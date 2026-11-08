"""
Build Graphormer-ready graphs from PeptideBERT NPZ inputs.
- Reads: ./data/{task}/{split}.npz with arrays 'inputs' [N, L], 'labels' [N]
- Removes special tokens ([PAD]=0, [CLS]=2, [SEP]=3, [MASK]=4) and padding
- Creates a sequence graph per peptide with k-hop edges (default k=1 → backbone)
- Saves: ./data/{task}/graphs_{split}.pt (list[torch_geometric.data.Data])
- Also provides helper to get PyG DataLoaders

Usage:
    python graph_data_builder.py --task hemo --k 2
    python graph_data_builder.py --task sol --k 1
    python graph_data_builder.py --task nf --k 3

Notes:
- This matches the NPZ layout used in PeptideBERT and prepares tensors
  commonly used by Graphormer-like models: x (token ids), edge_index, edge_attr (distance),
  spatial_pos (|i-j|), in_degree, out_degree, y (label).
- You can change `USE_ONE_HOT` to True if you prefer one-hot node features.
"""
from __future__ import annotations
import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

# try:
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# except Exception as e:
    # raise RuntimeError("torch-geometric is required. Install with: pip install torch-geometric") from e

# ===== Constants follow PeptideBERT token space used in your repo =====
PAD_ID = 0
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4
SPECIAL_IDS = {PAD_ID, CLS_ID, SEP_ID, MASK_ID}
VOCAB_SIZE = 30  # as used by the provided mapping

# If True, export x as one-hot [L, VOCAB_SIZE] (float32). If False, export token ids [L] (long)
USE_ONE_HOT = False


def _strip_and_trim(ids: np.ndarray) -> List[int]:   #delet pad, cls , mask just keep amino acids ids
    """Return the unpadded token ids, dropping specials (CLS/SEP/MASK) and PADs.
    Expects a 1D array of ints for one sequence.
    """
    # Keep tokens >0 (non-PAD), then drop the 3 special ids if present
    ids = ids.astype(np.int64)
    valid = ids[ids > 0].tolist()
    return [int(t) for t in valid if int(t) not in SPECIAL_IDS]


def _make_edges(n: int, k: int) -> Tuple[Tensor, Tensor]:
    """Build undirected k-hop edges for a chain of length n.
    Returns edge_index [2, E] (long) and edge_attr distances [E] (long).
    """
    src, dst, dist = [], [], []
    for d in range(1, k + 1):
        for i in range(n - d):
            j = i + d
            # i -> j and j -> i (undirected as two directed edges)
            src.append(i); dst.append(j); dist.append(d)
            src.append(j); dst.append(i); dist.append(d)
    if len(src) == 0:
        # Graph with a single node: no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(dist, dtype=torch.long)
    return edge_index, edge_attr


def _spatial_pos(n: int) -> Tensor:
    """|i-j| distances matrix used by Graphormer-style biases. Shape [n, n] (long)."""
    # Note: spatial_pos is not used in TransformerConv and causes batching issues
    # Return None to skip this feature
    return None


def _degrees(edge_index: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    """Compute in/out degree per node (long), given edge_index [2, E]."""    #count each edge and know of degrees
    if edge_index.numel() == 0:
        return torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    out_deg = torch.bincount(src, minlength=n)
    in_deg = torch.bincount(dst, minlength=n)
    return in_deg.to(torch.long), out_deg.to(torch.long)


def sequence_to_graph(ids_1d: np.ndarray, label: float, k_hop: int = 1) -> Data:
    """Convert one peptide token-id sequence to a PyG Data graph suitable for Graphormer-like encoders."""
    tokens = _strip_and_trim(ids_1d)
    n = len(tokens)
    if n == 0:
        # Edge case: after stripping specials, sequence is empty → keep one PAD node
        tokens = [PAD_ID]
        n = 1

    if USE_ONE_HOT:
        x = torch.zeros((n, VOCAB_SIZE), dtype=torch.float32)
        x[torch.arange(n), torch.tensor(tokens, dtype=torch.long)] = 1.0
    else:
        # Store token ids; the model can use an nn.Embedding(VOCAB_SIZE, hidden_dim)
        x = torch.tensor(tokens, dtype=torch.long)

    edge_index, edge_attr = _make_edges(n, k_hop)
    in_deg, out_deg = _degrees(edge_index, n)

    y = torch.tensor([float(label)], dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,  # distance in hops
        in_degree=in_deg,
        out_degree=out_deg,
        y=y,
        num_nodes=n,
    )
    return data


def _load_npz_split(task: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    path = f"./data/{task}/{split}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with np.load(path) as f:
        inputs = f["inputs"]  # [N, L]
        labels = f["labels"]  # [N]
    return inputs, labels


def process_split(task: str, split: str, k_hop: int) -> str:
    inputs, labels = _load_npz_split(task, split)
    N = inputs.shape[0]
    graphs: List[Data] = []
    for i in range(N):
        g = sequence_to_graph(inputs[i], float(labels[i]), k_hop=k_hop)
        graphs.append(g)
    out_path = f"./data/{task}/graphs_{split}.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(graphs, out_path)
    return out_path


def build_all(task: str, k_hop: int = 1) -> None:
    for split in ("train", "val", "test"):
        out = process_split(task, split, k_hop)
        print(f"Saved {split:>5} → {out}")


def get_graph_loaders(task: str, batch_size: int = 8, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience: return PyG DataLoaders for train/val/test after building graphs."""
    paths = {s: f"./data/{task}/graphs_{s}.pt" for s in ("train", "val", "test")}
    for s, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run build first.")
    train_graphs: List[Data] = torch.load(paths["train"], weights_only=False)
    val_graphs: List[Data] = torch.load(paths["val"], weights_only=False)
    test_graphs: List[Data] = torch.load(paths["test"], weights_only=False)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Graphormer-ready graphs from PeptideBERT NPZ inputs")
    parser.add_argument("--task", required=True, choices=["hemo", "sol", "nf"], help="Which dataset to process")
    parser.add_argument("--k", type=int, default=1, help="Max hop distance for edges (1=backbone only)")
    args = parser.parse_args()

    print(f"Building graphs for task={args.task} with k-hop edges k={args.k} …")
    build_all(task=args.task, k_hop=args.k)
    print("Done.")
