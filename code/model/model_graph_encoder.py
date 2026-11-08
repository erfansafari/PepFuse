"""
GraphEncoder: Graphormer‑lite encoder for peptide graphs (PyTorch Geometric)
===========================================================================

This module encodes peptide graphs produced by `graph_data_builder.py` into a
fixed‑size graph embedding suitable for fusion with PeptideBERT.

Requirements
------------
- torch >= 1.12
- torch_geometric >= 2.3

Install (examples):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch-geometric

Usage
-----
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from model.graph_encoder import GraphEncoder

    enc = GraphEncoder(vocab_size=30, hidden_dim=128, num_layers=4, heads=4, k_max=3)
    out = enc(data)  # data is a PyG Data with fields: x, edge_index, edge_attr, in_degree, out_degree
    graph_emb = out["graph_emb"]  # [B, hidden_dim]

Notes
-----
- Node features `x` are expected either as token ids [N] (int64) or one‑hot [N,V] (float32).
- Edge attributes `edge_attr` are hop distances in [1..k]; they are embedded and passed to TransformerConv.
- Degrees are embedded (in+out) as a centrality bias and added to node states.
- Spatial matrix `spatial_pos` is not used directly here because TransformerConv
  does not accept pairwise bias; we already encode local distances via `edge_attr`.
  (You can extend with GPSConv/GraphormerConv if available in your environment.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool, global_max_pool


# ------------------------------ helpers ------------------------------------

def _is_one_hot(x: torch.Tensor) -> bool:
    return x.dim() == 2 and x.dtype.is_floating_point


def _divisible(v: int, d: int) -> int:
    if v % d != 0:
        raise ValueError(f"hidden_dim={v} must be divisible by heads={d}")
    return v // d


# ---------------------------- configuration ---------------------------------

@dataclass
class GraphEncoderConfig:
    vocab_size: int = 30
    hidden_dim: int = 128
    heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    k_max: int = 3                 # maximum hop distance seen in edge_attr
    max_degree: int = 64           # clamp degree embeddings to this range
    readout: str = "mean"          # one of {"mean","add","max"}
    use_degree: bool = True


# ------------------------------ encoder -------------------------------------

class GraphEncoder(nn.Module):
    """Graphormer‑lite encoder using TransformerConv and hop/degree encodings."""

    def __init__(self, **kwargs):
        super().__init__()
        cfg = GraphEncoderConfig(**kwargs)
        self.cfg = cfg

        # Node encoding
        self.node_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.node_proj = nn.Linear(cfg.vocab_size, cfg.hidden_dim)  # for one‑hot input

        # Centrality (degree) encoding
        self.use_degree = cfg.use_degree
        if self.use_degree:
            self.in_deg_emb = nn.Embedding(cfg.max_degree + 1, cfg.hidden_dim)
            self.out_deg_emb = nn.Embedding(cfg.max_degree + 1, cfg.hidden_dim)

        # Edge (hop distance) encoding → provided to TransformerConv via edge_dim
        self.edge_dim = cfg.hidden_dim
        self.hop_emb = nn.Embedding(cfg.k_max + 1, self.edge_dim)  # idx 0 unused; 1..k

        # Stacked TransformerConv layers (residual + norm)
        out_per_head = _divisible(cfg.hidden_dim, cfg.heads)
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=cfg.hidden_dim,
                out_channels=out_per_head,
                heads=cfg.heads,
                edge_dim=self.edge_dim,
                dropout=cfg.dropout,
                beta=True,
            )
            for _ in range(cfg.num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)])
        self.dropout = nn.Dropout(cfg.dropout)

    # ---------------------------- forward -----------------------------------
    def forward(self, data) -> Dict[str, torch.Tensor]:
        """Encode a batch of graphs.
        Expected fields on `data`:
          - x: [num_nodes] (long token ids) or [num_nodes, vocab] (float one‑hot)
          - edge_index: [2, E] (long)
          - edge_attr: [E] (long hop distances)
          - in_degree, out_degree: [num_nodes] (long) (optional if use_degree=False)
          - batch: [num_nodes] (long) — from PyG DataLoader
        Returns dict with keys: node_emb, graph_emb
        """
        x = data.x
        if _is_one_hot(x):
            h = self.node_proj(x)
        else:
            if x.dim() != 1:
                raise ValueError(f"x must be [N] token ids or [N,V] one‑hot; got {tuple(x.shape)}")
            h = self.node_emb(x)

        if self.use_degree:
            if not hasattr(data, "in_degree") or not hasattr(data, "out_degree"):
                raise ValueError("Degree features missing: 'in_degree' and 'out_degree' required when use_degree=True")
            in_deg = data.in_degree.clamp(max=self.cfg.max_degree)
            out_deg = data.out_degree.clamp(max=self.cfg.max_degree)
            h = h + self.in_deg_emb(in_deg) + self.out_deg_emb(out_deg)

        edge_index = data.edge_index
        if not hasattr(data, "edge_attr"):
            raise ValueError("edge_attr (hop distances) is required")
        hop = data.edge_attr.clamp(min=0, max=self.cfg.k_max)
        edge_feat = self.hop_emb(hop)  # [E, edge_dim]

        # TransformerConv stack with pre‑norm + residual
        for conv, ln in zip(self.layers, self.norms):
            h_in = h
            h = ln(h)
            h = F.relu(conv(h, edge_index, edge_attr=edge_feat))
            h = self.dropout(h)
            h = h + h_in

        # Graph readout
        if not hasattr(data, "batch"):
            # Single graph path: create a dummy batch of zeros
            batch = x.new_zeros(x.size(0))
        else:
            batch = data.batch

        if self.cfg.readout == "add":
            g = global_add_pool(h, batch)
        elif self.cfg.readout == "max":
            g = global_max_pool(h, batch)
        else:
            g = global_mean_pool(h, batch)

        return {"node_emb": h, "graph_emb": g}


# ------------------------------- tests --------------------------------------

def _make_toy(batch_size: int = 2, k: int = 2):
    """Create a tiny batch of chain graphs for testing.
    Each graph has 4 tokens: [5,7,9,11] and [6,8,10,12].
    edge_attr are hop distances in [1..k].
    """
    from torch_geometric.data import Data
    import itertools

    graphs = []
    for b in range(batch_size):
        x = torch.tensor([5 + b, 7 + b, 9 + b, 11 + b], dtype=torch.long)
        # edges for chain with k hops
        src, dst, dist = [], [], []
        n = x.numel()
        for d in range(1, k + 1):
            for i in range(n - d):
                j = i + d
                src += [i, j]; dst += [j, i]; dist += [d, d]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(dist, dtype=torch.long)
        in_deg = torch.bincount(edge_index[1], minlength=n)
        out_deg = torch.bincount(edge_index[0], minlength=n)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, in_degree=in_deg, out_degree=out_deg))

    # mini‑batch
    from torch_geometric.loader import DataLoader
    loader = DataLoader(graphs, batch_size=batch_size)
    batch = next(iter(loader))
    return batch


def _run_smoke_test():
    enc = GraphEncoder(vocab_size=30, hidden_dim=64, heads=4, num_layers=2, k_max=2)
    batch = _make_toy()
    out = enc(batch)
    g = out["graph_emb"]
    assert g.shape[0] == 2 and g.shape[1] == 64, f"Unexpected graph_emb shape: {tuple(g.shape)}"
    print("[OK] GraphEncoder smoke test passed.")


if __name__ == "__main__":
    try:
        _run_smoke_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[FAIL] GraphEncoder test failed:", e)
