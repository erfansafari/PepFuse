"""
Fusion model: PeptideBERT (sequence) ⟷ GraphEncoder (graph) → binary classifier
==============================================================================

This module defines a compact **FusionModel** that combines:
  • a sequence encoder that returns a CLS embedding for each sequence,
  • the provided GraphEncoder (Graphormer‑lite via TransformerConv) that returns
    a fixed‑size graph embedding per peptide graph.

It is tailored to work with the two files you already have:
  - model/graph_encoder.py  (GraphEncoder)
  - data/graph_data_builder.py  (produces graphs_{split}.pt)

Minimal usage
-------------
from model.fusion_model import FusionModel, DummySeqEncoder
from model.graph_encoder import GraphEncoder

seq_enc   = DummySeqEncoder(d_model=256)      # swap with your PeptideBERT later
graph_enc = GraphEncoder(vocab_size=30, hidden_dim=128, heads=4, num_layers=4, k_max=3)
model     = FusionModel(seq_encoder=seq_enc, graph_encoder=graph_enc,
                        seq_dim=256, graph_dim=128, fusion="concat", hidden=256)

# Forward (toy):
logits = model(batch_graph, input_ids, attention_mask)  # [B]

Notes
-----
- Keep `seq_dim` equal to your PeptideBERT CLS embedding size.
- Keep `graph_dim` equal to GraphEncoder's hidden_dim.
- Two fusion types provided: 'concat' (default) and 'gated'.
- A built‑in **smoke test** at the bottom runs a tiny forward pass without any
  project data (useful to verify wiring and shapes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

# Import mean pooling from refactored network.py
try:
    from model.network import mean_pooling
except ImportError:
    # Fallback if import fails
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over valid tokens (non-padding)."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


# --------------------------- sequence encoders ------------------------------

class SeqEncoderBase(nn.Module):
    """Abstract: returns a CLS embedding [B, d_seq] from tokenized input."""
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DummySeqEncoder(SeqEncoderBase):
    """Small Transformer‑like encoder for sanity checks (no pretraining)."""
    def __init__(self, vocab_size: int = 30, d_model: int = 256, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4*d_model, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        # TransformerEncoder uses True for masked positions; attention_mask has 1 for keep
        key_padding_mask = (attention_mask == 0)
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        cls = h[:, 0, :]  # assumes CLS token at position 0
        return cls


class PeptideBERTSeqEncoder(SeqEncoderBase):
    """Adapter aligned with **refactored network.py PeptideBERT**.

    Uses mean pooling instead of pooler_output for better representations.
    This class exposes the **mean-pooled embedding (pre‑head)** for fusion.
    It builds a BertModel with the SAME config knobs as your repo and loads the
    `Rostlab/prot_bert_bfd` weights, matching your refactored pipeline.
    """
    def __init__(self, vocab_size: int, hidden_size: int, hidden_layers: int, attn_heads: int,
                 dropout: float, d_out: int):
        super().__init__()
        from transformers import BertModel, BertConfig
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attn_heads,
            hidden_dropout_prob=dropout,
        )
        self.backbone = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd', config=cfg, ignore_mismatched_sizes=True
        )
        self.proj = nn.Identity() if d_out == hidden_size else nn.Linear(hidden_size, d_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass using mean pooling (aligned with refactored PeptideBERT)."""
        out = self.backbone(input_ids, attention_mask=attention_mask, return_dict=True)
        # Use mean pooling instead of pooler_output (matches refactored network.py)
        pooled = mean_pooling(out.last_hidden_state, attention_mask)
        return self.proj(pooled)


# ------------------------------- fusion -------------------------------------

@dataclass
class FusionConfig:
    seq_dim: int = 256
    graph_dim: int = 128
    fusion: str = "concat"   # {'concat','gated','cross_attention'}
    hidden: int = 256
    dropout: float = 0.1
    num_heads: int = 4  # For cross-attention


class FusionModel(nn.Module):
    """Fuse a sequence encoder (CLS) with the GraphEncoder (graph_emb) and optionally handcrafted features."""
    def __init__(self, seq_encoder: SeqEncoderBase, graph_encoder: nn.Module,
                 seq_dim: int, graph_dim: int, fusion: str = "concat",
                 hidden: int = 256, dropout: float = 0.1, num_heads: int = 4,
                 handcrafted_encoder: Optional[nn.Module] = None,
                 hand_dim: Optional[int] = None):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.graph_encoder = graph_encoder
        self.handcrafted_encoder = handcrafted_encoder
        self.use_handcrafted = handcrafted_encoder is not None
        
        self.seq_dim = seq_dim
        self.graph_dim = graph_dim
        self.hand_dim = hand_dim if self.use_handcrafted else 0
        
        self.cfg = FusionConfig(seq_dim=seq_dim, graph_dim=graph_dim,
                                 fusion=fusion, hidden=hidden, dropout=dropout,
                                 num_heads=num_heads)

        if fusion == "concat":
            in_dim = seq_dim + graph_dim + self.hand_dim
            self.gate = None
            self.cross_attn = None
            self.seq_proj = None
            self.graph_proj = None
        elif fusion == "gated":
            if self.use_handcrafted:
                raise NotImplementedError("Handcrafted features currently supported only for fusion='concat'")
            in_dim = seq_dim
            self.gate = nn.Sequential(
                nn.Linear(seq_dim + graph_dim, hidden), nn.ReLU(), nn.Linear(hidden, seq_dim), nn.Sigmoid()
            )
            self.cross_attn = None
            self.seq_proj = None
            self.graph_proj = None
        elif fusion == "cross_attention":
            if self.use_handcrafted:
                raise NotImplementedError("Handcrafted features currently supported only for fusion='concat'")
            # Project to same dimension if needed
            common_dim = min(seq_dim, graph_dim)
            self.seq_proj = nn.Linear(seq_dim, common_dim) if seq_dim != common_dim else nn.Identity()
            self.graph_proj = nn.Linear(graph_dim, common_dim) if graph_dim != common_dim else nn.Identity()
            
            # Cross-attention: sequence attends to graph and vice versa
            self.seq_to_graph_attn = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.graph_to_seq_attn = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Layer normalization
            self.seq_norm = nn.LayerNorm(common_dim)
            self.graph_norm = nn.LayerNorm(common_dim)
            
            # Fusion after cross-attention
            in_dim = common_dim * 2  # Concatenate attended features
            self.gate = None
            self.cross_attn = True
        else:
            raise ValueError("fusion must be one of {'concat','gated','cross_attention'}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def get_representation(self, batch_graph: Batch, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                          classic_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract the fused representation before the final classification layer.
        
        This method computes the same fused representation z that is used in forward()
        right before applying the final MLP classifier. This representation can be used
        for auxiliary losses like supervised contrastive loss.
        
        Args:
            batch_graph: PyG Batch with graph data
            input_ids: [B, L] tokenized peptide ids
            attention_mask: [B, L] attention mask (1 for valid tokens, 0 for padding)
            classic_feats: [B, L, 45] handcrafted features (optional)
            
        Returns:
            z: [B, d_rep] fused representation vector
               - For concat fusion: d_rep = seq_dim + graph_dim (+ hand_dim if enabled)
               - For gated fusion: d_rep = seq_dim
               - For cross_attention fusion: d_rep = common_dim * 2
        """
        # 1) Encode sequence
        seq_cls = self.seq_encoder(input_ids, attention_mask)   # [B, seq_dim]
        
        # 2) Encode graph
        g_out = self.graph_encoder(batch_graph)
        graph_emb = g_out["graph_emb"]                          # [B, graph_dim]
        
        # 3) Encode handcrafted features if enabled
        if self.use_handcrafted:
            if classic_feats is None:
                raise ValueError("classic_feats must be provided when handcrafted branch is enabled")
            z_hand = self.handcrafted_encoder(classic_feats, attention_mask)  # [B, hand_dim]
        else:
            z_hand = None

        # 4) Apply fusion logic (same as forward, but stop before MLP)
        if self.cfg.fusion == "concat":
            parts = [seq_cls, graph_emb]
            if z_hand is not None:
                parts.append(z_hand)
            z = torch.cat(parts, dim=-1)
        elif self.cfg.fusion == "gated":
            gate = self.gate(torch.cat([seq_cls, graph_emb], dim=-1))
            z = seq_cls * gate
        elif self.cfg.fusion == "cross_attention":
            # Project to common dimension
            seq_proj = self.seq_proj(seq_cls)      # [B, common_dim]
            graph_proj = self.graph_proj(graph_emb)  # [B, common_dim]
            
            # Add sequence dimension for attention (batch_first=True expects [B, S, D])
            seq_proj = seq_proj.unsqueeze(1)       # [B, 1, common_dim]
            graph_proj = graph_proj.unsqueeze(1)   # [B, 1, common_dim]
            
            # Cross-attention: sequence attends to graph
            seq_attended, _ = self.seq_to_graph_attn(
                query=seq_proj,
                key=graph_proj,
                value=graph_proj
            )  # [B, 1, common_dim]
            seq_attended = self.seq_norm(seq_attended.squeeze(1) + seq_proj.squeeze(1))  # Residual + LN
            
            # Cross-attention: graph attends to sequence
            graph_attended, _ = self.graph_to_seq_attn(
                query=graph_proj,
                key=seq_proj,
                value=seq_proj
            )  # [B, 1, common_dim]
            graph_attended = self.graph_norm(graph_attended.squeeze(1) + graph_proj.squeeze(1))  # Residual + LN
            
            # Concatenate attended features
            z = torch.cat([seq_attended, graph_attended], dim=-1)  # [B, common_dim*2]
        
        return z

    def forward(self, batch_graph: Batch, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                classic_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the fusion model.
        
        Args:
            batch_graph: PyG Batch with fields expected by GraphEncoder
            input_ids: [B, L] tokenized peptide ids
            attention_mask: [B, L] attention mask (1 for tokens, 0 for padding)
            classic_feats: [B, L, 45] handcrafted features (optional, required if use_handcrafted=True)
            
        Returns:
            logits: [B] classification logits (before sigmoid)
        """
        # Get fused representation (reuses get_representation for consistency)
        z = self.get_representation(batch_graph, input_ids, attention_mask, classic_feats)
        
        # Apply final classification layer
        logits = self.mlp(z).squeeze(-1)  # [B]
        return logits


# ------------------------------- tests --------------------------------------

def _toy_batch(B: int = 2, L: int = 8):
    """Create a tiny fake batch (no disk IO) to validate forward pass."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    graphs = []
    for b in range(B):
        # 4-token chain with hop=1 edges
        x = torch.tensor([5, 7, 9, 11], dtype=torch.long)
        n = x.numel()
        src, dst, dist = [], [], []
        for i in range(n-1):
            j = i + 1
            src += [i, j]; dst += [j, i]; dist += [1, 1]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(dist, dtype=torch.long)
        in_deg = torch.bincount(edge_index[1], minlength=n)
        out_deg = torch.bincount(edge_index[0], minlength=n)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                           in_degree=in_deg, out_degree=out_deg))

    batch_graph = Batch.from_data_list(graphs)
    input_ids = torch.zeros((B, L), dtype=torch.long)
    input_ids[:, 0] = 2  # CLS id
    input_ids[:, 1:5] = torch.tensor([5, 7, 9, 11])
    attention_mask = (input_ids > 0).long()
    return batch_graph, input_ids, attention_mask


def _smoke_test():
    from model.graph_encoder import GraphEncoder
    seq = DummySeqEncoder(d_model=256)
    genc = GraphEncoder(vocab_size=30, hidden_dim=128, heads=4, num_layers=2, k_max=2)
    model = FusionModel(seq, genc, seq_dim=256, graph_dim=128, fusion="concat", hidden=128)

    g, ids, mask = _toy_batch()
    logits = model(g, ids, mask)
    assert logits.shape == (2,), f"Unexpected logits shape: {tuple(logits.shape)}"
    print("[OK] FusionModel smoke test passed.")


if __name__ == "__main__":
    try:
        _smoke_test()
    except Exception:
        import traceback
        traceback.print_exc()
        print("[FAIL] FusionModel test failed.")
