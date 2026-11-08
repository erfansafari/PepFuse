"""
Handcrafted Features GCNN Encoder
=================================

This module implements a GCNN-based encoder for handcrafted per-residue features:
- One-hot encoding (20 dims)
- BLOSUM62 substitution scores (20 dims)
- Z-scale physicochemical descriptors (5 dims)

The encoder uses multi-kernel 1D convolutions with GLU activation and residual
connections, followed by global max pooling to produce a fixed-size embedding.

Architecture inspired by CG-AMP paper's GCNN design.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GCNNLayer(nn.Module):
    """
    GCNN layer with multi-kernel convolutions, GLU activation, and residual connection.
    
    Uses 3 parallel convolution paths with kernel sizes 3, 5, 7 to capture features
    at different scales. The outputs are concatenated and projected through a gated
    linear unit (GLU) for better feature selection.
    """
    
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        """
        Args:
            dim: Input/output dimension
            hidden: Hidden dimension for each kernel path
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        
        # Multi-kernel convolutions (3, 5, 7)
        self.conv3 = nn.Conv1d(dim, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, hidden, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(dim, hidden, kernel_size=7, padding=3)
        
        # Projection with GLU (needs 2x channels for gating)
        self.proj = nn.Conv1d(3 * hidden, 2 * dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable residual weight
        self.gamma = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, L, D]
            
        Returns:
            Output tensor of shape [B, L, D]
        """
        # Store input for residual
        x_in = x
        
        # Transpose for Conv1d: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        
        # Apply multi-kernel convolutions
        z3 = self.conv3(x)  # [B, H, L]
        z5 = self.conv5(x)  # [B, H, L]
        z7 = self.conv7(x)  # [B, H, L]
        
        # Concatenate multi-scale features
        z = torch.cat([z3, z5, z7], dim=1)  # [B, 3H, L]
        
        # Project to 2D for GLU
        z = self.proj(z)  # [B, 2D, L]
        
        # Gated Linear Unit (GLU)
        A, B = torch.chunk(z, 2, dim=1)  # Each [B, D, L]
        h = A * torch.sigmoid(B)  # Element-wise gating
        
        # Dropout
        h = self.dropout(h)
        
        # Transpose back: [B, D, L] -> [B, L, D]
        h = h.transpose(1, 2)
        
        # Residual connection with learnable weight
        out = h + self.gamma * x_in
        
        return out


class HandcraftedGCNN(nn.Module):
    """
    GCNN encoder for handcrafted per-residue features.
    
    Takes handcrafted features (one-hot + BLOSUM62 + Z-scale) and produces
    a fixed-size sequence-level embedding using multi-layer GCNN followed by
    global max pooling.
    """
    
    def __init__(
        self,
        in_dim: int = 45,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            in_dim: Input feature dimension (45 for one-hot + BLOSUM62 + Z-scale)
            hidden_dim: Hidden dimension for GCNN layers
            num_layers: Number of GCNN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection: map 45-dim features to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Stack of GCNN layers
        self.layers = nn.ModuleList([
            GCNNLayer(dim=hidden_dim, hidden=hidden_dim * 2, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        feats: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feats: Handcrafted features of shape [B, L, 45]
            attention_mask: Optional mask of shape [B, L] (1 for valid, 0 for padding)
            
        Returns:
            Sequence-level embedding of shape [B, hidden_dim]
        """
        # Project input features
        x = self.input_proj(feats)  # [B, L, hidden_dim]
        
        # Apply GCNN layers
        for layer in self.layers:
            x = layer(x)  # [B, L, hidden_dim]
        
        # Layer normalization
        x = self.norm(x)  # [B, L, hidden_dim]
        
        # Mask padding positions before pooling
        if attention_mask is not None:
            # Expand mask: [B, L] -> [B, L, 1]
            mask = attention_mask.unsqueeze(-1).float()
            # Mask out padding by setting to -inf (will be ignored by max pooling)
            x = x.masked_fill(mask == 0, float('-inf'))
        
        # Global max pooling over sequence length
        z_hand = x.max(dim=1).values  # [B, hidden_dim]
        
        # Handle case where entire sequence was masked (shouldn't happen in practice)
        # Replace -inf with 0
        z_hand = torch.where(
            torch.isinf(z_hand),
            torch.zeros_like(z_hand),
            z_hand
        )
        
        return z_hand


# ===== Test functions =====

def test_gcnn_layer():
    """Test GCNNLayer forward pass."""
    print("\n" + "="*60)
    print("Testing GCNNLayer")
    print("="*60)
    
    B, L, D = 4, 20, 128
    layer = GCNNLayer(dim=D, hidden=D*2, dropout=0.1)
    
    x = torch.randn(B, L, D)
    out = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, L, D), f"Expected shape ({B}, {L}, {D}), got {out.shape}"
    print("✓ GCNNLayer test passed")


def test_handcrafted_gcnn():
    """Test HandcraftedGCNN forward pass."""
    print("\n" + "="*60)
    print("Testing HandcraftedGCNN")
    print("="*60)
    
    B, L = 4, 20
    in_dim = 45
    hidden_dim = 128
    num_layers = 2
    
    encoder = HandcraftedGCNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # Test without mask
    feats = torch.randn(B, L, in_dim)
    z = encoder(feats)
    
    print(f"Input features shape: {feats.shape}")
    print(f"Output embedding shape: {z.shape}")
    assert z.shape == (B, hidden_dim), f"Expected shape ({B}, {hidden_dim}), got {z.shape}"
    print("✓ Forward pass without mask passed")
    
    # Test with mask
    attention_mask = torch.ones(B, L)
    attention_mask[:, 15:] = 0  # Mask last 5 positions
    z_masked = encoder(feats, attention_mask)
    
    print(f"Output with mask shape: {z_masked.shape}")
    assert z_masked.shape == (B, hidden_dim), f"Expected shape ({B}, {hidden_dim}), got {z_masked.shape}"
    print("✓ Forward pass with mask passed")
    
    # Test backward pass
    loss = z_masked.sum()
    loss.backward()
    print("✓ Backward pass passed")
    
    print("\n✓ All HandcraftedGCNN tests passed")


if __name__ == "__main__":
    print("="*60)
    print("Handcrafted GCNN Encoder Tests")
    print("="*60)
    
    test_gcnn_layer()
    test_handcrafted_gcnn()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
