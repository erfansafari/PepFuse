#!/usr/bin/env python3
"""
Quick Test for Fusion Model Setup
=================================

Simple test to verify that the fusion model components work correctly.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports."""
    print("Testing imports...")
    try:
        import torch
        import torch_geometric
        from model.model_fusion_model import FusionModel, PeptideBERTSeqEncoder, DummySeqEncoder
        from model.model_graph_encoder import GraphEncoder
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test creating the fusion model."""
    print("\nTesting model creation...")
    try:
        from model.model_fusion_model import FusionModel, DummySeqEncoder
        from model.model_graph_encoder import GraphEncoder

        # Create components
        seq_encoder = DummySeqEncoder(vocab_size=30, d_model=256)
        graph_encoder = GraphEncoder(vocab_size=30, hidden_dim=128, heads=4, num_layers=2, k_max=3)

        # Create fusion model
        fusion_model = FusionModel(
            seq_encoder=seq_encoder,
            graph_encoder=graph_encoder,
            seq_dim=256,
            graph_dim=128,
            fusion="concat",
            hidden=128
        )

        print("✓ Fusion model created successfully")
        return fusion_model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return None

def test_forward_pass(model):
    """Test a forward pass through the model."""
    print("\nTesting forward pass...")
    try:
        import torch
        from torch_geometric.data import Data, Batch

        # Create toy data
        batch_size = 2
        seq_length = 8

        # Sequence data
        input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
        input_ids[:, 0] = 2  # CLS token
        input_ids[:, 1:5] = torch.tensor([5, 7, 9, 11])
        attention_mask = (input_ids > 0).long()

        # Graph data
        graphs = []
        for i in range(batch_size):
            x = torch.tensor([5, 7, 9, 11], dtype=torch.long)
            n = x.size(0)

            # Create edges
            src, dst, dist = [], [], []
            for j in range(n-1):
                k = j + 1
                src += [j, k]
                dst += [k, j]
                dist += [1, 1]

            edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(dist, dtype=torch.long) if dist else torch.empty((0,), dtype=torch.long)

            in_deg = torch.bincount(edge_index[1], minlength=n) if src else torch.zeros(n, dtype=torch.long)
            out_deg = torch.bincount(edge_index[0], minlength=n) if src else torch.zeros(n, dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                in_degree=in_deg,
                out_degree=out_deg
            )
            graphs.append(graph)

        batch_graph = Batch.from_data_list(graphs)

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(batch_graph, input_ids, attention_mask)

        print(f"✓ Forward pass successful, output shape: {logits.shape}")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("QUICK FUSION MODEL TEST")
    print("=" * 50)

    # Test imports
    if not test_basic_imports():
        return False

    # Test model creation
    model = test_model_creation()
    if model is None:
        return False

    # Test forward pass
    if not test_forward_pass(model):
        return False

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("Your fusion model setup is working correctly.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
