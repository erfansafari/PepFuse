#!/usr/bin/env python3
"""
Test Fusion Model Setup
======================

This script verifies that the fusion model setup is working correctly by:
1. Testing all imports
2. Creating model components
3. Testing data loading
4. Running a small forward pass
5. Checking GPU availability

Usage:
    python test_fusion_setup.py
    python test_fusion_setup.py --quick
"""

import os
import sys
import argparse
import traceback
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        # Core PyTorch
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print(f"✓ PyTorch {torch.__version__}")

        # PyTorch Geometric
        import torch_geometric
        from torch_geometric.data import Data, Batch
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch_geometric.nn import TransformerConv
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")

        # Transformers
        import transformers
        from transformers import BertModel, BertConfig
        print(f"✓ Transformers {transformers.__version__}")

        # Other libraries
        import numpy as np
        import sklearn
        import yaml
        import tqdm
        print(f"✓ NumPy {np.__version__}")
        print(f"✓ Scikit-learn {sklearn.__version__}")

        # Project modules
        from model.model_fusion_model import FusionModel, PeptideBERTSeqEncoder, DummySeqEncoder
        from model.model_graph_encoder import GraphEncoder
        from data.graph_data_builder import sequence_to_graph
        print("✓ Project modules")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_device_setup():
    """Test device availability and setup."""
    print("\nTesting device setup...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device: {device}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ Current GPU: {torch.cuda.get_device_name()}")

            # Test GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU memory: {gpu_memory:.1f} GB")
        else:
            print("! Using CPU (CUDA not available)")

        return True

    except Exception as e:
        print(f"✗ Device setup error: {e}")
        return False


def test_model_creation():
    """Test creating model components."""
    print("\nTesting model creation...")

    try:
        # Test sequence encoder
        seq_encoder = DummySeqEncoder(vocab_size=30, d_model=256)
        print("✓ Dummy sequence encoder created")

        # Test graph encoder
        graph_encoder = GraphEncoder(
            vocab_size=30,
            hidden_dim=128,
            heads=4,
            num_layers=2,
            k_max=3
        )
        print("✓ Graph encoder created")

        # Test fusion model
        fusion_model = FusionModel(
            seq_encoder=seq_encoder,
            graph_encoder=graph_encoder,
            seq_dim=256,
            graph_dim=128,
            fusion="concat",
            hidden=128
        )
        print("✓ Fusion model created")

        # Count parameters
        total_params = sum(p.numel() for p in fusion_model.parameters())
        trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        return fusion_model

    except Exception as e:
        print(f"✗ Model creation error: {e}")
        traceback.print_exc()
        return None


def test_data_creation():
    """Test creating toy data."""
    print("\nTesting data creation...")

    try:
        from torch_geometric.data import Data, Batch

        # Create toy graph data
        graphs = []
        batch_size = 4

        for i in range(batch_size):
            # Create a small peptide graph (4 amino acids)
            x = torch.tensor([5, 7, 9, 11], dtype=torch.long)  # Token IDs
            n = x.size(0)

            # Create edges (backbone + 1-hop)
            src, dst, dist = [], [], []
            for j in range(n-1):
                k = j + 1
                src += [j, k]
                dst += [k, j]
                dist += [1, 1]

            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(dist, dtype=torch.long)

            # Compute degrees
            in_deg = torch.bincount(edge_index[1], minlength=n)
            out_deg = torch.bincount(edge_index[0], minlength=n)

            # Create label
            y = torch.tensor([float(i % 2)], dtype=torch.float32)

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                in_degree=in_deg,
                out_degree=out_deg,
                y=y
            )
            graphs.append(graph)

        # Batch graphs
        batch_graph = Batch.from_data_list(graphs)
        print(f"✓ Created batch with {batch_graph.num_graphs} graphs")
        print(f"✓ Total nodes: {batch_graph.x.size(0)}")
        print(f"✓ Total edges: {batch_graph.edge_index.size(1)}")

        # Create sequence data
        seq_length = 8
        input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
        input_ids[:, 0] = 2  # CLS token
        input_ids[:, 1:5] = torch.tensor([5, 7, 9, 11])  # Match graph tokens
        attention_mask = (input_ids > 0).long()

        print(f"✓ Created sequence data: {input_ids.shape}")

        return batch_graph, input_ids, attention_mask

    except Exception as e:
        print(f"✗ Data creation error: {e}")
        traceback.print_exc()
        return None, None, None


def test_forward_pass(model, batch_graph, input_ids, attention_mask):
    """Test forward pass through the model."""
    print("\nTesting forward pass...")

    try:
        model.eval()

        with torch.no_grad():
            # Forward pass
            logits = model(batch_graph, input_ids, attention_mask)

            print(f"✓ Forward pass successful")
            print(f"✓ Output shape: {logits.shape}")
            print(f"✓ Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

            # Test predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            print(f"✓ Probabilities: {probs.tolist()}")
            print(f"✓ Predictions: {preds.tolist()}")

        return True

    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow(model, batch_graph, input_ids, attention_mask):
    """Test that gradients flow properly."""
    print("\nTesting gradient flow...")

    try:
        model.train()

        # Create fake targets
        targets = torch.randint(0, 2, (input_ids.size(0),)).float()

        # Forward pass
        logits = model(batch_graph, input_ids, attention_mask)

        # Compute loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, targets)

        print(f"✓ Loss computed: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = False
        gradient_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)

        if has_gradients:
            print(f"✓ Gradients computed successfully")
            print(f"✓ Average gradient norm: {np.mean(gradient_norms):.6f}")
            print(f"✓ Max gradient norm: {max(gradient_norms):.6f}")
        else:
            print("✗ No gradients computed")
            return False

        return True

    except Exception as e:
        print(f"✗ Gradient flow error: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test loading configuration files."""
    print("\nTesting configuration loading...")

    try:
        import yaml

        # Test default config
        config_path = "config_fusion.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded config from {config_path}")
            print(f"✓ Task: {config.get('task', 'unknown')}")
            print(f"✓ Batch size: {config.get('batch_size', 'unknown')}")
            return True
        else:
            print(f"! Config file {config_path} not found (this is okay)")
            return True

    except Exception as e:
        print(f"✗ Config loading error: {e}")
        return False


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")

    try:
        data_dir = Path("data")
        if not data_dir.exists():
            print("! Data directory not found (run data preparation first)")
            return True

        # Check for NPZ files
        tasks = ['hemo', 'sol', 'nf']
        splits = ['train', 'val', 'test']

        found_data = False
        for task in tasks:
            task_dir = data_dir / task
            if task_dir.exists():
                print(f"✓ Found task directory: {task}")
                for split in splits:
                    npz_file = task_dir / f"{split}.npz"
                    if npz_file.exists():
                        print(f"  ✓ {split}.npz")
                        found_data = True
                    else:
                        print(f"  ! {split}.npz not found")

        if found_data:
            print("✓ Some data files found")
        else:
            print("! No data files found (run data preparation first)")

        return True

    except Exception as e:
        print(f"✗ Data file check error: {e}")
        return False


def run_full_test():
    """Run all tests."""
    print("=" * 60)
    print("FUSION MODEL SETUP TEST")
    print("=" * 60)

    tests_passed = 0
    total_tests = 0

    # Test 1: Imports
    total_tests += 1
    if test_imports():
        tests_passed += 1

    # Test 2: Device setup
    total_tests += 1
    if test_device_setup():
        tests_passed += 1

    # Test 3: Model creation
    total_tests += 1
    model = test_model_creation()
    if model is not None:
        tests_passed += 1

        # Test 4: Data creation
        total_tests += 1
        batch_graph, input_ids, attention_mask = test_data_creation()
        if batch_graph is not None:
            tests_passed += 1

            # Test 5: Forward pass
            total_tests += 1
            if test_forward_pass(model, batch_graph, input_ids, attention_mask):
                tests_passed += 1

            # Test 6: Gradient flow
            total_tests += 1
            if test_gradient_flow(model, batch_graph, input_ids, attention_mask):
                tests_passed += 1

    # Test 7: Config loading
    total_tests += 1
    if test_config_loading():
        tests_passed += 1

    # Test 8: Data files
    total_tests += 1
    if test_data_files():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your fusion model setup is ready.")
        print("\nNext steps:")
        print("1. Prepare graph data: python prepare_graph_data.py --all")
        print("2. Start training: python train_fusion.py --config config_fusion.yaml")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Please check the errors above.")
        return False


def run_quick_test():
    """Run a quick subset of tests."""
    print("=" * 60)
    print("QUICK FUSION MODEL TEST")
    print("=" * 60)

    # Just test imports and basic model creation
    if not test_imports():
        return False

    model = test_model_creation()
    if model is None:
        return False

    batch_graph, input_ids, attention_mask = test_data_creation()
    if batch_graph is None:
        return False

    if not test_forward_pass(model, batch_graph, input_ids, attention_mask):
        return False

    print("\n✅ Quick test passed! Basic setup is working.")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test fusion model setup")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test (subset of all tests)'
    )

    args = parser.parse_args()

    try:
        if args.quick:
            success = run_quick_test()
        else:
            success = run_full_test()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
