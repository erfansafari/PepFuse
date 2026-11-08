#!/usr/bin/env python3
"""
Fusion Model Testing Script
==========================

This script loads a trained fusion model checkpoint and evaluates it on the test set.

Usage:
    python test_model.py --checkpoint checkpoints/fusion_nf_0820_1652/best_model.pt
    python test_model.py --checkpoint checkpoints/fusion_nf_0820_1652/best_model.pt --config config_fusion.yaml
    python test_model.py --checkpoint checkpoints/fusion_nf_0820_1652/best_model.pt --save-predictions
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data.fusion_dataloader import load_fusion_data
from model.model_fusion_model import FusionModel, PeptideBERTSeqEncoder
from model.model_graph_encoder import GraphEncoder
from model.fusion_utils import test_fusion, print_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - ROC curve plotting disabled")


def load_config_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load configuration from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Configuration dictionary
    """
    # Try to find config in checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.yaml")

    if os.path.exists(config_path):
        logger.info(f"Loading config from checkpoint directory: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"No config found in checkpoint directory. Using default config.")
        # Return default config
        return {
            'task': 'nf',
            'batch_size': 32,
            'vocab_size': 25,
            'graph': {'k_hop': 2, 'readout': 'mean'},
            'sequence': {'hidden_size': 480, 'hidden_layers': 12, 'attn_heads': 12, 'dropout': 0.15},
            'graph_encoder': {'hidden_dim': 128, 'heads': 8, 'num_layers': 4, 'dropout': 0.1, 'k_max': 3, 'max_degree': 64, 'use_degree': True},
            'fusion': {'fusion_type': 'concat', 'hidden_dim': 256, 'dropout': 0.1},
            'data': {'num_workers': 4, 'pin_memory': True}
        }


def create_fusion_model(config: Dict) -> nn.Module:
    """
    Create the fusion model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized fusion model
    """
    # Create sequence encoder (PeptideBERT)
    seq_encoder = PeptideBERTSeqEncoder(
        vocab_size=config['vocab_size'],
        hidden_size=config['sequence']['hidden_size'],
        hidden_layers=config['sequence']['hidden_layers'],
        attn_heads=config['sequence']['attn_heads'],
        dropout=config['sequence']['dropout'],
        d_out=config['sequence']['hidden_size']
    )

    # Create graph encoder (Graphormer Lite)
    graph_encoder = GraphEncoder(
        vocab_size=config['vocab_size'],
        hidden_dim=config['graph_encoder']['hidden_dim'],
        heads=config['graph_encoder']['heads'],
        num_layers=config['graph_encoder']['num_layers'],
        dropout=config['graph_encoder']['dropout'],
        k_max=config['graph_encoder']['k_max'],
        max_degree=config['graph_encoder']['max_degree'],
        readout=config['graph'].get('readout', 'mean'),
        use_degree=config['graph_encoder']['use_degree']
    )

    # Create fusion model
    fusion_model = FusionModel(
        seq_encoder=seq_encoder,
        graph_encoder=graph_encoder,
        seq_dim=config['sequence']['hidden_size'],
        graph_dim=config['graph_encoder']['hidden_dim'],
        fusion=config['fusion']['fusion_type'],
        hidden=config['fusion']['hidden_dim'],
        dropout=config['fusion']['dropout']
    )

    return fusion_model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        Checkpoint metadata
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Log checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})

    logger.info(f"Loaded checkpoint from epoch {epoch}")
    if metrics:
        logger.info("Checkpoint validation metrics:")
        for key, value in metrics.items():
            if 'time' in key.lower():
                logger.info(f"  {key}: {value:.4f}s")
            elif 'samples' in key.lower():
                logger.info(f"  {key}: {int(value)}")
            else:
                logger.info(f"  {key}: {value:.2f}%")

    return checkpoint


def save_detailed_results(test_metrics: Dict, predictions: np.ndarray,
                         labels: np.ndarray, probabilities: np.ndarray,
                         config: Dict, save_dir: str):
    """Save detailed test results including predictions and visualizations."""

    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics
    results_file = os.path.join(save_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"# Fusion Model Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n")
        f.write(f"- Task: {config['task']}\n")
        f.write(f"- Fusion Type: {config['fusion']['fusion_type']}\n")
        f.write(f"- Sequence Hidden Size: {config['sequence']['hidden_size']}\n")
        f.write(f"- Graph Hidden Size: {config['graph_encoder']['hidden_dim']}\n")
        f.write(f"- Graph K-hop: {config['graph']['k_hop']}\n\n")

        f.write("## Test Metrics\n")
        for key, value in test_metrics.items():
            if 'time' in key.lower():
                f.write(f"- {key.title()}: {value:.4f}s\n")
            elif 'samples' in key.lower():
                f.write(f"- {key.title()}: {int(value)}\n")
            else:
                f.write(f"- {key.title()}: {value:.2f}%\n")

    # Save predictions
    predictions_file = os.path.join(save_dir, "predictions.npz")
    np.savez(predictions_file,
             predictions=predictions,
             labels=labels,
             probabilities=probabilities)

    # Save classification report
    report_file = os.path.join(save_dir, "classification_report.txt")
    with open(report_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("===================\n\n")
        f.write(classification_report(labels, predictions, target_names=['Negative', 'Positive']))

    # Save confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_file = os.path.join(save_dir, "confusion_matrix.txt")
    with open(cm_file, 'w') as f:
        f.write("Confusion Matrix\n")
        f.write("===============\n\n")
        f.write(f"True Negative:  {cm[0,0]}\n")
        f.write(f"False Positive: {cm[0,1]}\n")
        f.write(f"False Negative: {cm[1,0]}\n")
        f.write(f"True Positive:  {cm[1,1]}\n")

    # Create ROC curve if possible
    if HAS_MATPLOTLIB:
        try:
            fpr, tpr, _ = roc_curve(labels, probabilities)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            roc_file = os.path.join(save_dir, "roc_curve.png")
            plt.savefig(roc_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"ROC curve saved to {roc_file}")

        except Exception as e:
            logger.warning(f"Could not create ROC curve: {e}")
    else:
        logger.info("Skipping ROC curve generation (matplotlib not available)")

    logger.info(f"Detailed results saved to {save_dir}")


def test_model_with_predictions(model: nn.Module, dataloader, device: torch.device):
    """Test model and return predictions along with metrics."""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            batch_graph = batch['batch_graph'].to(device)

            # Forward pass
            logits = model(batch_graph, input_ids, attention_mask)

            # Convert logits to predictions and probabilities
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities))


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test Fusion Model from Checkpoint")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional, will try to load from checkpoint dir)'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save detailed predictions and analysis'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save results (default: test_results_TIMESTAMP)'
    )

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = load_config_from_checkpoint(args.checkpoint)

    logger.info(f"Testing model for task: {config['task']}")

    try:
        # Load data
        logger.info("Loading test data...")
        _, _, test_loader = load_fusion_data(config)
        logger.info(f"Test batches: {len(test_loader)}")

        # Create model
        logger.info("Creating fusion model...")
        model = create_fusion_model(config)
        model = model.to(device)

        # Load checkpoint
        checkpoint_info = load_checkpoint(model, args.checkpoint, device)

        # Run testing
        logger.info("Running model evaluation...")
        print(f"\n{'='*60}")
        print(f"{'FUSION MODEL TESTING':^60}")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Task: {config['task']}")
        print(f"Device: {device}")
        print("-" * 60)

        # Test with detailed metrics
        test_metrics = test_fusion(model, test_loader, device)

        print_metrics(test_metrics, "Test Results")

        # Save detailed results if requested
        if args.save_predictions:
            logger.info("Generating detailed predictions...")
            predictions, labels, probabilities = test_model_with_predictions(model, test_loader, device)

            # Create output directory
            if args.output_dir:
                save_dir = args.output_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = f"test_results_{timestamp}"

            save_detailed_results(test_metrics, predictions, labels, probabilities, config, save_dir)
            print(f"\n📁 Detailed results saved to: {save_dir}")

        # Summary
        print(f"\n{'='*60}")
        print(f"{'TEST SUMMARY':^60}")
        print(f"{'='*60}")
        print(f"🎯 Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"📊 F1 Score: {test_metrics['f1']:.2f}%")
        print(f"🎪 AUC-ROC: {test_metrics['auc']:.2f}%")
        print(f"⚡ Avg Inference Time: {test_metrics['avg_inference_time']:.4f}s")
        print(f"📈 Total Samples: {test_metrics['total_samples']}")
        print("✅ Testing completed successfully!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
