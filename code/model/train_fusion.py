#!/usr/bin/env python3
"""
Fusion Model Training Script: PeptideBERT + Graphormer Lite
==========================================================

This script trains a fusion model that combines:
- PeptideBERT (sequence encoder) for peptide sequence understanding
- Graphormer Lite (graph encoder) for peptide structural graph understanding

The fusion model learns to combine both sequence and structural information
for improved peptide property prediction.

Usage:
    python train_fusion.py --config config_fusion.yaml
    python train_fusion.py --config config_fusion.yaml --debug
    python train_fusion.py --config config_fusion.yaml --resume checkpoints/fusion_model.pt
"""

import os
import sys
import argparse
import yaml
import logging
import wandb
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data.fusion_dataloader import load_fusion_data, verify_fusion_data
from model.model_fusion_model import FusionModel, PeptideBERTSeqEncoder
from model.model_graph_encoder import GraphEncoder
from model.fusion_utils import FusionModelManager, print_metrics
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fusion_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        d_out=config['sequence']['hidden_size']  # Keep same dimension
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

    logger.info(f"Created fusion model:")
    logger.info(f"  Sequence encoder: PeptideBERT ({config['sequence']['hidden_size']}D)")
    logger.info(f"  Graph encoder: Graphormer Lite ({config['graph_encoder']['hidden_dim']}D)")
    logger.info(f"  Fusion type: {config['fusion']['fusion_type']}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")

    return fusion_model


def create_optimizer_and_scheduler(model: nn.Module, config: Dict, steps_per_epoch: int):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        config: Configuration dictionary
        steps_per_epoch: Number of training steps per epoch

    Returns:
        Tuple of (criterion, optimizer, scheduler)
    """
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optim']['lr'],
        weight_decay=1e-5
    )

    # Scheduler
    scheduler = None
    sch_config = config.get('sch', {})
    sch_name = sch_config.get('name', 'lronplateau')

    if sch_name == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch
        )
    elif sch_name == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=sch_config.get('factor', 0.1),
            patience=sch_config.get('patience', 5),
            verbose=True
        )

    logger.info(f"Created optimizer: AdamW (lr={config['optim']['lr']})")
    logger.info(f"Created scheduler: {sch_name}")

    return criterion, optimizer, scheduler


def save_results(metrics: Dict[str, float], config: Dict, save_dir: str,
                prefix: str = "results"):
    """Save training/validation/test results to file."""
    results_file = os.path.join(save_dir, f"{prefix}.txt")

    with open(results_file, 'w') as f:
        f.write(f"# Fusion Model {prefix.title()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n")
        f.write(f"- Task: {config['task']}\n")
        f.write(f"- Batch Size: {config['batch_size']}\n")
        f.write(f"- Learning Rate: {config['optim']['lr']}\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Fusion Type: {config['fusion']['fusion_type']}\n")
        f.write(f"- Graph K-hop: {config['graph']['k_hop']}\n")
        f.write(f"- Sequence Hidden Size: {config['sequence']['hidden_size']}\n")
        f.write(f"- Graph Hidden Size: {config['graph_encoder']['hidden_dim']}\n\n")

        f.write("## Performance Metrics\n")
        for key, value in metrics.items():
            if 'time' in key.lower():
                f.write(f"- {key.title()}: {value:.4f}s\n")
            elif 'samples' in key.lower():
                f.write(f"- {key.title()}: {int(value)}\n")
            else:
                f.write(f"- {key.title()}: {value:.2f}%\n")


def train_fusion_model(config: Dict, args: argparse.Namespace):
    """
    Main training function for fusion model.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create save directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"fusion_{config['task']}_{timestamp}"
    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Save directory: {save_dir}")

    try:
        # Verify data first
        logger.info("Verifying fusion data loading...")
        verify_fusion_data(config, num_batches=1)

        # Load data
        logger.info("Loading fusion data...")
        train_loader, val_loader, test_loader = load_fusion_data(config)
        config['sch']['steps'] = len(train_loader)

        # Create model
        logger.info("Creating fusion model...")
        model = create_fusion_model(config)
        model = model.to(device)

        # Create optimizer and scheduler
        criterion, optimizer, scheduler = create_optimizer_and_scheduler(
            model, config, len(train_loader)
        )

        # Create model manager
        manager = FusionModelManager(model, device)

        # Resume training if requested
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            start_epoch, _ = manager.load_checkpoint(args.resume, optimizer, scheduler)

        # Initialize W&B if not in debug mode
        if not args.debug:
            wandb.init(
                project=config.get('logging', {}).get('wandb_project', 'PeptideBERT-Fusion'),
                name=run_name,
                config=config
            )

        # Training loop
        logger.info("Starting training...")
        print(f"\n{'='*60}")
        print(f"{'FUSION MODEL TRAINING':^60}")
        print(f"{'='*60}")

        best_val_f1 = 0.0
        patience_counter = 0
        max_patience = 10

        for epoch in range(start_epoch, config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print("-" * 50)

            # Training
            train_metrics = manager.train_epoch(
                train_loader, optimizer, criterion, scheduler, device,
                epoch + 1, config.get('logging', {}).get('log_interval', 10)
            )

            print_metrics(train_metrics, "Training")

            # Validation
            val_metrics = manager.validate(val_loader, criterion, device)
            print_metrics(val_metrics, "Validation")

            # Update learning rate scheduler
            if scheduler is not None and hasattr(scheduler, 'patience'):
                scheduler.step(val_metrics['f1'])

            # Log to W&B
            if not args.debug:
                log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch + 1
                log_dict['lr'] = optimizer.param_groups[0]['lr']
                wandb.log(log_dict)

            # Save best model
            is_best = manager.update_best_metrics(val_metrics, 'f1')
            if is_best:
                best_val_f1 = val_metrics['f1']
                best_model_path = os.path.join(save_dir, "best_model.pt")
                manager.save_checkpoint(
                    best_model_path, epoch + 1, optimizer, scheduler,
                    val_metrics, config
                )
                print(f"\n🎉 New best model saved! F1: {best_val_f1:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {max_patience} epochs without improvement")
                break

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                manager.save_checkpoint(
                    checkpoint_path, epoch + 1, optimizer, scheduler,
                    val_metrics, config
                )

        # Load best model for testing
        best_model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            logger.info("Loading best model for testing...")
            manager.load_checkpoint(best_model_path)

        # Final testing
        logger.info("Running final evaluation...")
        test_metrics = manager.test(test_loader, device)

        print(f"\n{'='*60}")
        print(f"{'FINAL TEST RESULTS':^60}")
        print(f"{'='*60}")
        print_metrics(test_metrics, "Test")

        # Save results
        save_results(train_metrics, config, save_dir, "final_train")
        save_results(val_metrics, config, save_dir, "final_val")
        save_results(test_metrics, config, save_dir, "final_test")

        # Log final results to W&B
        if not args.debug:
            final_log = {f'final_test_{k}': v for k, v in test_metrics.items()}
            wandb.log(final_log)
            wandb.finish()

        logger.info(f"Training completed! Results saved to {save_dir}")
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {save_dir}")
        print(f"🏆 Best validation F1: {best_val_f1:.2f}%")
        print(f"🎯 Final test F1: {test_metrics['f1']:.2f}%")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if not args.debug:
            wandb.finish()
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train PeptideBERT + Graphormer Fusion Model")
    parser.add_argument(
        '--config',
        type=str,
        default='config_fusion.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (no W&B logging, faster)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--verify-data',
        action='store_true',
        help='Only verify data loading and exit'
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with debug mode
    config['debug'] = args.debug

    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Task: {config['task']}")
    logger.info(f"Debug mode: {args.debug}")

    # Verify data loading only
    if args.verify_data:
        logger.info("Verifying data loading only...")
        verify_fusion_data(config)
        logger.info("Data verification completed successfully!")
        return

    # Run training
    train_fusion_model(config, args)


if __name__ == "__main__":
    main()
