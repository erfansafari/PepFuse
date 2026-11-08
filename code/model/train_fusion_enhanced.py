#!/usr/bin/env python3
"""
Enhanced Fusion Model Training Script with Results Saving
========================================================

This enhanced script trains fusion models and automatically saves the best results
to text files with comprehensive tracking and analysis.

Features:
- Automatic best model tracking by accuracy, F1, and validation accuracy
- Comprehensive results saving and reporting
- Integration with FusionResultsSaver
- Enhanced metrics collection and analysis
- Automatic summary generation after training

Usage:
    python train_fusion_enhanced.py --config config_fusion.yaml
    python train_fusion_enhanced.py --config config_fusion.yaml --debug
    python train_fusion_enhanced.py --config config_fusion.yaml --resume checkpoints/fusion_model.pt
"""

import os
import sys
import argparse
import yaml
import logging
import wandb
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules
from data.fusion_dataloader import load_fusion_data, verify_fusion_data
from model.model_fusion_model import FusionModel, PeptideBERTSeqEncoder
from model.model_graph_encoder import GraphEncoder
from model.fusion_utils import FusionModelManager, print_metrics
from model.fusion_results_saver import FusionResultsSaver, save_fusion_training_results
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fusion_training_enhanced.log'),
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

    # Create handcrafted encoder if enabled
    handcrafted_encoder = None
    hand_dim = None
    hand_cfg = config.get('handcrafted', {})
    use_handcrafted = hand_cfg.get('use', False)
    
    if use_handcrafted:
        from model.model_handcrafted import HandcraftedGCNN
        hand_dim = hand_cfg.get('hidden_dim', 128)
        handcrafted_encoder = HandcraftedGCNN(
            in_dim=hand_cfg.get('in_dim', 45),
            hidden_dim=hand_dim,
            num_layers=hand_cfg.get('num_layers', 2),
            dropout=hand_cfg.get('dropout', 0.1)
        )
        logger.info(f"  Handcrafted encoder enabled: GCNN ({hand_dim}D)")

    # Create fusion model
    fusion_model = FusionModel(
        seq_encoder=seq_encoder,
        graph_encoder=graph_encoder,
        seq_dim=config['sequence']['hidden_size'],
        graph_dim=config['graph_encoder']['hidden_dim'],
        fusion=config['fusion']['fusion_type'],
        hidden=config['fusion']['hidden_dim'],
        dropout=config['fusion']['dropout'],
        num_heads=config['fusion'].get('num_heads', 4),  # Default 4 heads for cross-attention
        handcrafted_encoder=handcrafted_encoder,
        hand_dim=hand_dim
    )

    logger.info(f"Created fusion model:")
    logger.info(f"  Sequence encoder: PeptideBERT ({config['sequence']['hidden_size']}D)")
    logger.info(f"  Graph encoder: Graphormer Lite ({config['graph_encoder']['hidden_dim']}D)")
    if use_handcrafted:
        logger.info(f"  Handcrafted encoder: GCNN ({hand_dim}D)")
    logger.info(f"  Fusion type: {config['fusion']['fusion_type']}")
    if config['fusion']['fusion_type'] == 'cross_attention':
        logger.info(f"  Cross-attention heads: {config['fusion'].get('num_heads', 4)}")
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
            patience=sch_config.get('patience', 5)
        )

    logger.info(f"Created optimizer: AdamW (lr={config['optim']['lr']})")
    logger.info(f"Created scheduler: {sch_name}")

    return criterion, optimizer, scheduler


def plot_learning_curves(history: Dict, save_dir: str, task: str = ""):
    """
    Plot and save learning curves for training metrics.
    
    Args:
        history: Dictionary containing training history with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        save_dir: Directory to save the plots
        task: Task name for plot titles
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 12)
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'learning_curves')
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = history['epochs']
    
    # Create a 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Learning Curves - {task.upper()}', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss over Epochs', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy over Epochs', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: F1 Score
    ax = axes[0, 2]
    ax.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_f1'], 'r-s', label='Val F1', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score over Epochs', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 4: Precision
    ax = axes[1, 0]
    ax.plot(epochs, history['train_precision'], 'b-o', label='Train Precision', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_precision'], 'r-s', label='Val Precision', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax.set_title('Precision over Epochs', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 5: Recall
    ax = axes[1, 1]
    ax.plot(epochs, history['train_recall'], 'b-o', label='Train Recall', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_recall'], 'r-s', label='Val Recall', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Recall over Epochs', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 6: Learning Rate
    ax = axes[1, 2]
    ax.plot(epochs, history['lr'], 'g-o', label='Learning Rate', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined plot
    combined_path = os.path.join(plots_dir, 'all_metrics.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Combined learning curves saved to: {combined_path}")
    plt.close()
    
    # Create individual plots for each metric
    # 1. Loss (individual)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2.5, markersize=6)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2.5, markersize=6)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Validation Loss - {task.upper()}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy (individual)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2.5, markersize=6)
    ax.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2.5, markersize=6)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Validation Accuracy - {task.upper()}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 Score (individual)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2.5, markersize=6)
    ax.plot(epochs, history['val_f1'], 'r-s', label='Validation F1', linewidth=2.5, markersize=6)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Training and Validation F1 Score - {task.upper()}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_score.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision & Recall (combined)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_precision'], 'b-o', label='Train Precision', linewidth=2, markersize=5)
    ax.plot(epochs, history['val_precision'], 'r-s', label='Val Precision', linewidth=2, markersize=5)
    ax.plot(epochs, history['train_recall'], 'g-^', label='Train Recall', linewidth=2, markersize=5)
    ax.plot(epochs, history['val_recall'], 'm-d', label='Val Recall', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision and Recall - {task.upper()}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Learning Rate (individual)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['lr'], 'g-o', label='Learning Rate', linewidth=2.5, markersize=6)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'Learning Rate Schedule - {task.upper()}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    logger.info(f"✓ Individual plots saved:")
    logger.info(f"  - Loss: {os.path.join(plots_dir, 'loss.png')}")
    logger.info(f"  - Accuracy: {os.path.join(plots_dir, 'accuracy.png')}")
    logger.info(f"  - F1 Score: {os.path.join(plots_dir, 'f1_score.png')}")
    logger.info(f"  - Precision & Recall: {os.path.join(plots_dir, 'precision_recall.png')}")
    logger.info(f"  - Learning Rate: {os.path.join(plots_dir, 'learning_rate.png')}")
    
    # Save history to file
    history_file = os.path.join(plots_dir, 'training_history.npz')
    np.savez(history_file, **history)
    logger.info(f"✓ Training history saved to: {history_file}")
    
    print(f"\n{'='*80}")
    print(f"📊 LEARNING CURVES SAVED")
    print(f"{'='*80}")
    print(f"Location: {plots_dir}/")
    print(f"Files:")
    print(f"  • all_metrics.png         - All metrics in one view")
    print(f"  • loss.png                - Training and validation loss")
    print(f"  • accuracy.png            - Training and validation accuracy")
    print(f"  • f1_score.png            - Training and validation F1 score")
    print(f"  • precision_recall.png    - Precision and recall metrics")
    print(f"  • learning_rate.png       - Learning rate schedule")
    print(f"  • training_history.npz    - Raw data for all metrics")
    print(f"{'='*80}\n")

    return plots_dir


def train_fusion_model_enhanced(config: Dict, args: argparse.Namespace):
    """
    Enhanced training function for fusion model with comprehensive results saving.

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

    # Initialize results saver
    results_saver = FusionResultsSaver()

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

        # Create model manager (pass config for SupCon support)
        manager = FusionModelManager(model, device, config=config)

        # Resume training if requested
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            start_epoch, _ = manager.load_checkpoint(args.resume, optimizer, scheduler)

        # Initialize W&B if not in debug mode
        if not args.debug:
            try:
                # Set offline mode to avoid connection errors
                os.environ['WANDB_MODE'] = 'offline'
                wandb.init(
                    project=config.get('logging', {}).get('wandb_project', 'PeptideBERT-Fusion'),
                    name=run_name,
                    config=config,
                    mode='offline'
                )
                logger.info("W&B initialized in offline mode")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}. Continuing without W&B logging.")
                args.debug = True  # Disable W&B logging for this run

        # Training loop
        logger.info("Starting enhanced fusion training...")
        print(f"\n{'='*70}")
        print(f"{'ENHANCED FUSION MODEL TRAINING':^70}")
        print(f"{'='*70}")

        best_val_f1 = 0.0
        best_val_acc = 0.0
        best_test_acc = 0.0
        patience_counter = 0
        max_patience = 8

        # Storage for best metrics across training
        best_train_metrics = {}
        best_val_metrics = {}
        best_test_metrics = {}

        # Storage for learning curves
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'train_precision': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'epochs': [],
            'lr': []
        }

        for epoch in range(start_epoch, config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print("-" * 60)

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

            # Store metrics for learning curves
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_precision'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['epochs'].append(epoch + 1)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # Log to W&B
            if not args.debug:
                log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch + 1
                log_dict['lr'] = optimizer.param_groups[0]['lr']
                wandb.log(log_dict)

            # Check for best models and save accordingly
            is_best_f1 = manager.update_best_metrics(val_metrics, 'f1')
            is_best_val_acc = val_metrics['accuracy'] > best_val_acc

            if is_best_f1:
                best_val_f1 = val_metrics['f1']
                best_f1_model_path = os.path.join(save_dir, "best_f1_model.pt")
                manager.save_checkpoint(
                    best_f1_model_path, epoch + 1, optimizer, scheduler,
                    val_metrics, config
                )
                print(f"\n🎯 New best F1 model saved! F1: {best_val_f1:.2f}%")

            if is_best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_model_path = os.path.join(save_dir, "best_val_accuracy_model.pt")
                manager.save_checkpoint(
                    best_val_model_path, epoch + 1, optimizer, scheduler,
                    val_metrics, config
                )
                print(f"\n🏆 New best validation accuracy model saved! Val Acc: {best_val_acc:.2f}%")

            # Update best metrics for final saving
            if is_best_f1 or is_best_val_acc:
                best_train_metrics = train_metrics.copy()
                best_val_metrics = val_metrics.copy()
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

        # Load best F1 model for testing
        best_f1_model_path = os.path.join(save_dir, "best_f1_model.pt")
        if os.path.exists(best_f1_model_path):
            logger.info("Loading best F1 model for testing...")
            manager.load_checkpoint(best_f1_model_path)

        # Final testing on best F1 model
        logger.info("Running final evaluation on best F1 model...")
        test_metrics_f1 = manager.test(test_loader, device)

        # Load best validation accuracy model for testing
        best_val_model_path = os.path.join(save_dir, "best_val_accuracy_model.pt")
        test_metrics_val = test_metrics_f1  # Default to F1 model results

        if os.path.exists(best_val_model_path) and best_val_model_path != best_f1_model_path:
            logger.info("Loading best validation accuracy model for testing...")
            manager.load_checkpoint(best_val_model_path)
            test_metrics_val = manager.test(test_loader, device)

        # Determine which model performed better on test set
        if test_metrics_f1['accuracy'] >= test_metrics_val['accuracy']:
            best_test_metrics = test_metrics_f1.copy()
            best_model_path = best_f1_model_path
            best_model_type = "F1"
        else:
            best_test_metrics = test_metrics_val.copy()
            best_model_path = best_val_model_path
            best_model_type = "Validation Accuracy"

        # Save the overall best model
        overall_best_path = os.path.join(save_dir, "best_overall_model.pt")
        if best_model_path != overall_best_path:
            import shutil
            shutil.copy2(best_model_path, overall_best_path)

        print(f"\n{'='*70}")
        print(f"{'FINAL TEST RESULTS':^70}")
        print(f"{'='*70}")
        print(f"Best Model Type: {best_model_type}")
        print(f"Best Model Path: {best_model_path}")
        print_metrics(best_test_metrics, "Best Test Results")

        # Also print results for the other model if different
        if best_model_type == "F1" and test_metrics_val != test_metrics_f1:
            print_metrics(test_metrics_val, "Val Accuracy Model Test Results")
        elif best_model_type == "Validation Accuracy" and test_metrics_f1 != test_metrics_val:
            print_metrics(test_metrics_f1, "F1 Model Test Results")

        # Plot and save learning curves
        logger.info("Plotting learning curves...")
        plot_learning_curves(history, save_dir, task=config['task'])

        # Save comprehensive results using FusionResultsSaver
        logger.info("Saving comprehensive results...")

        # Use the best training and validation metrics from the epoch that produced the best model
        result_file = results_saver.save_run_results(
            train_metrics=best_train_metrics if best_train_metrics else train_metrics,
            val_metrics=best_val_metrics if best_val_metrics else val_metrics,
            test_metrics=best_test_metrics,
            config=config,
            model_path=overall_best_path,
            run_name=run_name
        )

        # Generate all summary reports
        logger.info("Generating summary reports...")
        summary_report = results_saver.generate_summary_report()
        best_models_summary = results_saver.save_best_models_summary()
        csv_export = results_saver.export_results_csv()

        # Log final results to W&B
        if not args.debug:
            final_log = {f'final_test_{k}': v for k, v in best_test_metrics.items()}
            final_log['best_model_type'] = best_model_type
            wandb.log(final_log)
            wandb.finish()

        # Print summary of saved files
        print(f"\n{'='*70}")
        print(f"{'TRAINING COMPLETED SUCCESSFULLY!':^70}")
        print(f"{'='*70}")
        print(f"🏆 Best Model Type: {best_model_type}")
        print(f"📊 Best Test Accuracy: {best_test_metrics['accuracy']:.2f}%")
        print(f"🎯 Best Test F1 Score: {best_test_metrics['f1']:.2f}%")
        print(f"🔗 Best Model Path: {overall_best_path}")
        print(f"\n📁 Results Files Generated:")
        print(f"   - Individual Run: {result_file}")
        print(f"   - Summary Report: {summary_report}")
        print(f"   - Best Models: {best_models_summary}")
        print(f"   - CSV Export: {csv_export}")
        print(f"   - Training Logs: fusion_training_enhanced.log")
        print(f"\n✨ All results have been saved to 'txt file' directory!")

        # Print quick stats
        stats = results_saver.get_summary_stats()
        if stats and stats.get('total_runs', 0) > 1:
            print(f"\n📈 Overall Performance Summary ({stats['total_runs']} runs):")
            print(f"   - Average Test Accuracy: {stats['test_accuracy']['mean']:.2f}% ± {stats['test_accuracy']['std']:.2f}%")
            print(f"   - Average Test F1: {stats['test_f1']['mean']:.2f}% ± {stats['test_f1']['std']:.2f}%")
            print(f"   - Best Ever Accuracy: {stats['test_accuracy']['max']:.2f}%")
            print(f"   - Best Ever F1: {stats['test_f1']['max']:.2f}%")

        logger.info("Enhanced fusion training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if not args.debug:
            wandb.finish()
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Fusion Model Training with Results Saving")
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
    parser.add_argument(
        '--show-summary',
        action='store_true',
        help='Show summary of previous fusion model results and exit'
    )

    args = parser.parse_args()

    # Show summary and exit if requested
    if args.show_summary:
        from model.fusion_results_saver import print_fusion_summary
        print_fusion_summary()
        return

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

    # Run enhanced training
    train_fusion_model_enhanced(config, args)


if __name__ == "__main__":
    main()
