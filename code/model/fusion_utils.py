"""
Fusion Model Training Utilities
==============================

This module provides training, validation, and testing utilities specifically designed
for the fusion model that combines PeptideBERT (sequence encoder) and GraphEncoder.

Key functions:
- train_fusion_epoch: Training loop for one epoch
- validate_fusion: Validation loop
- test_fusion: Testing loop with detailed metrics
- FusionModelManager: High-level model management class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


def train_fusion_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    supcon_loss: Optional[nn.Module] = None,
    supcon_lambda: float = 0.0
) -> Dict[str, float]:
    """
    Train the fusion model for one epoch with optional supervised contrastive loss.

    Args:
        model: Fusion model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (BCE)
        scheduler: Learning rate scheduler (optional)
        device: Device to run on
        epoch: Current epoch number
        log_interval: How often to log progress
        supcon_loss: Optional supervised contrastive loss module
        supcon_lambda: Weight for the contrastive loss term (if enabled)

    Returns:
        Dictionary with training metrics including BCE loss, SupCon loss (if used), and total loss
    """
    model.train()

    total_bce_loss = 0.0
    total_supcon_loss = 0.0
    total_combined_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []

    # Check if SupCon is enabled
    use_supcon = (supcon_loss is not None and supcon_lambda > 0.0)

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        batch_graph = batch['batch_graph'].to(device)
        
        # Get handcrafted features if available
        classic_feats = batch.get('classic_feats')
        if classic_feats is not None:
            classic_feats = classic_feats.to(device)

        # Zero gradients
        optimizer.zero_grad()

        try:
            # Forward pass for classification
            logits = model(batch_graph, input_ids, attention_mask, classic_feats)

            # Compute BCE loss
            if labels.dtype == torch.long:
                # Convert to float for BCE loss
                labels_float = labels.float()
            else:
                labels_float = labels

            bce_loss = criterion(logits, labels_float)

            # Compute supervised contrastive loss if enabled
            if use_supcon:
                # Get fused representation (allows gradients to flow)
                z = model.get_representation(batch_graph, input_ids, attention_mask, classic_feats)
                
                # SupCon expects integer labels
                labels_int = labels.long() if labels.dtype != torch.long else labels
                
                # Compute contrastive loss
                contrast_loss = supcon_loss(z, labels_int)
                
                # Combined loss
                loss = bce_loss + supcon_lambda * contrast_loss
                
                # Track individual losses
                total_supcon_loss += contrast_loss.item() * input_ids.size(0)
            else:
                loss = bce_loss
                contrast_loss = torch.tensor(0.0)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for fusion models)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Update scheduler if provided and it's a step-based scheduler
            if scheduler is not None and hasattr(scheduler, 'step') and not hasattr(scheduler, 'patience'):
                scheduler.step()

            # Compute metrics
            batch_size = input_ids.size(0)
            total_bce_loss += bce_loss.item() * batch_size
            total_combined_loss += loss.item() * batch_size
            total_samples += batch_size

            # Convert logits to predictions
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct_predictions += (predictions == labels_float).sum().item()

            # Store for epoch-level metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_float.cpu().numpy())

            # Update progress bar
            avg_loss = total_combined_loss / total_samples
            accuracy = correct_predictions / total_samples * 100
            
            if use_supcon:
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'BCE': f'{total_bce_loss/total_samples:.4f}',
                    'SCL': f'{total_supcon_loss/total_samples:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })

            # Log progress
            if batch_idx % log_interval == 0:
                if use_supcon:
                    logger.debug(
                        f'Train Epoch: {epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} '
                        f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                        f'Total Loss: {loss.item():.6f} (BCE: {bce_loss.item():.6f}, '
                        f'SupCon: {contrast_loss.item():.6f})'
                    )
                else:
                    logger.debug(
                        f'Train Epoch: {epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} '
                        f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                        f'Loss: {loss.item():.6f}'
                    )

        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {e}")
            # Skip this batch and continue
            continue

    # Compute final metrics
    avg_bce_loss = total_bce_loss / total_samples
    avg_combined_loss = total_combined_loss / total_samples
    accuracy = correct_predictions / total_samples * 100

    # Compute additional metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    try:
        precision = precision_score(all_labels, all_predictions, zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, zero_division=0) * 100
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        precision = recall = f1 = 0.0

    metrics = {
        'loss': avg_combined_loss,  # Total loss (BCE + λ*SupCon)
        'bce_loss': avg_bce_loss,   # BCE loss only
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Add SupCon loss to metrics if it was used
    if use_supcon:
        avg_supcon_loss = total_supcon_loss / total_samples
        metrics['supcon_loss'] = avg_supcon_loss

    return metrics


def validate_fusion(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the fusion model.

    Args:
        model: Fusion model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            batch_graph = batch['batch_graph'].to(device)
            
            # Get handcrafted features if available
            classic_feats = batch.get('classic_feats')
            if classic_feats is not None:
                classic_feats = classic_feats.to(device)

            try:
                # Forward pass
                logits = model(batch_graph, input_ids, attention_mask, classic_feats)

                # Compute loss
                if labels.dtype == torch.long:
                    labels = labels.float()

                loss = criterion(logits, labels)

                # Compute metrics
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Convert logits to predictions and probabilities
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()

                # Store for epoch-level metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

            except Exception as e:
                logger.error(f"Error in validation batch: {e}")
                continue

    # Compute final metrics
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples * 100

    # Compute additional metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    try:
        precision = precision_score(all_labels, all_predictions, zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, zero_division=0) * 100

        # AUC-ROC if we have both classes
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probabilities) * 100
        else:
            auc = 0.0

    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        precision = recall = f1 = auc = 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics


def test_fusion(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Test the fusion model and return detailed metrics.

    Args:
        model: Fusion model
        dataloader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with test metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            batch_graph = batch['batch_graph'].to(device)
            
            # Get handcrafted features if available
            classic_feats = batch.get('classic_feats')
            if classic_feats is not None:
                classic_feats = classic_feats.to(device)

            try:
                # Time inference
                start_time = time.time()

                # Forward pass
                logits = model(batch_graph, input_ids, attention_mask, classic_feats)

                end_time = time.time()
                inference_times.append(end_time - start_time)

                # Convert logits to predictions and probabilities
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

            except Exception as e:
                logger.error(f"Error in test batch: {e}")
                continue

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    accuracy = (all_predictions == all_labels).mean() * 100

    try:
        precision = precision_score(all_labels, all_predictions, zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, zero_division=0) * 100

        # AUC-ROC
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probabilities) * 100
        else:
            auc = 0.0

    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        precision = recall = f1 = auc = 0.0

    # Compute inference statistics
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'avg_inference_time': avg_inference_time,
        'total_samples': len(all_labels)
    }

    return metrics


class FusionModelManager:
    """
    High-level manager for fusion model training and evaluation.
    
    Supports optional supervised contrastive loss alongside BCE loss.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: Optional[Dict] = None):
        self.model = model
        self.device = device
        self.best_metrics = {}
        
        # Initialize supervised contrastive loss if enabled in config
        self.supcon_loss = None
        self.supcon_lambda = 0.0
        
        if config is not None:
            loss_config = config.get("loss", {})
            use_supcon = loss_config.get("use_supcon", False)
            
            if use_supcon:
                try:
                    from model.losses import SupConLoss
                    
                    temperature = loss_config.get("supcon_temperature", 0.07)
                    self.supcon_lambda = loss_config.get("supcon_lambda", 0.1)
                    self.supcon_loss = SupConLoss(temperature=temperature).to(device)
                    
                    logger.info(f"✓ Supervised Contrastive Loss enabled:")
                    logger.info(f"  - Temperature: {temperature}")
                    logger.info(f"  - Lambda (weight): {self.supcon_lambda}")
                except ImportError as e:
                    logger.error(f"Failed to import SupConLoss: {e}")
                    logger.warning("Continuing without supervised contrastive loss")

    def train_epoch(self, train_loader, optimizer, criterion, scheduler, device, epoch, log_interval=10):
        """
        Train for one epoch with optional supervised contrastive loss.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: BCE loss criterion
            scheduler: Learning rate scheduler
            device: Device
            epoch: Current epoch number
            log_interval: Logging interval
            
        Returns:
            Dictionary with training metrics
        """
        return train_fusion_epoch(
            self.model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            epoch,
            log_interval=log_interval,
            supcon_loss=self.supcon_loss,
            supcon_lambda=self.supcon_lambda
        )

    def validate(self, *args, **kwargs):
        """Validate the model."""
        return validate_fusion(self.model, *args, **kwargs)

    def test(self, *args, **kwargs):
        """Test the model."""
        return test_fusion(self.model, *args, **kwargs)

    def save_checkpoint(self, filepath: str, epoch: int, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], metrics: Dict[str, float], config: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

    def update_best_metrics(self, metrics: Dict[str, float], metric_key: str = 'f1'):
        """Update best metrics if current metrics are better."""
        current_score = metrics.get(metric_key, 0)
        best_score = self.best_metrics.get(metric_key, 0)

        if current_score > best_score:
            self.best_metrics = metrics.copy()
            return True
        return False


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")

    print("-" * 40)
    for key, value in metrics.items():
        if 'time' in key.lower():
            print(f"{key.capitalize():15}: {value:.4f}s")
        elif 'samples' in key.lower():
            print(f"{key.capitalize():15}: {int(value)}")
        else:
            print(f"{key.capitalize():15}: {value:.2f}%")
    print("-" * 40)
