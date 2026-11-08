#!/usr/bin/env python3
"""
Quick Demo Training Script for Fusion Model
==========================================

This script runs a quick demonstration of the fusion model training
with minimal epochs and simplified logging for testing purposes.
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fusion_dataloader import load_fusion_data
from model.model_fusion_model import FusionModel, DummySeqEncoder
from model.model_graph_encoder import GraphEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_config():
    """Create a minimal configuration for demo training."""
    return {
        'task': 'nf',
        'batch_size': 4,
        'epochs': 3,
        'vocab_size': 25,
        'debug': True,

        'graph': {
            'k_hop': 1,
            'readout': 'mean'
        },

        'sequence': {
            'hidden_size': 128,
            'hidden_layers': 2,
            'attn_heads': 4,
            'dropout': 0.1
        },

        'graph_encoder': {
            'hidden_dim': 64,
            'heads': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'k_max': 2,
            'max_degree': 32,
            'use_degree': True
        },

        'fusion': {
            'fusion_type': 'concat',
            'hidden_dim': 64,
            'dropout': 0.1
        },

        'optim': {
            'lr': 1e-3
        },

        'data': {
            'num_workers': 0,
            'pin_memory': False
        }
    }

def create_simple_model(config):
    """Create a simple fusion model for demo."""
    # Use dummy sequence encoder to avoid pretrained model loading
    seq_encoder = DummySeqEncoder(
        vocab_size=config['vocab_size'],
        d_model=config['sequence']['hidden_size'],
        nhead=config['sequence']['attn_heads'],
        nlayers=config['sequence']['hidden_layers']
    )

    # Create graph encoder
    graph_encoder = GraphEncoder(
        vocab_size=config['vocab_size'],
        hidden_dim=config['graph_encoder']['hidden_dim'],
        heads=config['graph_encoder']['heads'],
        num_layers=config['graph_encoder']['num_layers'],
        dropout=config['graph_encoder']['dropout'],
        k_max=config['graph_encoder']['k_max'],
        max_degree=config['graph_encoder']['max_degree'],
        readout=config['graph']['readout'],
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

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"Epoch {epoch + 1} - Training...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:  # Limit to 10 batches for demo
            break

        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        batch_graph = batch['batch_graph'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_graph, input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / min(10, len(dataloader))
    accuracy = 100 * correct / total

    print(f"  Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    print("Validating...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Limit to 5 batches for demo
                break

            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            batch_graph = batch['batch_graph'].to(device)

            # Forward pass
            logits = model(batch_graph, input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / min(5, len(dataloader))
    accuracy = 100 * correct / total

    print(f"  Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def main():
    """Main demo function."""
    print("=" * 60)
    print("FUSION MODEL TRAINING DEMO")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create config
    config = create_demo_config()
    print(f"Task: {config['task']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")

    try:
        # Load data
        print("\nLoading data...")
        train_loader, val_loader, test_loader = load_fusion_data(config)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Create model
        print("\nCreating model...")
        model = create_simple_model(config).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Create optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optim']['lr'])
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        print("\nStarting training...")
        best_val_acc = 0

        for epoch in range(config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  New best validation accuracy: {val_acc:.2f}%")

        # Final test
        print("\n--- Final Test ---")
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        # Summary
        print("\n" + "=" * 60)
        print("DEMO TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Final test accuracy: {test_acc:.2f}%")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
