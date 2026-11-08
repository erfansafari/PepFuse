"""
Fusion DataLoader: Handles both sequence and graph data for PeptideBERT + Graphormer fusion
========================================================================================

This module provides data loading functionality for the fusion model that combines:
- Sequence data (tokenized peptides) for PeptideBERT
- Graph data (peptide graphs) for GraphEncoder

The loader ensures that sequence and graph data are properly aligned and batched together.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Dict, List, Tuple, Optional
import logging

# Import existing components
import numpy as np
from .graph_data_builder import get_graph_loaders, build_all

logger = logging.getLogger(__name__)


class FusionPeptideDataset(Dataset):
    """
    Dataset that provides both sequence and graph representations of peptides.

    Args:
        task: Task name ('hemo', 'sol', 'nf')
        split: Data split ('train', 'val', 'test')
        k_hop: Maximum hop distance for graph edges
        transform: Optional transform to apply to data
    """

    def __init__(self, task: str, split: str, k_hop: int = 2, transform=None):
        self.task = task
        self.split = split
        self.k_hop = k_hop
        self.transform = transform

        # Load sequence data directly from NPZ files
        npz_path = f"./data/{task}/{split}.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ data not found: {npz_path}")

        with np.load(npz_path) as data:
            self.input_ids = torch.tensor(data['inputs'], dtype=torch.long)
            self.labels = torch.tensor(data['labels'], dtype=torch.float)
            self.attention_masks = torch.tensor(data['inputs'] > 0, dtype=torch.long)
            
            # Load handcrafted features if available
            if 'classic_feats' in data:
                self.classic_feats = torch.tensor(data['classic_feats'], dtype=torch.float)
                logger.info(f"Loaded classic_feats with shape: {self.classic_feats.shape}")
            else:
                self.classic_feats = None
                logger.info("No classic_feats found in NPZ file")

        # Ensure graph data exists with correct k_hop
        self._ensure_graph_data()

        # Load graph data
        graph_path = f"./data/{task}/graphs_{split}.pt"
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph data not found: {graph_path}")

        self.graphs = torch.load(graph_path, weights_only=False)

        # Verify graph data was built with correct k_hop by checking a sample
        if len(self.graphs) > 0:
            self._verify_graph_khop()

        # Verify alignment
        if len(self.input_ids) != len(self.graphs):
            raise ValueError(
                f"Sequence and graph data length mismatch: "
                f"seq={len(self.input_ids)}, graph={len(self.graphs)}"
            )

        logger.info(f"Loaded {len(self)} samples for {task}/{split}")

    def _ensure_graph_data(self):
        """Ensure graph data exists, build if necessary."""
        graph_path = f"./data/{self.task}/graphs_{self.split}.pt"
        if not os.path.exists(graph_path):
            logger.info(f"Graph data not found, building for {self.task} with k_hop={self.k_hop}")
            build_all(task=self.task, k_hop=self.k_hop)

    def _verify_graph_khop(self):
        """Verify that existing graphs match the requested k_hop parameter."""
        from .graph_data_builder import sequence_to_graph

        # Test with first sequence to see expected edge count
        test_graph = sequence_to_graph(self.input_ids[0].numpy(), self.labels[0].item(), self.k_hop)
        existing_graph = self.graphs[0]

        # Compare edge counts - if they don't match, rebuild
        if test_graph.edge_index.shape[1] != existing_graph.edge_index.shape[1]:
            logger.warning(f"Graph k_hop mismatch detected. Expected {test_graph.edge_index.shape[1]} edges, "
                          f"got {existing_graph.edge_index.shape[1]}. Rebuilding graphs with k_hop={self.k_hop}")
            build_all(task=self.task, k_hop=self.k_hop)

            # Reload the rebuilt graphs
            graph_path = f"./data/{self.task}/graphs_{self.split}.pt"
            self.graphs = torch.load(graph_path, weights_only=False)
            logger.info(f"Rebuilt and reloaded graphs for {self.task}/{self.split} with k_hop={self.k_hop}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Get sequence data
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]

        # Get corresponding graph
        graph = self.graphs[idx]

        # Create combined item
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'graph': graph
        }
        
        # Add handcrafted features if available
        if self.classic_feats is not None:
            item['classic_feats'] = self.classic_feats[idx]

        if self.transform:
            item = self.transform(item)

        return item


class FusionCollator:
    """
    Custom collator for fusion data that handles both sequence tensors and graph batching.
    """

    def __call__(self, batch: List[Dict]) -> Dict:
        # Extract components
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        graphs = [item['graph'] for item in batch]

        # Stack sequence data
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)

        # Batch graphs using PyG's Batch
        batch_graph = Batch.from_data_list(graphs)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'batch_graph': batch_graph
        }
        
        # Add handcrafted features if present in batch
        if 'classic_feats' in batch[0]:
            classic_feats = [item['classic_feats'] for item in batch]
            result['classic_feats'] = torch.stack(classic_feats)
        
        return result


def load_fusion_data(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load fusion data for training, validation, and testing.

    Args:
        config: Configuration dictionary containing data parameters

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    task = config['task']
    batch_size = config['batch_size']
    k_hop = config.get('graph', {}).get('k_hop', 2)
    num_workers = config.get('data', {}).get('num_workers', 0)
    pin_memory = config.get('data', {}).get('pin_memory', False)

    # Create datasets
    train_dataset = FusionPeptideDataset(task, 'train', k_hop)
    val_dataset = FusionPeptideDataset(task, 'val', k_hop)
    test_dataset = FusionPeptideDataset(task, 'test', k_hop)

    # Create collator
    collator = FusionCollator()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created fusion data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


def verify_fusion_data(config: Dict, num_batches: int = 2):
    """
    Verify that fusion data loading works correctly.

    Args:
        config: Configuration dictionary
        num_batches: Number of batches to check
    """
    logger.info("Verifying fusion data loading...")

    try:
        train_loader, val_loader, test_loader = load_fusion_data(config)

        # Check a few batches
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            logger.info(f"Batch {i+1}:")
            logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
            logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")
            logger.info(f"  labels shape: {batch['labels'].shape}")
            logger.info(f"  batch_graph.x shape: {batch['batch_graph'].x.shape}")
            logger.info(f"  batch_graph.edge_index shape: {batch['batch_graph'].edge_index.shape}")
            logger.info(f"  batch_graph.batch shape: {batch['batch_graph'].batch.shape}")
            logger.info(f"  Number of graphs in batch: {batch['batch_graph'].num_graphs}")

            # Verify alignment
            expected_graphs = batch['input_ids'].shape[0]
            actual_graphs = batch['batch_graph'].num_graphs
            if expected_graphs != actual_graphs:
                raise ValueError(
                    f"Batch size mismatch: expected {expected_graphs} graphs, "
                    f"got {actual_graphs}"
                )

        logger.info("✓ Fusion data loading verification successful!")

    except Exception as e:
        logger.error(f"✗ Fusion data loading verification failed: {e}")
        raise


if __name__ == "__main__":
    # Test the fusion data loader
    import yaml

    # Load config
    config_path = "./config_fusion.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback config for testing
        config = {
            'task': 'nf',
            'batch_size': 4,
            'graph': {'k_hop': 2},
            'data': {'num_workers': 0, 'pin_memory': False}
        }

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Verify data loading
    verify_fusion_data(config)
