#!/usr/bin/env python3
"""
Graph Data Preparation Script for Fusion Model Training
======================================================

This script prepares graph data for all peptide tasks (hemo, sol, nf) by converting
sequence data into graph representations suitable for the Graphormer Lite encoder.

Usage:
    python prepare_graph_data.py --all                    # Prepare all tasks
    python prepare_graph_data.py --task hemo --k 2        # Prepare specific task
    python prepare_graph_data.py --task sol --k 1         # Different k-hop
    python prepare_graph_data.py --verify                 # Verify existing data

Features:
- Converts NPZ sequence data to PyTorch Geometric graph format
- Supports different k-hop connectivity patterns
- Validates data consistency
- Provides detailed statistics
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.graph_data_builder import build_all, get_graph_loaders, process_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_npz_data(task: str) -> Dict[str, Tuple[int, int]]:
    """
    Check if NPZ data exists and return statistics.

    Args:
        task: Task name ('hemo', 'sol', 'nf')

    Returns:
        Dictionary with split -> (num_samples, max_length) mapping
    """
    stats = {}
    data_dir = f"./data/{task}"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Task directory not found: {data_dir}")

    for split in ['train', 'val', 'test']:
        npz_path = f"{data_dir}/{split}.npz"
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                inputs = data['inputs']
                labels = data['labels']
                stats[split] = (len(inputs), inputs.shape[1])
                logger.info(f"{task}/{split}: {len(inputs)} samples, max_length={inputs.shape[1]}")
        else:
            logger.warning(f"NPZ file not found: {npz_path}")

    return stats


def check_graph_data(task: str) -> Dict[str, Dict]:
    """
    Check if graph data exists and return statistics.

    Args:
        task: Task name

    Returns:
        Dictionary with split statistics
    """
    stats = {}
    data_dir = f"./data/{task}"

    for split in ['train', 'val', 'test']:
        graph_path = f"{data_dir}/graphs_{split}.pt"
        if os.path.exists(graph_path):
            try:
                graphs = torch.load(graph_path, weights_only=False)
                num_graphs = len(graphs)

                # Analyze first few graphs for statistics
                if num_graphs > 0:
                    sample_graph = graphs[0]
                    node_features = sample_graph.x.shape if hasattr(sample_graph, 'x') else None
                    num_edges = sample_graph.edge_index.shape[1] if hasattr(sample_graph, 'edge_index') else 0

                    # Get average graph size
                    total_nodes = sum(g.num_nodes for g in graphs[:min(100, num_graphs)])
                    avg_nodes = total_nodes / min(100, num_graphs)

                    stats[split] = {
                        'num_graphs': num_graphs,
                        'node_features_shape': node_features,
                        'avg_nodes': avg_nodes,
                        'sample_edges': num_edges
                    }

                    logger.info(f"{task}/{split} graphs: {num_graphs} graphs, avg {avg_nodes:.1f} nodes")
                else:
                    stats[split] = {'num_graphs': 0}

            except Exception as e:
                logger.error(f"Error loading {graph_path}: {e}")
                stats[split] = {'error': str(e)}
        else:
            stats[split] = {'missing': True}

    return stats


def prepare_task_data(task: str, k_hop: int = 2, force: bool = False) -> bool:
    """
    Prepare graph data for a specific task.

    Args:
        task: Task name
        k_hop: Maximum hop distance for edges
        force: Force rebuild even if data exists

    Returns:
        True if successful
    """
    logger.info(f"Preparing graph data for task: {task} (k_hop={k_hop})")

    # Check if NPZ data exists
    try:
        npz_stats = check_npz_data(task)
        if not npz_stats:
            logger.error(f"No NPZ data found for task {task}")
            return False
    except FileNotFoundError as e:
        logger.error(f"Task data not found: {e}")
        return False

    # Check if graph data already exists
    if not force:
        graph_stats = check_graph_data(task)
        all_exist = all('num_graphs' in stats and stats['num_graphs'] > 0
                       for stats in graph_stats.values())
        if all_exist:
            logger.info(f"Graph data already exists for {task}, use --force to rebuild")
            return True

    # Build graph data
    try:
        logger.info(f"Building graph data for {task}...")
        build_all(task=task, k_hop=k_hop)

        # Verify the built data
        graph_stats = check_graph_data(task)
        success = all('num_graphs' in stats and stats['num_graphs'] > 0
                     for stats in graph_stats.values())

        if success:
            logger.info(f"✅ Successfully built graph data for {task}")
            return True
        else:
            logger.error(f"❌ Failed to build complete graph data for {task}")
            return False

    except Exception as e:
        logger.error(f"Error building graph data for {task}: {e}")
        return False


def verify_data_consistency(task: str) -> bool:
    """
    Verify that sequence and graph data are consistent.

    Args:
        task: Task name

    Returns:
        True if data is consistent
    """
    logger.info(f"Verifying data consistency for {task}...")

    try:
        npz_stats = check_npz_data(task)
        graph_stats = check_graph_data(task)

        for split in ['train', 'val', 'test']:
            if split not in npz_stats:
                logger.error(f"Missing NPZ data for {task}/{split}")
                return False

            if split not in graph_stats or 'num_graphs' not in graph_stats[split]:
                logger.error(f"Missing graph data for {task}/{split}")
                return False

            npz_count = npz_stats[split][0]
            graph_count = graph_stats[split]['num_graphs']

            if npz_count != graph_count:
                logger.error(f"Count mismatch in {task}/{split}: "
                           f"NPZ={npz_count}, graphs={graph_count}")
                return False

            logger.info(f"✅ {task}/{split}: {npz_count} samples consistent")

        logger.info(f"✅ All data consistent for {task}")
        return True

    except Exception as e:
        logger.error(f"Error verifying {task}: {e}")
        return False


def test_data_loading(task: str, batch_size: int = 4) -> bool:
    """
    Test data loading with PyTorch Geometric DataLoader.

    Args:
        task: Task name
        batch_size: Batch size for testing

    Returns:
        True if loading successful
    """
    logger.info(f"Testing data loading for {task}...")

    try:
        # Get graph loaders
        train_loader, val_loader, test_loader = get_graph_loaders(
            task=task,
            batch_size=batch_size,
            shuffle_train=False
        )

        # Test loading a few batches
        for loader_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            logger.info(f"Testing {loader_name} loader...")

            for i, batch in enumerate(loader):
                if i >= 2:  # Test only first 2 batches
                    break

                # Check batch properties
                logger.info(f"  Batch {i+1}: {batch.num_graphs} graphs, "
                          f"{batch.x.shape[0]} total nodes, "
                          f"{batch.edge_index.shape[1]} total edges")

                # Verify required fields
                required_fields = ['x', 'edge_index', 'edge_attr', 'y', 'batch']
                for field in required_fields:
                    if not hasattr(batch, field):
                        logger.error(f"Missing field {field} in batch")
                        return False

        logger.info(f"✅ Data loading test successful for {task}")
        return True

    except Exception as e:
        logger.error(f"Error testing data loading for {task}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare graph data for fusion model training")
    parser.add_argument(
        '--task',
        type=str,
        choices=['hemo', 'sol', 'nf'],
        help='Specific task to prepare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Prepare data for all tasks'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='Maximum hop distance for graph edges (default: 2)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if data exists'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing data consistency'
    )
    parser.add_argument(
        '--test-loading',
        action='store_true',
        help='Test data loading with PyTorch Geometric'
    )

    args = parser.parse_args()

    # Determine tasks to process
    if args.all:
        tasks = ['hemo', 'sol', 'nf']
    elif args.task:
        tasks = [args.task]
    else:
        # Default to all tasks if no specific task given
        tasks = ['hemo', 'sol', 'nf']

    logger.info(f"Processing tasks: {tasks}")
    logger.info(f"K-hop distance: {args.k}")

    # Process each task
    success_count = 0
    total_tasks = len(tasks)

    for task in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing task: {task.upper()}")
        logger.info(f"{'='*50}")

        try:
            # Verify only mode
            if args.verify:
                if verify_data_consistency(task):
                    success_count += 1
                continue

            # Test loading only mode
            if args.test_loading:
                if test_data_loading(task):
                    success_count += 1
                continue

            # Prepare data
            if prepare_task_data(task, args.k, args.force):
                # Verify the prepared data
                if verify_data_consistency(task):
                    # Test loading
                    if test_data_loading(task):
                        success_count += 1
                        logger.info(f"✅ Task {task} completed successfully")
                    else:
                        logger.error(f"❌ Task {task} failed data loading test")
                else:
                    logger.error(f"❌ Task {task} failed consistency check")
            else:
                logger.error(f"❌ Task {task} failed to prepare data")

        except Exception as e:
            logger.error(f"❌ Task {task} failed with error: {e}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {success_count}/{total_tasks} tasks completed successfully")
    logger.info(f"{'='*60}")

    if success_count == total_tasks:
        logger.info("🎉 All tasks completed successfully!")
        logger.info("You can now run fusion model training with:")
        logger.info("  python train_fusion.py --config config_fusion.yaml")
        return 0
    else:
        logger.error(f"❌ {total_tasks - success_count} tasks failed")
        return 1


if __name__ == "__main__":
    exit(main())
