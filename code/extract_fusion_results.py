#!/usr/bin/env python3
"""
Extract Fusion Results Script
============================

This script extracts results from existing fusion model checkpoints and saves
them to comprehensive text files. It can process multiple checkpoints and
generate summary reports automatically.

Usage:
    # Extract from specific checkpoint
    python extract_fusion_results.py --checkpoint checkpoints/fusion_nf_0820_1652/best_model.pt

    # Extract from all checkpoints in a directory
    python extract_fusion_results.py --checkpoint-dir checkpoints/

    # Extract and test a model
    python extract_fusion_results.py --checkpoint checkpoints/fusion_nf_0820_1652/best_model.pt --test

    # Generate summary from existing results
    python extract_fusion_results.py --generate-summary
"""

import os
import sys
import argparse
import torch
import yaml
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.fusion_dataloader import load_fusion_data
    from model.model_fusion_model import FusionModel
    from model.fusion_utils import test_fusion
    from model.fusion_results_saver import FusionResultsSaver
except ImportError as e:
    print(f"Warning: Could not import fusion modules: {e}")
    print("Some functionality may be limited.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_checkpoint_info(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract information from a fusion model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint information or None if failed
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {
            'checkpoint_path': checkpoint_path,
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'timestamp': datetime.fromtimestamp(
                os.path.getmtime(checkpoint_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        }

        # Extract model info if available
        if 'model_state_dict' in checkpoint:
            model_keys = list(checkpoint['model_state_dict'].keys())
            info['model_parameters'] = len(model_keys)
            info['has_model'] = True
        else:
            info['has_model'] = False

        return info

    except Exception as e:
        logger.error(f"Failed to extract info from {checkpoint_path}: {e}")
        return None


def test_checkpoint_model(checkpoint_path: str, config_path: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Test a fusion model checkpoint and return metrics.

    Args:
        checkpoint_path: Path to the checkpoint
        config_path: Path to config file (optional)

    Returns:
        Test metrics dictionary or None if failed
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config')

        if not config and config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        if not config:
            logger.error("No configuration found for testing")
            return None

        # Load test data
        _, _, test_loader = load_fusion_data(config)

        # Create and load model
        from model.train_fusion import create_fusion_model
        model = create_fusion_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        test_metrics = test_fusion(model, test_loader, device)

        logger.info(f"Testing completed for {checkpoint_path}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"Test F1: {test_metrics['f1']:.2f}%")

        return test_metrics

    except Exception as e:
        logger.error(f"Failed to test checkpoint {checkpoint_path}: {e}")
        return None


def save_extracted_results(checkpoint_info: Dict[str, Any],
                          test_metrics: Optional[Dict[str, float]] = None) -> str:
    """
    Save extracted results to a text file.

    Args:
        checkpoint_info: Information extracted from checkpoint
        test_metrics: Optional test metrics

    Returns:
        Path to saved results file
    """
    # Determine save directory and filename
    save_dir = Path("txt file")
    save_dir.mkdir(exist_ok=True)

    # Extract run name from checkpoint path
    checkpoint_path = checkpoint_info['checkpoint_path']
    if 'fusion_' in checkpoint_path:
        run_name = checkpoint_path.split('fusion_')[1].split('/')[0]
    else:
        run_name = f"extracted_{datetime.now().strftime('%m%d_%H%M')}"

    filename = f"fusion_extracted_{run_name}.txt"
    filepath = save_dir / filename

    with open(filepath, 'w') as f:
        f.write("# Fusion Model Extracted Results\n")
        f.write(f"# Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Source: {checkpoint_path}\n\n")

        # Checkpoint information
        f.write("## Checkpoint Information\n")
        f.write(f"- Checkpoint Path: {checkpoint_path}\n")
        f.write(f"- Creation Date: {checkpoint_info['timestamp']}\n")
        f.write(f"- Epoch: {checkpoint_info['epoch']}\n")
        f.write(f"- Has Model Parameters: {checkpoint_info['has_model']}\n")
        if checkpoint_info['has_model']:
            f.write(f"- Model Parameter Count: {checkpoint_info['model_parameters']}\n")
        f.write("\n")

        # Configuration
        config = checkpoint_info.get('config', {})
        if config:
            f.write("## Model Configuration\n")
            f.write(f"- Task: {config.get('task', 'N/A')}\n")
            f.write(f"- Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"- Learning Rate: {config.get('optim', {}).get('lr', 'N/A')}\n")
            f.write(f"- Total Epochs: {config.get('epochs', 'N/A')}\n")
            f.write(f"- Fusion Type: {config.get('fusion', {}).get('fusion_type', 'N/A')}\n")
            f.write(f"- Graph K-hop: {config.get('graph', {}).get('k_hop', 'N/A')}\n")
            f.write(f"- Sequence Hidden Size: {config.get('sequence', {}).get('hidden_size', 'N/A')}\n")
            f.write(f"- Graph Hidden Size: {config.get('graph_encoder', {}).get('hidden_dim', 'N/A')}\n\n")

        # Stored metrics from checkpoint
        stored_metrics = checkpoint_info.get('metrics', {})
        if stored_metrics:
            f.write("## Stored Validation Metrics\n")
            for key, value in stored_metrics.items():
                if isinstance(value, (int, float)):
                    if 'time' in key.lower():
                        f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}s\n")
                    elif 'samples' in key.lower():
                        f.write(f"- {key.replace('_', ' ').title()}: {int(value)}\n")
                    else:
                        f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
            f.write("\n")

        # Test metrics if available
        if test_metrics:
            f.write("## Fresh Test Results\n")
            f.write(f"- Test Accuracy: {test_metrics.get('accuracy', 0):.2f}%\n")
            f.write(f"- Test Precision: {test_metrics.get('precision', 0):.2f}%\n")
            f.write(f"- Test Recall: {test_metrics.get('recall', 0):.2f}%\n")
            f.write(f"- Test F1 Score: {test_metrics.get('f1', 0):.2f}%\n")
            f.write(f"- Test AUC: {test_metrics.get('auc', 0):.2f}%\n")
            if 'avg_inference_time' in test_metrics:
                f.write(f"- Average Inference Time: {test_metrics['avg_inference_time']:.4f}s\n")
            if 'total_samples' in test_metrics:
                f.write(f"- Total Test Samples: {int(test_metrics['total_samples'])}\n")
            f.write("\n")

        # Summary
        f.write("## Summary\n")
        best_metric = "N/A"
        best_value = 0

        if test_metrics:
            accuracy = test_metrics.get('accuracy', 0)
            f1 = test_metrics.get('f1', 0)
            if accuracy >= f1:
                best_metric = "Accuracy"
                best_value = accuracy
            else:
                best_metric = "F1 Score"
                best_value = f1
        elif stored_metrics:
            accuracy = stored_metrics.get('accuracy', 0)
            f1 = stored_metrics.get('f1', 0)
            if accuracy >= f1:
                best_metric = "Accuracy"
                best_value = accuracy
            else:
                best_metric = "F1 Score"
                best_value = f1

        if best_value > 0:
            f.write(f"- Best Performance: {best_value:.2f}% {best_metric}\n")
            f.write(f"- Task: {config.get('task', 'Unknown').upper()}\n")
            f.write(f"- Model Type: Fusion ({config.get('fusion', {}).get('fusion_type', 'unknown')})\n")

            performance_level = "Excellent" if best_value > 85 else "Good" if best_value > 80 else "Moderate"
            f.write(f"- Performance Level: {performance_level}\n")

    logger.info(f"Extracted results saved to: {filepath}")
    return str(filepath)


def find_fusion_checkpoints(base_dir: str) -> List[str]:
    """
    Find all fusion model checkpoints in a directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of checkpoint paths
    """
    checkpoint_paths = []
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.error(f"Directory not found: {base_dir}")
        return []

    # Search patterns
    patterns = [
        "**/*fusion*/**/*.pt",
        "**/*fusion*/*.pt",
        "**/best_model.pt",
        "**/best_*.pt",
        "**/*.pt"
    ]

    found_files = set()
    for pattern in patterns:
        for path in base_path.glob(pattern):
            if path.is_file() and 'fusion' in str(path).lower():
                found_files.add(str(path))

    checkpoint_paths = sorted(list(found_files))
    logger.info(f"Found {len(checkpoint_paths)} potential fusion checkpoints")

    return checkpoint_paths


def generate_fusion_summary(results_dir: str = "txt file") -> str:
    """
    Generate a comprehensive summary from all extracted results.

    Args:
        results_dir: Directory containing results files

    Returns:
        Path to summary file
    """
    results_path = Path(results_dir)
    summary_file = results_path / "fusion_comprehensive_summary.txt"

    # Find all fusion result files
    result_files = list(results_path.glob("fusion_*.txt"))
    result_files = [f for f in result_files if "summary" not in f.name.lower()]

    if not result_files:
        logger.warning("No fusion result files found for summary")
        return str(summary_file)

    # Parse results from files
    all_results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                content = f.read()

            # Extract key metrics (simple parsing)
            result_data = {
                'file': result_file.name,
                'accuracy': 0,
                'f1': 0,
                'precision': 0,
                'recall': 0,
                'auc': 0,
                'task': 'unknown',
                'fusion_type': 'unknown'
            }

            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if 'Test Accuracy:' in line:
                    result_data['accuracy'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'Test F1 Score:' in line or 'Test F1:' in line:
                    result_data['f1'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'Test Precision:' in line:
                    result_data['precision'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'Test Recall:' in line:
                    result_data['recall'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'Test AUC:' in line:
                    result_data['auc'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'Task:' in line:
                    result_data['task'] = line.split(':')[1].strip().lower()
                elif 'Fusion Type:' in line:
                    result_data['fusion_type'] = line.split(':')[1].strip().lower()

            all_results.append(result_data)

        except Exception as e:
            logger.warning(f"Could not parse {result_file}: {e}")

    # Generate summary
    with open(summary_file, 'w') as f:
        f.write("# Comprehensive Fusion Model Results Summary\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total Models Analyzed: {len(all_results)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("                           BEST MODELS BY CRITERIA\n")
        f.write("=" * 80 + "\n\n")

        if all_results:
            # Find best models
            best_accuracy = max(all_results, key=lambda x: x['accuracy'])
            best_f1 = max(all_results, key=lambda x: x['f1'])

            f.write("## 🏆 CHAMPION MODELS\n\n")
            f.write(f"### Best Test Accuracy: {best_accuracy['accuracy']:.2f}%\n")
            f.write(f"- Model: {best_accuracy['file']}\n")
            f.write(f"- Task: {best_accuracy['task'].upper()}\n")
            f.write(f"- Fusion Type: {best_accuracy['fusion_type'].title()}\n")
            f.write(f"- F1 Score: {best_accuracy['f1']:.2f}%\n\n")

            f.write(f"### Best F1 Score: {best_f1['f1']:.2f}%\n")
            f.write(f"- Model: {best_f1['file']}\n")
            f.write(f"- Task: {best_f1['task'].upper()}\n")
            f.write(f"- Fusion Type: {best_f1['fusion_type'].title()}\n")
            f.write(f"- Accuracy: {best_f1['accuracy']:.2f}%\n\n")

            # Results table
            f.write("## 📊 ALL MODELS COMPARISON\n\n")
            f.write("| Model | Task | Fusion | Accuracy | F1 | Precision | Recall | AUC |\n")
            f.write("|-------|------|--------|----------|----|-----------|---------|\n")

            # Sort by accuracy
            sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
            for result in sorted_results:
                model_name = result['file'][:20] + "..." if len(result['file']) > 23 else result['file']
                f.write(f"| {model_name:20} | "
                       f"{result['task'][:4].upper():4} | "
                       f"{result['fusion_type'][:6]:6} | "
                       f"{result['accuracy']:8.2f}% | "
                       f"{result['f1']:6.2f}% | "
                       f"{result['precision']:9.2f}% | "
                       f"{result['recall']:7.2f}% | "
                       f"{result['auc']:7.2f}% |\n")

            # Statistics
            f.write("\n## 📈 PERFORMANCE STATISTICS\n\n")
            accuracies = [r['accuracy'] for r in all_results]
            f1s = [r['f1'] for r in all_results]

            f.write(f"### Test Accuracy Statistics\n")
            f.write(f"- Mean: {np.mean(accuracies):.2f}%\n")
            f.write(f"- Std Dev: {np.std(accuracies):.2f}%\n")
            f.write(f"- Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]\n\n")

            f.write(f"### F1 Score Statistics\n")
            f.write(f"- Mean: {np.mean(f1s):.2f}%\n")
            f.write(f"- Std Dev: {np.std(f1s):.2f}%\n")
            f.write(f"- Range: [{np.min(f1s):.2f}%, {np.max(f1s):.2f}%]\n\n")

            # Task analysis
            tasks = {}
            fusion_types = {}
            for result in all_results:
                task = result['task']
                fusion_type = result['fusion_type']

                if task not in tasks:
                    tasks[task] = []
                tasks[task].append(result['accuracy'])

                if fusion_type not in fusion_types:
                    fusion_types[fusion_type] = []
                fusion_types[fusion_type].append(result['accuracy'])

            f.write(f"### Performance by Task\n")
            for task, accuracies in tasks.items():
                f.write(f"- {task.upper()}: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}% ({len(accuracies)} models)\n")

            f.write(f"\n### Performance by Fusion Type\n")
            for fusion_type, accuracies in fusion_types.items():
                f.write(f"- {fusion_type.title()}: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}% ({len(accuracies)} models)\n")

        else:
            f.write("No valid results found to summarize.\n")

    logger.info(f"Generated comprehensive summary: {summary_file}")
    return str(summary_file)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract Fusion Model Results")
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='Directory to search for checkpoints')
    parser.add_argument('--test', action='store_true', help='Run fresh testing on the model')
    parser.add_argument('--config', type=str, help='Config file for testing (optional)')
    parser.add_argument('--generate-summary', action='store_true', help='Generate summary from existing results')
    parser.add_argument('--save-dir', type=str, default='txt file', help='Directory to save results')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    if args.generate_summary:
        logger.info("Generating comprehensive summary...")
        summary_file = generate_fusion_summary(args.save_dir)
        print(f"\n✅ Summary generated: {summary_file}")
        return

    checkpoints_to_process = []

    if args.checkpoint:
        # Process specific checkpoint
        if os.path.exists(args.checkpoint):
            checkpoints_to_process = [args.checkpoint]
        else:
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            return
    else:
        # Find all checkpoints in directory
        checkpoints_to_process = find_fusion_checkpoints(args.checkpoint_dir)

    if not checkpoints_to_process:
        logger.error("No checkpoints found to process")
        return

    print(f"\n🔍 Processing {len(checkpoints_to_process)} checkpoint(s)...")

    processed_files = []
    for i, checkpoint_path in enumerate(checkpoints_to_process, 1):
        print(f"\n[{i}/{len(checkpoints_to_process)}] Processing: {checkpoint_path}")

        # Extract checkpoint info
        checkpoint_info = extract_checkpoint_info(checkpoint_path)
        if not checkpoint_info:
            print(f"❌ Failed to extract info from {checkpoint_path}")
            continue

        # Test model if requested
        test_metrics = None
        if args.test:
            print("🧪 Running fresh model testing...")
            test_metrics = test_checkpoint_model(checkpoint_path, args.config)
            if test_metrics:
                print(f"✅ Test completed: {test_metrics['accuracy']:.2f}% accuracy, {test_metrics['f1']:.2f}% F1")
            else:
                print("❌ Testing failed")

        # Save results
        result_file = save_extracted_results(checkpoint_info, test_metrics)
        processed_files.append(result_file)
        print(f"📁 Results saved: {result_file}")

    # Generate summary if multiple files processed
    if len(processed_files) > 1:
        print("\n📊 Generating comprehensive summary...")
        summary_file = generate_fusion_summary(args.save_dir)
        processed_files.append(summary_file)
        print(f"📋 Summary generated: {summary_file}")

    print(f"\n🎉 Processing complete!")
    print(f"📁 Files generated: {len(processed_files)}")
    for file in processed_files:
        print(f"   - {file}")

    print(f"\n💡 Tip: Use --generate-summary to create a comprehensive report from all results.")


if __name__ == "__main__":
    main()
