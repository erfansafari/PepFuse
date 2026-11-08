#!/usr/bin/env python3
"""
Fusion Model Results Saver
==========================

This module provides comprehensive functionality to save, track, and manage
the best results from fusion model training runs. It automatically maintains
records of the best performing models across different criteria.

Features:
- Save individual run results
- Track best models by accuracy, F1 score, and validation accuracy
- Generate comprehensive summaries
- Update global best metrics across all runs
- Export results in various formats

Usage:
    from model.fusion_results_saver import FusionResultsSaver

    saver = FusionResultsSaver()
    saver.save_run_results(metrics, config, model_path)
    saver.update_global_best()
    saver.generate_summary_report()
"""

import os
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FusionResultsSaver:
    """
    Comprehensive results saver for fusion model training.

    This class manages saving individual run results, tracking best models,
    and generating comprehensive summary reports.
    """

    def __init__(self, base_dir: str = "txt file", results_file: str = "fusion_results.json"):
        """
        Initialize the results saver.

        Args:
            base_dir: Base directory to save results
            results_file: JSON file to store all results data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.results_file = self.base_dir / results_file
        self.results_data = self._load_existing_results()

        # Initialize tracking structures
        self.best_by_accuracy = {}
        self.best_by_f1 = {}
        self.best_by_val_accuracy = {}
        self.all_runs = []

        self._load_best_metrics()

    def _load_existing_results(self) -> Dict:
        """Load existing results from JSON file."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not load existing results, starting fresh")

        return {
            "runs": [],
            "best_models": {
                "accuracy": {},
                "f1": {},
                "val_accuracy": {}
            },
            "summary_stats": {},
            "last_updated": None
        }

    def _save_results_data(self):
        """Save results data to JSON file."""
        self.results_data["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results_data, f, indent=2)

    def _load_best_metrics(self):
        """Load best metrics from existing data."""
        best_models = self.results_data.get("best_models", {})
        self.best_by_accuracy = best_models.get("accuracy", {})
        self.best_by_f1 = best_models.get("f1", {})
        self.best_by_val_accuracy = best_models.get("val_accuracy", {})
        self.all_runs = self.results_data.get("runs", [])

    def save_run_results(self,
                        train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float],
                        test_metrics: Dict[str, float],
                        config: Dict[str, Any],
                        model_path: str,
                        run_name: Optional[str] = None) -> str:
        """
        Save results for a single training run.

        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
            config: Model configuration
            model_path: Path to saved model
            run_name: Optional run name

        Returns:
            Path to the saved results file
        """
        if run_name is None:
            run_name = datetime.now().strftime("fusion_%m%d_%H%M")

        # Create run data
        run_data = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "config": config,
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            }
        }

        # Add to runs list
        self.all_runs.append(run_data)
        self.results_data["runs"] = self.all_runs

        # Save individual run file
        run_file_path = self._save_individual_run(run_data)

        # Update best models
        self._update_best_models(run_data)

        # Save updated results data
        self._save_results_data()

        logger.info(f"Saved run results: {run_file_path}")
        return str(run_file_path)

    def _save_individual_run(self, run_data: Dict) -> Path:
        """Save individual run results to a text file."""
        run_name = run_data["run_name"]
        run_file = self.base_dir / f"{run_name}_results.txt"

        config = run_data["config"]
        metrics = run_data["metrics"]

        with open(run_file, 'w') as f:
            f.write(f"# Fusion Model Results - {run_name}\n")
            f.write(f"# Generated: {run_data['timestamp']}\n")
            f.write(f"# Model Path: {run_data['model_path']}\n\n")

            # Configuration
            f.write("## Model Configuration\n")
            f.write(f"- Task: {config.get('task', 'unknown')}\n")
            f.write(f"- Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"- Learning Rate: {config.get('optim', {}).get('lr', 'N/A')}\n")
            f.write(f"- Epochs: {config.get('epochs', 'N/A')}\n")
            f.write(f"- Fusion Type: {config.get('fusion', {}).get('fusion_type', 'N/A')}\n")
            f.write(f"- Graph K-hop: {config.get('graph', {}).get('k_hop', 'N/A')}\n")
            f.write(f"- Sequence Hidden Size: {config.get('sequence', {}).get('hidden_size', 'N/A')}\n")
            f.write(f"- Graph Hidden Size: {config.get('graph_encoder', {}).get('hidden_dim', 'N/A')}\n\n")

            # Training Metrics
            f.write("## Training Metrics\n")
            for key, value in metrics["train"].items():
                if 'time' in key.lower():
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}s\n")
                elif 'samples' in key.lower() or key == 'total_samples':
                    f.write(f"- {key.replace('_', ' ').title()}: {int(value)}\n")
                else:
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
            f.write("\n")

            # Validation Metrics
            f.write("## Validation Metrics\n")
            for key, value in metrics["validation"].items():
                if 'time' in key.lower():
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}s\n")
                elif 'samples' in key.lower() or key == 'total_samples':
                    f.write(f"- {key.replace('_', ' ').title()}: {int(value)}\n")
                else:
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
            f.write("\n")

            # Test Metrics
            f.write("## Test Metrics\n")
            for key, value in metrics["test"].items():
                if 'time' in key.lower():
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}s\n")
                elif 'samples' in key.lower() or key == 'total_samples':
                    f.write(f"- {key.replace('_', ' ').title()}: {int(value)}\n")
                else:
                    f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
            f.write("\n")

            # Quick Summary
            f.write("## Quick Summary\n")
            f.write(f"- **Test Accuracy**: {metrics['test'].get('accuracy', 0):.2f}%\n")
            f.write(f"- **Test F1 Score**: {metrics['test'].get('f1', 0):.2f}%\n")
            f.write(f"- **Test Precision**: {metrics['test'].get('precision', 0):.2f}%\n")
            f.write(f"- **Test Recall**: {metrics['test'].get('recall', 0):.2f}%\n")
            f.write(f"- **Test AUC**: {metrics['test'].get('auc', 0):.2f}%\n")
            f.write(f"- **Validation Accuracy**: {metrics['validation'].get('accuracy', 0):.2f}%\n")

            if 'avg_inference_time' in metrics['test']:
                f.write(f"- **Average Inference Time**: {metrics['test']['avg_inference_time']:.4f}s\n")

        return run_file

    def _update_best_models(self, run_data: Dict):
        """Update best models tracking."""
        test_metrics = run_data["metrics"]["test"]
        val_metrics = run_data["metrics"]["validation"]

        # Get current scores
        test_accuracy = test_metrics.get("accuracy", 0)
        test_f1 = test_metrics.get("f1", 0)
        val_accuracy = val_metrics.get("accuracy", 0)

        # Update best by test accuracy
        if not self.best_by_accuracy or test_accuracy > self.best_by_accuracy.get("score", 0):
            self.best_by_accuracy = {
                "score": test_accuracy,
                "run_data": run_data,
                "metric_type": "test_accuracy"
            }

        # Update best by F1 score
        if not self.best_by_f1 or test_f1 > self.best_by_f1.get("score", 0):
            self.best_by_f1 = {
                "score": test_f1,
                "run_data": run_data,
                "metric_type": "test_f1"
            }

        # Update best by validation accuracy
        if not self.best_by_val_accuracy or val_accuracy > self.best_by_val_accuracy.get("score", 0):
            self.best_by_val_accuracy = {
                "score": val_accuracy,
                "run_data": run_data,
                "metric_type": "val_accuracy"
            }

        # Update results data
        self.results_data["best_models"] = {
            "accuracy": self.best_by_accuracy,
            "f1": self.best_by_f1,
            "val_accuracy": self.best_by_val_accuracy
        }

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        summary_file = self.base_dir / "fusion_comprehensive_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("# Comprehensive Fusion Model Results Summary\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Runs: {len(self.all_runs)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("                           BEST MODELS BY CRITERIA\n")
            f.write("=" * 80 + "\n\n")

            # Best by Test Accuracy
            if self.best_by_accuracy:
                self._write_best_model_section(f, "Test Accuracy", self.best_by_accuracy)

            # Best by F1 Score
            if self.best_by_f1:
                self._write_best_model_section(f, "F1 Score", self.best_by_f1)

            # Best by Validation Accuracy
            if self.best_by_val_accuracy:
                self._write_best_model_section(f, "Validation Accuracy", self.best_by_val_accuracy)

            # All Runs Summary Table
            f.write("\n" + "=" * 80 + "\n")
            f.write("                           ALL RUNS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            self._write_all_runs_table(f)

            # Statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("                              STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            self._write_statistics(f)

            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("                            RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            self._write_recommendations(f)

        logger.info(f"Generated summary report: {summary_file}")
        return str(summary_file)

    def _write_best_model_section(self, f, criteria_name: str, best_model_data: Dict):
        """Write best model section to file."""
        run_data = best_model_data["run_data"]
        score = best_model_data["score"]
        metrics = run_data["metrics"]
        config = run_data["config"]

        f.write(f"## 🏆 BEST MODEL BY {criteria_name.upper()}\n")
        f.write(f"**Winner: {run_data['run_name']}**\n")
        f.write(f"**Score: {score:.2f}%**\n\n")

        f.write("### Complete Performance Profile\n")
        f.write(f"- Test Accuracy: {metrics['test'].get('accuracy', 0):.2f}%\n")
        f.write(f"- Test Precision: {metrics['test'].get('precision', 0):.2f}%\n")
        f.write(f"- Test Recall: {metrics['test'].get('recall', 0):.2f}%\n")
        f.write(f"- Test F1 Score: {metrics['test'].get('f1', 0):.2f}%\n")
        f.write(f"- Test AUC: {metrics['test'].get('auc', 0):.2f}%\n")
        f.write(f"- Validation Accuracy: {metrics['validation'].get('accuracy', 0):.2f}%\n")

        if 'avg_inference_time' in metrics['test']:
            f.write(f"- Average Inference Time: {metrics['test']['avg_inference_time']:.4f}s\n")
        f.write(f"- Total Test Samples: {int(metrics['test'].get('total_samples', 0))}\n\n")

        f.write("### Model Details\n")
        f.write(f"- Training Date: {run_data['timestamp'][:19]}\n")
        f.write(f"- Model Path: {run_data['model_path']}\n")
        f.write(f"- Run Name: {run_data['run_name']}\n\n")

        f.write("### Configuration\n")
        f.write(f"- Task: {config.get('task', 'N/A')}\n")
        f.write(f"- Batch Size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"- Learning Rate: {config.get('optim', {}).get('lr', 'N/A')}\n")
        f.write(f"- Epochs: {config.get('epochs', 'N/A')}\n")
        f.write(f"- Fusion Type: {config.get('fusion', {}).get('fusion_type', 'N/A')}\n")
        f.write(f"- Graph K-hop: {config.get('graph', {}).get('k_hop', 'N/A')}\n\n")

    def _write_all_runs_table(self, f):
        """Write table of all runs."""
        if not self.all_runs:
            f.write("No runs recorded yet.\n")
            return

        f.write("| Run Name | Date | Test Acc | F1 Score | Val Acc | Precision | Recall | AUC |\n")
        f.write("|----------|------|----------|----------|---------|-----------|--------|---------|\n")

        for run in sorted(self.all_runs, key=lambda x: x['metrics']['test'].get('accuracy', 0), reverse=True):
            test_metrics = run['metrics']['test']
            val_metrics = run['metrics']['validation']
            date = run['timestamp'][:10]

            f.write(f"| {run['run_name'][:15]} | {date} | "
                   f"{test_metrics.get('accuracy', 0):.1f}% | "
                   f"{test_metrics.get('f1', 0):.1f}% | "
                   f"{val_metrics.get('accuracy', 0):.1f}% | "
                   f"{test_metrics.get('precision', 0):.1f}% | "
                   f"{test_metrics.get('recall', 0):.1f}% | "
                   f"{test_metrics.get('auc', 0):.1f}% |\n")

    def _write_statistics(self, f):
        """Write performance statistics."""
        if not self.all_runs:
            f.write("No runs available for statistics.\n")
            return

        # Collect metrics
        test_accuracies = [run['metrics']['test'].get('accuracy', 0) for run in self.all_runs]
        test_f1s = [run['metrics']['test'].get('f1', 0) for run in self.all_runs]
        val_accuracies = [run['metrics']['validation'].get('accuracy', 0) for run in self.all_runs]

        f.write("### Performance Statistics\n")
        f.write(f"- **Test Accuracy**: Mean={np.mean(test_accuracies):.2f}%, "
               f"Std={np.std(test_accuracies):.2f}%, "
               f"Range=[{np.min(test_accuracies):.2f}%, {np.max(test_accuracies):.2f}%]\n")

        f.write(f"- **Test F1 Score**: Mean={np.mean(test_f1s):.2f}%, "
               f"Std={np.std(test_f1s):.2f}%, "
               f"Range=[{np.min(test_f1s):.2f}%, {np.max(test_f1s):.2f}%]\n")

        f.write(f"- **Validation Accuracy**: Mean={np.mean(val_accuracies):.2f}%, "
               f"Std={np.std(val_accuracies):.2f}%, "
               f"Range=[{np.min(val_accuracies):.2f}%, {np.max(val_accuracies):.2f}%]\n\n")

        # Task distribution
        tasks = [run['config'].get('task', 'unknown') for run in self.all_runs]
        task_counts = {}
        for task in tasks:
            task_counts[task] = task_counts.get(task, 0) + 1

        f.write("### Task Distribution\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"- {task.upper()}: {count} runs\n")
        f.write("\n")

        # Fusion type distribution
        fusion_types = [run['config'].get('fusion', {}).get('fusion_type', 'unknown') for run in self.all_runs]
        fusion_counts = {}
        for fusion_type in fusion_types:
            fusion_counts[fusion_type] = fusion_counts.get(fusion_type, 0) + 1

        f.write("### Fusion Type Distribution\n")
        for fusion_type, count in sorted(fusion_counts.items()):
            f.write(f"- {fusion_type.title()}: {count} runs\n")

    def _write_recommendations(self, f):
        """Write model recommendations."""
        if not self.all_runs:
            f.write("No runs available for recommendations.\n")
            return

        f.write("### 🎯 PRODUCTION MODEL SELECTION\n\n")

        if self.best_by_accuracy:
            best_acc = self.best_by_accuracy["run_data"]
            f.write("#### For Highest Test Accuracy\n")
            f.write(f"- **Model**: {best_acc['run_name']}\n")
            f.write(f"- **Path**: {best_acc['model_path']}\n")
            f.write(f"- **Performance**: {self.best_by_accuracy['score']:.2f}% accuracy, "
                   f"{best_acc['metrics']['test'].get('f1', 0):.2f}% F1\n")
            f.write(f"- **Use Case**: Maximum prediction accuracy\n\n")

        if self.best_by_f1:
            best_f1 = self.best_by_f1["run_data"]
            f.write("#### For Best F1 Score\n")
            f.write(f"- **Model**: {best_f1['run_name']}\n")
            f.write(f"- **Path**: {best_f1['model_path']}\n")
            f.write(f"- **Performance**: {best_f1['metrics']['test'].get('accuracy', 0):.2f}% accuracy, "
                   f"{self.best_by_f1['score']:.2f}% F1\n")
            f.write(f"- **Use Case**: Balanced precision-recall performance\n\n")

        if self.best_by_val_accuracy:
            best_val = self.best_by_val_accuracy["run_data"]
            f.write("#### For Best Generalization\n")
            f.write(f"- **Model**: {best_val['run_name']}\n")
            f.write(f"- **Path**: {best_val['model_path']}\n")
            f.write(f"- **Performance**: {self.best_by_val_accuracy['score']:.2f}% validation accuracy\n")
            f.write(f"- **Use Case**: Best generalization to unseen data\n\n")

    def save_best_models_summary(self) -> str:
        """Save a focused summary of just the best models."""
        best_file = self.base_dir / "fusion_best_models_summary.txt"

        with open(best_file, 'w') as f:
            f.write("# Fusion Model Best Results Summary\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 🏆 CHAMPION MODELS\n\n")

            if self.best_by_accuracy:
                run_data = self.best_by_accuracy["run_data"]
                f.write(f"### Best Test Accuracy: {self.best_by_accuracy['score']:.2f}%\n")
                f.write(f"- **Model**: {run_data['run_name']}\n")
                f.write(f"- **Path**: {run_data['model_path']}\n")
                f.write(f"- **F1 Score**: {run_data['metrics']['test'].get('f1', 0):.2f}%\n")
                f.write(f"- **Task**: {run_data['config'].get('task', 'N/A').upper()}\n")
                f.write(f"- **Fusion Type**: {run_data['config'].get('fusion', {}).get('fusion_type', 'N/A')}\n\n")

            if self.best_by_f1:
                run_data = self.best_by_f1["run_data"]
                f.write(f"### Best F1 Score: {self.best_by_f1['score']:.2f}%\n")
                f.write(f"- **Model**: {run_data['run_name']}\n")
                f.write(f"- **Path**: {run_data['model_path']}\n")
                f.write(f"- **Accuracy**: {run_data['metrics']['test'].get('accuracy', 0):.2f}%\n")
                f.write(f"- **Task**: {run_data['config'].get('task', 'N/A').upper()}\n")
                f.write(f"- **Fusion Type**: {run_data['config'].get('fusion', {}).get('fusion_type', 'N/A')}\n\n")

            if self.best_by_val_accuracy:
                run_data = self.best_by_val_accuracy["run_data"]
                f.write(f"### Best Validation Accuracy: {self.best_by_val_accuracy['score']:.2f}%\n")
                f.write(f"- **Model**: {run_data['run_name']}\n")
                f.write(f"- **Path**: {run_data['model_path']}\n")
                f.write(f"- **Test Accuracy**: {run_data['metrics']['test'].get('accuracy', 0):.2f}%\n")
                f.write(f"- **Task**: {run_data['config'].get('task', 'N/A').upper()}\n")
                f.write(f"- **Fusion Type**: {run_data['config'].get('fusion', {}).get('fusion_type', 'N/A')}\n\n")

            # Quick comparison table
            f.write("## 📊 Quick Comparison\n\n")
            f.write("| Criteria | Model | Score | Task | Fusion Type |\n")
            f.write("|----------|--------|--------|------|-------------|\n")

            if self.best_by_accuracy:
                acc_run = self.best_by_accuracy["run_data"]
                f.write(f"| Test Accuracy | {acc_run['run_name'][:15]} | {self.best_by_accuracy['score']:.2f}% | "
                       f"{acc_run['config'].get('task', 'N/A').upper()} | "
                       f"{acc_run['config'].get('fusion', {}).get('fusion_type', 'N/A')} |\n")

            if self.best_by_f1:
                f1_run = self.best_by_f1["run_data"]
                f.write(f"| F1 Score | {f1_run['run_name'][:15]} | {self.best_by_f1['score']:.2f}% | "
                       f"{f1_run['config'].get('task', 'N/A').upper()} | "
                       f"{f1_run['config'].get('fusion', {}).get('fusion_type', 'N/A')} |\n")

            if self.best_by_val_accuracy:
                val_run = self.best_by_val_accuracy["run_data"]
                f.write(f"| Val Accuracy | {val_run['run_name'][:15]} | {self.best_by_val_accuracy['score']:.2f}% | "
                       f"{val_run['config'].get('task', 'N/A').upper()} | "
                       f"{val_run['config'].get('fusion', {}).get('fusion_type', 'N/A')} |\n")

        logger.info(f"Generated best models summary: {best_file}")
        return str(best_file)

    def export_results_csv(self) -> str:
        """Export results to CSV format."""
        csv_file = self.base_dir / "fusion_results.csv"

        if not self.all_runs:
            logger.warning("No runs available for CSV export")
            return str(csv_file)

        with open(csv_file, 'w') as f:
            # Header
            f.write("run_name,date,task,fusion_type,batch_size,learning_rate,epochs,"
                   "test_accuracy,test_f1,test_precision,test_recall,test_auc,"
                   "val_accuracy,val_f1,model_path\n")

            # Data rows
            for run in self.all_runs:
                config = run['config']
                test_metrics = run['metrics']['test']
                val_metrics = run['metrics']['validation']

                f.write(f"{run['run_name']},{run['timestamp'][:10]},"
                       f"{config.get('task', '')},"
                       f"{config.get('fusion', {}).get('fusion_type', '')},"
                       f"{config.get('batch_size', '')},"
                       f"{config.get('optim', {}).get('lr', '')},"
                       f"{config.get('epochs', '')},"
                       f"{test_metrics.get('accuracy', 0):.2f},"
                       f"{test_metrics.get('f1', 0):.2f},"
                       f"{test_metrics.get('precision', 0):.2f},"
                       f"{test_metrics.get('recall', 0):.2f},"
                       f"{test_metrics.get('auc', 0):.2f},"
                       f"{val_metrics.get('accuracy', 0):.2f},"
                       f"{val_metrics.get('f1', 0):.2f},"
                       f"{run['model_path']}\n")

        logger.info(f"Exported results to CSV: {csv_file}")
        return str(csv_file)

    def get_best_model_path(self, criteria: str = "accuracy") -> Optional[str]:
        """
        Get the path to the best model by specified criteria.

        Args:
            criteria: One of 'accuracy', 'f1', or 'val_accuracy'

        Returns:
            Path to the best model, or None if no models found
        """
        criteria_map = {
            "accuracy": self.best_by_accuracy,
            "f1": self.best_by_f1,
            "val_accuracy": self.best_by_val_accuracy
        }

        if criteria not in criteria_map:
            logger.error(f"Unknown criteria: {criteria}. Must be one of: {list(criteria_map.keys())}")
            return None

        best_model = criteria_map[criteria]
        if not best_model:
            logger.warning(f"No best model found for criteria: {criteria}")
            return None

        return best_model["run_data"]["model_path"]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all runs."""
        if not self.all_runs:
            return {}

        # Collect all metrics
        test_accuracies = [run['metrics']['test'].get('accuracy', 0) for run in self.all_runs]
        test_f1s = [run['metrics']['test'].get('f1', 0) for run in self.all_runs]
        val_accuracies = [run['metrics']['validation'].get('accuracy', 0) for run in self.all_runs]

        return {
            "total_runs": len(self.all_runs),
            "test_accuracy": {
                "mean": np.mean(test_accuracies),
                "std": np.std(test_accuracies),
                "min": np.min(test_accuracies),
                "max": np.max(test_accuracies)
            },
            "test_f1": {
                "mean": np.mean(test_f1s),
                "std": np.std(test_f1s),
                "min": np.min(test_f1s),
                "max": np.max(test_f1s)
            },
            "validation_accuracy": {
                "mean": np.mean(val_accuracies),
                "std": np.std(val_accuracies),
                "min": np.min(val_accuracies),
                "max": np.max(val_accuracies)
            }
        }


# Example usage functions
def save_fusion_training_results(train_metrics: Dict[str, float],
                                val_metrics: Dict[str, float],
                                test_metrics: Dict[str, float],
                                config: Dict[str, Any],
                                model_path: str,
                                run_name: Optional[str] = None) -> str:
    """
    Convenience function to save fusion training results.

    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        test_metrics: Test metrics dictionary
        config: Training configuration dictionary
        model_path: Path where the model was saved
        run_name: Optional name for this run

    Returns:
        Path to the saved results file
    """
    saver = FusionResultsSaver()
    result_file = saver.save_run_results(
        train_metrics, val_metrics, test_metrics,
        config, model_path, run_name
    )

    # Generate comprehensive reports
    saver.generate_summary_report()
    saver.save_best_models_summary()
    saver.export_results_csv()

    return result_file


def get_best_fusion_model(criteria: str = "accuracy") -> Optional[str]:
    """
    Get the path to the best fusion model by specified criteria.

    Args:
        criteria: One of 'accuracy', 'f1', or 'val_accuracy'

    Returns:
        Path to the best model, or None if no models found
    """
    saver = FusionResultsSaver()
    return saver.get_best_model_path(criteria)


def print_fusion_summary():
    """Print a summary of all fusion model results."""
    saver = FusionResultsSaver()
    stats = saver.get_summary_stats()

    if not stats:
        print("No fusion model results found.")
        return

    print("=" * 60)
    print("FUSION MODEL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Runs: {stats['total_runs']}")
    print(f"\nTest Accuracy: {stats['test_accuracy']['mean']:.2f}% ± {stats['test_accuracy']['std']:.2f}%")
    print(f"  Range: [{stats['test_accuracy']['min']:.2f}%, {stats['test_accuracy']['max']:.2f}%]")
    print(f"\nTest F1 Score: {stats['test_f1']['mean']:.2f}% ± {stats['test_f1']['std']:.2f}%")
    print(f"  Range: [{stats['test_f1']['min']:.2f}%, {stats['test_f1']['max']:.2f}%]")
    print(f"\nValidation Accuracy: {stats['validation_accuracy']['mean']:.2f}% ± {stats['validation_accuracy']['std']:.2f}%")
    print(f"  Range: [{stats['validation_accuracy']['min']:.2f}%, {stats['validation_accuracy']['max']:.2f}%]")
    print("=" * 60)
