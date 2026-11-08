#!/usr/bin/env python3
"""
Fusion Model Results Integration Script
======================================

This script integrates with existing fusion model training to automatically
save the best results to text files. It can be imported and used with minimal
changes to existing training code.

Features:
- Simple integration with existing training loops
- Automatic tracking of best models by accuracy, F1, and validation accuracy
- Comprehensive text file generation with metrics and model paths
- Easy-to-use functions that work with current fusion training setup

Usage:
    from model.save_fusion_results import FusionResultsManager

    # In your training script
    results_manager = FusionResultsManager()
    results_manager.save_epoch_results(epoch, train_metrics, val_metrics, test_metrics, config, model_path)
    results_manager.finalize_and_save_best()
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


class FusionResultsManager:
    """
    Simple results manager for fusion model training.

    This class tracks metrics across epochs and automatically saves
    the best results to comprehensive text files.
    """

    def __init__(self, save_dir: str = "txt file"):
        """
        Initialize the results manager.

        Args:
            save_dir: Directory to save results files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize tracking
        self.best_accuracy = {"score": 0, "data": None}
        self.best_f1 = {"score": 0, "data": None}
        self.best_val_accuracy = {"score": 0, "data": None}

        self.all_epochs = []
        self.current_run_name = None
        self.current_config = None

    def start_run(self, run_name: str, config: Dict[str, Any]):
        """
        Start tracking a new training run.

        Args:
            run_name: Name for this training run
            config: Configuration dictionary
        """
        self.current_run_name = run_name
        self.current_config = config
        self.all_epochs = []

        logger.info(f"Started tracking fusion run: {run_name}")

    def save_epoch_results(self,
                          epoch: int,
                          train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float],
                          test_metrics: Optional[Dict[str, float]] = None,
                          model_path: Optional[str] = None):
        """
        Save results for a single epoch and update best models.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            test_metrics: Test metrics dictionary (optional)
            model_path: Path to saved model for this epoch (optional)
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": model_path
        }

        self.all_epochs.append(epoch_data)

        # Update best models based on validation metrics
        val_accuracy = val_metrics.get("accuracy", 0)
        val_f1 = val_metrics.get("f1", 0)
        test_accuracy = test_metrics.get("accuracy", 0) if test_metrics else 0

        # Best validation accuracy
        if val_accuracy > self.best_val_accuracy["score"]:
            self.best_val_accuracy["score"] = val_accuracy
            self.best_val_accuracy["data"] = epoch_data.copy()
            logger.info(f"New best validation accuracy: {val_accuracy:.2f}% at epoch {epoch}")

        # Best F1 score
        if val_f1 > self.best_f1["score"]:
            self.best_f1["score"] = val_f1
            self.best_f1["data"] = epoch_data.copy()
            logger.info(f"New best F1 score: {val_f1:.2f}% at epoch {epoch}")

        # Best test accuracy (if test metrics provided)
        if test_metrics and test_accuracy > self.best_accuracy["score"]:
            self.best_accuracy["score"] = test_accuracy
            self.best_accuracy["data"] = epoch_data.copy()
            logger.info(f"New best test accuracy: {test_accuracy:.2f}% at epoch {epoch}")

    def finalize_and_save_best(self, final_test_metrics: Optional[Dict[str, float]] = None,
                              final_model_path: Optional[str] = None) -> List[str]:
        """
        Finalize training and save comprehensive results.

        Args:
            final_test_metrics: Final test metrics on best model
            final_model_path: Path to final best model

        Returns:
            List of generated file paths
        """
        if not self.current_run_name or not self.current_config:
            logger.warning("No run in progress, cannot save results")
            return []

        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual run results
        run_file = self._save_run_results(final_test_metrics, final_model_path, timestamp)
        generated_files.append(run_file)

        # Save best models summary
        best_file = self._save_best_models_summary(timestamp)
        generated_files.append(best_file)

        # Update global fusion results
        global_file = self._update_global_results(final_test_metrics, final_model_path)
        generated_files.append(global_file)

        # Generate comprehensive summary
        summary_file = self._generate_comprehensive_summary()
        generated_files.append(summary_file)

        logger.info(f"Saved fusion results to {len(generated_files)} files")
        return generated_files

    def _save_run_results(self, final_test_metrics: Optional[Dict],
                         final_model_path: Optional[str], timestamp: str) -> str:
        """Save results for the current run."""
        filename = f"fusion_{self.current_run_name}_{timestamp}.txt"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            f.write(f"# Fusion Model Training Results - {self.current_run_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Epochs: {len(self.all_epochs)}\n\n")

            # Configuration
            f.write("## Model Configuration\n")
            config = self.current_config
            f.write(f"- Task: {config.get('task', 'N/A')}\n")
            f.write(f"- Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"- Learning Rate: {config.get('optim', {}).get('lr', 'N/A')}\n")
            f.write(f"- Epochs: {config.get('epochs', 'N/A')}\n")
            f.write(f"- Fusion Type: {config.get('fusion', {}).get('fusion_type', 'N/A')}\n")
            f.write(f"- Graph K-hop: {config.get('graph', {}).get('k_hop', 'N/A')}\n")
            f.write(f"- Sequence Hidden Size: {config.get('sequence', {}).get('hidden_size', 'N/A')}\n")
            f.write(f"- Graph Hidden Size: {config.get('graph_encoder', {}).get('hidden_dim', 'N/A')}\n\n")

            # Best Results Summary
            f.write("## Best Results Summary\n\n")

            if self.best_accuracy["data"]:
                self._write_best_result(f, "Test Accuracy", self.best_accuracy)

            if self.best_f1["data"]:
                self._write_best_result(f, "F1 Score", self.best_f1)

            if self.best_val_accuracy["data"]:
                self._write_best_result(f, "Validation Accuracy", self.best_val_accuracy)

            # Final test results if provided
            if final_test_metrics:
                f.write("## Final Test Results\n")
                if final_model_path:
                    f.write(f"- Model Path: {final_model_path}\n")
                for key, value in final_test_metrics.items():
                    if 'time' in key.lower():
                        f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}s\n")
                    elif 'samples' in key.lower():
                        f.write(f"- {key.replace('_', ' ').title()}: {int(value)}\n")
                    else:
                        f.write(f"- {key.replace('_', ' ').title()}: {value:.2f}%\n")
                f.write("\n")

            # Epoch-by-epoch summary table
            f.write("## Training Progress\n\n")
            f.write("| Epoch | Train Loss | Train Acc | Val Acc | Val F1 | Val AUC |\n")
            f.write("|-------|------------|-----------|---------|--------|---------|\n")

            for epoch_data in self.all_epochs:
                epoch = epoch_data["epoch"]
                train_metrics = epoch_data["train_metrics"]
                val_metrics = epoch_data["val_metrics"]

                f.write(f"| {epoch:5d} | "
                       f"{train_metrics.get('loss', 0):10.4f} | "
                       f"{train_metrics.get('accuracy', 0):9.2f}% | "
                       f"{val_metrics.get('accuracy', 0):7.2f}% | "
                       f"{val_metrics.get('f1', 0):6.2f}% | "
                       f"{val_metrics.get('auc', 0):7.2f}% |\n")

        return str(filepath)

    def _write_best_result(self, f, criteria_name: str, best_data: Dict):
        """Write best result section to file."""
        data = best_data["data"]
        score = best_data["score"]

        f.write(f"### Best {criteria_name}: {score:.2f}%\n")
        f.write(f"- Achieved at Epoch: {data['epoch']}\n")
        f.write(f"- Timestamp: {data['timestamp'][:19]}\n")

        if data.get('model_path'):
            f.write(f"- Model Path: {data['model_path']}\n")

        # Write all metrics for this epoch
        train_metrics = data['train_metrics']
        val_metrics = data['val_metrics']
        test_metrics = data.get('test_metrics')

        f.write(f"- Train Accuracy: {train_metrics.get('accuracy', 0):.2f}%\n")
        f.write(f"- Train F1: {train_metrics.get('f1', 0):.2f}%\n")
        f.write(f"- Val Accuracy: {val_metrics.get('accuracy', 0):.2f}%\n")
        f.write(f"- Val F1: {val_metrics.get('f1', 0):.2f}%\n")
        f.write(f"- Val AUC: {val_metrics.get('auc', 0):.2f}%\n")

        if test_metrics:
            f.write(f"- Test Accuracy: {test_metrics.get('accuracy', 0):.2f}%\n")
            f.write(f"- Test F1: {test_metrics.get('f1', 0):.2f}%\n")

        f.write("\n")

    def _save_best_models_summary(self, timestamp: str) -> str:
        """Save summary of best models across all criteria."""
        filename = f"fusion_best_models_{timestamp}.txt"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            f.write(f"# Fusion Model Best Models Summary\n")
            f.write(f"# Run: {self.current_run_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 🏆 CHAMPION MODELS BY CRITERIA\n\n")

            if self.best_accuracy["data"]:
                f.write("### 🎯 Best Test Accuracy\n")
                self._write_champion_model(f, self.best_accuracy)

            if self.best_f1["data"]:
                f.write("### 🎪 Best F1 Score\n")
                self._write_champion_model(f, self.best_f1)

            if self.best_val_accuracy["data"]:
                f.write("### 🔝 Best Validation Accuracy\n")
                self._write_champion_model(f, self.best_val_accuracy)

            # Quick comparison
            f.write("## 📊 Quick Comparison\n\n")
            f.write("| Criteria | Score | Epoch | Model Path |\n")
            f.write("|----------|--------|-------|------------|\n")

            for name, best_data in [("Test Acc", self.best_accuracy),
                                   ("F1 Score", self.best_f1),
                                   ("Val Acc", self.best_val_accuracy)]:
                if best_data["data"]:
                    data = best_data["data"]
                    model_path = data.get('model_path', 'N/A')
                    if isinstance(model_path, str) and len(model_path) > 30:
                        model_path = "..." + model_path[-27:]
                    f.write(f"| {name:8s} | {best_data['score']:6.2f}% | "
                           f"{data['epoch']:5d} | {model_path} |\n")

        return str(filepath)

    def _write_champion_model(self, f, best_data: Dict):
        """Write champion model details."""
        data = best_data["data"]
        score = best_data["score"]

        f.write(f"**Score: {score:.2f}%**\n")
        f.write(f"- Epoch: {data['epoch']}\n")
        f.write(f"- Date: {data['timestamp'][:19]}\n")

        if data.get('model_path'):
            f.write(f"- Model Path: {data['model_path']}\n")

        # Performance profile
        train_metrics = data['train_metrics']
        val_metrics = data['val_metrics']
        test_metrics = data.get('test_metrics')

        f.write(f"\n**Performance Profile:**\n")
        f.write(f"- Training: {train_metrics.get('accuracy', 0):.2f}% acc, {train_metrics.get('f1', 0):.2f}% F1\n")
        f.write(f"- Validation: {val_metrics.get('accuracy', 0):.2f}% acc, {val_metrics.get('f1', 0):.2f}% F1, {val_metrics.get('auc', 0):.2f}% AUC\n")

        if test_metrics:
            f.write(f"- Test: {test_metrics.get('accuracy', 0):.2f}% acc, {test_metrics.get('f1', 0):.2f}% F1\n")

        f.write("\n")

    def _update_global_results(self, final_test_metrics: Optional[Dict],
                              final_model_path: Optional[str]) -> str:
        """Update global fusion results file."""
        global_file = self.save_dir / "fusion_global_results.txt"

        # Load existing results or create new
        all_runs = []
        if global_file.exists():
            try:
                # Parse existing results (simple approach)
                with open(global_file, 'r') as f:
                    content = f.read()
                    # Extract run count for basic tracking
                    if "Total Runs:" in content:
                        run_count = int(content.split("Total Runs:")[1].split()[0]) + 1
                    else:
                        run_count = 1
            except:
                run_count = 1
        else:
            run_count = 1

        # Write updated global results
        with open(global_file, 'w') as f:
            f.write(f"# Global Fusion Model Results\n")
            f.write(f"# Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Runs: {run_count}\n\n")

            f.write(f"## Latest Run: {self.current_run_name}\n")

            if self.best_accuracy["data"]:
                f.write(f"- Best Test Accuracy: {self.best_accuracy['score']:.2f}% "
                       f"(Epoch {self.best_accuracy['data']['epoch']})\n")

            if self.best_f1["data"]:
                f.write(f"- Best F1 Score: {self.best_f1['score']:.2f}% "
                       f"(Epoch {self.best_f1['data']['epoch']})\n")

            if self.best_val_accuracy["data"]:
                f.write(f"- Best Validation Accuracy: {self.best_val_accuracy['score']:.2f}% "
                       f"(Epoch {self.best_val_accuracy['data']['epoch']})\n")

            if final_test_metrics:
                f.write(f"\n**Final Test Performance:**\n")
                f.write(f"- Accuracy: {final_test_metrics.get('accuracy', 0):.2f}%\n")
                f.write(f"- F1 Score: {final_test_metrics.get('f1', 0):.2f}%\n")
                f.write(f"- Precision: {final_test_metrics.get('precision', 0):.2f}%\n")
                f.write(f"- Recall: {final_test_metrics.get('recall', 0):.2f}%\n")
                f.write(f"- AUC: {final_test_metrics.get('auc', 0):.2f}%\n")

            if final_model_path:
                f.write(f"- Final Model Path: {final_model_path}\n")

            f.write(f"\n## Configuration\n")
            config = self.current_config
            f.write(f"- Task: {config.get('task', 'N/A')}\n")
            f.write(f"- Fusion Type: {config.get('fusion', {}).get('fusion_type', 'N/A')}\n")
            f.write(f"- Learning Rate: {config.get('optim', {}).get('lr', 'N/A')}\n")
            f.write(f"- Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"- Total Epochs: {len(self.all_epochs)}\n")

        return str(global_file)

    def _generate_comprehensive_summary(self) -> str:
        """Generate comprehensive summary with statistics."""
        summary_file = self.save_dir / "fusion_training_summary.txt"

        with open(summary_file, 'w') as f:
            f.write(f"# Comprehensive Fusion Training Summary\n")
            f.write(f"# Run: {self.current_run_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"{'TRAINING SUMMARY':^80}\n")
            f.write("=" * 80 + "\n\n")

            # Overall stats
            if self.all_epochs:
                val_accuracies = [epoch['val_metrics'].get('accuracy', 0) for epoch in self.all_epochs]
                val_f1s = [epoch['val_metrics'].get('f1', 0) for epoch in self.all_epochs]
                train_losses = [epoch['train_metrics'].get('loss', 0) for epoch in self.all_epochs]

                f.write(f"## Training Statistics\n")
                f.write(f"- Total Epochs: {len(self.all_epochs)}\n")
                f.write(f"- Final Training Loss: {train_losses[-1]:.4f}\n")
                f.write(f"- Final Validation Accuracy: {val_accuracies[-1]:.2f}%\n")
                f.write(f"- Final Validation F1: {val_f1s[-1]:.2f}%\n")
                f.write(f"- Max Validation Accuracy: {max(val_accuracies):.2f}%\n")
                f.write(f"- Max Validation F1: {max(val_f1s):.2f}%\n")
                f.write(f"- Average Validation Accuracy: {np.mean(val_accuracies):.2f}%\n")
                f.write(f"- Validation Accuracy Std: {np.std(val_accuracies):.2f}%\n\n")

            # Best models summary
            f.write("## Best Models Achieved\n\n")

            if self.best_accuracy["data"]:
                f.write(f"🎯 **Best Test Accuracy**: {self.best_accuracy['score']:.2f}% at Epoch {self.best_accuracy['data']['epoch']}\n")

            if self.best_f1["data"]:
                f.write(f"🎪 **Best F1 Score**: {self.best_f1['score']:.2f}% at Epoch {self.best_f1['data']['epoch']}\n")

            if self.best_val_accuracy["data"]:
                f.write(f"🔝 **Best Validation Accuracy**: {self.best_val_accuracy['score']:.2f}% at Epoch {self.best_val_accuracy['data']['epoch']}\n")

            f.write("\n## Model Configuration Used\n")
            config = self.current_config
            for key, value in config.items():
                if isinstance(value, dict):
                    f.write(f"- {key.title()}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  - {subkey}: {subvalue}\n")
                else:
                    f.write(f"- {key.title()}: {value}\n")

        return str(summary_file)


# Convenience functions for easy integration
def save_fusion_results(run_name: str,
                       config: Dict[str, Any],
                       best_train_metrics: Dict[str, float],
                       best_val_metrics: Dict[str, float],
                       final_test_metrics: Dict[str, float],
                       final_model_path: str,
                       save_dir: str = "txt file") -> List[str]:
    """
    Convenient function to save fusion model results.

    Args:
        run_name: Name for this training run
        config: Configuration dictionary
        best_train_metrics: Best training metrics achieved
        best_val_metrics: Best validation metrics achieved
        final_test_metrics: Final test metrics
        final_model_path: Path to the final best model
        save_dir: Directory to save results

    Returns:
        List of generated file paths
    """
    manager = FusionResultsManager(save_dir)
    manager.start_run(run_name, config)

    # Add a single "epoch" representing the best results
    manager.save_epoch_results(
        epoch=1,
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
        test_metrics=final_test_metrics,
        model_path=final_model_path
    )

    return manager.finalize_and_save_best(final_test_metrics, final_model_path)


def quick_save_fusion_best(accuracy: float,
                          f1: float,
                          precision: float,
                          recall: float,
                          auc: float,
                          model_path: str,
                          task: str = "unknown",
                          fusion_type: str = "unknown") -> str:
    """
    Quick save function for fusion results with minimal data.

    Args:
        accuracy: Test accuracy percentage
        f1: Test F1 score percentage
        precision: Test precision percentage
        recall: Test recall percentage
        auc: Test AUC percentage
        model_path: Path to the model
        task: Task name
        fusion_type: Type of fusion used

    Returns:
        Path to saved results file
    """
    save_dir = Path("txt file")
    save_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fusion_quick_results_{timestamp}.txt"
    filepath = save_dir / filename

    with open(filepath, 'w') as f:
        f.write(f"# Fusion Model Quick Results\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Best Model Results\n")
        f.write(f"- **Task**: {task.upper()}\n")
        f.write(f"- **Fusion Type**: {fusion_type}\n")
        f.write(f"- **Model Path**: {model_path}\n\n")

        f.write(f"## Performance Metrics\n")
        f.write(f"- **Test Accuracy**: {accuracy:.2f}%\n")
        f.write(f"- **Test F1 Score**: {f1:.2f}%\n")
        f.write(f"- **Test Precision**: {precision:.2f}%\n")
        f.write(f"- **Test Recall**: {recall:.2f}%\n")
        f.write(f"- **Test AUC**: {auc:.2f}%\n\n")

        f.write(f"## Summary\n")
        f.write(f"This fusion model achieved {accuracy:.2f}% accuracy and {f1:.2f}% F1 score.\n")
        f.write(f"The model shows {'excellent' if accuracy > 85 else 'good' if accuracy > 80 else 'moderate'} performance.\n")

    print(f"✅ Fusion results saved to: {filepath}")
    return str(filepath)


# Integration example for existing fusion training
def integrate_with_training_loop():
    """
    Example of how to integrate with existing fusion training code.
    """
    print("""
    # Integration Example:
    # Add this to your existing fusion training script

    from model.save_fusion_results import FusionResultsManager

    # At the beginning of training
    results_manager = FusionResultsManager()
    results_manager.start_run(run_name, config)

    # In your training loop (after each epoch)
    results_manager.save_epoch_results(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=None,  # or test_metrics if available
        model_path=model_path if is_best_epoch else None
    )

    # After training completes
    final_test_metrics = test_model()  # Your final testing
    files = results_manager.finalize_and_save_best(final_test_metrics, final_model_path)
    print(f"Results saved to {len(files)} files!")
    """)


if __name__ == "__main__":
    # Show integration example
    integrate_with_training_loop()
