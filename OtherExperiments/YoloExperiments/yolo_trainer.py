"""
YOLO Classification Training Script for ODIR-2019
Supports YOLOv11 and YOLOv26 classification models
Includes F1-score calculation, confusion matrix, inference timing, and optional W&B integration
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import wandb
import cv2
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class ODIRYOLOTrainer:
    """
    YOLO Classification Trainer for ODIR-2019 dataset
    """

    def __init__(self, model_version='yolo11n-cls',
                 use_wandb=False, wandb_project_name=None, wandb_run_name=None):
        """
        Initialize trainer

        Args:
            model_version: Model version (yolo11n-cls, yolo11s-cls, yolo11m-cls, 
                           yolo11l-cls, yolo11x-cls or yolo26 variants)
            run_name: Project name for saving results
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model_version = model_version
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.results = None
        self.test_predictions = []
        self.test_labels = []
        self.test_probabilities = []  # Store probabilities for F1 curve
        self.inference_times = []
        self.use_wandb = use_wandb
        self.training_history = []  # Store training metrics history
        self.val_metrics_history = []  # Store validation metrics history
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.wandb_run_id = None
        # ODIR-2019 class names (8 classes)
        self.class_names = [
            'normal', 'diabetes', 'glaucoma', 'cataract',
            'ageing', 'hypertension', 'myopia', 'other'
        ]
        self.num_classes = len(self.class_names)

        # Setup run directory
        self.run_dir = Path('yolo-runs')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
        print(f"Model version: {model_version}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        print(f"W&B logging: {use_wandb}")
        print(f"Run directory: {self.run_dir}")

    def wandb_init(self):
        """Initialize W&B run"""
        if self.use_wandb and (self.wandb_project_name is not None and self.wandb_run_name is not None):
            if self.wandb_run_id is None:
                # start new run
                wandb_run = wandb.init(project=self.wandb_project_name,
                                            name=self.wandb_run_name)
                self.wandb_run_id = wandb_run.id
            else:
                # resume existing run
                wandb.init(project=self.wandb_project_name,name=self.wandb_run_name,id=self.wandb_run_id,resume="must")
            
    
    def load_model(self, model_path=None):
        """
        Load a trained model from file

        Args:
            model_path: Path to model weights (.pt file)
                       If None, loads pretrained model based on model_version
        """
        if model_path is None:
            # Load pretrained model
            print(f"Loading pretrained model: {self.model_version}")
            self.model = YOLO(f'{self.model_version}.pt')
        else:
            # Load trained model
            model_path = Path(model_path)
            if model_path.exists():
                print(f"Loading model from: {model_path}")
                self.model = YOLO(str(model_path))
            else:
                print(f"Model not found at: {model_path}")
                print(f"Loading pretrained model: {self.model_version}")
                self.model = YOLO(f'{self.model_version}.pt')

        # Move model to device
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")

    def save_model(self, save_dir=None, model_name=None,
                   save_format='pt', include_optimizer=False):
        """
        Save trained model to disk

        Args:
            save_dir: Directory to save the model (default: run_dir/saved_models)
            model_name: Name for saved model (uses timestamp if None)
            save_format: Format to save model ('pt', 'onnx', 'torchscript')
            include_optimizer: Whether to include optimizer state (for .pt format)

        Returns:
            save_path: Path to saved model
        """
        if self.model is None:
            print("No model to save. Please train or load a model first.")
            return None

        # Set default save directory
        if save_dir is None:
            save_dir = self.run_dir / 'saved_models'
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_version}_{timestamp}"

        # Save based on format
        if save_format == 'pt':
            # Save PyTorch model
            save_path = save_dir / f"{model_name}.pt"

            # Export using Ultralytics save
            self.model.save(str(save_path))

            # Save additional metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = {
                'model_version': self.model_version,
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'save_time': timestamp,
                'device': self.device,
                'training_history': self.training_history,
                'val_metrics_history': self.val_metrics_history
            }

            # Save metadata
            metadata_path = save_dir / f"{model_name}_metadata.pkl"
            joblib.dump(metadata, metadata_path)

            print(f"Model saved to: {save_path}")
            print(f"Metadata saved to: {metadata_path}")

        elif save_format == 'onnx':
            # Export to ONNX format
            save_path = save_dir / f"{model_name}.onnx"

            # Export model to ONNX with 512 image size
            success = self.model.export(
                format='onnx', imgsz=512, simplify=True)

            if success:
                # Find the exported model
                export_dir = Path('runs/classify') / self.model_version
                onnx_files = list(export_dir.glob('*.onnx'))

                if onnx_files:
                    # Move to our save directory
                    onnx_files[0].rename(save_path)
                    print(f"ONNX model saved to: {save_path}")
                else:
                    print("Failed to find exported ONNX model")
            else:
                print("Failed to export model to ONNX")

        elif save_format == 'torchscript':
            # Export to TorchScript format
            save_path = save_dir / f"{model_name}.torchscript"

            # Export model to TorchScript with 512 image size
            success = self.model.export(format='torchscript', imgsz=512)

            if success:
                # Find the exported model
                export_dir = Path('runs/classify') / self.model_version
                torchscript_files = list(export_dir.glob('*.torchscript'))

                if torchscript_files:
                    # Move to our save directory
                    torchscript_files[0].rename(save_path)
                    print(f"TorchScript model saved to: {save_path}")
                else:
                    print("Failed to find exported TorchScript model")
            else:
                print("Failed to export model to TorchScript")

        return save_path

    def load_saved_model(self, model_path, metadata_path=None):
        """
        Load a saved model with metadata

        Args:
            model_path: Path to saved model (.pt file)
            metadata_path: Path to metadata file (if None, tries to find automatically)
        """
        model_path = Path(model_path)

        # Load metadata if available
        if metadata_path is None:
            # Try to find metadata file automatically
            metadata_path = model_path.parent / \
                f"{model_path.stem}_metadata.pkl"

        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_version = metadata.get(
                'model_version', self.model_version)
            self.class_names = metadata.get('class_names', self.class_names)
            self.num_classes = metadata.get('num_classes', self.num_classes)
            self.training_history = metadata.get('training_history', [])
            self.val_metrics_history = metadata.get('val_metrics_history', [])
            print(f"Loaded metadata from: {metadata_path}")
            print(f"Model version: {self.model_version}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Classes: {self.class_names}")
            if self.training_history:
                print(
                    f"Loaded training history with {len(self.training_history)} epochs")

        # Load the model
        self.load_model(model_path)

    def train_from_folder(self, data_folder, run_name=None, epochs=100, imgsz=512,
                          batch_size=32, patience=50):
        """
        Train YOLO classification model directly from folder structure
        Includes validation during training, F1 curve plotting, and confusion matrix

        Args:
            data_folder: Path to data folder with train/val/test subfolders
            run_name: Name for this training run (uses timestamp if None)
            epochs: Number of training epochs
            imgsz: Image size (512)
            batch_size: Batch size
            patience: Early stopping patience            
        Returns:
            results: Training results
        """
        print(f"\n{'='*50}")
        print(f"Training {self.model_version} on ODIR-2019")
        print(f"Image size: {imgsz}x{imgsz}")
        print(f"{'='*50}")

        # Load model
        print(f"Loading model: {self.model_version}")
        self.model = YOLO(f'{self.model_version}.pt')

        # Create run name
        if run_name is None:
            run_name = f"{self.model_version}_imgsz{imgsz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare data folder structure
        data_folder = Path(data_folder)
        print(f"Data folder: {data_folder}")

        # Check if data folder has train/val structure
        if not (data_folder / 'train').exists():
            raise ValueError(f"Train folder not found in {data_folder}")

        if not (data_folder / 'val').exists():
            print(f"Warning: Validation folder not found in {data_folder}")

        # Set up training arguments - validation will be done by Ultralytics automatically
        train_args = {
            # Ultralytics expects a folder with train/val subfolders
            'data': str(data_folder),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': patience,
            'project': str(self.run_dir),
            'name': run_name,
            'exist_ok': True,
            'device': self.device,
            'workers': 2,
            'seed': 42,
            'val': True,  # Enable validation during training
            'save': True,
            'plots': True,
            'save_period': -1,
            'cache': False,
            'amp': True,
            'fraction': 1.0,
            'resume': False,
            'verbose': True,
        }

        # Log to W&B if enabled
        if self.use_wandb:
            self.wandb_init()
            train_args['project'] = str(self.run_dir)
            print("W&B integration enabled for training")


        # Train the model
        print("\nStarting training...")
        start_time = time.time()
        self.results = self.model.train(**train_args)
        training_time = time.time() - start_time
        metrics=self.model.val()  # Final validation after training
        print(f"\nFinal validation metrics: {metrics}")
        if self.use_wandb:
            self.wandb_finish()
        print(f"\nTraining completed in {training_time:.2f} seconds")
        # Extract training history
        self._extract_training_history()

        # Save training results
        self._save_training_summary(run_name, training_time)

        # Plot training metrics including F1 curve
        self.plot_training_metrics(run_name)

        # Log training summary to W&B
        if self.use_wandb:
            metrics = self.results.results_dict if hasattr(
                self.results, 'results_dict') else {}

            self.wandb_init()
            wandb.log({
                "training_time": training_time,
                "image_size": imgsz,
                "batch_size": batch_size,
                "best_accuracy": metrics.get('metrics/accuracy_top1', 0),
                "best_precision": metrics.get('metrics/precision', 0),
                "best_recall": metrics.get('metrics/recall', 0),
                "best_f1": metrics.get('metrics/f1', 0),
                "best_epoch": metrics.get('epoch', 0)
            })

            # Log training curves to W&B
            self._log_training_curves_to_wandb()
            self.wandb_finish()

        return self.results

    def _extract_training_history(self):
        """Extract training history from results"""
        if self.results is None:
            return

        # Try to extract metrics from results
        try:
            # Ultralytics stores metrics in results.metrics
            if hasattr(self.results, 'metrics'):
                # Check if it's a dict or list
                if isinstance(self.results.metrics, dict):
                    self.training_history.append({
                        'epoch': len(self.training_history) + 1,
                        'train_loss': self.results.metrics.get('train/loss', 0),
                        'val_loss': self.results.metrics.get('val/loss', 0),
                        'accuracy': self.results.metrics.get('metrics/accuracy_top1', 0),
                        'f1': self.results.metrics.get('metrics/f1', 0),
                        'precision': self.results.metrics.get('metrics/precision', 0),
                        'recall': self.results.metrics.get('metrics/recall', 0)
                    })
                elif isinstance(self.results.metrics, list) and len(self.results.metrics) > 0:
                    # Handle list format
                    latest_metrics = self.results.metrics[-1]
                    if isinstance(latest_metrics, dict):
                        self.training_history.append({
                            'epoch': len(self.training_history) + 1,
                            'train_loss': latest_metrics.get('train/loss', 0),
                            'val_loss': latest_metrics.get('val/loss', 0),
                            'accuracy': latest_metrics.get('metrics/accuracy_top1', 0),
                            'f1': latest_metrics.get('metrics/f1', 0),
                            'precision': latest_metrics.get('metrics/precision', 0),
                            'recall': latest_metrics.get('metrics/recall', 0)
                        })
        except Exception as e:
            print(f"Warning: Could not extract training history: {e}")

    def _save_training_summary(self, run_name, training_time):
        """Save training summary to file"""
        summary_path = self.run_dir / run_name / 'training_summary.txt'
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            f.write(f"Training Summary - {run_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {self.model_version}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write(f"Number of classes: {self.num_classes}\n")

            if self.training_history:
                f.write("\nTraining History (Last 5 epochs):\n")
                for epoch_data in self.training_history[-5:]:
                    f.write(f"Epoch {epoch_data['epoch']}: "
                            f"Train Loss: {epoch_data.get('train_loss', 0):.4f}, "
                            f"Val Loss: {epoch_data.get('val_loss', 0):.4f}, "
                            f"Accuracy: {epoch_data.get('accuracy', 0):.4f}, "
                            f"F1: {epoch_data.get('f1', 0):.4f}\n")

        print(f"Training summary saved to: {summary_path}")

    def plot_training_metrics(self, run_name):
        """
        Plot training metrics including F1 curve

        Args:
            run_name: Name of the training run
        """
        if not self.training_history:
            print("No training history to plot")
            return

        # Create plots directory
        plots_dir = self.run_dir / run_name / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Extract metrics
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h.get('train_loss', 0) for h in self.training_history]
        val_losses = [h.get('val_loss', 0) for h in self.training_history]
        accuracies = [h.get('accuracy', 0) for h in self.training_history]
        f1_scores = [h.get('f1', 0) for h in self.training_history]

        # Create comprehensive training metrics plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(epochs, train_losses, 'b-',
                        label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-',
                        label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, accuracies, 'g-',
                        label='Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: F1 Score
        axes[0, 2].plot(epochs, f1_scores, 'm-', label='F1 Score', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('F1 Score over Epochs')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: F1 Curve (Precision-Recall)
        axes[1, 0].set_title('Precision-Recall Curve with F1 Contours')
        # Generate F1 contours
        f1_scores_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
        p = np.linspace(0.01, 1, 100)

        for f1 in f1_scores_levels:
            r = f1 * p / (2 * p - f1)
            r = np.clip(r, 0, 1)
            valid = p > f1/2
            axes[1, 0].plot(p[valid], r[valid], '--',
                            alpha=0.5, label=f'F1={f1}')

        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend(loc='lower left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])

        # Plot 5: Training vs Validation Loss Difference
        if len(train_losses) > 0 and len(val_losses) > 0:
            loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
            axes[1, 1].plot(epochs, loss_diff, 'c-',
                            label='|Train Loss - Val Loss|', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].set_title('Training vs Validation Loss Difference')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Learning Rate Schedule (if available)
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].text(0.5, 0.5, 'LR schedule\navailable in\nmodel checkpoints',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure
        plot_path = plots_dir / 'comprehensive_training_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {plot_path}")

        # Also save individual plots
        self._save_individual_plots(
            plots_dir, epochs, train_losses, val_losses, accuracies, f1_scores)

        plt.show()

    def _save_individual_plots(self, plots_dir, epochs, train_losses, val_losses, accuracies, f1_scores):
        """Save individual plot files"""
        # F1 Score plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, f1_scores, 'm-', linewidth=3, label='F1 Score')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score Progression During Training', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        f1_plot_path = plots_dir / 'f1_score_progression.png'
        plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
        print(f"F1 score plot saved to: {f1_plot_path}")
        plt.close()

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-',
                 linewidth=3, label='Training Loss')
        plt.plot(epochs, val_losses, 'r-',
                 linewidth=3, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path = plots_dir / 'loss_curves.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {loss_plot_path}")
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, 'g-', linewidth=3, label='Accuracy')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy Progression During Training', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_plot_path = plots_dir / 'accuracy_progression.png'
        plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved to: {acc_plot_path}")
        plt.close()

    def _log_training_curves_to_wandb(self):
        """Log training curves to W&B"""
        if not self.training_history or not self.use_wandb:
            return

        # Create a figure with all training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h.get('train_loss', 0) for h in self.training_history]
        val_losses = [h.get('val_loss', 0) for h in self.training_history]
        accuracies = [h.get('accuracy', 0) for h in self.training_history]
        f1_scores = [h.get('f1', 0) for h in self.training_history]

        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-',
                        label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-',
                        label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, accuracies, 'g-',
                        label='Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: F1 Score
        axes[1, 0].plot(epochs, f1_scores, 'm-', label='F1 Score', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score over Epochs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: F1 contours
        axes[1, 1].set_title('Precision-Recall Curve with F1 Contours')
        f1_scores_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
        p = np.linspace(0.01, 1, 100)

        for f1 in f1_scores_levels:
            r = f1 * p / (2 * p - f1)
            r = np.clip(r, 0, 1)
            valid = p > f1/2
            axes[1, 1].plot(p[valid], r[valid], '--',
                            alpha=0.5, label=f'F1={f1}')

        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend(loc='lower left')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

        # Log to W&B
        wandb.log({"training_curves": wandb.Image(fig)})

        # Also log individual metrics as line plots
        for epoch_data in self.training_history:
            wandb.log({
                "epoch": epoch_data['epoch'],
                "train_loss": epoch_data.get('train_loss', 0),
                "val_loss": epoch_data.get('val_loss', 0),
                "accuracy": epoch_data.get('accuracy', 0),
                "f1_score": epoch_data.get('f1', 0),
                "precision": epoch_data.get('precision', 0),
                "recall": epoch_data.get('recall', 0)
            })

        plt.close()

    def evaluate_on_validation(self, data_folder, imgsz=512, generate_confusion_matrix=True):
        """
        Evaluate model on validation set during training
        Includes confusion matrix plotting

        Args:
            data_folder: Path to data folder
            imgsz: Image size for evaluation
            generate_confusion_matrix: Whether to generate confusion matrix

        Returns:
            metrics: Dictionary with evaluation metrics
        """
        data_folder = Path(data_folder)
        val_folder = data_folder / 'val'

        if not val_folder.exists():
            print(f"Validation folder not found: {val_folder}")
            return None

        return self._evaluate_on_split(val_folder, 'validation', imgsz, generate_confusion_matrix)

    def _evaluate_on_split(self, split_folder, split_name='validation', imgsz=512, generate_confusion_matrix=True):
        """
        Evaluate model on a specific split with confusion matrix

        Args:
            split_folder: Path to split folder
            split_name: Name of the split
            imgsz: Image size for evaluation
            generate_confusion_matrix: Whether to generate confusion matrix

        Returns:
            metrics: Dictionary with evaluation metrics
        """
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None

        print(f"\nEvaluating on {split_name} set...")

        # Get all image files
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_folder.glob(f'*/*.{ext}'))

        if not image_files:
            # Try flat structure
            for ext in image_extensions:
                image_files.extend(split_folder.glob(f'*.{ext}'))

        print(f"Found {len(image_files)} images in {split_name} set")

        # Run inference
        predictions = []
        true_labels = []
        probabilities = []
        inference_times = []

        for img_path in tqdm(image_files, desc=f"Evaluating {split_name}"):
            # Get true label from folder structure
            true_label = img_path.parent.name

            # Convert class name to index
            if true_label in self.class_names:
                true_idx = self.class_names.index(true_label)
            else:
                # Try to parse as integer
                try:
                    true_idx = int(true_label)
                except ValueError:
                    print(
                        f"Warning: Could not parse label '{true_label}' for image {img_path.name} - skipping")
                    continue

            # Measure inference time
            start_time = time.time()

            # Run inference
            results = self.model(img_path, imgsz=imgsz, verbose=False)

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Get prediction
            if hasattr(results[0], 'probs'):
                pred_probs = results[0].probs.data.cpu().numpy()
                pred_class = np.argmax(pred_probs)
                probabilities.append(pred_probs)
            else:
                # Fallback for detection models
                pred_class = results[0].boxes.cls[0].int(
                ).item() if len(results[0].boxes) > 0 else 0
                probabilities.append(np.zeros(self.num_classes))

            predictions.append(pred_class)
            true_labels.append(true_idx)

        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))

        # Calculate F1 scores
        f1_micro = f1_score(true_labels, predictions, average='micro')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')

        # Per-class F1 scores
        f1_per_class = f1_score(true_labels, predictions, average=None)

        # Classification report
        class_report = classification_report(
            true_labels,
            predictions,
            target_names=self.class_names,
            output_dict=True
        )

        # Calculate inference statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        # Print results
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS for {split_name.upper()} SET")
        print(f"{'='*50}")
        print(f"Total images: {len(predictions)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Micro): {f1_micro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"\nAverage Inference Time: {avg_inference_time:.4f} seconds")
        print(f"Inference Time Std: {std_inference_time:.4f} seconds")
        print(f"FPS: {fps:.2f}")

        print(f"\nPer-class F1 Scores:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:15}: {f1_per_class[i]:.4f}")

        # Store validation metrics history
        self.val_metrics_history.append({
            'split': split_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_inference_time': avg_inference_time,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        })

        # Generate confusion matrix if requested
        if generate_confusion_matrix:
            cm = confusion_matrix(true_labels, predictions)
            self.plot_confusion_matrix(cm, split_name, imgsz)

        # Return metrics dictionary
        metrics_dict = {
            'split': split_name,
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.class_names, f1_per_class)),
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'fps': fps,
            'image_size': imgsz,
            'total_images': len(predictions),
            'classification_report': class_report
        }

        if generate_confusion_matrix:
            metrics_dict['confusion_matrix'] = cm

        return metrics_dict

    def plot_confusion_matrix(self, cm, split_name='validation', imgsz=512):
        """
        Plot and save confusion matrix with enhanced visualization

        Args:
            cm: Confusion matrix array
            split_name: Name of the data split
            imgsz: Image size used for evaluation
        """
        # Create plots directory
        plots_dir = self.run_dir / 'confusion_matrices'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Absolute values confusion matrix
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title(
            f'Confusion Matrix - {split_name.capitalize()} Set\n(Absolute Values)', fontsize=14)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xticks(np.arange(len(self.class_names)))
        ax1.set_yticks(np.arange(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.set_yticklabels(self.class_names)

        # Add text annotations for absolute values
        thresh = cm.max() / 2.
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                ax1.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=10)

        # Plot 2: Normalized confusion matrix
        im2 = ax2.imshow(
            cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title(
            f'Confusion Matrix - {split_name.capitalize()} Set\n(Normalized)', fontsize=14)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xticks(np.arange(len(self.class_names)))
        ax2.set_yticks(np.arange(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.set_yticklabels(self.class_names)

        # Add text annotations for normalized values
        thresh = cm_normalized.max() / 2.
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                ax2.text(j, i, format(cm_normalized[i, j], '.2f'),
                         ha="center", va="center",
                         color="white" if cm_normalized[i,
                                                        j] > thresh else "black",
                         fontsize=10)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'confusion_matrix_{self.model_version}_{split_name}_{imgsz}_{timestamp}.png'
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {plot_path}")

        # Also save a simplified version
        self._save_simple_confusion_matrix(
            cm, split_name, imgsz, plots_dir, timestamp)

        # Log to W&B if enabled
        if self.use_wandb:
            self.wandb_init()
            wandb.log({
                f"{split_name}_confusion_matrix": wandb.Image(fig),
                f"{split_name}_confusion_matrix_absolute": wandb.Image(ax1.figure),
                f"{split_name}_confusion_matrix_normalized": wandb.Image(ax2.figure)
            })
            self.wandb_finish()

        plt.show()

    def _save_simple_confusion_matrix(self, cm, split_name, imgsz, plots_dir, timestamp):
        """Save a simplified confusion matrix"""
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)

        plt.title(f'Confusion Matrix - {self.model_version}\n{split_name.capitalize()} Set ({imgsz}x{imgsz})',
                  fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save simplified version
        simple_filename = f'confusion_matrix_simple_{self.model_version}_{split_name}_{imgsz}_{timestamp}.png'
        simple_path = plots_dir / simple_filename
        plt.savefig(simple_path, dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_on_test(self, data_folder, imgsz=512):
        """
        Evaluate model on test set

        Args:
            data_folder: Path to data folder
            imgsz: Image size for evaluation

        Returns:
            metrics: Dictionary with evaluation metrics
        """
        data_folder = Path(data_folder)
        test_folder = data_folder / 'test'

        if not test_folder.exists():
            print(f"Test folder not found: {test_folder}")
            return None

        return self._evaluate_on_split(test_folder, 'test', imgsz, generate_confusion_matrix=True)

    def predict_single_image(self, image_path, show_result=True, save_result=False):
        """
        Make prediction on a single image

        Args:
            image_path: Path to image file
            show_result: Whether to display the result
            save_result: Whether to save the result image

        Returns:
            result_dict: Dictionary containing prediction results
        """
        if self.model is None:
            print("Model not loaded. Please load or train a model first.")
            return None

        # Load and preprocess image
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"Image not found: {image_path}")
            return None

        print(f"\nPredicting on: {img_path.name}")

        # Measure inference time
        start_time = time.time()

        # Run inference
        results = self.model(img_path, verbose=False)

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")

        # Get prediction results
        if hasattr(results[0], 'probs'):
            # Classification model
            pred_probs = results[0].probs.data.cpu().numpy()
            pred_class_idx = np.argmax(pred_probs)
            pred_class_name = self.class_names[pred_class_idx]
            pred_confidence = float(pred_probs[pred_class_idx])

            # Get top-3 predictions
            top_k = min(3, len(self.class_names))
            top_indices = np.argsort(pred_probs)[-top_k:][::-1]
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(pred_probs[idx]),
                    'class_idx': int(idx)
                }
                for idx in top_indices
            ]

            # Print results
            print(f"\nPrediction Results:")
            print(
                f"  Predicted Class: {pred_class_name} (Index: {pred_class_idx})")
            print(f"  Confidence: {pred_confidence:.4f}")
            print(f"\nTop-{top_k} Predictions:")
            for i, pred in enumerate(top_predictions):
                print(
                    f"  {i+1}. {pred['class']:15} - Confidence: {pred['confidence']:.4f}")

        else:
            # Detection model (if applicable)
            print("Warning: Model appears to be a detection model, not classification")
            pred_class_idx = 0
            pred_class_name = self.class_names[0]
            pred_confidence = 0.0
            top_predictions = []
            pred_probs = np.zeros(len(self.class_names))

        # Display result if requested
        if show_result:
            self._display_prediction_result(
                img_path,
                pred_class_name,
                pred_confidence,
                top_predictions,
                pred_probs
            )

        # Save result if requested
        if save_result:
            save_path = self._save_prediction_result(
                img_path,
                pred_class_name,
                pred_confidence,
                top_predictions,
                pred_probs
            )
            print(f"Result saved to: {save_path}")

        # Create result dictionary
        result_dict = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'predicted_class': pred_class_name,
            'predicted_class_idx': pred_class_idx,
            'confidence': pred_confidence,
            'inference_time': inference_time,
            'top_predictions': top_predictions,
            'all_probabilities': pred_probs.tolist()
        }

        return result_dict

    def _display_prediction_result(self, img_path, pred_class, confidence, top_predictions, pred_probs):
        """Display prediction result with image"""
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Display image
        ax1.imshow(img)
        ax1.set_title(f"Input Image: {img_path.name}", fontsize=14)
        ax1.axis('off')

        # Add prediction text overlay
        pred_text = f"Predicted: {pred_class}\nConfidence: {confidence:.2%}"
        ax1.text(0.02, 0.98, pred_text, transform=ax1.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create bar chart for all probabilities
        x_pos = np.arange(len(self.class_names))
        colors = ['red' if i == np.argmax(
            pred_probs) else 'skyblue' for i in range(len(self.class_names))]

        ax2.barh(x_pos, pred_probs, color=colors)
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(self.class_names)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim([0, 1])
        ax2.invert_yaxis()

        # Add probability values on bars
        for i, prob in enumerate(pred_probs):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def _save_prediction_result(self, img_path, pred_class, confidence, top_predictions, pred_probs):
        """Save prediction result to file"""
        # Create results directory
        results_dir = self.run_dir / 'predictions'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Read image
        img = cv2.imread(str(img_path))

        # Add text overlay to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Add prediction text
        text = f"Predicted: {pred_class} ({confidence:.2%})"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 30

        # Draw background rectangle
        cv2.rectangle(img,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)

        # Draw text
        cv2.putText(img, text, (text_x, text_y),
                    font, font_scale, (0, 255, 0), thickness)

        # Add top predictions
        for i, pred in enumerate(top_predictions[:3]):
            pred_text = f"{i+1}. {pred['class']}: {pred['confidence']:.3f}"
            y_pos = text_y + (i + 1) * 30

            # Draw background
            pred_text_size = cv2.getTextSize(
                pred_text, font, font_scale-0.2, thickness-1)[0]
            cv2.rectangle(img,
                          (text_x - 5, y_pos - pred_text_size[1] - 5),
                          (text_x + pred_text_size[0] + 5, y_pos + 5),
                          (0, 0, 0), -1)

            # Draw text
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(img, pred_text, (text_x, y_pos),
                        font, font_scale-0.2, color, thickness-1)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = results_dir / f"pred_{img_path.stem}_{timestamp}.jpg"
        cv2.imwrite(str(save_path), img)

        # Save prediction details to CSV
        csv_path = results_dir / f"predictions_{timestamp}.csv"

        prediction_data = {
            'timestamp': [timestamp],
            'image_path': [str(img_path)],
            'image_name': [img_path.name],
            'predicted_class': [pred_class],
            'confidence': [confidence],
            'model_version': [self.model_version]
        }

        # Add all probabilities to CSV
        for i, class_name in enumerate(self.class_names):
            prediction_data[f'prob_{class_name}'] = [pred_probs[i]]

        df = pd.DataFrame(prediction_data)

        # Append to existing CSV or create new
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(csv_path, index=False)

        return save_path

    def run_inference_benchmark(self, data_folder, num_iterations=100, batch_sizes=[1, 4, 8, 16], imgsz=512):
        """
        Run inference timing benchmark with different batch sizes and 512x512 image size

        Args:
            data_folder: Path to data folder
            num_iterations: Number of inference iterations per batch size
            batch_sizes: List of batch sizes to test
            imgsz: Image size for benchmarking (512)

        Returns:
            benchmark_results: Dictionary with benchmark results
        """
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None

        benchmark_results = {}

        # Load a sample image for benchmarking
        data_folder = Path(data_folder)
        test_folder = data_folder / 'test'

        if not test_folder.exists():
            print(f"Test folder not found: {test_folder}")
            return None

        # Find a sample image
        image_files = list(test_folder.glob('*/*.jpg')
                           ) or list(test_folder.glob('*.jpg'))

        if not image_files:
            print("No test images found!")
            return None

        sample_image = str(image_files[0])

        for batch_size in batch_sizes:
            print(
                f"\nBenchmarking with batch size: {batch_size}, Image size: {imgsz}")

            # Create batch of images
            batch_images = [sample_image] * batch_size

            # Warm-up
            print("  Warming up...")
            for _ in range(10):
                _ = self.model(batch_images, imgsz=imgsz, verbose=False)

            # Benchmark
            print(f"  Running {num_iterations} iterations...")
            times = []

            for _ in tqdm(range(num_iterations), desc=f"Batch {batch_size}"):
                start_time = time.time()
                _ = self.model(batch_images, imgsz=imgsz, verbose=False)
                times.append(time.time() - start_time)

            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = batch_size / avg_time

            print(
                f"  Average time per batch: {avg_time:.4f} ± {std_time:.4f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Time per image: {avg_time/batch_size:.4f} seconds")

            benchmark_results[f'batch_{batch_size}'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': fps,
                'time_per_image': avg_time/batch_size,
                'image_size': imgsz
            }

        return benchmark_results

    def save_evaluation_results(self, metrics_dict, benchmark_results=None, split_name='test'):
        """Save evaluation results to CSV"""
        # Create results directory
        results_dir = self.run_dir / 'evaluation_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main metrics
        metrics_df = pd.DataFrame([{
            'model': self.model_version,
            'timestamp': timestamp,
            'split': split_name,
            'accuracy': metrics_dict.get('accuracy', 0),
            'f1_micro': metrics_dict.get('f1_micro', 0),
            'f1_macro': metrics_dict.get('f1_macro', 0),
            'f1_weighted': metrics_dict.get('f1_weighted', 0),
            'avg_inference_time': metrics_dict.get('avg_inference_time', 0),
            'fps': metrics_dict.get('fps', 0),
            'image_size': metrics_dict.get('image_size', 512),
            'total_images': metrics_dict.get('total_images', 0)
        }])

        metrics_path = results_dir / \
            f'{split_name}_metrics_{self.model_version}_{timestamp}.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")

        # Save per-class F1 scores
        f1_per_class = metrics_dict.get('f1_per_class', {})
        if f1_per_class:
            f1_df = pd.DataFrame(list(f1_per_class.items()),
                                 columns=['class', 'f1_score'])
            f1_path = results_dir / \
                f'{split_name}_f1_scores_{self.model_version}_{timestamp}.csv'
            f1_df.to_csv(f1_path, index=False)
            print(f"Per-class F1 scores saved to: {f1_path}")

        # Save benchmark results
        if benchmark_results:
            bench_df = pd.DataFrame(benchmark_results).T.reset_index()
            bench_df.columns = ['batch_size'] + list(bench_df.columns[1:])
            bench_path = results_dir / \
                f'{split_name}_benchmark_{self.model_version}_{timestamp}.csv'
            bench_df.to_csv(bench_path, index=False)
            print(f"Benchmark results saved to: {bench_path}")

    def wandb_finish(self):
        """Finish training and clean up"""
        if self.use_wandb:
            wandb.finish()
            print("W&B run finished")



