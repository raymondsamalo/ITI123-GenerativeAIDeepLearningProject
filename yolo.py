"""
YOLO Classification Training Script for ODIR-2019
Supports YOLOv11 and YOLOv26 classification models
Includes F1-score calculation and inference timing with W&B integration
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import wandb
import cv2
import joblib
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
    
    def __init__(self, model_version, run_name, 
                 use_wandb=False):
        """
        Initialize trainer
        
        Args:
            model_version: Model version (yolo11n-cls, yolo11s-cls, yolo11m-cls, 
                           yolo11l-cls, yolo11x-cls or yolo26 variants)
            run_name: Project name for saving results
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
        """
        self.model_version = model_version
        self.run_name = run_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.results = None
        self.test_predictions = []
        self.test_labels = []
        self.inference_times = []
        self.use_wandb = use_wandb
        
        # ODIR-2019 class names (8 classes)
        self.class_names = [
            'normal', 'diabetes', 'glaucoma', 'cataract',
            'ageing', 'hypertension', 'myopia', 'other'
        ]
        self.num_classes = len(self.class_names)
        
        print(f"Using device: {self.device}")
        print(f"Model version: {model_version}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        print(f"W&B logging: {use_wandb}")

    
    def create_data_yaml(self, data_dir='./datasets/odir'):
        """
        Create YAML configuration file for dataset
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            yaml_path: Path to created YAML file
        """
        data_dir = Path(data_dir)
        
        data_config = {
            'path': str(data_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        yaml_path = data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created data.yaml at: {yaml_path}")
        return yaml_path
    
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
        
    def save_model(self, save_dir='./saved_models', model_name=None, 
                   save_format='pt', include_optimizer=False):
        """
        Save trained model to disk
        
        Args:
            save_dir: Directory to save the model
            model_name: Name for saved model (uses timestamp if None)
            save_format: Format to save model ('pt', 'onnx', 'torchscript')
            include_optimizer: Whether to include optimizer state (for .pt format)
            
        Returns:
            save_path: Path to saved model
        """
        if self.model is None:
            print("No model to save. Please train or load a model first.")
            return None
        
        # Create save directory
        save_dir = Path(save_dir) / self.run_name
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
            metadata = {
                'model_version': self.model_version,
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'save_time': timestamp,
                'device': self.device
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
            success = self.model.export(format='onnx', imgsz=512, simplify=True)
            
            if success:
                # Find the exported model (Ultralytics creates it in runs/export)
                export_dir = Path(f'./runs/export/{self.model_version}')
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
                export_dir = Path(f'./runs/export/{self.model_version}')
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
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.pkl"
        
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_version = metadata.get('model_version', self.model_version)
            self.class_names = metadata.get('class_names', self.class_names)
            self.num_classes = metadata.get('num_classes', self.num_classes)
            print(f"Loaded metadata from: {metadata_path}")
            print(f"Model version: {self.model_version}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Classes: {self.class_names}")
        
        # Load the model
        self.load_model(model_path)
        
    def train_model(self, data_yaml, epochs=100, imgsz=512, batch_size=32, 
                    patience=50, save_dir='./runs'):
        """
        Train YOLO classification model with 512x512 image size
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            imgsz: Image size (512)
            batch_size: Batch size
            patience: Early stopping patience
            save_dir: Directory to save results
        """
        print(f"\n{'='*50}")
        print(f"Training {self.model_version} on ODIR-2019")
        print(f"Image size: {imgsz}x{imgsz}")
        print(f"{'='*50}")
                
        # Load model
        print(f"Loading model: {self.model_version}")
        self.model = YOLO(f'{self.model_version}.pt')
        
        # Training arguments - using 512 image size
        train_args = {
            'data': str(data_yaml),
            'epochs': epochs,
            'imgsz': imgsz,  # Using 512
            'batch': batch_size,
            'patience': patience,
            'project': save_dir,
            'name': f'{self.model_version}_imgsz{imgsz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'device': self.device,
            'workers': 8 if self.device == 'cuda' else 4,
            'seed': 42,
            'val': True,
            'save': True,
            'plots': True,
            'save_period': -1,
            'cache': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'source': None,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'resume': False,
            'verbose': True,
        }
        
        # Add W&B arguments if enabled
        if self.use_wandb:
            train_args['project'] = save_dir
            print("W&B integration enabled for training")
        
        # Train the model
        print("\nStarting training...")
        start_time = time.time()
        self.results = self.model.train(**train_args)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Log training summary to W&B
        if self.use_wandb:
            metrics = self.results.results_dict if hasattr(self.results, 'results_dict') else {}
            
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
        
        return self.results
    
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
            print(f"  Predicted Class: {pred_class_name} (Index: {pred_class_idx})")
            print(f"  Confidence: {pred_confidence:.4f}")
            print(f"\nTop-{top_k} Predictions:")
            for i, pred in enumerate(top_predictions):
                print(f"  {i+1}. {pred['class']:15} - Confidence: {pred['confidence']:.4f}")
            
        else:
            # Detection model (if applicable)
            print("Warning: Model appears to be a detection model, not classification")
            pred_class_idx = 0
            pred_class_name = self.class_names[0]
            pred_confidence = 0.0
            top_predictions = []
        
        # Display result if requested
        if show_result:
            self._display_prediction_result(
                img_path, 
                pred_class_name, 
                pred_confidence, 
                top_predictions
            )
        
        # Save result if requested
        if save_result:
            save_path = self._save_prediction_result(
                img_path, 
                pred_class_name, 
                pred_confidence, 
                top_predictions
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
            'all_probabilities': pred_probs.tolist() if 'pred_probs' in locals() else []
        }
        
        # Log to W&B if enabled
        if self.use_wandb:
            # Create prediction visualization
            if show_result or save_result:
                # Read image for W&B logging
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Create a figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Show image
                ax1.imshow(img)
                ax1.set_title(f"Input Image: {img_path.name}")
                ax1.axis('off')
                
                # Create bar chart for probabilities
                x_pos = np.arange(len(self.class_names))
                ax2.barh(x_pos, pred_probs, color='skyblue')
                ax2.set_yticks(x_pos)
                ax2.set_yticklabels(self.class_names)
                ax2.set_xlabel('Probability')
                ax2.set_title('Class Probabilities')
                ax2.invert_yaxis()
                
                # Highlight predicted class
                ax2.barh(pred_class_idx, pred_probs[pred_class_idx], color='red')
                
                plt.tight_layout()
                
                # Log to W&B
                wandb.log({
                    "single_prediction": wandb.Image(fig),
                    "predicted_class": pred_class_name,
                    "confidence": pred_confidence,
                    "inference_time": inference_time
                })
                
                plt.close()
        
        return result_dict
    
    def _display_prediction_result(self, img_path, pred_class, confidence, top_predictions):
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
        
        # Create bar chart for top predictions
        if top_predictions:
            classes = [p['class'] for p in top_predictions]
            confidences = [p['confidence'] for p in top_predictions]
            
            colors = ['red' if i == 0 else 'skyblue' for i in range(len(classes))]
            y_pos = np.arange(len(classes))
            
            ax2.barh(y_pos, confidences, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(classes)
            ax2.set_xlabel('Confidence')
            ax2.set_title('Top Predictions')
            ax2.set_xlim([0, 1])
            
            # Add confidence values on bars
            for i, conf in enumerate(confidences):
                ax2.text(conf + 0.01, i, f'{conf:.3f}', 
                        va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def _save_prediction_result(self, img_path, pred_class, confidence, top_predictions):
        """Save prediction result to file"""
        # Create results directory
        results_dir = Path(f'./prediction_results/{self.run_name}')
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
        if top_predictions:
            for i, pred in enumerate(top_predictions[:3]):
                pred_text = f"{i+1}. {pred['class']}: {pred['confidence']:.3f}"
                y_pos = text_y + (i + 1) * 30
                
                # Draw background
                pred_text_size = cv2.getTextSize(pred_text, font, font_scale-0.2, thickness-1)[0]
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
        
        # Add top predictions to CSV
        for i, pred in enumerate(top_predictions[:3]):
            prediction_data[f'top_{i+1}_class'] = [pred['class']]
            prediction_data[f'top_{i+1}_confidence'] = [pred['confidence']]
        
        df = pd.DataFrame(prediction_data)
        
        # Append to existing CSV or create new
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(csv_path, index=False)
        
        return save_path
    
    def evaluate_model(self, data_yaml, split='test', imgsz=512):
        """
        Evaluate model on test/validation set with 512x512 image size
        
        Args:
            data_yaml: Path to data YAML file
            split: Dataset split to evaluate ('val' or 'test')
            imgsz: Image size for evaluation (512)
            
        Returns:
            metrics_dict: Dictionary containing evaluation metrics
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {self.model_version} on {split} set")
        print(f"Image size: {imgsz}x{imgsz}")
        print(f"{'='*50}")
        
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None
        
        # Clear previous results
        self.test_predictions = []
        self.test_labels = []
        self.inference_times = []
        
        # Load dataset path from YAML
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_config['path'])
        split_path = dataset_path / split / 'images'
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_path.glob(ext))
        
        print(f"Found {len(image_files)} images in {split} set")
        
        # Run inference on all images
        correct = 0
        total = 0
        
        for img_path in tqdm(image_files, desc=f"Evaluating on {split}"):
            # Measure inference time
            start_time = time.time()
            
            # Run inference with specified image size
            results = self.model(img_path, imgsz=imgsz, verbose=False)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Get prediction
            if hasattr(results[0], 'probs'):
                pred_probs = results[0].probs.data.cpu().numpy()
                pred_class = np.argmax(pred_probs)
            else:
                pred_class = results[0].boxes.cls[0].int().item() if len(results[0].boxes) > 0 else 0
            
            # Get true label from label file
            label_path = dataset_path / split / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    try:
                        true_class = int(f.read().strip())
                    except:
                        continue
            else:
                # If no label file, skip
                continue
            
            self.test_predictions.append(pred_class)
            self.test_labels.append(true_class)
            
            if pred_class == true_class:
                correct += 1
            total += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        # Calculate F1 scores
        f1_micro = f1_score(self.test_labels, self.test_predictions, average='micro')
        f1_macro = f1_score(self.test_labels, self.test_predictions, average='macro')
        f1_weighted = f1_score(self.test_labels, self.test_predictions, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(self.test_labels, self.test_predictions, average=None)
        
        # Classification report
        class_report = classification_report(
            self.test_labels, 
            self.test_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.test_labels, self.test_predictions)
        
        # Calculate inference statistics
        avg_inference_time = np.mean(self.inference_times)
        std_inference_time = np.std(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Print results
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS for {self.model_version}")
        print(f"Image Size: {imgsz}x{imgsz}")
        print(f"{'='*50}")
        print(f"Total images: {total}")
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
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(cm, split, imgsz)
        
        # Log to W&B
        if self.use_wandb :
            # Create confusion matrix figure
            cm_fig = self.create_confusion_matrix_figure(cm, imgsz)
            
            # Log metrics
            wandb.log({
                f"{split}_accuracy": accuracy,
                f"{split}_f1_micro": f1_micro,
                f"{split}_f1_macro": f1_macro,
                f"{split}_f1_weighted": f1_weighted,
                f"{split}_avg_inference_time": avg_inference_time,
                f"{split}_fps": fps,
                f"{split}_image_size": imgsz,
                f"{split}_confusion_matrix": wandb.Image(cm_fig)
            })
            
            # Log per-class F1 scores
            for i, class_name in enumerate(self.class_names):
                wandb.log({f"{split}_f1_{class_name}": f1_per_class[i]})
            
            # Log class report as table
            class_report_df = pd.DataFrame(class_report).transpose()
            wandb.log({f"{split}_classification_report": wandb.Table(dataframe=class_report_df)})
            
            plt.close('all')
        
        # Return metrics
        metrics_dict = {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.class_names, f1_per_class)),
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'fps': fps,
            'image_size': imgsz,
            'total_images': total,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        return metrics_dict
    
    def plot_confusion_matrix(self, cm, split='test', imgsz=512):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_version} - {split} - {imgsz}x{imgsz}', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        save_path = Path(f'./runs/{self.run_name}/confusion_matrix_{split}_{imgsz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        # Show plot in notebook
        plt.show()
    
    def create_confusion_matrix_figure(self, cm, imgsz=512):
        """Create confusion matrix figure for W&B logging"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        ax.set_title(f'Confusion Matrix - {self.model_version} - {imgsz}x{imgsz}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        return fig
    
    def run_inference_benchmark(self, data_yaml, num_iterations=100, batch_sizes=[1, 4, 8, 16], imgsz=512):
        """
        Run inference timing benchmark with different batch sizes and 512x512 image size
        
        Args:
            data_yaml: Path to data YAML file
            num_iterations: Number of inference iterations per batch size
            batch_sizes: List of batch sizes to test
            imgsz: Image size for benchmarking (512)
            
        Returns:
            benchmark_results: Dictionary with benchmark results
        """
        print(f"\n{'='*50}")
        print(f"Running Inference Benchmark for {self.model_version}")
        print(f"Image Size: {imgsz}x{imgsz}")
        print(f"{'='*50}")
        
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None
        
        benchmark_results = {}
        
        # Load a sample image for benchmarking
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_config['path'])
        test_path = dataset_path / 'test' / 'images'
        image_files = list(test_path.glob('*.jpg'))[:1]
        
        if not image_files:
            print("No test images found!")
            return None
        
        sample_image = str(image_files[0])
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking with batch size: {batch_size}, Image size: {imgsz}")
            
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
            
            print(f"  Average time per batch: {avg_time:.4f} ± {std_time:.4f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Time per image: {avg_time/batch_size:.4f} seconds")
            
            benchmark_results[f'batch_{batch_size}'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': fps,
                'time_per_image': avg_time/batch_size,
                'image_size': imgsz
            }
            
            # Log to W&B
            if self.use_wandb :
                wandb.log({
                    f"benchmark_batch_{batch_size}_avg_time": avg_time,
                    f"benchmark_batch_{batch_size}_fps": fps,
                    f"benchmark_batch_{batch_size}_time_per_image": avg_time/batch_size,
                    f"benchmark_batch_{batch_size}_image_size": imgsz
                })
        
        return benchmark_results
    
    def save_results(self, metrics_dict, benchmark_results=None):
        """Save evaluation results to CSV"""
        # Create results directory
        results_dir = Path(f'./results/{self.run_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main metrics
        metrics_df = pd.DataFrame([{
            'model': self.model_version,
            'timestamp': timestamp,
            'accuracy': metrics_dict.get('accuracy', 0),
            'f1_micro': metrics_dict.get('f1_micro', 0),
            'f1_macro': metrics_dict.get('f1_macro', 0),
            'f1_weighted': metrics_dict.get('f1_weighted', 0),
            'avg_inference_time': metrics_dict.get('avg_inference_time', 0),
            'fps': metrics_dict.get('fps', 0),
            'image_size': metrics_dict.get('image_size', 512),
            'total_images': metrics_dict.get('total_images', 0)
        }])
        
        metrics_path = results_dir / f'metrics_{self.model_version}_{timestamp}.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save per-class F1 scores
        f1_per_class = metrics_dict.get('f1_per_class', {})
        if f1_per_class:
            f1_df = pd.DataFrame(list(f1_per_class.items()), columns=['class', 'f1_score'])
            f1_path = results_dir / f'f1_scores_{self.model_version}_{timestamp}.csv'
            f1_df.to_csv(f1_path, index=False)
            print(f"Per-class F1 scores saved to: {f1_path}")
        
        # Save benchmark results
        if benchmark_results:
            bench_df = pd.DataFrame(benchmark_results).T.reset_index()
            bench_df.columns = ['batch_size'] + list(bench_df.columns[1:])
            bench_path = results_dir / f'benchmark_{self.model_version}_{timestamp}.csv'
            bench_df.to_csv(bench_path, index=False)
            print(f"Benchmark results saved to: {bench_path}")
    
    def finish(self):
        """Finish training and clean up"""
        if self.use_wandb :
            wandb.finish()
            print("W&B run finished")


# Main execution function
def main():
    """Main function to run the training pipeline"""
    # Configuration
    DATA_DIR = './datasets/odir'  # Update this path
    MODELS_TO_TRAIN = ['yolo11n-cls', 'yolo11s-cls']  # Add more models as needed
    EPOCHS = 50
    IMG_SIZE = 512  # Using 512x512 image size
    BATCH_SIZE = 16  # Reduced batch size for 512x512 images (adjust based on GPU memory)
    USE_WANDB = True
    
    # Create data YAML
    trainer = ODIRYOLOTrainer(model_version='yolo11n-cls', use_wandb=USE_WANDB)
    data_yaml = trainer.create_data_yaml(DATA_DIR)
    
    # Train and evaluate each model
    all_results = {}
    
    for model_version in MODELS_TO_TRAIN:
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_version}")
        print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
        print(f"{'='*60}")
        
        # Initialize trainer for this model
        trainer = ODIRYOLOTrainer(model_version=model_version, use_wandb=USE_WANDB)
        
        # Train model
        results = trainer.train_model(
            data_yaml=data_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch_size=BATCH_SIZE,
            patience=20,
            save_dir='./runs'
        )
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate_model(data_yaml, split='val', imgsz=IMG_SIZE)
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_model(data_yaml, split='test', imgsz=IMG_SIZE)
        
        # Run inference benchmark
        benchmark_results = trainer.run_inference_benchmark(data_yaml, imgsz=IMG_SIZE)
        
        # Save the trained model
        trainer.save_model(
            save_dir='./saved_models',
            model_name=f'{model_version}_imgsz{IMG_SIZE}',
            save_format='pt'
        )
        
        # Save results
        if test_metrics:
            trainer.save_results(test_metrics, benchmark_results)
        
        # Example: Predict on a single test image
        print(f"\n{'='*60}")
        print("EXAMPLE: Predicting on a single test image")
        print(f"{'='*60}")
        
        # Find a test image
        test_images = list(Path(DATA_DIR) / 'test' / 'images').glob('*.jpg')
        if test_images:
            test_image = test_images[0]
            print(f"Predicting on test image: {test_image.name}")
            prediction = trainer.predict_single_image(
                test_image,
                show_result=True,
                save_result=True
            )
        
        # Store results
        all_results[model_version] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'benchmark': benchmark_results
        }
        
        # Finish W&B run for this model
        trainer.finish()
    
    # Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'='*60}")
    
    comparison_data = []
    for model_version, results in all_results.items():
        test_metrics = results.get('test_metrics', {})
        comparison_data.append({
            'Model': model_version,
            'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
            'F1-Macro': f"{test_metrics.get('f1_macro', 0):.4f}",
            'FPS': f"{test_metrics.get('fps', 0):.2f}",
            'Inference Time (ms)': f"{test_metrics.get('avg_inference_time', 0)*1000:.2f}",
            'Image Size': f"{IMG_SIZE}x{IMG_SIZE}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = Path(f'./results/model_comparison_imgsz{IMG_SIZE}.csv')
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()