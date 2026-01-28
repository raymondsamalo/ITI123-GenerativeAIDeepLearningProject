# efficientnet_plain_odir.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import wandb
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class ODIRMultiClassDataset(Dataset):
    """Multi-class dataset for ODIR-2019"""
    
    def __init__(self, root_dir, split='train', transform=None, image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # ODIR-2019 classes (correct names)
        self.class_names = [
            'normal',
            'diabetes',
            'glaucoma',
            'cataract',
            'ageing',
            'hypertension',
            'myopia',
            'other'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Load samples
        self.samples = []
        split_dir = os.path.join(root_dir, split)
        
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {split} split")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of samples per class"""
        class_counts = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            class_counts[self.class_names[label]] += 1
        
        print(f"\nClass distribution in {self.split} split:")
        for cls in self.class_names:
            count = class_counts[cls]
            if count > 0:
                print(f"  {cls}: {count} samples ({count/len(self.samples)*100:.1f}%)")
            else:
                print(f"  {cls}: 0 samples (0.0%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


class PlainEfficientNet(nn.Module):
    """Plain EfficientNet for multi-class classification"""
    
    def __init__(self, model_name='efficientnet-b0', num_classes=8, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name)
            print(f"✅ Loaded pretrained EfficientNet-{model_name}")
        else:
            self.model = EfficientNet.from_name(model_name)
            print(f"✅ Created EfficientNet-{model_name} from scratch")
        
        # Get number of features in the classifier
        in_features = self.model._fc.in_features
        
        # Replace classifier head with custom head
        self.model._fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Store class names
        self.class_names = [
            'normal', 'diabetes', 'glaucoma', 'cataract',
            'ageing', 'hypertension', 'myopia', 'other'
        ]
        
        print(f"   Number of classes: {num_classes}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        return self.model(x)


class EfficientNetTrainer:
    """Trainer for plain EfficientNet on ODIR-2019"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set seeds
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.get('seed', 42))
        
        # Create model
        self.model = PlainEfficientNet(
            model_name=config.get('model_name', 'efficientnet-b0'),
            num_classes=config.get('num_classes', 8),
            pretrained=config.get('pretrained', True)
        ).to(self.device)
        
        # Get class names from model
        self.class_names = self.model.class_names
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30)
        )
        
        # Data transforms
        self.train_transform, self.val_transform = self._create_transforms()
        
        # Training tracking
        self.best_f1 = 0
        self.best_model_path = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
        
        # Timing
        self.training_times = []
        self.inference_times = []
        
        print(f"\n✅ Plain EfficientNet Trainer Initialized")
        print(f"   Model: EfficientNet-{config.get('model_name', 'efficientnet-b0')}")
        print(f"   Classes: {config.get('num_classes', 8)}")
        print(f"   Class names: {', '.join(self.class_names)}")
    
    def _create_transforms(self):
        """Create data augmentation transforms"""
        image_size = self.config.get('image_size', 224)
        
        # Training transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self, data_dir):
        """Create train, val, test dataloaders"""
        print(f"\n📁 Loading datasets from: {data_dir}")
        
        # Create datasets
        train_dataset = ODIRMultiClassDataset(
            data_dir, 'train', self.train_transform
        )
        val_dataset = ODIRMultiClassDataset(
            data_dir, 'val', self.val_transform
        )
        test_dataset = ODIRMultiClassDataset(
            data_dir, 'test', self.val_transform
        )
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples:   {len(val_dataset)}")
        print(f"   Test samples:  {len(test_dataset)}")
        print(f"   Image size:    {self.config.get('image_size', 224)}")
        print(f"   Batch size:    {batch_size}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self, dataloader=None, split_name='Validation'):
        """Validate the model"""
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=split_name)
            for images, targets, _ in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{total_loss/len(pbar):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        # Store inference times
        self.inference_times.extend(inference_times)
        
        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_targets, all_probs, avg_inference_time
    
    def train(self, epochs=None):
        """Main training loop"""
        if epochs is None:
            epochs = self.config.get('epochs', 30)
        
        print(f"\n🚀 Starting EfficientNet training for {epochs} epochs...")
        print(f"🎯 Classes: {', '.join(self.class_names)}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_targets, _, val_inf_time = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            self.training_times.append(epoch_time)
            
            # Print results
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Time - Epoch: {epoch_time:.1f}s, Inference: {val_inf_time:.1f}ms")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time,
                    'inference_time': val_inf_time
                })
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.best_model_path = f"efficientnet_best_epoch{epoch+1}_f1_{val_f1:.4f}_{timestamp}.pth"
                self.save_model(self.best_model_path)
                print(f"✅ New best model saved! F1: {val_f1:.4f}")
            
            # Plot confusion matrix every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_confusion_matrix(val_targets, val_preds, split='val')
        
        # Training summary
        print(f"\n🎉 Training completed!")
        print(f"📈 Best validation F1: {self.best_f1:.4f}")
        print(f"⏱️  Average epoch time: {np.mean(self.training_times):.1f}s")
        print(f"⚡ Average inference time: {np.mean(self.inference_times)*1000:.1f}ms")
        
        # Plot training history
        self.plot_training_history()
    
    def plot_confusion_matrix(self, y_true, y_pred, split='val'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'EfficientNet Confusion Matrix - {split.capitalize()} Set\nODIR-2019 Dataset')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        save_path = f'efficientnet_cm_{split}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f"confusion_matrix_{split}": wandb.Image(plt)})
        
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.suptitle('EfficientNet Training History - ODIR-2019', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        save_path = 'efficientnet_training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({"training_history": wandb.Image(fig)})
        
        plt.close()
        print(f"📈 Training history saved to {save_path}")
    
    def test(self):
        """Test the model on test set"""
        print(f"\n{'='*60}")
        print("🧪 Testing EfficientNet Model")
        print(f"{'='*60}")
        
        # Load best model if available
        if self.best_model_path and os.path.exists(self.best_model_path):
            print(f"📂 Loading best model: {self.best_model_path}")
            self.load_model(self.best_model_path)
        
        # Test
        test_loss, test_acc, test_f1, test_preds, test_targets, test_probs, test_inf_time = self.validate(
            self.test_loader, split_name='Testing'
        )
        
        # Detailed classification report
        report = classification_report(
            test_targets, test_preds,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        
        print(f"\n📊 EfficientNet Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test F1 Score (weighted): {test_f1:.4f}")
        print(f"  Average Inference Time: {test_inf_time:.1f}ms")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(test_targets, test_preds, target_names=self.class_names))
        
        # Plot confusion matrix for test set
        self.plot_confusion_matrix(test_targets, test_preds, split='test')
        
        # Calculate per-class metrics
        print(f"\n🎯 Per-class Performance:")
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(test_targets) == i
            if class_mask.sum() > 0:
                accuracy = (np.array(test_preds)[class_mask] == i).sum() / class_mask.sum()
                precision = report[class_name]['precision'] if class_name in report else 0
                recall = report[class_name]['recall'] if class_name in report else 0
                f1 = report[class_name]['f1-score'] if class_name in report else 0
                print(f"  {class_name}: Acc={accuracy:.2%}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_inference_time': test_inf_time,
                'test_classification_report': report_df
            })
        
        # Save detailed results
        self._save_test_results(test_targets, test_preds, test_probs)
        
        return test_loss, test_acc, test_f1
    
    def _save_test_results(self, y_true, y_pred, y_probs):
        """Save detailed test results to CSV"""
        results = []
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            results.append({
                'true_class': self.class_names[true],
                'predicted_class': self.class_names[pred],
                'correct': true == pred,
                'confidence': y_probs[i][pred] if i < len(y_probs) else 0
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('efficientnet_test_results.csv', index=False)
        print(f"📄 Detailed test results saved to efficientnet_test_results.csv")
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'history': self.history,
            'config': self.config,
            'class_names': self.class_names,
            'model_architecture': 'Plain_EfficientNet'
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_f1 = checkpoint.get('best_f1', 0)
        self.history = checkpoint.get('history', self.history)
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        print(f"📂 Model loaded from {path}")
        print(f"📊 Previously achieved F1: {self.best_f1:.4f}")


class EfficientNetDemo:
    """Demo for loading and testing plain EfficientNet"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model = None
        self.class_names = None
        self.transform = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        print(f"\n🔧 Loading EfficientNet from: {model_path}")
        start_time = time.time()
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get config
            config = checkpoint.get('config', {})
            
            # Create model
            self.model = PlainEfficientNet(
                model_name=config.get('model_name', 'efficientnet-b0'),
                num_classes=config.get('num_classes', 8),
                pretrained=False  # Don't load pretrained weights
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Get class names from checkpoint
            self.class_names = checkpoint.get('class_names', [
                'normal',
                'diabetes',
                'glaucoma',
                'cataract',
                'ageing',
                'hypertension',
                'myopia',
                'other'
            ])
            
            # Create transforms
            image_size = config.get('image_size', 224)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            load_time = time.time() - start_time
            print(f"✅ EfficientNet loaded in {load_time:.3f} seconds")
            print(f"📊 Best F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
            print(f"🎯 Classes: {', '.join(self.class_names)}")
            print(f"🔧 Model: EfficientNet-{config.get('model_name', 'efficientnet-b0')}")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def predict_single_image(self, image_path, top_k=3):
        """Predict class for a single image with timing"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"\n🔍 Predicting: {os.path.basename(image_path)}")
        start_time = time.time()
        
        # Load image
        load_start = time.time()
        try:
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        load_time = time.time() - load_start
        
        # Transform
        transform_start = time.time()
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        transform_time = time.time() - transform_start
        
        # Predict
        inference_start = time.time()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        inference_time = time.time() - inference_start
        
        # Process results
        process_start = time.time()
        probs, indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            class_idx = indices[0, i].item()
            class_name = self.class_names[class_idx]
            confidence = probs[0, i].item() * 100
            predictions.append((class_name, confidence))
        
        # Get predicted class
        predicted_class, confidence = predictions[0]
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        
        # Print results
        print(f"\n🎯 Prediction: {predicted_class} ({confidence:.1f}%)")
        print(f"\n🏆 Top {top_k} Predictions:")
        for i, (cls, conf) in enumerate(predictions, 1):
            print(f"   {i}. {cls}: {conf:.1f}%")
        
        print(f"\n⏱️ Timing (ms):")
        print(f"   Total: {total_time*1000:.1f}")
        print(f"   Loading: {load_time*1000:.1f}")
        print(f"   Transform: {transform_time*1000:.1f}")
        print(f"   Inference: {inference_time*1000:.1f}")
        print(f"   Processing: {process_time*1000:.1f}")
        
        # Display image with prediction
        self._display_prediction(original_image, predicted_class, confidence, predictions)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': predictions,
            'all_probabilities': probabilities[0].cpu().numpy(),
            'timing_ms': {
                'total': total_time * 1000,
                'loading': load_time * 1000,
                'transform': transform_time * 1000,
                'inference': inference_time * 1000,
                'processing': process_time * 1000
            }
        }
    
    def _display_prediction(self, image, predicted_class, confidence, top_predictions):
        """Display image with prediction results"""
        plt.figure(figsize=(12, 5))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"EfficientNet\n{predicted_class} ({confidence:.1f}%)", 
                 fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Display confidence bars
        plt.subplot(1, 2, 2)
        classes = [cls for cls, _ in top_predictions]
        confidences = [conf for _, conf in top_predictions]
        
        colors = ['green' if i == 0 else 'blue' for i in range(len(classes))]
        bars = plt.barh(range(len(classes)), confidences, color=colors)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Confidence (%)')
        plt.title('Top Predictions')
        plt.xlim(0, 100)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{conf:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def benchmark(self, image_path, num_runs=10):
        """Benchmark inference speed"""
        print(f"\n🏃‍♂️ Benchmarking EfficientNet ({num_runs} runs)...")
        
        times = []
        for i in range(num_runs):
            start = time.time()
            _ = self.predict_single_image(image_path)
            times.append(time.time() - start)
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average: {np.mean(times)*1000:.1f}ms")
        print(f"   Min: {np.min(times)*1000:.1f}ms")
        print(f"   Max: {np.max(times)*1000:.1f}ms")
        print(f"   Throughput: {1/np.mean(times):.1f} images/sec")
        
        return times


# Main function with WandB
def main():
    """Main training function"""
    # Configuration
    config = {
        # Model
        'model_name': 'efficientnet-b0',  # Options: b0, b1, b2, b3, b4, b5, b6, b7
        'num_classes': 8,
        'pretrained': True,
        
        # Training
        'epochs': 30,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        
        # Data
        'data_dir': './odir_dataset',
        'image_size': 224,
        'num_workers': 4,
        
        # Misc
        'seed': 42
    }
    
    # Initialize WandB
    wandb.init(
        project="ODIR-2019-EfficientNet",
        config=config,
        name=f"EfficientNet-{config['model_name']}",
        tags=["odir-2019", "eye-disease", "classification", "efficientnet"]
    )
    
    # Create trainer
    trainer = EfficientNetTrainer(config)
    
    # Create dataloaders
    trainer.create_dataloaders(config['data_dir'])
    
    # Train
    trainer.train()
    
    # Test
    trainer.test()
    
    # Save final model
    final_model_path = 'efficientnet_final_model.pth'
    trainer.save_model(final_model_path)
    
    wandb.finish()
    print(f"\n🎉 EfficientNet training completed!")
    print(f"💾 Final model saved to: {final_model_path}")


# Quick usage examples
if __name__ == "__main__":
    # Install requirements:
    # pip install efficientnet-pytorch wandb scikit-learn seaborn
    
    # Option 1: Full training
    # main()
    
    # Option 2: Quick demo
    demo = EfficientNetDemo()
    
    # Load a trained model
    # demo.load_model('efficientnet_best_model.pth')
    
    # Make prediction
    # result = demo.predict_single_image('path/to/eye_image.jpg')
    
    # Option 3: Compare different EfficientNet variants
    def compare_models():
        """Compare different EfficientNet variants"""
        variants = ['b0', 'b1', 'b2', 'b3', 'b4']
        
        results = []
        for variant in variants:
            print(f"\n{'='*50}")
            print(f"Testing EfficientNet-{variant}")
            print(f"{'='*50}")
            
            # Create model
            model = PlainEfficientNet(
                model_name=f'efficientnet-{variant}',
                num_classes=8,
                pretrained=True
            )
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {params:,}")
            
            # Test inference time (simplified)
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Warmup
            for _ in range(3):
                _ = model(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            print(f"Average inference: {avg_time:.1f}ms")
            
            results.append({
                'variant': variant,
                'parameters': params,
                'inference_ms': avg_time
            })
        
        # Print comparison table
        print(f"\n{'='*50}")
        print("EfficientNet Variants Comparison")
        print(f"{'='*50}")
        for r in results:
            print(f"EfficientNet-{r['variant']}: {r['parameters']:,} params, {r['inference_ms']:.1f}ms")
    
    # compare_models()

"""
# 1. Training
config = {
    'model_name': 'efficientnet-b0',  # Try b0, b1, b2, b3, b4
    'num_classes': 8,
    'data_dir': './odir_dataset',
    'epochs': 30,
    'batch_size': 32,
    'lr': 1e-3
}

trainer = EfficientNetTrainer(config)
trainer.create_dataloaders(config['data_dir'])
trainer.train()
trainer.test()

# 2. Demo with saved model
demo = EfficientNetDemo()
demo.load_model('efficientnet_best_model.pth')
result = demo.predict_single_image('test_eye_image.jpg')

# 3. Benchmark
demo.benchmark('test_eye_image.jpg', num_runs=10)
"""