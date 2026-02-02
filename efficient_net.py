# train_efficientnet_folder_structure.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import wandb
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

warnings.filterwarnings('ignore')

class ODIRFolderDataset(Dataset):
    """Dataset class for folder-structured ODIR-2019 dataset"""
    
    def __init__(self, root_dir, split='train', transform=None, image_size=224):
        """
        Args:
            root_dir: Root directory with folder structure
                root_dir/
                ├── train/
                │   ├── Normal/
                │   │   ├── image1.jpg
                │   │   └── ...
                │   ├── Diabetes/
                │   │   ├── image2.jpg
                │   │   └── ...
                │   ├── Glaucoma/
                │   └── ...
                ├── val/
                │   ├── Normal/
                │   ├── Diabetes/
                │   └── ...
                └── test/
                    ├── Normal/
                    ├── Diabetes/
                    └── ...
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
            image_size: Target image size
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Define ODIR-2019 classes (8 diseases)
        self.class_names = [
            'normal', 'diabetes', 'glaucoma', 'cataract',
            'ageing', 'hypertension', 'myopia', 'other'
        ]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Collect all images and their labels
        self.samples = []  # List of (image_path, class_idx)
        
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, split, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping class {class_name}.")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(image_extensions):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} images in {split} split across {len(self.class_names)} classes")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of samples per class"""
        class_counts = {class_name: 0 for class_name in self.class_names}
        for _, class_idx in self.samples:
            class_name = self.class_names[class_idx]
            class_counts[class_name] += 1
        
        print(f"\nClass distribution in {self.split} split:")
        for class_name in self.class_names:
            count = class_counts[class_name]
            if count > 0:
                print(f"  {class_name}: {count} samples ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx, img_path


class EfficientNetTrainer:
    """Trainer class for EfficientNet on ODIR-2019 dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Data transforms
        self.train_transform, self.val_transform = self._create_transforms()
        
        # Datasets
        self._create_datasets()
        
        # Dataloaders
        self._create_dataloaders()
        
        # Loss function (multi-class classification)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.best_val_f1 = 0
        self.best_model_path = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': []
        }
        # Timing
        self.training_times = []
        self.inference_times = []

    def _create_model(self):
        """Create EfficientNet model for multi-class classification"""
        model_name = self.config.get('model_name', 'efficientnet-b0')
        num_classes = self.config.get('num_classes', 8)
        
        print(f"Loading {model_name} with {num_classes} classes...")
        
        # Load pretrained EfficientNet
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        
        # Fine-tuning configuration
        fine_tune = self.config.get('fine_tune', True)
        if fine_tune:
            # Freeze all layers initially
            for param in model.parameters():
                param.requires_grad = False
            
            # Strategy 1: Unfreeze the classifier
            for param in model._fc.parameters():
                param.requires_grad = True
            
            # Strategy 2: Unfreeze last few blocks
            unfreeze_blocks = self.config.get('unfreeze_blocks', 3)
            for block in list(model._blocks)[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            # Strategy 3: Gradually unfreeze (implemented in training loop if needed)
            self.unfreeze_strategy = self.config.get('unfreeze_strategy', 'all_at_once')
        
        return model
    
    def _create_transforms(self):
        """Create data augmentation transforms for training and validation"""
        image_size = self.config.get('image_size', 224)
        
        # Advanced training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 48, image_size + 48)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                  saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
        
        # Validation transforms (no augmentation, just normalization)
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def _create_datasets(self):
        """Create train, validation, and test datasets"""
        data_dir = self.config['data_dir']
        
        print(f"Loading datasets from {data_dir}...")
        
        self.train_dataset = ODIRFolderDataset(
            root_dir=data_dir,
            split='train',
            transform=self.train_transform,
            image_size=self.config.get('image_size', 224)
        )
        
        self.val_dataset = ODIRFolderDataset(
            root_dir=data_dir,
            split='val',
            transform=self.val_transform,
            image_size=self.config.get('image_size', 224)
        )
        
        self.test_dataset = ODIRFolderDataset(
            root_dir=data_dir,
            split='test',
            transform=self.val_transform,
            image_size=self.config.get('image_size', 224)
        )
        
        # Store class names for plotting
        self.class_names = self.train_dataset.class_names
    
    def _create_dataloaders(self):
        """Create data loaders with optional weighted sampling for class imbalance"""
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        use_weighted_sampler = self.config.get('use_weighted_sampler', False)
        
        if use_weighted_sampler and self.train_dataset is not None:
            # Create weighted sampler to handle class imbalance
            train_labels = []
            for _, class_idx in self.train_dataset.samples:
                train_labels.append(class_idx)
            
            class_counts = np.bincount(train_labels)
            print("Class counts for weighted sampler:", class_counts)
            print("Labels are:", train_labels)
            print("Using WeightedRandomSampler to address class imbalance.")
            class_weights = 1. / class_counts
            sample_weights = class_weights[train_labels]
            print("Sample weights are:", sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True  # Drop last incomplete batch
            )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print("\n📊 Dataset Statistics:")
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Val samples:   {len(self.val_dataset)}")
        print(f"   Test samples:  {len(self.test_dataset)}")
        print(f"   Image size:    {self.config.get('image_size', 224)}")
        print(f"   Batch size:    {batch_size}")

    def _create_optimizer(self):
        """Create optimizer with different parameter groups
           Optimizer supports 'adamw', 'adam', 'sgd'
           optimizer changes learning rates for backbone and classifier differently 
        """
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Separate parameters into different groups for different learning rates
        params_dict = {
            'backbone': [],
            'classifier': []
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'fc' in name or '_fc' in name:
                params_dict['classifier'].append(param)
            else:
                params_dict['backbone'].append(param)
        
        # Different learning rates for backbone and classifier
        optimizer_params = [
            {'params': params_dict['backbone'], 'lr': lr * 0.1},  # Lower LR for backbone
            {'params': params_dict['classifier'], 'lr': lr}       # Higher LR for classifier
        ]
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_params, lr=lr, momentum=0.9, 
                                 weight_decay=weight_decay, nesterov=True)
        else:
            optimizer = optim.AdamW(optimizer_params, lr=lr)
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler
           Supported schedulers: 'plateau', 'cosine', 'onecycle'
           Scheduler is optional but recommended for better training
           Scheduler adjusts learning rate during training based on performance or epoch count
           It modifies the optimizer's learning rate according to the chosen strategy
        """
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            )
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50),
                eta_min=1e-6
            )
        elif scheduler_name == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('lr', 0.001),
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.get('epochs', 50)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if available
            if self.config.get('use_amp', False) and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step scheduler if it's OneCycleLR
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1

    
    def validate(self, dataloader=None, split_name='Validation'):
        """Validate the model"""
        if dataloader is None:
            dataloader = self.val_loader
            
        self.model.eval()
        running_loss = 0.0
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
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                #copy tensor from gpu to cpu numpy and append to list
                all_preds.extend(predicted.cpu().numpy()) 
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/len(pbar):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        # Store inference times
        self.inference_times.extend(inference_times)

        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_targets, all_probs, avg_inference_time
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, split='val', save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 10})
        plt.title(f'Confusion Matrix - {split.capitalize()} Set', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add percentage annotations
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_percent = cm / cm_sum.astype(float) * 100
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if cm[i, j] > 0:
                    text = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
                    ax.text(j + 0.5, i + 0.5, text, 
                           ha='center', va='center', 
                           color='red' if i != j else 'black',
                           fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb
        wandb.log({f"confusion_matrix_{split}": wandb.Image(plt)})
        plt.close()
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, pad=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, pad=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('Training and Validation F1 Score', fontsize=14, pad=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Learning rate plot
        if self.history['learning_rates']:
            axes[1, 1].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, pad=10)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.suptitle('Training History', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb
        wandb.log({"training_history": wandb.Image(fig)})
        plt.close()
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'config': self.config,
            'class_names': self.class_names
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        self.class_names = checkpoint['class_names']
        print(f"Model loaded from {path}")
    
    def train(self):
        """Main training loop"""
        epochs = self.config.get('epochs', 50)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Gradually unfreeze layers (if configured)
            if self.config.get('unfreeze_strategy', 'all_at_once') == 'gradual':
                self._gradually_unfreeze(epoch, epochs)
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_targets, _,val_inf_time = self.validate()
            
            # Update scheduler (if not OneCycleLR)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            self.training_times.append(epoch_time)
            
            # Print epoch results
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"  Time - Epoch: {epoch_time:.1f}s, Inference: {val_inf_time:.1f}ms")
            
            # Log to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'learning_rate': current_lr,
            })
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_model_path = os.path.join(wandb.run.dir, 'best_model.pth')
                self.save_model(best_model_path)
                self.best_model_path = best_model_path
                print(f"✅ New best model saved with F1: {val_f1:.4f}")
            
            # Plot confusion matrix every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.plot_confusion_matrix(val_targets, val_preds, 
                                          self.class_names, split='val')
        # Training summary
        print(f"\n🎉 Training completed!")
        print(f"📈 Best validation F1: {self.best_val_f1:.4f}")
        print(f"⏱️  Average epoch time: {np.mean(self.training_times):.1f}s")
        print(f"⚡ Average inference time: {np.mean(self.inference_times)*1000:.1f}ms")

        # Plot training history
        self.plot_training_history(save_path=os.path.join(wandb.run.dir, 'training_history.png'))
    
    def _gradually_unfreeze(self, epoch, total_epochs):
        """Gradually unfreeze layers during training"""
        # Example: Unfreeze one additional block every 10 epochs
        blocks_to_unfreeze = min((epoch // 10) + 1, len(list(self.model._blocks)))
        
        # Freeze all blocks initially
        for block in self.model._blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # Unfreeze the last N blocks
        for block in list(self.model._blocks)[-blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
        
        if epoch % 10 == 0:
            print(f"Unfroze last {blocks_to_unfreeze} blocks")
    
    def test(self):
        """Test the model on test set"""
        print(f"\n{'='*60}")
        print("Testing Model")
        print(f"{'='*60}")
        
        if self.best_model_path:
            self.load_model(self.best_model_path)
        
        test_loss, test_acc, test_f1, test_preds, test_targets, test_probs, test_inf_time = self.validate(
            self.test_loader, split_name='Testing'
        )
        
        # Generate detailed classification report
        report = classification_report(test_targets, test_preds, 
                                      target_names=self.class_names,
                                      output_dict=True)
        
        print(f"\n📊 Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Weighted F1 Score: {test_f1:.4f}")
        print(f"Average Inference Time per Image: {test_inf_time:.2f} ms")
        
        print("\n📋 Detailed Classification Report:")
        print(classification_report(test_targets, test_preds, 
                                   target_names=self.class_names))
        
        # Plot confusion matrix for test set
        self.plot_confusion_matrix(test_targets, test_preds, 
                                  self.class_names, split='test',
                                  save_path=os.path.join(wandb.run.dir, 'test_confusion_matrix.png'))
        
        
        # Log test results to wandb
        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_classification_report': report
        })
        
        # Create a detailed results table
        self._create_results_table(test_targets, test_preds, test_probs)
        
        return test_loss, test_acc, test_f1
    
    def _create_results_table(self, y_true, y_pred, y_probs):
        """Create detailed results table for wandb"""
        # Calculate per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        per_class_acc = []
        
        for i in range(len(self.class_names)):
            mask = np.array(y_true) == i
            if mask.sum() > 0:
                acc = (np.array(y_pred)[mask] == i).sum() / mask.sum()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        # Create table
        results_data = []
        for i, class_name in enumerate(self.class_names):
            results_data.append([
                class_name,
                per_class_acc[i],
                per_class_f1[i],
                f"{(np.array(y_true) == i).sum()} samples"
            ])
        
        # Log table to wandb
        results_table = wandb.Table(
            columns=["Class", "Accuracy", "F1 Score", "Samples"],
            data=results_data
        )
        wandb.log({"per_class_results": results_table})
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        # Load and preprocess image
        print(f"\n🔍 Predicting: {os.path.basename(image_path)}")
        start_time = time.time()
        
        # Load image
        load_start = time.time()
        transform = self.val_transform
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as exc:
            raise ValueError(f"Could not load image from {image_path}") from exc
        load_time = time.time() - load_start
        
        transform_start = time.time()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        transform_time = time.time() - transform_start

        # Predict
        inference_start = time.time()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
        inference_time = time.time() - inference_start

        # Get results
        process_start = time.time()
        class_idx = predicted.item()
        class_name = self.class_names[class_idx]
        confidence = probabilities[0, class_idx].item()

        # Get top-3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        print(f"Prediction completed in {total_time*1000:.2f} ms "
              f"(Load: {load_time*1000:.2f} ms, Transform: {transform_time*1000:.2f} ms, "
              f"Inference: {inference_time*1000:.2f} ms, Process: {process_time*1000:.2f} ms)")        
        return {
            'predicted_class': class_name,
            'confidence': confidence,
            'class_index': class_idx,
            'all_probabilities': probabilities[0].cpu().numpy(),
            'top3_predictions': top3_predictions,
            'total_time_ms': total_time * 1000,
            'inference_time_ms': inference_time * 1000
        }


def create_sample_folder_structure(base_dir='./odir_dataset'):
    """Helper function to create sample folder structure"""
    import os
    
    splits = ['train', 'val', 'test']
    classes = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
               'AMD', 'Hypertension', 'Myopia', 'Other']
    
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(base_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"Created: {class_dir}")
    
    print(f"\nFolder structure created at {base_dir}")
    print("Now you can copy your images to respective class folders.")
    print("\nExample:")
    print("  ./odir_dataset/train/Normal/image1.jpg")
    print("  ./odir_dataset/train/Diabetes/image2.jpg")
    print("  ...")


def main():
    """Main function to run training"""
    # Configuration
    config = {
        # Data
        'data_dir': './odir_dataset',  # Path to your folder-structured dataset
        'num_classes': 8,
        'image_size': 224,
        
        # Model
        'model_name': 'efficientnet-b0',
        'fine_tune': True,
        'unfreeze_blocks': 3,
        'unfreeze_strategy': 'all_at_once',  # or 'gradual'
        
        # Training
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'label_smoothing': 0.1,
        
        # Data loading
        'num_workers': 4,
        'use_weighted_sampler': False,  # Set to True if class imbalance
        'use_amp': True,  # Automatic Mixed Precision
        
        # Others
        'seed': 42
    }
    
    # Initialize wandb
    wandb.init(
        project="ODIR-2019-EfficientNet-Folder",
        config=config,
        name=f"{config['model_name']}_folder_structure",
        tags=["folder-structure", "multi-class", "efficientnet"]
    )
    
    # Create trainer
    trainer = EfficientNetTrainer(config)
    
    # Train
    trainer.train()
    
    # Test
    trainer.test()
    
    # Save final model
    final_model_path = os.path.join(wandb.run.dir, 'final_model.pth')
    trainer.save_model(final_model_path)
    
    wandb.finish()
    print("🎉 Training completed!")


if __name__ == "__main__":
    # Uncomment to create sample folder structure
    # create_sample_folder_structure()
    
    main()