import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import timm

# Suppress warnings
warnings.filterwarnings('ignore')

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, balance_dataset=True):
        """
        Dataset for real and fake face images
        Args:
            real_dir: Directory containing real face images
            fake_dir: Directory containing fake face images
            transform: Image transformations
            balance_dataset: Whether to balance real and fake samples
        """
        self.real_images = []
        self.fake_images = []
        self.labels = []
        self.transform = transform
        
        # Load real images
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.real_images.append(os.path.join(real_dir, img_file))
        else:
            print(f"⚠️ Warning: Real directory not found: {real_dir}")
        
        # Load fake images
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.fake_images.append(os.path.join(fake_dir, img_file))
        else:
            print(f"⚠️ Warning: Fake directory not found: {fake_dir}")
        
        # Balance dataset if requested
        if balance_dataset and self.real_images and self.fake_images:
            min_count = min(len(self.real_images), len(self.fake_images))
            if len(self.real_images) > min_count:
                self.real_images = np.random.choice(self.real_images, min_count, replace=False).tolist()
            if len(self.fake_images) > min_count:
                self.fake_images = np.random.choice(self.fake_images, min_count, replace=False).tolist()
        
        # Combine images and create labels
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0=real, 1=fake
        
        if len(self.all_images) == 0:
            print(f"❌ No images found in {real_dir} or {fake_dir}")
        else:
            print(f"✅ Loaded {len(self.real_images)} real images and {len(self.fake_images)} fake images")
            print(f"📊 Total: {len(self.all_images)} images")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 299, 299)
            return placeholder, torch.tensor(label, dtype=torch.long)

class XceptionDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionDeepfakeModel, self).__init__()
        
        # Load Xception backbone
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        # Enhanced classifier for deepfake detection
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class XceptionFineTuneTrainer:
    def __init__(self, device=None, pretrained_path=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = XceptionDeepfakeModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Load pre-trained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading pre-trained Xception model from: {pretrained_path}")
            self.load_model(pretrained_path)
        else:
            print("🆕 Training from scratch (no pre-trained weights found)")
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': []
        }
    
    def load_model(self, checkpoint_path):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                model_state_dict = checkpoint
            
            # Load model state dict
            self.model.load_state_dict(model_state_dict, strict=False)
            
            print(f"✅ Successfully loaded pre-trained Xception model")
            
            # Print previous performance if available
            if 'val_acc' in checkpoint:
                print(f"📊 Previous validation accuracy: {checkpoint['val_acc']:.4f}")
            if 'val_f1' in checkpoint:
                print(f"📊 Previous validation F1: {checkpoint['val_f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🆕 Continuing with randomly initialized weights")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint for auto-resume"""
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading training checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state
                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Load training history
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
                
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_f1 = checkpoint.get('val_f1', 0.0)
                
                print(f"📅 Resuming from epoch {start_epoch}")
                print(f"📊 Previous best F1: {best_val_f1:.4f}")
                
                return start_epoch, best_val_f1
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print("🔄 Starting from epoch 0")
                return 0, 0.0
        else:
            print("🚀 No checkpoint found. Starting from scratch.")
            return 0, 0.0
    
    def get_transforms(self, augment=True):
        """Define data transformations for Xception"""
        input_size = 299
        resize_size = 331
        
        if augment:
            # Training transforms with augmentation
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Xception normalization
            ])
        else:
            # Validation transforms without augmentation
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        return transform
    
    def setup_optimizer(self, learning_rate=1e-4, weight_decay=1e-4):
        """Setup optimizer with differential learning rates"""
        
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
                print(f"📈 Classifier layer: {name}")
            else:
                backbone_params.append(param)
        
        # Different learning rates for backbone and classifier
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate / 10, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': learning_rate, 'weight_decay': weight_decay}
        ])
        
        print(f"🎯 Optimizer Configuration:")
        print(f"   - Backbone LR: {learning_rate / 10:.1e}")
        print(f"   - Classifier LR: {learning_rate:.1e}")
        print(f"   - Weight Decay: {weight_decay:.1e}")
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=learning_rate * 0.001
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, accuracy, f1
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, accuracy, precision, recall, f1, all_probabilities, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=15, save_dir='xception_finetuned_model', patience=7):
        """Fine-tuning loop with auto-resume"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Define checkpoint paths
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        best_model_path = os.path.join(save_dir, 'best_xception_finetuned.pth')
        final_model_path = os.path.join(save_dir, 'final_xception_model.pth')
        
        # AUTO-RESUME: Load checkpoint if exists
        start_epoch, best_val_f1 = self.load_checkpoint(checkpoint_path)
        
        patience_counter = 0
        
        print(f"🚀 Starting Xception fine-tuning for {epochs} epochs on {self.device}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"🤖 Model: Xception with enhanced classifier")
        print(f"🔄 Auto-resume: {'ENABLED' if start_epoch > 0 else 'DISABLED'}")
        
        for epoch in range(start_epoch, epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1, val_probs, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            self.history['learning_rates'].append(current_lr)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print epoch results
            print(f"✅ Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"🎯 Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"📈 Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            print(f"📉 Learning Rate: {current_lr:.2e}")
            
            # AUTO-RESUME: Save checkpoint after every epoch
            self.save_checkpoint(checkpoint_path, epoch, val_f1)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model(best_model_path, epoch, val_f1)
                print(f"💾 New best model saved with F1: {val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            # Save checkpoint every 5 epochs (additional backup)
            if (epoch + 1) % 5 == 0:
                backup_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(backup_path, epoch, val_f1)
                print(f"💾 Backup checkpoint saved: {backup_path}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_model(final_model_path, epochs-1, val_f1)
        print(f"\n🎉 Xception fine-tuning completed! Best validation F1: {best_val_f1:.4f}")
        
        # Clean up: Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"🧹 Cleanup: Removed checkpoint file {checkpoint_path}")
        
        return self.history
    
    def save_checkpoint(self, path, epoch, val_f1):
        """Save training checkpoint for auto-resume"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(checkpoint, path)
    
    def save_model(self, path, epoch, val_f1):
        """Save model checkpoint (best/final model)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0.0,
            'val_f1': val_f1,
            'history': self.history,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(checkpoint, path)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Xception Fine-tuning Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Xception Fine-tuning Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', linewidth=2)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', linewidth=2)
        axes[1, 0].set_title('Xception Fine-tuning F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate', linewidth=2, color='purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader, save_path=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Class-wise metrics
        class_report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n📊 Xception Fine-tuned Model Evaluation Results:")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print(f"⭐ F1 Score: {f1:.4f}")
        print(f"\n📋 Classification Report:")
        print(class_report)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Xception Fine-tuned Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probabilities,
            'true_labels': all_labels
        }

def create_data_loaders(base_dir, batch_size=16, balance_dataset=True, val_split=0.2):
    """
    Create data loaders from dataset directory structure
    Expected directory structure:
    base_dir/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
    """
    
    # Normalize path
    base_dir = os.path.normpath(base_dir)
    
    # Define paths
    train_real_dir = os.path.join(base_dir, 'train', 'real')
    train_fake_dir = os.path.join(base_dir, 'train', 'fake')
    val_real_dir = os.path.join(base_dir, 'val', 'real')
    val_fake_dir = os.path.join(base_dir, 'val', 'fake')
    
    print(f"🔍 Checking dataset structure in: {base_dir}")
    
    # Check if directories exist
    directories = {
        'train_real': train_real_dir,
        'train_fake': train_fake_dir,
        'val_real': val_real_dir,
        'val_fake': val_fake_dir
    }
    
    missing_dirs = []
    for name, dir_path in directories.items():
        if not os.path.exists(dir_path):
            missing_dirs.append((name, dir_path))
            print(f"❌ Missing directory: {dir_path}")
        else:
            # Count images in directory
            image_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ {name}: {image_count} images")
    
    if missing_dirs:
        print(f"\n⚠️  Missing {len(missing_dirs)} directories:")
        for name, path in missing_dirs:
            print(f"   - {name}: {path}")
        return None, None
    
    # Get transforms
    trainer = XceptionFineTuneTrainer()
    train_transform = trainer.get_transforms(augment=True)
    val_transform = trainer.get_transforms(augment=False)
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = DeepfakeDataset(train_real_dir, train_fake_dir, transform=train_transform, balance_dataset=balance_dataset)
    val_dataset = DeepfakeDataset(val_real_dir, val_fake_dir, transform=val_transform, balance_dataset=balance_dataset)
    
    # Check if datasets have images
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ One or more datasets are empty!")
        return None, None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Input size: 299x299 (Xception)")
    
    return train_loader, val_loader

def fine_tune_xception():
    """Main function for Xception fine-tuning with auto-resume"""
    
    # Configuration
    config = {
        'dataset_base_dir': 'combined_dataset',  # UPDATE THIS PATH
        'pretrained_model_path': 'DEEPFAKE MODELS/xception_deepfake_model/best_face_model.pth',  # Your pre-trained Xception model
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'save_dir': 'xception_finetuned_model',
        'patience': 7,
        'balance_dataset': True
    }
    
    print("🚀 Xception Fine-tuning with Enhanced Classifier & AUTO-RESUME")
    print("=" * 60)
    print("🎯 Features:")
    print("   ✅ Auto-resume from last checkpoint")
    print("   ✅ Differential learning rates")
    print("   ✅ Automatic checkpoint saving every epoch")
    print("   ✅ Early stopping with patience")
    print("   ✅ Cleanup after successful completion")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(config['dataset_base_dir']):
        print(f"❌ Dataset directory not found: {config['dataset_base_dir']}")
        print("📁 Please update the 'dataset_base_dir' in the config")
        return
    
    # Check if pre-trained model exists
    if not os.path.exists(config['pretrained_model_path']):
        print(f"⚠️ Pre-trained model not found: {config['pretrained_model_path']}")
        print("🔄 Training from ImageNet pre-trained weights")
        config['pretrained_model_path'] = None
    
    # Prepare data loaders
    print(f"📦 Loading datasets from: {config['dataset_base_dir']}")
    train_loader, val_loader = create_data_loaders(
        config['dataset_base_dir'],
        batch_size=config['batch_size'],
        balance_dataset=config['balance_dataset']
    )
    
    if train_loader is None:
        print("❌ Failed to create data loaders. Please check your dataset structure.")
        return
    
    # Initialize trainer with pre-trained model
    print("🤖 Initializing Xception model for fine-tuning...")
    trainer = XceptionFineTuneTrainer(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_path=config['pretrained_model_path']
    )
    
    # Setup optimizer
    trainer.setup_optimizer(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
    
    # Start fine-tuning with auto-resume
    print("🎯 Starting Xception fine-tuning with auto-resume...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config['epochs'],
        save_dir=config['save_dir'],
        patience=config['patience']
    )
    
    # Plot training history
    print("📈 Plotting fine-tuning history...")
    trainer.plot_training_history(os.path.join(config['save_dir'], 'xception_fine_tuning_history.png'))
    
    # Evaluate on validation set
    print("🧪 Evaluating fine-tuned Xception model...")
    evaluation_results = trainer.evaluate_model(
        val_loader,
        save_path=os.path.join(config['save_dir'], 'xception_confusion_matrix.png')
    )
    
    # Save configuration and results
    results = {
        'config': config,
        'final_metrics': {
            'val_accuracy': float(evaluation_results['accuracy']),
            'val_precision': float(evaluation_results['precision']),
            'val_recall': float(evaluation_results['recall']),
            'val_f1': float(evaluation_results['f1_score'])
        },
        'dataset_statistics': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset)
        },
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': 'Xception',
            'pretrained_model': config['pretrained_model_path']
        },
        'training_history': history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(config['save_dir'], 'xception_fine_tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Xception Fine-tuning Completed!")
    print(f"💾 Fine-tuned model saved in: {config['save_dir']}")
    print(f"📊 Final Validation Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"📊 Final Validation F1 Score: {evaluation_results['f1_score']:.4f}")

if __name__ == "__main__":
    # Run Xception fine-tuning with auto-resume
    fine_tune_xception()