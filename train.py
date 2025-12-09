import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU setup and memory check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_props.name}")
    print(f"Total GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}")

# Auto-detect batch size based on GPU memory
if device.type == 'cuda':
    if gpu_props.total_memory < 8 * 1024**3:  # Less than 8GB
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 32
else:
    BATCH_SIZE = 8  # CPU fallback

print(f"Using batch size: {BATCH_SIZE}")

# Enhanced Configuration
CONFIG = {
    'data_path': '/content/combined_dataset',
    'image_size': 224,
    'num_epochs': 30,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'patience': 7,
    'num_workers': 2,
    'model_save_path': 'best_deepfake_vit_model.pth',
    'checkpoint_dir': '/content/drive/MyDrive/deepfake_checkpoints',
    'gradient_accumulation_steps': 2,
    'max_grad_norm': 1.0,
    'warmup_epochs': 3,
    'freeze_backbone_epochs': 2,
    'use_amp': True,
    'use_focal_loss': True,
}

# Create checkpoint directory
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Check dataset structure first
def check_dataset_structure():
    """Check if the dataset exists and has proper structure"""
    dataset_path = CONFIG['data_path']
    print("📁 Checking dataset structure...")

    required_folders = [
        ('train', 'real'),
        ('train', 'fake'),
        ('val', 'real'),
        ('val', 'fake')
    ]

    all_exists = True
    for split, cls in required_folders:
        folder_path = os.path.join(dataset_path, split, cls)
        if os.path.exists(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ {split}/{cls}: {num_images} images")
            if num_images == 0:
                print(f"   ⚠️  Warning: No images found in {split}/{cls}")
                all_exists = False
        else:
            print(f"❌ {split}/{cls}: Directory not found")
            all_exists = False

    return all_exists

# Check dataset
dataset_ready = check_dataset_structure()

if not dataset_ready:
    print("\n❌ Dataset structure incomplete! Please make sure your dataset has:")
    print("   /content/combined_dataset/train/real/")
    print("   /content/combined_dataset/train/fake/")
    print("   /content/combined_dataset/val/real/")
    print("   /content/combined_dataset/val/fake/")
else:
    print("\n✅ Dataset structure looks good!")

"""## Enhanced Dataset Class with Safety Features"""

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train'):
        """
        Custom Dataset for Deepfake Detection with enhanced safety
        """
        self.root_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.phase = phase

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []

        # Real images (label 1)
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(1)  # Real = 1

        # Fake images (label 0)
        fake_dir = os.path.join(self.root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fake_dir, img_name))
                    self.labels.append(0)  # Fake = 0

        print(f"{phase} dataset: {len(self.image_paths)} images ({sum(self.labels)} real, {len(self.labels)-sum(self.labels)} fake)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image with enhanced error handling
        try:
            image = Image.open(img_path).convert('RGB')
            # Verify image is valid
            image.verify()
            image = Image.open(img_path).convert('RGB')  # Reopen after verify
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a safe fallback image
            image = Image.new('RGB', (224, 224), color='white')
            label = 0  # Default to fake for corrupted images

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

"""## Safe Data Augmentations"""

def safe_jpeg_compression(image, quality_range=(80, 95)):
    """
    Safe JPEG compression simulation
    """
    import io
    try:
        quality = np.random.randint(quality_range[0], quality_range[1])
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    except:
        return image  # Return original if compression fails

def get_transforms(phase='train', image_size=224):
    """
    Get safe data transformations for training and validation
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Mild rotation
            transforms.ColorJitter(
                brightness=0.1,  # Reduced from 0.2
                contrast=0.1,    # Reduced from 0.2
                saturation=0.1,  # Reduced from 0.2
                hue=0.05         # Reduced from 0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Reduced sigma
            transforms.Lambda(lambda x: safe_jpeg_compression(x)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    else:  # validation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

"""## Enhanced Vision Transformer with Numerical Stability"""

class StableDeepfakeViT(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load pre-trained ViT base model
        self.vit = timm.create_model('vit_base_patch16_224',
                                   pretrained=pretrained,
                                   num_classes=0)  # Remove classification head

        # Get feature dimension
        feature_dim = self.vit.num_features

        # Enhanced classification head with better stability
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6),  # Increased epsilon for stability
            nn.Dropout(0.1),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Initialize classifier weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.vit(x)
        # Add numerical stability
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        """Freeze ViT backbone parameters"""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ViT backbone parameters"""
        for param in self.vit.parameters():
            param.requires_grad = True

"""## Enhanced Loss Functions"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(use_focal=True):
    """Get appropriate loss function"""
    if use_focal:
        return FocalLoss(alpha=0.75, gamma=2.0)
    else:
        return nn.CrossEntropyLoss()

"""## Enhanced Model Setup with SAM Optimizer"""

def setup_model(num_classes=2, resume_from_checkpoint=None):
    """Initialize model with enhanced stability features"""
    model = StableDeepfakeViT(num_classes=num_classes)
    model = model.to(device)

    # Loss function
    criterion = get_loss_function(CONFIG['use_focal_loss'])

    # Base optimizer
    base_optimizer = optim.AdamW

    # Enhanced optimizer with gradient clipping built-in
    optimizer = base_optimizer(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999),  # Stable betas
        eps=1e-8  # Increased epsilon for stability
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['num_epochs'],
        steps_per_epoch=1,  # Will be updated in training
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"📅 Resuming from epoch {start_epoch}")

    return model, criterion, optimizer, scheduler, start_epoch

"""## Enhanced Trainer with All Safety Features"""

class EnhancedTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler() if CONFIG['use_amp'] else None

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [],
            'learning_rates': [], 'grad_norms': []
        }

        # Training state
        self.best_auc = 0.0
        self.epochs_without_improvement = 0

    def check_nan_values(self, tensor, name=""):
        """Check for NaN values in tensors"""
        if torch.isnan(tensor).any():
            print(f"⚠️ NaN detected in {name}")
            return True
        return False

    def safe_backward(self, loss, retain_graph=False):
        """Safe backward pass with NaN checking"""
        if self.check_nan_values(loss, "loss"):
            return False

        if self.scaler:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

        return True

    def safe_optimizer_step(self):
        """Safe optimizer step with gradient clipping and NaN protection"""
        # Check for NaN gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None and self.check_nan_values(param.grad, f"grad_{name}"):
                self.optimizer.zero_grad()
                return False

        # Gradient clipping
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['max_grad_norm'])

        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return True

    def train_epoch(self, dataloader, epoch):
        """Enhanced training epoch with safety features"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        skipped_batches = 0

        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision training (optional)
            if CONFIG['use_amp'] and self.scaler:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    # Numerical stability
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                loss = self.criterion(outputs, labels)

            # Skip batch if loss is NaN
            if self.check_nan_values(loss, "batch_loss"):
                skipped_batches += 1
                self.optimizer.zero_grad()
                continue

            # Gradient accumulation
            loss = loss / CONFIG['gradient_accumulation_steps']

            # Backward pass
            if not self.safe_backward(loss):
                skipped_batches += 1
                continue

            # Optimizer step only at accumulation steps
            if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                if not self.safe_optimizer_step():
                    skipped_batches += 1
                    continue

            # Update metrics
            running_loss += loss.item() * CONFIG['gradient_accumulation_steps']
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item() * CONFIG["gradient_accumulation_steps"]:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Skipped': skipped_batches
            })

        # Final optimizer step if there are remaining gradients
        if total > 0 and (len(dataloader) % CONFIG['gradient_accumulation_steps'] != 0):
            self.safe_optimizer_step()

        if skipped_batches > 0:
            print(f"⚠️ Skipped {skipped_batches} batches due to numerical issues")

        epoch_loss = running_loss / (len(dataloader) - skipped_batches) if (len(dataloader) - skipped_batches) > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def validate_epoch(self, dataloader):
        """Enhanced validation epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if CONFIG['use_amp'] and self.scaler:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(images)
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds) * 100

        # Calculate AUC
        if len(np.unique(all_labels)) > 1:
            epoch_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            epoch_auc = 0.5

        return epoch_loss, epoch_acc, epoch_auc, all_preds, all_labels, all_probs

    def save_checkpoint(self, epoch, is_best=False, is_epoch=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'history': self.history,
            'config': CONFIG
        }

        if is_best:
            path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"🏆 Best model saved: {path}")

        if is_epoch:
            path = os.path.join(CONFIG['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
            print(f"💾 Epoch checkpoint saved: {path}")

        # Always save latest
        path = os.path.join(CONFIG['checkpoint_dir'], 'latest_model.pth')
        torch.save(checkpoint, path)

    def train(self, train_loader, val_loader, epochs, start_epoch=0):
        """Enhanced training loop with all safety features"""
        print("🚀 Starting enhanced training with safety features...")

        # Update scheduler steps
        self.scheduler.steps_per_epoch = len(train_loader) // CONFIG['gradient_accumulation_steps']

        for epoch in range(start_epoch, epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Freeze/unfreeze backbone based on schedule
            if epoch < CONFIG['freeze_backbone_epochs']:
                if epoch == 0:
                    print("🔒 Freezing backbone for warmup...")
                    self.model.freeze_backbone()
            else:
                if epoch == CONFIG['freeze_backbone_epochs']:
                    print("🔓 Unfreezing backbone...")
                    self.model.unfreeze_backbone()

            # Enable/disable AMP based on stability
            if epoch < 2 and CONFIG['use_amp']:
                print("⚠️ Disabling AMP for first 2 epochs for stability...")
                current_amp = False
            else:
                current_amp = CONFIG['use_amp']

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)

            # Validation phase
            val_loss, val_acc, val_auc, _, _, _ = self.validate_epoch(val_loader)

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(current_lr)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')
            print(f'Learning Rate: {current_lr:.2e}')

            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f'🎯 New best model with AUC: {val_auc:.4f}')
            else:
                self.epochs_without_improvement += 1

            # Save epoch checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_epoch=True)

            # Save latest checkpoint
            self.save_checkpoint(epoch)

            # Early stopping
            if self.epochs_without_improvement >= CONFIG['patience']:
                print(f'🛑 Early stopping at epoch {epoch+1}')
                break

        print(f'\n✅ Training completed!')
        print(f'🏆 Best validation AUC: {self.best_auc:.4f}')

        return {
            'best_auc': self.best_auc,
            'final_epoch': epoch,
            'history': self.history
        }

"""## Enhanced Training Function with Resume Capability"""

def enhanced_main_training(resume_from_checkpoint=None):
    """Enhanced main training with resume capability"""
    print("🚀 Enhanced Deepfake Detection Training with ViT")
    print("="*60)

    # Check if dataset exists
    train_real_dir = os.path.join(CONFIG['data_path'], 'train', 'real')
    train_fake_dir = os.path.join(CONFIG['data_path'], 'train', 'fake')

    if not os.path.exists(train_real_dir) or not os.path.exists(train_fake_dir):
        print(f"❌ Dataset not found at {CONFIG['data_path']}")
        return None

    print("📁 Setting up data loaders...")

    # Data transforms
    train_transform = get_transforms('train', CONFIG['image_size'])
    val_transform = get_transforms('val', CONFIG['image_size'])

    # Datasets
    train_dataset = DeepfakeDataset(CONFIG['data_path'], train_transform, 'train')
    val_dataset = DeepfakeDataset(CONFIG['data_path'], val_transform, 'val')

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No images found in dataset directories!")
        return None

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")

    # Setup model with resume capability
    print("🔄 Setting up model...")
    model, criterion, optimizer, scheduler, start_epoch = setup_model(
        num_classes=2, 
        resume_from_checkpoint=resume_from_checkpoint
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Model parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # Train model
    print("🎯 Starting enhanced training...")
    trainer = EnhancedTrainer(model, criterion, optimizer, scheduler, device)
    training_info = trainer.train(train_loader, val_loader, CONFIG['num_epochs'], start_epoch)

    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(trainer.history)

    # Final validation evaluation
    print("\n📊 Final validation evaluation...")
    val_loss, val_acc, val_auc, val_preds, val_labels, val_probs = trainer.validate_epoch(val_loader)
    val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
    print_metrics(val_metrics, 'Final Validation')
    plot_confusion_matrix(val_metrics['confusion_matrix'])

    return trainer, training_info

"""## Utility Functions (Keep from previous code)"""

def calculate_metrics(all_labels, all_preds, all_probs):
    """Calculate comprehensive evaluation metrics"""
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    else:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

    return metrics

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC
    axes[1, 0].plot(history['val_auc'], label='Val AUC', color='red', linewidth=2)
    axes[1, 0].set_title('Validation AUC Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rates' in history and history['learning_rates']:
        axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes=['Fake', 'Real']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16}, cbar_kws={"shrink": 0.8})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def print_metrics(metrics, phase='Validation'):
    """Print formatted metrics"""
    print(f"\n{phase} Metrics:")
    print("="*40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

"""## Resume Training Function"""

def resume_training():
    """Auto-resume training if a checkpoint exists, otherwise start fresh."""
    latest_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'latest_model.pth')

    if os.path.exists(latest_checkpoint):
        print(f"🔄 Found latest checkpoint: {latest_checkpoint}")
        print("✅ Auto-resume enabled. Resuming training from last saved state...")
        return latest_checkpoint
    
    print("ℹ️ No previous checkpoint found. Starting fresh training.")
    return None


# Check if we should resume training
resume_checkpoint = resume_training()

# Execute training
if dataset_ready:
    trainer, training_info = enhanced_main_training(resume_from_checkpoint=resume_checkpoint)
else:
    print("\n❌ Cannot start training. Please fix dataset structure first.")
    trainer, training_info = None, None

"""## Enhanced Prediction Function"""

def enhanced_predict_image(image_path, model_path=None):
    """
    Enhanced prediction with safety features
    """
    if model_path is None:
        model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = StableDeepfakeViT(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load and preprocess image
        transform = get_transforms('val', CONFIG['image_size'])
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction with safety
        with torch.no_grad():
            if CONFIG['use_amp']:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = model(input_tensor)
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                output = model(input_tensor)
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

            probs = torch.softmax(output, dim=1)

        # Get results
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
        confidence = max(fake_prob, real_prob)

        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'fake': fake_prob,
                'real': real_prob
            }
        }

        return result

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

"""## List Available Checkpoints"""

def list_checkpoints():
    """List all available checkpoints"""
    print("📁 Available checkpoints:")
    checkpoints = []
    
    if os.path.exists(CONFIG['checkpoint_dir']):
        for file in os.listdir(CONFIG['checkpoint_dir']):
            if file.endswith('.pth'):
                file_path = os.path.join(CONFIG['checkpoint_dir'], file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                checkpoints.append((file, file_size))
        
        for checkpoint, size in sorted(checkpoints):
            print(f"  - {checkpoint} ({size:.1f} MB)")
    
    return checkpoints

# List checkpoints
list_checkpoints()

"""## Enhanced Demo with Multiple Checkpoints"""

def enhanced_demo_predictions():
    """Demo with multiple checkpoint options"""
    print("\n🎪 ENHANCED DEMO: Model Predictions")
    print("="*50)

    checkpoints = list_checkpoints()
    if not checkpoints:
        print("❌ No checkpoints found. Please train a model first.")
        return

    # Use best model by default
    model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    
    # Test on validation images
    val_real_dir = os.path.join(CONFIG['data_path'], 'val', 'real')
    val_fake_dir = os.path.join(CONFIG['data_path'], 'val', 'fake')

    demo_images = []

    if os.path.exists(val_real_dir):
        real_images = [os.path.join(val_real_dir, f) for f in os.listdir(val_real_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if real_images:
            demo_images.extend(real_images[:2])

    if os.path.exists(val_fake_dir):
        fake_images = [os.path.join(val_fake_dir, f) for f in os.listdir(val_fake_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if fake_images:
            demo_images.extend(fake_images[:2])

    if not demo_images:
        print("❌ No test images found.")
        return

    print(f"🔍 Testing on {len(demo_images)} images using best model...\n")

    correct_predictions = 0
    total_predictions = 0

    for img_path in demo_images:
        result = enhanced_predict_image(img_path, model_path)
        if result:
            filename = os.path.basename(img_path)
            true_label = "real" if "real" in img_path else "fake"
            predicted_label = result['prediction'].lower()
            
            is_correct = (true_label == predicted_label)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            status = "✅" if is_correct else "❌"
            
            print(f"📁 {true_label}/{filename}:")
            print(f"   {status} Prediction: {result['prediction']} (True: {true_label.upper()})")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Fake prob: {result['probabilities']['fake']:.4f}")
            print(f"   Real prob: {result['probabilities']['real']:.4f}")
            print()

    if total_predictions > 0:
        accuracy = 100.0 * correct_predictions / total_predictions
        print(f"🎯 Demo Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")

# Run enhanced demo
if trainer is not None:
    enhanced_demo_predictions()

"""## Summary"""

print("\n" + "="*60)
print("✅ ENHANCED DEEPFAKE DETECTION SYSTEM SUMMARY")
print("="*60)

if trainer and training_info:
    print(f"🏆 Best Model Achieved:")
    print(f"   - Best AUC: {training_info['best_auc']:.4f}")
    print(f"   - Final Epoch: {training_info['final_epoch'] + 1}")
    print(f"📁 Checkpoints saved in: {CONFIG['checkpoint_dir']}")
else:
    print("❌ Training was not completed successfully")

print(f"\n🔧 Enhanced Features:")
print(f"   - NaN detection and protection")
print(f"   - Gradient clipping and accumulation")
print(f"   - Progressive backbone unfreezing")
print(f"   - Focal Loss for class imbalance")
print(f"   - Safe data augmentations")
print(f"   - Automatic checkpoint saving")
print(f"   - Training resume capability")

print(f"\n🎯 Next steps:")
print(f"1. Checkpoints are automatically saved to Google Drive")
print(f"2. Use enhanced_predict_image() for safe predictions")
print(f"3. Run resume_training() to continue from last checkpoint")
print(f"4. Monitor training with enhanced visualizations")

print(f"\n💡 To resume training later:")
print(f"   from google.colab import drive")
print(f"   drive.mount('/content/drive')")
print(f"   resume_checkpoint = '/content/drive/MyDrive/deepfake_checkpoints/latest_model.pth'")
print(f"   trainer, info = enhanced_main_training(resume_checkpoint)")