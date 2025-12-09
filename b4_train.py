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
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64
else:
    BATCH_SIZE = 16

print(f"Using batch size: {BATCH_SIZE}")

# Enhanced Configuration - Now supports custom dataset paths
CONFIG = {
    'data_path': 'final_image_dataset',  # Changed to local path for VS Code
    'image_size': 380,
    'num_epochs': 30,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'patience': 7,
    'num_workers': 0 if device.type == 'cuda' else 2,  # Optimized for VS Code
    'model_save_path': 'best_deepfake_efficientnet_model.pth',
    'checkpoint_dir': './checkpoints',  # Local checkpoint directory
    'gradient_accumulation_steps': 2,
    'max_grad_norm': 1.0,
    'warmup_epochs': 3,
    'freeze_backbone_epochs': 2,
    'use_amp': True if device.type == 'cuda' else False,  # Auto AMP based on device
    'use_focal_loss': True,
    'test_split': True,  # Enable test split
}

# Create checkpoint directory
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Dataset structure checker with flexible support
def check_dataset_structure(data_path=None):
    """Check if the dataset exists and has proper structure"""
    if data_path is None:
        data_path = CONFIG['data_path']
    
    print("📁 Checking dataset structure...")
    print(f"Dataset path: {data_path}")

    # Support multiple dataset structures
    possible_structures = [
        # Structure 1: train/val/test with real/fake subdirectories
        [
            ('train', 'real'),
            ('train', 'fake'),
            ('val', 'real'),
            ('val', 'fake'),
            ('test', 'real'),
            ('test', 'fake')
        ],
        # Structure 2: Only train/val with real/fake
        [
            ('train', 'real'),
            ('train', 'fake'),
            ('val', 'real'),
            ('val', 'fake')
        ],
        # Structure 3: Single directory with subdirectories
        [
            ('real',),
            ('fake',)
        ]
    ]

    all_exists = False
    found_structure = None
    
    for structure in possible_structures:
        structure_exists = True
        total_images = 0
        
        for folder_parts in structure:
            folder_path = os.path.join(data_path, *folder_parts)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                num_images = len(images)
                total_images += num_images
                print(f"✅ {'/'.join(folder_parts)}: {num_images} images")
                if num_images == 0:
                    print(f"   ⚠️  Warning: No images found in {'/'.join(folder_parts)}")
            else:
                structure_exists = False
                break
        
        if structure_exists and total_images > 0:
            all_exists = True
            found_structure = structure
            print(f"🎯 Found dataset structure: {structure}")
            break

    if not all_exists:
        print("\n❌ Dataset structure not recognized! Please use one of these structures:")
        print("Structure 1 (Recommended):")
        print("   dataset/train/real/")
        print("   dataset/train/fake/")
        print("   dataset/val/real/")
        print("   dataset/val/fake/")
        print("   dataset/test/real/")
        print("   dataset/test/fake/")
        print("\nStructure 2:")
        print("   dataset/train/real/")
        print("   dataset/train/fake/")
        print("   dataset/val/real/")
        print("   dataset/val/fake/")
        print("\nStructure 3:")
        print("   dataset/real/")
        print("   dataset/fake/")

    return all_exists, found_structure

# Check dataset
dataset_ready, dataset_structure = check_dataset_structure()

"""## Enhanced Dataset Class with Flexible Structure Support"""

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train', structure_type='standard'):
        """
        Custom Dataset for Deepfake Detection with flexible structure support
        """
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.structure_type = structure_type

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []

        self._load_images()
        
        print(f"{phase} dataset: {len(self.image_paths)} images ({sum(self.labels)} real, {len(self.labels)-sum(self.labels)} fake)")

    def _load_images(self):
        """Load images based on dataset structure"""
        if self.structure_type == 'standard':
            # Standard structure: root/phase/class/
            real_dir = os.path.join(self.root_dir, self.phase, 'real')
            fake_dir = os.path.join(self.root_dir, self.phase, 'fake')
            
            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(real_dir, img_name))
                        self.labels.append(1)  # Real = 1

            if os.path.exists(fake_dir):
                for img_name in os.listdir(fake_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(fake_dir, img_name))
                        self.labels.append(0)  # Fake = 0
                        
        elif self.structure_type == 'simple':
            # Simple structure: root/class/
            if self.phase == 'train':
                real_dir = os.path.join(self.root_dir, 'real')
                fake_dir = os.path.join(self.root_dir, 'fake')
                
                # For simple structure, we'll split the data later
                all_real = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_fake = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Simple split (you can modify this)
                split_idx_real = int(0.8 * len(all_real))
                split_idx_fake = int(0.8 * len(all_fake))
                
                if self.phase == 'train':
                    self.image_paths = all_real[:split_idx_real] + all_fake[:split_idx_fake]
                    self.labels = [1] * len(all_real[:split_idx_real]) + [0] * len(all_fake[:split_idx_fake])
                elif self.phase == 'val':
                    self.image_paths = all_real[split_idx_real:split_idx_real + len(all_real)//10] + all_fake[split_idx_fake:split_idx_fake + len(all_fake)//10]
                    self.labels = [1] * len(all_real[split_idx_real:split_idx_real + len(all_real)//10]) + [0] * len(all_fake[split_idx_fake:split_idx_fake + len(all_fake)//10])
                else:  # test
                    self.image_paths = all_real[split_idx_real + len(all_real)//10:] + all_fake[split_idx_fake + len(all_fake)//10:]
                    self.labels = [1] * len(all_real[split_idx_real + len(all_real)//10:]) + [0] * len(all_fake[split_idx_fake + len(all_fake)//10:])

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
            image = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']), color='white')
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

def get_transforms(phase='train', image_size=380):
    """
    Get safe data transformations for training and validation
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.Lambda(lambda x: safe_jpeg_compression(x)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    else:  # validation and test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

"""## Enhanced EfficientNet-B4 with Numerical Stability"""

class StableDeepfakeEfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load pre-trained EfficientNet-B4 model
        self.backbone = timm.create_model('efficientnet_b4',
                                        pretrained=pretrained,
                                        num_classes=0,
                                        drop_rate=0.2,
                                        drop_path_rate=0.2)

        # Get feature dimension
        feature_dim = self.backbone.num_features

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
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
        features = self.backbone(x)
        # Add numerical stability
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        output = self.classifier(features)
        return output

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
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

"""## Enhanced Model Setup"""

def setup_model(num_classes=2, resume_from_checkpoint=None):
    """Initialize model with enhanced stability features"""
    model = StableDeepfakeEfficientNet(num_classes=num_classes)
    model = model.to(device)

    # Loss function
    criterion = get_loss_function(CONFIG['use_focal_loss'])

    # Enhanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['num_epochs'],
        steps_per_epoch=1,
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
            'test_loss': [], 'test_acc': [], 'test_auc': [],
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

            # Mixed precision training
            if CONFIG['use_amp'] and self.scaler:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
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

    def evaluate_epoch(self, dataloader, phase='val'):
        """Enhanced evaluation epoch for val/test"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f'{phase.capitalize()}'):
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

    def train(self, train_loader, val_loader, test_loader=None, epochs=30, start_epoch=0):
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

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)

            # Validation phase
            val_loss, val_acc, val_auc, _, _, _ = self.evaluate_epoch(val_loader, 'val')

            # Test phase (if available)
            if test_loader is not None:
                test_loss, test_acc, test_auc, _, _, _ = self.evaluate_epoch(test_loader, 'test')
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                self.history['test_auc'].append(test_auc)

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
            if test_loader is not None:
                print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}')
            print(f'Learning Rate: {current_lr:.2e}')

            # Save best model based on validation AUC
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

"""## Enhanced Training Function with Test Support"""

def enhanced_main_training(resume_from_checkpoint=None):
    """Enhanced main training with test split support"""
    print("🚀 Enhanced Deepfake Detection Training with EfficientNet-B4")
    print("="*60)

    # Check dataset structure
    dataset_ready, structure = check_dataset_structure()
    if not dataset_ready:
        print("❌ Dataset not found or structure incorrect!")
        return None

    print("📁 Setting up data loaders...")

    # Data transforms
    train_transform = get_transforms('train', CONFIG['image_size'])
    val_transform = get_transforms('val', CONFIG['image_size'])
    test_transform = get_transforms('val', CONFIG['image_size'])  # Same as val for test

    # Determine dataset structure type
    structure_type = 'standard'
    if len(structure[0]) == 1:  # Only real/fake directories
        structure_type = 'simple'

    # Create datasets
    train_dataset = DeepfakeDataset(CONFIG['data_path'], train_transform, 'train', structure_type)
    val_dataset = DeepfakeDataset(CONFIG['data_path'], val_transform, 'val', structure_type)
    
    # Test dataset (if available)
    test_dataset = None
    if any('test' in str(part) for part in structure for part in part):
        test_dataset = DeepfakeDataset(CONFIG['data_path'], test_transform, 'test', structure_type)

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

    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )

    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    if test_loader:
        print(f"📊 Test samples: {len(test_dataset)}")

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
    training_info = trainer.train(train_loader, val_loader, test_loader, CONFIG['num_epochs'], start_epoch)

    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(trainer.history, test_loader is not None)

    # Final evaluation
    print("\n📊 Final evaluation...")
    
    # Validation evaluation
    val_loss, val_acc, val_auc, val_preds, val_labels, val_probs = trainer.evaluate_epoch(val_loader, 'val')
    val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
    print_metrics(val_metrics, 'Final Validation')
    plot_confusion_matrix(val_metrics['confusion_matrix'], 'Validation Confusion Matrix')
    
    # Test evaluation (if available)
    if test_loader:
        test_loss, test_acc, test_auc, test_preds, test_labels, test_probs = trainer.evaluate_epoch(test_loader, 'test')
        test_metrics = calculate_metrics(test_labels, test_preds, test_probs)
        print_metrics(test_metrics, 'Final Test')
        plot_confusion_matrix(test_metrics['confusion_matrix'], 'Test Confusion Matrix')

    return trainer, training_info

"""## Utility Functions"""

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
    
    # Additional metrics
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return metrics

def plot_training_history(history, has_test=False):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    if has_test and history['test_loss']:
        axes[0, 0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    if has_test and history['test_acc']:
        axes[0, 1].plot(history['test_acc'], label='Test Acc', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC
    axes[1, 0].plot(history['val_auc'], label='Val AUC', color='red', linewidth=2)
    if has_test and history['test_auc']:
        axes[1, 0].plot(history['test_auc'], label='Test AUC', color='purple', linewidth=2)
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

def plot_confusion_matrix(cm, title='Confusion Matrix', classes=['Fake', 'Real']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 16}, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=16, fontweight='bold')
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
        model = StableDeepfakeEfficientNet(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load and preprocess image
        transform = get_transforms('val', CONFIG['image_size'])
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction with safety
        with torch.no_grad():
            if CONFIG['use_amp'] and device.type == 'cuda':
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

"""## Main Execution Block"""

if __name__ == "__main__":
    # Check if we should resume training
    resume_checkpoint = resume_training()

    # Execute training
    if dataset_ready:
        trainer, training_info = enhanced_main_training(resume_from_checkpoint=resume_checkpoint)
        
        # Demo predictions if training was successful
        if trainer is not None:
            print("\n🎪 Running demo predictions...")
            # You can add demo prediction code here
    else:
        print("\n❌ Cannot start training. Please fix dataset structure first.")
        trainer, training_info = None, None

    print("\n" + "="*60)
    print("✅ ENHANCED DEEPFAKE DETECTION SYSTEM READY")
    print("="*60)
    
    print(f"\n📁 Expected dataset structure:")
    print("   dataset/train/real/")
    print("   dataset/train/fake/") 
    print("   dataset/val/real/")
    print("   dataset/val/fake/")
    print("   dataset/test/real/")
    print("   dataset/test/fake/")
    
    print(f"\n💡 Usage:")
    print("   1. Place your dataset in the structure above")
    print("   2. Run this script to train the model")
    print("   3. Use enhanced_predict_image() for predictions")
    print("   4. Checkpoints are saved in ./checkpoints/")