"""
Training script for dual-branch malware detection model
Handles class imbalance and provides comprehensive training with validation
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from models import DualBranchModel
from utils import create_data_loaders


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    # Calculate weights: n_samples / (n_classes * class_count)
    weights = n_samples / (n_classes * class_counts)
    
    print(f"\nClass distribution:")
    for cls, count, weight in zip(unique_classes, class_counts, weights):
        class_name = "Benign" if cls == 0 else "Malware"
        print(f"  {class_name} (class {cls}): {count} samples, weight: {weight:.4f}")
    
    return torch.FloatTensor(weights)


# ... (imports)
from utils.mutator import ThreatMutator

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, train_loader, criterion, optimizer, device, adversarial=False, mutation_rate=0.1, mutation_strategy='random_insert', vocab_size=307, max_batches=None):
    """Train for one epoch with optional adversarial training"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize mutator if needed
    mutator = None
    if adversarial:
        # Assuming vocab_size is provided, otherwise default fallback
        mutator = ThreatMutator(vocab_size=vocab_size)
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        cnn_input = batch['cnn_input'].to(device)
        lstm_input = batch['lstm_input'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass (Clean)
        optimizer.zero_grad()
        outputs = model(cnn_input, lstm_input)
        loss = criterion(outputs, labels)
        
        # Adversarial Training Step
        if adversarial:
            # Generate mutated sequences (Objective 2)
            lstm_adv = mutator.mutate_batch(
                lstm_input, 
                mutation_rate=mutation_rate, 
                strategy=mutation_strategy
            )
            outputs_adv = model(cnn_input, lstm_adv)
            loss_adv = criterion(outputs_adv, labels)
            
            # Robust Loss = Clean + Adversarial
            loss = loss + loss_adv
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics (based on clean data usually, or robust acc)
        total_loss += loss.item() # Note: this loss is now higher if adv
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        status = {
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        }
        if adversarial:
            status['mode'] = 'ADV'
            
        pbar.set_postfix(status)
        
        # Max batches limit for testing
        if max_batches is not None and (pbar.n + 1) >= max_batches:
            print(f"\n✋ Reached execution limit of {max_batches} batches.")
            break
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            cnn_input = batch['cnn_input'].to(device)
            lstm_input = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(cnn_input, lstm_input)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'\n{"="*60}')
    print(f'Dual-Branch Malware Detection Training')
    print(f'{"="*60}')
    print(f'Device: {device}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(args.data_dir)
    
    # Fast Dev Run: Subsample data or generate synthetic data
    if args.fast_dev_run:
        print("\n⚡ FAST DEV RUN ENABLED: Generating synthetic data (skipping file load)")
        limit = 100
        # Create random data matching the expected shapes
        # CNN: (limit, 1, 64, 64) - Assuming 64x64 based on standard malware imgs, checking metadata later? 
        # Actually better to read metadata first.
        
        # Load metadata first to get shapes
        print("Loading metadata.pkl...")
        with open(data_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            
        img_size = metadata['image_size']
        seq_len = metadata['seq_length']
        vocab_size = metadata['vocab_size']
        
        print(f"Generating synthetic data: {limit} samples")
        train_cnn = np.random.rand(limit, 1, img_size, img_size).astype(np.float32)
        train_lstm = np.random.randint(0, vocab_size, (limit, seq_len)).astype(np.int64)
        train_labels = np.random.randint(0, 2, limit).astype(np.int64)
        
        val_cnn = np.random.rand(limit, 1, img_size, img_size).astype(np.float32)
        val_lstm = np.random.randint(0, vocab_size, (limit, seq_len)).astype(np.int64)
        val_labels = np.random.randint(0, 2, limit).astype(np.int64)

    else:
        # Load data normally
        print(f'\nLoading data from {args.data_dir}...', flush=True)
        # Load metadata
        print("Loading metadata.pkl...", flush=True)
        with open(data_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        print("Loading train_cnn.npy...", flush=True)
        train_cnn = np.load(data_path / 'train_cnn.npy', mmap_mode='r' if args.mmap else None)
        print("Loading train_lstm.npy...", flush=True)
        train_lstm = np.load(data_path / 'train_lstm.npy')
        print("Loading train_labels.npy...", flush=True)
        train_labels = np.load(data_path / 'train_labels.npy')
        
        print("Loading val_cnn.npy...", flush=True)
        val_cnn = np.load(data_path / 'val_cnn.npy', mmap_mode='r' if args.mmap else None)
        print("Loading val_lstm.npy...", flush=True)
        val_lstm = np.load(data_path / 'val_lstm.npy')
        print("Loading val_labels.npy...", flush=True)
        val_labels = np.load(data_path / 'val_labels.npy')
    
    print(f'Training samples: {len(train_labels)}')
    print(f'Validation samples: {len(val_labels)}')
    print(f'Vocabulary size: {metadata["vocab_size"]}')
    print(f'Image size: {metadata["image_size"]}x{metadata["image_size"]}')
    print(f'Sequence length: {metadata["seq_length"]}')
    
    # Calculate class weights for sampler
    # class_weights = calculate_class_weights(train_labels) # Old manual weights
    
    # Create WeightedRandomSampler
    print(f'\nPreparing WeightedRandomSampler...')
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    print(f'\nCreating data loaders (batch size: {args.batch_size})...')
    train_loader, val_loader = create_data_loaders(
        train_cnn, train_lstm, train_labels,
        val_cnn, val_lstm, val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, # Will be set to False if sampler is passed
        sampler=sampler
    )
    
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    
    # Initialize model
    print(f'\nInitializing model...')
    model = DualBranchModel(
        cnn_config={
            'input_channels': 1,
            'num_classes': 2,
            'feature_dim': args.cnn_feature_dim
        },
        lstm_config={
            'vocab_size': metadata['vocab_size'],
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.lstm_hidden_size,
            'num_layers': args.lstm_num_layers,
            'feature_dim': args.lstm_feature_dim,
            'bidirectional': True
        },
        num_classes=2,
        fusion_method=args.fusion_method
    )
    
    model = model.to(device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f'\nLoading checkpoint from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✓ Checkpoint loaded successfully')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Fusion method: {args.fusion_method}')
    
    # Loss and optimizer
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using Focal Loss...")
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.scheduler_patience, 
        factor=args.scheduler_factor
    )
    
    # Training loop
    print(f'\n{"="*60}')
    print(f'Starting training for {args.epochs} epochs')
    print(f'{"="*60}')
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            adversarial=args.adversarial,
            mutation_rate=args.mutation_rate,
            mutation_strategy=args.mutation_strategy,
            vocab_size=metadata['vocab_size'],
            max_batches=args.max_batches
        )
        
        # Validate
        if not args.skip_validation:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                
                checkpoint_path = checkpoint_dir / args.save_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'metadata': metadata,
                    'model_config': {
                        'cnn_feature_dim': args.cnn_feature_dim,
                        'lstm_hidden_size': args.lstm_hidden_size,
                        'lstm_num_layers': args.lstm_num_layers,
                        'lstm_feature_dim': args.lstm_feature_dim,
                        'embedding_dim': args.embedding_dim,
                        'fusion_method': args.fusion_method
                    }
                }, checkpoint_path)
                
                print(f'  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            # If validation is skipped, just save the model every epoch (or last epoch)
            # For this specific test request, saving every epoch (overwrite) is fine as "best"
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Validation skipped.')
            
            checkpoint_path = checkpoint_dir / args.save_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'metadata': metadata,
                'model_config': {
                    'cnn_feature_dim': args.cnn_feature_dim,
                    'lstm_hidden_size': args.lstm_hidden_size,
                    'lstm_num_layers': args.lstm_num_layers,
                    'lstm_feature_dim': args.lstm_feature_dim,
                    'embedding_dim': args.embedding_dim,
                    'fusion_method': args.fusion_method
                }
            }, checkpoint_path)
            print(f'  ✓ Model saved (Validation skipped)')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\n⚠️ Early stopping triggered after {epoch+1} epochs')
            break
    
    # Save training history
    results = {
        'training_history': history,
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'total_epochs': epoch + 1,
        'model_config': {
            'cnn_feature_dim': args.cnn_feature_dim,
            'lstm_hidden_size': args.lstm_hidden_size,
            'lstm_num_layers': args.lstm_num_layers,
            'lstm_feature_dim': args.lstm_feature_dim,
            'embedding_dim': args.embedding_dim,
            'fusion_method': args.fusion_method
        }
    }
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'{"="*60}')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Total epochs trained: {epoch+1}')
    print(f'Total epochs trained: {epoch+1}')
    print(f'Model saved to: {checkpoint_dir / args.save_name}')
    print(f'History saved to: {checkpoint_dir / "training_history.json"}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dual-branch malware detection model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--cnn-feature-dim', type=int, default=256,
                        help='CNN feature dimension')
    parser.add_argument('--lstm-hidden-size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--lstm-num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm-feature-dim', type=int, default=256,
                        help='LSTM feature dimension')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Embedding dimension for API sequences')
    parser.add_argument('--fusion-method', type=str, default='concat',
                        choices=['concat', 'add', 'attention'],
                        help='Feature fusion method')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--scheduler-patience', type=int, default=3,
                        help='Learning rate scheduler patience')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='Learning rate reduction factor')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers (use 0 for Windows)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mmap', action='store_true',
                        help='Use memory mapping for large arrays')
    
    # Adversarial parameters
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable adversarial training (Threat Mutation)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Rate of mutation for adversarial samples')
    parser.add_argument('--mutation-strategy', type=str, default='random_insert',
                        choices=['random_insert', 'random_replace', 'noise_append'],
                        help='Strategy for adversarial mutation')
    
    # Checkpoint loading and saving parameters
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save-name', type=str, default='best_model.pth',
                        help='Filename to save the best model')
    
    # Testing/Debug parameters
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Limit number of batches per epoch (for testing)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation step')
    
    # Adaptive Architecture parameter (Objective 1)
    parser.add_argument('--adaptive', action='store_true',
                        help='Enable self-adaptive architecture (dynamic expansion)')
    
    # Fast verification for demo/testing
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='Run on small subset of data for quick verification')
    
    args = parser.parse_args()
    main(args)
