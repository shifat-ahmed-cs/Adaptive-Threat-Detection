import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from torch.utils.data import DataLoader
from models import DualBranchModel
from utils.data_loader import MalwareDataset
import pickle
import torch.nn.functional as F

# Set style
plt.style.use('seaborn-v0_8')

def plot_training_history(history_path, output_dir):
    print("Plotting training history...")
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    history = data.get('training_history', data) # Handle nested or flat structure
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    
    # Accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    plt.close()
    print(f"Saved training_history.png")

def plot_confusion_matrix(y_true, y_pred, output_dir):
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malware'],
                yticklabels=['Benign', 'Malware'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    print(f"Saved confusion_matrix.png")

def plot_roc_curve(y_true, y_probs, output_dir):
    print("Plotting ROC curve...")
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()
    print(f"Saved roc_curve.png")

# Hook storage
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def plot_activation_heatmaps(model, device, test_loader, output_dir, num_samples=5):
    print("Generating activation heatmaps...")
    
    # Register hook on the last convolutional layer (bn4 in CNNBranch)
    # Accessing via model.cnn_branch.bn4
    model.cnn_branch.bn4.register_forward_hook(get_activation('last_conv'))
    
    model.eval()
    
    # Find some malware samples
    found = 0
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    
    with torch.no_grad():
        for batch in test_loader:
            cnn_input = batch['cnn_input'].to(device)
            lstm_input = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass trigger hooks
            outputs = model(cnn_input, lstm_input)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Act map: (batch, channels, h, w) -> (batch, 256, 8, 8) likely
            act_maps = activations['last_conv']
            
            for i in range(len(labels)):
                if labels[i] == 1 and probs[i] > 0.9: # High confidence malware
                    if found >= num_samples:
                        break
                    
                    # Original Image
                    img = cnn_input[i, 0].cpu().numpy()
                    
                    # Heatmap: Average across channels
                    # act_maps[i] shape: (256, H, W) -> (Batched: 1, 256, H, W)
                    feature_map = act_maps[i:i+1] # Keep batch dim
                    heatmap = torch.mean(feature_map, dim=1, keepdim=True) # (1, 1, H, W)
                    heatmap = F.relu(heatmap)
                    
                    # Resize to input dimensions
                    # cnn_input shape: (batch, 1, H_in, W_in)
                    heatmap = F.interpolate(heatmap, size=(img.shape[0], img.shape[1]), 
                                          mode='bilinear', align_corners=False)
                    
                    heatmap = heatmap.squeeze().cpu().numpy() # (H_in, W_in)
                    
                    if np.max(heatmap) > 0:
                        heatmap /= np.max(heatmap) # Normalize
                    
                    # Resize handled by F.interpolate above
                    
                    # Plot Original
                    axes[found, 0].imshow(img, cmap='gray')
                    axes[found, 0].set_title(f'Malware Input (Prob: {probs[i]:.4f})')
                    axes[found, 0].axis('off')
                    
                    # Plot Heatmap Overlay
                    axes[found, 1].imshow(img, cmap='gray')
                    axes[found, 1].imshow(heatmap, cmap='jet', alpha=0.5) # Overlay
                    axes[found, 1].set_title('Activation Heatmap')
                    axes[found, 1].axis('off')
                    
                    found += 1
            
            if found >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(output_dir / 'activation_heatmaps.png')
    plt.close()
    print(f"Saved activation_heatmaps.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    args = parser.parse_args()
    
    device = torch.device('cpu') # Use CPU for stable plotting
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot Training History
    if Path('checkpoints/training_history.json').exists():
        plot_training_history('checkpoints/training_history.json', output_dir)
    else:
        print("Warning: training_history.json not found.")
        
    # 2. Load Model & Data
    print("Loading model and data...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Metadata for model config
    try:
        with open(Path(args.data_dir) / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = {'vocab_size': 1000} # Fallback
        
    model_config = checkpoint.get('model_config', {})
    
    model = DualBranchModel(
        cnn_config={
            'input_channels': 1, 
            'num_classes': 2,
            'feature_dim': model_config.get('cnn_feature_dim', 256)
        },
        lstm_config={
            'vocab_size': metadata.get('vocab_size', 1000),
            'bidirectional': True,
            'embedding_dim': model_config.get('embedding_dim', 128),
            'hidden_size': model_config.get('lstm_hidden_size', 128),
            'num_layers': model_config.get('lstm_num_layers', 2),
            'feature_dim': model_config.get('lstm_feature_dim', 256)
        },
        num_classes=2,
        fusion_method=model_config.get('fusion_method', 'concat')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load Test Data
    data_path = Path(args.data_dir)
    test_cnn = np.load(data_path / 'test_cnn.npy')
    test_lstm = np.load(data_path / 'test_lstm.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    test_dataset = MalwareDataset(test_cnn, test_lstm, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Get Predictions
    all_labels = []
    all_probs = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            cnn_input = batch['cnn_input'].to(device)
            lstm_input = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(cnn_input, lstm_input)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # 4. Plot Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, output_dir)
    
    # 5. Plot ROC Curve
    plot_roc_curve(all_labels, all_probs, output_dir)
    
    # 6. Activation Heatmaps
    plot_activation_heatmaps(model, device, test_loader, output_dir)
    
    print("Done! Check 'visualizations' directory.")

if __name__ == "__main__":
    main()
