import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from models import DualBranchModel
from utils.data_loader import MalwareDataset
from torch.utils.data import DataLoader

def train_fast():
    print("Loading a balanced subset of real data for fast training...")
    data_path = Path("data/processed")
    
    # Load labels first
    train_labels = np.load(data_path / 'train_labels.npy')
    
    # Find indices
    benign_idx = np.where(train_labels == 0)[0]
    malware_idx = np.where(train_labels == 1)[0]
    
    # We will use ALL benign samples (~142) and an equal number of malware samples
    # to perfectly balance the dataset and prevent mode collapse
    num_samples = len(benign_idx)
    selected_malware = np.random.choice(malware_idx, num_samples, replace=False)
    
    selected_idx = np.concatenate([benign_idx, selected_malware])
    np.random.shuffle(selected_idx)
    
    print(f"Selected {num_samples} Benign and {num_samples} Malware samples for training.")
    
    # Load actual data but only the selected indices
    # mmap is crucial to avoid loading the whole 2GB file
    print("Loading CNN and LSTM features from mmap...")
    train_cnn_full = np.load(data_path / 'train_cnn.npy', mmap_mode='r')
    train_lstm_full = np.load(data_path / 'train_lstm.npy', mmap_mode='r')
    
    train_cnn = np.array([train_cnn_full[i] for i in selected_idx])
    train_lstm = np.array([train_lstm_full[i] for i in selected_idx])
    train_y = train_labels[selected_idx]
    
    # Validation data (just take 50 of each)
    val_labels = np.load(data_path / 'val_labels.npy')
    val_benign = np.where(val_labels == 0)[0][:50]
    val_malware = np.where(val_labels == 1)[0][:50]
    val_idx = np.concatenate([val_benign, val_malware])
    
    val_cnn_full = np.load(data_path / 'val_cnn.npy', mmap_mode='r')
    val_lstm_full = np.load(data_path / 'val_lstm.npy', mmap_mode='r')
    
    val_cnn = np.array([val_cnn_full[i] for i in val_idx])
    val_lstm = np.array([val_lstm_full[i] for i in val_idx])
    val_y = val_labels[val_idx]
    
    # Load metadata
    with open(data_path / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        
    # Create datasets
    train_dataset = MalwareDataset(train_cnn, train_lstm, train_y)
    val_dataset = MalwareDataset(val_cnn, val_lstm, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = DualBranchModel(
        cnn_config={'input_channels': 1, 'num_classes': 2, 'feature_dim': 256},
        lstm_config={
            'vocab_size': metadata['vocab_size'],
            'embedding_dim': 128,
            'hidden_size': 128,
            'num_layers': 2,
            'feature_dim': 256,
            'bidirectional': True
        },
        num_classes=2,
        fusion_method='concat'
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting Training...")
    best_acc = 0
    
    for epoch in range(5):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            cnn_in = batch['cnn_input'].to(device)
            lstm_in = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            out = model(cnn_in, lstm_in)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(out, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100 * correct / total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                cnn_in = batch['cnn_input'].to(device)
                lstm_in = batch['lstm_input'].to(device)
                labels = batch['label'].to(device)
                
                out = model(cnn_in, lstm_in)
                pred = torch.argmax(out, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata,
                'model_config': {
                    'cnn_feature_dim': 256,
                    'lstm_hidden_size': 128,
                    'lstm_num_layers': 2,
                    'lstm_feature_dim': 256,
                    'embedding_dim': 128,
                    'fusion_method': 'concat'
                }
            }, 'checkpoints/best_model_v1.pth')
            print("Saved new best model!")
            
    print("\nTraining complete! The model has been saved to checkpoints/best_model_v1.pth")

if __name__ == "__main__":
    train_fast()
