import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from pathlib import Path
from models.dual_branch_model import DualBranchModel

def train_custom_model():
    data_dir = Path("data/custom_dataset")
    if not data_dir.exists():
        print(f"Error: Dataset directory {data_dir} not found. Please run safe_feature_extractor.py first.")
        return

    print("Loading custom dataset...")
    try:
        cnn_data = np.load(data_dir / 'train_cnn.npy')
        lstm_data = np.load(data_dir / 'train_lstm.npy')
        labels = np.load(data_dir / 'train_labels.npy')
        
        with open(data_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print("Error: Dataset files missing. Ensure the extraction process finished completely.")
        return

    # Convert to Tensors
    cnn_tensor = torch.FloatTensor(cnn_data)
    lstm_tensor = torch.LongTensor(lstm_data)
    labels_tensor = torch.LongTensor(labels)

    dataset = TensorDataset(cnn_tensor, lstm_tensor, labels_tensor)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize Baseline Model
    vocab_size = metadata['vocab_size']
    lstm_config = {
        'vocab_size': vocab_size,
        'input_size': 100,  # Embedding dimension
        'hidden_size': 128,
        'num_layers': 2,
        'feature_dim': 256,
        'bidirectional': True
    }
    model = DualBranchModel(lstm_config=lstm_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {device}...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for cnn_batch, lstm_batch, labels_batch in train_loader:
            cnn_batch, lstm_batch, labels_batch = cnn_batch.to(device), lstm_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(cnn_batch, lstm_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for cnn_batch, lstm_batch, labels_batch in val_loader:
                cnn_batch, lstm_batch, labels_batch = cnn_batch.to(device), lstm_batch.to(device), labels_batch.to(device)
                outputs = model(cnn_batch, lstm_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.2f}%")

    # Save the new model in the format expected by the Scanner
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'cnn_feature_dim': 256,
            'embedding_dim': 100,
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'lstm_feature_dim': 256,
            'fusion_method': 'concat'
        },
        'metadata': {
            'vocab_size': vocab_size,
            'image_size': 128,
            'seq_length': 100
        }
    }
    
    save_path = Path("new_custom_model.pth")
    torch.save(checkpoint, save_path)
    print(f"\n[SUCCESS] Model saved with metadata to {save_path}")
    print(f"Use vocab.pkl from {data_dir} for inference.")

if __name__ == "__main__":
    train_custom_model()
