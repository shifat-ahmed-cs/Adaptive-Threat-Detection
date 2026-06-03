import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
from models.dual_branch_model import DualBranchModel

def fine_tune_adaptive_model():
    data_dir = Path("data/custom_dataset")
    print("Loading dataset for fine-tuning...")
    cnn_data = np.load(data_dir / 'train_cnn.npy')
    lstm_data = np.load(data_dir / 'train_lstm.npy')
    labels = np.load(data_dir / 'train_labels.npy')
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 1. Initialize the Baseline Model
    vocab_size = metadata['vocab_size']
    lstm_config = {
        'vocab_size': vocab_size,
        'input_size': 100,
        'hidden_size': 128,
        'num_layers': 2,
        'feature_dim': 256,
        'bidirectional': True
    }
    model = DualBranchModel(lstm_config=lstm_config)
    
    # 2. Load the weights from our last successful training
    print("Loading previous weights...")
    checkpoint = torch.load("new_custom_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. TRIGGER DYNAMIC EXPANSION
    # This adds the new layer that we just talked about
    model.expand_capacity()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Fine-tuning Setup
    dataset = TensorDataset(torch.FloatTensor(cnn_data), torch.LongTensor(lstm_data), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # Use a smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Fine-tuning Adaptive Model on {device}...")
    num_epochs = 5 # Short burst to settle the new weights
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for cnn_batch, lstm_batch, labels_batch in loader:
            cnn_batch, lstm_batch, labels_batch = cnn_batch.to(device), lstm_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(cnn_batch, lstm_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader):.4f}")

    # 5. Save the final adaptive model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': checkpoint['model_config'],
        'metadata': checkpoint['metadata'],
        'is_expanded': True
    }
    
    # Update config for the scanner
    final_checkpoint['model_config']['is_expanded'] = True
    
    save_path = Path("adaptive_model_final.pth")
    torch.save(final_checkpoint, save_path)
    print(f"\n[SUCCESS] Final Self-Adaptive model saved to {save_path}")

if __name__ == "__main__":
    fine_tune_adaptive_model()
