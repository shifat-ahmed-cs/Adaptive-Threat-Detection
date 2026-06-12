import numpy as np
from pathlib import Path
import pickle

def split_existing():
    data_dir = Path("data/custom_dataset")
    print("Loading existing dataset files from:", data_dir)
    
    # Load the full unsplit files
    cnn = np.load(data_dir / 'train_cnn.npy')
    lstm = np.load(data_dir / 'train_lstm.npy')
    labels = np.load(data_dir / 'train_labels.npy')
    
    num_samples = len(labels)
    print(f"Loaded {num_samples} samples.")
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    cnn = cnn[indices]
    lstm = lstm[indices]
    labels = labels[indices]
    
    # Split: 70% train, 15% val, 15% test
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    # Save splits
    print("Saving train, val, and test splits...")
    np.save(data_dir / 'train_cnn.npy', cnn[:train_end])
    np.save(data_dir / 'train_lstm.npy', lstm[:train_end])
    np.save(data_dir / 'train_labels.npy', labels[:train_end])
    
    np.save(data_dir / 'val_cnn.npy', cnn[train_end:val_end])
    np.save(data_dir / 'val_lstm.npy', lstm[train_end:val_end])
    np.save(data_dir / 'val_labels.npy', labels[train_end:val_end])
    
    np.save(data_dir / 'test_cnn.npy', cnn[val_end:])
    np.save(data_dir / 'test_lstm.npy', lstm[val_end:])
    np.save(data_dir / 'test_labels.npy', labels[val_end:])
    
    print(f"Successfully split and saved dataset!")
    print(f"  Train: {train_end} samples")
    print(f"  Val:   {val_end - train_end} samples")
    print(f"  Test:  {num_samples - val_end} samples")

if __name__ == "__main__":
    split_existing()
