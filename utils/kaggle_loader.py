"""
Custom data loader for Kaggle malware datasets:
1. Malware Analysis Datasets: Raw PE as Image
2. Malware Analysis Datasets: API Call Sequences
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import pickle


class KaggleMalwareLoader:
    """
    Loader for the two Kaggle malware datasets
    """
    
    def __init__(self, pe_image_path, api_sequence_path):
        """
        Initialize the loader
        
        Args:
            pe_image_path (str): Path to PE images dataset directory
            api_sequence_path (str): Path to API call sequences dataset directory
        """
        self.pe_image_path = Path(pe_image_path)
        self.api_sequence_path = Path(api_sequence_path)
        
    def load_pe_images(self, target_size=(256, 256), normalize=True):
        """
        Load PE binary images
        
        Args:
            target_size (tuple): Target image size (height, width)
            normalize (bool): Whether to normalize pixel values to [0, 1]
            
        Returns:
            tuple: (images, labels, filenames)
        """
        print("Loading PE images...")
        
        images = []
        labels = []
        filenames = []
        
        # Look for common dataset structures
        # Typically: benign/ and malware/ folders or train.csv with labels
        
        # Check for CSV with labels
        csv_files = list(self.pe_image_path.glob("*.csv"))
        
        if csv_files:
            # Load from CSV
            df = pd.read_csv(csv_files[0])
            print(f"Found CSV file: {csv_files[0].name}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Common column names for labels
            label_col = None
            for col in ['label', 'Label', 'class', 'Class', 'target', 'Target', 'malware', 'Malware']:
                if col in df.columns:
                    label_col = col
                    break
            
            # Common column names for filenames
            filename_col = None
            for col in ['filename', 'Filename', 'file', 'File', 'name', 'Name', 'id', 'Id']:
                if col in df.columns:
                    filename_col = col
                    break
            
            if label_col and filename_col:
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
                    img_path = self.pe_image_path / row[filename_col]
                    
                    # Try different extensions
                    if not img_path.exists():
                        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.npy']:
                            test_path = self.pe_image_path / f"{row[filename_col]}{ext}"
                            if test_path.exists():
                                img_path = test_path
                                break
                    
                    if img_path.exists():
                        if img_path.suffix == '.npy':
                            img = np.load(img_path)
                        else:
                            img = Image.open(img_path).convert('L')  # Grayscale
                            img = img.resize(target_size)
                            img = np.array(img)
                        
                        # Ensure correct shape
                        if img.ndim == 2:
                            img = img[np.newaxis, :, :]  # Add channel dimension
                        
                        if normalize and img.max() > 1.0:
                            img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(row[label_col])
                        filenames.append(row[filename_col])
        
        else:
            # Load from directory structure (benign/ and malware/)
            for class_name, label in [('benign', 0), ('malware', 1)]:
                class_dir = self.pe_image_path / class_name
                
                if not class_dir.exists():
                    # Try capitalized versions
                    class_dir = self.pe_image_path / class_name.capitalize()
                
                if class_dir.exists():
                    image_files = list(class_dir.glob('*.png')) + \
                                 list(class_dir.glob('*.jpg')) + \
                                 list(class_dir.glob('*.npy'))
                    
                    for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                        if img_path.suffix == '.npy':
                            img = np.load(img_path)
                        else:
                            img = Image.open(img_path).convert('L')
                            img = img.resize(target_size)
                            img = np.array(img)
                        
                        if img.ndim == 2:
                            img = img[np.newaxis, :, :]
                        
                        if normalize and img.max() > 1.0:
                            img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(label)
                        filenames.append(img_path.name)
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"Loaded {len(images)} PE images")
        print(f"Image shape: {images.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return images, labels, filenames
    
    def load_api_sequences(self, max_length=1000, vocab_size=None):
        """
        Load API call sequences
        
        Args:
            max_length (int): Maximum sequence length (pad/truncate)
            vocab_size (int): Vocabulary size for encoding (None = auto-detect)
            
        Returns:
            tuple: (sequences, labels, filenames, vocab)
        """
        print("Loading API call sequences...")
        
        sequences = []
        labels = []
        filenames = []
        vocab = {}
        
        # Look for CSV, JSON, or pickle files
        csv_files = list(self.api_sequence_path.glob("*.csv"))
        json_files = list(self.api_sequence_path.glob("*.json"))
        pkl_files = list(self.api_sequence_path.glob("*.pkl"))
        
        if csv_files:
            # Load from CSV
            df = pd.read_csv(csv_files[0])
            print(f"Found CSV file: {csv_files[0].name}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Find label column
            label_col = None
            for col in ['label', 'Label', 'class', 'Class', 'target', 'Target', 'malware', 'Malware']:
                if col in df.columns:
                    label_col = col
                    break
            
            # Find sequence column
            seq_col = None
            for col in ['sequence', 'Sequence', 'api_calls', 'API_calls', 'calls', 'Calls', 'apis', 'APIs']:
                if col in df.columns:
                    seq_col = col
                    break
            
            if not seq_col:
                # If no specific sequence column, assume all other columns are features
                feature_cols = [col for col in df.columns if col != label_col]
                seq_col = feature_cols
            
            # Build vocabulary from all sequences
            all_tokens = set()
            
            if isinstance(seq_col, list):
                # Multiple columns represent the sequence
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building vocabulary"):
                    seq = row[seq_col].values
                    all_tokens.update(seq.astype(str))
            else:
                # Single column with sequence (might be string or list)
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building vocabulary"):
                    seq = row[seq_col]
                    
                    # Parse if it's a string representation
                    if isinstance(seq, str):
                        # Try to parse as JSON list
                        try:
                            seq = json.loads(seq)
                        except:
                            # Split by common delimiters
                            seq = seq.replace('[', '').replace(']', '').replace("'", "").split(',')
                            seq = [s.strip() for s in seq]
                    
                    all_tokens.update([str(s) for s in seq])
            
            # Create vocabulary mapping
            vocab = {token: idx + 1 for idx, token in enumerate(sorted(all_tokens))}  # 0 reserved for padding
            vocab['<PAD>'] = 0
            
            print(f"Vocabulary size: {len(vocab)}")
            
            # Encode sequences
            if isinstance(seq_col, list):
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding sequences"):
                    seq = row[seq_col].values.astype(str)
                    encoded = [vocab.get(token, 0) for token in seq]
                    
                    # Pad or truncate
                    if len(encoded) > max_length:
                        encoded = encoded[:max_length]
                    else:
                        encoded = encoded + [0] * (max_length - len(encoded))
                    
                    sequences.append(encoded)
                    if label_col:
                        labels.append(row[label_col])
            else:
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding sequences"):
                    seq = row[seq_col]
                    
                    if isinstance(seq, str):
                        try:
                            seq = json.loads(seq)
                        except:
                            seq = seq.replace('[', '').replace(']', '').replace("'", "").split(',')
                            seq = [s.strip() for s in seq]
                    
                    encoded = [vocab.get(str(token), 0) for token in seq]
                    
                    # Pad or truncate
                    if len(encoded) > max_length:
                        encoded = encoded[:max_length]
                    else:
                        encoded = encoded + [0] * (max_length - len(encoded))
                    
                    sequences.append(encoded)
                    if label_col:
                        labels.append(row[label_col])
        
        elif json_files:
            # Load from JSON
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for item in tqdm(data, desc="Processing JSON"):
                    if 'sequence' in item or 'api_calls' in item:
                        seq = item.get('sequence', item.get('api_calls'))
                        sequences.append(seq)
                        labels.append(item.get('label', item.get('class', 0)))
        
        sequences = np.array(sequences, dtype=np.int32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"Loaded {len(sequences)} API sequences")
        print(f"Sequence shape: {sequences.shape}")
        if len(labels) > 0:
            print(f"Label distribution: {np.bincount(labels)}")
        
        return sequences, labels, filenames, vocab
    
    def load_combined_dataset(self, pe_size=(256, 256), seq_length=1000):
        """
        Load both datasets and align them
        
        Args:
            pe_size (tuple): Target size for PE images
            seq_length (int): Maximum sequence length
            
        Returns:
            dict: Dictionary containing aligned data
        """
        print("Loading combined dataset...")
        
        # Load both datasets
        pe_images, pe_labels, pe_files = self.load_pe_images(target_size=pe_size)
        api_seqs, api_labels, api_files, vocab = self.load_api_sequences(max_length=seq_length)
        
        # Try to align datasets by filename
        # This assumes both datasets have corresponding samples
        
        return {
            'pe_images': pe_images,
            'pe_labels': pe_labels,
            'pe_files': pe_files,
            'api_sequences': api_seqs,
            'api_labels': api_labels,
            'api_files': api_files,
            'vocab': vocab
        }


def prepare_dual_branch_data(pe_image_path, api_sequence_path, 
                             train_split=0.8, val_split=0.1,
                             pe_size=(256, 256), seq_length=1000,
                             save_path=None):
    """
    Prepare data for dual-branch model training
    
    Args:
        pe_image_path (str): Path to PE images dataset
        api_sequence_path (str): Path to API sequences dataset
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        pe_size (tuple): Target size for PE images
        seq_length (int): Maximum sequence length
        save_path (str, optional): Path to save processed data
        
    Returns:
        dict: Dictionary containing train/val/test splits
    """
    loader = KaggleMalwareLoader(pe_image_path, api_sequence_path)
    
    # Load datasets
    pe_images, pe_labels, _ = loader.load_pe_images(target_size=pe_size)
    api_seqs, api_labels, _, vocab = loader.load_api_sequences(max_length=seq_length)
    
    # Ensure same number of samples
    min_samples = min(len(pe_images), len(api_seqs))
    pe_images = pe_images[:min_samples]
    pe_labels = pe_labels[:min_samples]
    api_seqs = api_seqs[:min_samples]
    api_labels = api_labels[:min_samples]
    
    # Use PE labels as primary (they should match)
    labels = pe_labels
    
    # Shuffle data
    indices = np.random.permutation(min_samples)
    pe_images = pe_images[indices]
    api_seqs = api_seqs[indices]
    labels = labels[indices]
    
    # Split data
    train_end = int(train_split * min_samples)
    val_end = int((train_split + val_split) * min_samples)
    
    data = {
        'train_cnn': pe_images[:train_end],
        'train_lstm': api_seqs[:train_end],
        'train_labels': labels[:train_end],
        
        'val_cnn': pe_images[train_end:val_end],
        'val_lstm': api_seqs[train_end:val_end],
        'val_labels': labels[train_end:val_end],
        
        'test_cnn': pe_images[val_end:],
        'test_lstm': api_seqs[val_end:],
        'test_labels': labels[val_end:],
        
        'vocab': vocab,
        'vocab_size': len(vocab)
    }
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / 'train_cnn.npy', data['train_cnn'])
        np.save(save_path / 'train_lstm.npy', data['train_lstm'])
        np.save(save_path / 'train_labels.npy', data['train_labels'])
        
        np.save(save_path / 'val_cnn.npy', data['val_cnn'])
        np.save(save_path / 'val_lstm.npy', data['val_lstm'])
        np.save(save_path / 'val_labels.npy', data['val_labels'])
        
        np.save(save_path / 'test_cnn.npy', data['test_cnn'])
        np.save(save_path / 'test_lstm.npy', data['test_lstm'])
        np.save(save_path / 'test_labels.npy', data['test_labels'])
        
        with open(save_path / 'vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        
        print(f"Data saved to {save_path}")
    
    print("\nDataset Statistics:")
    print(f"Training samples: {len(data['train_labels'])}")
    print(f"Validation samples: {len(data['val_labels'])}")
    print(f"Test samples: {len(data['test_labels'])}")
    print(f"Vocabulary size: {data['vocab_size']}")
    
    return data
