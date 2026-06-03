"""
Data loading utilities for malware detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path


class MalwareDataset(Dataset):
    """
    Custom Dataset for dual-branch malware detection
    Handles both image-based (CNN) and sequence-based (LSTM) data
    """
    
    def __init__(self, cnn_data, lstm_data, labels, transform=None):
        """
        Initialize dataset
        
        Args:
            cnn_data (np.ndarray or list): Data for CNN branch
            lstm_data (np.ndarray or list): Data for LSTM branch
            labels (np.ndarray or list): Labels for classification
            transform (callable, optional): Optional transform to be applied on samples
        """
        self.cnn_data = cnn_data
        self.lstm_data = lstm_data
        self.labels = labels
        self.transform = transform
        
        assert len(cnn_data) == len(lstm_data) == len(labels), \
            "All data arrays must have the same length"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing CNN input, LSTM input, and label
        """
        cnn_sample = self.cnn_data[idx]
        lstm_sample = self.lstm_data[idx]
        label = self.labels[idx]
        
        # Convert to tensors if not already
        if not isinstance(cnn_sample, torch.Tensor):
            cnn_sample = torch.FloatTensor(cnn_sample)
        
        if not isinstance(lstm_sample, torch.Tensor):
            lstm_sample = torch.FloatTensor(lstm_sample)
        
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label])[0]
        
        # Apply transforms if any
        if self.transform:
            cnn_sample = self.transform(cnn_sample)
        
        return {
            'cnn_input': cnn_sample,
            'lstm_input': lstm_sample,
            'label': label
        }


def create_data_loaders(train_cnn, train_lstm, train_labels,
                       val_cnn, val_lstm, val_labels,
                       batch_size=32, num_workers=4, shuffle=True, sampler=None):
    """
    Create training and validation data loaders
    
    Args:
        train_cnn (np.ndarray): Training data for CNN branch
        train_lstm (np.ndarray): Training data for LSTM branch
        train_labels (np.ndarray): Training labels
        val_cnn (np.ndarray): Validation data for CNN branch
        val_lstm (np.ndarray): Validation data for LSTM branch
        val_labels (np.ndarray): Validation labels
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle training data (ignored if sampler is provided)
        sampler (Sampler): Custom sampler for training data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MalwareDataset(train_cnn, train_lstm, train_labels)
    val_dataset = MalwareDataset(val_cnn, val_lstm, val_labels)
    
    # If sampler is provided, shuffle must be False
    if sampler is not None:
        shuffle = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_kaggle_dataset(data_path, dataset_name='malware'):
    """
    Load malware dataset from Kaggle data directory
    
    Args:
        data_path (str or Path): Path to data directory
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dictionary containing loaded data
    """
    data_path = Path(data_path)
    
    # This is a template function - adjust based on your specific Kaggle dataset structure
    print(f"Loading dataset from {data_path}")
    
    # Example: Load CSV files
    # train_df = pd.read_csv(data_path / 'train.csv')
    # test_df = pd.read_csv(data_path / 'test.csv')
    
    # Return placeholder - implement based on your dataset
    return {
        'train': None,
        'test': None,
        'labels': None
    }
