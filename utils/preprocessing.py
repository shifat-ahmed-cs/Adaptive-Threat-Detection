"""
Preprocessing utilities for malware data
"""

import numpy as np
from PIL import Image
import torch


def preprocess_binary(binary_data, target_size=(256, 256), normalize=True):
    """
    Preprocess binary malware data for CNN input
    Converts binary data to image representation
    
    Args:
        binary_data (bytes or np.ndarray): Raw binary data
        target_size (tuple): Target image size (height, width)
        normalize (bool): Whether to normalize pixel values
        
    Returns:
        np.ndarray: Preprocessed image data
    """
    if isinstance(binary_data, bytes):
        # Convert bytes to numpy array
        binary_array = np.frombuffer(binary_data, dtype=np.uint8)
    else:
        binary_array = binary_data
    
    # Calculate dimensions for square-ish image
    total_pixels = target_size[0] * target_size[1]
    
    # Pad or truncate to fit target size
    if len(binary_array) < total_pixels:
        # Pad with zeros
        padded = np.zeros(total_pixels, dtype=np.uint8)
        padded[:len(binary_array)] = binary_array
        binary_array = padded
    else:
        # Truncate
        binary_array = binary_array[:total_pixels]
    
    # Reshape to image
    image = binary_array.reshape(target_size)
    
    # Add channel dimension (grayscale)
    image = image[np.newaxis, :, :]
    
    # Normalize if requested
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def preprocess_sequence(sequence_data, max_length=1000, vocab_size=256, padding='post'):
    """
    Preprocess sequential data (e.g., API calls, opcodes) for LSTM input
    
    Args:
        sequence_data (list or np.ndarray): Sequential data
        max_length (int): Maximum sequence length
        vocab_size (int): Size of vocabulary for one-hot encoding
        padding (str): 'pre' or 'post' padding
        
    Returns:
        np.ndarray: Preprocessed sequence data
    """
    if isinstance(sequence_data, list):
        sequence_data = np.array(sequence_data)
    
    # Truncate or pad sequence
    if len(sequence_data) > max_length:
        # Truncate
        sequence_data = sequence_data[:max_length]
    else:
        # Pad
        pad_length = max_length - len(sequence_data)
        if padding == 'post':
            sequence_data = np.concatenate([sequence_data, np.zeros(pad_length)])
        else:  # 'pre'
            sequence_data = np.concatenate([np.zeros(pad_length), sequence_data])
    
    # One-hot encode if needed
    # For now, return as-is assuming it's already encoded
    # You can add one-hot encoding here if needed
    
    return sequence_data.astype(np.float32)


def normalize_features(features, method='standard'):
    """
    Normalize feature vectors
    
    Args:
        features (np.ndarray): Feature array
        method (str): Normalization method ('standard', 'minmax', 'l2')
        
    Returns:
        np.ndarray: Normalized features
    """
    if method == 'standard':
        # Standardization (zero mean, unit variance)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        # Min-max normalization
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        return (features - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'l2':
        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def augment_binary_image(image, rotation_range=15, flip=True):
    """
    Apply data augmentation to binary image representations
    
    Args:
        image (np.ndarray): Input image
        rotation_range (int): Range for random rotation in degrees
        flip (bool): Whether to apply random flipping
        
    Returns:
        np.ndarray: Augmented image
    """
    # Convert to PIL Image for augmentation
    if image.ndim == 3:
        image = image[0]  # Remove channel dimension
    
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        pil_image = pil_image.rotate(angle)
    
    # Random flip
    if flip and np.random.random() > 0.5:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Convert back to numpy
    augmented = np.array(pil_image).astype(np.float32) / 255.0
    augmented = augmented[np.newaxis, :, :]  # Add channel dimension
    
    return augmented
