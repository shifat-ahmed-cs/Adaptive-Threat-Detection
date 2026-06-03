import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
import random

# Set style
plt.style.use('seaborn-v0_8')

def plot_class_distribution(labels_path, output_dir):
    print(f"Plotting class distribution from {labels_path}...")
    labels = np.load(labels_path)
    
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Benign', 'Malware']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=['#2ecc71', '#e74c3c'])
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/len(labels)*100:.1f}%)',
                 ha='center', va='bottom')
        
    plt.title('Class Distribution (Training Set)')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    print("Saved class_distribution.png")

def plot_sample_images(cnn_path, labels_path, output_dir, samples_per_class=5):
    print(f"Plotting sample images from {cnn_path}...")
    
    # Use mmap_mode='r' for large files
    try:
        images = np.load(cnn_path, mmap_mode='r')
        labels = np.load(labels_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    benign_indices = np.where(labels == 0)[0]
    malware_indices = np.where(labels == 1)[0]
    
    # Randomly select samples
    if len(benign_indices) > samples_per_class:
        sel_benign = np.random.choice(benign_indices, samples_per_class, replace=False)
    else:
        sel_benign = benign_indices
        
    if len(malware_indices) > samples_per_class:
        sel_malware = np.random.choice(malware_indices, samples_per_class, replace=False)
    else:
        sel_malware = malware_indices
    
    fig, axes = plt.subplots(2, samples_per_class, figsize=(15, 6))
    
    # Plot Benign
    for i, idx in enumerate(sel_benign):
        img = images[idx, 0] # (1, H, W) -> (H, W)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Benign Sample {i+1}')
        axes[0, i].axis('off')
        
    # Plot Malware
    for i, idx in enumerate(sel_malware):
        img = images[idx, 0]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Malware Sample {i+1}')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_samples.png')
    plt.close()
    print("Saved dataset_samples.png")

def plot_sequence_lengths(lstm_path, labels_path, output_dir):
    print(f"Plotting sequence lengths from {lstm_path}...")
    
    try:
        sequences = np.load(lstm_path) # Integers, should fit in memory (12MB)
        labels = np.load(labels_path)
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return

    # Calculate non-padding length (assuming 0 is padding)
    # This might be slow if loop, so vectorise:
    # shape: (N, L)
    # non_zeros = (sequences != 0).sum(axis=1)
    
    # Actually, let's just plot the distribution of non-zero values per sample
    lengths = np.count_nonzero(sequences, axis=1)
    
    # Filter outliers for better visualization (99th percentile)
    limit = np.percentile(lengths, 99)
    filtered_indices = lengths <= limit
    
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        'Length': lengths[filtered_indices],
        'Class': ['Malware' if l==1 else 'Benign' for l in labels[filtered_indices]]
    })
    
    # Use Seaborn histplot with KDE
    sns.histplot(data=df, x='Length', hue='Class', element='step', stat='density', 
                 common_norm=False, kde=True, 
                 palette={'Malware': '#e74c3c', 'Benign': '#2ecc71'})
    
    plt.title(f'Sequence Length Distribution (Capped at 99th Percentile: {int(limit)})')
    plt.xlabel('Sequence Length (Non-padding API calls)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sequence_lengths.png')
    plt.close()
    print("Saved sequence_lengths.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    plot_class_distribution(data_path / 'train_labels.npy', output_dir)
    
    # 2. Sample Images
    plot_sample_images(data_path / 'train_cnn.npy', data_path / 'train_labels.npy', output_dir)
    
    # 3. Sequence Lengths
    plot_sequence_lengths(data_path / 'train_lstm.npy', data_path / 'train_labels.npy', output_dir)
    
    print("Dataset visualization completed.")

if __name__ == "__main__":
    main()
