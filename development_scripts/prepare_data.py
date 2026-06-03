"""
Ultra-memory-efficient loader for Kaggle malware datasets
Processes data in small batches and saves incrementally
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle


def prepare_kaggle_data_efficient(pe_csv_path, api_csv_path, save_path='data/processed', 
                                  max_seq_length=100, image_size=128):
    """
    Memory-efficient data preparation - processes in batches and saves incrementally
    
    Note: Using smaller image size (128x128) to reduce memory usage
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Kaggle Malware Dataset Preparation (Memory-Efficient)")
    print("="*60)
    print(f"Image size: {image_size}x{image_size}")
    print(f"Sequence length: {max_seq_length}")
    
    # Step 1: Load API sequences (smaller dataset)
    print("\n[1/5] Loading API sequences...")
    api_df = pd.read_csv(api_csv_path)
    
    api_hashes = api_df['hash'].values
    api_labels = api_df['malware'].values
    
    seq_cols = [col for col in api_df.columns if col.startswith('t_')][:max_seq_length]
    api_sequences = api_df[seq_cols].values.astype(np.int32)
    
    print(f"   ✓ Loaded {len(api_df)} API samples")
    print(f"   ✓ Benign: {np.sum(api_labels==0)}, Malware: {np.sum(api_labels==1)}")
    
    # Create hash lookup
    api_hash_set = set(api_hashes)
    api_hash_to_idx = {h: i for i, h in enumerate(api_hashes)}
    
    # Step 2: Process PE images in chunks and collect matched samples
    print("\n[2/5] Processing PE images (this may take a few minutes)...")
    
    matched_data = []
    chunk_size = 1000
    total_processed = 0
    
    for chunk in tqdm(pd.read_csv(pe_csv_path, chunksize=chunk_size), desc="   Processing"):
        chunk_hashes = chunk['hash'].values
        pix_cols = [col for col in chunk.columns if col.startswith('pix_')]
        
        for idx, h in enumerate(chunk_hashes):
            if h in api_hash_set:
                # Get pixels and reshape
                pixels = chunk[pix_cols].iloc[idx].values.astype(np.uint8)
                
                # Resize to target size
                num_pixels = image_size * image_size
                if len(pixels) > num_pixels:
                    pixels = pixels[:num_pixels]
                else:
                    pixels = np.pad(pixels, (0, max(0, num_pixels - len(pixels))))[:num_pixels]
                
                # Store as tuple (api_idx, pixels)
                api_idx = api_hash_to_idx[h]
                matched_data.append((api_idx, pixels))
        
        total_processed += len(chunk)
        
        # Periodically save to free memory
        if len(matched_data) >= 5000:
            print(f"   Processed {total_processed} PE samples, matched {len(matched_data)} so far...")
    
    print(f"   ✓ Matched {len(matched_data)} samples out of {total_processed} processed")
    
    # Step 3: Convert to arrays
    print("\n[3/5] Creating final arrays...")
    
    api_indices = [item[0] for item in matched_data]
    
    # Process images in batches to avoid memory issues
    num_samples = len(matched_data)
    pe_images = np.zeros((num_samples, 1, image_size, image_size), dtype=np.float32)
    
    batch_size = 1000
    for i in tqdm(range(0, num_samples, batch_size), desc="   Converting images"):
        end_idx = min(i + batch_size, num_samples)
        for j in range(i, end_idx):
            pixels = matched_data[j][1]
            pe_images[j, 0] = pixels.reshape(image_size, image_size) / 255.0
    
    # Get corresponding API sequences and labels
    matched_api_seqs = api_sequences[api_indices]
    matched_labels = api_labels[api_indices]
    
    print(f"   ✓ PE images shape: {pe_images.shape}")
    print(f"   ✓ API sequences shape: {matched_api_seqs.shape}")
    
    # Step 4: Shuffle and split
    print("\n[4/5] Shuffling and splitting data...")
    
    indices = np.random.permutation(num_samples)
    pe_images = pe_images[indices]
    matched_api_seqs = matched_api_seqs[indices]
    matched_labels = matched_labels[indices]
    
    # Split: 70% train, 15% val, 15% test
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)
    
    # Step 5: Save data
    print("\n[5/5] Saving processed data...")
    
    np.save(save_path / 'train_cnn.npy', pe_images[:train_end])
    np.save(save_path / 'train_lstm.npy', matched_api_seqs[:train_end])
    np.save(save_path / 'train_labels.npy', matched_labels[:train_end])
    
    np.save(save_path / 'val_cnn.npy', pe_images[train_end:val_end])
    np.save(save_path / 'val_lstm.npy', matched_api_seqs[train_end:val_end])
    np.save(save_path / 'val_labels.npy', matched_labels[train_end:val_end])
    
    np.save(save_path / 'test_cnn.npy', pe_images[val_end:])
    np.save(save_path / 'test_lstm.npy', matched_api_seqs[val_end:])
    np.save(save_path / 'test_labels.npy', matched_labels[val_end:])
    
    # Save metadata
    vocab_size = int(np.max(api_sequences)) + 1
    metadata = {
        'vocab_size': vocab_size,
        'image_size': image_size,
        'seq_length': max_seq_length
    }
    with open(save_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Dataset Preparation Complete!")
    print("="*60)
    print(f"Training samples: {train_end}")
    print(f"  Benign: {np.sum(matched_labels[:train_end]==0)}, Malware: {np.sum(matched_labels[:train_end]==1)}")
    print(f"Validation samples: {val_end - train_end}")
    print(f"  Benign: {np.sum(matched_labels[train_end:val_end]==0)}, Malware: {np.sum(matched_labels[train_end:val_end]==1)}")
    print(f"Test samples: {num_samples - val_end}")
    print(f"  Benign: {np.sum(matched_labels[val_end:]==0)}, Malware: {np.sum(matched_labels[val_end:]==1)}")
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Sequence length: {max_seq_length}")
    print(f"\nData saved to: {save_path}")
    print("="*60)
    
    print("\n🚀 Ready for training!")
    print("Run: jupyter notebook notebooks/03_train_with_kaggle_data.ipynb")


if __name__ == "__main__":
    prepare_kaggle_data_efficient(
        pe_csv_path='data/raw/pe_images/raw_pe_images.csv',
        api_csv_path='data/raw/api_sequences/dynamic_api_call_sequence_per_malware_100_0_306.csv',
        save_path='data/processed',
        max_seq_length=100,  # Use actual sequence length from dataset
        image_size=128  # Reduced from 256 to save memory
    )
