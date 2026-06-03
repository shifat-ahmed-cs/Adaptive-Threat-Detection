import os
import io
import json
import zipfile
import requests
import pefile
import numpy as np
import pickle
import itertools
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

def preprocess_binary_array(binary_data, target_size=(128, 128)):
    binary_array = np.frombuffer(binary_data, dtype=np.uint8)
    total_pixels = target_size[0] * target_size[1]
    
    if len(binary_array) < total_pixels:
        padded = np.zeros(total_pixels, dtype=np.uint8)
        padded[:len(binary_array)] = binary_array
        binary_array = padded
    else:
        binary_array = binary_array[:total_pixels]
        
    image = binary_array.reshape(target_size)
    image = image[np.newaxis, :, :] # (1, 128, 128)
    image = image.astype(np.float32) / 255.0
    return image

def extract_apis_from_binary(binary_data):
    try:
        pe = pefile.PE(data=binary_data)
        api_list = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name is not None:
                        api_list.append(imp.name.decode('utf-8'))
        return api_list
    except Exception:
        return None

def process_safe_feature_extraction(num_samples=2000, seq_length=100, image_size=128):
    print("="*60)
    print("Diverse Benign Extraction & Malware Reuse Pipeline")
    print("="*60)
    
    save_dir = Path("data/custom_dataset")
    
    # 1. Reuse existing Malware data if available
    malware_cnn = []
    malware_api_sequences = []
    
    if (save_dir / 'train_cnn.npy').exists() and (save_dir / 'vocab.pkl').exists():
        print("\n[1/4] Reconstructing Malware data from existing files...")
        old_cnn = np.load(save_dir / 'train_cnn.npy')
        old_lstm = np.load(save_dir / 'train_lstm.npy')
        old_labels = np.load(save_dir / 'train_labels.npy')
        with open(save_dir / 'vocab.pkl', 'rb') as f:
            old_vocab = pickle.load(f)
            
        # Reverse vocab for reconstruction
        inv_vocab = {v: k for k, v in old_vocab.items()}
        
        # Extract only malware samples
        malware_indices = np.where(old_labels == 1)[0]
        for idx in tqdm(malware_indices, desc="Reconstructing Malware"):
            malware_cnn.append(old_cnn[idx])
            # Reconstruct API names from indices
            seq = [inv_vocab.get(i, "<UNK>") for i in old_lstm[idx] if i != 0]
            malware_api_sequences.append(seq)
            
        print(f"  [OK] Successfully reused {len(malware_cnn)} Malware samples.")
    else:
        print("\n[1/4] No existing malware data found. (Aborting to avoid download as requested)")
        return

    # 2. Extract Diverse Benign Features (System32 + Program Files)
    print("\n[2/4] Extracting DIVERSE Benign Features (System + Apps)...")
    benign_dirs = [
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
        Path(r"C:\Windows\System32")
    ]
    
    benign_cnn = []
    benign_api_sequences = []
    benign_count = 0
    quota_per_dir = num_samples // len(benign_dirs)
    
    for b_dir in benign_dirs:
        if benign_count >= num_samples:
            break
        if not b_dir.exists():
            continue
            
        print(f"  Scanning: {b_dir} (Target: {quota_per_dir} samples)")
        dir_count = 0
        
        # Search recursively for .exe and .dll
        files = []
        for ext in ['*.exe', '*.dll']:
            files = itertools.chain(files, b_dir.rglob(ext))
        
        for file_path in tqdm(files, desc=f"Scanning {b_dir.name}"):
            if dir_count >= quota_per_dir or benign_count >= num_samples:
                break
            try:
                # Basic filter to avoid system driver/temp files
                if any(x in str(file_path).lower() for x in ['temp', 'cache', 'driver']):
                    continue
                    
                with open(file_path, "rb") as f:
                    binary_data = f.read()
                
                if not binary_data.startswith(b'MZ'):
                    continue
                    
                apis = extract_apis_from_binary(binary_data)
                if apis is not None and len(apis) > 0:
                    cnn = preprocess_binary_array(binary_data, (image_size, image_size))
                    benign_cnn.append(cnn)
                    benign_api_sequences.append(apis)
                    benign_count += 1
                    dir_count += 1
            except Exception:
                continue

    print(f"  [OK] Extracted {benign_count} Diverse Benign samples")

    print(f"  [OK] Extracted {benign_count} Diverse Benign samples")

    # 3. Build New Unified Vocabulary
    print("\n[3/4] Building NEW Unified API Vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab_idx = 2
    
    # Combine sequences
    all_sequences = malware_api_sequences + benign_api_sequences
    all_cnn = malware_cnn + benign_cnn
    all_labels = [1]*len(malware_api_sequences) + [0]*len(benign_api_sequences)
    
    for apis in tqdm(all_sequences, desc="Building Vocab"):
        for api in apis:
            if api not in vocab:
                vocab[api] = vocab_idx
                vocab_idx += 1
                
    print(f"  [OK] New Vocabulary size: {len(vocab)}")
    
    # 4. Map Sequences & Save Dataset
    print("\n[4/4] Mapping Sequences and Saving Dataset...")
    final_lstm = np.zeros((len(all_sequences), seq_length), dtype=np.int64)
    
    for i, apis in enumerate(all_sequences):
        for j, api in enumerate(apis[:seq_length]):
            final_lstm[i, j] = vocab.get(api, 1)
            
    final_cnn = np.array(all_cnn, dtype=np.float32)
    final_labels = np.array(all_labels, dtype=np.int64)
    
    np.save(save_dir / 'train_cnn.npy', final_cnn)
    np.save(save_dir / 'train_lstm.npy', final_lstm)
    np.save(save_dir / 'train_labels.npy', final_labels)
    
    with open(save_dir / 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    metadata = {
        'vocab_size': len(vocab),
        'image_size': image_size,
        'seq_length': seq_length
    }
    with open(save_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"  [OK] Saved arrays to {save_dir}")
    print("\nDone! Your model now understands the difference between system files, user apps, and malware.")

if __name__ == "__main__":
    process_safe_feature_extraction(num_samples=2000)
