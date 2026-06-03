import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import shutil
import random
from pathlib import Path
from prototype.backend.config import settings
from models.dual_branch_model import DualBranchModel
from utils.mutator import ThreatMutator

def calculate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for cnn, lstm, lbl in loader:
            cnn, lstm = cnn.to(device), lstm.to(device)
            outputs = model(cnn, lstm)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, recall

def run_autonomous_evolution(selected_strategy=None, epochs=5, log_callback=None, quick=False):
    def log(msg, error=False):
        if log_callback:
            log_callback(msg, error)
        else:
            print(msg)

    log("="*60)
    log("AUTONOMOUS SECTION 6: ADVERSARIAL HARDENING LOOP")
    log("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data/custom_dataset")
    
    # 1. LOAD BASE DATASET (2k + 2k)
    if quick:
        log("[1/6] Loading Base Dataset (QUICK Mode: 100 samples)...")
    else:
        log("[1/6] Loading Base Dataset (4000 samples)...")
        
    cnn_base = np.load(data_dir / 'train_cnn.npy')
    lstm_base = np.load(data_dir / 'train_lstm.npy')
    labels_base = np.load(data_dir / 'train_labels.npy')
    
    if quick:
        cnn_base = cnn_base[:100]
        lstm_base = lstm_base[:100]
        labels_base = labels_base[:100]
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 2. BASELINE MODEL (Current Best Path)
    log(f"[2/6] Evaluating Current Best Path: {settings.CHECKPOINT_PATH.name}")
    checkpoint = torch.load(settings.CHECKPOINT_PATH, map_location='cpu')
    
    lstm_config = {
        'vocab_size': metadata['vocab_size'],
        'input_size': 100, 'hidden_size': 128, 'num_layers': 2, 'feature_dim': 256, 'bidirectional': True
    }
    model = DualBranchModel(lstm_config=lstm_config)
    if checkpoint.get('is_expanded'): model.expand_capacity()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Use a fixed 20% validation set for comparison
    val_indices = list(range(len(labels_base)))[-800:]
    val_loader = DataLoader(TensorDataset(
        torch.FloatTensor(cnn_base[val_indices]) / 255.0, 
        torch.LongTensor(lstm_base[val_indices]), 
        torch.LongTensor(labels_base[val_indices])
    ), batch_size=32)
    
    base_acc, base_rec = calculate_metrics(model, val_loader, device)
    log(f"  > Current Accuracy: {base_acc:.4f}, Recall: {base_rec:.4f}")

    # 3. DATA MUTATOR (Autonomous Adversarial Generation)
    log(f"[3/6] Generating Adversarial Mutations (Strategy: {selected_strategy or 'Random Mixed'})...")
    mutator = ThreatMutator(vocab_size=metadata['vocab_size'])
    
    # Pick a random 25% of malware samples to mutate
    malware_indices = np.where(labels_base == 1)[0]
    mutation_indices = random.sample(list(malware_indices), k=int(len(malware_indices) * 0.25))
    
    aug_cnn, aug_lstm, aug_labels = list(cnn_base), list(lstm_base), list(labels_base)
    
    default_strategies = ['random_insert', 'noise_append', 'random_replace']
    for idx in mutation_indices:
        lstm_t = torch.LongTensor(lstm_base[idx]).unsqueeze(0)
        # Apply the specific strategy or a random one
        strat = selected_strategy if selected_strategy in default_strategies else random.choice(default_strategies)
        mutated = mutator.mutate_batch(lstm_t, mutation_rate=0.2, strategy=strat)
        
        aug_cnn.append(cnn_base[idx])
        aug_lstm.append(mutated.squeeze(0).numpy())
        aug_labels.append(1) # Still malware
        
    log(f"  > Generated {len(mutation_indices)} adversarial variants.")

    # 4. RESUME TRAINING (Candidate Model)
    log("[4/6] Resume Training: Hardening Candidate Model...")
    # Use the same model weights but train on augmented data
    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(np.array(aug_cnn)) / 255.0, 
        torch.LongTensor(np.array(aug_lstm)), 
        torch.LongTensor(np.array(aug_labels))
    ), batch_size=16, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for c_b, l_b, lbl_b in train_loader:
            c_b, l_b, lbl_b = c_b.to(device), l_b.to(device), lbl_b.to(device)
            optimizer.zero_grad()
            outputs = model(c_b, l_b)
            loss = criterion(outputs, lbl_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        log(f"  > Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # 5. PERFORMANCE COMPARISON
    log("[5/6] Model Evaluation & Comparison...")
    cand_acc, cand_rec = calculate_metrics(model, val_loader, device)
    log(f"  > Candidate Accuracy: {cand_acc:.4f}, Recall: {cand_rec:.4f}")
    
    is_better = (cand_acc > base_acc) or (cand_acc == base_acc and cand_rec > base_rec)

    # 6. UPDATE BEST PATH
    if is_better:
        log("  [YES] Evolution Successful. Candidate is better than Baseline.")
        shutil.copy(settings.CHECKPOINT_PATH, "adaptive_model_backup.pth")
        new_checkpoint = checkpoint.copy()
        new_checkpoint['model_state_dict'] = model.state_dict()
        new_checkpoint['is_hardened'] = True
        torch.save(new_checkpoint, settings.CHECKPOINT_PATH)
        log("[SUCCESS] Section 6 Cycle Complete: Best Path Deployed & Updated.")
    else:
        log("  [NO] Evolution Stagnated. Discarding candidate.")
        log("[SUCCESS] Section 6 Cycle Complete: Baseline Maintained.")
        
    # Write stats to JSON for backend / client UI consumption
    import json
    results = {
        "base_accuracy": float(base_acc),
        "base_recall": float(base_rec),
        "candidate_accuracy": float(cand_acc),
        "candidate_recall": float(cand_rec),
        "evolution_successful": bool(is_better),
        "strategy": selected_strategy or "random_mixed",
        "epochs": epochs
    }
    try:
        with open("model_evolution_results.json", "w") as f:
            json.dump(results, f, indent=2)
        log("  [OK] Saved evolution results to model_evolution_results.json")
    except Exception as e:
        log(f"  [ERROR] Failed to save evolution results: {e}", error=True)
        
    return is_better, (base_acc, base_rec), (cand_acc, cand_rec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_autonomous_evolution(selected_strategy=args.strategy, epochs=args.epochs, quick=args.quick)
