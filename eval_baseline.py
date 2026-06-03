"""
eval_baseline.py — Manual evaluation of adaptive_model_final.pth
Runs against the full custom_dataset (train_cnn.npy / train_lstm.npy / train_labels.npy)
using an 80/20 train-val split (same as continuous_learning.py uses).

Usage:
    python eval_baseline.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.dual_branch_model import DualBranchModel
from prototype.backend.config import settings

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt(v, width=12):
    return f"{v:.4f}".rjust(width)

def fmt_pct(v, width=12):
    return f"{v*100:.2f}%".rjust(width)

def compute_metrics(model, loader, device, label=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for cnn_b, lstm_b, lbl_b in loader:
            cnn_b, lstm_b = cnn_b.to(device), lstm_b.to(device)
            logits = model(cnn_b, lstm_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl_b.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    # Per-class metrics
    results = {}
    for cls_idx, cls_name in [(0, "Benign"), (1, "Malware")]:
        tp = int(np.sum((preds == cls_idx) & (labels == cls_idx)))
        fp = int(np.sum((preds == cls_idx) & (labels != cls_idx)))
        fn = int(np.sum((preds != cls_idx) & (labels == cls_idx)))
        tn = int(np.sum((preds != cls_idx) & (labels != cls_idx)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cls_name] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                  precision=precision, recall=recall, f1=f1)

    accuracy  = float(np.mean(preds == labels))
    macro_f1  = (results["Benign"]["f1"] + results["Malware"]["f1"]) / 2

    # False-positive / false-negative rates
    fpr = results["Malware"]["fp"] / (results["Malware"]["fp"] + results["Malware"]["tn"]) \
          if (results["Malware"]["fp"] + results["Malware"]["tn"]) > 0 else 0.0
    fnr = results["Malware"]["fn"] / (results["Malware"]["fn"] + results["Malware"]["tp"]) \
          if (results["Malware"]["fn"] + results["Malware"]["tp"]) > 0 else 0.0

    return dict(accuracy=accuracy, macro_f1=macro_f1, fpr=fpr, fnr=fnr,
                per_class=results, n_samples=len(labels))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    CHECKPOINT = ROOT / "adaptive_model_final.pth"
    DATA_DIR   = ROOT / "data" / "custom_dataset"

    print("=" * 65)
    print("  BASELINE MODEL EVALUATION — adaptive_model_final.pth")
    print("=" * 65)
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Data dir   : {DATA_DIR}")
    print()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("[1/4] Loading dataset...")
    cnn_data    = np.load(DATA_DIR / "train_cnn.npy")
    lstm_data   = np.load(DATA_DIR / "train_lstm.npy")
    labels      = np.load(DATA_DIR / "train_labels.npy")

    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    n_total  = len(labels)
    benign_total  = int(np.sum(labels == 0))
    malware_total = int(np.sum(labels == 1))
    print(f"  Total samples : {n_total:,}  (Benign={benign_total:,}, Malware={malware_total:,})")

    # Stratified 20% val split — SAME seed and method as continuous_learning.py
    # (Dataset is class-ordered so a naive tail-slice gives zero malware in val set)
    all_idx = list(range(n_total))
    train_idx, val_idx_list = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=labels
    )
    val_benign  = int(np.sum(labels[val_idx_list] == 0))
    val_malware = int(np.sum(labels[val_idx_list] == 1))
    print(f"  Stratified val: {len(val_idx_list):,} samples  (Benign={val_benign}, Malware={val_malware})")

    # ── 2. Load model ────────────────────────────────────────────────────────
    print()
    print("[2/4] Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    ckpt_keys = [k for k in checkpoint.keys() if k != "model_state_dict"]
    for k in ckpt_keys:
        v = checkpoint[k]
        if not hasattr(v, "__len__") or len(str(v)) < 80:
            print(f"  {k:25s}: {v}")

    lstm_config = {
        "vocab_size":   metadata["vocab_size"],
        "input_size":   100,
        "hidden_size":  128,
        "num_layers":   2,
        "feature_dim":  256,
        "bidirectional": True,
    }
    model = DualBranchModel(lstm_config=lstm_config)

    if checkpoint.get("is_expanded"):
        print("  [!] Model has expanded capacity — calling expand_capacity()")
        model.expand_capacity()

    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    print()
    print("[3/4] Running evaluation on stratified validation set (20%, seed=42)...")

    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(cnn_data[val_idx_list]) / 255.0,
            torch.LongTensor(lstm_data[val_idx_list]),
            torch.LongTensor(labels[val_idx_list]),
        ),
        batch_size=64,
        shuffle=False,
    )

    val_metrics = compute_metrics(model, val_loader, device, label="Validation")

    # Also run on the FULL dataset for a complete picture
    print("[4/4] Running evaluation on FULL dataset...")
    full_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(cnn_data) / 255.0,
            torch.LongTensor(lstm_data),
            torch.LongTensor(labels),
        ),
        batch_size=64,
        shuffle=False,
    )
    full_metrics = compute_metrics(model, full_loader, device, label="Full")

    # ── 4. Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)

    for split_name, m in [("Validation (20%)", val_metrics), ("Full Dataset", full_metrics)]:
        print()
        print(f"  -- {split_name} ({m['n_samples']:,} samples) --")
        print(f"  {'Accuracy':30s}: {m['accuracy']*100:.4f}%")
        print(f"  {'Macro F1':30s}: {m['macro_f1']*100:.4f}%")
        print(f"  {'False Positive Rate (FPR)':30s}: {m['fpr']*100:.4f}%")
        print(f"  {'False Negative Rate (FNR)':30s}: {m['fnr']*100:.4f}%")
        print()
        print(f"  {'Class':12s} {'Precision':>12} {'Recall':>12} {'F1':>12} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}")
        print(f"  {'-'*72}")
        for cls, cm in m["per_class"].items():
            print(
                f"  {cls:12s}"
                f"{fmt_pct(cm['precision']):>12}"
                f"{fmt_pct(cm['recall']):>12}"
                f"{fmt_pct(cm['f1']):>12}"
                f"{cm['tp']:>7}"
                f"{cm['fp']:>7}"
                f"{cm['fn']:>7}"
                f"{cm['tn']:>7}"
            )

    # ── Summary line (for easy copy-paste into GUI / README) ─────────────────
    vm = val_metrics
    print()
    print("=" * 65)
    print("  SUMMARY (Validation Set — matches continuous_learning.py)")
    print("=" * 65)
    print(f"  Base Accuracy      : {vm['accuracy']*100:.4f}%")
    print(f"  Malware Recall     : {vm['per_class']['Malware']['recall']*100:.4f}%")
    print(f"  Malware Precision  : {vm['per_class']['Malware']['precision']*100:.4f}%")
    print(f"  Macro F1           : {vm['macro_f1']*100:.4f}%")
    print(f"  FPR                : {vm['fpr']*100:.4f}%")
    print(f"  FNR                : {vm['fnr']*100:.4f}%")
    print()
    print("  These are the true baseline values to use in your thesis and GUI.")
    print("=" * 65)


if __name__ == "__main__":
    main()
