"""
Robustness Evaluation Script
Evaluates model checkpoints against:
  1. Base test set  (original)
  2. random_insert  mutated test set
  3. noise_append   mutated test set
  4. random_replace mutated test set

Produces a full N-models x 4-datasets comparison matrix.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

sys.path.append(str(Path(__file__).parent.parent))

from models import DualBranchModel
from utils.mutator import ThreatMutator

# ── Mutation rate used to generate adversarial test sets ──────────────────────
MUTATION_RATE = 0.2


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> DualBranchModel:
    """Load a DualBranchModel from a checkpoint, handling expanded architectures."""
    ck = torch.load(checkpoint_path, map_location=device)
    mc = ck.get('model_config', {})
    meta = ck.get('metadata')

    if meta is None:
        raise RuntimeError(f"No metadata in checkpoint: {checkpoint_path}")

    model = DualBranchModel(
        cnn_config={
            'input_channels': 1,
            'num_classes': 2,
            'feature_dim': mc.get('cnn_feature_dim', 256)
        },
        lstm_config={
            'vocab_size': meta['vocab_size'],
            'embedding_dim': mc.get('embedding_dim', 100),
            'hidden_size': mc.get('lstm_hidden_size', 128),
            'num_layers': mc.get('lstm_num_layers', 2),
            'feature_dim': mc.get('lstm_feature_dim', 256),
            'bidirectional': True
        },
        num_classes=2,
        fusion_method=mc.get('fusion_method', 'concat')
    )

    if ck.get('is_expanded', False):
        model.expand_capacity()

    model.load_state_dict(ck['model_state_dict'])
    model.to(device)
    model.eval()
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_arrays(model, cnn_arr, lstm_arr, labels_arr, device,
                       batch_size=32, threshold=0.5):
    """Run inference on raw numpy arrays and return metrics dict."""
    all_preds, all_labels, all_probs = [], [], []

    n = len(labels_arr)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cnn_b  = torch.FloatTensor(cnn_arr[start:end])  / 255.0
        lstm_b = torch.LongTensor(lstm_arr[start:end])
        lbl_b  = labels_arr[start:end]

        with torch.no_grad():
            out   = model(cnn_b.to(device), lstm_b.to(device))
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        preds = (probs >= threshold).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(lbl_b.tolist())
        all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    return {
        'accuracy':  float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall':    float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1_score':  float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc':       float(roc_auc_score(all_labels, all_probs)),
        'n_samples': int(n),
        'n_malware': int(np.sum(all_labels == 1)),
        'n_benign':  int(np.sum(all_labels == 0)),
        'tp': int(np.sum((all_preds==1) & (all_labels==1))),
        'fp': int(np.sum((all_preds==1) & (all_labels==0))),
        'tn': int(np.sum((all_preds==0) & (all_labels==0))),
        'fn': int(np.sum((all_preds==0) & (all_labels==1))),
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_mutated_arrays(cnn_arr, lstm_arr, labels_arr,
                         strategy, vocab_size, mutation_rate=MUTATION_RATE):
    """Apply mutation ONLY to malware samples (label=1); benign stays clean."""
    mutator   = ThreatMutator(vocab_size=vocab_size)
    lstm_mut  = lstm_arr.copy()

    malware_idx = np.where(labels_arr == 1)[0]
    lstm_tensor = torch.LongTensor(lstm_arr[malware_idx])
    mutated     = mutator.mutate_batch(lstm_tensor, mutation_rate=mutation_rate,
                                       strategy=strategy)
    lstm_mut[malware_idx] = mutated.numpy()

    return cnn_arr, lstm_mut, labels_arr


# ─────────────────────────────────────────────────────────────────────────────
def print_row(label, m):
    print(f"  {label:<18} | Acc={m['accuracy']*100:5.2f}%  "
          f"Prec={m['precision']*100:5.2f}%  "
          f"Rec={m['recall']*100:5.2f}%  "
          f"F1={m['f1_score']*100:5.2f}%  "
          f"AUC={m['auc']*100:5.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = Path(args.data_dir)

    # ── Load base test set ────────────────────────────────────────────────
    print(f"\nLoading test data from {data_path} ...")
    test_cnn    = np.load(data_path / 'test_cnn.npy')
    test_lstm   = np.load(data_path / 'test_lstm.npy')
    test_labels = np.load(data_path / 'test_labels.npy')

    with open(data_path / 'metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']

    print(f"  Test samples: {len(test_labels)}  "
          f"(Benign={np.sum(test_labels==0)}, Malware={np.sum(test_labels==1)})")

    # ── Pre-build mutated test sets ───────────────────────────────────────
    STRATEGIES = ['random_insert', 'noise_append', 'random_replace']
    print(f"\nGenerating mutated test sets (rate={MUTATION_RATE}) ...")
    mutated_sets = {}
    for strat in STRATEGIES:
        c, l, y = build_mutated_arrays(test_cnn, test_lstm, test_labels,
                                        strat, vocab_size)
        mutated_sets[strat] = (c, l, y)
        print(f"  {strat}: done")

    datasets = {'base': (test_cnn, test_lstm, test_labels), **mutated_sets}

    # ── Evaluate each checkpoint ──────────────────────────────────────────
    checkpoints = {
        'baseline':       args.baseline,
        'random_insert':  args.random_insert,
        'noise_append':   args.noise_append,
        'random_replace': args.random_replace,
    }
    # Filter out missing / 'None' string / non-existent paths
    checkpoints = {
        k: v for k, v in checkpoints.items()
        if v and v.lower() != 'none' and Path(v).exists()
    }
    if not checkpoints:
        print("ERROR: No valid checkpoint paths provided.")
        return

    all_results = {}

    for model_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}  ({ckpt_path})")
        print(f"{'='*70}")
        print(f"  {'Dataset':<18} | Acc       Prec      Rec       F1        AUC")
        print(f"  {'-'*65}")

        model, _ = load_model(ckpt_path, device)
        model_results = {}

        for ds_name, (c, l, y) in datasets.items():
            m = evaluate_on_arrays(model, c, l, y, device,
                                   batch_size=args.batch_size,
                                   threshold=args.threshold)
            model_results[ds_name] = m
            print_row(ds_name, m)

        all_results[model_name] = model_results
        del model  # free memory

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — Accuracy across all models and datasets")
    print(f"{'='*70}")
    header = f"{'Model':<20}" + "".join(f"  {d:<18}" for d in datasets)
    print(header)
    print("-" * len(header))
    for model_name, model_results in all_results.items():
        row = f"{model_name:<20}"
        for ds_name in datasets:
            acc = model_results[ds_name]['accuracy'] * 100
            row += f"  {acc:5.2f}%            "
        print(row)

    # ── Save results ──────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[OK] Full results saved to {out_path}")

    print(f"\n{'='*70}")
    print("Robustness evaluation complete!")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Robustness evaluation: model x dataset matrix'
    )
    parser.add_argument('--data-dir', type=str, default='data/custom_dataset',
                        help='Directory with test_cnn.npy, test_lstm.npy, test_labels.npy')
    parser.add_argument('--baseline', type=str,
                        default='checkpoints/baseline_concat.pth',
                        help='Path to baseline (untouched) checkpoint')
    parser.add_argument('--random-insert', type=str,
                        default='checkpoints/mutation_random_insert.pth',
                        help='Path to random_insert-trained checkpoint')
    parser.add_argument('--noise-append', type=str,
                        default='checkpoints/mutation_noise_append.pth',
                        help='Path to noise_append-trained checkpoint')
    parser.add_argument('--random-replace', type=str,
                        default='checkpoints/mutation_random_replace.pth',
                        help='Path to random_replace-trained checkpoint')
    parser.add_argument('--output', type=str,
                        default='checkpoints/robustness_results.json',
                        help='Path to save full results JSON')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Mutation rate for test set generation')

    args = parser.parse_args()
    MUTATION_RATE = args.mutation_rate
    main(args)
