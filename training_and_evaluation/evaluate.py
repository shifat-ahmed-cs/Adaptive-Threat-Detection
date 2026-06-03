"""
Evaluation script for dual-branch malware detection model
Provides comprehensive metrics and visualizations
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from tqdm import tqdm

from models import DualBranchModel
from utils.data_loader import MalwareDataset
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, device, threshold=0.5):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            cnn_input = batch['cnn_input'].to(device)
            lstm_input = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(cnn_input, lstm_input)
            probs = torch.softmax(outputs, dim=1)
            
            # Use threshold for prediction instead of argmax
            malware_probs = probs[:, 1]
            predicted = (malware_probs >= threshold).long()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
            
    # Calculate and print probability stats
    all_probs_np = np.array(all_probs)
    stats = {
        'min': float(np.min(all_probs_np)),
        'max': float(np.max(all_probs_np)),
        'mean': float(np.mean(all_probs_np)),
        'median': float(np.median(all_probs_np)),
        'p25': float(np.percentile(all_probs_np, 25)),
        'p75': float(np.percentile(all_probs_np, 75))
    }
    
    print(f'\n{"="*60}')
    print('PROBABILITY DISTRIBUTION STATISTICS (Malware Class)')
    print(f'{"="*60}')
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"25th Percentile: {stats['p25']:.4f}")
    print(f"75th Percentile: {stats['p75']:.4f}")
    print(f'{"="*60}\n')
    
    # Save stats to file
    with open('checkpoints/prob_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(labels, preds, probs):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
        'f1_score': f1_score(labels, preds, average='binary', zero_division=0),
        'auc': roc_auc_score(labels, probs),
    }
    
    # Per-class metrics
    cm = confusion_matrix(labels, preds)
    # Handle cases where confusion matrix is not 2x2 (e.g. only one class predicted)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback logic if needed, but usually for binary classification with standard labels 0/1 it should be 2x2 
        # unless one class is completely missing from labels (which shouldn't happen here as we have both)
        # But if predictions are all 1, we still get 2 columns if labels have both.
        # If labels only have 0s or only 1s, then shape might differ.
        # Our test set has both.
        tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Per-class precision and recall
    metrics['benign_precision'] = precision_score(labels, preds, pos_label=0, zero_division=0)
    metrics['benign_recall'] = recall_score(labels, preds, pos_label=0, zero_division=0)
    metrics['malware_precision'] = precision_score(labels, preds, pos_label=1, zero_division=0)
    metrics['malware_recall'] = recall_score(labels, preds, pos_label=1, zero_division=0)
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print(f'\n{"="*60}')
    print('TEST SET EVALUATION METRICS')
    print(f'{"="*60}')
    print(f'Accuracy:    {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.2f}%)')
    print(f'Precision:   {metrics["precision"]:.4f}')
    print(f'Recall:      {metrics["recall"]:.4f}')
    print(f'F1-Score:    {metrics["f1_score"]:.4f}')
    print(f'AUC:         {metrics["auc"]:.4f}')
    print(f'Specificity: {metrics["specificity"]:.4f}')
    print(f'Sensitivity: {metrics["sensitivity"]:.4f}')
    
    print(f'\n{"="*60}')
    print('PER-CLASS METRICS')
    print(f'{"="*60}')
    print(f'Benign:')
    print(f'  Precision: {metrics["benign_precision"]:.4f}')
    print(f'  Recall:    {metrics["benign_recall"]:.4f}')
    print(f'Malware:')
    print(f'  Precision: {metrics["malware_precision"]:.4f}')
    print(f'  Recall:    {metrics["malware_recall"]:.4f}')
    
    print(f'\n{"="*60}')
    print('CONFUSION MATRIX')
    print(f'{"="*60}')
    print(f'True Negatives:  {metrics["true_negatives"]}')
    print(f'False Positives: {metrics["false_positives"]}')
    print(f'False Negatives: {metrics["false_negatives"]}')
    print(f'True Positives:  {metrics["true_positives"]}')
    print(f'{"="*60}')


def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'\n{"="*60}')
    print(f'Dual-Branch Malware Detection Evaluation')
    print(f'{"="*60}')
    print(f'Device: {device}')
    
    # Load checkpoint
    print(f'\nLoading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load metadata
    metadata = checkpoint.get('metadata')
    if metadata is None:
        # Try loading from data directory
        data_path = Path(args.data_dir)
        with open(data_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    
    model_config = checkpoint.get('model_config', {})
    
    print(f'Checkpoint epoch: {checkpoint.get("epoch", "N/A")}')
    print(f'Validation accuracy: {checkpoint.get("val_acc", "N/A"):.2f}%')
    print(f'Validation loss: {checkpoint.get("val_loss", "N/A"):.4f}')
    
    # Load test data
    print(f'\nLoading test data from {args.data_dir}...')
    data_path = Path(args.data_dir)
    
    test_cnn = np.load(data_path / 'test_cnn.npy')
    test_lstm = np.load(data_path / 'test_lstm.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    print(f'Test samples: {len(test_labels)}')
    print(f'  Benign: {np.sum(test_labels==0)}')
    print(f'  Malware: {np.sum(test_labels==1)}')
    
    # Create test loader
    test_dataset = MalwareDataset(test_cnn, test_lstm, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print(f'\nInitializing model...')
    model = DualBranchModel(
        cnn_config={
            'input_channels': 1,
            'num_classes': 2,
            'feature_dim': model_config.get('cnn_feature_dim', 256)
        },
        lstm_config={
            'vocab_size': metadata['vocab_size'],
            'embedding_dim': model_config.get('embedding_dim', 128),
            'hidden_size': model_config.get('lstm_hidden_size', 128),
            'num_layers': model_config.get('lstm_num_layers', 2),
            'feature_dim': model_config.get('lstm_feature_dim', 256),
            'bidirectional': True
        },
        num_classes=2,
        fusion_method=model_config.get('fusion_method', 'concat')
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Evaluate
    print(f'\nEvaluating model on test set (Threshold: {args.threshold})...')
    preds, labels, probs = evaluate_model(model, test_loader, device, threshold=args.threshold)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, preds, probs)
    metrics['threshold'] = args.threshold
    
    # Print metrics
    print_metrics(metrics)
    
    # Print classification report
    print(f'\n{"="*60}')
    print('CLASSIFICATION REPORT')
    print(f'{"="*60}')
    print(classification_report(labels, preds, target_names=['Benign', 'Malware'], digits=4, zero_division=0))
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in metrics.items()},
            'model_config': model_config,
            'checkpoint_path': str(args.checkpoint),
            'test_samples': int(len(labels)),
            'benign_samples': int(np.sum(labels==0)),
            'malware_samples': int(np.sum(labels==1)),
            'threshold': args.threshold
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n✓ Results saved to {output_path}')
    
    print(f'\n{"="*60}')
    print('Evaluation completed!')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dual-branch malware detection model')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output', type=str, default='checkpoints/evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for classification (default: 0.5)')
    
    args = parser.parse_args()
    main(args)
