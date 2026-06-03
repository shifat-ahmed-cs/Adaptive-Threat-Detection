"""
Adversarial Robustness Test
Measures the resilience of the trained model against mutated malware samples.
Baseline for Thesis Objective 2.
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import DualBranchModel
from utils.data_loader import MalwareDataset
from utils.mutator import ThreatMutator

def load_data_and_model(data_dir, checkpoint_path, batch_size, device):
    # Load metadata/checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get('metadata', {})
    model_config = checkpoint.get('model_config', {})
    
    # Load Test Data
    data_path = Path(data_dir)
    test_cnn = np.load(data_path / 'test_cnn.npy')
    test_lstm = np.load(data_path / 'test_lstm.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    # Filter for Malware only (we want to see if we can hide malware)
    malware_indices = np.where(test_labels == 1)[0]
    test_cnn = test_cnn[malware_indices]
    test_lstm = test_lstm[malware_indices]
    test_labels = test_labels[malware_indices]
    
    dataset = MalwareDataset(test_cnn, test_lstm, test_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    model = DualBranchModel(
        cnn_config={
            'input_channels': 1,
            'num_classes': 2,
            'feature_dim': model_config.get('cnn_feature_dim', 256)
        },
        lstm_config={
            'vocab_size': metadata.get('vocab_size', 307),
            'embedding_dim': model_config.get('embedding_dim', 128),
            'hidden_size': model_config.get('lstm_hidden_size', 128),
            'num_layers': model_config.get('lstm_num_layers', 2),
            'feature_dim': model_config.get('lstm_feature_dim', 256),
            'bidirectional': True
        },
        num_classes=2,
        fusion_method=model_config.get('fusion_method', 'concat')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, loader, metadata.get('vocab_size', 307)

def evaluate_robustness(model, loader, mutator, mutation_rate, device, strategy='random_insert'):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            cnn_input = batch['cnn_input'].to(device)
            lstm_input = batch['lstm_input'].to(device)
            labels = batch['label'].to(device)
            
            # Apply Mutation to LSTM input
            if mutation_rate > 0:
                lstm_input = mutator.mutate_batch(
                    lstm_input, 
                    mutation_rate=mutation_rate, 
                    strategy=strategy
                )
            
            outputs = model(cnn_input, lstm_input)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--output', default=None, help='Path to save results json')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for testing')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running robustness test on {device}")
    
    model, loader, vocab_size = load_data_and_model(args.data_dir, args.checkpoint, 32, device)
    
    # Slice dataset if limit is set
    if args.limit:
        print(f"Limiting test to {args.limit} samples")
        loader.dataset.cnn_data = loader.dataset.cnn_data[:args.limit]
        loader.dataset.lstm_data = loader.dataset.lstm_data[:args.limit]
        loader.dataset.labels = loader.dataset.labels[:args.limit]
    
    mutator = ThreatMutator(vocab_size)
    
    print("\n" + "="*50)
    print("🛡️  ADVERSARIAL COMPONENT ROBUSTNESS TEST")
    print("="*50)
    print(f"Evaluating Model on {len(loader.dataset)} MALWARE samples only.")
    print("Goal: Maintain high detection rate despite mutation.")
    print("-" * 50)
    
    results = {}
    
    # Test across different mutation rates
    rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    strategies = ['random_insert', 'random_replace', 'noise_append']
    
    for strategy in strategies:
        print(f"\nTesting Strategy: {strategy}")
        results[strategy] = {}
        for rate in rates:
            acc = evaluate_robustness(model, loader, mutator, rate, device, strategy)
            print(f"  Rate {int(rate*100)}%: \tDetection Rate = {acc:.2f}%")
            results[strategy][rate] = acc
        
    print("="*50)
    
    # Save results
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
