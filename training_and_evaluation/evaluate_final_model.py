import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from models.dual_branch_model import DualBranchModel

def evaluate_model():
    data_dir = Path("data/custom_dataset")
    print("Loading dataset for evaluation...")
    cnn_data = np.load(data_dir / 'train_cnn.npy')
    lstm_data = np.load(data_dir / 'train_lstm.npy')
    labels = np.load(data_dir / 'train_labels.npy')
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 1. Initialize and Load Final Model
    print("Loading final adaptive model...")
    checkpoint = torch.load("adaptive_model_final.pth", map_location='cpu')
    
    lstm_config = {
        'vocab_size': metadata['vocab_size'],
        'input_size': 100,
        'hidden_size': 128,
        'num_layers': 2,
        'feature_dim': 256,
        'bidirectional': True
    }
    model = DualBranchModel(lstm_config=lstm_config)
    
    # Expand if necessary
    if checkpoint.get('is_expanded'):
        model.expand_capacity()
        
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Prepare Data (Val Split)
    dataset = TensorDataset(torch.FloatTensor(cnn_data), torch.LongTensor(lstm_data), torch.LongTensor(labels))
    # Using 20% for validation (consistent with training)
    val_size = int(0.2 * len(dataset))
    _, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 3. Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for cnn_batch, lstm_batch, labels_batch in val_loader:
            cnn_batch, lstm_batch = cnn_batch.to(device), lstm_batch.to(device)
            outputs = model(cnn_batch, lstm_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    # 4. Metrics Calculation
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "="*40)
    print("FINAL MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"Total Samples Evaluated: {len(all_labels)}")
    print("-" * 40)
    print(f"True Negatives (Benign ✅): {tn}")
    print(f"False Positives (Alarm ❌):  {fp}")
    print(f"False Negatives (Missed ⚠️): {fn}")
    print(f"True Positives (Malware 🛡️): {tp}")
    print("-" * 40)
    print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"Overall Accuracy: {(tp+tn)/len(all_labels):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malware']))

if __name__ == "__main__":
    evaluate_model()
