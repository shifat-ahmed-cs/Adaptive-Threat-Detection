import numpy as np
import json
from pathlib import Path
import os

def export_samples():
    print("Exporting real test samples from Kaggle dataset...")
    data_path = Path("data/processed")
    
    if not data_path.exists():
        print("Data path not found!")
        return
        
    test_cnn = np.load(data_path / 'test_cnn.npy')
    test_lstm = np.load(data_path / 'test_lstm.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    samples_dir = Path("prototype/samples/kaggle_extracted")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    benign_count = 0
    malware_count = 0
    
    for i in range(len(test_labels)):
        label = test_labels[i]
        
        # We want 3 benign and 3 malware samples
        if label == 0 and benign_count < 3:
            name = f"benign_sample_{benign_count+1}.json"
            benign_count += 1
        elif label == 1 and malware_count < 3:
            name = f"malware_sample_{malware_count+1}.json"
            malware_count += 1
        else:
            continue
            
        # Create a dict with the features
        sample_data = {
            "type": "extracted_features",
            "cnn_input": test_cnn[i].tolist(),
            "lstm_input": test_lstm[i].tolist(),
            "true_label": int(label)
        }
        
        with open(samples_dir / name, 'w') as f:
            json.dump(sample_data, f)
            
        print(f"Exported {name}")
        
        if benign_count == 3 and malware_count == 3:
            break
            
    print("Done! You can now upload these .json files to the dashboard.")

if __name__ == "__main__":
    export_samples()
