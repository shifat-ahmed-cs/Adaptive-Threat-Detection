import json
import torch
from pathlib import Path
from models import DualBranchModel
from prototype.backend.config import settings

def debug():
    device = torch.device('cpu')
    print("Loading model...")
    checkpoint = torch.load(settings.CHECKPOINT_PATH, map_location=device)
    model_config = checkpoint.get('model_config', {})
    metadata = checkpoint.get('metadata', {})
    
    model = DualBranchModel(
        cnn_config={'input_channels': 1, 'num_classes': 2, 'feature_dim': model_config.get('cnn_feature_dim', 256)},
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
    model.eval()
    
    samples_dir = Path("prototype/samples/kaggle_extracted")
    
    for f in samples_dir.glob("*.json"):
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        cnn_t = torch.FloatTensor(data['cnn_input']).unsqueeze(0)
        lstm_t = torch.LongTensor(data['lstm_input']).unsqueeze(0)
        
        with torch.no_grad():
            out = model(cnn_t, lstm_t)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
        print(f"Sample: {f.name}")
        print(f"  True label: {data['true_label']}")
        print(f"  Probs: {probs.tolist()[0]}")
        print(f"  Pred: {'Malware' if pred == 1 else 'Benign'}")
        
if __name__ == "__main__":
    debug()
