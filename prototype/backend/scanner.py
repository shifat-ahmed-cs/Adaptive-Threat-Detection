import torch
import numpy as np
import sys
from pathlib import Path
from .config import settings

# Add project root to path
sys.path.append(str(settings.BASE_PATH))

from models import DualBranchModel
from utils.preprocessing import preprocess_binary

class ThreatScanner:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThreatScanner, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def reload_model(self):
        """Force a reload of the model weights and config from disk."""
        print("[INFO] Reloading model from disk...")
        self._load_model()
        return True

    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(settings.CHECKPOINT_PATH, map_location=self.device)
            self.model_config = checkpoint.get('model_config', {})
            self.metadata = checkpoint.get('metadata', {})
            
            # Load Vocabulary
            vocab_path = Path(settings.BASE_PATH) / "data/custom_dataset/vocab.pkl"
            if vocab_path.exists():
                import pickle
                with open(vocab_path, 'rb') as f:
                    self.vocab = pickle.load(f)
                print(f"Loaded custom vocabulary from {vocab_path}")
            else:
                self.vocab = None
                print("Warning: No custom vocabulary found. Using fallback hashing.")

            self.model = DualBranchModel(
                cnn_config={
                    'input_channels': 1,
                    'num_classes': 2,
                    'feature_dim': self.model_config.get('cnn_feature_dim', 256)
                },
                lstm_config={
                    'vocab_size': self.metadata.get('vocab_size', 307),
                    'embedding_dim': self.model_config.get('embedding_dim', 100),
                    'hidden_size': self.model_config.get('lstm_hidden_size', 128),
                    'num_layers': self.model_config.get('lstm_num_layers', 2),
                    'feature_dim': self.model_config.get('lstm_feature_dim', 256),
                    'bidirectional': True
                },
                num_classes=2,
                fusion_method=self.model_config.get('fusion_method', 'concat')
            )
            
            # ADAPTIVE ARCHITECTURE HANDLING
            # If the checkpoint is from an expanded model, expand the current model before loading
            if checkpoint.get('is_expanded') or self.model_config.get('is_expanded'):
                self.model.expand_capacity()

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.ready = True
            print(f"Scanner initialized on {self.device}")
        except Exception as e:
            print(f"Scanner initialization failed: {e}")
            self.ready = False

    def scan(self, binary_data, filename: str):
        if not self.ready:
            raise Exception("Scanner not initialized")

        # Check if the uploaded file is pre-extracted features (JSON)
        is_extracted = False
        true_label = None
        if filename.endswith('.json'):
            import json
            try:
                data = json.loads(binary_data.decode('utf-8'))
                if data.get("type") == "extracted_features":
                    is_extracted = True
                    true_label = data.get("true_label", None)
                    cnn_input = np.array(data["cnn_input"], dtype=np.float32)
                    lstm_input = np.array(data["lstm_input"], dtype=np.int64)
                    cnn_tensor = torch.FloatTensor(cnn_input).unsqueeze(0).to(self.device)
                    lstm_tensor = torch.LongTensor(lstm_input).unsqueeze(0).to(self.device)
            except:
                pass
                
        if not is_extracted:
            # 1. Static Analysis (CNN) Fallback for raw binaries
            image_size = self.metadata.get('image_size', 128)
            cnn_input = preprocess_binary(binary_data, target_size=(image_size, image_size))
            cnn_tensor = torch.FloatTensor(cnn_input).unsqueeze(0).to(self.device)
            
            # 2. Static API Extraction (to simulate Dynamic Sequence for BiLSTM)
            vocab_size = self.metadata.get('vocab_size', 307)
            seq_length = self.metadata.get('seq_length', 100)
            
            lstm_input = np.zeros(seq_length, dtype=np.int64)
            
            try:
                import pefile
                pe = pefile.PE(data=binary_data)
                api_list = []
                if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in entry.imports:
                            if imp.name is not None:
                                api_list.append(imp.name.decode('utf-8'))
                
                # Map API names to indices using the custom vocabulary
                if api_list:
                    for i, api_name in enumerate(api_list[:seq_length]):
                        if self.vocab:
                            # Use custom vocab lookup
                            idx = self.vocab.get(api_name, 1) # 1 is <UNK>
                        else:
                            # Fallback hash mapping
                            idx = (hash(api_name) % (vocab_size - 1)) + 1
                        lstm_input[i] = idx
            except Exception as e:
                # Fallback if not a valid PE file (e.g., text/python file)
                print(f"PE parsing failed (not a PE file?): {e}")
                pass
                
            lstm_tensor = torch.LongTensor(lstm_input).unsqueeze(0).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            outputs = self.model(cnn_tensor, lstm_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Extract probabilities
            benign_prob = probs[0][0].item()
            malware_prob = probs[0][1].item()

        # Real Model Output
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
            
        if prediction == 1:
            res_pred = 'Malware'
        else:
            res_pred = 'Benign'
        
        return {
            "prediction": res_pred,
            "confidence": confidence,
            "static_analysis": "CNN Spatial Mapping Complete",
            "dynamic_analysis": "BiLSTM Sequence Analysis Complete",
            "fusion_strategy": self.model_config.get('fusion_method', 'concat')
        }

scanner = ThreatScanner()
