"""
Dual-Branch Malware Detection Model
Combines CNN and LSTM branches for comprehensive malware detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_branch import CNNBranch
from .lstm_branch import LSTMBranch


class DualBranchModel(nn.Module):
    """
    Dual-branch model that combines CNN (spatial features) and LSTM (temporal features)
    for enhanced malware detection capabilities.
    """
    
    def __init__(self, cnn_config=None, lstm_config=None, num_classes=2, fusion_method='concat'):
        """
        Initialize Dual-Branch Model
        
        Args:
            cnn_config (dict): Configuration for CNN branch
            lstm_config (dict): Configuration for LSTM branch
            num_classes (int): Number of output classes
            fusion_method (str): Method to fuse features ('concat', 'add', 'attention')
        """
        super(DualBranchModel, self).__init__()
        
        # Default configurations
        if cnn_config is None:
            cnn_config = {
                'input_channels': 1,
                'num_classes': num_classes,
                'feature_dim': 256
            }
        
        if lstm_config is None:
            lstm_config = {
                'input_size': 100,
                'hidden_size': 128,
                'num_layers': 2,
                'feature_dim': 256,
                'bidirectional': True
            }
        
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Initialize branches
        self.cnn_branch = CNNBranch(**cnn_config)
        self.lstm_branch = LSTMBranch(**lstm_config)
        
        # Get feature dimensions
        cnn_feature_dim = self.cnn_branch.get_feature_dim()
        lstm_feature_dim = self.lstm_branch.get_feature_dim()
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = cnn_feature_dim + lstm_feature_dim
        elif fusion_method == 'add':
            assert cnn_feature_dim == lstm_feature_dim, "Feature dimensions must match for addition fusion"
            fusion_dim = cnn_feature_dim
        elif self.fusion_method == 'attention':
            fusion_dim = cnn_feature_dim + lstm_feature_dim
            # Cross-Branch Attention: CNN attends to LSTM and vice-versa
            self.query_cnn = nn.Linear(cnn_feature_dim, cnn_feature_dim)
            self.key_lstm = nn.Linear(lstm_feature_dim, cnn_feature_dim)
            self.query_lstm = nn.Linear(lstm_feature_dim, lstm_feature_dim)
            self.key_cnn = nn.Linear(cnn_feature_dim, lstm_feature_dim)
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def fuse_features(self, cnn_features, lstm_features):
        """
        Fuse features using Cross-Branch Attention (Methodology Section 3)
        """
        if self.fusion_method == 'concat':
            return torch.cat([cnn_features, lstm_features], dim=1)
        
        elif self.fusion_method == 'add':
            return cnn_features + lstm_features
        
        elif self.fusion_method == 'attention':
            # 1. CNN attends to LSTM
            q_c = self.query_cnn(cnn_features) # (batch, c_dim)
            k_l = self.key_lstm(lstm_features) # (batch, c_dim)
            # Dot product attention score
            attn_c = torch.sum(q_c * k_l, dim=1, keepdim=True) # (batch, 1)
            attn_c = torch.sigmoid(attn_c)
            weighted_cnn = cnn_features * attn_c
            
            # 2. LSTM attends to CNN
            q_l = self.query_lstm(lstm_features) # (batch, l_dim)
            k_c = self.key_cnn(cnn_features) # (batch, l_dim)
            attn_l = torch.sum(q_l * k_c, dim=1, keepdim=True) # (batch, 1)
            attn_l = torch.sigmoid(attn_l)
            weighted_lstm = lstm_features * attn_l
            
            return torch.cat([weighted_cnn, weighted_lstm], dim=1)
        
    def forward(self, cnn_input, lstm_input):
        """
        Forward pass through dual-branch model
        
        Args:
            cnn_input (torch.Tensor): Input for CNN branch (batch_size, channels, height, width)
            lstm_input (torch.Tensor): Input for LSTM branch (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Class predictions of shape (batch_size, num_classes)
        """
        # Extract features from both branches
        cnn_features = self.cnn_branch(cnn_input)
        lstm_features = self.lstm_branch(lstm_input)
        
        # Fuse features
        fused_features = self.fuse_features(cnn_features, lstm_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output
    
    def get_branch_features(self, cnn_input, lstm_input):
        """
        Get features from both branches without classification
        Useful for feature visualization and analysis
        
        Args:
            cnn_input (torch.Tensor): Input for CNN branch
            lstm_input (torch.Tensor): Input for LSTM branch
            
        Returns:
            tuple: (cnn_features, lstm_features, fused_features)
        """
        cnn_features = self.cnn_branch(cnn_input)
        lstm_features = self.lstm_branch(lstm_input)
        fused_features = self.fuse_features(cnn_features, lstm_features)
        
        return cnn_features, lstm_features, fused_features

    def expand_capacity(self, new_layer_size=256):
        """
        Dynamically expand the model capacity by adding a new layer to the classifier.
        Implements 'Self-Adaptive Architecture' (Thesis Objective 1).
        
        Args:
            new_layer_size (int): Size of the new hidden layer
            
        Returns:
            bool: True if expansion successful
        """
        device = next(self.parameters()).device
        print(f"\n[Self-Adaptive Architecture] Triggering Dynamic Expansion...")
        
        # Current classifier structure:
        # 0: Linear(fusion -> 512)
        # 1: ReLU
        # 2: Dropout
        # 3: Linear(512 -> 256)
        # ...
        
        # We will insert a new layer after the first block
        # New structure: Linear(fusion->512) -> ReLU -> Dropout -> NEW_LAYER -> ReLU -> Dropout -> Linear(512->256) ...
        
        # Create new layers
        new_linear = nn.Linear(512, new_layer_size).to(device)
        new_relu = nn.ReLU().to(device)
        new_dropout = nn.Dropout(0.3).to(device)
        
        # Adjust existing layers to match dimensions
        # The layer at index 3 was taking 512 input, now it must take new_layer_size
        # BUT simpler strategy: Just append feature extraction layers or expand the fusion mapping.
        
        # SIMPLIFIED STRATEGY for P1 Thesis Deployment:
        # Add a new block at the beginning of classifier? No, dimensions must match.
        # Add a new block at the end? No, output is fixed.
        # Let's Insert a layer in the middle.
        
        # Reconstruct classifier with added depth
        # For simplicity in this implementation, we will append a layer to the start of the bottleneck
        # Old: Fusion(256) -> 512 -> ...
        # New: Fusion(256) -> 512 -> 512(New) -> ...
        
        # Note: Modifying compiled/running model graph requires care.
        # We will simply create a new layer sequence.
        
        current_layers = list(self.classifier.children())
        
        # Let's insert after layer 2 (Dropout)
        # Layer 0: Linear(Fusion -> 512)
        # Layer 1: ReLU
        # Layer 2: Dropout
        # INSERT HERE: Linear(512 -> 512) -> ReLU -> Dropout
        # Layer 3: Linear(512 -> 256)
        
        new_block = [
            nn.Linear(512, 512).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.3).to(device)
        ]
        
        # Combine
        new_classifier_list = current_layers[:3] + new_block + current_layers[3:]
        self.classifier = nn.Sequential(*new_classifier_list)
        
        print(f"Model Expanded: Added hidden layer (512->512). Total depth increased.")
        return True
