"""
CNN Branch for Malware Detection
Handles spatial feature extraction from malware data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    """
    Convolutional Neural Network branch for extracting spatial features
    from malware binary representations or image-based malware visualizations.
    """
    
    def __init__(self, input_channels=1, num_classes=2, feature_dim=256):
        """
        Initialize CNN Branch
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes (2 for binary classification)
            feature_dim (int): Dimension of extracted features
        """
        super(CNNBranch, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, feature_dim)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through CNN branch
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, feature_dim)
        """
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        return x
    
    def get_feature_dim(self):
        """Return the dimension of extracted features"""
        return self.feature_dim
