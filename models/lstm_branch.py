"""
LSTM Branch for Malware Detection
Handles temporal/sequential feature extraction from malware data
"""

import torch
import torch.nn as nn


class LSTMBranch(nn.Module):
    """
    LSTM branch for extracting temporal/sequential features
    from malware API call sequences, opcode sequences, or behavioral patterns.
    """
    
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, 
                 feature_dim=256, bidirectional=True, vocab_size=None, embedding_dim=None):
        """
        Initialize LSTM Branch
        
        Args:
            input_size (int): Size of input features at each time step (or embedding dimension if using embeddings)
            hidden_size (int): Number of features in hidden state
            num_layers (int): Number of recurrent layers
            feature_dim (int): Dimension of extracted features
            bidirectional (bool): If True, becomes a bidirectional LSTM
            vocab_size (int, optional): Vocabulary size for embedding layer. If provided, expects integer inputs.
            embedding_dim (int, optional): Dimension of embeddings. Defaults to input_size if not provided.
        """
        super(LSTMBranch, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer for integer sequences
        if vocab_size is not None:
            self.embedding_dim = embedding_dim if embedding_dim is not None else input_size
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
            lstm_input_size = self.embedding_dim
        else:
            self.embedding = None
            self.embedding_dim = None
            lstm_input_size = input_size
        
        self.input_size = lstm_input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * self.num_directions, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, feature_dim)
        self.dropout2 = nn.Dropout(0.3)
        
    def attention_net(self, lstm_output):
        """
        Apply attention mechanism to LSTM outputs
        
        Args:
            lstm_output (torch.Tensor): LSTM output of shape (batch_size, seq_len, hidden_size * num_directions)
            
        Returns:
            torch.Tensor: Attention-weighted output of shape (batch_size, hidden_size * num_directions)
        """
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector
        
    def forward(self, x):
        """
        Forward pass through LSTM branch
        
        Args:
            x (torch.Tensor): Input tensor of shape:
                - (batch_size, seq_len) for integer sequences (if embedding is used)
                - (batch_size, seq_len, input_size) for continuous features
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, feature_dim)
        """
        # Apply embedding if using integer sequences
        if self.embedding is not None:
            # x shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            x = self.embedding(x.long())
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attn_output = self.attention_net(lstm_out)
        
        # Fully connected layers
        x = torch.relu(self.fc1(attn_output))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        return x
    
    def get_feature_dim(self):
        """Return the dimension of extracted features"""
        return self.feature_dim
