"""
Dual-Branch Malware Detection Models
Contains CNN and LSTM architectures for malware detection
"""

from .cnn_branch import CNNBranch
from .lstm_branch import LSTMBranch
from .dual_branch_model import DualBranchModel

__all__ = ['CNNBranch', 'LSTMBranch', 'DualBranchModel']
