# Adaptive Threat Detection

A dual-branch deep learning model for malware detection that combines CNN and LSTM architectures to analyze both spatial and temporal features of malware samples.

## Project Structure

```
Adaptive-Threat-Detection/
│
├── data/                          # Dataset directory
│   ├── raw/                       # Raw Kaggle datasets
│   └── processed/                 # Preprocessed data
│
├── models/                        # Model architectures
│   ├── __init__.py
│   ├── cnn_branch.py             # CNN branch for spatial features
│   ├── lstm_branch.py            # LSTM branch for temporal features
│   └── dual_branch_model.py      # Combined dual-branch model
│
├── notebooks/                     # Jupyter notebooks for experiments
│   └── (your experiment notebooks here)
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Data preprocessing functions
│   └── metrics.py                # Evaluation metrics
│
├── checkpoints/                   # Model checkpoints
│
└── README.md                      # This file
```

## Model Architecture

### Dual-Branch Design

The model uses a dual-branch architecture to capture complementary features:

1. **CNN Branch**: Extracts spatial features from binary visualizations
   - 4 convolutional blocks with batch normalization
   - Adaptive pooling for variable input sizes
   - Fully connected layers for feature extraction

2. **LSTM Branch**: Captures temporal/sequential patterns
   - Bidirectional LSTM with attention mechanism
   - Processes API call sequences or opcode sequences
   - Attention-weighted feature aggregation

3. **Fusion Layer**: Combines features from both branches
   - Multiple fusion strategies: concatenation, addition, attention
   - Classification head with dropout for regularization

## Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow
```

### Dataset Preparation

1. Download your malware dataset from Kaggle
2. Place raw data in `data/raw/`
3. Use preprocessing utilities to prepare data for training

### Training Example

```python
import torch
from models import DualBranchModel
from utils import create_data_loaders

# Initialize model
model = DualBranchModel(
    cnn_config={'input_channels': 1, 'feature_dim': 256},
    lstm_config={'input_size': 100, 'hidden_size': 128, 'feature_dim': 256},
    num_classes=2,
    fusion_method='concat'
)

# Load data
train_loader, val_loader = create_data_loaders(
    train_cnn, train_lstm, train_labels,
    val_cnn, val_lstm, val_labels,
    batch_size=32
)

# Training loop (implement in notebooks)
```

## Features

- **Dual-branch architecture** for comprehensive feature extraction
- **Flexible fusion strategies** (concatenation, addition, attention)
- **Attention mechanism** in LSTM branch for better sequence modeling
- **Comprehensive evaluation metrics** (accuracy, precision, recall, F1, AUC)
- **Data augmentation** for improved generalization
- **Visualization tools** for confusion matrices, ROC curves, and training history

## Notebooks

Use the `notebooks/` folder for:
- Data exploration and analysis
- Model training experiments
- Hyperparameter tuning
- Results visualization
- Model evaluation

## Model Checkpoints

Trained models will be saved in the `checkpoints/` directory with the following naming convention:
- `model_epoch_XX.pth` - Model weights
- `best_model.pth` - Best performing model

## Contributing

Feel free to experiment with:
- Different CNN architectures (ResNet, EfficientNet, etc.)
- Alternative LSTM configurations
- New fusion strategies
- Additional preprocessing techniques
- Ensemble methods

## License

This project is for research and educational purposes.

## Acknowledgments

- Kaggle for providing malware datasets
- PyTorch team for the deep learning framework
