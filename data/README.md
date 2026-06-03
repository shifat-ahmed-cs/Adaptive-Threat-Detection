# Data Directory

This directory contains datasets for the malware detection project.

## Structure

- **raw/**: Place your raw Kaggle datasets here
- **processed/**: Preprocessed data ready for training

## Dataset Guidelines

### Expected Data Format

1. **For CNN Branch**: Binary files converted to image representations
   - Format: `.npy` files containing image arrays
   - Shape: (num_samples, 1, height, width)
   - Normalized: 0-1 range

2. **For LSTM Branch**: Sequential data (API calls, opcodes, etc.)
   - Format: `.npy` files containing sequence arrays
   - Shape: (num_samples, sequence_length, feature_dim)
   - Preprocessed and padded to uniform length

### Kaggle Datasets

Popular malware datasets on Kaggle:
- Microsoft Malware Classification Challenge
- Malware Analysis Datasets
- Android Malware Dataset
- PE Malware Detection Dataset

### Data Preprocessing

Use the utilities in `utils/preprocessing.py` to:
1. Convert binary files to image representations
2. Extract and encode sequential features
3. Normalize and augment data
4. Split into train/validation/test sets

## Usage

```python
from utils import load_kaggle_dataset, preprocess_binary, preprocess_sequence

# Load your dataset
data = load_kaggle_dataset('data/raw/your_dataset')

# Preprocess for CNN
cnn_data = preprocess_binary(data['binary'], target_size=(256, 256))

# Preprocess for LSTM
lstm_data = preprocess_sequence(data['sequences'], max_length=1000)
```

## Notes

- Keep raw data separate from processed data
- Document any preprocessing steps in your notebooks
- Consider data versioning for reproducibility
