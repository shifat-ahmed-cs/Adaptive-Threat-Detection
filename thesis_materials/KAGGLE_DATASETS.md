# Using Kaggle Malware Datasets

This guide explains how to use the two Kaggle datasets with your dual-branch malware detection model.

## Datasets

### 1. Malware Analysis Datasets: Raw PE as Image
**URL**: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-raw-pe-as-image

This dataset contains PE (Portable Executable) files visualized as images for the CNN branch.

### 2. Malware Analysis Datasets: API Call Sequences
**URL**: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences

This dataset contains API call sequences for the LSTM branch.

## Quick Start

### Step 1: Download Datasets

#### Option A: Manual Download
1. Visit the dataset pages above
2. Click "Download" on each page
3. Extract to:
   - `data/raw/pe_images/`
   - `data/raw/api_sequences/`

#### Option B: Using Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (see https://www.kaggle.com/docs/api)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download datasets
kaggle datasets download -d ang3loliveira/malware-analysis-datasets-raw-pe-as-image -p data/raw/pe_images
kaggle datasets download -d ang3loliveira/malware-analysis-datasets-api-call-sequences -p data/raw/api_sequences

# Extract (Windows PowerShell)
Expand-Archive data/raw/pe_images/*.zip -DestinationPath data/raw/pe_images/
Expand-Archive data/raw/api_sequences/*.zip -DestinationPath data/raw/api_sequences/
```

### Step 2: Load and Preprocess Data

Use the provided notebook:
```bash
jupyter notebook notebooks/02_kaggle_data_loading.ipynb
```

Or use Python directly:
```python
from utils import prepare_dual_branch_data

data = prepare_dual_branch_data(
    pe_image_path='data/raw/pe_images',
    api_sequence_path='data/raw/api_sequences',
    train_split=0.7,
    val_split=0.15,
    pe_size=(256, 256),
    seq_length=1000,
    save_path='data/processed'
)
```

### Step 3: Train the Model

Use the training notebook:
```bash
jupyter notebook notebooks/03_train_with_kaggle_data.ipynb
```

## Data Loader Features

The `KaggleMalwareLoader` class provides:

- **Automatic format detection**: Handles CSV, JSON, and directory structures
- **Vocabulary building**: Creates vocabulary from API call sequences
- **Data alignment**: Ensures PE images and API sequences are properly matched
- **Preprocessing**: Resizes images, pads/truncates sequences
- **Train/Val/Test splitting**: Automatic data splitting with shuffling

## Expected Data Structure

### PE Images Dataset
```
data/raw/pe_images/
├── train.csv (optional, with labels)
├── benign/ (or)
│   ├── sample1.png
│   └── sample2.png
└── malware/
    ├── sample3.png
    └── sample4.png
```

### API Sequences Dataset
```
data/raw/api_sequences/
├── sequences.csv (with labels and sequences)
└── (or JSON/pickle files)
```

## Troubleshooting

### Issue: "Path not found"
- Ensure you've extracted the datasets to the correct directories
- Check that the paths in your code match the actual directory structure

### Issue: "Vocabulary size too large"
- Reduce `max_length` parameter
- Filter rare API calls
- Use embedding layers to handle large vocabularies

### Issue: "Mismatched sample counts"
- The loader automatically aligns datasets by taking the minimum count
- Ensure both datasets have corresponding samples

## Advanced Usage

### Custom Preprocessing
```python
from utils import KaggleMalwareLoader

loader = KaggleMalwareLoader(
    pe_image_path='data/raw/pe_images',
    api_sequence_path='data/raw/api_sequences'
)

# Load separately for custom processing
pe_images, pe_labels, _ = loader.load_pe_images(target_size=(512, 512))
api_seqs, api_labels, _, vocab = loader.load_api_sequences(max_length=2000)
```

### Adjusting Model for Dataset
```python
from models import DualBranchModel

# Configure based on your data
model = DualBranchModel(
    cnn_config={
        'input_channels': 1,  # Grayscale images
        'feature_dim': 256
    },
    lstm_config={
        'input_size': len(vocab),  # Vocabulary size
        'hidden_size': 128,
        'feature_dim': 256
    },
    fusion_method='concat'  # or 'add', 'attention'
)
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA if available
2. **Batch Size**: Adjust based on your GPU memory
3. **Data Augmentation**: Implement for PE images to improve generalization
4. **Sequence Length**: Tune based on actual API call sequence lengths
5. **Fusion Method**: Experiment with different fusion strategies

## References

- [Kaggle PE Images Dataset](https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-raw-pe-as-image)
- [Kaggle API Sequences Dataset](https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
