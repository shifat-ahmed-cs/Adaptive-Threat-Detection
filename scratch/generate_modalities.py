import numpy as np
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
os.makedirs('thesis_materials/Thesis/images', exist_ok=True)

# Load custom dataset
train_cnn = np.load('data/custom_dataset/train_cnn.npy')
train_lstm = np.load('data/custom_dataset/train_lstm.npy')
train_labels = np.load('data/custom_dataset/train_labels.npy')

# Find a malware sample
malware_indices = np.where(train_labels == 1)[0]
idx = malware_indices[0]  # Use the first malware sample

cnn_sample = train_cnn[idx, 0]  # Shape: (128, 128)
lstm_sample = train_lstm[idx]    # Shape: (100,)

# Set up the plot with a modern aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# Left Panel: CNN Input (Spatial representation of the binary)
im = ax1.imshow(cnn_sample, cmap='gray', interpolation='nearest')
ax1.set_title('Spatial Modality: PE Binary 2D Grayscale', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Width (128 pixels)', fontsize=10)
ax1.set_ylabel('Height (128 pixels)', fontsize=10)
ax1.grid(False)
# Add a subtle colorbar
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# Right Panel: LSTM Input (Temporal sequence of API calls)
time_steps = np.arange(len(lstm_sample))
ax2.step(time_steps, lstm_sample, where='mid', color='#1f77b4', linewidth=1.5, alpha=0.8)
ax2.scatter(time_steps, lstm_sample, color='#d62728', s=10, zorder=3, alpha=0.6)
ax2.set_title('Temporal Modality: Integer-Encoded API Calls', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Execution Step (Time)', fontsize=10)
ax2.set_ylabel('API Token ID (Vocabulary)', fontsize=10)
ax2.set_xlim(-2, 102)
ax2.set_ylim(-10, max(lstm_sample) + 20)

plt.tight_layout()

# Save the generated image
output_path = 'thesis_materials/Thesis/images/input_modalities.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Successfully generated and saved input modalities visualization to {output_path}")
