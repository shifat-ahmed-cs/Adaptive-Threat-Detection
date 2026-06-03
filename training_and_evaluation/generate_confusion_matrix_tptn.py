"""
Regenerate confusion matrix with TP/TN/FP/FN labels
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load evaluation results
results_path = Path('checkpoints/evaluation_results.json')
if not results_path.exists():
    print(f"Error: {results_path} not found")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

# Extract confusion matrix values from test_metrics
metrics = results['test_metrics']
tn = metrics['true_negatives']
fp = metrics['false_positives']
fn = metrics['false_negatives']
tp = metrics['true_positives']

# Build confusion matrix
# Row 0: Actual Benign (TN, FP)
# Row 1: Actual Malware (FN, TP)
cm = np.array([[tn, fp],
               [fn, tp]])

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Count'},
            ax=ax)

# Add TP/TN/FP/FN annotations
# cm[0,0] = TN (True Negative - Correctly predicted Benign)
# cm[0,1] = FP (False Positive - Benign predicted as Malware)
# cm[1,0] = FN (False Negative - Malware predicted as Benign)
# cm[1,1] = TP (True Positive - Correctly predicted Malware)

labels = [['TN', 'FP'],
          ['FN', 'TP']]

# Use green for correct predictions (TN, TP) and red for errors (FP, FN)
colors = [['green', 'red'],
          ['red', 'green']]

for i in range(2):
    for j in range(2):
        text = ax.text(j + 0.5, i + 0.7, labels[i][j],
                      ha="center", va="center",
                      color=colors[i][j], fontsize=14, weight='bold')

plt.title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
plt.tight_layout()

# Save to visualizations/Concat
output_path = Path('visualizations/Concat/confusion_matrix_tptn.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to {output_path}")

# Also save to experiments/visualization
exp_path = Path('experiments/visualization/confusion_matrix_tptn.png')
exp_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(exp_path, dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to {exp_path}")

plt.close()
