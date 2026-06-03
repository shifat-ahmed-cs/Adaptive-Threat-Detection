"""
Generate per-class classification report for baseline model
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Metrics are hardcoded from 1.png

# Use metrics from user request
benign_precision = 0.5598
benign_recall = 0.9225
benign_f1 = 0.6968
benign_support = 142

malware_precision = 0.9982
malware_recall = 0.9838
malware_f1 = 0.9910
malware_support = 6354

# Overall metrics
accuracy = 0.9825
total_samples = 6496

# Create classification report data
data = [
    ['Benign', f'{benign_precision:.3f}', f'{benign_recall:.3f}', f'{benign_f1:.3f}', benign_support],
    ['Malware', f'{malware_precision:.3f}', f'{malware_recall:.3f}', f'{malware_f1:.3f}', malware_support],
    ['', '', '', '', ''],
    ['Accuracy', '', '', f'{accuracy:.3f}', total_samples]
]

df = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.axis('tight')

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Style
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Highlight headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4a4a4a')
    elif row == 3:  # Empty separator row
        cell.set_facecolor('#ffffff')
        cell.set_text_props(color='white')
    elif row == 4:  # Accuracy row
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8e8e8')
    else:
        cell.set_facecolor('#f5f5f5')

plt.title('Classification Report - Baseline Model', fontsize=14, weight='bold', pad=20)
plt.tight_layout()

# Save to experiments/visualization
output_path = Path('experiments/visualization/classification_report.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Classification report saved to {output_path}")

plt.close()
