import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Data from experiments
data = {
    'Method': ['Concat (Baseline)', 'Add', 'Attention'],
    'Recall (Malware)': ['98.38%', '97.81%', '97.83%'],
    'Specificity (Benign)': ['92.25%', '90.85%', '93.66%'],
    'Accuracy': ['98.25%', '97.66%', '97.74%'],
    'Verdict': ['Best Recall & Acc', 'Worst', 'Best Specificity']
}

df = pd.DataFrame(data)

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

# Create output directory
output_dir = Path('visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Render and save
ax = render_mpl_table(df, header_columns=0, col_width=4.0)
plt.savefig(output_dir / 'fusion_comparison_table.png', bbox_inches='tight', dpi=300)
print(f"Table saved to {output_dir / 'fusion_comparison_table.png'}")
