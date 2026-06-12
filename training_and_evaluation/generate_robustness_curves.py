"""
Script to generate robustness decay curves for Chapter 6.
Plots detection accuracy vs. mutation rate for RANDOM_INSERT, RANDOM_REPLACE, and NOISE_APPEND attacks.
Saves outputs directly to the LaTeX images directory.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# Style setup to match existing visualizations
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  True,
    'axes.spines.right':True,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.color':       '#e2e8f0'
})

# Define output directories
EXP_DIR = Path('experiments/visualization')
LATEX_DIR = Path('thesis_materials/Thesis/images')
EXP_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Definition
# -----------------------------------------------------------------------------
x_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# Colors matching the PALETTE in generate_visualizations.py
c_insert  = '#2563EB'  # blue
c_replace = '#D97706'  # orange/amber (matching 'noise_append' style or orange)
c_append  = '#16A34A'  # green

# 1. RANDOM_INSERT Attack
insert_metrics = {
    'Adv. Insert':  [99.0, 98.5, 99.2, 99.2, 98.8, 98.5],
    'Adv. Replace': [98.8, 98.3, 98.5, 98.7, 98.8, 98.3],
    'Adv. Append':  [98.5, 98.2, 98.3, 98.3, 98.5, 98.2]
}

# 2. RANDOM_REPLACE Attack
replace_metrics = {
    'Adv. Insert':  [98.8, 98.0, 98.5, 98.3, 98.2, 98.5],
    'Adv. Replace': [98.5, 97.8, 98.0, 97.8, 98.2, 98.5],
    'Adv. Append':  [98.0, 97.3, 97.3, 97.8, 97.3, 97.8]
}

# 3. NOISE_APPEND Attack
append_metrics = {
    'Adv. Insert':  [95.3, 96.2, 95.7, 96.0, 96.2, 96.5],
    'Adv. Replace': [95.7, 97.0, 96.0, 96.7, 96.5, 96.5],
    'Adv. Append':  [98.2, 98.8, 98.8, 98.7, 98.8, 98.7]
}

plots_config = [
    ('Robustness against RANDOM_INSERT Attack', 'robustness_insert_adv.png', insert_metrics),
    ('Robustness against RANDOM_REPLACE Attack', 'robustness_replace_adv.png', replace_metrics),
    ('Robustness against NOISE_APPEND Attack', 'robustness_append_adv.png', append_metrics)
]

for title, filename, metrics in plots_config:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(x_rates, metrics['Adv. Insert'], 'o-', color=c_insert, linewidth=2, label='Adv. Insert')
    ax.plot(x_rates, metrics['Adv. Replace'], 'o-', color='#f97316', linewidth=2, label='Adv. Replace') # standard orange
    ax.plot(x_rates, metrics['Adv. Append'], 'o-', color=c_append, linewidth=2, label='Adv. Append')
    
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Mutation Rate')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_xlim([-0.03, 0.53])
    ax.set_ylim([80.0, 100.0])
    ax.set_yticks([80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save files
    plt.savefig(EXP_DIR / filename, dpi=300, bbox_inches='tight')
    plt.savefig(LATEX_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated and saved {filename}")
