import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Local Inference\n(Pre-commit Hook, CPU)', 'Cloud Retraining\n(GitHub Actions, GPU)', 'Local Retraining\n(Developer Machine, CPU)']
times_seconds = [0.0238, 56.4, 397.2]  # 23.8ms, 0.94 mins, 6.62 mins
colors = ['#2ECC71', '#3498DB', '#E74C3C']  # Green, Blue, Red

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars = ax.bar(labels, times_seconds, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# Set logarithmic scale to handle massive disparity
ax.set_yscale('log')

# Formatting
ax.set_ylabel('Execution Latency (Seconds, Log Scale)', fontsize=14, fontweight='bold')
ax.set_title('Computational Latency: Inference vs. Retraining', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add grid
ax.grid(True, which="both", ls="--", alpha=0.3, axis='y')

# Add text labels on top of bars
def add_labels(bars, times):
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        # Format the text differently depending on the scale
        if times[idx] < 1:
            text = f"{times[idx]*1000:.0f} ms"
        else:
            text = f"{times[idx]/60:.1f} mins"
            
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.2, text, 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

add_labels(bars, times_seconds)

# Adjust layout and save
plt.tight_layout()
plt.savefig('d:/Adaptive-Threat-Detection/thesis_materials/Thesis/images/fig9_latency_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig9_latency_comparison.png successfully.")
