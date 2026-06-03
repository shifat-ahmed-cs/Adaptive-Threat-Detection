import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def create_comparison_table_image():
    # Define the data
    data = [
        ["Our System", "Dual-Branch (CNN + BiLSTM)", "98.65%", "99.76%", "98.87%"],
        ["Kalash et al. (2018)", "M-CNN (Deep CNN on Images)", "98.52%", "98.50%", "98.50%"],
        ["Kolosnjaji et al. (2016)", "CNN + LSTM (Sequential API)", "97.70%", "98.10%", "97.50%"],
        ["Catak et al. (2020)", "Standard LSTM (API Sequence)", "95.00%", "95.00%", "95.00%"],
        ["Oliveira & Sassi (2019)", "Graph CNN (API Graphs)", "92.44%", "~92.00%", "~92.00%"]
    ]
    
    columns = ["Reference", "Method", "Accuracy", "Precision", "Recall"]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Create the table
    # Define column widths: Reference (0.2), Method (0.35), Metrics (0.15 each)
    col_widths = [0.2, 0.35, 0.15, 0.15, 0.15]
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     colWidths=col_widths,
                     loc='center',
                     cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Slightly larger font
    table.scale(1.2, 1.8)   # More vertical padding
    
    # Add colors
    # distinct highlight color for our system
    for i in range(len(columns)):
        # Header
        cell = table[0, i]
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white', weight='bold')
        cell.set_edgecolor('white')
        
        # Our System Row (Row 1)
        cell = table[1, i]
        cell.set_facecolor('#e6f3ff')  # Light blue highlight
        cell.set_text_props(weight='bold')
        
    # Save the figure
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'comparison_table.png'
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Table image saved to: {output_path.absolute()}")

if __name__ == "__main__":
    create_comparison_table_image()
