import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_results(filename):
    path = Path('experiments') / filename
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def generate_table():
    models = {
        'Random Insert': 'robustness_insert.json',
        'Random Replace': 'robustness_replace.json',
        'Noise Append': 'robustness_append.json'
    }
    
    # We will pick a specific mutation rate to compare, e.g., 0.3 (30%)
    target_rate = 0.3
    str_rate = "0.3" 
    
    data = []
    
    for model_name, fname in models.items():
        res = load_results(fname)
        if not res:
            continue
            
        # Res structure: {'random_insert': {'0.0': ...}, ...}
        
        # Get score for each attack type at target_rate
        insert_score = res.get('random_insert', {}).get(str_rate, 0)
        replace_score = res.get('random_replace', {}).get(str_rate, 0)
        append_score = res.get('noise_append', {}).get(str_rate, 0)
        
        # Also clean accuracy (Rate 0)
        clean_score = res.get('random_insert', {}).get("0.0", 0) # Should be same across strategies
        
        data.append({
            'Strategy': model_name,
            'Clean Acc.': f"{clean_score:.1f}%",
            'Insert (30%)': f"{insert_score:.1f}%",
            'Replace (30%)': f"{replace_score:.1f}%",
            'Append (30%)': f"{append_score:.1f}%"
        })
        
    df = pd.DataFrame(data)
    
    # Render Table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Highlight headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4a4a4a')
        else:
            cell.set_facecolor('#f5f5f5')
            
    plt.title(f'Adversarial Model Robustness Comparison (Mutation Rate {int(target_rate*100)}%)', y=0.8, fontsize=13, weight='bold')
    
    output_temp = 'robustness_comparison_table.png'
    plt.savefig(output_temp, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Table saved to {output_temp}")

if __name__ == "__main__":
    generate_table()
