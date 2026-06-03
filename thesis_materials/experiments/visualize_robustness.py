import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename):
    path = Path('experiments') / filename
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_robustness():
    # Load all results
    models = {
        'Adv. Insert': 'robustness_insert.json',
        'Adv. Replace': 'robustness_replace.json',
        'Adv. Append': 'robustness_append.json'
    }
    
    data = {}
    for name, fname in models.items():
        res = load_results(fname)
        if res:
            data[name] = res
            
    if not data:
        print("No data found!")
        return

    strategies = ['random_insert', 'random_replace', 'noise_append']
    rates = ['0', '5', '10', '20', '30', '50'] # Keys in json are strings or floats? 
    # In json dump, keys might be floats. Let's check one.
    # Actually keys are likely strings of floats e.g. "0.0", "0.05"
    
    # We'll parse dynamically
    sample_res = next(iter(data.values()))
    sample_strat = next(iter(sample_res.values()))
    # rates_float = sorted([float(k) for k in sample_strat.keys()])
    rates_float = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Create a plot for each strategy
    for strategy in strategies:
        plt.figure(figsize=(10, 6))
        
        for model_name, model_data in data.items():
            if strategy not in model_data:
                continue
                
            strat_data = model_data[strategy]
            # Extract y values matching rates_float
            # JSON keys might be stringified floats, so we stick to float matching
            y_values = []
            for r in rates_float:
                # Try exact string match or float match
                val = strat_data.get(str(r)) or strat_data.get(r)
                if val is None:
                    # Try looking for close key
                    for k, v in strat_data.items():
                        if abs(float(k) - r) < 1e-5:
                            val = v
                            break
                y_values.append(val if val is not None else 0)
                
            plt.plot(rates_float, y_values, marker='o', label=model_name, linewidth=2)
            
        plt.title(f'Robustness against {strategy.upper()} Attack', fontsize=14)
        plt.xlabel('Mutation Rate', fontsize=12)
        plt.ylabel('Detection Accuracy (%)', fontsize=12)
        plt.ylim(80, 100) # Zoom in to relevant range
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save
        output_path = Path(f'current_working_directory') # Placeholder, will use CWD
        plt.savefig(f'robustness_{strategy}.png', dpi=300, bbox_inches='tight')
        print(f"Saved robustness_{strategy}.png")
        plt.close()

if __name__ == "__main__":
    plot_robustness()
