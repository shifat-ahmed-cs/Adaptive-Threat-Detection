import subprocess
import json
from pathlib import Path

mutation_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
results_dir = Path('checkpoints')

# Run evaluate_robustness.py for each mutation rate
for p in mutation_rates:
    print(f"Evaluating at mutation rate {p}...")
    output_file = results_dir / f'robustness_{p}.json'
    cmd = [
        'python', 'training_and_evaluation/evaluate_robustness.py',
        '--mutation-rate', str(p),
        '--output', str(output_file)
    ]
    subprocess.run(cmd, check=True)

print("Evaluation complete. Compiling decay matrices...")

# Read the generated JSONs and compile matrices
insert_metrics = {'Adv. Insert': [], 'Adv. Replace': [], 'Adv. Append': []}
replace_metrics = {'Adv. Insert': [], 'Adv. Replace': [], 'Adv. Append': []}
append_metrics = {'Adv. Insert': [], 'Adv. Replace': [], 'Adv. Append': []}

for p in mutation_rates:
    with open(results_dir / f'robustness_{p}.json', 'r') as f:
        data = json.load(f)
    
    # 1. RANDOM_INSERT Attack Test Set ('random_insert' dataset)
    insert_metrics['Adv. Insert'].append(round(data['random_insert']['random_insert']['accuracy'] * 100, 1))
    insert_metrics['Adv. Replace'].append(round(data['random_replace']['random_insert']['accuracy'] * 100, 1))
    insert_metrics['Adv. Append'].append(round(data['noise_append']['random_insert']['accuracy'] * 100, 1))
    
    # 2. RANDOM_REPLACE Attack Test Set ('random_replace' dataset)
    replace_metrics['Adv. Insert'].append(round(data['random_insert']['random_replace']['accuracy'] * 100, 1))
    replace_metrics['Adv. Replace'].append(round(data['random_replace']['random_replace']['accuracy'] * 100, 1))
    replace_metrics['Adv. Append'].append(round(data['noise_append']['random_replace']['accuracy'] * 100, 1))

    # 3. NOISE_APPEND Attack Test Set ('noise_append' dataset)
    append_metrics['Adv. Insert'].append(round(data['random_insert']['noise_append']['accuracy'] * 100, 1))
    append_metrics['Adv. Replace'].append(round(data['random_replace']['noise_append']['accuracy'] * 100, 1))
    append_metrics['Adv. Append'].append(round(data['noise_append']['noise_append']['accuracy'] * 100, 1))

# Write to a JSON file to easily copy-paste or read later
decay_data = {
    'insert_metrics': insert_metrics,
    'replace_metrics': replace_metrics,
    'append_metrics': append_metrics
}

with open(results_dir / 'compiled_decay_metrics.json', 'w') as f:
    json.dump(decay_data, f, indent=4)

print("Decay metrics compiled to compiled_decay_metrics.json")
