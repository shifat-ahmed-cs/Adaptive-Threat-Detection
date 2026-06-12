import json
from pathlib import Path

for f in ['concat', 'add', 'attention', 'gated']:
    p = Path(f'checkpoints/evaluation_results_{f}.json')
    if p.exists():
        d = json.load(open(p))
        m = d['test_metrics']
        print(f"{f:10}: Acc={m['accuracy']*100:.2f}% Rec={m['recall']*100:.2f}% Spec={m['specificity']*100:.2f}%")
