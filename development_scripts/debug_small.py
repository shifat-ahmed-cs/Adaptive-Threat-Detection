import pickle
from pathlib import Path
import time
import sys

print("Starting small debug...", flush=True)
p = Path('data/processed/metadata.pkl')
try:
    print(f"Size: {p.stat().st_size}", flush=True)
    with open(p, 'rb') as f:
        meta = pickle.load(f)
    print("Loaded metadata key count:", len(meta), flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
print("Done.", flush=True)
