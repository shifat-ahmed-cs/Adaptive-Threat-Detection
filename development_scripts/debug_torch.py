print("Importing torch...", flush=True)
import torch
print(f"Torch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print("Done.", flush=True)
