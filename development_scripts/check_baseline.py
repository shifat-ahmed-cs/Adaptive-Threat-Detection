import torch
checkpoint = torch.load('checkpoints/baseline_model.pth', map_location='cpu')
print(checkpoint.keys())
if 'model_config' in checkpoint:
    print("Model Config:", checkpoint['model_config'])
if 'metadata' in checkpoint:
    print("Metadata:", checkpoint['metadata'])
