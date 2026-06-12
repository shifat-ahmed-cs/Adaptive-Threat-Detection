import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.dual_branch_model import DualBranchModel

def main():
    print("Initializing Dual-Branch Model for Latency Measurement...")
    
    # 1. Initialize model
    model = DualBranchModel()
    
    # Check if GPU is available (optional, for comparison if they have one)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model.to(device)
    
    # Create dummy data for a single forward pass (Local Inference pre-commit hook)
    # CNN expects: (batch_size, channels, height, width) -> (1, 1, 128, 128)
    cnn_input_single = torch.randn(1, 1, 128, 128).to(device)
    
    # LSTM expects: (batch_size, seq_len, input_size) -> (1, 100, 100)
    lstm_input_single = torch.randn(1, 100, 100).to(device)
    
    print("-" * 50)
    print("1. MEASURING LOCAL INFERENCE (PRE-COMMIT HOOK)")
    print("-" * 50)
    
    # Warmup
    for _ in range(5):
        _ = model(cnn_input_single, lstm_input_single)
        
    # Measure 100 passes to get a reliable average
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(cnn_input_single, lstm_input_single)
    end_time = time.time()
    
    avg_inference_ms = ((end_time - start_time) / iterations) * 1000
    print(f"Average Local Inference Time: {avg_inference_ms:.2f} milliseconds\n")
    
    
    print("-" * 50)
    print("2. MEASURING LOCAL RETRAINING SIMULATION (DEVELOPER MACHINE)")
    print("-" * 50)
    
    # To simulate retraining, we will do forward + backward passes on a batch
    batch_size = 16
    dataset_size = 2000 # As stated in chapter 5
    batches_per_epoch = dataset_size // batch_size
    
    cnn_batch = torch.randn(batch_size, 1, 128, 128).to(device)
    lstm_batch = torch.randn(batch_size, 100, 100).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    # Warmup
    optimizer.zero_grad()
    out = model(cnn_batch, lstm_batch)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    
    # Measure 10 batches to average
    train_iterations = 10
    start_time_train = time.time()
    
    for _ in range(train_iterations):
        optimizer.zero_grad()
        out = model(cnn_batch, lstm_batch)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
    end_time_train = time.time()
    
    avg_batch_train_sec = (end_time_train - start_time_train) / train_iterations
    
    # Estimate time for 1 full epoch (simulating the Cloud / Local Retraining)
    estimated_epoch_sec = avg_batch_train_sec * batches_per_epoch
    estimated_epoch_min = estimated_epoch_sec / 60
    
    print(f"Average time per training batch ({batch_size} samples): {avg_batch_train_sec:.4f} seconds")
    print(f"Estimated time to retrain 1 full epoch locally ({dataset_size} samples): {estimated_epoch_min:.2f} minutes")
    print(f"Estimated time to retrain 5 full epochs locally: {estimated_epoch_min * 5:.2f} minutes\n")
    
    print("-" * 50)
    print("INSTRUCTIONS FOR YOUR THESIS:")
    print("-" * 50)
    print("1. Update the '45 ms' in Table 5.10 / Figure 5.9 with the 'Average Local Inference Time' above.")
    print("2. Update the '12.5 mins' in Table 5.10 / Figure 5.9 with the 5-epoch estimate above.")
    print("3. If you have a GPU runner on GitHub Actions, it is typically 5-10x faster than CPU.")
    print("   You can conservatively estimate Cloud Retraining as (Local Time / 7).")

if __name__ == "__main__":
    main()
