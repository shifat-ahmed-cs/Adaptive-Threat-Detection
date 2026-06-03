"""
CI/CD Pipeline Simulation
Demonstrates how the Adaptive Threat Detection system integrates into a DevOps pipeline.
Simulates a Commit -> Build -> Scan -> Deploy workflow.
"""

import os
import time
import subprocess
import shutil
from pathlib import Path

def print_step(step_name):
    print(f"\n{'-'*60}")
    print(f"📡 PIPELINE STEP: {step_name}")
    print(f"{'-'*60}")

def simulate_pipeline_run(file_path, description):
    print(f"\n\n[PIPELINE] STARTING PIPELINE RUN: {description}")
    print(f"Target Artifact: {file_path}")
    
    # Step 1: Source Control (Simulated Commit)
    print_step("SOURCE CONTROL")
    print(f"[OK] User committed code changes.")
    print(f"[OK] Triggered webhook: logic/deploy-pipeline")
    time.sleep(1)
    
    # Step 2: Build (Simulated)
    print_step("BUILD STAGE")
    print(f"[BUILD] Compiling source code...")
    time.sleep(1.5)
    print(f"[PACKAGE] Packaging artifact: {Path(file_path).name}")
    time.sleep(0.5)
    print(f"[OK] Build successful.")
    
    # Step 3: Security Scan (The Thesis Integration)
    print_step("SECURITY GATE (Adaptive Threat Detection)")
    print(f"[SCAN] Analyzing artifact for zero-day threats...")
    print(f"   - Static Analysis (CNN) running...")
    print(f"   - Dynamic Analysis (LSTM) running...")
    
    start_time = time.time()
    
    # Run the scan script
    result = subprocess.run(
        ['python', 'cicd/scan.py', file_path],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    elapsed = time.time() - start_time
    print(f"[TIME] Analysis completed in {elapsed:.2f}s")
    
    # Check result
    if result.returncode == 0:
        # Benign
        print(f"\n[PASS] SECURITY CHECK PASSED")
        print(f"[REPORT] Scan Report:\n{result.stdout}")
        
        # Step 4: Deploy
        print_step("DEPLOYMENT")
        print(f"[DEPLOY] Deploying to Staging Environment...")
        time.sleep(1)
        print(f"[OK] Deployment Complete.")
        
    else:
        # Malware
        print(f"\n[FAIL] SECURITY CHECK FAILED - THREAT DETECTED")
        print(f"[ALERT] Pipeline stopped to prevent malware injection.")
        print(f"[REPORT] Scan Report:\n{result.stdout}")
        
        # Step 4: Block/Rollback
        print_step("AUTOMATED REMEDIATION")
        print(f"[BLOCK] Build Blocked.")
        print(f"[ISOLATE] Isolated artifact.")
        print(f"[NOTIFY] Notification sent to Security Ops.")
        print(f"[ROLLBACK] Rolling back to last stable version...")
    
    print("\n[DONE] PIPELINE FINISHED")

def main():
    # Create dummy samples for simulation
    Path("temp_artifacts").mkdir(exist_ok=True)
    
    # 1. Create a benign sample (just some random text/code)
    benign_path = "temp_artifacts/app_v1.0.exe"
    with open(benign_path, "wb") as f:
        f.write(b"print('Hello World') " * 1000)
    
    # 2. Create a "malware" sample 
    # (In reality we need a real sample, but here we depend on the model's prediction on random data)
    # To force a specific result in this simulation, we strictly rely on the model.
    # However, since the model is trained on specific data, random noise might be unpredictable.
    # For the simulation to look good, we might need to mock the scanner output OR trust the model.
    # Let's trust the model but creating a slightly different file.
    malware_path = "temp_artifacts/trojan_update.exe"
    with open(malware_path, "wb") as f:
        # Different content pattern
        f.write(os.urandom(1024 * 100)) 
        
    # Run Simulation 1: Benign
    simulate_pipeline_run(benign_path, "Standard Feature Update")
    
    # Run Simulation 2: Malware
    simulate_pipeline_run(malware_path, "Suspicious Dependency Update")
    
    # Cleanup
    shutil.rmtree("temp_artifacts")

if __name__ == "__main__":
    main()
