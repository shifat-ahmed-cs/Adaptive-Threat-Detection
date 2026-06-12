# Collaborator Setup Guide: Adaptive Threat Detection

Welcome to the Adaptive Threat Detection project! Follow these exact steps to get the CI/CD Threat Detection system running on your local Windows machine.

## Prerequisites
1. **Python 3.9 - 3.11** (Ensure `Add Python to PATH` is checked during installation).
2. **Git** for version control.
3. Access to this GitHub Repository.

---

## Step 1: Clone the Repository
Open your terminal (PowerShell or Command Prompt) and download the code:
```bash
git clone https://github.com/Sahil-Ishan-Tonmoy/Adaptive-Threat-Detection.git
cd Adaptive-Threat-Detection
```

## Step 2: Install Dependencies
Install all required Python libraries, including PyTorch for the Dual-Branch Neural Network:
```bash
pip install -r requirements.txt
```
*(Note: If you have an NVIDIA GPU, you may want to install the CUDA version of PyTorch from the official PyTorch website for faster neural network inference).*

## Step 3: Create the Server Launcher
For security reasons, the `run_server.bat` file is ignored by Git because it requires a Personal Access Token. You must create this file manually.

1. Go to your GitHub Settings -> **Developer Settings** -> **Personal access tokens (classic)**.
2. Generate a new token with `repo` and `workflow` scopes.
3. Create a new file in the root directory named `run_server.bat`.
4. Paste the following code into `run_server.bat` and insert your token:

```bat
@echo off
set GITHUB_TOKEN=your_personal_access_token_here
set GITHUB_REPO=Sahil-Ishan-Tonmoy/Adaptive-Threat-Detection
python -m uvicorn prototype.backend.main:app --host 0.0.0.0 --port 8000
pause
```
*(Git will automatically ignore this file so you don't accidentally push your token to the repo).*

## Step 4: Run the System
The system runs in two parts: the backend server (which handles AI inference) and the frontend UI Dashboard.

**First Terminal (Start Backend):**
```bash
.\run_server.bat
```
Wait for it to say `Application startup complete`. Leave this terminal open.

**Second Terminal (Start UI Dashboard):**
Open a new terminal window in the project folder and run:
```bash
python desktop_app.py
```

Because the `adaptive_model_final.pth` checkpoint is already in the repository, the neural network will automatically load and the developer dashboard will open immediately. You are ready to go!
