from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import sys
import logging
from pathlib import Path
from fastapi.security import OAuth2PasswordRequestForm
from typing import List

# Add project root to sys.path to import git_threat_protector
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from .database import engine, Base, get_db
from .models import Scan, SystemMetric
from .scanner import scanner
from .config import settings
from .auth import get_current_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, FAKE_USERS_DB, verify_password, get_user
from datetime import timedelta

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Metrics if they don't exist
def init_metrics():
    db = next(get_db())
    if db.query(SystemMetric).count() == 0:
        db.add(SystemMetric(total_scans=1245, threats_blocked=87))
        db.commit()
    db.close()

@app.on_event("startup")
async def startup_event():
    init_metrics()
    logger.info("Database initialized and metrics seeded.")

@app.get("/")
async def root():
    return {"status": "ready", "engine": "Dual-Branch AI", "version": "2.0.0"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(FAKE_USERS_DB, form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/scan")
async def scan_file(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    if not scanner.ready:
        raise HTTPException(status_code=503, detail="AI Engine not ready")
        
    temp_path = settings.TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with open(temp_path, 'rb') as f:
            binary_data = f.read()
            
        # Run Scan
        result = scanner.scan(binary_data, file.filename)
        
        # Save to DB
        db_scan = Scan(
            filename=file.filename,
            prediction=result['prediction'],
            confidence=result['confidence'],
            static_analysis=result['static_analysis'],
            dynamic_analysis=result['dynamic_analysis'],
            fusion_strategy=result['fusion_strategy']
        )
        db.add(db_scan)
        
        # Update Stats
        metric = db.query(SystemMetric).first()
        metric.total_scans += 1
        if result['prediction'] == 'Malware':
            metric.threats_blocked += 1
            
        db.commit()
        db.refresh(db_scan)
        
        return {
            "id": db_scan.id,
            "filename": db_scan.filename,
            "prediction": db_scan.prediction,
            "confidence": db_scan.confidence,
            "analysis": {
                "static": db_scan.static_analysis,
                "dynamic": db_scan.dynamic_analysis,
                "fusion": db_scan.fusion_strategy
            },
            "timestamp": db_scan.timestamp
        }
    except Exception as e:
        logger.error(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)

@app.get("/history")
async def get_history(limit: int = 10, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    scans = db.query(Scan).order_by(Scan.timestamp.desc()).limit(limit).all()
    return scans

@app.get("/system/stats")
async def get_stats(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    metric = db.query(SystemMetric).first()
    return {
        "total_scans": metric.total_scans,
        "threats_blocked": metric.threats_blocked,
        "model_accuracy": metric.model_accuracy,
        "last_evolution": metric.last_evolution,
        "system_health": "Optimal"
    }

@app.post("/scans/{scan_id}/feedback")
async def provide_feedback(scan_id: int, correct_label: str, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if correct_label not in ["Benign", "Malware"]:
        raise HTTPException(status_code=400, detail="Invalid label")
        
    scan.correct_label = correct_label
    db.commit()
    return {"status": "success", "message": f"Feedback received: {correct_label}"}

@app.post("/system/retrain")
async def trigger_retrain(strategy: str = None, epochs: int = 5, current_user: dict = Depends(get_current_user)):
    # Check if GitHub Actions is configured
    if settings.GITHUB_TOKEN and settings.GITHUB_REPO:
        import requests
        headers = {
            "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        url = f"https://api.github.com/repos/{settings.GITHUB_REPO}/actions/workflows/retrain.yml/dispatches"
        data = {
            "ref": "main",
            "inputs": {
                "strategy": strategy or "random_insert",
                "epochs": str(epochs)
            }
        }
        try:
            resp = requests.post(url, json=data, headers=headers)
            if resp.status_code in [204, 201, 200]:
                return {
                    "status": "github_triggered",
                    "message": "GitHub Actions retraining workflow triggered successfully.",
                    "repo": settings.GITHUB_REPO
                }
            else:
                return {
                    "status": "error",
                    "message": f"GitHub dispatch failed ({resp.status_code}): {resp.text}"
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GitHub API Error: {str(e)}")
            
    # Fallback to local subprocess training
    import subprocess
    import sys
    try:
        cmd = [sys.executable, "continuous_learning.py"]
        if strategy:
            cmd.extend(["--strategy", strategy])
        cmd.extend(["--epochs", str(epochs)])
        cmd.append("--quick")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            from .scanner import scanner
            scanner.reload_model()
            return {"status": "success", "output": result.stdout}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/sync-model")
async def sync_model(current_user: dict = Depends(get_current_user)):
    """Downloads the latest model weights from GitHub repository raw files and reloads scanner."""
    if not settings.GITHUB_REPO:
        raise HTTPException(status_code=400, detail="GITHUB_REPO setting not configured.")
        
    import requests
    url = f"https://raw.githubusercontent.com/{settings.GITHUB_REPO}/main/adaptive_model_final.pth"
    headers = {}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
        
    try:
        logger.info(f"Syncing model weights from: {url}")
        resp = requests.get(url, headers=headers, stream=True)
        if resp.status_code == 200:
            # Save to checkpoint path
            with open(settings.CHECKPOINT_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Reload scanner
            from .scanner import scanner
            scanner.reload_model()
            return {"status": "success", "message": "Model synced and reloaded successfully."}
        else:
            raise HTTPException(
                status_code=resp.status_code, 
                detail=f"Failed to fetch model from GitHub raw: {resp.text[:200]}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync Error: {str(e)}")

@app.get("/system/retrain-status")
async def get_retrain_status(current_user: dict = Depends(get_current_user)):
    """Checks the status of the latest Model Retraining GitHub Action workflow run."""
    if not settings.GITHUB_REPO or not settings.GITHUB_TOKEN:
        return {"status": "local_only", "message": "GitHub Integration not configured."}
        
    import requests
    url = f"https://api.github.com/repos/{settings.GITHUB_REPO}/actions/runs"
    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            # Find the latest Model Retraining workflow run
            retrain_runs = [r for r in runs if "retrain.yml" in r.get("path", "")]
            if not retrain_runs:
                return {"status": "idle", "message": "No retraining runs found."}
                
            latest_run = retrain_runs[0]
            return {
                "status": latest_run.get("status"), # e.g. "queued", "in_progress", "completed"
                "conclusion": latest_run.get("conclusion"), # e.g. "success", "failure", "cancelled"
                "html_url": latest_run.get("html_url"),
                "run_id": latest_run.get("id")
            }
        else:
            return {"status": "error", "message": f"GitHub API failed: {resp.text[:200]}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Git Integration Endpoints
@app.get("/git/status", dependencies=[Depends(get_current_user)])
async def get_git_status():
    pre_commit_path = Path(".git/hooks/pre-commit")
    
    # Get staged files
    staged_files = []
    try:
        from git_threat_protector import get_staged_files
        staged_files = get_staged_files()
    except Exception as e:
        logger.warning(f"Could not get staged files: {e}")
        
    return {
        "installed": pre_commit_path.exists(),
        "hook_path": str(pre_commit_path),
        "staged_files": staged_files,
        "staged_count": len(staged_files)
    }

@app.post("/git/install", dependencies=[Depends(get_current_user)])
async def post_git_install():
    try:
        from git_threat_protector import install_hook
        success = install_hook()
        if success:
            return {"status": "success", "message": "Pre-commit hook installed successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to install pre-commit hook.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/git/uninstall", dependencies=[Depends(get_current_user)])
async def post_git_uninstall():
    try:
        from git_threat_protector import uninstall_hook
        success = uninstall_hook()
        if success:
            return {"status": "success", "message": "Pre-commit hook uninstalled successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to uninstall pre-commit hook.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/git/scan-staged", dependencies=[Depends(get_current_user)])
async def post_git_scan_staged(db: Session = Depends(get_db)):
    try:
        from git_threat_protector import get_staged_files
        staged_files = get_staged_files()
        
        if not staged_files:
            return {"status": "success", "message": "No staged files to scan.", "scanned": 0, "results": []}
            
        results = []
        for file_path_str in staged_files:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue
                
            # Filter non-binary or safe source files
            from git_threat_protector import is_binary_file
            file_ext = file_path.suffix.lower()
            is_binary = is_binary_file(file_path)
            is_executable = file_ext in ['.exe', '.dll', '.bin', '.elf', '.sys', '.msi']
            
            if not (is_binary or is_executable):
                results.append({
                    "filename": file_path.name,
                    "filepath": file_path_str,
                    "prediction": "Skipped",
                    "confidence": 1.0,
                    "reason": "Non-binary / source file skipped"
                })
                continue
                
            # Read and scan
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            scan_res = scanner.scan(binary_data, file_path.name)
            
            # Save to DB
            db_scan = Scan(
                filename=file_path.name,
                prediction=scan_res['prediction'],
                confidence=scan_res['confidence'],
                static_analysis=scan_res['static_analysis'],
                dynamic_analysis=scan_res['dynamic_analysis'],
                fusion_strategy=scan_res['fusion_strategy']
            )
            db.add(db_scan)
            
            # Update metrics
            metric = db.query(SystemMetric).first()
            if metric:
                metric.total_scans += 1
                if scan_res['prediction'] == 'Malware':
                    metric.threats_blocked += 1
            
            results.append({
                "filename": file_path.name,
                "filepath": file_path_str,
                "prediction": scan_res['prediction'],
                "confidence": scan_res['confidence'],
                "details": scan_res
            })
            
        db.commit()
        return {"status": "success", "scanned": len(results), "results": results}
    except Exception as e:
        logger.error(f"Git scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
