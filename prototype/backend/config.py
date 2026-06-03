from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Adaptive Threat Detection"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_PATH: Path = Path(__file__).parent.parent.parent
    # Use the final self-adaptive expanded model
    CHECKPOINT_PATH: Path = BASE_PATH / "adaptive_model_final.pth"
    # Older versions: 
    # CHECKPOINT_PATH: Path = BASE_PATH / "new_custom_model.pth"
    # CHECKPOINT_PATH: Path = BASE_PATH / "checkpoints" / "baseline_model.pth"
    TEMP_DIR: Path = Path(__file__).parent / "temp"
    
    # Database
    DATA_DIR: Path = BASE_PATH / "data"
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./threat_detection.db"
    
    # App Settings
    DEBUG: bool = True
    
    # GitHub Actions Integration (for server-side retraining)
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = "" # Format: "owner/repo"
    
    class Config:
        case_sensitive = True

settings = Settings()
settings.TEMP_DIR.mkdir(exist_ok=True)
