import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "VocalMood: Speech Emotion & Voice Journal"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = ROOT_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    DATABASE_URL: str = f"sqlite:///{ROOT_DIR}/data/voice_journal.db"
    UPLOAD_DIR: Path = ROOT_DIR / "data" / "audio_uploads"
    MODEL_DIR: Path = ROOT_DIR / "models"
    MODEL_PATH: Path = ROOT_DIR / "models" / "cnn_lstm_ser.pt"
    
    # Audio Feature Parameters
    SAMPLE_RATE: int = 16000
    N_MFCC: int = 40
    N_MELS: int = 64
    HOP_LENGTH: int = 512
    N_FFT: int = 1024
    MAX_AUDIO_DURATION_SEC: float = 60.0
    TARGET_FRAMES: int = 128  # Fixed time frame length for CNN-LSTM input
    
    # Emotional Classes (RAVDESS standard)
    EMOTIONS: list = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised"
    ]
    
    # Optional External APIs
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    model_config = ConfigDict(extra="ignore")

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
