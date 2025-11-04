from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    DATA_DIR: Path = Path("data/raw")
    PROCESSED_DIR: Path = Path("data/processed")
    MODEL_DIR: Path = Path("models")
    GOOGLE_API_KEY: str = ""

    DIABETES_PATH: Path = DATA_DIR / "diabetes.csv"
    HEART_PATH: Path = DATA_DIR / "heart.csv"
    KIDNEY_PATH: Path = DATA_DIR / "kidney_disease.csv"

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    SMOTE_K_NEIGHBORS: int = 5
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()