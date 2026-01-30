from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Настройки приложения."""

    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    ASSETS_DIR = BACKEND_ROOT / "assets"

    IMAGES_DIR = ASSETS_DIR / "images"

    YOLO_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n.pt")
    DB_PATH = ASSETS_DIR / "database" / "attendance.db"
