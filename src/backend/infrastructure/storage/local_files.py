import os
import shutil
from pathlib import Path


class LocalFileStorage:
    """Сервис для сохранения файлов на локальный диск."""

    def __init__(self, base_path: str):
        """Инициализация с базовой директорией (assets/images)."""
        self.base_path = Path(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def save_image(self, file_bytes: bytes, filename: str) -> str:
        """Сохраняет байты изображения и возвращает полный путь к файлу."""
        file_path = self.base_path / filename

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return str(file_path.absolute())

    def delete_image(self, filename: str) -> None:
        """Удаляет файл, если он существует."""
        file_path = self.base_path / filename
        if file_path.exists():
            os.remove(file_path)