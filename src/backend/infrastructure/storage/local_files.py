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
        """
        Saves image.
        
        Args:
            file_bytes: Input value for `file_bytes`.
            filename: Input value for `filename`.
        
        Returns:
            The result of the operation.
        """
        file_path = self.base_path / filename

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return str(file_path.absolute())

    def delete_image(self, filename: str) -> None:
        """
        Deletes image.
        
        Args:
            filename: Input value for `filename`.
        
        Returns:
            Does not return a value.
        """
        file_path = self.base_path / filename
        if file_path.exists():
            os.remove(file_path)
