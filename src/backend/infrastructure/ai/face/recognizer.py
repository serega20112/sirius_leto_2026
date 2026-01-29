import os
import pandas as pd
from deepface import DeepFace
import numpy as np


class FaceRecognizer:
    """Сервис распознавания лиц через DeepFace."""

    def __init__(self, db_path: str):
        """
        db_path: Путь к папке с фото студентов (assets/images).
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

    def recognize(self, face_img: np.ndarray) -> str | None:
        """
        Пытается найти лицо в базе данных.
        Возвращает имя файла (например, 'student_id.jpg') или None.
        """
        try:
            dfs = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name="VGG-Face",
                detector_backend="opencv",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True
            )

            if len(dfs) > 0 and not dfs[0].empty:
                full_path = dfs[0].iloc[0]["identity"]
                return os.path.basename(full_path)

            return None

        except Exception as e:
            return(f"{e} ошибочка вышла XD")

    def refresh_db(self):
        """Удаляет кэш DeepFace, чтобы подхватить новые фото."""
        pkl_path = os.path.join(self.db_path, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)