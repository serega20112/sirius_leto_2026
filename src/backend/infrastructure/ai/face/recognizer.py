import os
from deepface import DeepFace


class FaceRecognizer:
    """Сервис распознавания лиц через DeepFace."""

    def __init__(self, db_path: str):
        """Инициализация модели и путей базы лиц."""
        self.db_path = db_path
        self.model_name = "Facenet"

        os.makedirs(db_path, exist_ok=True)

        try:
            DeepFace.build_model(self.model_name)
        except Exception as e:
            print(f"[AI] Ошибка загрузки модели: {e}")

    def recognize(self, face_img):
        """Распознавание лица на переданном изображении."""
        try:
            results = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )

            if results and len(results) > 0 and not results[0].empty:
                full_path = results[0].iloc[0]["identity"]
                identity = os.path.basename(full_path)
                print(f"[AI] Распознан: {identity}")
                return identity

            return None
        except Exception as e:
            print(f"[AI] Ошибка при распознавании: {e}")
            return None

    def refresh_db(self):
        """Удаление кэша базы лиц для обновления данных."""
        pkl_path = os.path.join(
            self.db_path, f"representations_{self.model_name.lower()}.pkl"
        )
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
