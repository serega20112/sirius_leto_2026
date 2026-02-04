import os
import cv2
from deepface import DeepFace


class FaceRecognizer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_name = "Facenet"

        os.makedirs(db_path, exist_ok=True)

        print(f"[AI] Инициализация DeepFace. Папка с лицами: {self.db_path}")
        # Предзагрузка, чтобы потом работало быстрее
        try:
            DeepFace.build_model(self.model_name)
        except Exception as e:
            print(f"[AI] Ошибка загрузки весов модели: {e}")

    def recognize(self, face_img):
        """
        face_img: это numpy массив (вырезанный квадрат лица)
        """
        # 1. Проверка на мусор
        if face_img is None or face_img.size == 0:
            return None

        # Если лицо слишком маленькое (меньше 50x50 пикселей), DeepFace его не поймет
        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            # print("[AI] Лицо слишком мелкое для распознавания")
            return None

        try:
            # 2. Поиск
            results = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend="skip",  # Мы верим, что YOLO вырезала лицо правильно
                enforce_detection=False,
                distance_metric="cosine",
                silent=True,
            )

            # 3. Анализ результата
            if results and len(results) > 0 and not results[0].empty:
                match = results[0].iloc[0]
                full_path = match["identity"]
                distance = match["distance"]

                # Если уверенность слабая (distance > 0.4 для Facenet - это уже сомнительно)
                # Но для теста пока выводим всё
                print(
                    f"[AI] НАШЕЛ! Файл: {os.path.basename(full_path)} | Дистанция: {distance:.4f}"
                )

                return os.path.basename(full_path)

            else:
                # print("[AI] Лицо четкое, но в базе нет совпадений")
                return None

        except Exception as e:
            print(f"[AI] Критическая ошибка DeepFace: {e}")
            return None

    def refresh_db(self):
        pkl_path = os.path.join(
            self.db_path, f"representations_{self.model_name.lower()}.pkl"
        )
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            print("[AI] Кэш базы лиц очищен")
