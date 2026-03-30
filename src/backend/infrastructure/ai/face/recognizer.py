import os
import cv2
from deepface import DeepFace


class FaceRecognizer:
    """
    Распознавание лиц через DeepFace с контролем дистанции и стабилизацией
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_name = "SFace"
        self.distance_threshold = 0.55
        self.identity_stability = {}

        os.makedirs(db_path, exist_ok=True)

        print(f"[AI] Инициализация DeepFace. Папка с лицами: {self.db_path}")

        try:
            DeepFace.build_model(self.model_name)
        except Exception as e:
            print(f"[AI] Ошибка загрузки весов модели: {e}")

    def recognize(self, face_img, track_id=None):
        """
        face_img: numpy массив лица
        track_id: id трека (для стабилизации)
        """

        if face_img is None or face_img.size == 0:
            return None

        if face_img.shape[0] < 80 or face_img.shape[1] < 80:
            return None

        try:
            results = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                distance_metric="cosine",
                silent=True,
            )

            if results and len(results) > 0 and not results[0].empty:
                match = results[0].iloc[0]

                full_path = match["identity"]
                distance = float(match["distance"])

                if distance > self.distance_threshold:
                    return None

                student_id = os.path.basename(full_path)

                if track_id is not None:
                    stable = self.identity_stability.get(track_id)

                    if stable is None:
                        self.identity_stability[track_id] = {
                            "id": student_id,
                            "count": 1,
                        }
                        return None

                    if stable["id"] == student_id:
                        stable["count"] += 1
                    else:
                        stable["count"] = 0
                        stable["id"] = student_id

                    if stable["count"] >= 2:
                        return student_id

                    return None

                return student_id

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