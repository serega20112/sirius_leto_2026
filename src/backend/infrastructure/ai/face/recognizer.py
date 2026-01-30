from deepface import DeepFace
import os


class FaceRecognizer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        DeepFace.build_model("Facenet")

    def recognize(self, face_img):
        try:
            results = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name="FaceNet",
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            if results and not results[0].empty:
                return os.path.basename(results[0].iloc[0]["identity"])
        except:
            return None
        return None

    def refresh_db(self):
        """Удаляет кэш DeepFace, чтобы подхватить новые фото."""
        pkl_path = os.path.join(self.db_path, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
