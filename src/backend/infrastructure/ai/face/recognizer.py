from __future__ import annotations

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from src.backend.infrastructure.ai.config import FaceRecognitionConfig
from src.backend.dependencies import settings


class FaceRecognizer:
    """Lightweight face recognizer using MediaPipe landmarks as compact embeddings.

    This implementation replaces heavy deep backends with a compact landmark-based
    embedding and cosine-distance matching. Designed for CPU-only edge devices.
    """

    def __init__(self, db_path: str, config: FaceRecognitionConfig | None = None):
        self.db_path = Path(db_path)
        self.config = config or FaceRecognitionConfig()
        self.gallery: List[tuple[str, Path, np.ndarray]] = []

        self.mp = None
        self.face_detection = None
        self.face_mesh = None

        try:
            import mediapipe as mp

            self.mp = mp
            # Use MP settings tuned for realtime by default (lower confidence than face model config)
            mp_conf = float(getattr(settings, "MP_MIN_DETECTION_CONFIDENCE", 0.5))
            mp_track_conf = float(getattr(settings, "MP_MIN_TRACKING_CONFIDENCE", 0.5))

            self.face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=mp_conf
            )
            # realtime mesh (tracking enabled) for lower latency
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=mp_conf,
                min_tracking_confidence=mp_track_conf,
            )
        except Exception:
            self.mp = None
            self.face_detection = None
            self.face_mesh = None

        self._selected_indices = [1, 33, 61, 199, 263, 291, 10, 152]

        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.refresh_db()

    def refresh_db(self) -> None:
        self.gallery = []
        for photo_path in sorted(self.db_path.iterdir()):
            if photo_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            emb = self._extract_embedding_from_image_path(photo_path)
            if emb is None:
                continue
            student_id = self._student_id_from_path(photo_path)
            self.gallery.append((student_id, photo_path, emb))

        print(
            f"[AI] Face gallery refreshed: {len(self.gallery)} embeddings | backend: light_face"
        )

    def detect_faces(self, frame: Any) -> list[dict]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        if self.face_detection is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb)
                if not results or not getattr(results, "detections", None):
                    return []

                prepared = []
                h, w = frame.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(max(0, min((bbox.xmin) * w, w - 1)))
                    y1 = int(max(0, min((bbox.ymin) * h, h - 1)))
                    x2 = int(max(0, min((bbox.xmin + bbox.width) * w, w)))
                    y2 = int(max(0, min((bbox.ymin + bbox.height) * h, h)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    conf = 0.0
                    try:
                        conf = (
                            float(det.score[0])
                            if getattr(det, "score", None) is not None
                            else 0.0
                        )
                    except Exception:
                        conf = 0.0
                    prepared.append(
                        {"bbox": [x1, y1, x2, y2], "crop": crop, "confidence": conf}
                    )
                return prepared
            except Exception:
                return []

        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            prepared = []
            for x, y, w_f, h_f in faces:
                x1, y1, x2, y2 = x, y, x + w_f, y + h_f
                crop = frame[y1:y2, x1:x2]
                prepared.append(
                    {"bbox": [x1, y1, x2, y2], "crop": crop, "confidence": 1.0}
                )
            return prepared
        except Exception:
            return []

    def recognize(self, face_img: Any, track_id: int | None = None) -> str | None:
        emb = self._extract_embedding(face_img)
        if emb is None or len(self.gallery) == 0:
            return None

        best = None
        best_dist = float("inf")
        for student_id, _, gemb in self.gallery:
            d = 1.0 - float(np.clip(np.dot(emb, gemb), -1.0, 1.0))
            if d < best_dist:
                best_dist = d
                best = student_id

        if best is None:
            return None

        if best_dist > float(self.config.distance_threshold):
            return None

        return best

    def forget_track(self, track_id: int) -> None:
        return

    def _extract_embedding_from_image_path(self, path: Path) -> np.ndarray | None:
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None
            # For gallery images use a static mesh for more stable landmarks if available
            if self.mp is not None and self.face_mesh is not None:
                mesh_backup = self.face_mesh
                try:
                    mp_conf = float(
                        getattr(settings, "MP_MIN_DETECTION_CONFIDENCE", 0.5)
                    )
                    mp_track_conf = float(
                        getattr(settings, "MP_MIN_TRACKING_CONFIDENCE", 0.5)
                    )
                    self.face_mesh = self.mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=mp_conf,
                        min_tracking_confidence=mp_track_conf,
                    )
                    emb = self._extract_embedding(img)
                finally:
                    self.face_mesh = mesh_backup
                return emb

            return self._extract_embedding(img)
        except Exception:
            return None

    def _extract_embedding(self, image: Any) -> np.ndarray | None:
        if image is None:
            return None

        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            if image is None:
                return None

        if not isinstance(image, np.ndarray):
            return None

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

        if self.face_mesh is None:
            try:
                small = cv2.resize(rgb, (64, 64))
                gray = (
                    cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                )
                vec = gray.flatten()
                norm = np.linalg.norm(vec)
                if norm == 0:
                    return None
                return (vec / norm).astype(np.float32)
            except Exception:
                return None

        try:
            results = self.face_mesh.process(rgb)
            if not results or not getattr(results, "multi_face_landmarks", None):
                return None
            lm = results.multi_face_landmarks[0]
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width = max(1e-6, maxx - minx)
            height = max(1e-6, maxy - miny)
            points = []
            for idx in self._selected_indices:
                if idx >= len(lm.landmark):
                    points.extend([0.0, 0.0])
                    continue
                p = lm.landmark[idx]
                nx = (p.x - minx) / width
                ny = (p.y - miny) / height
                points.extend([nx, ny])

            vec = np.asarray(points, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None
            return (vec / norm).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _student_id_from_path(photo_path: Path) -> str:
        stem = photo_path.stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[0]
