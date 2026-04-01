from collections import Counter, deque
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from statistics import mean

import numpy as np
from deepface import DeepFace

from src.backend.infrastructure.ai.config import FaceRecognitionConfig


@dataclass
class GalleryEmbedding:
    student_id: str
    photo_path: Path
    embedding: np.ndarray


@dataclass
class StudentMatchCandidate:
    student_id: str
    best_distance: float
    mean_best_distance: float
    support_count: int
    newest_mtime: float


@dataclass
class TrackIdentityState:
    votes: deque[str] = field(default_factory=deque)
    confirmed_student_id: str | None = None


class FaceRecognizer:
    """
    Распознавание лиц через FaceNet-эмбеддинги с локальной галереей и
    временной стабилизацией по track_id.
    """

    def __init__(
        self,
        db_path: str,
        config: FaceRecognitionConfig | None = None,
    ):
        self.db_path = Path(db_path)
        self.config = config or FaceRecognitionConfig()
        self.identity_stability: dict[int, TrackIdentityState] = {}
        self.gallery: list[GalleryEmbedding] = []

        self.db_path.mkdir(parents=True, exist_ok=True)

        print(
            "[AI] Инициализация FaceNet-распознавания. "
            f"Папка с лицами: {self.db_path}"
        )
        self._log_runtime()

        try:
            DeepFace.build_model(self.config.model_name)
        except Exception as e:
            print(f"[AI] Ошибка загрузки весов модели: {e}")

        self.refresh_db()

    def recognize(self, face_img, track_id=None):
        """
        Возвращает student_id при уверенном совпадении, иначе None.
        """
        if face_img is None or face_img.size == 0:
            return None

        if (
            face_img.shape[0] < self.config.min_face_size
            or face_img.shape[1] < self.config.min_face_size
        ):
            return None

        try:
            if not self.gallery:
                return None

            embedding = self._extract_embedding(face_img)
            if embedding is None:
                return None

            student_id = self._match_embedding(embedding)
            if track_id is None:
                return student_id
            return self._stabilize_identity(track_id, student_id)
        except Exception as e:
            print(f"[AI] Критическая ошибка DeepFace: {e}")
            return None

    def refresh_db(self):
        self.gallery = []
        self.identity_stability.clear()

        for cache_file in self.db_path.glob("representations_*.pkl"):
            try:
                cache_file.unlink()
            except OSError:
                pass

        for photo_path in sorted(self.db_path.iterdir()):
            if photo_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            embedding = self._extract_embedding(str(photo_path))
            if embedding is None:
                continue

            student_id = self._student_id_from_path(photo_path)
            self.gallery.append(
                GalleryEmbedding(
                    student_id=student_id,
                    photo_path=photo_path,
                    embedding=embedding,
                )
            )

        print(f"[AI] Галерея FaceNet обновлена: {len(self.gallery)} эмбеддингов")

    def forget_track(self, track_id: int) -> None:
        self.identity_stability.pop(track_id, None)

    def detect_faces(self, frame) -> list[dict]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        backends = [self.config.detector_backend]
        if self.config.detector_backend != "opencv":
            backends.append("opencv")

        for backend in backends:
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                    normalize_face=False,
                )
            except Exception as error:
                print(f"[AI] Ошибка детекции лица ({backend}): {error}")
                continue

            prepared_faces = self._prepare_detected_faces(frame, faces)
            if prepared_faces:
                return prepared_faces

        return []

    def _extract_embedding(self, image_source) -> np.ndarray | None:
        represent_signature = signature(DeepFace.represent)
        backends = [self.config.detector_backend]
        if self.config.detector_backend != "skip":
            backends.append("skip")

        for backend in backends:
            try:
                kwargs = {
                    "img_path": image_source,
                    "model_name": self.config.model_name,
                    "detector_backend": backend,
                    "enforce_detection": False,
                    "align": True,
                    "normalization": self.config.normalization,
                }
                if "silent" in represent_signature.parameters:
                    kwargs["silent"] = True
                representations = DeepFace.represent(**kwargs)
            except Exception as error:
                print(
                    "[AI] Ошибка извлечения эмбеддинга "
                    f"({backend}) для {image_source}: {error}"
                )
                continue

            if not representations:
                continue

            best_face = max(
                representations,
                key=lambda item: float(item.get("face_confidence", 0.0)),
            )
            embedding = np.asarray(best_face.get("embedding"), dtype=np.float32)

            if embedding.size == 0:
                continue

            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue

            return embedding / norm

        return None

    def _match_embedding(self, embedding: np.ndarray) -> str | None:
        candidates = self._build_student_candidates(embedding)
        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                -item.support_count,
                item.best_distance,
                item.mean_best_distance,
                -item.newest_mtime,
            )
        )

        best = candidates[0]
        if len(candidates) == 1:
            return best.student_id

        second = candidates[1]
        if best.support_count > second.support_count:
            return best.student_id

        if (
            abs(best.best_distance - second.best_distance) < 1e-6
            and abs(best.mean_best_distance - second.mean_best_distance) < 1e-6
        ):
            print(
                "[AI] Найдены дублирующие регистрации одного лица. "
                f"Выбираю более новую запись: {best.student_id}"
            )
            return best.student_id

        if (second.best_distance - best.best_distance) < self.config.min_margin:
            return None

        return best.student_id

    def _stabilize_identity(
        self,
        track_id: int,
        student_id: str | None,
    ) -> str | None:
        state = self.identity_stability.get(track_id)
        if state is None:
            state = TrackIdentityState(
                votes=deque(maxlen=self.config.vote_window),
            )
            self.identity_stability[track_id] = state

        if state.confirmed_student_id and (
            student_id is None or student_id == state.confirmed_student_id
        ):
            return state.confirmed_student_id

        if student_id is None:
            return None

        state.votes.append(student_id)
        candidate, count = Counter(state.votes).most_common(1)[0]
        if count >= self.config.min_stable_votes:
            state.confirmed_student_id = candidate
            state.votes.clear()
            state.votes.append(candidate)
            return candidate

        return None

    @staticmethod
    def _student_id_from_path(photo_path: Path) -> str:
        stem = photo_path.stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[0]

    @staticmethod
    def _cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(1.0 - np.clip(np.dot(first, second), -1.0, 1.0))

    def _build_student_candidates(
        self,
        embedding: np.ndarray,
    ) -> list[StudentMatchCandidate]:
        grouped_distances: dict[str, list[tuple[float, Path]]] = {}
        for record in self.gallery:
            distance = self._cosine_distance(embedding, record.embedding)
            grouped_distances.setdefault(record.student_id, []).append(
                (distance, record.photo_path)
            )

        candidates: list[StudentMatchCandidate] = []
        for student_id, entries in grouped_distances.items():
            ordered_entries = sorted(entries, key=lambda item: item[0])
            best_distance = ordered_entries[0][0]
            if best_distance > self.config.distance_threshold:
                continue

            top_distances = [
                distance
                for distance, _ in ordered_entries[: min(2, len(ordered_entries))]
            ]
            support_count = sum(
                1
                for distance, _ in ordered_entries
                if distance <= self.config.distance_threshold
            )
            newest_mtime = max(path.stat().st_mtime for _, path in ordered_entries)

            candidates.append(
                StudentMatchCandidate(
                    student_id=student_id,
                    best_distance=best_distance,
                    mean_best_distance=float(mean(top_distances)),
                    support_count=support_count,
                    newest_mtime=newest_mtime,
                )
            )

        return candidates

    @staticmethod
    def _prepare_detected_faces(frame, faces: list[dict]) -> list[dict]:
        height, width = frame.shape[:2]
        prepared_faces = []

        for face in faces:
            area = face.get("facial_area") or {}
            x = int(area.get("x", 0))
            y = int(area.get("y", 0))
            w = int(area.get("w", 0))
            h = int(area.get("h", 0))
            if w <= 0 or h <= 0:
                continue

            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(0, min(x + w, width))
            y2 = max(0, min(y + h, height))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if getattr(crop, "size", 0) == 0:
                continue

            prepared_faces.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop,
                    "confidence": float(face.get("confidence", 0.0)),
                }
            )

        return prepared_faces

    @staticmethod
    def _log_runtime() -> None:
        try:
            import tensorflow as tf
        except Exception:
            print("[AI] FaceNet device: tensorflow unavailable")
            return

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[AI] FaceNet device: GPU ({gpus[0].name})")
            return

        print("[AI] FaceNet device: CPU")
