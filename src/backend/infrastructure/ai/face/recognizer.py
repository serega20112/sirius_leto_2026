from collections import Counter, deque
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from src.backend.infrastructure.ai.config import FaceRecognitionConfig

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
except Exception:
    InceptionResnetV1 = None
    MTCNN = None


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
    Provides face recognition with GPU-first FaceNet embeddings and a stable local gallery.
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
        self.backend_name = "deepface"
        self.device = self._resolve_device()
        self.mtcnn = None
        self.facenet_model = None
        self.deepface = None

        self.db_path.mkdir(parents=True, exist_ok=True)

        print(
            "[AI] Initializing face recognition. "
            f"Faces directory: {self.db_path}"
        )

        self._initialize_backend()
        self._log_runtime()
        self.refresh_db()

    def recognize(self, face_img, track_id=None):
        """
        Recognizes a person from a face crop and stabilizes the identity across frames.

        Args:
            face_img: Face crop or image source for the current person.
            track_id: Optional tracker identifier used for temporal stabilization.

        Returns:
            The recognized student identifier or `None` when no stable match is found.
        """
        if face_img is None or getattr(face_img, "size", 0) == 0:
            return None

        if (
            hasattr(face_img, "shape")
            and (
                face_img.shape[0] < self.config.min_face_size
                or face_img.shape[1] < self.config.min_face_size
            )
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
        except Exception as error:
            print(f"[AI] Critical face recognition error: {error}")
            return None

    def refresh_db(self):
        """
        Rebuilds the embedding gallery from all stored student photos.

        Args:
            None.

        Returns:
            Does not return a value.
        """
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

            embedding = self._extract_embedding(photo_path)
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

        print(
            "[AI] Face gallery refreshed: "
            f"{len(self.gallery)} embeddings | backend: {self.backend_name}"
        )

    def forget_track(self, track_id: int) -> None:
        """
        Removes cached stabilization state for the given tracker identifier.

        Args:
            track_id: Tracker identifier that should be cleared.

        Returns:
            Does not return a value.
        """
        self.identity_stability.pop(track_id, None)

    def detect_faces(self, frame) -> list[dict]:
        """
        Detects faces on a frame and returns prepared crops with bounding boxes.

        Args:
            frame: Full video frame in OpenCV format.

        Returns:
            A list of detected faces with bounding boxes, crops, and confidence.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        if self.backend_name == "facenet_pytorch":
            faces = self._detect_faces_facenet(frame)
            if faces or self.config.runtime_backend != "auto":
                return faces

        return self._detect_faces_deepface(frame)

    def _initialize_backend(self) -> None:
        """
        Initializes the preferred recognition backend and keeps a CPU fallback.

        Args:
            None.

        Returns:
            Does not return a value.
        """
        preferred_backend = str(self.config.runtime_backend).strip().lower()
        if preferred_backend in {"auto", "facenet_pytorch"} and self._try_init_facenet():
            self.backend_name = "facenet_pytorch"
            return

        if preferred_backend == "facenet_pytorch":
            print("[AI] facenet-pytorch unavailable, falling back to DeepFace")

        self._try_init_deepface()

    def _try_init_facenet(self) -> bool:
        """
        Initializes the facenet-pytorch stack when the package is available.

        Args:
            None.

        Returns:
            `True` when the backend is ready, otherwise `False`.
        """
        if MTCNN is None or InceptionResnetV1 is None:
            return False

        try:
            self.mtcnn = MTCNN(
                image_size=self.config.embedding_image_size,
                margin=self.config.embedding_margin,
                keep_all=True,
                post_process=True,
                device=self.device,
            )
            self.facenet_model = InceptionResnetV1(
                pretrained=self.config.embedding_model_name
            ).eval().to(self.device)
        except Exception as error:
            print(f"[AI] facenet-pytorch init error: {error}")
            self.mtcnn = None
            self.facenet_model = None
            return False

        return True

    def _try_init_deepface(self) -> None:
        """
        Initializes the DeepFace fallback backend when available.

        Args:
            None.

        Returns:
            Does not return a value.
        """
        deepface_module = self._get_deepface()
        if deepface_module is None:
            print("[AI] DeepFace fallback unavailable")
            return

        try:
            deepface_module.build_model(self.config.model_name)
        except Exception as error:
            print(f"[AI] DeepFace model init error: {error}")

        self.backend_name = "deepface"

    def _extract_embedding(self, image_source) -> np.ndarray | None:
        """
        Extracts a normalized embedding from the selected backend.

        Args:
            image_source: Image path or image array containing a face.

        Returns:
            A normalized embedding vector or `None` when extraction fails.
        """
        if self.backend_name == "facenet_pytorch":
            embedding = self._extract_embedding_facenet(image_source)
            if embedding is not None or self.config.runtime_backend != "auto":
                return embedding

        return self._extract_embedding_deepface(image_source)

    def _extract_embedding_facenet(self, image_source) -> np.ndarray | None:
        """
        Extracts a FaceNet embedding with facenet-pytorch on the configured device.

        Args:
            image_source: Image path or image array containing a face.

        Returns:
            A normalized embedding vector or `None` when extraction fails.
        """
        if self.mtcnn is None or self.facenet_model is None:
            return None

        image = self._prepare_image_source(image_source)
        if image is None:
            return None

        try:
            boxes, probabilities = self.mtcnn.detect(image)
        except Exception as error:
            print(f"[AI] facenet-pytorch detection error: {error}")
            return None

        boxes_array, probabilities_array = self._normalize_detection_output(
            boxes,
            probabilities,
        )
        if boxes_array is None or probabilities_array is None:
            return None

        try:
            faces = self.mtcnn.extract(image, boxes_array, None)
        except Exception as error:
            print(f"[AI] facenet-pytorch crop error: {error}")
            return None

        face_tensor = self._select_face_tensor(faces, probabilities_array)
        if face_tensor is None:
            return None

        try:
            with torch.inference_mode():
                embedding = self.facenet_model(face_tensor.unsqueeze(0).to(self.device))
                embedding = functional.normalize(embedding, p=2, dim=1)
        except Exception as error:
            print(f"[AI] facenet-pytorch embedding error: {error}")
            return None

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _extract_embedding_deepface(self, image_source) -> np.ndarray | None:
        """
        Extracts a FaceNet embedding with the DeepFace fallback backend.

        Args:
            image_source: Image path or image array containing a face.

        Returns:
            A normalized embedding vector or `None` when extraction fails.
        """
        deepface_module = self._get_deepface()
        if deepface_module is None:
            return None

        represent_signature = signature(deepface_module.represent)
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
                representations = deepface_module.represent(**kwargs)
            except Exception as error:
                print(
                    "[AI] DeepFace embedding error "
                    f"({backend}) for {image_source}: {error}"
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

    def _detect_faces_facenet(self, frame) -> list[dict]:
        """
        Detects faces on a frame with facenet-pytorch and prepares OpenCV crops.

        Args:
            frame: Full video frame in OpenCV format.

        Returns:
            A list of detected faces with crops and confidence values.
        """
        if self.mtcnn is None:
            return []

        image = self._prepare_image_source(frame)
        if image is None:
            return []

        try:
            boxes, probabilities = self.mtcnn.detect(image)
        except Exception as error:
            print(f"[AI] facenet-pytorch frame detection error: {error}")
            return []

        boxes_array, probabilities_array = self._normalize_detection_output(
            boxes,
            probabilities,
        )
        if boxes_array is None or probabilities_array is None:
            return []

        height, width = frame.shape[:2]
        prepared_faces = []
        for box, probability in zip(boxes_array, probabilities_array):
            if probability < self.config.min_face_confidence:
                continue

            x1, y1, x2, y2 = self._clamp_box(box, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if getattr(crop, "size", 0) == 0:
                continue

            prepared_faces.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop,
                    "confidence": float(probability),
                }
            )

        return prepared_faces

    def _detect_faces_deepface(self, frame) -> list[dict]:
        """
        Detects faces on a frame with the DeepFace fallback backend.

        Args:
            frame: Full video frame in OpenCV format.

        Returns:
            A list of detected faces with crops and confidence values.
        """
        deepface_module = self._get_deepface()
        if deepface_module is None:
            return []

        backends = [self.config.detector_backend]
        if self.config.detector_backend != "opencv":
            backends.append("opencv")

        for backend in backends:
            try:
                faces = deepface_module.extract_faces(
                    img_path=frame,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                )
            except Exception as error:
                print(f"[AI] DeepFace detect error ({backend}): {error}")
                continue

            prepared_faces = self._prepare_detected_faces(frame, faces)
            if prepared_faces:
                return prepared_faces

        return []

    def _match_embedding(self, embedding: np.ndarray) -> str | None:
        """
        Matches the current embedding against the gallery and resolves duplicates.

        Args:
            embedding: Normalized embedding for the current face.

        Returns:
            The best matching student identifier or `None`.
        """
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
                "[AI] Duplicate face registrations detected. "
                f"Choosing newest entry: {best.student_id}"
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
        """
        Stabilizes recognition decisions for a tracker over multiple frames.

        Args:
            track_id: Tracker identifier for the current person.
            student_id: Current raw recognition result.

        Returns:
            A stable student identifier or `None` until enough votes are collected.
        """
        state = self.identity_stability.get(track_id)
        if state is None:
            state = TrackIdentityState(votes=deque(maxlen=self.config.vote_window))
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
        """
        Converts a stored photo filename into the corresponding student identifier.

        Args:
            photo_path: Path to a stored student image.

        Returns:
            The student identifier parsed from the filename.
        """
        stem = photo_path.stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[0]

    @staticmethod
    def _cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
        """
        Computes cosine distance between two normalized embeddings.

        Args:
            first: First embedding vector.
            second: Second embedding vector.

        Returns:
            The cosine distance between the embeddings.
        """
        return float(1.0 - np.clip(np.dot(first, second), -1.0, 1.0))

    def _build_student_candidates(
        self,
        embedding: np.ndarray,
    ) -> list[StudentMatchCandidate]:
        """
        Builds grouped matching candidates for each student in the gallery.

        Args:
            embedding: Normalized embedding for the current face.

        Returns:
            A list of matching candidates that pass the configured threshold.
        """
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
        """
        Normalizes DeepFace detections into the internal face representation.

        Args:
            frame: Full video frame in OpenCV format.
            faces: Raw detections returned by DeepFace.

        Returns:
            A list of prepared face entries with crops and bounding boxes.
        """
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

    def _prepare_image_source(self, image_source: Any) -> Image.Image | None:
        """
        Converts supported image sources into a RGB PIL image.

        Args:
            image_source: Image path, PIL image, or OpenCV image array.

        Returns:
            A RGB PIL image or `None` when conversion fails.
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if isinstance(image_source, (str, Path)):
            try:
                return Image.open(image_source).convert("RGB")
            except Exception as error:
                print(f"[AI] Image open error for {image_source}: {error}")
                return None

        if not isinstance(image_source, np.ndarray):
            return None

        if image_source.size == 0:
            return None

        try:
            if image_source.ndim == 2:
                rgb_image = cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            elif image_source.ndim == 3 and image_source.shape[2] == 4:
                rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        except Exception as error:
            print(f"[AI] Image conversion error: {error}")
            return None

        return Image.fromarray(rgb_image)

    @staticmethod
    def _normalize_detection_output(
        boxes: Any,
        probabilities: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Normalizes detector outputs into two aligned numpy arrays.

        Args:
            boxes: Raw bounding boxes returned by the detector.
            probabilities: Raw detection confidence scores.

        Returns:
            A tuple with normalized boxes and probabilities or `(None, None)`.
        """
        if boxes is None or probabilities is None:
            return None, None

        boxes_array = np.asarray(boxes, dtype=np.float32)
        probabilities_array = np.asarray(probabilities, dtype=np.float32)
        if boxes_array.ndim == 1:
            boxes_array = np.expand_dims(boxes_array, axis=0)
        if probabilities_array.ndim == 0:
            probabilities_array = np.expand_dims(probabilities_array, axis=0)

        valid_indices = [
            index
            for index, (box, probability) in enumerate(
                zip(boxes_array, probabilities_array)
            )
            if np.isfinite(box).all() and np.isfinite(probability)
        ]
        if not valid_indices:
            return None, None

        return boxes_array[valid_indices], probabilities_array[valid_indices]

    @staticmethod
    def _select_face_tensor(
        faces: torch.Tensor | None,
        probabilities: np.ndarray,
    ) -> torch.Tensor | None:
        """
        Selects the highest-confidence aligned face tensor from detector output.

        Args:
            faces: Aligned face tensor batch produced by MTCNN.
            probabilities: Confidence scores for the aligned faces.

        Returns:
            A single aligned face tensor or `None` when no tensor is available.
        """
        if faces is None:
            return None

        if isinstance(faces, torch.Tensor) and faces.ndim == 3:
            return faces

        if not isinstance(faces, torch.Tensor) or faces.ndim != 4:
            return None

        best_index = int(np.argmax(probabilities))
        if best_index >= faces.shape[0]:
            return None

        return faces[best_index]

    @staticmethod
    def _clamp_box(
        box: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        """
        Clamps a floating-point detection box to valid image coordinates.

        Args:
            box: Floating-point bounding box.
            width: Frame width.
            height: Frame height.

        Returns:
            A clamped integer bounding box.
        """
        x1 = max(0, min(int(round(float(box[0]))), width - 1))
        y1 = max(0, min(int(round(float(box[1]))), height - 1))
        x2 = max(0, min(int(round(float(box[2]))), width))
        y2 = max(0, min(int(round(float(box[3]))), height))
        return x1, y1, x2, y2

    def _resolve_device(self) -> str:
        """
        Resolves the preferred inference device from configuration and hardware.

        Args:
            None.

        Returns:
            The device string that should be used for torch-based inference.
        """
        preferred_device = str(self.config.device).strip().lower()
        if preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred_device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_deepface(self):
        """
        Lazily imports and caches the DeepFace module for fallback inference.

        Args:
            None.

        Returns:
            The imported DeepFace module or `None` when it is unavailable.
        """
        if self.deepface is not None:
            return self.deepface

        try:
            from deepface import DeepFace
        except Exception as error:
            print(f"[AI] DeepFace import error: {error}")
            self.deepface = None
            return None

        self.deepface = DeepFace
        return self.deepface

    def _log_runtime(self) -> None:
        """
        Logs the selected recognition backend and execution device.

        Args:
            None.

        Returns:
            Does not return a value.
        """
        if self.backend_name == "facenet_pytorch":
            device_name = self.device
            if self.device == "cuda":
                try:
                    device_name = torch.cuda.get_device_name(0)
                except Exception:
                    device_name = "cuda"
            print(f"[AI] FaceNet backend: facenet_pytorch | device: {device_name}")
            return

        try:
            import tensorflow as tf
        except Exception:
            print("[AI] FaceNet backend: deepface | device: tensorflow unavailable")
            return

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[AI] FaceNet backend: deepface | device: GPU ({gpus[0].name})")
            return

        print("[AI] FaceNet backend: deepface | device: CPU")
