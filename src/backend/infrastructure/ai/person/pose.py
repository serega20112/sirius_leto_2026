from collections import deque

import cv2
import numpy as np

from src.backend.infrastructure.ai.config import EngagementConfig


HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


class PoseEstimator:
    def __init__(self, config: EngagementConfig | None = None):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Для оценки вовлеченности нужен пакет mediapipe."
            ) from exc

        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is None:
            raise RuntimeError(
                "Установленный mediapipe "
                f"{getattr(mp, '__version__', 'unknown')} "
                "не содержит legacy API mp.solutions. "
                "Текущая реализация engagement использует Face Mesh и Pose "
                "из mp.solutions, поэтому с этим wheel она не запустится."
            )

        self.config = config or EngagementConfig()
        self.mp = mp
        self.face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.track_scores: dict[int, deque[float]] = {}

    def estimate_engagement(self, frame, bbox, track_id=None):
        if bbox is None:
            return "unknown"

        sanitized_bbox = self._sanitize_bbox(frame, bbox)
        if sanitized_bbox is None:
            return "unknown"

        x1, y1, x2, y2 = sanitized_bbox
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return "unknown"

        head_bottom = y1 + int((y2 - y1) * self.config.head_crop_ratio)
        face_roi = frame[y1:head_bottom, x1:x2]

        face_score = self._estimate_face_attention(face_roi)
        body_score = self._estimate_body_attention(person_roi)

        weighted_scores = []
        if face_score is not None:
            weighted_scores.append((face_score, 0.7))
        if body_score is not None:
            weighted_scores.append((body_score, 0.3))

        if not weighted_scores:
            return "unknown"

        score = sum(value * weight for value, weight in weighted_scores) / sum(
            weight for _, weight in weighted_scores
        )
        score = self._smooth_score(score, track_id)
        return self._score_to_label(score)

    def forget_track(self, track_id: int) -> None:
        self.track_scores.pop(track_id, None)

    def _estimate_face_attention(self, face_roi) -> float | None:
        if face_roi is None or face_roi.size == 0:
            return None

        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_face)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        image_points = self._build_face_image_points(landmarks, face_roi.shape[1], face_roi.shape[0])
        if image_points is None:
            return None

        focal_length = float(face_roi.shape[1])
        camera_matrix = np.array(
            [
                [focal_length, 0.0, face_roi.shape[1] / 2],
                [0.0, focal_length, face_roi.shape[0] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        success, rotation_vector, _ = cv2.solvePnP(
            HEAD_POSE_MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        pitch = float(angles[0])
        yaw = float(angles[1])

        yaw_score = 1.0 - self._clamp(
            abs(yaw) / self.config.max_yaw_degrees,
            0.0,
            1.0,
        )
        pitch_score = 1.0 - self._clamp(
            abs(pitch) / self.config.max_pitch_degrees,
            0.0,
            1.0,
        )
        iris_score = self._estimate_iris_focus(landmarks)
        if iris_score is None:
            return 0.6 * yaw_score + 0.4 * pitch_score

        return 0.45 * yaw_score + 0.25 * pitch_score + 0.30 * iris_score

    def _estimate_body_attention(self, person_roi) -> float | None:
        rgb_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_person)
        if result.pose_landmarks is None:
            return None

        landmarks = result.pose_landmarks.landmark
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        visibility = float(
            np.mean([nose.visibility, left_shoulder.visibility, right_shoulder.visibility])
        )
        if visibility < self.config.min_body_visibility:
            return None

        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        shoulder_balance = 1.0 - self._clamp(
            abs(left_shoulder.y - right_shoulder.y) / 0.15,
            0.0,
            1.0,
        )
        torso_center = (left_shoulder.x + right_shoulder.x) / 2
        center_score = 1.0 - self._clamp(abs(nose.x - torso_center) / 0.2, 0.0, 1.0)
        width_score = self._clamp(shoulder_width / 0.35, 0.0, 1.0)

        return (
            0.45 * visibility
            + 0.20 * shoulder_balance
            + 0.20 * center_score
            + 0.15 * width_score
        )

    def _build_face_image_points(self, landmarks, width: int, height: int) -> np.ndarray | None:
        indices = [1, 152, 33, 263, 61, 291]
        if max(indices) >= len(landmarks):
            return None

        return np.array(
            [
                (landmarks[index].x * width, landmarks[index].y * height)
                for index in indices
            ],
            dtype=np.float64,
        )

    def _estimate_iris_focus(self, landmarks) -> float | None:
        left_score = self._eye_focus_score(landmarks, 33, 133, [468, 469, 470, 471, 472])
        right_score = self._eye_focus_score(landmarks, 263, 362, [473, 474, 475, 476, 477])

        valid_scores = [score for score in (left_score, right_score) if score is not None]
        if not valid_scores:
            return None

        return float(np.mean(valid_scores))

    def _eye_focus_score(
        self,
        landmarks,
        outer_corner_idx: int,
        inner_corner_idx: int,
        iris_indices: list[int],
    ) -> float | None:
        if max(iris_indices + [outer_corner_idx, inner_corner_idx]) >= len(landmarks):
            return None

        outer_x = landmarks[outer_corner_idx].x
        inner_x = landmarks[inner_corner_idx].x
        left_bound = min(outer_x, inner_x)
        right_bound = max(outer_x, inner_x)
        eye_width = right_bound - left_bound
        if eye_width <= 1e-6:
            return None

        iris_x = float(np.mean([landmarks[index].x for index in iris_indices]))
        iris_ratio = (iris_x - left_bound) / eye_width
        return 1.0 - self._clamp(abs(iris_ratio - 0.5) / 0.35, 0.0, 1.0)

    def _smooth_score(self, score: float, track_id: int | None) -> float:
        if track_id is None:
            return score

        history = self.track_scores.get(track_id)
        if history is None:
            history = deque(maxlen=self.config.smoothing_window)
            self.track_scores[track_id] = history

        history.append(score)
        return float(np.mean(history))

    def _score_to_label(self, score: float) -> str:
        if score >= self.config.high_threshold:
            return "high"
        if score >= self.config.medium_threshold:
            return "medium"
        return "low"

    @staticmethod
    def _sanitize_bbox(frame, bbox) -> tuple[int, int, int, int] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))
