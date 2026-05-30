from collections import deque
from pathlib import Path

import cv2
import numpy as np
import time

from src.backend.dependencies import settings
from src.backend.infrastructure.ai.config import EngagementConfig
from src.backend.infrastructure.ai.pose.pose_estimator import (
    PoseEstimator as YoloPoseEstimator,
)

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
    """Оценщик вовлеченности с несколькими backend'ами и устойчивым fallback."""

    def __init__(self, config: EngagementConfig | None = None):
        self.config = config or EngagementConfig()
        self.track_scores: dict[int, deque[float]] = {}
        # EWMA state per track for lightweight temporal smoothing
        self.track_ewma: dict[int, float] = {}
        # Per-track state machine and hysteresis counters
        self.track_states: dict[int, dict] = {}
        self.state_counters: dict[int, dict] = {}
        self.backend_name = "heuristic_face"

        self.mp = None
        self.face_mesh = None
        self.pose = None
        self.face_landmarker = None
        self.pose_landmarker = None
        self.yolo_pose_estimator = None

        self.face_task_path = Path(
            getattr(settings, "MP_FACE_LANDMARKER_MODEL_PATH", "")
        )
        self.pose_task_path = Path(
            getattr(settings, "MP_POSE_LANDMARKER_MODEL_PATH", "")
        )
        self.yolo_pose_model_path = Path(getattr(settings, "YOLO_POSE_MODEL_PATH", ""))

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # throttling / caching to reduce repeated heavy inference per-track
        self.eval_interval_seconds = float(
            getattr(settings, "ENGAGEMENT_EVAL_INTERVAL", 0.35)
        )
        self._last_eval_time: dict[int, float] = {}
        self._last_eval_score: dict[int, float] = {}

        self._initialize_backend()
        print(f"[AI] engagement backend: {self.backend_name}")

    def estimate_engagement(self, frame, bbox, track_id=None, face_bbox=None):
        """
        Estimates engagement.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            track_id: Input value for `track_id`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The computed or transformed result.
        """
        if bbox is None:
            return "unknown"

        sanitized_bbox = self._sanitize_bbox(frame, bbox)
        if sanitized_bbox is None:
            return "unknown"

        sanitized_face_bbox = (
            self._sanitize_bbox(frame, face_bbox) if face_bbox else None
        )

        # Throttle per-track heavy evaluation to reduce CPU usage on high frame rates
        now_ts = time.time()
        if track_id is not None:
            last = self._last_eval_time.get(track_id)
            if last is not None and (now_ts - last) < float(self.eval_interval_seconds):
                cached = self._last_eval_score.get(track_id)
                if cached is not None:
                    sm = self._smooth_score(cached, track_id)
                    return self._score_to_label(sm)

        if self.backend_name == "mediapipe_legacy":
            score = self._estimate_engagement_mediapipe_legacy(
                frame,
                sanitized_bbox,
                sanitized_face_bbox,
            )
        elif self.backend_name == "mediapipe_tasks":
            score = self._estimate_engagement_mediapipe_tasks(
                frame,
                sanitized_bbox,
                sanitized_face_bbox,
            )
        elif self.backend_name == "yolo_pose":
            score = self._estimate_engagement_yolo(
                frame,
                sanitized_bbox,
                sanitized_face_bbox,
            )
            if score is None:
                score = self._estimate_engagement_heuristic(
                    frame,
                    sanitized_bbox,
                    sanitized_face_bbox,
                )
        else:
            score = self._estimate_engagement_heuristic(
                frame,
                sanitized_bbox,
                sanitized_face_bbox,
            )

        if score is None:
            return "unknown"
        # store cached numeric score for throttling
        try:
            if track_id is not None:
                self._last_eval_time[track_id] = now_ts
                self._last_eval_score[track_id] = float(score)
        except Exception:
            pass

        score = self._smooth_score(score, track_id)
        return self._score_to_label(score)

    def forget_track(self, track_id: int) -> None:
        """
        Clears track.

        Args:
            track_id: Input value for `track_id`.

        Returns:
            Does not return a value.
        """
        self.track_scores.pop(track_id, None)

    def _initialize_backend(self) -> None:
        """
        Runs the internal step initialize backend.

        Args:
            None.

        Returns:
            Does not return a value.
        """
        try:
            import mediapipe as mp
        except ImportError:
            mp = None

        # Prefer MediaPipe (tasks or legacy) for realtime performance on CPU
        if mp is not None:
            self.mp = mp
            if self._try_init_mediapipe_tasks(mp):
                return
            if self._try_init_mediapipe_legacy(mp):
                return

        # Fallback to YOLO pose if MediaPipe not available or failed
        if self._try_init_yolo_pose():
            return

        self.backend_name = "heuristic_face"

    def _try_init_mediapipe_legacy(self, mp) -> bool:
        """
        Runs the internal step try init mediapipe legacy.

        Args:
            mp: Input value for `mp`.

        Returns:
            The function result.
        """
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is None:
            return False

        # Use fastest settings for realtime CPU execution
        self.face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        # model_complexity=0 is fastest and recommended for edge/CPU
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.backend_name = "mediapipe_legacy"
        return True

    def _try_init_mediapipe_tasks(self, mp) -> bool:
        """
        Runs the internal step try init mediapipe tasks.

        Args:
            mp: Input value for `mp`.

        Returns:
            The function result.
        """
        if not self.face_task_path.exists() or not self.pose_task_path.exists():
            return False

        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except Exception as error:
            print(f"[AI] mediapipe tasks import error: {error}")
            return False

        try:
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.face_task_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=self.config.min_detection_confidence,
                    min_face_presence_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                )
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.pose_task_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=self.config.min_detection_confidence,
                    min_pose_presence_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                )
            )
        except Exception as error:
            print(f"[AI] mediapipe tasks init error: {error}")
            self.face_landmarker = None
            self.pose_landmarker = None
            return False

        self.backend_name = "mediapipe_tasks"
        return True

    def _try_init_yolo_pose(self) -> bool:
        """
        Runs the internal step try init yolo pose.

        Args:
            None.

        Returns:
            The function result.
        """
        # YOLO pose disabled for edge deployment — rely on MediaPipe only
        print("[AI] YOLO pose disabled for edge deployment; using MediaPipe only")
        self.yolo_pose_estimator = None
        return False

    def _estimate_engagement_mediapipe_legacy(
        self,
        frame,
        bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """
        Runs the internal step estimate engagement mediapipe legacy.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None

        face_roi = self._extract_face_roi(frame, bbox, face_bbox)

        face_score = self._estimate_face_attention_legacy(face_roi)
        body_score = self._estimate_body_attention_legacy(person_roi)
        return self._combine_scores(face_score, body_score)

    def _estimate_engagement_mediapipe_tasks(
        self,
        frame,
        bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """
        Runs the internal step estimate engagement mediapipe tasks.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        if not self.mp or not self.face_landmarker or not self.pose_landmarker:
            return None

        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None

        face_roi = self._extract_face_roi(frame, bbox, face_bbox)
        if face_roi.size == 0:
            return None

        # Downscale face/person crops before running heavy detectors to speed up inference
        try:
            max_face = int(getattr(settings, "FACE_PROCESS_SIZE", 160))
            fw, fh = face_roi.shape[1], face_roi.shape[0]
            scale = min(1.0, float(max_face) / max(1.0, max(fw, fh)))
            if scale < 1.0:
                nw = max(1, int(fw * scale))
                nh = max(1, int(fh * scale))
                face_proc = cv2.resize(face_roi, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                face_proc = face_roi

            rgb_face = cv2.cvtColor(face_proc, cv2.COLOR_BGR2RGB)
            face_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB, data=rgb_face
            )
            face_result = self.face_landmarker.detect(face_image)
            face_score = self._estimate_face_attention_from_landmarks(
                face_result.face_landmarks[0] if face_result.face_landmarks else None,
                face_proc.shape[1],
                face_proc.shape[0],
            )
        except Exception:
            face_score = None

        try:
            max_person = int(getattr(settings, "PERSON_PROCESS_SIZE", 320))
            pw, ph = person_roi.shape[1], person_roi.shape[0]
            pscale = min(1.0, float(max_person) / max(1.0, max(pw, ph)))
            if pscale < 1.0:
                npw = max(1, int(pw * pscale))
                nph = max(1, int(ph * pscale))
                person_proc = cv2.resize(
                    person_roi, (npw, nph), interpolation=cv2.INTER_AREA
                )
            else:
                person_proc = person_roi

            rgb_person = cv2.cvtColor(person_proc, cv2.COLOR_BGR2RGB)
            person_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=rgb_person,
            )
            pose_result = self.pose_landmarker.detect(person_image)
            body_score = self._estimate_body_attention_from_landmarks(
                pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
            )
        except Exception:
            body_score = None

        return self._combine_scores(face_score, body_score)

    def _estimate_engagement_yolo(
        self,
        frame,
        bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """
        Runs the internal step estimate engagement yolo.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        if self.yolo_pose_estimator is None:
            return None

        pose_result = self.yolo_pose_estimator.estimate_pose(frame, list(bbox))
        if not pose_result:
            return None

        keypoints = pose_result.get("keypoints") or []
        if len(keypoints) < 7:
            return None

        def _kp(index: int):
            """
            Runs the internal step kp.

            Args:
                index: Input value for `index`.

            Returns:
                The function result.
            """
            point = keypoints[index]
            return (
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("conf", 0.0)),
            )

        nose_x, nose_y, nose_conf = _kp(0)
        left_shoulder_x, left_shoulder_y, left_shoulder_conf = _kp(5)
        right_shoulder_x, right_shoulder_y, right_shoulder_conf = _kp(6)

        visibility = float(
            np.mean([nose_conf, left_shoulder_conf, right_shoulder_conf])
        )
        if visibility < self.config.min_body_visibility:
            return None

        shoulder_width = abs(left_shoulder_x - right_shoulder_x)
        shoulder_balance = 1.0 - self._clamp(
            abs(left_shoulder_y - right_shoulder_y) / 30.0,
            0.0,
            1.0,
        )
        torso_center = (left_shoulder_x + right_shoulder_x) / 2.0
        center_score = 1.0 - self._clamp(
            abs(nose_x - torso_center) / max(30.0, shoulder_width),
            0.0,
            1.0,
        )
        width_score = self._clamp(shoulder_width / 120.0, 0.0, 1.0)

        return (
            0.45 * visibility
            + 0.20 * shoulder_balance
            + 0.20 * center_score
            + 0.15 * width_score
        )

    def _estimate_engagement_heuristic(
        self,
        frame,
        bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """
        Runs the internal step estimate engagement heuristic.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        if face_bbox is not None:
            return self._estimate_engagement_from_face_bbox(bbox, face_bbox)

        x1, y1, x2, y2 = self._build_head_region(bbox)
        head_roi = frame[y1:y2, x1:x2]
        if head_roi.size == 0:
            return None

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return 0.20

        face_x, face_y, face_w, face_h = max(
            faces,
            key=lambda face: int(face[2]) * int(face[3]),
        )
        head_width = max(1, x2 - x1)
        head_height = max(1, y2 - y1)
        face_area_ratio = (face_w * face_h) / float(head_width * head_height)

        face_center_x = face_x + face_w / 2.0
        face_center_y = face_y + face_h / 2.0
        head_center_x = head_width / 2.0
        head_center_y = head_height / 2.0

        horizontal_offset = abs(face_center_x - head_center_x) / max(
            1.0, head_width / 2.0
        )
        vertical_offset = abs(face_center_y - head_center_y) / max(
            1.0, head_height / 2.0
        )
        centered_score = 1.0 - self._clamp(
            0.7 * horizontal_offset + 0.3 * vertical_offset,
            0.0,
            1.0,
        )
        size_score = self._clamp(face_area_ratio / 0.16, 0.0, 1.0)
        frontal_score = 1.0 - self._clamp(
            abs(face_w - face_h) / max(face_w, face_h), 0.0, 1.0
        )

        return 0.45 * centered_score + 0.35 * size_score + 0.20 * frontal_score

    def _estimate_engagement_from_face_bbox(
        self,
        person_bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> float:
        """
        Runs the internal step estimate engagement from face bbox.

        Args:
            person_bbox: Input value for `person_bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        px1, py1, px2, py2 = person_bbox
        fx1, fy1, fx2, fy2 = face_bbox
        person_width = max(1, px2 - px1)
        person_height = max(1, py2 - py1)
        face_width = max(1, fx2 - fx1)
        face_height = max(1, fy2 - fy1)

        face_center_x = (fx1 + fx2) / 2.0
        face_center_y = (fy1 + fy2) / 2.0
        expected_center_x = (px1 + px2) / 2.0
        expected_center_y = py1 + person_height * 0.32

        horizontal_offset = abs(face_center_x - expected_center_x) / max(
            1.0,
            person_width * 0.35,
        )
        vertical_offset = abs(face_center_y - expected_center_y) / max(
            1.0,
            person_height * 0.25,
        )
        centered_score = 1.0 - self._clamp(
            0.65 * horizontal_offset + 0.35 * vertical_offset,
            0.0,
            1.0,
        )

        face_area_ratio = (face_width * face_height) / float(
            person_width * person_height
        )
        size_score = self._clamp(face_area_ratio / 0.18, 0.0, 1.0)
        frontal_score = 1.0 - self._clamp(
            abs(face_width - face_height) / max(face_width, face_height),
            0.0,
            1.0,
        )
        upper_body_score = 1.0 - self._clamp(
            (face_center_y - py1) / max(1.0, person_height) / 0.6,
            0.0,
            1.0,
        )

        return (
            0.40 * centered_score
            + 0.25 * size_score
            + 0.15 * frontal_score
            + 0.20 * upper_body_score
        )

    def _estimate_face_attention_legacy(self, face_roi) -> float | None:
        """
        Runs the internal step estimate face attention legacy.

        Args:
            face_roi: Input value for `face_roi`.

        Returns:
            The function result.
        """
        if face_roi is None or face_roi.size == 0 or self.face_mesh is None:
            return None

        try:
            max_face = int(getattr(settings, "FACE_PROCESS_SIZE", 160))
            fw, fh = face_roi.shape[1], face_roi.shape[0]
            scale = min(1.0, float(max_face) / max(1.0, max(fw, fh)))
            if scale < 1.0:
                nw = max(1, int(fw * scale))
                nh = max(1, int(fh * scale))
                face_proc = cv2.resize(face_roi, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                face_proc = face_roi

            rgb_face = cv2.cvtColor(face_proc, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_face)
            if not result.multi_face_landmarks:
                return None

            return self._estimate_face_attention_from_landmarks(
                result.multi_face_landmarks[0].landmark,
                face_proc.shape[1],
                face_proc.shape[0],
            )
        except Exception:
            return None

    def _estimate_body_attention_legacy(self, person_roi) -> float | None:
        """
        Runs the internal step estimate body attention legacy.

        Args:
            person_roi: Input value for `person_roi`.

        Returns:
            The function result.
        """
        if self.pose is None:
            return None

        try:
            max_person = int(getattr(settings, "PERSON_PROCESS_SIZE", 320))
            pw, ph = person_roi.shape[1], person_roi.shape[0]
            pscale = min(1.0, float(max_person) / max(1.0, max(pw, ph)))
            if pscale < 1.0:
                npw = max(1, int(pw * pscale))
                nph = max(1, int(ph * pscale))
                person_proc = cv2.resize(
                    person_roi, (npw, nph), interpolation=cv2.INTER_AREA
                )
            else:
                person_proc = person_roi

            rgb_person = cv2.cvtColor(person_proc, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_person)
            if result.pose_landmarks is None:
                return None

            return self._estimate_body_attention_from_landmarks(
                result.pose_landmarks.landmark
            )
        except Exception:
            return None

    def _estimate_face_attention_from_landmarks(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> float | None:
        """
        Runs the internal step estimate face attention from landmarks.

        Args:
            landmarks: Input value for `landmarks`.
            width: Input value for `width`.
            height: Input value for `height`.

        Returns:
            The function result.
        """
        if landmarks is None:
            return None

        image_points = self._build_face_image_points(landmarks, width, height)
        if image_points is None:
            return None

        focal_length = float(width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, width / 2],
                [0.0, focal_length, height / 2],
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

        yaw_score = 1.0 - self._clamp(abs(yaw) / self.config.max_yaw_degrees, 0.0, 1.0)
        pitch_score = 1.0 - self._clamp(
            abs(pitch) / self.config.max_pitch_degrees, 0.0, 1.0
        )
        iris_score = self._estimate_iris_focus(landmarks)

        if iris_score is None:
            return 0.6 * yaw_score + 0.4 * pitch_score

        return 0.45 * yaw_score + 0.25 * pitch_score + 0.30 * iris_score

    def _estimate_body_attention_from_landmarks(self, landmarks) -> float | None:
        """
        Runs the internal step estimate body attention from landmarks.

        Args:
            landmarks: Input value for `landmarks`.

        Returns:
            The function result.
        """
        if landmarks is None or len(landmarks) <= 12:
            return None

        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        nose_visibility = float(getattr(nose, "visibility", 1.0))
        left_visibility = float(getattr(left_shoulder, "visibility", 1.0))
        right_visibility = float(getattr(right_shoulder, "visibility", 1.0))
        visibility = float(
            np.mean([nose_visibility, left_visibility, right_visibility])
        )
        if visibility < self.config.min_body_visibility:
            return None

        shoulder_width = abs(float(left_shoulder.x) - float(right_shoulder.x))
        shoulder_balance = 1.0 - self._clamp(
            abs(float(left_shoulder.y) - float(right_shoulder.y)) / 0.15,
            0.0,
            1.0,
        )
        torso_center = (float(left_shoulder.x) + float(right_shoulder.x)) / 2.0
        center_score = 1.0 - self._clamp(
            abs(float(nose.x) - torso_center) / 0.2, 0.0, 1.0
        )
        width_score = self._clamp(shoulder_width / 0.35, 0.0, 1.0)

        base_score = (
            0.45 * visibility
            + 0.20 * shoulder_balance
            + 0.20 * center_score
            + 0.15 * width_score
        )

        # Hand gestures: detect raised hand (positive engagement) and hands near face (negative)
        try:
            hand_indices = []
            if len(landmarks) > 15:
                # mediapipe pose: left_wrist=15, right_wrist=16
                left_wrist = landmarks[15]
                right_wrist = landmarks[16] if len(landmarks) > 16 else None
                hand_indices = [left_wrist, right_wrist]
            else:
                hand_indices = []

            hand_raise_count = 0
            near_face = False
            nose_x = float(getattr(nose, "x", 0.0))
            nose_y = float(getattr(nose, "y", 0.0))
            shoulder_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0

            dists = []
            for hw in hand_indices:
                if hw is None:
                    continue
                try:
                    wx = float(getattr(hw, "x", 0.0))
                    wy = float(getattr(hw, "y", 0.0))
                except Exception:
                    continue
                if wy < shoulder_y - 0.02:
                    hand_raise_count += 1
                d = np.sqrt((wx - nose_x) ** 2 + (wy - nose_y) ** 2)
                dists.append(d)

            if dists:
                min_d = float(min(dists))
                if min_d < 0.12:
                    near_face = True

            hand_raise_score = float(hand_raise_count) / 2.0 if hand_indices else 0.0
            hand_effect = 0.35 * hand_raise_score - (0.45 if near_face else 0.0)
            # moderate influence of hands on overall score
            final = base_score + hand_effect * 0.25
            final = float(self._clamp(final, 0.0, 1.0))
            return final
        except Exception:
            return float(self._clamp(base_score, 0.0, 1.0))

    def _combine_scores(
        self,
        face_score: float | None,
        body_score: float | None,
    ) -> float | None:
        """
        Runs the internal step combine scores.

        Args:
            face_score: Input value for `face_score`.
            body_score: Input value for `body_score`.

        Returns:
            The function result.
        """
        # Use configurable weights from settings, fall back to sensible defaults
        face_w = float(getattr(settings, "ENGAGEMENT_FACE_WEIGHT", 0.7))
        body_w = float(getattr(settings, "ENGAGEMENT_BODY_WEIGHT", 0.3))

        weighted_scores = []
        if face_score is not None:
            weighted_scores.append((float(self._clamp(face_score, 0.0, 1.0)), face_w))
        if body_score is not None:
            weighted_scores.append((float(self._clamp(body_score, 0.0, 1.0)), body_w))

        if not weighted_scores:
            return None

        total_w = sum(w for _, w in weighted_scores)
        if total_w <= 0.0:
            # If weights are misconfigured, fallback to simple mean
            return float(sum(v for v, _ in weighted_scores) / len(weighted_scores))

        return float(sum(value * weight for value, weight in weighted_scores) / total_w)

    def _extract_face_roi(
        self,
        frame,
        person_bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None,
    ):
        """
        Runs the internal step extract face roi.

        Args:
            frame: Input value for `frame`.
            person_bbox: Input value for `person_bbox`.
            face_bbox: Input value for `face_bbox`.

        Returns:
            The function result.
        """
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            crop = frame[y1:y2, x1:x2]
            if getattr(crop, "size", 0) > 0:
                return crop

        x1, y1, x2, y2 = person_bbox
        head_bottom = y1 + int((y2 - y1) * self.config.head_crop_ratio)
        return frame[y1:head_bottom, x1:x2]

    def _build_face_image_points(
        self, landmarks, width: int, height: int
    ) -> np.ndarray | None:
        """
        Runs the internal step build face image points.

        Args:
            landmarks: Input value for `landmarks`.
            width: Input value for `width`.
            height: Input value for `height`.

        Returns:
            The function result.
        """
        indices = [1, 152, 33, 263, 61, 291]
        if max(indices) >= len(landmarks):
            return None

        return np.array(
            [
                (float(landmarks[index].x) * width, float(landmarks[index].y) * height)
                for index in indices
            ],
            dtype=np.float64,
        )

    def _estimate_iris_focus(self, landmarks) -> float | None:
        """
        Runs the internal step estimate iris focus.

        Args:
            landmarks: Input value for `landmarks`.

        Returns:
            The function result.
        """
        left_score = self._eye_focus_score(
            landmarks, 33, 133, [468, 469, 470, 471, 472]
        )
        right_score = self._eye_focus_score(
            landmarks, 263, 362, [473, 474, 475, 476, 477]
        )
        valid_scores = [
            score for score in (left_score, right_score) if score is not None
        ]
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
        """
        Runs the internal step eye focus score.

        Args:
            landmarks: Input value for `landmarks`.
            outer_corner_idx: Input value for `outer_corner_idx`.
            inner_corner_idx: Input value for `inner_corner_idx`.
            iris_indices: Input value for `iris_indices`.

        Returns:
            The function result.
        """
        if max(iris_indices + [outer_corner_idx, inner_corner_idx]) >= len(landmarks):
            return None

        outer_x = float(landmarks[outer_corner_idx].x)
        inner_x = float(landmarks[inner_corner_idx].x)
        left_bound = min(outer_x, inner_x)
        right_bound = max(outer_x, inner_x)
        eye_width = right_bound - left_bound
        if eye_width <= 1e-6:
            return None

        iris_x = float(np.mean([float(landmarks[index].x) for index in iris_indices]))
        iris_ratio = (iris_x - left_bound) / eye_width
        return 1.0 - self._clamp(abs(iris_ratio - 0.5) / 0.35, 0.0, 1.0)

    def _build_head_region(
        self, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """
        Runs the internal step build head region.

        Args:
            bbox: Input value for `bbox`.

        Returns:
            The function result.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * (1.0 - self.config.head_crop_ratio) / 2.0)
        head_height = int(height * max(self.config.head_crop_ratio, 0.55))
        return x1 + padding_x, y1, x2 - padding_x, y1 + head_height

    def _smooth_score(self, score: float, track_id: int | None) -> float:
        """
        Runs the internal step smooth score.

        Args:
            score: Input value for `score`.
            track_id: Input value for `track_id`.

        Returns:
            The function result.
        """
        if track_id is None:
            return score

        # EWMA smoothing: alpha derived from smoothing window
        alpha = 2.0 / max(3.0, float(self.config.smoothing_window) + 1.0)
        prev = self.track_ewma.get(track_id)
        if prev is None:
            self.track_ewma[track_id] = float(score)
            smoothed = float(score)
        else:
            smoothed = alpha * float(score) + (1.0 - alpha) * float(prev)
            self.track_ewma[track_id] = smoothed

        # update lightweight state machine (hysteresis counters)
        try:
            self._update_state(track_id, smoothed)
        except Exception:
            pass

        return float(smoothed)

    def _update_state(self, track_id: int, smoothed_score: float) -> None:
        """Lightweight state machine with simple hysteresis.

        States: Focused, Participating, Distracted, Absent
        """
        from datetime import datetime

        # determine candidate state from thresholds
        if smoothed_score >= self.config.high_threshold:
            candidate = "Focused"
        elif smoothed_score >= self.config.medium_threshold:
            candidate = "Participating"
        elif smoothed_score >= 0.10:
            candidate = "Distracted"
        else:
            candidate = "Absent"

        state_entry = self.track_states.get(track_id)
        if state_entry is None:
            self.track_states[track_id] = {
                "state": candidate,
                "last_change": datetime.now(),
            }
            self.state_counters[track_id] = {"candidate": None, "count": 0}
            return

        current = state_entry.get("state")
        counter = self.state_counters.get(track_id, {"candidate": None, "count": 0})

        if candidate == current:
            # stable, reset counters
            counter["candidate"] = None
            counter["count"] = 0
            self.state_counters[track_id] = counter
            return

        if counter.get("candidate") == candidate:
            counter["count"] = counter.get("count", 0) + 1
        else:
            counter["candidate"] = candidate
            counter["count"] = 1

        # require two confirmations to change state (simple hysteresis)
        if counter.get("count", 0) >= 2:
            self.track_states[track_id] = {
                "state": candidate,
                "last_change": datetime.now(),
            }
            counter["candidate"] = None
            counter["count"] = 0

        self.state_counters[track_id] = counter

    def _score_to_label(self, score: float) -> str:
        """
        Runs the internal step score to label.

        Args:
            score: Input value for `score`.

        Returns:
            The function result.
        """
        if score >= self.config.high_threshold:
            return "high"
        if score >= self.config.medium_threshold:
            return "medium"
        return "low"

    @staticmethod
    def _sanitize_bbox(frame, bbox) -> tuple[int, int, int, int] | None:
        """
        Runs the internal step sanitize bbox.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.

        Returns:
            The function result.
        """
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
        """
        Runs the internal step clamp.

        Args:
            value: Input value for `value`.
            min_value: Input value for `min_value`.
            max_value: Input value for `max_value`.

        Returns:
            The function result.
        """
        return max(min_value, min(value, max_value))
