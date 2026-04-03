from datetime import date as date_type
from datetime import datetime, timedelta

from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.infrastructure.ai.config import AttendanceTrackingConfig
from src.backend.infrastructure.ai.contracts import (
    EngagementEstimator,
    FaceRecognizerContract,
    PersonTracker,
)


class TrackAttendanceUseCase:
    """Сценарий обработки кадров с устойчивым распознаванием и корректной фиксацией присутствия."""

    def __init__(
        self,
        person_detector: PersonTracker | None,
        face_recognizer: FaceRecognizerContract | None,
        pose_estimator: EngagementEstimator | None,
        student_repo,
        attendance_repo,
        config: AttendanceTrackingConfig | None = None,
    ):
        self.person_detector = person_detector
        self.face_recognizer = face_recognizer
        self.pose_estimator = pose_estimator
        self.student_repo = student_repo
        self.attendance_repo = attendance_repo
        self.config = config or AttendanceTrackingConfig()

        self.identity_cache: dict[int, dict[str, str | None]] = {}
        self.log_cooldowns: dict[str, datetime] = {}
        self.track_last_seen: dict[int, datetime] = {}
        self.frame_count = 0

        self.presence_tracker: dict[str, dict[str, datetime]] = {}
        self.marked_students: dict[str, date_type] = {}

    def execute(self, frame):
        """
        Executes the main scenario for TrackAttendanceUseCase.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The scenario execution result.
        """
        self.frame_count += 1
        now = datetime.now()
        tracked_people = []
        detected_faces = self._detect_faces(frame)

        try:
            if self.person_detector:
                tracked_people = self.person_detector.track_people(frame)
        except Exception as e:
            print(f"[Track] person_detector error: {e}")

        print(f"[Track] frame {self.frame_count} - tracked_people: {len(tracked_people)}")

        final_results = []
        h, w = frame.shape[:2]

        for person in tracked_people:
            raw_bbox = person.get("bbox")
            tid = person.get("track_id")
            if tid is None:
                continue

            try:
                x1, y1, x2, y2 = map(int, raw_bbox)
            except Exception:
                continue

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            bbox = [x1, y1, x2, y2]
            self.track_last_seen[tid] = now
            matched_face = self._match_face_to_person(bbox, detected_faces)

            self._resolve_identity(frame, bbox, tid, matched_face)
            cached = self.identity_cache.get(tid, {"name": "Unknown", "sid": None})
            student_name = cached["name"]
            student_id = cached["sid"]

            engagement = self._estimate_engagement(frame, bbox, tid, matched_face)

            print(
                f"[Track] track_id {tid} - name: {student_name} - engagement: {engagement}"
            )

            if student_id and student_id != "Unknown":
                presence_state = self.presence_tracker.get(student_id)
                if presence_state is None or (
                    presence_state["first_seen"].date() != now.date()
                ) or (
                    now - presence_state["last_seen"]
                ).total_seconds() > self.config.stale_track_ttl_seconds:
                    self.presence_tracker[student_id] = {
                        "first_seen": now,
                        "last_seen": now,
                    }
                else:
                    self.presence_tracker[student_id]["last_seen"] = now

                if self.marked_students.get(student_id) != now.date():
                    first_seen = self.presence_tracker[student_id]["first_seen"]

                    if (
                        now - first_seen
                    ).total_seconds() >= self.config.presence_confirmation_seconds:
                        is_accounted_for_today = self._log_visit(student_id, engagement, now)
                        if is_accounted_for_today:
                            self.marked_students[student_id] = now.date()

            final_results.append(
                {
                    "bbox": bbox,
                    "display_bbox": self._build_display_bbox(frame, bbox, matched_face),
                    "name": student_name,
                    "engagement": engagement,
                    "track_id": tid,
                }
            )

        self._cleanup_stale_tracks(now)
        return {"students": final_results}

    def _detect_faces(self, frame) -> list[dict]:
        """
        Runs the internal step detect faces.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The function result.
        """
        if not self.face_recognizer or not hasattr(self.face_recognizer, "detect_faces"):
            return []

        try:
            detected_faces = self.face_recognizer.detect_faces(frame)
        except Exception as e:
            print(f"[Track] face detector error: {e}")
            return []

        return detected_faces if isinstance(detected_faces, list) else []

    def _resolve_identity(self, frame, bbox, track_id: int, matched_face: dict | None) -> None:
        """
        Runs the internal step resolve identity.
        
        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            track_id: Input value for `track_id`.
            matched_face: Input value for `matched_face`.
        
        Returns:
            Does not return a value.
        """
        cached_identity = self.identity_cache.get(track_id)
        if cached_identity and cached_identity.get("sid"):
            return

        face_crop = self._select_face_crop(frame, bbox, matched_face)
        if face_crop is None or getattr(face_crop, "size", 0) == 0:
            self.identity_cache[track_id] = {"name": "Unknown", "sid": None}
            return

        student_id = None
        if self.face_recognizer:
            try:
                student_id = self.face_recognizer.recognize(face_crop, track_id=track_id)
            except Exception as e:
                print(f"[Track] face recognition error: {e}")

        print(f"[Track] recognition student_id: {student_id}")

        if not student_id:
            self.identity_cache[track_id] = {"name": "Unknown", "sid": None}
            return

        student = self.student_repo.find_by_id(student_id)
        if student is None:
            print(f"[Track] student {student_id} not found in repository")
            self.identity_cache[track_id] = {"name": "Unknown", "sid": None}
            return

        self.identity_cache[track_id] = {"name": student.name, "sid": student.id}

    def _estimate_engagement(self, frame, bbox, track_id: int, matched_face: dict | None) -> str:
        """
        Runs the internal step estimate engagement.
        
        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            track_id: Input value for `track_id`.
            matched_face: Input value for `matched_face`.
        
        Returns:
            The function result.
        """
        if not self.pose_estimator:
            return "unknown"

        try:
            return self.pose_estimator.estimate_engagement(
                frame,
                bbox,
                track_id=track_id,
                face_bbox=matched_face["bbox"] if matched_face else None,
            )
        except TypeError:
            return self.pose_estimator.estimate_engagement(frame, bbox)
        except Exception as e:
            print(f"[Track] pose estimator error: {e}")
            return "unknown"

    def _select_face_crop(self, frame, bbox, matched_face: dict | None):
        """
        Runs the internal step select face crop.
        
        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
            matched_face: Input value for `matched_face`.
        
        Returns:
            The function result.
        """
        if matched_face is not None:
            return matched_face["crop"]

        return self._extract_face_crop(frame, bbox)

    def _match_face_to_person(self, bbox, detected_faces: list[dict]) -> dict | None:
        """
        Runs the internal step match face to person.
        
        Args:
            bbox: Input value for `bbox`.
            detected_faces: Input value for `detected_faces`.
        
        Returns:
            The function result.
        """
        if not detected_faces:
            return None

        head_region = self._build_head_region(bbox)
        best_face = None
        best_score = 0.0

        for face in detected_faces:
            score = self._intersection_over_face_area(head_region, face["bbox"])
            if score > best_score:
                best_score = score
                best_face = face

        if best_score < 0.35:
            return None

        return best_face

    def _build_head_region(self, bbox) -> list[int]:
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
        padding_x = int(width * (1.0 - self.config.face_crop_width_ratio) / 2)
        head_height = int(height * max(self.config.face_crop_height_ratio, 0.6))

        return [
            x1 + padding_x,
            y1,
            x2 - padding_x,
            y1 + head_height,
        ]

    def _extract_face_crop(self, frame, bbox):
        """
        Runs the internal step extract face crop.
        
        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.
        
        Returns:
            The function result.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        face_width = int(width * self.config.face_crop_width_ratio)
        face_height = int(height * self.config.face_crop_height_ratio)
        center_x = x1 + width // 2

        crop_x1 = max(0, center_x - face_width // 2)
        crop_x2 = min(frame.shape[1], center_x + face_width // 2)
        crop_y1 = max(0, y1)
        crop_y2 = min(frame.shape[0], y1 + face_height)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    def _build_display_bbox(self, frame, bbox, matched_face: dict | None) -> list[int]:
        """
        Builds a larger bbox for drawing so the frame surrounds the student
        more naturally on frontal and close camera shots.

        Args:
            frame: Current frame.
            bbox: Tracker bbox.
            matched_face: Matched face metadata when available.

        Returns:
            A sanitized bbox that should be used for drawing overlays.
        """
        expanded_bbox = self._expand_bbox(
            frame,
            bbox,
            left_ratio=0.18,
            top_ratio=0.12,
            right_ratio=0.18,
            bottom_ratio=0.18,
        )
        if matched_face is None:
            return expanded_bbox

        face_bbox = self._sanitize_bbox(frame, matched_face.get("bbox"))
        if face_bbox is None:
            return expanded_bbox

        portrait_bbox = self._build_portrait_bbox_from_face(frame, face_bbox)
        return self._merge_bboxes(expanded_bbox, portrait_bbox)

    def _build_portrait_bbox_from_face(self, frame, face_bbox) -> list[int]:
        """
        Builds a portrait-style bbox around a detected face to better frame
        a student when the camera is close and the full body is not visible.

        Args:
            frame: Current frame.
            face_bbox: Sanitized face bbox.

        Returns:
            Expanded portrait-style bbox.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_width = max(1, fx2 - fx1)
        face_height = max(1, fy2 - fy1)

        portrait_width = int(face_width * 2.6)
        portrait_height = int(face_height * 4.8)
        center_x = (fx1 + fx2) // 2
        top = fy1 - int(face_height * 0.35)

        return self._sanitize_bbox(
            frame,
            [
                center_x - portrait_width // 2,
                top,
                center_x + portrait_width // 2,
                top + portrait_height,
            ],
        ) or list(face_bbox)

    def _expand_bbox(
        self,
        frame,
        bbox,
        *,
        left_ratio: float,
        top_ratio: float,
        right_ratio: float,
        bottom_ratio: float,
    ) -> list[int]:
        """
        Expands bbox edges by configurable ratios and clamps them to the frame.

        Args:
            frame: Current frame.
            bbox: Source bbox.
            left_ratio: Left expansion ratio.
            top_ratio: Top expansion ratio.
            right_ratio: Right expansion ratio.
            bottom_ratio: Bottom expansion ratio.

        Returns:
            Expanded bbox.
        """
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return self._sanitize_bbox(
            frame,
            [
                x1 - int(width * left_ratio),
                y1 - int(height * top_ratio),
                x2 + int(width * right_ratio),
                y2 + int(height * bottom_ratio),
            ],
        ) or list(bbox)

    @staticmethod
    def _intersection_over_face_area(first_bbox, second_bbox) -> float:
        """
        Runs the internal step intersection over face area.
        
        Args:
            first_bbox: Input value for `first_bbox`.
            second_bbox: Input value for `second_bbox`.
        
        Returns:
            The function result.
        """
        ax1, ay1, ax2, ay2 = first_bbox
        bx1, by1, bx2, by2 = second_bbox

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        face_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return intersection / face_area

    @staticmethod
    def _merge_bboxes(first_bbox, second_bbox) -> list[int]:
        """
        Merges two bbox rectangles into one bbox containing both.

        Args:
            first_bbox: First bbox.
            second_bbox: Second bbox.

        Returns:
            Bounding box containing both inputs.
        """
        return [
            min(first_bbox[0], second_bbox[0]),
            min(first_bbox[1], second_bbox[1]),
            max(first_bbox[2], second_bbox[2]),
            max(first_bbox[3], second_bbox[3]),
        ]

    @staticmethod
    def _sanitize_bbox(frame, bbox) -> list[int] | None:
        """
        Clamps bbox coordinates to the current frame.

        Args:
            frame: Current frame.
            bbox: Source bbox.

        Returns:
            Sanitized bbox or `None`.
        """
        if bbox is None:
            return None

        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return None

        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _cleanup_stale_tracks(self, now: datetime) -> None:
        """
        Runs the internal step cleanup stale tracks.
        
        Args:
            now: Input value for `now`.
        
        Returns:
            Does not return a value.
        """
        stale_track_ids = [
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if (now - last_seen).total_seconds() > self.config.stale_track_ttl_seconds
        ]

        for track_id in stale_track_ids:
            self.track_last_seen.pop(track_id, None)
            self.identity_cache.pop(track_id, None)

            if self.face_recognizer and hasattr(self.face_recognizer, "forget_track"):
                self.face_recognizer.forget_track(track_id)

            if self.pose_estimator and hasattr(self.pose_estimator, "forget_track"):
                self.pose_estimator.forget_track(track_id)

    def _log_visit(self, student_id, engagement, now: datetime) -> bool:
        """
        Runs the internal step log visit.
        
        Args:
            student_id: Input value for `student_id`.
            engagement: Input value for `engagement`.
            now: Input value for `now`.
        
        Returns:
            The function result.
        """
        cooldown_deadline = timedelta(seconds=self.config.log_cooldown_seconds)
        last_logged_at = self.log_cooldowns.get(student_id)
        if last_logged_at and last_logged_at.date() == now.date() and (
            now - last_logged_at
        ) <= cooldown_deadline:
            return True

        if self._has_log_for_date(student_id, now.date()):
            self.log_cooldowns[student_id] = now
            return True

        lesson_start_at = self._resolve_lesson_start(now)
        delay = max(0.0, (now - lesson_start_at).total_seconds())
        is_late = now > lesson_start_at + timedelta(
            seconds=self.config.late_after_seconds
        )

        status_map = {
            "high": EngagementStatus.HIGH,
            "medium": EngagementStatus.MEDIUM,
            "low": EngagementStatus.LOW,
        }
        status_enum = status_map.get(engagement, EngagementStatus.UNKNOWN)

        log = AttendanceLog(
            id=None,
            student_id=student_id,
            timestamp=now,
            is_late=is_late,
            engagement_score=status_enum,
        )

        try:
            self.attendance_repo.add_log(log)
            self.log_cooldowns[student_id] = now
            print(
                f"[DB] {student_id} отмечен | опоздание: {is_late} | задержка: {delay:.1f}s"
            )
            return True
        except Exception as e:
            print(f"[DB] Ошибка: {e}")
            return False

    def _has_log_for_date(self, student_id: str, target_date: date_type) -> bool:
        """
        Runs the internal step has log for date.
        
        Args:
            student_id: Input value for `student_id`.
            target_date: Input value for `target_date`.
        
        Returns:
            The function result.
        """
        try:
            logs = self.attendance_repo.get_logs_by_student(student_id)
        except Exception as e:
            print(f"[DB] Ошибка чтения логов {student_id}: {e}")
            return False

        return any(log.timestamp.date() == target_date for log in logs)

    def _resolve_lesson_start(self, now: datetime) -> datetime:
        """
        Runs the internal step resolve lesson start.
        
        Args:
            now: Input value for `now`.
        
        Returns:
            The function result.
        """
        return datetime.combine(now.date(), self.config.lesson_start_time)
