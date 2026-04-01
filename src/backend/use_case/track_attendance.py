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
        self.marked_students = set()
        self.lesson_start_time = datetime.now()

    def execute(self, frame):
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

            self._resolve_identity(frame, bbox, tid, detected_faces)
            cached = self.identity_cache.get(tid, {"name": "Unknown", "sid": None})
            student_name = cached["name"]
            student_id = cached["sid"]

            engagement = self._estimate_engagement(frame, bbox, tid)

            print(
                f"[Track] track_id {tid} - name: {student_name} - engagement: {engagement}"
            )

            if student_id and student_id != "Unknown":
                presence_state = self.presence_tracker.get(student_id)
                if presence_state is None or (
                    now - presence_state["last_seen"]
                ).total_seconds() > self.config.stale_track_ttl_seconds:
                    self.presence_tracker[student_id] = {
                        "first_seen": now,
                        "last_seen": now,
                    }
                else:
                    self.presence_tracker[student_id]["last_seen"] = now

                if student_id not in self.marked_students:
                    first_seen = self.presence_tracker[student_id]["first_seen"]

                    if (
                        now - first_seen
                    ).total_seconds() >= self.config.presence_confirmation_seconds:
                        self._log_visit(student_id, engagement, now)
                        self.marked_students.add(student_id)

            final_results.append(
                {
                    "bbox": bbox,
                    "name": student_name,
                    "engagement": engagement,
                    "track_id": tid,
                }
            )

        self._cleanup_stale_tracks(now)
        return {"students": final_results}

    def _detect_faces(self, frame) -> list[dict]:
        if not self.face_recognizer or not hasattr(self.face_recognizer, "detect_faces"):
            return []

        try:
            return self.face_recognizer.detect_faces(frame)
        except Exception as e:
            print(f"[Track] face detector error: {e}")
            return []

    def _resolve_identity(self, frame, bbox, track_id: int, detected_faces: list[dict]) -> None:
        cached_identity = self.identity_cache.get(track_id)
        if cached_identity and cached_identity.get("sid"):
            return

        face_crop = self._select_face_crop(frame, bbox, detected_faces)
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

    def _estimate_engagement(self, frame, bbox, track_id: int) -> str:
        if not self.pose_estimator:
            return "unknown"

        try:
            return self.pose_estimator.estimate_engagement(
                frame,
                bbox,
                track_id=track_id,
            )
        except TypeError:
            return self.pose_estimator.estimate_engagement(frame, bbox)
        except Exception as e:
            print(f"[Track] pose estimator error: {e}")
            return "unknown"

    def _select_face_crop(self, frame, bbox, detected_faces: list[dict]):
        matched_face = self._match_face_to_person(bbox, detected_faces)
        if matched_face is not None:
            return matched_face["crop"]

        return self._extract_face_crop(frame, bbox)

    def _match_face_to_person(self, bbox, detected_faces: list[dict]) -> dict | None:
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

    @staticmethod
    def _intersection_over_face_area(first_bbox, second_bbox) -> float:
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

    def _cleanup_stale_tracks(self, now: datetime) -> None:
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

    def _log_visit(self, student_id, engagement, now: datetime):
        """Запись посещения с учетом времени и опоздания."""
        if student_id not in self.log_cooldowns or (
            now - self.log_cooldowns[student_id]
        ) > timedelta(seconds=self.config.log_cooldown_seconds):

            delay = (now - self.lesson_start_time).total_seconds()
            is_late = delay > self.config.late_after_seconds

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
            except Exception as e:
                print(f"[DB] Ошибка: {e}")
