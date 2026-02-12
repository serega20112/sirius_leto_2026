from datetime import datetime, timedelta
from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus


class TrackAttendanceUseCase:
    """Сценарий обработки кадров с кэшированием распознавания."""

    def __init__(
        self,
        person_detector,
        face_recognizer,
        pose_estimator,
        student_repo,
        attendance_repo,
    ):
        self.person_detector = person_detector
        self.face_recognizer = face_recognizer
        self.pose_estimator = pose_estimator
        self.student_repo = student_repo
        self.attendance_repo = attendance_repo

        self.identity_cache = {}
        self.log_cooldowns = {}
        self.frame_count = 0

    def execute(self, frame):
        self.frame_count += 1
        tracked_people = []
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

            try:
                x1, y1, x2, y2 = map(int, raw_bbox)
            except Exception:
                continue

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            bbox = [x1, y1, x2, y2]

            if tid not in self.identity_cache or self.frame_count % 30 == 0:
                face_crop = None
                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]

                if face_crop is not None and getattr(face_crop, 'size', 0) > 0:
                    try:
                        filename = self.face_recognizer.recognize(face_crop)
                    except Exception as e:
                        print(f"[Track] face recognition error: {e}")
                        filename = None
                    print(f"[Track] recognition filename: {filename}")
                    if filename:
                        sid = filename.split(".")[0]
                        student = self.student_repo.find_by_id(sid)
                        name = student.name if student else "Unknown"
                        self.identity_cache[tid] = {"name": name, "sid": sid}
                    else:
                        self.identity_cache[tid] = {"name": "Unknown", "sid": None}

            cached = self.identity_cache.get(tid, {"name": "Unknown", "sid": None})
            student_name = cached["name"]
            student_id = cached["sid"]

            engagement = "unknown"
            try:
                if self.pose_estimator:
                    engagement = self.pose_estimator.estimate_engagement(frame, bbox)
            except Exception as e:
                print(f"[Track] pose estimator error: {e}")
            print(f"[Track] track_id {tid} - name: {student_name} - engagement: {engagement}")

            if student_id and student_id != "Unknown":
                self._log_visit(student_id, engagement)

            final_results.append(
                {
                    "bbox": bbox,
                    "name": student_name,
                    "engagement": engagement,
                    "track_id": tid,
                }
            )

        return {"students": final_results}

    def _log_visit(self, student_id, engagement):
        """Запись в базу данных с проверкой кулдауна."""
        now = datetime.now()
        if student_id not in self.log_cooldowns or (
            now - self.log_cooldowns[student_id]
        ) > timedelta(minutes=1):

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
                is_late=False,
                engagement_score=status_enum,
            )

            try:
                self.attendance_repo.add_log(log)
                self.log_cooldowns[student_id] = now
                print(f"[DB] Запись для {student_id} добавлена.")
            except Exception as e:
                print(f"[DB] Ошибка: {e}")
