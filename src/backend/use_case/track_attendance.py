from datetime import datetime, timedelta
from typing import Dict, Any

from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus


class TrackAttendanceUseCase:
    """Сценарий обработки видеокадра: детекция, распознавание, анализ вовлеченности."""

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

        # Кэш: {track_id: "Имя Студента"}
        self.identity_cache = {}
        # Кэш: {track_id: datetime_last_db_log}
        self.log_cooldowns = {}

    def execute(self, frame):
        # 1. Детекция и трекинг (GPU) - очень быстро
        tracked_people = self.person_detector.track_people(frame)

        final_results = []

        for person in tracked_people:
            bbox = person["bbox"]
            tid = person["track_id"]

            # 2. РАСПОЗНАВАНИЕ (Тяжелая часть)
            # Проверяем, знаем ли мы уже этого человека по его ID трека
            if tid not in self.identity_cache:
                # Вызываем DeepFace только ОДИН РАЗ для нового ID
                x1, y1, x2, y2 = bbox
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    filename = self.face_recognizer.recognize(face_crop)
                    if filename:
                        sid = filename.split(".")[0]
                        student = self.student_repo.find_by_id(sid)
                        name = student.name if student else "Unknown"
                        self.identity_cache[tid] = name
                        # Записываем в БД факт прихода
                        self._log_to_db(sid)
                    else:
                        self.identity_cache[tid] = "Unknown"

            student_name = self.identity_cache.get(tid, "Unknown")

            # 3. ВОВЛЕЧЕННОСТЬ (GPU)
            # Считаем на каждом кадре, так как твоя 5060 Ti это потянет
            engagement = self.pose_estimator.estimate_engagement(frame, bbox)

            final_results.append(
                {
                    "bbox": bbox,
                    "name": student_name,
                    "engagement": engagement,
                    "track_id": tid,
                }
            )

        return {"students": final_results}

    def _log_visit(self, student_id: str, engagement: str):
        """Записывает лог, если прошло достаточно времени с последней записи."""
        now = datetime.now()
        last_time = self._log_cache.get(student_id)

        if last_time is None or (now - last_time) > self._LOG_COOLDOWN:
            lesson_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
            is_late = now > lesson_start

            status_enum = (
                EngagementStatus(engagement)
                if engagement in ["high", "medium", "low"]
                else EngagementStatus.UNKNOWN
            )

            log = AttendanceLog(
                id=None,
                student_id=student_id,
                timestamp=now,
                is_late=is_late,
                engagement_score=status_enum,
            )

            self.attendance_repo.add_log(log)
            self._log_cache[student_id] = now
