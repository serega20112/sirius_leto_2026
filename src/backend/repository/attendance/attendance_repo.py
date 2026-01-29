from typing import List

from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.infrastructure.database import AttendanceModel
from src.backend.repository import BaseRepository


class SqliteAttendanceRepository(BaseRepository):
    """Репозиторий для работы с журналом посещаемости через SQLite."""

    def add_log(self, log: AttendanceLog) -> AttendanceLog:
        """Добавляет новую запись в журнал и возвращает её с присвоенным ID."""
        model = AttendanceModel(
            student_id=log.student_id,
            timestamp=log.timestamp,
            is_late=log.is_late,
            engagement_score=log.engagement_score.value
        )
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)

        return self._to_entity(model)

    def get_logs_by_student(self, student_id: str) -> List[AttendanceLog]:
        """Возвращает историю посещений для конкретного студента."""
        models = self.session.query(AttendanceModel).filter(
            AttendanceModel.student_id == student_id
        ).all()
        return [self._to_entity(m) for m in models]

    def get_all_logs(self) -> List[AttendanceLog]:
        """Возвращает все записи журнала."""
        models = self.session.query(AttendanceModel).order_by(AttendanceModel.timestamp.desc()).all()
        return [self._to_entity(m) for m in models]

    def _to_entity(self, model: AttendanceModel) -> AttendanceLog:
        """Преобразует ORM модель в доменную сущность."""
        return AttendanceLog(
            id=model.id,
            student_id=model.student_id,
            timestamp=model.timestamp,
            is_late=model.is_late,
            engagement_score=EngagementStatus(model.engagement_score)
        )