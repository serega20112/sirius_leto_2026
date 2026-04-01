from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.domain.attendance.repository import AttendanceRepository
from src.backend.infrastructure.database import AttendanceModel
from src.backend.infrastructure.persistence.sqlite.base_repository import SqliteRepository


class SqliteAttendanceRepository(SqliteRepository, AttendanceRepository):
    def add_log(self, log: AttendanceLog) -> AttendanceLog:
        """
        Adds log.
        
        Args:
            log: Input value for `log`.
        
        Returns:
            The result of the operation.
        """
        model = AttendanceModel(
            student_id=log.student_id,
            timestamp=log.timestamp,
            is_late=log.is_late,
            engagement_score=log.engagement_score.value,
        )
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        return self._to_entity(model)

    def get_logs_by_student(self, student_id: str) -> list[AttendanceLog]:
        """
        Gets logs by student.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        models = (
            self.session.query(AttendanceModel)
            .filter(AttendanceModel.student_id == student_id)
            .all()
        )
        return [self._to_entity(model) for model in models]

    def get_all_logs(self) -> list[AttendanceLog]:
        """
        Gets all logs.
        
        Args:
            None.
        
        Returns:
            The requested data or prepared result.
        """
        models = (
            self.session.query(AttendanceModel)
            .order_by(AttendanceModel.timestamp.desc())
            .all()
        )
        return [self._to_entity(model) for model in models]

    def get_stats_by_student(self, student_id: str) -> list[AttendanceLog]:
        """
        Gets stats by student.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        models = (
            self.session.query(AttendanceModel)
            .filter(AttendanceModel.student_id == student_id)
            .order_by(AttendanceModel.timestamp.asc())
            .all()
        )
        return [self._to_entity(model) for model in models]

    @staticmethod
    def _to_entity(model: AttendanceModel) -> AttendanceLog:
        """
        Runs the internal step to entity.
        
        Args:
            model: Input value for `model`.
        
        Returns:
            The function result.
        """
        return AttendanceLog(
            id=model.id,
            student_id=model.student_id,
            timestamp=model.timestamp,
            is_late=model.is_late,
            engagement_score=EngagementStatus(model.engagement_score),
        )
