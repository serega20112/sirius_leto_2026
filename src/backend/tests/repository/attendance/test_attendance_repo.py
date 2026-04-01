from datetime import datetime
from types import SimpleNamespace

from src.backend.domain.attendance.entity import EngagementStatus
from src.backend.infrastructure.persistence.sqlite.attendance_repository import (
    SqliteAttendanceRepository,
)


def test_add_log_persists_and_returns_entity(mock_session, mock_attendance_log):
    """
    Verifies scenario add log persists and returns entity.
    
    Args:
        mock_session: Input value for `mock_session`.
        mock_attendance_log: Input value for `mock_attendance_log`.
    
    Returns:
        Does not return a value.
    """
    refreshed_model = SimpleNamespace(
        id=1,
        student_id=mock_attendance_log.student_id,
        timestamp=mock_attendance_log.timestamp,
        is_late=mock_attendance_log.is_late,
        engagement_score=mock_attendance_log.engagement_score.value,
    )

    def refresh_side_effect(model):
        """
        Verifies scenario refresh side effect.
        
        Args:
            model: Input value for `model`.
        
        Returns:
            The result of the operation.
        """
        model.id = refreshed_model.id
        model.student_id = refreshed_model.student_id
        model.timestamp = refreshed_model.timestamp
        model.is_late = refreshed_model.is_late
        model.engagement_score = refreshed_model.engagement_score

    mock_session.refresh.side_effect = refresh_side_effect

    repo = SqliteAttendanceRepository(mock_session)
    saved_log = repo.add_log(mock_attendance_log)

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()
    assert saved_log.id == 1
    assert saved_log.engagement_score is EngagementStatus.HIGH


def test_get_logs_by_student_returns_entities(mock_session):
    """
    Verifies scenario get logs by student returns entities.
    
    Args:
        mock_session: Input value for `mock_session`.
    
    Returns:
        Does not return a value.
    """
    model = SimpleNamespace(
        id=1,
        student_id="123",
        timestamp=datetime(2026, 2, 13, 10, 0, 0),
        is_late=False,
        engagement_score="medium",
    )
    mock_session.query.return_value.filter.return_value.all.return_value = [model]

    repo = SqliteAttendanceRepository(mock_session)
    logs = repo.get_logs_by_student("123")

    assert len(logs) == 1
    assert logs[0].student_id == "123"
    assert logs[0].engagement_score is EngagementStatus.MEDIUM
