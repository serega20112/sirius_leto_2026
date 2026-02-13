import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_session():
    """
    Фикстура для создания мок-сессии базы данных.
    """
    return MagicMock()


@pytest.fixture
def mock_attendance_log():
    """
    Фикстура для создания тестового объекта AttendanceLog.
    """
    from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus

    return AttendanceLog(
        id=None,
        student_id="123",
        timestamp="2026-02-13 10:00:00",
        is_late=False,
        engagement_score=EngagementStatus.ACTIVE,
    )


@pytest.fixture
def mock_student():
    """
    Фикстура для создания тестового объекта Student.
    """
    from src.backend.domain.student.entity import Student

    return Student(
        id="123",
        name="Иван Иванов",
        group_name="10А",
        photo_path="/path/to/photo.jpg",
        created_at="2026-02-13",
    )
