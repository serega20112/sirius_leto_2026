from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.domain.student.entity import Student


@pytest.fixture
def mock_session():
    """
    Verifies scenario mock session.
    
    Args:
        None.
    
    Returns:
        The function result.
    """
    return MagicMock()


@pytest.fixture
def mock_attendance_log():
    """
    Verifies scenario mock attendance log.
    
    Args:
        None.
    
    Returns:
        The function result.
    """
    return AttendanceLog(
        id=None,
        student_id="123",
        timestamp=datetime(2026, 2, 13, 10, 0, 0),
        is_late=False,
        engagement_score=EngagementStatus.HIGH,
    )


@pytest.fixture
def mock_student():
    """
    Verifies scenario mock student.
    
    Args:
        None.
    
    Returns:
        The function result.
    """
    return Student(
        id="123",
        name="Иван Иванов",
        group_name="10А",
        photo_paths=[
            "/path/to/photo_1.jpg",
            "/path/to/photo_2.jpg",
            "/path/to/photo_3.jpg",
        ],
        created_at=datetime(2026, 2, 13, 0, 0, 0),
    )
