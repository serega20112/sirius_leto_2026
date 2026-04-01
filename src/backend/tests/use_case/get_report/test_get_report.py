import pytest
from unittest.mock import MagicMock
from datetime import datetime
from src.backend.use_case.get_report import GetReportUseCase


@pytest.fixture
def mock_dependencies():
    """
    Verifies scenario mock dependencies.
    
    Args:
        None.
    
    Returns:
        The function result.
    """
    return {
        "attendance_repo": MagicMock(),
        "student_repo": MagicMock(),
    }


@pytest.mark.parametrize(
    "attendance_exists, student_exists, expected_report",
    [
        (
            True,
            True,
            [
                {
                    "id": 1,
                    "student_name": "Иван Иванов",
                    "timestamp": "2026-02-13T10:00:00",
                    "is_late": False,
                    "status": "present",
                    "arrived_at": "10:00",
                    "engagement": "high",
                }
            ],
        ),
        (
            False,
            False,
            [],
        ),
    ],
)
def test_execute(mock_dependencies, attendance_exists, student_exists, expected_report):
    """
    Verifies scenario execute.
    
    Args:
        mock_dependencies: Input value for `mock_dependencies`.
        attendance_exists: Input value for `attendance_exists`.
        student_exists: Input value for `student_exists`.
        expected_report: Input value for `expected_report`.
    
    Returns:
        Does not return a value.
    """

    if attendance_exists:
        log = MagicMock()
        log.student_id = "123"
        log.timestamp = datetime.strptime("2026-02-13 10:00:00", "%Y-%m-%d %H:%M:%S")
        log.id = 1
        log.is_late = False

        engagement = MagicMock()
        engagement.value = "high"
        log.engagement_score = engagement

        mock_dependencies["attendance_repo"].get_all_logs.return_value = [log]
    else:
        mock_dependencies["attendance_repo"].get_all_logs.return_value = []

    if student_exists:
        student = MagicMock()
        student.id = "123"
        student.name = "Иван Иванов"
        mock_dependencies["student_repo"].get_all.return_value = [student]
    else:
        mock_dependencies["student_repo"].get_all.return_value = []

    use_case = GetReportUseCase(**mock_dependencies)
    report = use_case.execute()

    assert report == expected_report


def test_execute_returns_only_latest_log_per_student(mock_dependencies):
    """
    Verifies scenario execute returns only the newest log for one student.
    
    Args:
        mock_dependencies: Input value for `mock_dependencies`.
    
    Returns:
        Does not return a value.
    """
    older_log = MagicMock()
    older_log.student_id = "123"
    older_log.timestamp = datetime.strptime("2026-02-13 09:00:00", "%Y-%m-%d %H:%M:%S")
    older_log.id = 1
    older_log.is_late = False
    older_engagement = MagicMock()
    older_engagement.value = "medium"
    older_log.engagement_score = older_engagement

    newer_log = MagicMock()
    newer_log.student_id = "123"
    newer_log.timestamp = datetime.strptime("2026-02-13 10:00:00", "%Y-%m-%d %H:%M:%S")
    newer_log.id = 2
    newer_log.is_late = True
    newer_engagement = MagicMock()
    newer_engagement.value = "low"
    newer_log.engagement_score = newer_engagement

    mock_dependencies["attendance_repo"].get_all_logs.return_value = [older_log, newer_log]

    student = MagicMock()
    student.id = "123"
    student.name = "Иван Иванов"
    mock_dependencies["student_repo"].get_all.return_value = [student]

    use_case = GetReportUseCase(**mock_dependencies)
    report = use_case.execute()

    assert report == [
        {
            "id": 2,
            "student_name": "Иван Иванов",
            "timestamp": "2026-02-13T10:00:00",
            "is_late": True,
            "status": "late",
            "arrived_at": "10:00",
            "engagement": "low",
        }
    ]
