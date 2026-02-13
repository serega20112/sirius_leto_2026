import pytest
from unittest.mock import MagicMock
from datetime import datetime
from src.backend.use_case.get_report import GetReportUseCase


@pytest.fixture
def mock_dependencies():
    """Создает моки репозиториев для изоляции use case."""
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
    Тестирует формирование журнала посещаемости.
    Ожидается корректный список логов с именами студентов.
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
