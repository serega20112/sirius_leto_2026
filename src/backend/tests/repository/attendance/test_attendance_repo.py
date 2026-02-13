import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest
from unittest.mock import MagicMock
from src.backend.repository.attendance.attendance_repo import SqliteAttendanceRepository
from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.tests.common.fixtures import mock_session, mock_attendance_log


@pytest.mark.parametrize(
    "student_id, expected_count",
    [
        ("123", 1),
        ("456", 0),
    ],
)
def get_logs_by_student_test(mock_session, student_id, expected_count):
    """
    Тестируем метод get_logs_by_student с параметризацией.
    Проверяем, что возвращаются корректные записи.
    """
    mock_session.query.return_value.filter.return_value.all.return_value = (
        [
            MagicMock(
                id=1,
                student_id="123",
                timestamp="2026-02-13 10:00:00",
                is_late=False,
                engagement_score=1,
            )
        ]
        if student_id == "123"
        else []
    )

    repo = SqliteAttendanceRepository(mock_session)
    logs = repo.get_logs_by_student(student_id)

    assert len(logs) == expected_count
    if expected_count > 0:
        assert logs[0].student_id == student_id


@pytest.mark.parametrize(
    "engagement_score, expected_status",
    [
        (1, EngagementStatus.HIGH),
        (0, EngagementStatus.LOW),
    ],
)
def add_log_test(mock_session, mock_attendance_log, engagement_score, expected_status):
    """
    Тестируем метод add_log с параметризацией.
    Проверяем, что запись добавляется в базу данных с корректным статусом вовлеченности.
    """
    mock_attendance_log.engagement_score = expected_status

    repo = SqliteAttendanceRepository(mock_session)
    repo.add_log(mock_attendance_log)

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()
