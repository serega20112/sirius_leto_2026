import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest
from unittest.mock import MagicMock
from src.backend.repository.student.student_repo import SqliteStudentRepository
from src.backend.domain.student.entity import Student
from src.backend.tests.common.fixtures import mock_session, mock_student


def save_test(mock_session, mock_student):
    """
    Тестируем метод save, проверяем, что данные студента сохраняются в базу данных.
    """
    repo = SqliteStudentRepository(mock_session)

    repo.save(mock_student)

    mock_session.merge.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.parametrize(
    "student_id, expected_name",
    [
        ("123", "Иван Иванов"),
        ("456", None),
    ],
)
def find_by_id_test(mock_session, student_id, expected_name):
    """
    Тестируем метод find_by_id с параметризацией.
    Проверяем, что возвращается корректный студент или None.
    """
    mock_session.query.return_value.filter.return_value.first.return_value = (
        MagicMock(
            id="123",
            name="Иван Иванов",
            group_name="10А",
            photo_path="/path/to/photo.jpg",
            created_at="2026-02-13",
        )
        if student_id == "123"
        else None
    )

    repo = SqliteStudentRepository(mock_session)
    student = repo.find_by_id(student_id)

    if expected_name:
        assert student.name == expected_name
    else:
        assert student is None


@pytest.mark.parametrize(
    "group_name, expected_count",
    [
        ("10А", 1),
        ("11Б", 0),
    ],
)
def get_groups_with_students_test(mock_session, group_name, expected_count):
    """
    Тестируем метод get_groups_with_students с параметризацией.
    Проверяем, что группы возвращаются корректно.
    """
    mock_session.query.return_value.all.return_value = (
        [
            MagicMock(
                id="123",
                name="Иван Иванов",
                group_name="10А",
                photo_path="/path/to/photo.jpg",
                created_at="2026-02-13",
            )
        ]
        if group_name == "10А"
        else []
    )

    repo = SqliteStudentRepository(mock_session)
    groups = repo.get_groups_with_students()

    assert len(groups.get(group_name, [])) == expected_count
