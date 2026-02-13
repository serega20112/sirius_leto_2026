import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import pytest
from unittest.mock import MagicMock
from src.backend.use_case.register_student import RegisterStudentUseCase
from src.backend.domain.student.entity import Student


@pytest.fixture
def mock_dependencies():
    """
    Фикстура для создания моков зависимостей.
    """
    return {
        "student_repo": MagicMock(),
        "file_storage": MagicMock(),
        "face_recognizer": MagicMock(),
    }


@pytest.mark.parametrize(
    "name, group_name, photo_bytes",
    [
        ("Иван Иванов", "10А", b"fake_photo_data"),
        ("Петр Петров", "11Б", b"another_fake_photo_data"),
    ],
)
def test_execute(mock_dependencies, name, group_name, photo_bytes):
    """
    Тестируем метод execute с параметризацией.
    Проверяем, что студент сохраняется корректно.
    """
    mock_dependencies["file_storage"].save_image.return_value = (
        "/path/to/saved/image.jpg"
    )
    mock_dependencies["face_recognizer"].refresh_db.return_value = None

    use_case = RegisterStudentUseCase(**mock_dependencies)
    student = use_case.execute(name, group_name, photo_bytes)

    assert student.name == name
    assert student.group_name == group_name
    assert student.photo_path == "/path/to/saved/image.jpg"
    mock_dependencies["student_repo"].save.assert_called_once_with(student)
    mock_dependencies["face_recognizer"].refresh_db.assert_called_once()
