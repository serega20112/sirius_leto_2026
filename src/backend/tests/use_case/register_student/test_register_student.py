import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import pytest
from unittest.mock import MagicMock
from src.backend.use_case.register_student import RegisterStudentUseCase


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
        (
            "Иван Иванов",
            "10А",
            [b"fake_photo_data_1", b"fake_photo_data_2", b"fake_photo_data_3"],
        ),
        (
            "Петр Петров",
            "11Б",
            [b"another_fake_photo_data_1", b"another_fake_photo_data_2", b"another_fake_photo_data_3"],
        ),
    ],
)
def test_execute(mock_dependencies, name, group_name, photo_bytes):
    """
    Тестируем метод execute с параметризацией.
    Проверяем, что студент сохраняется корректно.
    """
    mock_dependencies["file_storage"].save_image.side_effect = [
        "/path/to/saved/image_1.jpg",
        "/path/to/saved/image_2.jpg",
        "/path/to/saved/image_3.jpg",
    ]
    mock_dependencies["face_recognizer"].refresh_db.return_value = None

    use_case = RegisterStudentUseCase(**mock_dependencies)
    student = use_case.execute(name, group_name, photo_bytes)

    assert student.name == name
    assert student.group_name == group_name
    assert student.photo_paths == [
        "/path/to/saved/image_1.jpg",
        "/path/to/saved/image_2.jpg",
        "/path/to/saved/image_3.jpg",
    ]
    assert mock_dependencies["file_storage"].save_image.call_count == 3
    mock_dependencies["student_repo"].save.assert_called_once_with(student)
    mock_dependencies["face_recognizer"].refresh_db.assert_called_once()


def test_execute_requires_exactly_three_photos(mock_dependencies):
    use_case = RegisterStudentUseCase(**mock_dependencies)

    with pytest.raises(ValueError, match="Нужно ровно 3 фото"):
        use_case.execute(
            "Иван Иванов",
            "10А",
            [b"only_one_photo"],
        )
