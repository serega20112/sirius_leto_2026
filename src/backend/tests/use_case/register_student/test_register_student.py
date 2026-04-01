import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import pytest
from unittest.mock import MagicMock
from src.backend.application.exceptions import ValidationError
from src.backend.use_case.register_student import RegisterStudentUseCase


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
    Verifies scenario execute.
    
    Args:
        mock_dependencies: Input value for `mock_dependencies`.
        name: Input value for `name`.
        group_name: Input value for `group_name`.
        photo_bytes: Input value for `photo_bytes`.
    
    Returns:
        Does not return a value.
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
    """
    Verifies scenario execute requires exactly three photos.
    
    Args:
        mock_dependencies: Input value for `mock_dependencies`.
    
    Returns:
        Does not return a value.
    """
    use_case = RegisterStudentUseCase(**mock_dependencies)

    with pytest.raises(ValidationError, match="Нужно ровно 3 фото"):
        use_case.execute(
            "Иван Иванов",
            "10А",
            [b"only_one_photo"],
        )


def test_execute_requires_name_and_group(mock_dependencies):
    """
    Verifies scenario execute requires name and group.
    
    Args:
        mock_dependencies: Input value for `mock_dependencies`.
    
    Returns:
        Does not return a value.
    """
    use_case = RegisterStudentUseCase(**mock_dependencies)

    with pytest.raises(ValidationError, match="Имя обязательно"):
        use_case.execute("", "10А", [b"1", b"2", b"3"])

    with pytest.raises(ValidationError, match="Группа обязательна"):
        use_case.execute("Иван Иванов", "", [b"1", b"2", b"3"])
