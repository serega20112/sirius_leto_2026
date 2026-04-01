from datetime import datetime
from types import SimpleNamespace

from src.backend.infrastructure.persistence.sqlite.student_repository import (
    SqliteStudentRepository,
)


def test_save_persists_student(mock_session, mock_student):
    """
    Verifies scenario save persists student.
    
    Args:
        mock_session: Input value for `mock_session`.
        mock_student: Input value for `mock_student`.
    
    Returns:
        Does not return a value.
    """
    repo = SqliteStudentRepository(mock_session)

    repo.save(mock_student)

    mock_session.merge.assert_called_once()
    mock_session.commit.assert_called_once()


def test_find_by_id_returns_domain_student(mock_session):
    """
    Verifies scenario find by id returns domain student.
    
    Args:
        mock_session: Input value for `mock_session`.
    
    Returns:
        Does not return a value.
    """
    model = SimpleNamespace(
        to_domain=lambda: SimpleNamespace(id="123", name="Иван Иванов"),
    )
    mock_session.query.return_value.filter_by.return_value.first.return_value = model

    repo = SqliteStudentRepository(mock_session)
    student = repo.find_by_id("123")

    assert student.id == "123"
    assert student.name == "Иван Иванов"


def test_find_by_name_returns_domain_student(mock_session):
    """
    Verifies scenario find by name returns domain student.
    
    Args:
        mock_session: Input value for `mock_session`.
    
    Returns:
        Does not return a value.
    """
    model = SimpleNamespace(
        to_domain=lambda: SimpleNamespace(id="123", name="Иван Иванов"),
    )
    mock_session.query.return_value.filter_by.return_value.first.return_value = model

    repo = SqliteStudentRepository(mock_session)
    student = repo.find_by_name("Иван Иванов")

    assert student.id == "123"
    assert student.name == "Иван Иванов"


def test_get_all_returns_domain_students(mock_session):
    """
    Verifies scenario get all returns domain students.
    
    Args:
        mock_session: Input value for `mock_session`.
    
    Returns:
        Does not return a value.
    """
    created_at = datetime(2026, 2, 13)
    models = [
        SimpleNamespace(
            to_domain=lambda: SimpleNamespace(
                id="123",
                name="Иван Иванов",
                group_name="10А",
                created_at=created_at,
            )
        )
    ]
    mock_session.query.return_value.all.return_value = models

    repo = SqliteStudentRepository(mock_session)
    students = repo.get_all()

    assert len(students) == 1
    assert students[0].group_name == "10А"
