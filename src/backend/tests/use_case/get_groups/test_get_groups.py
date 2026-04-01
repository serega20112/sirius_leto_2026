from datetime import datetime

from src.backend.domain.student.entity import Student
from src.backend.use_case.get_groups import GetGroupsUseCase


def test_execute_groups_students_by_group_and_name():
    """
    Verifies scenario execute groups students by group and name.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    students = [
        Student(
            id="2",
            name="Борис",
            group_name="Б",
            photo_paths=["/tmp/2.jpg"],
            created_at=datetime(2026, 2, 13),
        ),
        Student(
            id="1",
            name="Алексей",
            group_name="А",
            photo_paths=["/tmp/1.jpg"],
            created_at=datetime(2026, 2, 13),
        ),
        Student(
            id="3",
            name="Анна",
            group_name="А",
            photo_paths=["/tmp/3.jpg"],
            created_at=datetime(2026, 2, 13),
        ),
    ]

    class StubStudentRepository:
        def get_all(self):
            """
            Verifies scenario get all.
            
            Args:
                None.
            
            Returns:
                The requested data or prepared result.
            """
            return students

    result = GetGroupsUseCase(StubStudentRepository()).execute()

    assert list(result.keys()) == ["А", "Б"]
    assert [student.name for student in result["А"]] == ["Алексей", "Анна"]
