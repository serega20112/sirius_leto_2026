from src.backend.domain.student.entity import Student
from src.backend.repository import SqliteStudentRepository, BaseRepository


class StudentDependencies:
    """Содержит все зависимости для работы со студентами."""

    def __init__(self):
        self.student_repo = SqliteStudentRepository(BaseRepository)

    def get_all_students(self):
        return self.student_repo.get_all()

    def get_student_by_id(self, student_id: int):
        return self.student_repo.get_by_id(student_id)
