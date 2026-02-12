from typing import List, Dict

from src.backend.domain.student.entity import Student


class StudentService:
    def __init__(self, student_repo):
        self.student_repo = student_repo

    def get_all_students(self) -> List[Student]:
        """
        Возвращает список всех студентов.

        Returns:
            List[Student]: Список студентов.
        """
        return self.student_repo.get_all()

    def get_student_by_id(self, student_id: str) -> Student | None:
        """
        Находит студента по ID.

        Args:
            student_id (str): ID студента.

        Returns:
            Student | None: Студент или None.
        """
        return self.student_repo.find_by_id(student_id)

    def get_groups_with_students(self) -> Dict[str, List[dict]]:
        """
        Возвращает группы с студентами.

        Returns:
            Dict[str, List[dict]]: Группы.
        """
        return self.student_repo.get_groups_with_students()
