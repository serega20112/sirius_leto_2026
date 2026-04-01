from src.backend.domain.student.entity import Student


class GetGroupsUseCase:
    """Query-use-case группировки студентов для UI."""

    def __init__(self, student_repo):
        self.student_repo = student_repo

    def execute(self) -> dict[str, list[Student]]:
        """
        Executes the main scenario for GetGroupsUseCase.
        
        Args:
            None.
        
        Returns:
            The scenario execution result.
        """
        grouped_students: dict[str, list[Student]] = {}
        for student in self.student_repo.get_all():
            grouped_students.setdefault(student.group_name, []).append(student)

        for students in grouped_students.values():
            students.sort(key=lambda student: student.name.casefold())

        return dict(sorted(grouped_students.items(), key=lambda item: item[0].casefold()))
