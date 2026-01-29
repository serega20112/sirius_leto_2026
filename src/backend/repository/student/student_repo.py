from typing import List, Optional

from src.backend.domain.student.entity import Student
from src.backend.infrastructure.database import StudentModel
from src.backend.repository import BaseRepository


class SqliteStudentRepository(BaseRepository):
    """Репозиторий для работы со студентами через SQLite."""

    def save(self, student: Student) -> None:
        """Сохраняет или обновляет данные студента в БД."""
        student_model = StudentModel(
            id=student.id,
            name=student.name,
            group_name=student.group_name,
            photo_path=student.photo_path,
            created_at=student.created_at
        )
        self.session.merge(student_model)
        self.session.commit()

    def find_by_id(self, student_id: str) -> Optional[Student]:
        """Находит студента по уникальному ID."""
        model = self.session.query(StudentModel).filter(StudentModel.id == student_id).first()
        if not model:
            return None
        return self._to_entity(model)

    def find_by_name(self, name: str) -> Optional[Student]:
        """Находит студента по имени."""
        model = self.session.query(StudentModel).filter(StudentModel.name == name).first()
        if not model:
            return None
        return self._to_entity(model)

    def get_all(self) -> List[Student]:
        """Возвращает список всех студентов."""
        models = self.session.query(StudentModel).all()
        return [self._to_entity(m) for m in models]

    def _to_entity(self, model: StudentModel) -> Student:
        """Преобразует ORM модель в доменную сущность."""
        return Student(
            id=model.id,
            name=model.name,
            group_name=model.group_name,
            photo_path=model.photo_path,
            created_at=model.created_at
        )