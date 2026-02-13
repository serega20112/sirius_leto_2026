from typing import List, Dict
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
            created_at=student.created_at,
        )
        self.session.merge(student_model)
        self.session.commit()

    def find_by_id(self, student_id: str) -> Student | None:
        """Находит студента по уникальному ID."""
        model = (
            self.session.query(StudentModel)
            .filter(StudentModel.id == student_id)
            .first()
        )
        return self._to_entity(model) if model else None

    def find_by_name(self, name: str) -> Student | None:
        """Находит студента по имени."""
        model = (
            self.session.query(StudentModel).filter(StudentModel.name == name).first()
        )
        return self._to_entity(model) if model else None

    def get_all(self) -> List[Student]:
        """Возвращает список всех студентов."""
        models = self.session.query(StudentModel).all()
        return [self._to_entity(m) for m in models]

    def get_groups_with_students(self) -> Dict[str, List[dict]]:
        """
        Возвращает студентов, сгруппированных по группам.
        Сортирует студентов в каждой группе по имени и формирует путь к фото.
        """
        groups = {}
        for student in self.get_all():
            groups.setdefault(student.group_name, []).append(
                {
                    "id": student.id,
                    "name": student.name,
                    "photo": f"/src/assets/images/{student.id}.jpg",
                }
            )
        for group in groups.values():
            group.sort(key=lambda x: x["name"])
        return groups

    def _to_entity(self, model: StudentModel) -> Student:
        """Преобразует ORM модель в доменную сущность."""
        return Student(
            id=model.id,
            name=model.name,
            group_name=model.group_name,
            photo_path=model.photo_path,
            created_at=model.created_at,
        )
