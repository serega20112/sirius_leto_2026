# src/backend/repository/student_repository.py
from typing import List
from src.backend.domain.student.entity import Student
from src.backend.infrastructure.database import StudentModel
from src.backend.repository import BaseRepository

class SqliteStudentRepository(BaseRepository):
    def save(self, student: Student) -> None:
        model = StudentModel.from_domain(student)
        self.session.merge(model)
        self.session.commit()

    def find_by_id(self, student_id: str) -> Student | None:
        model = self.session.query(StudentModel).filter_by(id=student_id).first()
        return model.to_domain() if model else None

    def get_all(self) -> List[Student]:
        return [m.to_domain() for m in self.session.query(StudentModel).all()]