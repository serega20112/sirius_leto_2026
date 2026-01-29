import uuid
from datetime import datetime
from src.backend.domain.student.entity import Student


class RegisterStudentUseCase:
    """Сценарий регистрации нового студента с сохранением фото."""

    def __init__(self, student_repo, file_storage, face_recognizer):
        self.repo = student_repo
        self.storage = file_storage
        self.recognizer = face_recognizer

    def execute(self, name: str, group_name: str, photo_bytes: bytes) -> Student:
        """
        Регистрирует студента, сохраняет фото и обновляет базу лиц.
        Возвращает созданную сущность Student.
        """
        student_id = str(uuid.uuid4())
        filename = f"{student_id}.jpg"

        full_path = self.storage.save_image(photo_bytes, filename)

        student = Student(
            id=student_id,
            name=name,
            group_name=group_name,
            photo_path=full_path,
            created_at=datetime.now()
        )

        self.repo.save(student)
        self.recognizer.refresh_db()

        return student