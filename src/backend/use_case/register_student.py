# src/backend/use_case/register_student.py
import uuid
from datetime import datetime
from src.backend.domain.student.entity import Student

class RegisterStudentUseCase:
    def __init__(self, student_repo, file_storage, face_recognizer):
        self.repo = student_repo
        self.storage = file_storage
        self.recognizer = face_recognizer

    def execute(self, name: str, group_name: str, photos_bytes: list[bytes]) -> Student:
        if len(photos_bytes) != 3:
            raise ValueError("Нужно ровно 3 фото")

        student_id = str(uuid.uuid4())
        photo_paths = []

        for idx, photo in enumerate(photos_bytes, start=1):
            filename = f"{student_id}_{idx}.jpg"
            full_path = self.storage.save_image(photo, filename)
            photo_paths.append(full_path)

        student = Student(
            id=student_id,
            name=name,
            group_name=group_name,
            photo_paths=photo_paths,
            created_at=datetime.now()
        )

        self.repo.save(student)
        self.recognizer.refresh_db()
        return student  # возвращаем объект Student