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
        # 1. Генерируем ID
        student_id = str(uuid.uuid4())

        # Имя файла должно быть уникальным.
        # DeepFace любит, когда имя файла = ID или Имя.
        filename = f"{student_id}.jpg"

        print(f"[UseCase] Сохраняю фото: {filename}")

        # 2. Сохраняем фото на диск
        full_path = self.storage.save_image(photo_bytes, filename)
        print(f"[UseCase] Фото сохранено по пути: {full_path}")

        # 3. Создаем объект
        student = Student(
            id=student_id,
            name=name,
            group_name=group_name,
            photo_path=full_path,
            created_at=datetime.now(),
        )

        # 4. Пишем в БД
        print("[UseCase] Сохраняю в БД...")
        self.repo.save(student)

        # 5. Обновляем DeepFace (удаляем старый кэш)
        print("[UseCase] Обновляю кэш DeepFace...")
        self.recognizer.refresh_db()

        return student
