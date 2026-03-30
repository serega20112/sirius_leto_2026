from pathlib import Path


class RegisterService:
    def __init__(self, register_use_case):
        self.register_use_case = register_use_case

    def register_student(self, request) -> dict:
        """
        Регистрирует студента из запроса с 3 фото.
        """
        files = request.files.getlist("photos")
        name = request.form.get("name")
        group = request.form.get("group")

        if not name or not group or len(files) != 3:
            raise ValueError("Нужно ровно 3 фото для каждого студента")

        photos_bytes = [f.read() for f in files]

        # UseCase возвращает объект Student
        student = self.register_use_case.execute(name, group, photos_bytes)

        # Здесь мы преобразуем в dict только для API
        return {
            "id": student.id,
            "name": student.name,
            "group": student.group_name,
            "photo_paths": [f"/src/assets/images/{Path(p).name}" for p in student.photo_paths],
        }
