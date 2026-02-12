class RegisterService:
    def __init__(self, register_use_case):
        self.register_use_case = register_use_case

    def register_student(self, request) -> dict:
        """
        Регистрирует студента из запроса и возвращает данные для ответа.

        Args:
            request: Flask request объект.

        Returns:
            dict: Данные студента.

        Raises:
            ValueError: Если отсутствуют обязательные поля.
        """
        file = request.files.get("photo") or next(iter(request.files.values()), None)
        name = request.form.get("name")
        group = request.form.get("group")

        if not name or not group or not file:
            raise ValueError("Missing name, group or photo")

        photo_bytes = file.read()
        student = self.register_use_case.execute(name, group, photo_bytes)
        return {
            "id": student.id,
            "name": student.name,
            "group": student.group_name,
            "photo": f"{student.id}.jpg"
        }
