from pathlib import Path


class StudentApplicationService:
    def __init__(self, register_student_use_case, get_groups_use_case):
        self.register_student_use_case = register_student_use_case
        self.get_groups_use_case = get_groups_use_case

    def register_student(
        self,
        name: str,
        group_name: str,
        photos_bytes: list[bytes],
    ) -> dict:
        """
        Register a student and prepare the response payload for the route layer.

        Args:
            name: Student name received from the registration form.
            group_name: Student group name received from the registration form.
            photos_bytes: Raw bytes for the uploaded student photos.

        Returns:
            A serialized student payload ready for JSON output.
        """
        student = self.register_student_use_case.execute(
            name=name,
            group_name=group_name,
            photos_bytes=photos_bytes,
        )
        return {
            "id": student.id,
            "name": student.name,
            "group": student.group_name,
            "photo_paths": [
                self._build_public_photo_url(path) for path in student.photo_paths
            ],
        }

    def get_groups(self) -> dict[str, list[dict]]:
        """
        Load all students grouped by their class name for the groups page.

        Args:
            None.

        Returns:
            A dictionary where each key is a group name and each value is a list
            of serialized students.
        """
        grouped_students = self.get_groups_use_case.execute()
        return {
            group_name: [
                {
                    "id": student.id,
                    "name": student.name,
                    "photo": self._build_public_photo_url(student.photo_paths[0])
                    if student.photo_paths
                    else "",
                }
                for student in students
            ]
            for group_name, students in grouped_students.items()
        }

    @staticmethod
    def _build_public_photo_url(photo_path: str) -> str:
        """
        Convert a stored file path into a public image URL.

        Args:
            photo_path: Stored image path returned by the file storage layer.

        Returns:
            A URL that can be used by the frontend to request the image.
        """
        return f"/src/assets/images/{Path(photo_path).name}"
