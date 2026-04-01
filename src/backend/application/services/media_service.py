class MediaApplicationService:
    def __init__(self, student_photo_provider):
        self.student_photo_provider = student_photo_provider

    def get_student_photo(self, filename: str) -> bytes:
        """
        Load a prepared thumbnail for a public student photo request.

        Args:
            filename: File name requested by the delivery layer.

        Returns:
            JPEG bytes for the transformed student image.
        """
        return self.student_photo_provider.read_student_thumbnail(filename)
