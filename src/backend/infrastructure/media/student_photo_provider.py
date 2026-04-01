from pathlib import Path

import cv2


class StudentPhotoProvider:
    def __init__(self, images_dir: str | Path, thumbnail_size: int = 40):
        self.images_dir = Path(images_dir)
        self.thumbnail_size = thumbnail_size

    def read_student_thumbnail(self, filename: str) -> bytes:
        """
        Reads student thumbnail.
        
        Args:
            filename: Input value for `filename`.
        
        Returns:
            The requested data or prepared result.
        """
        image_path = self.images_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(filename)

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image {filename}")

        height, width = image.shape[:2]
        min_side = min(height, width)
        start_x = (width - min_side) // 2
        start_y = (height - min_side) // 2
        image_cropped = image[start_y : start_y + min_side, start_x : start_x + min_side]
        image_resized = cv2.resize(
            image_cropped,
            (self.thumbnail_size, self.thumbnail_size),
            interpolation=cv2.INTER_AREA,
        )

        success, buffer = cv2.imencode(".jpg", image_resized)
        if not success:
            raise RuntimeError(f"Cannot encode image {filename}")

        return buffer.tobytes()
