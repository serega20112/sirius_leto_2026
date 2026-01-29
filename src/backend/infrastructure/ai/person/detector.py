from ultralytics import YOLO
import numpy as np


class PersonDetector:
    """Обертка над YOLO для обнаружения людей."""

    def __init__(self, model_path: str):
        """Загружает модель YOLO."""
        self.model = YOLO(model_path)

    def detect_people(self, frame: np.ndarray):
        """
        Находит людей на изображении.
        Возвращает список bounding boxes [x1, y1, x2, y2].
        """
        results = self.model(frame, classes=[0], verbose=False)

        boxes = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(coords)

        return boxes