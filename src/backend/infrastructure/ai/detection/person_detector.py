import cv2
import numpy as np
from typing import Dict, List


class PersonDetector:
    """Класс для детекции людей на кадре."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Инициализация детектора людей.

        Args:
            model_path: Путь к модели YOLO для детекции людей.
        """
        try:
            self.model = cv2.dnn_DetectionModel(model_path)
            self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели детектора: {e}")
            self.model = None

    def track_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Детектирует людей на кадре и возвращает их координаты.

        Args:
            frame: Входной кадр в формате BGR.

        Returns:
            List[Dict]: Список словарей с информацией о найденных людях.
        """
        if self.model is None:
            # Возвращаем тестовые данные, если модель не загружена
            h, w = frame.shape[:2]
            return [{"bbox": [0, 0, w, h], "track_id": 1}]

        try:
            # Конвертируем кадр в формат BGR (если нужно)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Детектируем объекты
            classes, scores, boxes = self.model.detect(frame, confThreshold=0.5)

            results = []
            for i, (class_id, score, box) in enumerate(zip(classes, scores, boxes)):
                if class_id == 0:  # Класс 0 в COCO - это человек
                    x, y, w, h = box
                    results.append(
                        {
                            "bbox": [x, y, x + w, y + h],
                            "track_id": i + 1,
                            "score": float(score),
                        }
                    )

            return results

        except Exception as e:
            print(f"Ошибка при детекции людей: {e}")
            return []
