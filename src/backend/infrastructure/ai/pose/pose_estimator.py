import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class PoseEstimator:

    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        """
        Инициализация оценщика позы.
        
        Args:
            model_path: Путь к модели для оценки позы.
        """
        self.model: Any = None
        self.is_ultralytics: bool = False
        self.input_size = (192, 256)
        try:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.is_ultralytics = True
            except Exception:
                self.model = cv2.dnn.readNetFromONNX(model_path)
                self.is_ultralytics = False
        except Exception as e:
            print(f"Ошибка при загрузке модели оценки позы: {e}")
            self.model = None
            self.is_ultralytics = False

    def estimate_pose(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict]:
        """
        Оценивает позу человека в заданном bounding box.
        
        Args:
            frame: Входной кадр в формате BGR.
            bbox: Координаты bounding box [x1, y1, x2, y2].
            
        Returns:
            Optional[Dict]: Словарь с ключевыми точками или None при ошибке.
        """
        if self.model is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                return None
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None
            if self.is_ultralytics:
                try:
                    results = self.model(person_img)
                    # explicit length check to avoid ambiguous truth value of Results
                    if hasattr(results, '__len__') and len(results) == 0:
                        return {'keypoints': [], 'bbox': bbox}
                    result = results[0]
                    keypoints = self._process_output(result, (x1, y1))
                except Exception as e:
                    print(f"Ошибка при вызове ultralytics модели: {e}")
                    return None
            else:
                resized = cv2.resize(person_img, self.input_size)
                blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0/255.0, size=self.input_size, swapRB=True)
                self.model.setInput(blob)
                output = self.model.forward()
                keypoints = self._process_output(output, (x1, y1))
            return {'keypoints': keypoints, 'bbox': bbox}
        except Exception as e:
            print(f"Ошибка при оценке позы: {e}")
            return None
    
    def _process_output(self, output: Any, offset: Tuple[int, int]) -> List[Dict]:
        """
        Обрабатывает выход модели в ключевые точки.
        
        Args:
            output: Выход модели.
            offset: Смещение координат (x, y) для перевода в координаты исходного изображения.
            
        Returns:
            List[Dict]: Список ключевых точек с координатами и уверенностью.
        """
        ox, oy = offset
        keypoints: List[Dict] = []
        try:
            if hasattr(output, 'keypoints'):
                kps = output.keypoints
                try:
                    arr = np.asarray(kps)
                except Exception:
                    try:
                        arr = kps.xy
                    except Exception:
                        arr = None
                if arr is None:
                    return []
                arr = np.array(arr)
                if arr.ndim == 2:
                    for kp in arr:
                        if kp.size >= 2:
                            x, y = float(kp[0]) + ox, float(kp[1]) + oy
                            conf = float(kp[2]) if kp.size > 2 else 1.0
                            keypoints.append({'x': x, 'y': y, 'conf': conf})
                elif arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                    for kp in arr:
                        if kp.size >= 2:
                            x, y = float(kp[0]) + ox, float(kp[1]) + oy
                            conf = float(kp[2]) if kp.size > 2 else 1.0
                            keypoints.append({'x': x, 'y': y, 'conf': conf})
                return keypoints
            if isinstance(output, np.ndarray):
                return []
            try:
                out_arr = np.asarray(output)
                return []
            except Exception:
                return []
        except Exception:
            return []
