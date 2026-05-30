import cv2
import numpy as np
import importlib
from typing import Dict, List, Optional, Tuple, Any


class PoseEstimator:

    def _detect_device(self) -> str:
        """Detects best device to run pose estimator on (prefer CUDA if available)."""
        try:
            import importlib

            torch = importlib.import_module("torch")
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def __init__(self, model_path: str = ""):
        """Initialize pose estimator with lazy imports.

        The class avoids importing heavy libraries at module import time so the
        package can be imported on edge devices that do not have torch/ultralytics.
        """
        self.model: Any = None
        self.is_ultralytics: bool = False
        self.device = self._detect_device()
        self.input_size = (192, 256)

        # YOLO/ultralytics/ONNX disabled for edge deployment.
        # Rely on MediaPipe backends in higher-level code for lightweight pose estimation.
        try:
            self.model = None
            self.is_ultralytics = False
        except Exception as e:
            print(f"Ошибка при инициализации pose estimator: {e}")
            self.model = None
            self.is_ultralytics = False
        print("[AI] YOLO pose disabled; using MediaPipe-only pose estimator")

    def estimate_pose(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict]:
        """
        Estimates pose.

        Args:
            frame: Input value for `frame`.
            bbox: Input value for `bbox`.

        Returns:
            The computed or transformed result.
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
                    results = self.model(
                        person_img,
                        verbose=False,
                        device=self.device,
                    )
                    if hasattr(results, "__len__") and len(results) == 0:
                        return {"keypoints": [], "bbox": bbox}
                    result = results[0]
                    keypoints = self._process_output(result, (x1, y1))
                except Exception as e:
                    print(f"Ошибка при вызове ultralytics модели: {e}")
                    return None
            else:
                resized = cv2.resize(person_img, self.input_size)
                blob = cv2.dnn.blobFromImage(
                    resized, scalefactor=1.0 / 255.0, size=self.input_size, swapRB=True
                )
                self.model.setInput(blob)
                output = self.model.forward()
                keypoints = self._process_output(output, (x1, y1))
            return {"keypoints": keypoints, "bbox": bbox}
        except Exception as e:
            print(f"Ошибка при оценке позы: {e}")
            return None

    def _process_output(self, output: Any, offset: Tuple[int, int]) -> List[Dict]:
        """
        Runs the internal step process output.

        Args:
            output: Input value for `output`.
            offset: Input value for `offset`.

        Returns:
            The function result.
        """
        ox, oy = offset
        keypoints: List[Dict] = []
        try:
            if hasattr(output, "keypoints"):
                kps = output.keypoints
                xy = getattr(kps, "xy", None)
                if xy is None:
                    return []

                xy_array = self._to_numpy(xy)
                if xy_array is None or xy_array.size == 0:
                    return []

                conf = getattr(kps, "conf", None)
                conf_array = self._to_numpy(conf) if conf is not None else None

                if xy_array.ndim == 2:
                    xy_array = np.expand_dims(xy_array, axis=0)
                if conf_array is not None and conf_array.ndim == 1:
                    conf_array = np.expand_dims(conf_array, axis=0)

                points = xy_array[0]
                confidences = conf_array[0] if conf_array is not None else None
                for index, point in enumerate(points):
                    if len(point) < 2:
                        continue

                    point_confidence = (
                        float(confidences[index])
                        if confidences is not None and index < len(confidences)
                        else 1.0
                    )
                    keypoints.append(
                        {
                            "x": float(point[0]) + ox,
                            "y": float(point[1]) + oy,
                            "conf": point_confidence,
                        }
                    )
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

    @staticmethod
    def _to_numpy(value: Any) -> Optional[np.ndarray]:
        """
        Converts tensor-like objects to a detached numpy array.

        Args:
            value: Tensor-like object returned by the backend.

        Returns:
            A numpy array or `None` when conversion is not possible.
        """
        if value is None:
            return None

        if hasattr(value, "detach"):
            try:
                return value.detach().cpu().numpy()
            except Exception:
                try:
                    # fallback for other tensor-like types
                    return np.asarray(value)
                except Exception:
                    return None

        try:
            return np.asarray(value)
        except Exception:
            return None

    def _configure_opencv_backend(self) -> None:
        """
        Configures the OpenCV DNN backend for the available device.

        Args:
            None.

        Returns:
            Does not return a value.
        """
        if self.model is None or self.device != "cuda":
            return

        try:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as error:
            print(f"[AI] OpenCV DNN CUDA init error: {error}")
