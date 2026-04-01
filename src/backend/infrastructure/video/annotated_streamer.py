import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.backend.dependencies import settings
from src.backend.utils.cv_tools import draw_overlays


class AnnotatedVideoStreamer:
    """Инфраструктурный сервис видеопотока с AI-аннотациями."""

    def __init__(self, track_attendance_use_case):
        self.track_attendance_use_case = track_attendance_use_case

    def stream(self):
        """
        Streams operation.
        
        Args:
            None.
        
        Returns:
            A stream or generator with prepared data.
        """
        source = getattr(settings, "CAMERA_SOURCE", 0)

        def generate():
            """
            Runs the operation generate.
            
            Args:
                None.
            
            Returns:
                A stream or generator with prepared data.
            """
            cap = None
            try:
                cap = cv2.VideoCapture(
                    int(source)
                    if isinstance(source, (str, int)) and str(source).isdigit()
                    else source
                )
                if not cap.isOpened():
                    yield from self._fallback_image_generator()
                    return

                retry = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        retry += 1
                        if retry > 10:
                            print(
                                "[Video] camera read failed repeatedly, switching to fallback image"
                            )
                            yield from self._fallback_image_generator()
                            return
                        time.sleep(0.05)
                        continue
                    retry = 0
                    yield self._encode_frame(self._annotate_frame(frame))
            finally:
                if cap is not None:
                    cap.release()

        return generate()

    def _fallback_image_generator(self):
        """
        Runs the internal step fallback image generator.
        
        Args:
            None.
        
        Returns:
            The function result.
        """
        images_dir = Path(getattr(settings, "IMAGES_DIR", ""))
        first_image = None
        if images_dir and images_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files = list(images_dir.glob(ext))
                if files:
                    first_image = files[0]
                    break

        if first_image is None:
            yield (
                b"--frame\r\nContent-Type: text/plain\r\n\r\n"
                b"Camera not available and no fallback image found\r\n"
            )
            return

        image = self._read_image(first_image)
        if image is None:
            yield (
                b"--frame\r\nContent-Type: text/plain\r\n\r\n"
                b"Fallback image invalid\r\n"
            )
            return

        while True:
            yield self._encode_frame(self._annotate_frame(image.copy()))

    def _annotate_frame(self, frame):
        """
        Runs the internal step annotate frame.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The function result.
        """
        if not self.track_attendance_use_case:
            return frame

        try:
            tracking_result = self.track_attendance_use_case.execute(frame)
        except Exception as error:
            print(f"[Video] tracking error: {error}")
            print(traceback.format_exc())
            return frame

        if isinstance(tracking_result, dict) and "students" in tracking_result:
            return draw_overlays(frame, tracking_result)

        print("[Video] unexpected tracking_result type:", type(tracking_result))
        return frame

    @staticmethod
    def _encode_frame(frame):
        """
        Runs the internal step encode frame.
        
        Args:
            frame: Input value for `frame`.
        
        Returns:
            The function result.
        """
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            time.sleep(0.05)
            return b""

        chunk = jpeg.tobytes()
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"

    @staticmethod
    def _read_image(path: Path):
        """
        Runs the internal step read image.
        
        Args:
            path: Input value for `path`.
        
        Returns:
            The function result.
        """
        image = cv2.imread(str(path))
        if image is not None:
            return image

        try:
            pil = Image.open(str(path)).convert("RGB")
            array = np.array(pil)
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
