import cv2
from typing import Dict, Any
from src.backend.utils.cv_tools import draw_overlays
from src.backend.dependencies import settings
from pathlib import Path
import time
from PIL import Image
import numpy as np
import traceback


class MonitorService:
    def __init__(self, track_attendance_use_case, get_report_use_case, student_service):
        self.track_attendance_use_case = track_attendance_use_case
        self.get_report_use_case = get_report_use_case
        self.student_service = student_service

    def _read_image(self, path: Path):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        try:
            pil = Image.open(str(path)).convert('RGB')
            arr = np.array(pil)
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _fallback_image_generator(self):
        images_dir = Path(getattr(settings, 'IMAGES_DIR', ''))
        first_image = None
        if images_dir and images_dir.exists():
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                files = list(images_dir.glob(ext))
                if files:
                    first_image = files[0]
                    break
        if first_image is None:
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available and no fallback image found\r\n'
            return

        img = self._read_image(first_image)
        if img is None:
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nFallback image invalid\r\n'
            return

        while True:
            frame = img.copy()
            if self.track_attendance_use_case and getattr(self.track_attendance_use_case, 'person_detector', None):
                try:
                    tracking_result = self.track_attendance_use_case.execute(frame)
                    if isinstance(tracking_result, dict) and 'students' in tracking_result:
                        frame = draw_overlays(frame, tracking_result)
                    else:
                        print('[Monitor] unexpected tracking_result type:', type(tracking_result))
                except Exception as e:
                    print(f"[Monitor] fallback tracking error: {e}")
                    print(traceback.format_exc())
            ret2, jpeg = cv2.imencode('.jpg', frame)
            if not ret2:
                time.sleep(0.1)
                continue
            chunk = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')

    def stream_video(self):
        source = getattr(settings, 'CAMERA_SOURCE', 0)

        def generate():
            cap = None
            try:
                cap = cv2.VideoCapture(int(source) if isinstance(source, (str, int)) and str(source).isdigit() else source)
                if not cap.isOpened():
                    yield from self._fallback_image_generator()
                    return

                retry = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        retry += 1
                        if retry > 10:
                            print("[Monitor] camera read failed repeatedly, switching to fallback image")
                            yield from self._fallback_image_generator()
                            return
                        time.sleep(0.05)
                        continue
                    retry = 0

                    if self.track_attendance_use_case and getattr(self.track_attendance_use_case, 'person_detector', None):
                        try:
                            tracking_result = self.track_attendance_use_case.execute(frame)
                            if isinstance(tracking_result, dict) and 'students' in tracking_result:
                                frame = draw_overlays(frame, tracking_result)
                            else:
                                print('[Monitor] unexpected tracking_result type:', type(tracking_result))
                        except Exception as e:
                            print(f"[Monitor] tracking error: {e}")
                            print(traceback.format_exc())

                    ret2, jpeg = cv2.imencode('.jpg', frame)
                    if not ret2:
                        continue

                    chunk = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')
            finally:
                if cap is not None:
                    cap.release()

        return generate()

    def get_report(self) -> list:
        """
        Получает отчет о посещаемости.

        Returns:
            list: Отчет.
        """
        return self.get_report_use_case.execute()

    def get_groups(self) -> Dict[str, list]:
        """
        Получает группы с студентами.

        Returns:
            Dict[str, list]: Группы.
        """
        return self.student_service.get_groups_with_students()

    def update_manual_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обновляет статус вручную.

        Args:
            data (Dict[str, Any]): Данные.

        Returns:
            Dict[str, Any]: Результат.
        """
        # Placeholder for manual status update
        return {"status": "updated"}
