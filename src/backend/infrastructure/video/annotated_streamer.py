import time
import traceback
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image

from src.backend.dependencies import settings
from src.backend.utils.cv_tools import draw_overlays


class AnnotatedVideoStreamer:
    """Video streamer with background capture and lightweight inference scheduling.

    The streamer keeps capture and inference in background threads so the web
    response loop can continue delivering frames even when inference is slower.
    """

    def __init__(self, track_attendance_use_case):
        self.track_attendance_use_case = track_attendance_use_case

        self._latest_frame = None
        self._annotated_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._inference_thread = None
        self._engagement_thread = None

    def stream(self):
        source = getattr(settings, "CAMERA_SOURCE", 0)

        def generate():
            cap = None
            try:
                # start capture in background thread
                cap = cv2.VideoCapture(
                    int(source)
                    if isinstance(source, (str, int)) and str(source).isdigit()
                    else source
                )

                if not cap.isOpened():
                    yield from self._fallback_image_generator()
                    return

                # apply requested resolution when possible
                try:
                    w = int(getattr(settings, "CAMERA_WIDTH", 0) or 0)
                    h = int(getattr(settings, "CAMERA_HEIGHT", 0) or 0)
                    if w > 0 and h > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                except Exception:
                    pass

                # shared capture loop
                def _capture_loop():
                    retry = 0
                    while not self._stop_event.is_set():
                        try:
                            ret, frame = cap.read()
                        except Exception:
                            ret, frame = False, None

                        if not ret or frame is None:
                            retry += 1
                            if retry > 10:
                                print(
                                    "[Video] camera read failed repeatedly, switching to fallback image"
                                )
                                self._stop_event.set()
                                return
                            time.sleep(0.05)
                            continue

                        retry = 0
                        # resize to configured resolution to reduce CPU usage
                        try:
                            w = int(getattr(settings, "CAMERA_WIDTH", 0) or 0)
                            h = int(getattr(settings, "CAMERA_HEIGHT", 0) or 0)
                            if (
                                w > 0
                                and h > 0
                                and (frame.shape[1], frame.shape[0]) != (w, h)
                            ):
                                frame = cv2.resize(
                                    frame, (w, h), interpolation=cv2.INTER_AREA
                                )
                        except Exception:
                            pass

                        with self._lock:
                            self._latest_frame = frame.copy()

                # inference loop runs at configured INFERENCE_RATE
                def _inference_loop():
                    inference_rate = float(
                        getattr(settings, "INFERENCE_RATE", 7.0) or 7.0
                    )
                    interval = 1.0 / max(0.0001, inference_rate)
                    while not self._stop_event.is_set():
                        frame = None
                        with self._lock:
                            if self._latest_frame is not None:
                                frame = self._latest_frame.copy()
                        if frame is None:
                            time.sleep(0.01)
                            continue

                        if self.track_attendance_use_case:
                            try:
                                tracking_result = (
                                    self.track_attendance_use_case.execute(frame)
                                )
                                if (
                                    isinstance(tracking_result, dict)
                                    and "students" in tracking_result
                                ):
                                    annotated = draw_overlays(
                                        frame.copy(), tracking_result
                                    )
                                else:
                                    annotated = frame.copy()
                            except Exception as error:
                                print(f"[Video] tracking error: {error}")
                                annotated = frame.copy()
                        else:
                            annotated = frame.copy()

                        with self._lock:
                            self._annotated_frame = annotated
                        time.sleep(interval)

                # start background threads
                self._stop_event.clear()
                self._capture_thread = threading.Thread(
                    target=_capture_loop, daemon=True
                )
                self._capture_thread.start()
                self._inference_thread = threading.Thread(
                    target=_inference_loop, daemon=True
                )
                self._inference_thread.start()

                # engagement logger: periodically persist latest engagement to DB
                def _engagement_loop():
                    interval = float(getattr(settings, "ENGAGEMENT_LOG_INTERVAL", 60.0) or 60.0)
                    from time import sleep
                    while not self._stop_event.is_set():
                        try:
                            # Sleep first so we log only after the first interval
                            sleep(interval)
                            latest = getattr(self.track_attendance_use_case, "latest_engagements", None)
                            if not latest:
                                continue
                            try:
                                from src.backend.infrastructure.persistence.engagement_repository import EngagementRepository

                                repo = EngagementRepository()
                                for sid, info in list(latest.items()):
                                    try:
                                        if not sid or sid == "Unknown":
                                            continue
                                        score = info.get("score", "unknown")
                                        confidence = int(info.get("confidence", 0) or 0)
                                        repo.add_engagement_record(sid, score, confidence)
                                    except Exception as e:
                                        print(f"[Engage] record write error: {e}")
                            except Exception as e:
                                print(f"[Engage] repo init error: {e}")
                        except Exception as e:
                            print(f"[Engage] logger error: {e}")

                self._engagement_thread = threading.Thread(target=_engagement_loop, daemon=True)
                self._engagement_thread.start()

                stream_fps = float(getattr(settings, "STREAM_FPS", 20.0) or 20.0)
                stream_interval = 1.0 / max(0.0001, stream_fps)

                # main loop yields the latest annotated frame when available, otherwise latest raw frame
                while not self._stop_event.is_set():
                    out_frame = None
                    with self._lock:
                        if self._annotated_frame is not None:
                            out_frame = self._annotated_frame.copy()
                        elif self._latest_frame is not None:
                            out_frame = self._latest_frame.copy()

                    if out_frame is None:
                        time.sleep(0.01)
                        continue

                    yield self._encode_frame(out_frame)
                    time.sleep(stream_interval)

            finally:
                try:
                    self._stop_event.set()
                    if self._capture_thread is not None:
                        self._capture_thread.join(timeout=0.5)
                    if self._inference_thread is not None:
                        self._inference_thread.join(timeout=0.5)
                    if self._engagement_thread is not None:
                        self._engagement_thread.join(timeout=0.5)
                except Exception:
                    pass
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
