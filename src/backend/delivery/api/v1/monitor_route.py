import cv2
from flask import Blueprint, Response, jsonify
from src.backend.dependencies.container import container
from src.backend.utils.cv_tools import draw_overlays

monitor_bp = Blueprint("monitor", __name__, url_prefix="/api/v1/monitor")


def generate_frames():
    """Генератор видеопотока с оптимизированным разрешением."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            result = container.track_use_case.execute(frame)
            frame_with_overlay = draw_overlays(frame, result)
            ret, buffer = cv2.imencode(
                ".jpg", frame_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@monitor_bp.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@monitor_bp.route("/logs", methods=["GET"])
def get_logs():
    report = container.report_use_case.execute()
    return jsonify(report)
