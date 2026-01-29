import cv2
from flask import Blueprint, Response, jsonify
from src.backend.dependencies.container import container
from src.backend.utils.cv_tools import draw_overlays

monitor_bp = Blueprint('monitor', __name__, url_prefix='/api/v1/monitor')


def generate_frames():
    """Генератор видеопотока."""
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        result = container.track_use_case.execute(frame)
        frame_with_overlay = draw_overlays(frame, result)
        ret, buffer = cv2.imencode('.jpg', frame_with_overlay)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@monitor_bp.route('/video_feed')
def video_feed():
    """Эндпоинт для тега <img src="...">"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@monitor_bp.route('/logs', methods=['GET'])
def get_logs():
    """JSON API с журналом посещаемости."""
    report = container.report_use_case.execute()
    return jsonify(report)