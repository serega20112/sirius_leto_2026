import cv2
from flask import Blueprint, Response, jsonify, request
from src.backend.dependencies.container import container
from src.backend.utils.cv_tools import draw_overlays

monitor_bp = Blueprint("monitor", __name__, url_prefix="/api/v1/monitor")


def generate_frames():
    """Генератор видеопотока. БЕЗ cap.set для стабильности."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (640, 480))
            try:
                result = container.track_use_case.execute(frame)
                frame_with_overlay = draw_overlays(frame, result)
            except Exception as e:
                print(f"[ERROR] Ошибка в трекинге: {e}")
                frame_with_overlay = frame

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


@monitor_bp.route("/groups", methods=["GET"])
def get_groups():
    students = container.student_repo.get_all()
    groups = {}
    for s in students:
        if s.group_name not in groups:
            groups[s.group_name] = []
        groups[s.group_name].append(
            {"id": s.id, "name": s.name, "photo": f"/static/images/{s.id}.jpg"}
        )

    for g in groups:
        groups[g].sort(key=lambda x: x["name"])
    return jsonify(groups)


@monitor_bp.route("/manual_status", methods=["POST"])
def manual_status():
    data = request.json
    student_id = data.get("student_id")
    action = data.get("action")

    from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
    from datetime import datetime

    log = AttendanceLog(
        id=None,
        student_id=student_id,
        timestamp=datetime.now(),
        is_late=False,
        engagement_score=(
            EngagementStatus.HIGH if action == "present" else EngagementStatus.LOW
        ),
    )
    container.attendance_repo.add_log(log)
    return jsonify({"status": "updated"})
