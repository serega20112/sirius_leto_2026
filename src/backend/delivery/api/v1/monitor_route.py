import cv2
from flask import Blueprint, Response, jsonify, request, render_template
from src.backend.dependencies.container import container
from src.backend.utils.cv_tools import draw_overlays
from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from datetime import datetime

monitor_bp = Blueprint("monitor", __name__)


def generate_frames():
    """Генератор видеопотока с обработкой ошибок."""
    cap = cv2.VideoCapture(0)  # убрал CAP_DSHOW для кроссплатформенности

    if not cap.isOpened():
        print("[ERROR] Камера не открыта!")
        return  # генератор завершён

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue  # пропускаем итерацию, камера ещё не готова

            frame = cv2.resize(frame, (640, 480))

            try:
                result = container.track_use_case.execute(frame)
                frame = draw_overlays(frame, result)
            except Exception as e:
                print(f"[ERROR] Ошибка трекинга: {e}")

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
        groups.setdefault(s.group_name, []).append(
            {"id": s.id, "name": s.name, "photo": f"/src/assets/images/{s.id}.jpg"}
        )

    for g in groups:
        groups[g].sort(key=lambda x: x["name"])

    return render_template("groups.html", groups=groups)


@monitor_bp.route("/manual_status", methods=["POST"])
def manual_status():
    data = request.json
    student_id = data.get("student_id")
    action = data.get("action")

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
