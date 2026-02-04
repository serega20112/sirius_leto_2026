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
    # Диагностика: логируем вызов эндпоинта и источник запроса
    print(f"[MONITOR] GET /api/v1/monitor/logs called from {request.remote_addr} method={request.method}")
    try:
        report = container.report_use_case.execute()
        if report is None:
            print("[MONITOR] report_use_case.execute() вернул None, возвращаю пустой список")
            return jsonify([])

        # Если report содержит объекты, не сериализуемые напрямую, попробуем привести к простому виду
        try:
            return jsonify(report)
        except Exception as ser_exc:
            print(f"[MONITOR] Ошибка сериализации ответа: {ser_exc}. Попытка привести в список словарей.")
            # Попытка трансформировать итерацию объектов в словари (без предположений об атрибутах)
            try:
                safe = []
                for item in report:
                    if isinstance(item, dict):
                        safe.append(item)
                    else:
                        # берём атрибуты доступные через __dict__ как fallback
                        attrs = getattr(item, "__dict__", None)
                        if attrs:
                            safe.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in attrs.items()})
                        else:
                            safe.append(str(item))
                return jsonify(safe)
            except Exception as convert_exc:
                print(f"[MONITOR] Не удалось привести отчет в JSON: {convert_exc}")
                return jsonify({"error": "Failed to serialize report", "debug": str(convert_exc)}), 500
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[MONITOR] Ошибка при получении логов: {e}")
        return jsonify({"error": str(e)}), 500


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
    # добавлен лог для диагностики входящих данных
    try:
        data = request.get_json(force=False, silent=True) or request.json or {}
    except Exception:
        data = request.json or {}

    print(f"[MONITOR] POST /api/v1/monitor/manual_status payload from {request.remote_addr}: {data}")

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
    print(f"[MONITOR] Добавлен лог вручную для student_id={student_id} action={action}")
    return jsonify({"status": "updated"})
