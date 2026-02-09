from flask import Blueprint, Response, jsonify, request, render_template
from src.backend.dependencies.container import container

monitor_bp = Blueprint("monitor", __name__)


@monitor_bp.route("/video_feed")
def video_feed():
    """Stream live video feed."""
    return Response(container.video_stream_service.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@monitor_bp.route("/logs", methods=["GET"])
def get_logs():
    """Return attendance report in JSON."""
    return jsonify(container.report_service.get_report())


@monitor_bp.route("/groups", methods=["GET"])
def get_groups():
    """Render student groups in HTML."""
    groups_data = container.group_service.get_all_groups()
    return render_template("groups.html", groups=groups_data)


@monitor_bp.route("/manual_status", methods=["POST"])
def manual_status():
    """Update attendance status for a student manually."""
    data = request.json
    container.attendance_service.update_status(data)
    return jsonify({"status": "updated"})
