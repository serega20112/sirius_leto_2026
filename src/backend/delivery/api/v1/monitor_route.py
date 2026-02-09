from flask import Blueprint, Response, jsonify, request, render_template
from src.backend.dependencies import container

monitor_bp = Blueprint("monitor", __name__)


@monitor_bp.route("/video_feed")
def video_feed():
    return Response(
        container.monitor_service.stream_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@monitor_bp.route("/logs")
def get_logs():
    return jsonify(container.monitor_service.get_report())

@monitor_bp.route("/groups")
def get_groups():
    return render_template(
        "groups.html",
        groups=container.monitor_service.get_groups()
    )

@monitor_bp.route("/manual_status", methods=["POST"])
def manual_status():
    return jsonify(container.monitor_service.update_manual_status(request.json))
