from flask import Blueprint, Response, jsonify, request, render_template
from src.backend.dependencies.container import get_monitor_service

monitor_bp = Blueprint("monitor", __name__)

service = get_monitor_service()


@monitor_bp.route("/video_feed")
def video_feed():
    return Response(
        service.stream_video(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@monitor_bp.route("/logs")
def get_logs():
    return jsonify(service.get_report())


@monitor_bp.route("/groups")
def get_groups():
    return render_template(
        "groups.html",
        groups=service.get_groups(),
        show_video=False,
    )


@monitor_bp.route("/manual_status", methods=["POST"])
def manual_status():
    return jsonify(service.update_manual_status(request.json))
