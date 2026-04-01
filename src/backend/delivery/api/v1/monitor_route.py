from flask import Blueprint, Response, jsonify, render_template, request


def create_monitor_blueprint(attendance_service, student_service):
    """
    Create the monitoring blueprint for video, logs, and attendance analytics.

    Args:
        attendance_service: Application service for attendance and monitoring.
        student_service: Application service for student directory operations.

    Returns:
        A configured Flask blueprint.
    """
    monitor_bp = Blueprint("monitor", __name__)

    @monitor_bp.route("/video_feed")
    def video_feed():
        """
        Return the live annotated video stream.

        Args:
            None.

        Returns:
            A streaming HTTP response with MJPEG frames.
        """
        return Response(
            attendance_service.stream_video(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @monitor_bp.route("/logs")
    def get_logs():
        """
        Return the attendance journal for the monitoring dashboard.

        Args:
            None.

        Returns:
            A JSON response with serialized attendance entries.
        """
        return jsonify(attendance_service.get_report())

    @monitor_bp.route("/groups")
    def get_groups():
        """
        Render the groups page with all grouped students.

        Args:
            None.

        Returns:
            The rendered groups template.
        """
        return render_template(
            "groups.html",
            groups=student_service.get_groups(),
            show_video=False,
        )

    @monitor_bp.route("/students/<student_id>/attendance")
    def get_student_attendance(student_id):
        """
        Return the attendance summary for one student.

        Args:
            student_id: Student identifier from the route path.

        Returns:
            A JSON response with the student's attendance statistics.
        """
        return jsonify(attendance_service.get_student_attendance(student_id))

    @monitor_bp.route("/manual_status", methods=["POST"])
    def manual_status():
        """
        Accept a manual attendance status update request.

        Args:
            None.

        Returns:
            A JSON response confirming request validation.
        """
        payload = request.get_json(silent=True) or {}
        return jsonify(
            attendance_service.update_manual_status(
                student_id=payload.get("student_id"),
                status=payload.get("status"),
                payload=payload,
            )
        )

    return monitor_bp
