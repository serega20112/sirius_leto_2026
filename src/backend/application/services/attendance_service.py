from typing import Any

from src.backend.application.exceptions import ValidationError


class AttendanceApplicationService:
    def __init__(
        self,
        video_streamer,
        get_report_use_case,
        get_student_attendance_use_case,
    ):
        self.video_streamer = video_streamer
        self.get_report_use_case = get_report_use_case
        self.get_student_attendance_use_case = get_student_attendance_use_case

    def stream_video(self):
        """
        Start the annotated camera stream used by the monitoring page.

        Args:
            None.

        Returns:
            A generator that yields MJPEG chunks for the HTTP response.
        """
        return self.video_streamer.stream()

    def get_report(self) -> list[dict]:
        """
        Load the attendance journal in a frontend-friendly representation.

        Args:
            None.

        Returns:
            A list of serialized attendance entries.
        """
        return self.get_report_use_case.execute()

    def get_student_attendance(self, student_id: str) -> dict[str, Any]:
        """
        Load detailed attendance statistics for a single student.

        Args:
            student_id: Student identifier from the delivery layer.

        Returns:
            A serialized attendance summary with absences and late arrivals.
        """
        return self.get_student_attendance_use_case.execute(student_id)

    def update_manual_status(
        self,
        student_id: str | None,
        status: str | None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validate and acknowledge a manual attendance status update request.

        Args:
            student_id: Student identifier extracted from the request payload.
            status: Requested attendance status.
            payload: Raw payload received from the delivery layer.

        Returns:
            A normalized payload confirming that the request was accepted.
        """
        if not student_id or not status:
            raise ValidationError("Нужны поля student_id и status")

        return {
            "status": "accepted",
            "student_id": student_id,
            "presence_status": status,
            "payload": payload or {},
        }
