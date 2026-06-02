from typing import Any

from src.backend.application.exceptions import ValidationError


class AttendanceApplicationService:
    def __init__(
        self,
        video_streamer,
        get_report_use_case,
        get_student_attendance_use_case,
        update_lesson_times_use_case=None,
        manual_mark_use_case=None,
    ):
        self.video_streamer = video_streamer
        self.get_report_use_case = get_report_use_case
        self.get_student_attendance_use_case = get_student_attendance_use_case
        self.update_lesson_times_use_case = update_lesson_times_use_case
        self.manual_mark_use_case = manual_mark_use_case
        # Placeholder for a pre‑started video generator. The container will set this
        # attribute after initializing the background streamer so that the video
        # feed can be served continuously without re‑creating threads on each request.
        self._video_generator = None

    def stream_video(self):
        """Return the video generator for the MJPEG stream.

        If the container has already started the background streamer, the
        pre‑created generator stored in ``self._video_generator`` will be used.
        Otherwise we fall back to creating a new generator (e.g. during tests).
        """
        if self._video_generator is not None:
            return self._video_generator
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

    def update_lesson_times(self, start_str: str, end_str: str) -> dict[str, str]:
        """Delegate validation and runtime update of lesson times to the use-case.

        Args:
            start_str: Start time as HH:MM
            end_str: End time as HH:MM

        Returns:
            A dict with updated times and status.
        """
        if not self.update_lesson_times_use_case:
            raise NotImplementedError("UpdateLessonTimesUseCase not configured")
        return self.update_lesson_times_use_case.execute(start_str, end_str)

    def get_lesson_times(self) -> dict[str, str]:
        """Return current lesson start/end as strings (HH:MM).

        Delegates to the use-case when available.
        """
        if not self.update_lesson_times_use_case:
            raise NotImplementedError("UpdateLessonTimesUseCase not configured")
        return self.update_lesson_times_use_case.get_times()

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

        if self.manual_mark_use_case:
            return self.manual_mark_use_case.execute(student_id, status)

        return {
            "status": "accepted",
            "student_id": student_id,
            "presence_status": status,
            "payload": payload or {},
        }
