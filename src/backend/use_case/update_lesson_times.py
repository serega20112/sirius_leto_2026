from datetime import time as dt_time

from src.backend.application.exceptions import ValidationError


class UpdateLessonTimesUseCase:
    """Use-case that encapsulates validation and runtime update of lesson times.

    It updates both the runtime container config and the settings module so that
    the change is visible across the running application.
    """

    def __init__(self, attendance_tracking_config, settings_module):
        self.attendance_tracking_config = attendance_tracking_config
        self.settings = settings_module

    def get_times(self) -> dict[str, str]:
        start = getattr(self.attendance_tracking_config, "lesson_start_time", None)
        end = getattr(self.attendance_tracking_config, "lesson_end_time", None)
        if start is None:
            start = self.settings.LESSON_START_TIME
        if end is None:
            end = self.settings.LESSON_END_TIME
        return {
            "lesson_start_time": start.strftime("%H:%M"),
            "lesson_end_time": end.strftime("%H:%M"),
        }

    def execute(self, start_str: str, end_str: str) -> dict[str, str]:
        if not start_str or not end_str:
            raise ValidationError("Both start and end times must be provided")
        try:
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
            start_time_obj = dt_time(start_h, start_m)
            end_time_obj = dt_time(end_h, end_m)
        except Exception:
            raise ValidationError("Invalid time format, expected HH:MM")

        # Update module-level settings (useful for other modules that import settings)
        self.settings.LESSON_START_TIME = start_time_obj
        self.settings.LESSON_END_TIME = end_time_obj

        # Update the running container config so streaming logic uses new times
        if self.attendance_tracking_config is not None:
            setattr(self.attendance_tracking_config, "lesson_start_time", start_time_obj)
            setattr(self.attendance_tracking_config, "lesson_end_time", end_time_obj)

        return {
            "status": "updated",
            "lesson_start_time": start_time_obj.strftime("%H:%M"),
            "lesson_end_time": end_time_obj.strftime("%H:%M"),
        }
