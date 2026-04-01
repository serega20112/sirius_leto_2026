from src.backend.application.exceptions import ValidationError


class GetStudentAttendanceUseCase:
    """Query-use-case детальной статистики посещаемости ученика."""

    def __init__(self, attendance_repo, student_repo):
        self.attendance_repo = attendance_repo
        self.student_repo = student_repo

    def execute(self, student_id: str) -> dict:
        """
        Executes the main scenario for GetStudentAttendanceUseCase.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The scenario execution result.
        """
        if not student_id:
            raise ValidationError("Нужен student_id")

        student = self.student_repo.find_by_id(student_id)
        if student is None:
            raise ValidationError("Ученик не найден")

        all_students = self.student_repo.get_all()
        group_student_ids = {
            group_student.id
            for group_student in all_students
            if group_student.group_name == student.group_name
        }

        all_logs = self.attendance_repo.get_all_logs()
        group_logs = [log for log in all_logs if log.student_id in group_student_ids]
        lesson_dates = sorted(
            {log.timestamp.date() for log in group_logs},
            reverse=True,
        )

        student_daily_logs = {}
        student_logs = sorted(
            (log for log in group_logs if log.student_id == student_id),
            key=lambda log: log.timestamp,
        )
        for log in student_logs:
            student_daily_logs.setdefault(log.timestamp.date(), log)

        late_arrivals = []
        absences = []
        history = []

        for lesson_date in lesson_dates:
            arrival_log = student_daily_logs.get(lesson_date)
            if arrival_log is None:
                absences.append({"date": lesson_date.isoformat()})
                history.append(
                    {
                        "date": lesson_date.isoformat(),
                        "status": "absent",
                        "arrived_at": None,
                    }
                )
                continue

            arrived_at = arrival_log.timestamp.strftime("%H:%M")
            status = "late" if arrival_log.is_late else "present"
            history.append(
                {
                    "date": lesson_date.isoformat(),
                    "status": status,
                    "arrived_at": arrived_at,
                }
            )
            if arrival_log.is_late:
                late_arrivals.append(
                    {
                        "date": lesson_date.isoformat(),
                        "arrived_at": arrived_at,
                    }
                )

        attended_days = len(student_daily_logs)
        late_days = len(late_arrivals)
        on_time_days = attended_days - late_days
        lesson_days = len(lesson_dates)
        absent_days = len(absences)
        attendance_rate = round((attended_days / lesson_days) * 100) if lesson_days else 0

        return {
            "student": {
                "id": student.id,
                "name": student.name,
                "group": student.group_name,
            },
            "summary": {
                "lesson_days": lesson_days,
                "attended_days": attended_days,
                "on_time_days": on_time_days,
                "late_days": late_days,
                "absent_days": absent_days,
                "attendance_rate": attendance_rate,
            },
            "late_arrivals": late_arrivals,
            "absences": absences,
            "history": history,
        }
