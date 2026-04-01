from typing import List, Dict


class GetReportUseCase:
    """Сценарий получения журнала посещаемости для фронтенда."""

    def __init__(self, attendance_repo, student_repo):
        self.attendance_repo = attendance_repo
        self.student_repo = student_repo

    def execute(self) -> List[Dict]:
        """
        Executes the main scenario for GetReportUseCase.
        
        Args:
            None.
        
        Returns:
            The scenario execution result.
        """
        logs = self.attendance_repo.get_all_logs()
        students_cache = {s.id: s for s in self.student_repo.get_all()}
        latest_logs_by_student = {}
        for log in logs:
            cached_log = latest_logs_by_student.get(log.student_id)
            if cached_log is None or log.timestamp > cached_log.timestamp:
                latest_logs_by_student[log.student_id] = log

        report = []
        for log in sorted(
            latest_logs_by_student.values(),
            key=lambda item: item.timestamp,
            reverse=True,
        ):
            student = students_cache.get(log.student_id)
            student_name = student.name if student else "Unknown"

            report.append(
                {
                    "id": log.id,
                    "student_name": student_name,
                    "timestamp": log.timestamp.isoformat(),
                    "is_late": log.is_late,
                    "status": "late" if log.is_late else "present",
                    "arrived_at": log.timestamp.strftime("%H:%M"),
                    "engagement": log.engagement_score.value,
                }
            )

        return report
