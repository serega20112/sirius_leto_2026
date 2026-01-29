from typing import List, Dict


class GetReportUseCase:
    """Сценарий получения журнала посещаемости для фронтенда."""

    def __init__(self, attendance_repo, student_repo):
        self.attendance_repo = attendance_repo
        self.student_repo = student_repo

    def execute(self) -> List[Dict]:
        """
        Возвращает список логов, обогащенный именами студентов.
        """
        logs = self.attendance_repo.get_all_logs()
        students_cache = {s.id: s for s in self.student_repo.get_all()}

        report = []
        for log in logs:
            student = students_cache.get(log.student_id)
            student_name = student.name if student else "Unknown"

            report.append({
                "id": log.id,
                "student_name": student_name,
                "timestamp": log.timestamp.isoformat(),
                "is_late": log.is_late,
                "engagement": log.engagement_score.value
            })

        return report