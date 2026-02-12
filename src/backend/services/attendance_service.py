from typing import List

from src.backend.domain.attendance.entity import AttendanceLog


class AttendanceService:
    def __init__(self, attendance_repo):
        self.attendance_repo = attendance_repo

    def add_log(self, log: AttendanceLog) -> AttendanceLog:
        """
        Добавляет лог посещаемости.

        Args:
            log (AttendanceLog): Лог.

        Returns:
            AttendanceLog: Добавленный лог.
        """
        return self.attendance_repo.add_log(log)

    def get_logs_by_student(self, student_id: str) -> List[AttendanceLog]:
        """
        Получает логи по студенту.

        Args:
            student_id (str): ID студента.

        Returns:
            List[AttendanceLog]: Логи.
        """
        return self.attendance_repo.get_logs_by_student(student_id)

    def get_all_logs(self) -> List[AttendanceLog]:
        """
        Получает все логи.

        Returns:
            List[AttendanceLog]: Все логи.
        """
        return self.attendance_repo.get_all_logs()

    def get_stats_by_student(self, student_id: str) -> List[AttendanceLog]:
        """
        Получает статистику по студенту.

        Args:
            student_id (str): ID студента.

        Returns:
            List[AttendanceLog]: Статистика.
        """
        return self.attendance_repo.get_stats_by_student(student_id)
