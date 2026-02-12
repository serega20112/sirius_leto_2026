from abc import ABC, abstractmethod
from typing import List

from src.backend.domain.attendance.entity import AttendanceLog


class AbstractAttendanceRepository(ABC):
    @abstractmethod
    def add_log(self, log: AttendanceLog) -> AttendanceLog:
        """
        Добавляет новую запись в журнал и возвращает её с присвоенным ID.

        Args:
            log (AttendanceLog): Запись журнала посещаемости.

        Returns:
            AttendanceLog: Запись с присвоенным ID.
        """

    @abstractmethod
    def get_logs_by_student(self, student_id: str) -> List[AttendanceLog]:
        """
        Возвращает историю посещений для конкретного студента.

        Args:
            student_id (str): ID студента.

        Returns:
            List[AttendanceLog]: Список записей посещаемости.
        """

    @abstractmethod
    def get_all_logs(self) -> List[AttendanceLog]:
        """
        Возвращает все записи журнала.

        Returns:
            List[AttendanceLog]: Все записи посещаемости.
        """

    @abstractmethod
    def get_stats_by_student(self, student_id: str) -> List[AttendanceLog]:
        """
        Получить все логи конкретного студента для графика.

        Args:
            student_id (str): ID студента.

        Returns:
            List[AttendanceLog]: Список записей для графика.
        """
