from abc import ABC, abstractmethod

from src.backend.domain.attendance.entity import AttendanceLog


class AttendanceRepository(ABC):
    @abstractmethod
    def add_log(self, log: AttendanceLog) -> AttendanceLog:
        """
        Adds log.
        
        Args:
            log: Input value for `log`.
        
        Returns:
            The result of the operation.
        """
        ...

    @abstractmethod
    def get_logs_by_student(self, student_id: str) -> list[AttendanceLog]:
        """
        Gets logs by student.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        ...

    @abstractmethod
    def get_all_logs(self) -> list[AttendanceLog]:
        """
        Gets all logs.
        
        Args:
            None.
        
        Returns:
            The requested data or prepared result.
        """
        ...

    @abstractmethod
    def get_stats_by_student(self, student_id: str) -> list[AttendanceLog]:
        """
        Gets stats by student.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        ...
