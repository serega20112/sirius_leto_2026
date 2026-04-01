from abc import ABC, abstractmethod

from src.backend.domain.student.entity import Student


class StudentRepository(ABC):
    @abstractmethod
    def save(self, student: Student) -> None:
        """
        Runs the operation save.
        
        Args:
            student: Input value for `student`.
        
        Returns:
            Does not return a value.
        """
        ...

    @abstractmethod
    def find_by_id(self, student_id: str) -> Student | None:
        """
        Finds by id.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        ...

    @abstractmethod
    def find_by_name(self, name: str) -> Student | None:
        """
        Finds by name.
        
        Args:
            name: Input value for `name`.
        
        Returns:
            The requested data or prepared result.
        """
        ...

    @abstractmethod
    def get_all(self) -> list[Student]:
        """
        Gets all.
        
        Args:
            None.
        
        Returns:
            The requested data or prepared result.
        """
        ...
