from abc import ABC, abstractmethod
from typing import List, Dict

from src.backend.domain.student.entity import Student


class AbstractStudentRepository(ABC):
    @abstractmethod
    def save(self, student: Student) -> None:
        """
        Сохраняет или обновляет данные студента в БД.

        Args:
            student (Student): Сущность студента.
        """

    @abstractmethod
    def find_by_id(self, student_id: str) -> Student | None:
        """
        Находит студента по уникальному ID.

        Args:
            student_id (str): ID студента.

        Returns:
            Student | None: Сущность студента или None.
        """

    @abstractmethod
    def find_by_name(self, name: str) -> Student | None:
        """
        Находит студента по имени.

        Args:
            name (str): Имя студента.

        Returns:
            Student | None: Сущность студента или None.
        """

    @abstractmethod
    def get_all(self) -> List[Student]:
        """
        Возвращает список всех студентов.

        Returns:
            List[Student]: Список всех студентов.
        """

    @abstractmethod
    def get_groups_with_students(self) -> Dict[str, List[dict]]:
        """
        Возвращает студентов, сгруппированных по группам.

        Returns:
            Dict[str, List[dict]]: Словарь групп с списками студентов.
        """
