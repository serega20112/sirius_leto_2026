from src.backend.domain.student.entity import Student
from src.backend.domain.student.repository import StudentRepository
from src.backend.infrastructure.database import StudentModel
from src.backend.infrastructure.persistence.sqlite.base_repository import SqliteRepository


class SqliteStudentRepository(SqliteRepository, StudentRepository):
    def save(self, student: Student) -> None:
        """
        Runs the operation save.
        
        Args:
            student: Input value for `student`.
        
        Returns:
            Does not return a value.
        """
        model = StudentModel.from_domain(student)
        self.session.merge(model)
        self.session.commit()

    def find_by_id(self, student_id: str) -> Student | None:
        """
        Finds by id.
        
        Args:
            student_id: Input value for `student_id`.
        
        Returns:
            The requested data or prepared result.
        """
        model = self.session.query(StudentModel).filter_by(id=student_id).first()
        return model.to_domain() if model else None

    def find_by_name(self, name: str) -> Student | None:
        """
        Finds by name.
        
        Args:
            name: Input value for `name`.
        
        Returns:
            The requested data or prepared result.
        """
        model = self.session.query(StudentModel).filter_by(name=name).first()
        return model.to_domain() if model else None

    def get_all(self) -> list[Student]:
        """
        Gets all.
        
        Args:
            None.
        
        Returns:
            The requested data or prepared result.
        """
        return [model.to_domain() for model in self.session.query(StudentModel).all()]
