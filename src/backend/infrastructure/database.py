import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship, scoped_session
from src.backend.domain.student.entity import Student

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BACKEND_DIR / "assets" / "database"
DB_PATH = DB_DIR / "attendance.db"
DB_DIR.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

class Base(DeclarativeBase):
    pass

class StudentModel(Base):
    __tablename__ = "students"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    group_name = Column(String)
    photo1_path = Column(String)
    photo2_path = Column(String)
    photo3_path = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    logs = relationship("AttendanceModel", back_populates="student", cascade="all, delete-orphan")

    def to_domain(self) -> Student:
        """
        Runs the operation to domain.
        
        Args:
            None.
        
        Returns:
            The function result.
        """
        return Student(
            id=self.id,
            name=self.name,
            group_name=self.group_name,
            photo_paths=[self.photo1_path, self.photo2_path, self.photo3_path],
            created_at=self.created_at
        )

    @classmethod
    def from_domain(cls, student: Student) -> "StudentModel":
        """
        Runs the operation from domain.
        
        Args:
            student: Input value for `student`.
        
        Returns:
            The function result.
        """
        return cls(
            id=student.id,
            name=student.name,
            group_name=student.group_name,
            photo1_path=student.photo_paths[0],
            photo2_path=student.photo_paths[1],
            photo3_path=student.photo_paths[2],
            created_at=student.created_at
        )

class AttendanceModel(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"), index=True)
    timestamp = Column(DateTime, default=datetime.now)
    is_late = Column(Boolean, default=False)
    engagement_score = Column(String, default="unknown")

    student = relationship("StudentModel", back_populates="logs")

def init_db():
    """
    Runs the operation init db.
    
    Args:
        None.
    
    Returns:
        The function result.
    """
    Base.metadata.create_all(bind=engine)
