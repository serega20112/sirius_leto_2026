import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, DeclarativeBase, relationship, scoped_session

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BACKEND_DIR / "assets" / "database"
DB_PATH = DB_DIR / "attendance.db"

DB_DIR.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


class Base(DeclarativeBase):
    """Базовый класс для всех ORM моделей SQLAlchemy."""

    pass


class StudentModel(Base):
    """SQLAlchemy модель таблицы студентов."""

    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    group_name = Column(String)
    photo_path = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    logs = relationship("AttendanceModel", back_populates="student")


class AttendanceModel(Base):
    """SQLAlchemy модель таблицы журнала посещаемости."""

    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.id"), index=True)
    timestamp = Column(DateTime, default=datetime.now)
    is_late = Column(Boolean, default=False)
    engagement_score = Column(String, default="unknown")

    student = relationship("StudentModel", back_populates="logs")


def init_db():
    """Создает все таблицы в базе данных."""
    Base.metadata.create_all(bind=engine)
