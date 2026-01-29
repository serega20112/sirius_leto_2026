from dataclasses import dataclass
from datetime import datetime


@dataclass
class Student:
    """Сущность студента в бизнес-логике."""
    id: str
    name: str
    group_name: str
    photo_path: str
    created_at: datetime