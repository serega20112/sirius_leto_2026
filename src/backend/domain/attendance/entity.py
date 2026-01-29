from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EngagementStatus(Enum):
    """Перечисление возможных уровней вовлеченности."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class AttendanceLog:
    """Сущность записи о посещении."""
    id: int | None
    student_id: str
    timestamp: datetime
    is_late: bool
    engagement_score: EngagementStatus