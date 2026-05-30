from datetime import datetime
from typing import Dict, List, Optional
from src.backend.application.exceptions import ValidationError
from src.backend.infrastructure.persistence.engagement_repository import (
    EngagementRepository,
)


class GetEngagementHistoryUseCase:
    """Use case для получения истории вовлеченности ученика."""

    def __init__(self, engagement_repository: EngagementRepository):
        self.engagement_repository = engagement_repository

    def execute(self, student_id: str, days: int = 30) -> Dict:
        """
        Получить историю вовлеченности ученика.

        Args:
            student_id: ID ученика
            days: Количество дней для анализа

        Returns:
            Словарь с историей и статистикой вовлеченности
        """
        if not student_id:
            raise ValidationError("Нужен student_id")

        # Получаем статистику
        stats = self.engagement_repository.get_engagement_stats(student_id, days)

        # Получаем детальную историю
        start_date = datetime.now() - timedelta(days=days)
        history = self.engagement_repository.get_engagement_history(
            student_id, start_date=start_date
        )

        # Получаем последнюю запись
        latest = self.engagement_repository.get_latest_engagement(student_id)

        # Рассчитываем процентное соотношение
        total = stats["total"]
        if total > 0:
            stats["high_percent"] = round((stats["high"] / total) * 100, 1)
            stats["medium_percent"] = round((stats["medium"] / total) * 100, 1)
            stats["low_percent"] = round((stats["low"] / total) * 100, 1)
            stats["unknown_percent"] = round((stats["unknown"] / total) * 100, 1)
        else:
            stats["high_percent"] = 0
            stats["medium_percent"] = 0
            stats["low_percent"] = 0
            stats["unknown_percent"] = 0

        return {
            "student_id": student_id,
            "stats": stats,
            "history": history,
            "latest": latest,
            "period_days": days,
        }
