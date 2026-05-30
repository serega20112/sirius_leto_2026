from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from src.backend.infrastructure.database import SessionLocal, EngagementHistoryModel
from src.backend.domain.attendance.entity import EngagementStatus


class EngagementRepository:
    """Репозиторий для работы с историей вовлеченности."""

    def get_engagement_history(
        self,
        student_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Получить историю вовлеченности ученика за период.

        Args:
            student_id: ID ученика
            start_date: Начало периода
            end_date: Конец периода

        Returns:
            Список записей о вовлеченности
        """
        db = SessionLocal()
        try:
            query = db.query(EngagementHistoryModel).filter(
                EngagementHistoryModel.student_id == student_id
            )

            if start_date:
                query = query.filter(EngagementHistoryModel.timestamp >= start_date)
            if end_date:
                query = query.filter(EngagementHistoryModel.timestamp <= end_date)

            records = query.order_by(EngagementHistoryModel.timestamp.desc()).all()

            return [
                {
                    "id": record.id,
                    "student_id": record.student_id,
                    "timestamp": record.timestamp.isoformat(),
                    "engagement_score": record.engagement_score,
                    "confidence": record.confidence,
                }
                for record in records
            ]
        finally:
            db.close()

    def add_engagement_record(
        self, student_id: str, engagement_score: str, confidence: int = 0
    ) -> bool:
        """
        Добавить запись о вовлеченности.

        Args:
            student_id: ID ученика
            engagement_score: Уровень вовлеченности
            confidence: Уверенность в оценке (0-100)

        Returns:
            True если запись добавлена успешно
        """
        db = SessionLocal()
        try:
            record = EngagementHistoryModel(
                student_id=student_id,
                engagement_score=engagement_score,
                confidence=confidence,
            )
            db.add(record)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Ошибка добавления записи о вовлеченности: {e}")
            return False
        finally:
            db.close()

    def get_latest_engagement(self, student_id: str) -> Optional[dict]:
        """
        Получить последнюю запись о вовлеченности ученика.

        Args:
            student_id: ID ученика

        Returns:
            Последняя запись о вовлеченности или None
        """
        db = SessionLocal()
        try:
            record = (
                db.query(EngagementHistoryModel)
                .filter(EngagementHistoryModel.student_id == student_id)
                .order_by(EngagementHistoryModel.timestamp.desc())
                .first()
            )

            if record:
                return {
                    "id": record.id,
                    "student_id": record.student_id,
                    "timestamp": record.timestamp.isoformat(),
                    "engagement_score": record.engagement_score,
                    "confidence": record.confidence,
                }
            return None
        finally:
            db.close()

    def get_engagement_stats(self, student_id: str, days: int = 30) -> dict:
        """
        Получить статистику вовлеченности за последние N дней.

        Args:
            student_id: ID ученика
            days: Количество дней для анализа

        Returns:
            Статистика вовлеченности
        """
        db = SessionLocal()
        try:
            start_date = datetime.now() - timedelta(days=days)

            records = (
                db.query(EngagementHistoryModel)
                .filter(
                    EngagementHistoryModel.student_id == student_id,
                    EngagementHistoryModel.timestamp >= start_date,
                )
                .all()
            )

            stats = {
                "high": 0,
                "medium": 0,
                "low": 0,
                "unknown": 0,
                "total": len(records),
                "avg_confidence": 0,
            }

            if records:
                confidence_sum = 0
                for record in records:
                    stats[record.engagement_score] += 1
                    confidence_sum += record.confidence

                stats["avg_confidence"] = round(confidence_sum / len(records), 1)

            return stats
        finally:
            db.close()
