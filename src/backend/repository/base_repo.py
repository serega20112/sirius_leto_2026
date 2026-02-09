from abc import ABC

class BaseRepository(ABC):
    """Абстрактный базовый репозиторий."""

    def __init__(self, session):
        """Инициализация с сессией SQLAlchemy."""
        self.session = session
