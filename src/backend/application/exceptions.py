class ApplicationError(Exception):
    """Базовая ошибка application-слоя."""


class ValidationError(ApplicationError):
    """Ошибка валидации входных данных."""
