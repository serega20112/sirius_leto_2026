from flask import Flask

from src.backend.application.exceptions import ValidationError


def register_error_handlers(app: Flask) -> None:
    """
    Register shared HTTP error handlers for the application.

    Args:
        app: Flask application that owns the HTTP handlers.

    Returns:
        Does not return a value.
    """

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """
        Convert a validation failure into an HTTP 400 response.

        Args:
            error: Validation exception raised by the application layer.

        Returns:
            A JSON response body with HTTP status 400.
        """
        return {"error": str(error)}, 400

    @app.errorhandler(NotImplementedError)
    def handle_not_implemented_error(error):
        """
        Convert an unsupported operation into an HTTP 501 response.

        Args:
            error: Exception describing the unsupported branch.

        Returns:
            A JSON response body with HTTP status 501.
        """
        return {"error": str(error)}, 501
