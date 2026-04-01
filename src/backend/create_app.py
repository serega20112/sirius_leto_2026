from pathlib import Path

from flask import Flask
from flask_cors import CORS

from src.backend.dependencies.container import Container
from src.backend.delivery.error_handlers import register_error_handlers
from src.backend.infrastructure.database import init_db
from src.backend.delivery.api.v1.auth_route import create_auth_blueprint
from src.backend.delivery.api.v1.index_route import (
    web_bp,
)
from src.backend.delivery.api.v1.media_route import create_media_blueprint
from src.backend.delivery.api.v1.monitor_route import create_monitor_blueprint


def create_app():
    """
    Build the Flask application and wire the delivery layer to the DI container.

    Args:
        None.

    Returns:
        A fully configured Flask application instance.
    """
    backend_dir = Path(__file__).resolve().parent
    frontend_dir = backend_dir.parent / "frontend"
    app = Flask(
        __name__,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir / "static"),
        static_url_path="/static",
    )

    CORS(app)

    init_db()
    container = Container()
    app.extensions["container"] = container

    register_error_handlers(app)
    app.register_blueprint(web_bp)
    app.register_blueprint(create_auth_blueprint(container.student_service))
    app.register_blueprint(
        create_monitor_blueprint(
            container.attendance_service,
            container.student_service,
        )
    )
    app.register_blueprint(create_media_blueprint(container.media_service))

    return app
