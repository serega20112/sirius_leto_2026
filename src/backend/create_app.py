import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from src.backend.infrastructure.database import init_db
from src.backend.delivery.api.v1.auth_route import auth_bp
from src.backend.delivery.api.v1.monitor_route import monitor_bp
from src.backend.delivery.api.v1.index_route import (
    web_bp,
)  # Твой новый файл для фронта


def create_app():
    backend_dir = Path(__file__).resolve().parent
    frontend_dir = backend_dir.parent / "frontend"

    app = Flask(
        __name__, template_folder=str(frontend_dir), static_folder=str(frontend_dir)
    )

    CORS(app)

    init_db()

    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(monitor_bp)

    return app
