from pathlib import Path
from flask import Flask, send_from_directory

from flask_cors import CORS
from src.backend.infrastructure.database import init_db
from src.backend.delivery.api.v1.auth_route import auth_bp
from src.backend.delivery.api.v1.monitor_route import monitor_bp
from src.backend.delivery.api.v1.index_route import (
    web_bp,
)


def create_app():
    backend_dir = Path(__file__).resolve().parent
    frontend_dir = backend_dir.parent / "frontend"
    images_dir = backend_dir / "assets" / "images"

    # templates находятся в frontend/templates, а статические ресурсы — в frontend
    # Явно указываем static_url_path='/static', чтобы файлы отдавались по /static/...
    app = Flask(
        __name__,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir),
        static_url_path="/static",
    )

    CORS(app)

    init_db()

    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(monitor_bp)

    @app.route("/src/assets/images/<path:filename>")
    def serve_student_photos(filename):
        return send_from_directory(str(images_dir), filename)

    return app
