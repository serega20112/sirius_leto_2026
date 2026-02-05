from pathlib import Path
from flask import Flask

from io import BytesIO
import cv2
from flask import send_file, abort
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
    app = Flask(
        __name__,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir / "static"),
        static_url_path="/static",
    )

    CORS(app)

    init_db()

    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(monitor_bp)

    @app.route("/src/assets/images/<path:filename>")
    def serve_student_photos(filename):
        """Отдаёт фото студента, обрезанное и масштабированное до 40x40 через OpenCV"""
        img_path = images_dir / filename
        if not img_path.exists():
            abort(404)

        # читаем изображение
        img = cv2.imread(str(img_path))
        if img is None:
            abort(500)

        # центрируем и обрезаем в квадрат
        h, w = img.shape[:2]
        min_side = min(h, w)
        start_x = (w - min_side) // 2
        start_y = (h - min_side) // 2
        img_cropped = img[start_y:start_y + min_side, start_x:start_x + min_side]

        # масштабируем до 40x40
        img_resized = cv2.resize(img_cropped, (40, 40), interpolation=cv2.INTER_AREA)

        # кодируем в JPEG
        success, buffer = cv2.imencode(".jpg", img_resized)
        if not success:
            abort(500)

        return send_file(
            BytesIO(buffer.tobytes()),
            mimetype="image/jpeg",
            download_name=filename
        )

    return app
