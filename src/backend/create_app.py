from flask import Flask
from flask_cors import CORS
from src.backend.infrastructure.database import init_db
from src.backend.delivery.api.v1 import auth_bp
from src.backend.delivery.api.v1 import monitor_bp

def create_app():
    """Фабрика приложения."""
    app = Flask(__name__)
    CORS(app)
    init_db()
    app.register_blueprint(auth_bp)
    app.register_blueprint(monitor_bp)

    return app