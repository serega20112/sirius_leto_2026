from flask import Blueprint, render_template

# Создаем блюпринт без префикса /api, так как это главная страница
web_bp = Blueprint("web", __name__)


@web_bp.route("/")
def index():
    """Отдает главную страницу фронтенда."""
    return render_template("templates/index.html")
