from flask import Blueprint, request, jsonify, render_template
from pathlib import Path
from src.backend.dependencies.container import container

auth_bp = Blueprint("auth", __name__)

# Путь к папке с фото студентов
backend_dir = Path(__file__).resolve().parent.parent
images_dir = backend_dir / "assets" / "images"
images_dir.mkdir(parents=True, exist_ok=True)


@auth_bp.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@auth_bp.route("/register", methods=["POST"])
def register_api():
    """
    Обработка формы регистрации студента:
    - сохраняет студента через use_case
    - сохраняет фото в assets/images/<id>.jpg
    - возвращает JSON с id, именем, группой и именем фото
    """
    file = request.files.get("photo") or next(iter(request.files.values()), None)
    name = request.form.get("name")
    group = request.form.get("group")

    if not name or not group or not file:
        return jsonify({"error": "Missing name, group or photo"}), 400

    try:
        photo_bytes = file.read()
        result_student = container.register_use_case.execute(name, group, photo_bytes)

        # сохраняем фото
        photo_path = images_dir / f"{result_student.id}.jpg"
        with open(photo_path, "wb") as f:
            f.write(photo_bytes)

        return jsonify({
            "id": result_student.id,
            "name": result_student.name,
            "group": result_student.group_name,
            "photo": f"{result_student.id}.jpg"
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
