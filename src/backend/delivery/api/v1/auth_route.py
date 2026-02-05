from flask import Blueprint, request, jsonify, render_template
from src.backend.dependencies.container import container

auth_bp = Blueprint(
    "auth",
    __name__,
)


@auth_bp.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@auth_bp.route("/register", methods=["POST"])
def register_api():
    """Обработка формы регистрации"""
    file = request.files.get("photo") or next(iter(request.files.values()), None)
    name = request.form.get("name")
    group = request.form.get("group")

    if not name or not group or not file:
        return jsonify({"error": "Missing name, group or photo"}), 400

    try:
        photo_bytes = file.read()
        result_student = container.register_use_case.execute(name, group, photo_bytes)
        return (
            jsonify(
                {"message": "Student registered successfully", "id": result_student.id}
            ),
            201,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
