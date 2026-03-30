from flask import Blueprint, request, jsonify, render_template
from src.backend.dependencies.container import get_register_service
from pathlib import Path
auth_bp = Blueprint("auth", __name__)

service = get_register_service()


@auth_bp.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html", show_video=False)


@auth_bp.route("/register", methods=["POST"])
def register_api():
    try:
        student_data = service.register_student(request)  # dict
        return jsonify(student_data), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500