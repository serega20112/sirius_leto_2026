from flask import Blueprint, request, jsonify
from src.backend.dependencies.container import container

auth_bp = Blueprint('auth', __name__, url_prefix='/api/v1/auth')


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Формат: multipart/form-data
    Fields: name, group
    File: photo
    """
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files['photo']
    name = request.form.get('name')
    group = request.form.get('group')

    if not name or not group:
        return jsonify({"error": "Missing name or group"}), 400

    try:
        photo_bytes = file.read()
        result_student = container.register_use_case.execute(name, group, photo_bytes)

        return jsonify({
            "message": "Student registered successfully",
            "id": result_student.id
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500