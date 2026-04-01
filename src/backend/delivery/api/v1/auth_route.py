from flask import Blueprint, jsonify, render_template, request


def create_auth_blueprint(student_service):
    """
    Create the blueprint responsible for student registration routes.

    Args:
        student_service: Application service that handles student workflows.

    Returns:
        A configured Flask blueprint.
    """
    auth_bp = Blueprint("auth", __name__)

    @auth_bp.route("/register", methods=["GET"])
    def register_page():
        """
        Render the student registration page.

        Args:
            None.

        Returns:
            The rendered registration template.
        """
        return render_template("register.html", show_video=False)

    @auth_bp.route("/register", methods=["POST"])
    def register_api():
        """
        Accept the registration form and return the created student payload.

        Args:
            None.

        Returns:
            A JSON response with the created student and status 201.
        """
        files = request.files.getlist("photos")
        name = request.form.get("name", "")
        group_name = request.form.get("group", "")
        photos_bytes = [file.read() for file in files]

        student_data = student_service.register_student(
            name=name,
            group_name=group_name,
            photos_bytes=photos_bytes,
        )
        return jsonify(student_data), 201

    return auth_bp
