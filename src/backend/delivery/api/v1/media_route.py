from io import BytesIO

from flask import Blueprint, abort, send_file


def create_media_blueprint(media_service):
    """
    Create the blueprint that exposes transformed media assets.

    Args:
        media_service: Application service that prepares student media.

    Returns:
        A configured Flask blueprint.
    """
    media_bp = Blueprint("media", __name__)

    @media_bp.route("/src/assets/images/<path:filename>")
    def serve_student_photo(filename):
        """
        Return a generated thumbnail for a stored student photo.

        Args:
            filename: Requested image file name from the route path.

        Returns:
            An HTTP response with JPEG image bytes.
        """
        try:
            image_bytes = media_service.get_student_photo(filename)
        except FileNotFoundError:
            abort(404)
        except RuntimeError:
            abort(500)

        return send_file(
            BytesIO(image_bytes),
            mimetype="image/jpeg",
            download_name=filename,
        )

    return media_bp
