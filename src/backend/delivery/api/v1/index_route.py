from flask import Blueprint, render_template

web_bp = Blueprint("web", __name__)


@web_bp.route("/")
def index():
    """
    Render the main monitoring page.

    Args:
        None.

    Returns:
        The rendered index template.
    """
    return render_template("index.html", show_video=True)
