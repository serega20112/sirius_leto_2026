from .auth_route import create_auth_blueprint
from .media_route import create_media_blueprint
from .monitor_route import create_monitor_blueprint

__all__ = [
    "create_auth_blueprint",
    "create_media_blueprint",
    "create_monitor_blueprint",
]
