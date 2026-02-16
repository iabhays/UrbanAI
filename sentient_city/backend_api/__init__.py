"""Backend API for SENTIENTCITY AI."""

from .fastapi_server import app, WebSocketManager

__all__ = ["app", "WebSocketManager"]
