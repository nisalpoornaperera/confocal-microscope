"""HTTP and WebSocket transport (FastAPI). No business logic lives here."""

from confocal.api.app import create_app

__all__ = ["create_app"]
