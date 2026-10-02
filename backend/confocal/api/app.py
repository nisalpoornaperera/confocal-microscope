"""FastAPI application factory.

``create_app(settings)`` returns an app whose lifespan owns one
:class:`~confocal.services.ServiceContainer` (``app.state.services``): it is
created and started (data directories, interrupted-scan recovery, hardware
connection) before the first request and stopped - scan interrupted, stage
stopped, laser off, devices and storage closed - at shutdown.

When a built frontend exists (``<repo>/frontend/dist``) it is served at ``/``
after every API route; otherwise ``/`` returns a JSON pointer to ``/docs``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from confocal import __version__
from confocal.api.errors import install_exception_handlers
from confocal.api.routes import adc, calibration, ml, scans, stage, system, websocket
from confocal.config import Settings, load_settings
from confocal.services import ServiceContainer

log = logging.getLogger(__name__)

#: ``<repo>/frontend/dist`` (this file is ``<repo>/backend/confocal/api/app.py``).
DEFAULT_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"

API_DESCRIPTION = """
REST + WebSocket API of the confocal surface scanner (OpenFlexure Delta Stage,
OPT101 photodiode via ADS1115). All coordinates are micrometres.

* Manual hardware control (move, home, ADC read / gain, calibration) is refused
  with **409** while a scan is active. `POST /api/v1/stage/stop` is always allowed
  and latches the emergency stop until `POST /api/v1/stage/reset`.
* Errors use the body `{error, detail, violations?}`.
* Live scan events: WebSocket `/ws/scans/{scan_id}` (snapshot first, closes with
  1000 after the terminal state, 4404 for an unknown scan).
"""


def create_app(settings: Settings | None = None, *, frontend_dir: Path | None = None) -> FastAPI:
    """Build the application. ``settings`` defaults to :func:`load_settings`.

    ``frontend_dir`` overrides where a built UI is looked for (default
    ``<repo>/frontend/dist``); nothing is mounted when it does not exist.
    """
    resolved = settings if settings is not None else load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = ServiceContainer.create(resolved)
        try:
            await container.start()
        except BaseException:
            await container.stop()
            raise
        app.state.services = container
        try:
            yield
        finally:
            await container.stop()
            app.state.services = None

    app = FastAPI(
        title="Confocal surface scanner",
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.services = None
    install_exception_handlers(app)
    if resolved.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    for module in (system, stage, adc, calibration, scans, ml, websocket):
        app.include_router(module.router)
    _mount_frontend(app, frontend_dir if frontend_dir is not None else DEFAULT_FRONTEND_DIR)
    return app


def _mount_frontend(app: FastAPI, directory: Path) -> None:
    if directory.is_dir():
        log.info("serving the frontend from %s", directory)
        app.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
        return

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": "confocal-surface-scanner",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "api": "/api/v1",
            "frontend": "not built (see frontend/README.md)",
        }
