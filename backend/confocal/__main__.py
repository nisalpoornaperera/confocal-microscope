"""Server entry point: ``confocal-server`` or ``python -m confocal``.

Configuration comes from ``$CONFOCAL_CONFIG`` (a TOML file, e.g.
``config/confocal.pi.toml``); without it the full simulation runs.
``$CONFOCAL_DATA_DIR`` overrides the data directory.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from confocal.api.app import create_app
from confocal.config import load_settings


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="confocal-server", description="Confocal surface scanner backend (FastAPI)."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="TOML config (default: $CONFOCAL_CONFIG)"
    )
    parser.add_argument("--host", default=None, help="override [server] host")
    parser.add_argument("--port", type=int, default=None, help="override [server] port")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    server = settings.server
    host = args.host if args.host is not None else server.host
    port = args.port if args.port is not None else server.port
    logging.basicConfig(
        level=server.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    uvicorn.run(
        create_app(settings),
        host=host,
        port=port,
        log_level=server.log_level,
        ws_ping_interval=20.0,
    )


if __name__ == "__main__":
    main()
