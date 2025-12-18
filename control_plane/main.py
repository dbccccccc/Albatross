"""
Albatross Control Plane entry point.

Usage:
  python -m control_plane.main --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Albatross Control Plane")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".albatross",
        help="Directory for runtime data (logs, state).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(data_dir=args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

