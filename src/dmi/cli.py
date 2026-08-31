"""``dmi`` command-line entry point.

Currently one subcommand, ``dmi ui``, which serves DMI-configurator for a model
descriptor. Kept deliberately small: DMI is driven by inference frameworks, and
this CLI is an authoring tool, not a launcher for the runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .configuration.errors import ConfigurationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmi",
        description="DMI command-line tools.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ui = subcommands.add_parser(
        "ui",
        help="Serve DMI-configurator for a model descriptor.",
        description=(
            "Open DMI-configurator in a browser to author a capture "
            "configuration for MODEL_DESCRIPTOR."
        ),
    )
    ui.add_argument(
        "descriptor",
        metavar="MODEL_DESCRIPTOR",
        help="Path to a model descriptor YAML file.",
    )
    ui.add_argument(
        "--config",
        metavar="CONFIG",
        default=None,
        help=(
            "Configuration to load at startup and save back to. Defaults to "
            "<descriptor-dir>/<model-id>.dmi.yaml."
        ),
    )
    ui.add_argument("--host", default="127.0.0.1", help="Bind address.")
    ui.add_argument("--port", type=int, default=8000, help="Bind port.")
    ui.set_defaults(handler=_run_ui)

    return parser


def _run_ui(args: argparse.Namespace) -> int:
    descriptor = Path(args.descriptor)
    if not descriptor.exists():
        print(f"dmi ui: no such descriptor: {descriptor}", file=sys.stderr)
        return 2

    from .ui.server import serve

    try:
        serve(descriptor, args.config, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print()  # leave the shell prompt on its own line after Ctrl-C
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigurationError as exc:
        print(f"dmi {args.command}: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"dmi {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
