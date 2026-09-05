"""``dmi`` command-line entry point.

Two subcommands: ``dmi ui`` serves DMI-configurator, and ``dmi describe-model``
writes a descriptor from a framework config. Kept deliberately small: DMI is
driven by inference frameworks, and this CLI is an authoring tool, not a
launcher for the runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .configuration.errors import ConfigurationError
from .ui.errors import UIDependencyError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmi",
        description="DMI command-line tools.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ui = subcommands.add_parser(
        "ui",
        help="Serve DMI-configurator for a model.",
        description=(
            "Open DMI-configurator in a browser to author a capture "
            "configuration for MODEL."
        ),
    )
    ui.add_argument(
        "model",
        metavar="MODEL",
        nargs="?",
        default=None,
        help=(
            "A model directory, a config.json, a Hugging Face model id, or a "
            "DMI descriptor YAML. Omit it to use the descriptor in the current "
            "directory when there is exactly one."
        ),
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
    ui.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port. Defaults to 8000, or the next free port after it.",
    )
    ui.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window.",
    )
    ui.set_defaults(handler=_run_ui)

    describe = subcommands.add_parser(
        "describe-model",
        help="Write a DMI model descriptor from a framework config.",
        description=(
            "Read a Hugging-Face-shaped model config and write the equivalent "
            "DMI model descriptor. Descriptors are generated, not hand-typed."
        ),
    )
    describe.add_argument(
        "model",
        metavar="MODEL",
        help="A model directory, a config.json, or a Hugging Face model id.",
    )
    describe.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Where to write the descriptor. Defaults to stdout.",
    )
    describe.add_argument(
        "--name",
        default=None,
        help="Human-readable model name. Defaults to the source's own name.",
    )
    describe.set_defaults(handler=_run_describe_model)

    return parser


def _resolve_ui_model(model: Optional[str]) -> str:
    """Return the model to serve, discovering one if none was named.

    Discovery only succeeds when the answer is unambiguous. Picking for the
    user among several models would be worse than asking.
    """
    if model is not None:
        return model

    from .ui.server import DESCRIPTOR_GLOBS, find_descriptors

    found = find_descriptors(".")
    if len(found) == 1:
        print(f"Using {found[0]}")
        return str(found[0])

    patterns = ", ".join(DESCRIPTOR_GLOBS)
    if not found:
        raise ConfigurationError(
            "No model given and no descriptor found in the current directory "
            f"(looked for {patterns}).\n"
            "Name a model directly:\n"
            "    dmi ui ./my-model\n"
            "    dmi ui Qwen/Qwen3-8B\n"
            "or write a descriptor first:\n"
            "    dmi describe-model ./my-model --output my-model.model.yaml"
        )

    listing = "\n".join(f"    dmi ui {path}" for path in found)
    raise ConfigurationError(
        f"No model given and {len(found)} descriptors are present. "
        f"Name one:\n{listing}"
    )


def _run_ui(args: argparse.Namespace) -> int:
    from .ui.server import serve

    model = _resolve_ui_model(args.model)
    try:
        serve(
            model,
            args.config,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except KeyboardInterrupt:
        print()  # leave the shell prompt on its own line after Ctrl-C
    return 0


def _run_describe_model(args: argparse.Namespace) -> int:
    import yaml

    from .configuration import describe_model, descriptor_to_dict, save_descriptor

    descriptor = describe_model(args.model, name=args.name)
    if args.output is None:
        print(yaml.safe_dump(descriptor_to_dict(descriptor), sort_keys=False), end="")
        return 0

    target = Path(args.output)
    save_descriptor(descriptor, target)
    topology = descriptor.topology
    print(
        f"Wrote {target}\n"
        f"  {descriptor.model.name}: {topology.num_layers} layers, "
        f"hidden {topology.hidden_size}, "
        f"{topology.num_attention_heads} heads / {topology.num_kv_heads} KV"
        + (f", {topology.num_experts} experts" if topology.is_moe else "")
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigurationError as exc:
        print(f"dmi {args.command}: {exc}", file=sys.stderr)
        return 1
    except UIDependencyError as exc:
        # The optional-dependency failures serve()/create_app() raise. They
        # were historically RuntimeError, but that class also carries
        # genuine bugs (RecursionError is one), and a traceback is the right
        # output for those.
        print(f"dmi {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
