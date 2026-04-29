"""DMI Hook Compiler — auto-inject HookPoints into model files.

Usage:
    python -m monitoring.compiler extract <source> [-o output]
    python -m monitoring.compiler compile <spec> [--source path] [--framework hf|vllm] [-o output]
"""
from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "extract":
        from .extractor import main as extract_main
        extract_main(rest)
    elif cmd == "compile":
        from .compiler import main as compile_main
        compile_main(rest)
    else:
        print(f"Unknown command: {cmd}")
        print("Available: extract, compile")
        sys.exit(1)


if __name__ == "__main__":
    main()
