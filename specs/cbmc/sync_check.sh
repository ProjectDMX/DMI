#!/usr/bin/env bash
# Check that every block a CBMC harness copies from DMI's source still occurs
# there verbatim.
#
#   specs/cbmc/sync_check.sh       exit 0 when every copy matches, 1 if not
#
# A harness marks a copy with
#   // BEGIN COPY <file> <what> [:lines]
#   ...the copied lines...
#   // END COPY
# where <file> is a basename under native/csrc/. The line numbers in the
# marker are informational; the check is a substring match of the whole
# block against the file, so a copy goes stale when its source text changes,
# not when it merely moves.
set -uo pipefail

CBMC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$CBMC_DIR/../.." && pwd)

python3 - "$CBMC_DIR" "$ROOT" <<'EOF'
import pathlib, re, sys

cbmc_dir, root = map(pathlib.Path, sys.argv[1:3])
sources = {p.name: p for p in (root / "native" / "csrc").rglob("*")
           if p.is_file()}
bad = 0
total = 0
for harness in sorted(cbmc_dir.glob("*.cpp")):
    lines = harness.read_text().split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"\s*// BEGIN COPY (\S+) (.*)$", lines[i])
        if not m:
            i += 1
            continue
        name, what = m.group(1), m.group(2)
        j = i + 1
        while j < len(lines) and not re.match(r"\s*// END COPY", lines[j]):
            j += 1
        block = "\n".join(lines[i + 1:j])
        total += 1
        src = sources.get(name)
        if src is None:
            print(f"STALE {harness.name}: {name} not found under native/csrc")
            bad += 1
        elif block not in src.read_text():
            print(f"STALE {harness.name}: {name} {what} no longer occurs verbatim")
            bad += 1
        else:
            print(f"ok    {harness.name}: {name} {what}")
        i = j + 1
print(f"{total - bad} of {total} copied blocks match")
sys.exit(1 if bad else 0)
EOF
