#!/usr/bin/env bash
# Usage: run.sh <ConfigName> [extra TLC args]
# Env: TLA2TOOLS (path to tla2tools.jar), TLC_META (dir for TLC state; keep it outside the repo)
set -euo pipefail
cfg="$1"; shift || true
here="$(cd "$(dirname "$0")" && pwd)"
jar="${TLA2TOOLS:-tla2tools.jar}"
meta="${TLC_META:-${TMPDIR:-/tmp}/tlc-payloadring}/$cfg-$$"
mkdir -p "$meta"
cd "$here"
exec java -XX:+UseParallelGC -Xmx${TLC_XMX:-4g} -jar "$jar" -workers ${TLC_WORKERS:-auto} -metadir "$meta" -noGenerateSpecTE \
  -config "$cfg.cfg" "$@" PayloadRing.tla
