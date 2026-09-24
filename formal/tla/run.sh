#!/usr/bin/env bash
# Runs every faithful TLC config (the ones expected to pass) and reports PASS/FAIL.
#
# Usage: run.sh [Spec/Config ...]      e.g. run.sh VersionPublish/ReplayRest
# Env:   TLA2TOOLS    path to tla2tools.jar (default: ./tla2tools.jar)
#        TLC_META     dir for TLC state, outside the repo (default: $TMPDIR/tlc-meta)
#        TLC_WORKERS  default: auto;  TLC_XMX  default: 4g
#
# The faithful configs take roughly an hour in total on 4 cores (see each
# spec's README for per-config times). Expected-fail configs (mutations,
# findings, witnesses) are listed in the per-spec READMEs; run them by name.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
jar="$(realpath "${TLA2TOOLS:-tla2tools.jar}")"
meta="${TLC_META:-${TMPDIR:-/tmp}/tlc-meta}"

# Spec/Config -> module to check (the rest use <Spec>.tla)
module() { case "$1" in PublisherLease) echo MC.tla ;; *) echo "$1.tla" ;; esac; }

FAITHFUL=(
  PublisherLease/PublisherLease
  PublisherLease/PublisherLease_unknowns
  PublisherLease/PublisherLease_crash
  PublisherLease/Var_cpp_keep_lease
  VersionPublish/Faithful
  VersionPublish/FaithfulLarge
  VersionPublish/ReplayRest
  VersionPublish/SkipClaimReadbackRest
  PayloadRing/FaithfulRecord
  PayloadRing/FaithfulRecordLiveness
  PayloadRing/LegacyHooksWithinTaskCap
  PayloadRing/LegacyHooksWithinTaskCapLiveness
)
[ $# -gt 0 ] && FAITHFUL=("$@")

[ -f "$jar" ] || { echo "tla2tools.jar not found; set TLA2TOOLS (see README.md)" >&2; exit 2; }
fails=0
for sc in "${FAITHFUL[@]}"; do
  spec="${sc%%/*}"; cfg="${sc#*/}"
  log="$meta/$spec-$cfg.log"; md="$meta/$spec-$cfg-$$"
  mkdir -p "$md"
  start=$SECONDS
  if (cd "$here/$spec" && java -XX:+UseParallelGC -Xmx"${TLC_XMX:-4g}" -cp "$jar" tlc2.TLC \
        -workers "${TLC_WORKERS:-auto}" -metadir "$md" -noGenerateSpecTE -cleanup \
        -config "$cfg.cfg" "$(module "$spec")") > "$log" 2>&1; then
    res=PASS
  else
    res=FAIL; fails=$((fails + 1))
  fi
  printf '%-4s %-45s %5ss  %s\n' "$res" "$sc" $((SECONDS - start)) "$log"
done
echo "$(( ${#FAITHFUL[@]} - fails ))/${#FAITHFUL[@]} passed"
[ "$fails" -eq 0 ]
