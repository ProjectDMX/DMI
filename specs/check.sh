#!/usr/bin/env bash
# Run the spec checks and compare every verdict with the expected one.
#
#   specs/check.sh              the fast set: every config not marked manual,
#                               plus z3/clock_skew.py and the CBMC harness
#   specs/check.sh --all        the manual (multi-minute) configs as well
#   specs/check.sh PATTERN...   only the checks whose name matches a
#                               shell glob, e.g. 'LeaseLifecycle_O3_*' cbmc
#   specs/check.sh --list       print the expected-verdict table and exit
#
# Exits 0 when every verdict matches, 1 on any mismatch, 2 when a tool is
# missing or the table and specs/tla/ disagree.
#
# Tools, located through the environment:
#   TLA2TOOLS_JAR  path to tla2tools.jar (required for the TLA+ checks)
#   JAVA           java binary                  (default: java)
#   CBMC           cbmc binary                  (default: cbmc)
#   GOTO_CC        goto-cc binary               (default: goto-cc next to
#                                                $CBMC, else on PATH)
#   PYTHON         a Python with z3-solver      (default: python3)
#   TLC_WORKERS    TLC worker threads           (default: 4)
#   TLC_HEAP       JVM heap for TLC             (default: 4g)
#
# TLC runs in a scratch copy of specs/tla, so no states/ directory or trace
# file lands in the tree. The scratch directory is removed on success and
# kept, with every log, when something does not match.
set -uo pipefail

SPECS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# name                                expected   set
#   expected: holds | violated
#   set:      fast | manual  (manual: minutes each; run with --all)
EXPECTED=$(cat <<'EOF'
lin_distinct                          holds      fast
nocap3_distinct                       holds      fast
nocap3_ceiling                        holds      fast
nocap_distinct                        holds      fast
nocap_ceiling                         holds      fast
lin_ceiling                           violated   fast
lin_solerow                           violated   fast
lin_floor                             violated   fast
lin_publish                           violated   fast
lin_budget                            violated   fast
nocapf_distinct                       holds      fast
nocapf_ceiling                        holds      fast
frontier_distinct                     holds      fast
ec_distinct                           violated   fast
nocap_ec                              violated   fast
PublisherLease                        holds      manual
PublisherLease_base5                  holds      fast
PublisherLease_base5_ovr              violated   fast
PublisherLease_believers              holds      manual
PublisherLease_holderssafe            holds      manual
PublisherLease_holders                violated   fast
PublisherLease_noovr0                 holds      fast
PublisherLease_ovr0                   violated   fast
PublisherLease_noovr1                 holds      manual
PublisherLease_ovr1                   violated   fast
PublisherLease_overrun                holds      manual
PublisherLease_selfrace               violated   fast
PublisherLease_selfracelocked         holds      fast
PublisherLease_orphans                violated   fast
PublisherLease_chunks2_orphans        violated   fast
PublisherLease_chunks2_prefix         holds      fast
PublisherLease_chunks2                holds      fast
PublisherLease_stalepid               holds      manual
PublisherLease_nonlin_fence           holds      manual
PublisherLease_nonlin_admit           violated   fast
PublisherLease_nonlin_all             violated   fast
PublisherLease_vac_publish            violated   fast
PublisherLease_vac_fence              violated   fast
PublisherLease_vac_admit              violated   fast
PublisherLease_vac_contested          violated   fast
PublisherLease_vac_bothadmit          violated   fast
PublisherLease_vac0                   violated   fast
LeaseLifecycle_O1_tries               holds      fast
LeaseLifecycle_O1_tries12             holds      fast
LeaseLifecycle_O1_tries60             holds      fast
LeaseLifecycle_O1_tries5              violated   fast
LeaseLifecycle_O1_clean               holds      fast
LeaseLifecycle_O1_cut                 holds      fast
LeaseLifecycle_O1_late                holds      fast
LeaseLifecycle_O1_slowreq3            holds      fast
LeaseLifecycle_O1_slowreq             violated   fast
LeaseLifecycle_O1_absorb              violated   fast
LeaseLifecycle_O1_skip                violated   fast
LeaseLifecycle_O2_quar                holds      fast
LeaseLifecycle_O2_skew                violated   fast
LeaseLifecycle_O2_reuse               violated   fast
LeaseLifecycle_O2_selfref             violated   fast
LeaseLifecycle_O3_rival               violated   fast
LeaseLifecycle_O3_rivaljust           holds      fast
LeaseLifecycle_O3_stops2              holds      fast
LeaseLifecycle_O3_false               holds      fast
LeaseLifecycle_O3_falsenocut          holds      fast
LeaseLifecycle_O3_selflatch           violated   fast
LeaseLifecycle_O3_selflatch_latch     violated   fast
LeaseLifecycle_O3_stops               holds      fast
LeaseLifecycle_O4_same                holds      fast
LeaseLifecycle_O4_skew                holds      fast
LeaseLifecycle_O4_bigger              violated   fast
LeaseLifecycle_O4_compose             holds      fast
LeaseLifecycle_O5_refused             holds      fast
LeaseLifecycle_O5_cosweep             violated   fast
LeaseLifecycle_vac_held               violated   fast
LeaseLifecycle_vac_quar               violated   fast
LeaseLifecycle_vac_recov              violated   fast
LeaseLifecycle_vac_refus              violated   fast
LeaseLifecycle_vac_rival              violated   fast
LeaseLifecycle_vac_cut                violated   fast
LeaseLifecycle_vac_latch              violated   fast
LeaseLifecycle_vac_start              violated   fast
LeaseLifecycle_vac_o4waited           violated   fast
LeaseLifecycle_vac_refusedstart       violated   fast
LeaseLifecycle_vac_stops2refusal      violated   fast
LeaseLifecycle_vac_stops2rival        violated   fast
LeaseLifecycle_vac_stopsrefusal       holds      fast
z3_clock_skew                         as-expected fast
cbmc_span                             SSSSS      fast
cbmc_span_noprecondition              SSFFF      fast
EOF
)
# The two cbmc rows give the expected result of assertions P1..P5 in order,
# S for SUCCESS and F for FAILURE.

die() { echo "check.sh: $*" >&2; exit 2; }

ALL=0
LIST=0
PATTERNS=()
for arg in "$@"; do
  case "$arg" in
    --all) ALL=1 ;;
    --list) LIST=1 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option: $arg" ;;
    *) PATTERNS+=("$arg") ;;
  esac
done

if [[ $LIST -eq 1 ]]; then
  echo "$EXPECTED"
  exit 0
fi

# Every .cfg must have a row, and every TLA+ row a .cfg, or the table has
# drifted from the directory.
table_names=$(awk '{print $1}' <<<"$EXPECTED")
drift=0
for cfg in "$SPECS"/tla/*.cfg; do
  name=$(basename "$cfg" .cfg)
  grep -qx "$name" <<<"$table_names" || { echo "no expected verdict for tla/$name.cfg" >&2; drift=1; }
done
while read -r name _; do
  case "$name" in z3_*|cbmc_*) continue ;; esac
  [[ -f "$SPECS/tla/$name.cfg" ]] || { echo "table row $name has no tla/$name.cfg" >&2; drift=1; }
done <<<"$EXPECTED"
[[ $drift -eq 0 ]] || die "the expected-verdict table and specs/tla/ disagree"

selected() {  # name set
  local name=$1 set=$2
  if [[ ${#PATTERNS[@]} -gt 0 ]]; then
    local p
    for p in "${PATTERNS[@]}"; do
      # shellcheck disable=SC2053
      [[ $name == $p ]] && return 0
    done
    return 1
  fi
  [[ $set == fast || $ALL -eq 1 ]]
}

# Work out which tools the selected checks need, and fail early and clearly.
need_tlc=0 need_z3=0 need_cbmc=0
while read -r name _ set; do
  selected "$name" "$set" || continue
  case "$name" in
    z3_*) need_z3=1 ;;
    cbmc_*) need_cbmc=1 ;;
    *) need_tlc=1 ;;
  esac
done <<<"$EXPECTED"
[[ $need_tlc$need_z3$need_cbmc != 000 ]] || die "no check matches: ${PATTERNS[*]}"

JAVA=${JAVA:-java}
PYTHON=${PYTHON:-python3}
CBMC=${CBMC:-cbmc}
WORKERS=${TLC_WORKERS:-4}
HEAP=${TLC_HEAP:-4g}

if [[ $need_tlc -eq 1 ]]; then
  [[ -n ${TLA2TOOLS_JAR:-} ]] || die "set TLA2TOOLS_JAR to the path of tla2tools.jar (see specs/README.md, Getting the tools)"
  [[ -f $TLA2TOOLS_JAR ]] || die "TLA2TOOLS_JAR=$TLA2TOOLS_JAR does not exist"
  command -v "$JAVA" >/dev/null || die "java not found (set JAVA to a Java 11+ binary)"
fi
if [[ $need_z3 -eq 1 ]]; then
  command -v "$PYTHON" >/dev/null || die "PYTHON=$PYTHON not found"
  "$PYTHON" -c 'import z3' 2>/dev/null || die "PYTHON=$PYTHON cannot import z3 (pip install z3-solver, or point PYTHON at a Python that has it)"
fi
if [[ $need_cbmc -eq 1 ]]; then
  command -v "$CBMC" >/dev/null || die "cbmc not found (set CBMC to the cbmc binary; apt's cbmc on Ubuntu 20.04 is too old, see specs/README.md)"
  CBMC=$(command -v "$CBMC")
  if [[ -z ${GOTO_CC:-} ]]; then
    if [[ -x $(dirname "$CBMC")/goto-cc ]]; then GOTO_CC=$(dirname "$CBMC")/goto-cc; else GOTO_CC=goto-cc; fi
  fi
  command -v "$GOTO_CC" >/dev/null || die "goto-cc not found (set GOTO_CC; it ships with cbmc)"
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/dmi-specs-check.XXXXXX")
mkdir -p "$WORK/tla" "$WORK/logs" "$WORK/states" "$WORK/cbmc"
cp "$SPECS"/tla/*.tla "$SPECS"/tla/*.cfg "$WORK/tla/"

pass=0 fail=0
FAILED=()

report() {  # status name expected got detail
  printf '%-4s  %-36s expected %-11s got %-11s %s\n' "$1" "$2" "$3" "$4" "$5"
  if [[ $1 == PASS ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); FAILED+=("$2"); fi
}

run_tlc() {  # name expected
  local name=$1 expected=$2 module extra=() log got detail start secs
  case "$name" in
    LeaseLifecycle_*) module=LeaseLifecycle.tla ;;
    PublisherLease*) module=PublisherLease.tla ;;
    *) module=VersionAllocator.tla; extra=(-deadlock) ;;  # no stutter step
  esac
  log="$WORK/logs/$name.log"
  start=$(date +%s)
  (cd "$WORK/tla" && "$JAVA" -XX:+UseParallelGC "-Xmx$HEAP" -cp "$TLA2TOOLS_JAR" tlc2.TLC \
      -workers "$WORKERS" "${extra[@]}" -metadir "$WORK/states/$name" \
      -config "$name.cfg" "$module") </dev/null >"$log" 2>&1
  secs=$(( $(date +%s) - start ))
  if grep -q '^Model checking completed. No error has been found.' "$log"; then
    got=holds
  elif grep -qE '^Error: Invariant .* is violated|^Error: The invariant of .* is equal to FALSE' "$log"; then
    got=violated
  else
    got=error
  fi
  detail=$(grep -oE '^[0-9]+ states generated, [0-9]+ distinct states found' "$log" | tail -1 \
           | sed -E 's/ states generated, / \/ /; s/ distinct states found//')
  detail="${detail:-no state count}, ${secs}s"
  if [[ $got == violated ]]; then
    # Which invariant fell, and the trace length TLC printed (states,
    # including the initial one). Both vary with scheduling; see README.
    local inv depth
    inv=$(grep -oE '^Error: (Invariant [^ ]+ is violated|The invariant of [^ ]+ is equal to FALSE)' "$log" \
          | head -1 | sed -E 's/^Error: (Invariant |The invariant of )//; s/ is .*//')
    depth=$(grep -oE '^State [0-9]+:' "$log" | tail -1 | grep -oE '[0-9]+')
    detail="$inv${depth:+ at depth $depth}; $detail"
  fi
  if [[ $got == "$expected" ]]; then report PASS "$name" "$expected" "$got" "($detail)"
  else report FAIL "$name" "$expected" "$got" "($detail; log: $log)"; fi
}

run_z3() {  # name expected
  local log="$WORK/logs/$1.log" got
  if "$PYTHON" "$SPECS/z3/clock_skew.py" </dev/null >"$log" 2>&1; then got=as-expected; else got=mismatch; fi
  local n
  n=$(grep -cE '  OK$' "$log")
  if [[ $got == "$2" ]]; then report PASS "$1" "$2" "$got" "($n checks)"
  else report FAIL "$1" "$2" "$got" "(log: $log)"; fi
}

run_cbmc() {  # name expected
  local name=$1 expected=$2 defs=() gb log got
  [[ $name == *_noprecondition ]] && defs=(-DDROP_PRECONDITION)
  gb="$WORK/cbmc/$name.gb"
  log="$WORK/logs/$name.log"
  if ! "$GOTO_CC" -std=c++11 "${defs[@]}" "$SPECS/cbmc/payload_ring_span.cpp" -o "$gb" </dev/null >"$log" 2>&1; then
    report FAIL "$name" "$expected" "build-error" "(log: $log)"; return
  fi
  "$CBMC" --unwind 80 --unwinding-assertions "$gb" </dev/null >>"$log" 2>&1
  # [main.assertion.N] ... : SUCCESS|FAILURE, in assertion order.
  got=$(grep -E '^\[main\.assertion\.[0-9]+\]' "$log" \
        | sed -E 's/^\[main\.assertion\.([0-9]+)\].*: (SUCCESS|FAILURE)$/\1 \2/' \
        | sort -n | awk '{printf "%s", substr($2, 1, 1)}')
  if [[ $got == "$expected" ]]; then report PASS "$name" "$expected" "$got" "(P1..P5)"
  else report FAIL "$name" "$expected" "${got:-error}" "(log: $log)"; fi
}

while read -r name expected set; do
  selected "$name" "$set" || continue
  case "$name" in
    z3_*) run_z3 "$name" "$expected" ;;
    cbmc_*) run_cbmc "$name" "$expected" ;;
    *) run_tlc "$name" "$expected" ;;
  esac
done <<<"$EXPECTED"

echo
echo "$pass passed, $fail failed"
if [[ $fail -gt 0 ]]; then
  echo "mismatched: ${FAILED[*]}"
  echo "logs kept in $WORK/logs"
  exit 1
fi
rm -rf "$WORK"
if [[ ${#PATTERNS[@]} -eq 0 && $ALL -eq 0 ]]; then
  echo "manual configs not run (use --all):" $(awk '$3 == "manual" {print $1}' <<<"$EXPECTED")
fi
exit 0
