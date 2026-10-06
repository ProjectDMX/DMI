#!/usr/bin/env bash
# Run the spec checks and compare every verdict with the expected one.
#
#   specs/check.sh              the fast set: every row not marked manual --
#                               the TLA+ configs, z3/clock_skew.py, the CBMC
#                               harnesses and cbmc/sync_check.sh
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
lin_distinct                                  holds       fast
nocap3_distinct                               holds       fast
nocap3_ceiling                                holds       fast
nocap_distinct                                holds       fast
nocap_ceiling                                 holds       fast
lin_ceiling                                   violated    fast
lin_solerow                                   violated    fast
lin_floor                                     violated    fast
lin_publish                                   violated    fast
lin_budget                                    violated    fast
nocapf_distinct                               holds       fast
nocapf_ceiling                                holds       fast
frontier_distinct                             holds       fast
ec_distinct                                   violated    fast
nocap_ec                                      violated    fast
PublisherLease                                holds       manual
PublisherLease_base5                          holds       fast
PublisherLease_base5_ovr                      violated    fast
PublisherLease_believers                      holds       manual
PublisherLease_holderssafe                    holds       manual
PublisherLease_holders                        violated    fast
PublisherLease_noovr0                         holds       fast
PublisherLease_ovr0                           violated    fast
PublisherLease_noovr1                         holds       manual
PublisherLease_ovr1                           violated    fast
PublisherLease_overrun                        holds       manual
PublisherLease_selfrace                       violated    fast
PublisherLease_selfracelocked                 holds       fast
PublisherLease_orphans                        violated    fast
PublisherLease_chunks2_orphans                violated    fast
PublisherLease_chunks2_prefix                 holds       fast
PublisherLease_chunks2                        holds       fast
PublisherLease_stalepid                       holds       manual
PublisherLease_nonlin_fence                   holds       manual
PublisherLease_nonlin_admit                   violated    fast
PublisherLease_nonlin_all                     violated    fast
PublisherLease_vac_publish                    violated    fast
PublisherLease_vac_fence                      violated    fast
PublisherLease_vac_admit                      violated    fast
PublisherLease_vac_contested                  violated    fast
PublisherLease_vac_bothadmit                  violated    fast
PublisherLease_vac0                           violated    fast
PublisherLease_to_noovr0                      holds       fast
PublisherLease_to_ovr0                        violated    fast
PublisherLease_to_chunks2                     holds       fast
PublisherLease_to_chunks2_prefix              holds       fast
PublisherLease_to_retry                       holds       manual
PublisherLease_to_base5                       holds       manual
PublisherLease_vac_late                       violated    fast
PublisherLease_vac_gone                       violated    fast
PublisherLease_ovr0_mono                      violated    fast
LeaseLifecycle_O1_absorb                      violated    fast
LeaseLifecycle_O1_belief                      violated    fast
LeaseLifecycle_O1_clean                       holds       fast
LeaseLifecycle_O1_cut                         holds       fast
LeaseLifecycle_O1_fast_cy                     holds       fast
LeaseLifecycle_O1_fast_lt                     holds       fast
LeaseLifecycle_O1_late                        holds       fast
LeaseLifecycle_O1_skew_cy                     violated    fast
LeaseLifecycle_O1_skip                        holds       fast
LeaseLifecycle_O1_slow_cy                     violated    fast
LeaseLifecycle_O1_slow_lt                     violated    fast
LeaseLifecycle_O1_slowreq3                    holds       fast
LeaseLifecycle_O1_slowreq                     holds       fast
LeaseLifecycle_O1_slowreq_cycle               holds       fast
LeaseLifecycle_O1_slowreq_req                 holds       fast
LeaseLifecycle_O1_window                      holds       fast
LeaseLifecycle_O1_window_slowclaim            violated    fast
LeaseLifecycle_O2_quar                        holds       fast
LeaseLifecycle_O2_reuse                       violated    fast
LeaseLifecycle_O2_selfref                     violated    fast
LeaseLifecycle_O2_skew                        violated    fast
LeaseLifecycle_O3_false                       holds       fast
LeaseLifecycle_O3_falsenocut                  holds       fast
LeaseLifecycle_O3_rival                       violated    fast
LeaseLifecycle_O3_rivaljust                   holds       fast
LeaseLifecycle_O3_selflatch                   holds       fast
LeaseLifecycle_O3_selflatch_latch             holds       fast
LeaseLifecycle_O3_stops2                      holds       fast
LeaseLifecycle_O3_stops                       holds       fast
LeaseLifecycle_O3_unkrival                    violated    fast
LeaseLifecycle_O4_bigger                      violated    fast
LeaseLifecycle_O4_compose                     holds       fast
LeaseLifecycle_O4_deadline_noskew             violated    fast
LeaseLifecycle_O4_deadline_skew               holds       fast
LeaseLifecycle_O4_same                        holds       fast
LeaseLifecycle_O4_skew                        holds       fast
LeaseLifecycle_O4_slowstart                   holds       fast
LeaseLifecycle_O4_slowstart_pred              violated    fast
LeaseLifecycle_O4_slowstart_two               violated    fast
LeaseLifecycle_O5_cosweep                     holds       fast
LeaseLifecycle_O5_cosweep_nolock              violated    fast
LeaseLifecycle_O5_crash                       holds       fast
LeaseLifecycle_O5_refused                     holds       fast
LeaseLifecycle_vac_crash                      violated    fast
LeaseLifecycle_vac_cut                        violated    fast
LeaseLifecycle_vac_deadline_rival             violated    fast
LeaseLifecycle_vac_held                       violated    fast
LeaseLifecycle_vac_latch                      violated    fast
LeaseLifecycle_vac_lost                       violated    fast
LeaseLifecycle_vac_nolock                     violated    fast
LeaseLifecycle_vac_o4waited                   violated    fast
LeaseLifecycle_vac_ownrefusal                 violated    fast
LeaseLifecycle_vac_quar                       violated    fast
LeaseLifecycle_vac_recov                      violated    fast
LeaseLifecycle_vac_refus                      violated    fast
LeaseLifecycle_vac_refusedstart               violated    fast
LeaseLifecycle_vac_rival                      violated    fast
LeaseLifecycle_vac_slowstart                  violated    fast
LeaseLifecycle_vac_start                      violated    fast
LeaseLifecycle_vac_stops2refusal              violated    fast
LeaseLifecycle_vac_stops2rival                violated    fast
LeaseLifecycle_vac_stopsrefusal               holds       fast
PackRank_first                                holds       fast
PackRank_first_v4                             holds       fast
PackRank_rowversion                           violated    fast
PackRank_newest_pin                           holds       fast
PackRank_newest                               violated    fast
PackRank_diffreplay                           violated    fast
PackRank_unordered                            violated    fast
PackRank_vac_replay                           violated    fast
PackRank_vac_merge                            violated    fast
PackRank_vac_contest                          violated    fast
PackRank_vac_abovepin                         violated    fast
SpoolOwnership_quar                           holds       fast
SpoolOwnership_multi                          holds       fast
SpoolOwnership_bad                            holds       fast
SpoolOwnership_race                           holds       manual
SpoolOwnership_race_takeover                  holds       manual
SpoolOwnership_noreconcile                    violated    fast
SpoolOwnership_live                           holds       fast
SpoolOwnership_live_takeover                  violated    fast
SpoolOwnership_live_multi                     violated    fast
SpoolOwnership_live_fix                       holds       fast
SpoolOwnership_live_multi_fix                 holds       fast
SpoolOwnership_charge                         violated    fast
SpoolOwnership_charge_fix                     holds       fast
SpoolOwnership_m_nofilecheck                  holds       fast
SpoolOwnership_m_nodirlock                    violated    fast
SpoolOwnership_m_nodirlock_data               holds       fast
SpoolOwnership_m_nolocks                      violated    fast
SpoolOwnership_m_nolocks_data                 holds       fast
SpoolOwnership_m_sinkfirst                    violated    fast
SpoolOwnership_m_releaseunsealed              violated    fast
SpoolOwnership_m_noleasegate                  violated    fast
SpoolOwnership_m_noatfork                     violated    fast
SpoolOwnership_vac_adopted                    violated    fast
SpoolOwnership_vac_removed                    violated    fast
SpoolOwnership_vac_mismatch                   violated    fast
SpoolOwnership_vac_livesib                    violated    fast
SpoolOwnership_vac_blocked                    violated    fast
SpoolOwnership                                holds       manual
SpoolOwnership_multi3                         holds       manual
PackPipeline_main                             holds       fast
PackPipeline_main3                            holds       fast
PackPipeline_firstchunk                       violated    fast
PackPipeline_firstchunk_fixed                 holds       fast
PackPipeline_strictorphan                     violated    fast
PackPipeline_mut_nolease                      violated    fast
PackPipeline_mut_owed                         violated    fast
PackPipeline_mut_tail                         violated    fast
PackPipeline_mut_tail_close                   violated    fast
PackPipeline_mut_batches                      violated    fast
PackPipeline_mut_flushrecon                   violated    fast
PackPipeline_live_nocrash                     holds       fast
PackPipeline_live_periodic                    holds       fast
PackPipeline_live_periodic2                   holds       fast
PackPipeline_live_startonly                   violated    fast
PackPipeline_live_failuresowed                holds       fast
PackPipeline_live_noreconcile                 violated    fast
PackPipeline_vac_uploaded                     violated    fast
PackPipeline_vac_indexed                      violated    fast
PackPipeline_vac_flushtrue                    violated    fast
PackPipeline_vac_pastbatch                    violated    fast
PackPipeline_vac_owed                         violated    fast
PackPipeline_vac_twochunks                    violated    fast
PackPipeline_vac_closed                       violated    fast
PackPipelineLoop_between                      violated    fast
PackPipelineLoop_between_always               holds       fast
PackPipelineLoop_between_kick                 violated    fast
PackPipelineLoop_starve                       holds       fast
PackPipelineLoop_starve_always                violated    fast
PackPipelineLoop_vac_between                  violated    fast
PackPipelineLoop_vac_waitedagain              violated    fast
PackPipelineUpload_main                       holds       fast
PackPipelineUpload_mut_uwake                  violated    fast
PackPipelineUpload_mut_cwake                  violated    fast
PackPipelineUpload_mut_book                   violated    fast
PackPipelineUpload_vac_ok                     violated    fast
PackPipelineUpload_vac_cancelled              violated    fast
PackPipelineUpload_vac_fail                   violated    fast
PackPipelineUpload_vac_failaftercancel        violated    fast
z3_clock_skew                                 as-expected fast
cbmc_span                                     SSSSS       fast
cbmc_span_noprecondition                      SSFFF       fast
cbmc_room                                     SSSSSF      fast
cbmc_room_unsaturated                         FSFFSF      fast
cbmc_eager                                    SSSSSSFFF   fast
cbmc_eager_pre160_taskcheck                   FSSSSSFSF   fast
cbmc_eager_pre160_stepreserve                 SSSFFSFFF   fast
cbmc_eager_unsaturated                        SFFSSSFFF   fast
cbmc_eager_pre160                             FFFFSSFSF   fast
cbmc_eager_3x2                                SSSSSSFFF   manual
cbmc_cancel                                   SSSS        fast
cbmc_cancel_anyclock                          SSSF        fast
sync_cbmc_copies                              in-sync     fast
EOF
)
# A cbmc row gives the expected result of each assertion, S for SUCCESS and
# F for FAILURE: for payload_ring_span.cpp, P1..P5 in assertion order; for
# the other harnesses, in the order of the label each assertion's text
# starts with (room R1..R6, eager E1..E6 V1..V3, cancel C1..C4). cbmc_case
# below says which harness, -D flags and cbmc flags each row uses.
# sync_cbmc_copies runs cbmc/sync_check.sh: in-sync when every block a
# harness copies still occurs verbatim in native/csrc.

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
  case "$name" in z3_*|cbmc_*|sync_*) continue ;; esac
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
    sync_*) ;;
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
    PackRank_*) module=PackRank.tla; extra=(-deadlock) ;;  # bounded runs end
    SpoolOwnership*) module=SpoolOwnership.tla ;;
    # Loop and Upload before the PackPipeline_ prefix they would otherwise
    # not match anyway; kept first so the order reads as the routing.
    PackPipelineLoop_*) module=PackPipelineLoop.tla ;;
    PackPipelineUpload_*) module=PackPipelineUpload.tla ;;
    PackPipeline_*) module=PackPipeline.tla ;;
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
  elif grep -qE '^Error: Invariant .* is violated|^Error: The invariant of .* is equal to FALSE|^Error: Temporal propert(y|ies) .*(was|were|is) violated' "$log"; then
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
    inv=$(grep -oE '^Error: (Invariant [^ ]+ is violated|The invariant of [^ ]+ is equal to FALSE|Temporal property [^ ]+ was violated|Temporal properties were violated)' "$log" \
          | head -1 | sed -E 's/^Error: (Invariant |The invariant of |Temporal property )//; s/^Temporal properties were violated/temporal/; s/ (is|was) .*//')
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

# name -> harness|-D flags|cbmc flags
cbmc_case() {
  local eager="--unwind 4 --unwinding-assertions"
  case "$1" in
    cbmc_span)                     echo "payload_ring_span.cpp||--unwind 80 --unwinding-assertions" ;;
    cbmc_span_noprecondition)      echo "payload_ring_span.cpp|-DDROP_PRECONDITION|--unwind 80 --unwinding-assertions" ;;
    cbmc_room)                     echo "legacy_ring_room.cpp||" ;;
    cbmc_room_unsaturated)         echo "legacy_ring_room.cpp|-DUNSATURATED|" ;;
    cbmc_eager)                    echo "eager_safety_net.cpp|-DSTEPS=2 -DHMAX=2|$eager" ;;
    cbmc_eager_pre160_taskcheck)   echo "eager_safety_net.cpp|-DSTEPS=2 -DHMAX=2 -DPRE160_TASK_CHECK|$eager" ;;
    cbmc_eager_pre160_stepreserve) echo "eager_safety_net.cpp|-DSTEPS=2 -DHMAX=2 -DPRE160_STEP_RESERVE|$eager" ;;
    cbmc_eager_unsaturated)        echo "eager_safety_net.cpp|-DSTEPS=2 -DHMAX=2 -DUNSATURATED|$eager" ;;
    cbmc_eager_pre160)             echo "eager_safety_net.cpp|-DSTEPS=2 -DHMAX=2 -DPRE160|$eager" ;;
    cbmc_eager_3x2)                echo "eager_safety_net.cpp|-DSTEPS=3 -DHMAX=2|$eager" ;;
    cbmc_cancel)                   echo "cancel_sleep.cpp||" ;;
    cbmc_cancel_anyclock)          echo "cancel_sleep.cpp|-DANY_CLOCK|" ;;
  esac
}

run_cbmc() {  # name expected
  local name=$1 expected=$2 file defs flags gb log got
  IFS='|' read -r file defs flags <<<"$(cbmc_case "$name")"
  [[ -n $file ]] || { report FAIL "$name" "$expected" "no-case" "(add $name to cbmc_case)"; return; }
  gb="$WORK/cbmc/$name.gb"
  log="$WORK/logs/$name.log"
  # shellcheck disable=SC2086
  if ! "$GOTO_CC" -std=c++11 $defs "$SPECS/cbmc/$file" -o "$gb" </dev/null >"$log" 2>&1; then
    report FAIL "$name" "$expected" "build-error" "(log: $log)"; return
  fi
  # shellcheck disable=SC2086
  "$CBMC" $flags "$gb" </dev/null >>"$log" 2>&1
  if [[ $file == payload_ring_span.cpp ]]; then
    # [main.assertion.N] ... : SUCCESS|FAILURE, in assertion order.
    got=$(grep -E '^\[main\.assertion\.[0-9]+\]' "$log" \
          | sed -E 's/^\[main\.assertion\.([0-9]+)\].*: (SUCCESS|FAILURE)$/\1 \2/' \
          | sort -n | awk '{printf "%s", substr($2, 1, 1)}')
  else
    # [fn.assertion.N] line L <LABEL> text: SUCCESS|FAILURE, ordered by label.
    got=$(grep -E '^\[[^]]*\.assertion\.[0-9]+\] (line [0-9]+ )?[A-Z][0-9] ' "$log" \
          | sed -E 's/^\[[^]]*\] (line [0-9]+ )?([A-Z][0-9]) .*: (SUCCESS|FAILURE)$/\2 \3/' \
          | sort | awk '{printf "%s", substr($2, 1, 1)}')
  fi
  if [[ $got == "$expected" ]]; then report PASS "$name" "$expected" "$got" ""
  else report FAIL "$name" "$expected" "${got:-error}" "(log: $log)"; fi
}

run_sync() {  # name expected
  local log="$WORK/logs/$1.log" got
  if bash "$SPECS/cbmc/sync_check.sh" </dev/null >"$log" 2>&1; then got=in-sync; else got=drifted; fi
  if [[ $got == "$2" ]]; then report PASS "$1" "$2" "$got" ""
  else report FAIL "$1" "$2" "$got" "(log: $log)"; fi
}

while read -r name expected set; do
  selected "$name" "$set" || continue
  case "$name" in
    z3_*) run_z3 "$name" "$expected" ;;
    cbmc_*) run_cbmc "$name" "$expected" ;;
    sync_*) run_sync "$name" "$expected" ;;
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
