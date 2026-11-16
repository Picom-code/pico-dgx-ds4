#!/bin/bash
# Every batch shape the engine claims bit-exact, each session against the same session run alone
# (docs/batching.md B2/B3 "Outputs", docs/wide64.md "Exactness": steps of up to kDecodeRows = 16 rows;
# the >16-row wide path is opt-in and out of scope). The engine's deterministic modes print one hash a
# session, FNV-1a over the token it chose and that token's logit (golden/README.md), so equal hashes are
# equal bits:
#
#   alone    --mode decode --golden G --ctx C --steps K --big
#   plain    --mode batch --golden G --big --slots S --steps K --batch "C0,C1,..."        (B2, S = 1..16)
#   spec     the same and --spec --draft-fp8 (serve's drafter head)                     (B3, S = 2..8)
#   specced  the same and --spec, --batch "128,512,1024c,1536,2048c,2560,3072c,3584"     (drafts, CED sessions)
#   specbig  the same and --spec --slots 8 --big-slots 4 (the long contexts in the big slots)
#   spect    the same and --spec, --temp 0.8 --top-p 0.95 --top-k 40 --seed 7            (S = 3, 6, 8)
#   spectced the same, with CED sessions (drafts, CED and sampling in one batch)               (S = 8)
#   sampled  the same as plain and alone, and --temp 0.8 --top-p 0.95 --top-k 40 --seed 7 (S = 2,4,8,16)
#   ced      the same as plain and alone, and --ced, --batch "C0c,C1c,..."                (4 sessions)
#   cedmix   the same and --batch "C0c,..." (every fourth session CED; 12 and 16 sessions)
#   bigslots the same as plain, and --slots 16 --big-slots 4 (the long contexts in the big slots)
#   bigsampled / bigced  that layout with --temp ... / with CED sessions
#   tf alone --mode tf --big --golden G --prompt C --limit C + 16 --k G
#   batchtf  --mode batch --golden G --big --slots S --batch "..." --batch-tf --batch-rows G --steps 16/G
#            (B4's check: G * S = 16 rows; tf mode passes only --temp and --seed to the sampler, so the
#            sampled teacher-forced pair passes those two and not --top-p/--top-k)
#
# Each session reads its own window of the long transcript (the ladder: 128 512 1024 1536 2048 2560 3072
# 3584 4096 5120 6144 7168 10240 12288 16384 20480, ascending so --big-slots' long ones come last):
# different texts, context lengths and experts. A batch session's hash must equal its alone run's at
# that context, bit for bit. A session whose context ends in c (--batch "1024c") prefills through CED
# (docs/ced.md), alone and in the batch; its alone run passes --ced.
#
# The runs go through engine/server/run4.sh, JOB_RUNS to a ring job (engine/server/ring.sh), as
# golden/refhash.sh does. Each run loads the weights and prefills its contexts; the steps are small
# (docs/batching.md: a plain step is 18.55 + 4.6 (B - 1) ms, 50-70 ms with drafts), so a run takes 30-60 s:
# 111 runs in all, about an hour on four GB10s. --only runs some groups (the alone references they need
# come with them).
#
#   engine/server/tests/batch_matrix.sh                    # every group: plain spec sampled ced bigslots tf
#   engine/server/tests/batch_matrix.sh --only plain,spec   # some of them
#   engine/server/tests/batch_matrix.sh --plan              # print every run's command line, run none
#   G=DIR engine/server/tests/batch_matrix.sh --only tf     # another golden directory
#   engine/server/tests/batch_matrix.sh -- --weights PATH   # arguments after -- go to every run
#
# Environment: G the golden directory (default /data/dsgb10/golden/long: its tokens and positions, 28,489
# tokens); BIN the engine binary (run4.sh's default, engine/build/server/dsgb10_rank); OUT the logs
# (default ~/dsgb10/batch-matrix/<date>-<time>, made 0700); OWNER who holds the ring (default
# batch_matrix); JOB_RUNS runs per ring job (default 4, about 2-4 minutes); TIMEOUT per run in s
# (default 300); STEPS tokens a session decodes (default 64; the teacher-forced runs use 16 positions).
# Exit 0 when every session of every batch printed its alone run's hash.
set -u
cd "$(dirname "$0")/../../.." || exit 2
SELF=$PWD/engine/server/tests/batch_matrix.sh
G=${G:-/data/dsgb10/golden/long}
STEPS=${STEPS:-64}
TFROWS=16  # positions each session goes on by in the teacher-forced runs
CTX=(128 512 1024 1536 2048 2560 3072 3584 4096 5120 6144 7168 10240 12288 16384 20480)
CEDCTX=(1024 4096 12288 20480)
CEDMIX12=128c,512,1024,1536,2048c,2560,3072,3584,4096c,5120,6144,7168  # every fourth session CED
CEDMIX16=$CEDMIX12,10240,12288c,16384,20480
TEMP=(--temp 0.8 --top-p 0.95 --top-k 40 --seed 7)
log() { echo "$(date '+%F %T') batch_matrix: $*"; }
ladder() {  # n: the first n contexts, comma separated
  local n=$1 k out=
  for ((k = 0; k < n; ++k)); do out+=${out:+,}${CTX[k]}; done
  echo "$out"
}
ced_ladder() {
  local c out=
  for c in "${CEDCTX[@]}"; do out+=${out:+,}${c}c; done
  echo "$out"
}
add_run() {  # tag args...: a run's arguments, one to a line
  local tag=$1
  shift
  [ -f "$OUT/runs/$tag.args" ] && return 0
  local a
  { for a in "$@"; do printf '%s\n' "$a"; done; } > "$OUT/runs/$tag.args"
  printf '%s\n' "$tag" >> "$OUT/runs.list"
}
meta() { printf '%s %s\n' "$2" "$3" > "$OUT/runs/$1.meta"; }  # tag mode sessions
wants() {  # tag alone-prefix contexts...: the alone run each session is compared with
  local tag=$1 pfx=$2 c
  shift 2
  : > "$OUT/runs/$tag.want"
  for c in "$@"; do printf '%s alone.%s%s\n' "$c" "$pfx" "$c" >> "$OUT/runs/$tag.want"; done
}
# tag batch-list [ced-prefix plain-prefix]: the alone run each session is compared with (alone.c for a CED
# session, alone.g otherwise)
wants_list() {
  local tag=$1 list=$2 pc=${3:-c} pg=${4:-g} e c pfx
  : > "$OUT/runs/$tag.want"
  local IFS=,
  for e in $list; do
    c=${e%c}
    if [ "$c" = "$e" ]; then pfx=$pg; else pfx=$pc; fi
    printf '%s alone.%s%s\n' "$c" "$pfx" "$c" >> "$OUT/runs/$tag.want"
  done
}
alone_ced() {  # the alone (--ced) run of each context given
  local c
  for c in "$@"; do add_run "alone.c$c" --mode decode --golden "$G" --ctx "$c" --steps "$STEPS" --big --ced; done
}
alone_sampled() {  # the alone run of each context given, sampled as the batches are
  local c
  for c in "$@"; do add_run "alone.t$c" --mode decode --golden "$G" --ctx "$c" --steps "$STEPS" --big "${TEMP[@]}"; done
}

# ---- the groups
alone_greedy() {  # the alone run of every context in the ladder
  local c
  for c in "${CTX[@]}"; do add_run "alone.g$c" --mode decode --golden "$G" --ctx "$c" --steps "$STEPS" --big; done
}
group_plain() {  # B2: S = 1..16 sessions, one row each per step
  local S
  alone_greedy
  for ((S = 1; S <= 16; ++S)); do
    add_run "batch.p$S" --mode batch --golden "$G" --big --slots "$S" --steps "$STEPS" --batch "$(ladder "$S")"
    wants "batch.p$S" g "${CTX[@]:0:S}"
    meta "batch.p$S" plain "$S"
  done
  # the slots' order is not the sessions': S sessions in 16 slots sit in slots 15..(16 - S)
  add_run batch.p8s16 --mode batch --golden "$G" --big --slots 16 --steps "$STEPS" --batch "$(ladder 8)"
  wants batch.p8s16 g "${CTX[@]:0:8}"
  meta batch.p8s16 plain 8
}
group_spec() {  # B3: drafts across 2..8 sessions (serve: n <= spec_batch_rows() / 2)
  local S
  alone_greedy
  for ((S = 2; S <= 8; ++S)); do
    add_run "batch.s$S" --mode batch --golden "$G" --big --slots "$S" --steps "$STEPS" --spec --draft-fp8 --batch "$(ladder "$S")"
    wants "batch.s$S" g "${CTX[@]:0:S}"
    meta "batch.s$S" spec "$S"
  done
  # drafts with CED sessions in the same batch (three of the eight), the rest plain
  local sc=128,512,1024c,1536,2048c,2560,3072c,3584
  alone_ced 1024 2048 3072
  add_run batch.sc8 --mode batch --golden "$G" --big --slots 8 --steps "$STEPS" --spec --draft-fp8 --batch "$sc"
  wants_list batch.sc8 "$sc"
  meta batch.sc8 specced 8
  # drafts with --big-slots: the batch list's last four (the long contexts) get slots 3..0, the big ones
  local sb=128,512,1024,1536,10240,12288,16384,20480
  add_run batch.sbig8 --mode batch --golden "$G" --big --slots 8 --big-slots 4 --steps "$STEPS" --spec --draft-fp8 --batch "$sb"
  wants_list batch.sbig8 "$sb"
  meta batch.sbig8 specbig 8
  # drafts, sampled: the temperature, top-p, top-k and seed go to the batch's sampler as to the alone runs
  alone_sampled "${CTX[@]:0:8}"
  for S in 3 6 8; do
    add_run "batch.st$S" --mode batch --golden "$G" --big --slots "$S" --steps "$STEPS" --spec --draft-fp8 "${TEMP[@]}" \
      --batch "$(ladder "$S")"
    wants "batch.st$S" t "${CTX[@]:0:S}"
    meta "batch.st$S" spect "$S"
  done
  # all three at once: drafts, CED sessions and sampling
  local stc=128,512,1024c,1536,2048c,2560,3072c,3584
  for c in 1024 2048 3072; do
    add_run "alone.tc$c" --mode decode --golden "$G" --ctx "$c" --steps "$STEPS" --big --ced "${TEMP[@]}"
  done
  add_run batch.stc8 --mode batch --golden "$G" --big --slots 8 --steps "$STEPS" --spec --draft-fp8 "${TEMP[@]}" --batch "$stc"
  wants_list batch.stc8 "$stc" tc t
  meta batch.stc8 spectced 8
}
group_sampled() {
  local S
  alone_sampled "${CTX[@]}"
  for S in 2 4 8 16; do
    add_run "batch.t$S" --mode batch --golden "$G" --big --slots "$S" --steps "$STEPS" "${TEMP[@]}" --batch "$(ladder "$S")"
    wants "batch.t$S" t "${CTX[@]:0:S}"
    meta "batch.t$S" sampled "$S"
  done
}
group_ced() {
  alone_greedy
  alone_ced 128 1024 2048 4096 12288 20480
  add_run batch.ced --mode batch --golden "$G" --big --slots 4 --steps "$STEPS" --batch "$(ced_ladder)"
  wants batch.ced c "${CEDCTX[@]}"
  meta batch.ced ced 4
  # 12 and 16 rows: CED sessions beside plain ones in the same batch (the prefill path sets CED per
  # session), the plain ones at the contexts group_plain runs
  add_run batch.ced12 --mode batch --golden "$G" --big --slots 12 --steps "$STEPS" --batch "$CEDMIX12"
  wants_list batch.ced12 "$CEDMIX12"
  meta batch.ced12 cedmix 12
  add_run batch.ced16 --mode batch --golden "$G" --big --slots 16 --steps "$STEPS" --batch "$CEDMIX16"
  wants_list batch.ced16 "$CEDMIX16"
  meta batch.ced16 cedmix 16
}
group_bigslots() {  # 4 big slots (max_ctx), 12 of max_ctx / 8; the ladder ascends, so the long ones are last
  alone_greedy
  add_run batch.big --mode batch --golden "$G" --big --slots 16 --big-slots 4 --steps "$STEPS" --batch "$(ladder 16)"
  wants batch.big g "${CTX[@]}"
  meta batch.big bigslots 16
  # the same slot layout, sampled; and with CED sessions (the long ones, in the big slots)
  alone_sampled "${CTX[@]}"
  add_run batch.tbig --mode batch --golden "$G" --big --slots 16 --big-slots 4 --steps "$STEPS" "${TEMP[@]}" \
    --batch "$(ladder 16)"
  wants batch.tbig t "${CTX[@]}"
  meta batch.tbig sampledbig 16
  alone_ced 128 2048 4096 12288
  add_run batch.cedbig --mode batch --golden "$G" --big --slots 16 --big-slots 4 --steps "$STEPS" --batch "$CEDMIX16"
  wants_list batch.cedbig "$CEDMIX16"
  meta batch.cedbig cedbig 16
}
group_tf() {  # B4: teacher-forced rows, G rows of each session a step, G * S = 16 rows
  local c
  for c in "${CTX[@]}"; do  # G = 1: 16 sessions of one row
    add_run "alone.tf.g1.c$c" --mode tf --big --golden "$G" --prompt "$c" --limit "$((c + TFROWS))" --k 1
  done
  add_run batchtf.s16.g1 --mode batch --golden "$G" --big --slots 16 --steps "$TFROWS" --batch "$(ladder 16)" \
    --batch-tf --batch-rows 1
  wants batchtf.s16.g1 tf.g1.c "${CTX[@]}"
  meta batchtf.s16.g1 tf 16
  add_run batchtf.s8.g1 --mode batch --golden "$G" --big --slots 8 --steps "$TFROWS" --batch "$(ladder 8)" \
    --batch-tf --batch-rows 1
  wants batchtf.s8.g1 tf.g1.c "${CTX[@]:0:8}"
  meta batchtf.s8.g1 tf 8
  add_run batchtf.s1.g1 --mode batch --golden "$G" --big --slots 1 --steps "$TFROWS" --batch "$(ladder 1)" \
    --batch-tf --batch-rows 1
  wants batchtf.s1.g1 tf.g1.c "${CTX[@]:0:1}"
  meta batchtf.s1.g1 tf 1
  for c in "${CTX[@]:0:4}"; do  # G = 4: four rows of each of four sessions
    add_run "alone.tf.g4.c$c" --mode tf --big --golden "$G" --prompt "$c" --limit "$((c + TFROWS))" --k 4
  done
  add_run batchtf.s4.g4 --mode batch --golden "$G" --big --slots 4 --steps "$((TFROWS / 4))" --batch "$(ladder 4)" \
    --batch-tf --batch-rows 4
  wants batchtf.s4.g4 tf.g4.c "${CTX[@]:0:4}"
  meta batchtf.s4.g4 tf 4
  for c in "${CTX[@]:0:4}"; do  # sampled: --temp and --seed only (tf mode passes no top-p/top-k)
    add_run "alone.tf.g1t.c$c" --mode tf --big --golden "$G" --prompt "$c" --limit "$((c + TFROWS))" --k 1 --temp 0.8 --seed 7
  done
  add_run batchtf.s4.g1t --mode batch --golden "$G" --big --slots 4 --steps "$TFROWS" --batch "$(ladder 4)" \
    --batch-tf --batch-rows 1 --temp 0.8 --seed 7
  wants batchtf.s4.g1t tf.g1t.c "${CTX[@]:0:4}"
  meta batchtf.s4.g1t tf-sampled 4
}

# ---- a ring job: batch_matrix.sh --job OUT NN:tag... [-- engine args]; under ring.sh, each run through run4.sh
if [ "${1:-}" = --job ]; then
  OUT=$2
  shift 2
  RUNS=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do RUNS+=("$1") && shift; done
  [ $# -gt 0 ] && shift
  [ -n "${RING_HOLDER:-}" ] || { log "a job runs under engine/server/ring.sh"; exit 2; }
  for r in "${RUNS[@]}"; do
    i=${r%%:*} tag=${r#*:}
    A=()
    while IFS= read -r line; do A+=("$line"); done < "$OUT/runs/$tag.args"
    t0=$SECONDS
    TIMEOUT=${TIMEOUT:-300} engine/server/run4.sh "${A[@]}" "$@" > "$OUT/$tag.log" 2>&1 < /dev/null
    rc=$?
    h=$(grep -o 'hash [0-9a-f]\{16\}' "$OUT/$tag.log" | tail -n 1 | cut -d' ' -f2)
    echo "$i $tag $rc $((SECONDS - t0)) ${h:--}" >> "$OUT/results"
    log "$i $tag: exit $rc after $((SECONDS - t0)) s, hash ${h:-none}"
  done
  exit 0
fi

# ---- the driver
case ${1:-} in -h | --help) sed -n '2,/^# Exit 0/p' "$SELF"; exit 0 ;; esac
ONLY="plain spec sampled ced bigslots tf"
PLAN=0
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case $1 in
    --only)
      [ $# -ge 2 ] || { echo "batch_matrix.sh: --only needs a list" >&2; exit 2; }
      ONLY=${2//,/ }
      shift 2
      ;;
    --plan) PLAN=1; shift ;;
    *) echo "batch_matrix.sh: unknown argument $1 (--only plain,spec,sampled,ced,bigslots,tf, --plan, or -- engine args)" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] && shift
EXTRA=("$@")
for g in $ONLY; do
  case $g in
    plain | spec | sampled | ced | bigslots | tf) ;;
    *) echo "batch_matrix.sh: no group '$g' (plain, spec, sampled, ced, bigslots, tf)" >&2; exit 2 ;;
  esac
done
OUT=${OUT:-$HOME/dsgb10/batch-matrix/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT/runs" && chmod 700 "$OUT" || exit 2
: > "$OUT/results"
: > "$OUT/runs.list"
for g in $ONLY; do "group_$g"; done
TAGS=()
while IFS= read -r t; do TAGS+=("$t"); done < "$OUT/runs.list"
if [ $PLAN = 1 ]; then  # the command line of every run, and nothing else
  for t in "${TAGS[@]}"; do
    printf '%s:' "$t"
    while IFS= read -r a; do printf ' %s' "$a"; done < "$OUT/runs/$t.args"
    printf '\n'
  done
  exit 0
fi
export OWNER=${OWNER:-batch_matrix} OUT
B=${BIN:-engine/build/server/dsgb10_rank}
[ -f "$B" ] || { echo "batch_matrix.sh: no engine binary $B (BIN)" >&2; exit 2; }
log "binary $B (sha256 $(sha256sum "$B" | cut -c1-16)); groups ${ONLY}; G $G; STEPS $STEPS; logs in $OUT"
log "${#TAGS[@]} runs: $(for g in $ONLY; do printf '%s ' "$g"; done)"

. engine/server/ringlock.sh
ring_want_hold "batch_matrix: ${ONLY}" || log "no want held between the jobs"
trap ring_want_drop EXIT
have_run() { awk -v t="$1" '$2 == t && $3 == 0 { f = 1 } END { exit !f }' "$OUT/results" 2> /dev/null; }
per=${JOB_RUNS:-4}
for ((j = 0; j < ${#TAGS[@]}; j += per)); do
  [ $j = 0 ] || sleep 5  # a job waiting for the ring may go between ours
  JOB=()
  for ((k = j; k < j + per && k < ${#TAGS[@]}; ++k)); do JOB+=("$(printf '%02d' $((k + 1))):${TAGS[k]}"); done
  for _ in 1 2 3; do
    todo=()
    for t in "${JOB[@]}"; do have_run "${t#*:}" || todo+=("$t"); done
    [ ${#todo[@]} -gt 0 ] || break
    t0=$SECONDS
    engine/server/ring.sh bash "$SELF" --job "$OUT" "${todo[@]}" -- ${EXTRA[@]+"${EXTRA[@]}"} >> "$OUT/jobs.log" 2>&1
    rc=$?
    log "job ${todo[*]}: ring.sh exit $rc after $((SECONDS - t0)) s (the wait for the ring included)"
    [ $rc = 76 ] || break
    sleep 30  # preempted: the runs it did not finish go again
  done
done
ring_want_drop

# ---- the comparison: each session's hash against its alone run's
declare -A RC=()
while read -r _ tag rc _ _; do RC[$tag]=$rc; done < "$OUT/results"
alone_hash() {  # tag: the hash its log printed
  local f=$OUT/$1.log
  [ -f "$f" ] || return 0
  case $1 in
    alone.tf*) grep -o 'teacher-forced: .*hash [0-9a-f]\{16\}' "$f" | tail -n 1 | sed 's/.*hash //' ;;
    *) grep -o '^decode: context .*tokens hash [0-9a-f]\{16\}' "$f" | tail -n 1 | sed 's/.*tokens hash //' ;;
  esac
}
batch_pairs() {  # log: "context hash" a line for each session
  awk '
    /^batch: session / || /^batch-tf: session / {
      c = ""; h = ""
      if (match($0, /context [0-9]+/)) c = substr($0, RSTART + 8, RLENGTH - 8)
      if (match($0, /(tokens|samples) hash [0-9a-f]+/)) { h = substr($0, RSTART, RLENGTH); sub(/.*hash /, "", h) }
      if (c != "" && h != "") print c, h
    }' "$1"
}
bad=0
n=0
declare -A MODEN=() MODEBAD=()
check_batch() {  # tag mode sessions
  local tag=$1 mode=$2 S=$3
  local -A H=()
  local c h ctx alone want got k=0
  [ -f "$OUT/$tag.log" ] && while read -r c h; do H[$c]=$h; done < <(batch_pairs "$OUT/$tag.log")
  while read -r ctx alone; do
    want=$(alone_hash "$alone")
    got=${H[$ctx]:-}
    n=$((n + 1))
    MODEN[$mode]=$(( ${MODEN[$mode]:-0} + 1 ))
    if [ -z "$want" ]; then
      printf '  %-4s %-10s S=%-2s session %-2d ctx %-5s its alone run %s printed no hash (exit %s)\n' FAIL "$mode" "$S" "$k" "$ctx" "$alone" "${RC[$alone]:-not run}"
      bad=$((bad + 1))
      MODEBAD[$mode]=$(( ${MODEBAD[$mode]:-0} + 1 ))
    elif [ -z "$got" ]; then
      printf '  %-4s %-10s S=%-2s session %-2d ctx %-5s the batch printed no line for it (exit %s)\n' FAIL "$mode" "$S" "$k" "$ctx" "${RC[$tag]:-not run}"
      bad=$((bad + 1))
      MODEBAD[$mode]=$(( ${MODEBAD[$mode]:-0} + 1 ))
    elif [ "$got" = "$want" ]; then
      printf '  %-4s %-10s S=%-2s session %-2d ctx %-5s %s\n' ok "$mode" "$S" "$k" "$ctx" "$got"
    else
      printf '  %-4s %-10s S=%-2s session %-2d ctx %-5s batch %s, alone %s\n' FAIL "$mode" "$S" "$k" "$ctx" "$got" "$want"
      bad=$((bad + 1))
      MODEBAD[$mode]=$(( ${MODEBAD[$mode]:-0} + 1 ))
    fi
    k=$((k + 1))
  done < "$OUT/runs/$tag.want"
}
printf '\n%-3s %-10s %-6s %-9s %-7s %s\n' "" mode S session ctx hash
while IFS= read -r tag; do
  [ -f "$OUT/runs/$tag.meta" ] || continue
  read -r mode S < "$OUT/runs/$tag.meta"
  check_batch "$tag" "$mode" "$S"
done < "$OUT/runs.list"
if [ ${#MODEN[@]} -gt 0 ]; then
  for m in $(printf '%s\n' "${!MODEN[@]}" | sort); do
    log "$m: $(( MODEN[$m] - ${MODEBAD[$m]:-0} )) of ${MODEN[$m]} sessions equal their alone runs"
  done
fi
if [ $bad = 0 ]; then
  log "every session of every batch ($n) equals its alone run, bit for bit"
else
  log "$bad of $n sessions differ from their alone runs (logs in $OUT)"
fi
[ $bad = 0 ]
