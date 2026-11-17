#!/usr/bin/env bash
# deploy-switchless.sh — deploy pico-dgx-ds4 (ds41f-4xgx10-switchless) on a 4-node switchless DGX Spark ring.
#
# Run it on rank 0 (the box with passwordless SSH to the other three). It clones the recipe, writes the
# fleet file, downloads the checkpoint, builds the engine and the front-end, cuts the four per-rank
# weight files and the Engram shards, copies them to the boxes, and starts the service.
#
# Every phase is idempotent and resumable: a phase that finished leaves a marker in $WORK/.state, and a
# file already on a box (same size) is not copied again. The checkpoint download (475 GB) resumes.
#
#   ./deploy-switchless.sh                      # what this is, what it can do, and where things stand
#   ./deploy-switchless.sh deploy               # every phase, in order (the first run: hours, 475 GB)
#   ./deploy-switchless.sh start|stop|restart   # the service
#   ./deploy-switchless.sh status|logs          # what it is doing, its last log lines
#   ./deploy-switchless.sh crash                # why the engine last died: exit code, its last words, OOM
#   ./deploy-switchless.sh redeploy             # a new version of the recipe: fetch, rebuild, restart
#   ./deploy-switchless.sh redeploy --pr 2      # ... from a pull request's head (merged with main)
#   ./deploy-switchless.sh redeploy --ref <sha> # ... from any commit, branch or tag
#   ./deploy-switchless.sh redeploy --repo URL --ref <branch>  # ... from a fork's branch
#   ./deploy-switchless.sh redeploy --fork      # ... from the branch with PRs #2, #3, #7 and #15
#   ./deploy-switchless.sh redeploy --fork --no-stream-args  # ... without PR #7
#   ./deploy-switchless.sh redeploy --no-vision  # ... without the image encoder (text only)
#   ./deploy-switchless.sh --phase checkpoint   # one phase (repeatable; runs even if it finished before)
#   ./deploy-switchless.sh --skip checkpoint    # every phase but these
#   ./deploy-switchless.sh --force --phase build
#   ./deploy-switchless.sh --list               # the phases
#   REDEPLOY=1 ./deploy-switchless.sh deploy    # a full run that also fetches and rebuilds
#
# Every phase is idempotent: one that finished is skipped, and serve leaves a service that already
# serves with these settings alone. So `deploy` is safe to repeat, and `start` is the same as
# `--phase serve`.
#
# Settings come from the environment (the block below). The fleet file is written from the template in
# this script unless it already exists or FLEET_SRC names one. The template is one particular ring
# (qspark01-04, 192.168.2.102-104): another ring edits the file's hostnames and addresses, and
# preflight stops with what to change if they are not on the boxes.
#
# The front-end listens on 127.0.0.1 and on this box's address on the network that reaches the workers
# (the recipe refuses 0.0.0.0), on port 8888, the recipe's production endpoint. Set DSGB10_ADDR="127.0.0.1
# 192.168.2.101" to choose the addresses, DSGB10_PORT for another port (18893 is the recipe's test port;
# 8888 needs DSGB10_PROD=1, which follows the port unless you set it).
#
# By default this script deploys the repository it is in, at whatever that clone has checked out (or
# REPO_REF, main, when it cloned the recipe itself). `--fork` deploys FORK_REF from FORK_URL instead:
# the branch that carries every fix the ring needs, all four pull requests on top of main — PR #3
# (sampler open-interval endpoint, @ofan), PR #2 (DSML quarantine, @ofan), PR #7 (a stream's tool calls
# in pieces, @mpfaffenberger) and PR #15 (the service owns the image encoder, @mpfaffenberger). They are
# their authors' work, not this script's: it only merges them onto one branch. STREAM_TOOL_ARGS and
# VISION are on with it. `--fork --no-stream-args` deploys BASE_REF instead (PRs #2, #3 and #15, without
# #7); `--no-vision` leaves the encoder out.
#
# STREAM_TOOL_ARGS=1 starts the front-end with --stream-tool-args: a streamed answer carries each tool
# call in pieces (its name when that has parsed, then its arguments as they arrive) instead of in one
# chunk once the call is complete. On by default, since the branch under test has PR #7; `--no-stream-args`
# turns it off along with the branch.
#
# VISION=1 runs the image encoder (tools/vision/vision_encoder.py, docs/vision.md) on rank 0: service.sh
# is given its command (DSGB10_VISION_CMD, PR #15), so it starts the encoder before the engine, starts
# it again when it exits, stops it with the service and reports it in status. Without it dsfront
# refuses every image part with "image, audio and video input are not supported". The encoder runs in a
# container of the public image VISION_IMAGE (the
# base of the recipe's own kit image: torch, Pillow, safetensors, PyAV), so nothing is built here, and
# costs about 1 GiB of rank 0's GPU; text requests are unaffected while it is down. On by default;
# `--no-vision` turns it off. The encoder imports the checkpoint's own inference/vision.py and
# inference/image_processor.py, so the checkpoint phase fetches them (a few KB) even into a checkpoint
# that is already on disk. VISION_MEM (32g) is the container's memory limit, and on a GB10 the GPU's
# memory is host memory, so it caps the encoder's GPU memory too: a smaller limit fails in load() with
# "CUDA error: out of memory".
#
# Everything goes under one directory: the clone, the fleet file, the checkpoint, the weight files, the
# Engram shards, the spill and the API key. Run it from where you want them, or set WORK=/somewhere. The
# same directory must exist on all four boxes (the fleet file's paths are relative to it, or to each
# box's home when it is under the home directory).
#
# deploy-switchless.sh works from inside the recipe's clone too, which is how it is meant to be used:
# git clone the recipe, cd into it, ./tools/deploy-switchless/deploy-switchless.sh. Then the clone is
# the repo it is in, and everything it creates goes in one folder beside it, ./ds41deploy, which
# carries a .gitignore so that git leaves it alone.
#
# Only the model's data goes there. The toolchain keeps its usual places: rustup's ~/.cargo and
# ~/.rustup, pip's ~/.local, the Hugging Face cache in ~/.cache/huggingface. Nothing sets CARGO_HOME,
# RUSTUP_HOME, HF_HOME or XDG_*.
#
# Disk, per box: rank 0 needs the checkpoint (475 GB) + one rank's files at a time (78 GB) + its own
# weights (78 GB) + its Engram shards (51 GB) + 64 GiB of spill; ranks 1-3 need 78 + 51 + 64 GiB.
set -euo pipefail

# ---------------------------------------------------------------- settings (environment overrides)

UPSTREAM_URL=https://github.com/Picom-code/pico-dgx-ds4.git
FORK_URL=https://github.com/vesatikkanen/pico-dgx-ds4.git
# What a plain run deploys: the repository the script is in, at REPO_REF (main). That is what a reader
# of the repo expects, and it is what this script ships as.
#
# --fork deploys FORK_REF from FORK_URL instead: the branch that carries every fix the ring needs, all
# four pull requests on main — PR #3 (sampler open-interval endpoint, @ofan), PR #2 (DSML quarantine,
# @ofan), PR #7 (a stream's tool calls in pieces, @mpfaffenberger) and PR #15 (the service owns the
# image encoder, @mpfaffenberger). Their authors' work, not this script's. STREAM_TOOL_ARGS and VISION
# are on with it. --no-stream-args deploys BASE_REF instead: PRs #2, #3 and #15, without #7.
FORK_REF=${FORK_REF:-fix/sampler-dsml-stream-args}
BASE_REF=${BASE_REF:-fix/sampler-and-dsml}
FORK=${FORK:-0}                                      # 1: deploy FORK_REF from FORK_URL (--fork)
REPO_URL=${REPO_URL:-$UPSTREAM_URL}                  # the clone's origin
PR_REPO_URL=${PR_REPO_URL:-$UPSTREAM_URL}            # where refs/pull/<n>/head lives (the base repo)
REPO_REF=${REPO_REF:-main}
REF=${REF:-}                                         # what to check out: pr:N, or a commit/branch/tag
PR=${PR:-}                                           # the same as REF=pr:N (a pull request's head)
SRC_SUB=ds41f-4xgx10-switchless                      # the recipe's directory inside the repo

# Where deploy-switchless.sh is. It works from two places:
#   - in the recipe's clone at tools/deploy-switchless/deploy-switchless.sh (where the repo keeps it):
#     the repo is two levels up, and everything this script creates goes in one folder beside it,
#     <repo>/ds41deploy, which carries a .gitignore so that git leaves it alone;
#   - anywhere else: it clones the recipe into $HERE/pico-dgx-ds4 and keeps everything in $HERE.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -d "$HERE/../../$SRC_SUB" ]; then
  # tools/deploy-switchless/deploy-switchless.sh in the repo: the repo is two levels up
  IN_REPO=1
  REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
  WORK=${WORK:-$REPO/ds41deploy}
else
  IN_REPO=0
  REPO=${REPO:-$HERE/pico-dgx-ds4}
  WORK=${WORK:-$HERE}
fi

CKPT_REPO=${CKPT_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
CKPT_REV=${CKPT_REV:-1bae0e10}
CKPT=${CKPT:-$WORK/DeepSeek-V4.1-Flash-1017}         # the checkpoint's directory
OUT=${OUT:-$WORK/rankfiles}                          # staging: one rank's files at a time
FLEET=${FLEET:-$WORK/fleet.conf}                     # the fleet file the scripts read
FLEET_SRC=${FLEET_SRC:-}                             # use this fleet file instead of the template

JOBS=${JOBS:-4}                                      # cmake/cargo parallelism (the recipe uses 4)
KEEP_CHECKPOINT=${KEEP_CHECKPOINT:-1}                # 0: delete the checkpoint once the files are cut
LOSSLESS=${LOSSLESS:-0}                              # 1: drop DSGB10_OEA (the lossy expert re-route)
DSGB10_ADDR=${DSGB10_ADDR:-}                         # the front-end's addresses; empty: loopback + this box's LAN address
DSGB10_PORT=${DSGB10_PORT:-8888}                     # 8888 is the recipe's production endpoint
# service.sh refuses port 8888 unless DSGB10_PROD=1, so that follows the port unless set explicitly
if [ "$DSGB10_PORT" = 8888 ]; then DSGB10_PROD=${DSGB10_PROD:-1}; else DSGB10_PROD=${DSGB10_PROD:-0}; fi
MODEL_NAME=${MODEL_NAME:-deepseek-v4.1-flash}
ENCODING=${ENCODING:-1017}                           # dsfront's chat encoding: 1017 or v4.1
STREAM_TOOL_ARGS=${STREAM_TOOL_ARGS:-1}              # 1: a stream's tool calls in pieces (dsfront --stream-tool-args)
VISION=${VISION:-1}                                  # 1: run the image encoder (service.sh owns it)
VISION_IMAGE=${VISION_IMAGE:-lmsysorg/sglang:dev-dsv41}  # the image the encoder runs in (torch, Pillow, safetensors)
VISION_PORT=${VISION_PORT:-18950}                    # the encoder's port on rank 0's loopback
VISION_MAX_IMAGES=${VISION_MAX_IMAGES:-100}          # the encoder's --max-images
VISION_MAX_ROWS=${VISION_MAX_ROWS:-40000}            # the encoder's --max-rows (image tokens per request)
VISION_MEM=${VISION_MEM:-32g}                         # the encoder container's memory limit (see vision_cmd)
VISION_NAME=${VISION_NAME:-dsgx10-vision}            # the container an older version of this script left
INSTALL_DEPS=${INSTALL_DEPS:-0}                      # 1: apt-get the build dependencies (needs sudo)
NET_APPLY=${NET_APPLY:-0}                            # 1: --phase net adds the links' addresses (sudo); either way it writes netplan files
REDEPLOY=${REDEPLOY:-0}                              # 1: fetch the recipe, rebuild, restart (see redeploy)

STATE=$WORK/.state
SRC=$REPO/$SRC_SUB

# The fleet file's paths: WORK as the boxes see it. Under the home directory it is written ~/..., so
# every box uses its own home; otherwise it is the absolute path, which must then exist on all four.
case $WORK in
  "$HOME") FLEET_BASE='~' ;;
  "$HOME"/*) FLEET_BASE="~/${WORK#"$HOME"/}" ;;
  *) FLEET_BASE=$WORK ;;
esac

# ---------------------------------------------------------------- output

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
step() { printf '\033[1;36m  ·\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mdeploy-switchless.sh: %s\033[0m\n' "$*" >&2; exit 1; }
gib()  { awk -v b="$1" 'BEGIN{printf "%.1f GiB", b/1073741824}'; }

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  [ "${1:-0}" = 0 ] || exit "$1"
}

# ---------------------------------------------------------------- the fleet file

# The template below is one particular ring (qspark01-04, 192.168.2.10x, 10.1xx.80/81). Another ring
# gets the same file with its own addresses, and the script says so instead of failing later.
TEMPLATE_HOSTS='qspark01 qspark02 qspark03 qspark04'
TEMPLATE_SSH='192.168.2.102 192.168.2.103 192.168.2.104'

# 1 if the fleet file is still the template's ring (its hostnames and SSH addresses are unchanged)
fleet_is_template() {
  local r
  for r in 0 1 2 3; do
    case " $TEMPLATE_HOSTS " in
      *" $(fleet_get "RANK${r}_HOSTNAME") "*) ;;
      *) return 1 ;;
    esac
  done
  for r in 1 2 3; do
    case " $TEMPLATE_SSH " in
      *" $(fleet_get "RANK${r}_SSH") "*) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

# What a ring that is not the template's has to change, and how to find the values.
fleet_advice() {
  cat >&2 <<'EOF'

  The fleet file's addresses must be this ring's. The template's are qspark01-04 and 192.168.2.102-104;
  if this is another ring, edit the file before going on:

    RANK<r>_HOSTNAME   `hostname` on each box (the engine finds its rank by it)
    RANK<r>_SSH        how rank 0 reaches ranks 1-3 (passwordless: ssh-copy-id)
    RANK<r>_NEXT_IP<d> the address on the link to rank r+1, on PCI domain d
    RANK<r>_PREV_IP<d> the address on the link to rank r-1, on PCI domain d

  Rank r's NEXT_IP<d> and rank r+1's PREV_IP<d> must be in the same subnet, and each address must be
  on the netdev of the RDMA device the file names (NEXT_DEV<d>/PREV_DEV<d>, GID_INDEX). The recipe's
  docs/ring.md has the cabling; micro/ring_links.sh prints what a box has.

  Then re-run: ./deploy-switchless.sh --force --phase preflight
EOF
}

write_fleet() {
  mkdir -p "$(dirname "$FLEET")"
  # Everything this script creates lives in $WORK. When deploy-switchless.sh is in the recipe's clone, $WORK is
  # inside that clone, so it carries a .gitignore: git status stays clean and a pull never touches it.
  if [ "$IN_REPO" = 1 ] && [ ! -f "$WORK/.gitignore" ]; then
    printf '# deploy-switchless.sh keeps everything it creates here\n*\n' > "$WORK/.gitignore"
  fi
  if [ -n "$FLEET_SRC" ]; then
    [ -r "$FLEET_SRC" ] || die "FLEET_SRC=$FLEET_SRC is not readable"
    cp -f "$FLEET_SRC" "$FLEET"
    log "fleet file: copied $FLEET_SRC to $FLEET"
    return 0
  fi
  if [ -s "$FLEET" ] && [ "${FLEET_TEMPLATE:-0}" != 1 ]; then
    log "fleet file: keeping the existing $FLEET (FLEET_TEMPLATE=1 rewrites it from the template)"
    return 0
  fi
  cat > "$FLEET" <<EOF
# The fleet: the four boxes of the ring, how they are cabled and addressed, and where each keeps its
# files. Read by the engine (engine/comm/fleet.cpp) and the scripts (engine/server/fleet.sh).
# Format: KEY=value lines and # comments; a value has no spaces and no quotes. In a path a leading ~/
# is the home directory of the user on the box that uses the path, and {rank} is the rank.
#
# Every path below is under $FLEET_BASE, the directory deploy-switchless.sh runs from, on every box.

# Every box alike: the ConnectX-7's RDMA devices, port f0 (facing rank r + 1) and port f1 (facing
# rank r - 1), on PCI domain 0 (rocep1s0f*) and PCI domain 2 (roceP2p1s0f*); their RoCE v2 GID index
NEXT_DEV0=rocep1s0f0
NEXT_DEV1=roceP2p1s0f0
PREV_DEV0=rocep1s0f1
PREV_DEV1=roceP2p1s0f1
GID_INDEX=3

# Every box alike: where run4.sh copies binaries and writes logs, and the service's directory
BIN_DIR=$FLEET_BASE/dsgb10/bin
LOG_DIR=$FLEET_BASE/dsgb10/logs
SERVICE_DIR=$FLEET_BASE/dsgb10/service
# On rank 0: the ring lock; ring.owner, ring.wants/ and ring.reclaim go beside it
RING_LOCK=$FLEET_BASE/dsgb10/ring.lock

# Rank 0 (an ASUS Ascent GX10), where the scripts run
RANK0_NAME=qspark01
RANK0_HOSTNAME=qspark01
RANK0_NEXT_IP0=10.112.80.1
RANK0_NEXT_IP1=10.112.81.1
RANK0_PREV_IP0=10.141.80.2
RANK0_PREV_IP1=10.141.81.2
RANK0_WEIGHTS=$FLEET_BASE/dsgb10/weights/rank0.dsw
RANK0_HASHMAP=$FLEET_BASE/dsgb10/engram/engram_hashmap.bin
RANK0_ENGRAM=$FLEET_BASE/dsgb10/engram/rank0
RANK0_ENGRAM_READ_BATCH=1
RANK0_SPILL=$FLEET_BASE/dsgb10/spill

# Rank 1 (an ASUS Ascent GX10)
RANK1_NAME=qspark02
RANK1_HOSTNAME=qspark02
RANK1_SSH=192.168.2.102
RANK1_NEXT_IP0=10.123.80.1
RANK1_NEXT_IP1=10.123.81.1
RANK1_PREV_IP0=10.112.80.2
RANK1_PREV_IP1=10.112.81.2
RANK1_WEIGHTS=$FLEET_BASE/dsgb10/weights/rank1.dsw
RANK1_HASHMAP=$FLEET_BASE/dsgb10/engram/engram_hashmap.bin
RANK1_ENGRAM=$FLEET_BASE/dsgb10/engram/rank1
RANK1_ENGRAM_READ_BATCH=1
RANK1_SPILL=$FLEET_BASE/dsgb10/spill

# Rank 2 (an NVIDIA DGX Spark)
RANK2_NAME=qspark03
RANK2_HOSTNAME=qspark03
RANK2_SSH=192.168.2.103
RANK2_NEXT_IP0=10.134.80.1
RANK2_NEXT_IP1=10.134.81.1
RANK2_PREV_IP0=10.123.80.2
RANK2_PREV_IP1=10.123.81.2
RANK2_WEIGHTS=$FLEET_BASE/dsgb10/weights/rank2.dsw
RANK2_HASHMAP=$FLEET_BASE/dsgb10/engram/engram_hashmap.bin
RANK2_ENGRAM=$FLEET_BASE/dsgb10/engram/rank2
RANK2_ENGRAM_READ_BATCH=8
RANK2_SPILL=$FLEET_BASE/dsgb10/spill

# Rank 3 (an NVIDIA DGX Spark)
RANK3_NAME=qspark04
RANK3_HOSTNAME=qspark04
RANK3_SSH=192.168.2.104
RANK3_NEXT_IP0=10.141.80.1
RANK3_NEXT_IP1=10.141.81.1
RANK3_PREV_IP0=10.134.80.2
RANK3_PREV_IP1=10.134.81.2
RANK3_WEIGHTS=$FLEET_BASE/dsgb10/weights/rank3.dsw
RANK3_HASHMAP=$FLEET_BASE/dsgb10/engram/engram_hashmap.bin
RANK3_ENGRAM=$FLEET_BASE/dsgb10/engram/rank3
RANK3_ENGRAM_READ_BATCH=8
RANK3_SPILL=$FLEET_BASE/dsgb10/spill
EOF
  log "fleet file: wrote $FLEET (edit it, then re-run with --force --phase fleet)"
  log "the template is one particular ring: check its hostnames and addresses before going on"
  fleet_advice
}

load_fleet() {
  [ -r "$FLEET" ] || die "no fleet file $FLEET (run --phase fleet)"
  [ -r "$SRC/engine/server/fleet.sh" ] || die "no $SRC/engine/server/fleet.sh (run --phase clone)"
  export FLEET
  # shellcheck source=/dev/null
  . "$SRC/engine/server/fleet.sh"
  fleet_load "$FLEET" || die "the fleet file $FLEET is not one the scripts can use"
  WORKERS=("" "$FLEET_RANK1_SSH" "$FLEET_RANK2_SSH" "$FLEET_RANK3_SSH")
}

# ---------------------------------------------------------------- the boxes

run_on() {  # rank command
  if [ "$1" = 0 ]; then bash -c "$2"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "${WORKERS[$1]}" "$2"; fi
}

box_name() { fleet_get "RANK$1_NAME"; }

# a path value for this box (rank 0): a leading ~/ is $HOME
local_path() { fleet_local "$1"; }

# a path value for scp's remote side: relative to the home directory
remote_path() { fleet_rel "$1"; }

avail_gib() {  # path: GiB available on the filesystem that holds it (walks up to an existing directory)
  local p=$1
  while [ ! -d "$p" ]; do p=$(dirname "$p"); done
  df -B1 --output=avail "$p" | tail -1 | awk '{printf "%d", $1/1073741824}'
}

need_space() {  # path GiB what
  local have
  have=$(avail_gib "$1")
  [ "$have" -ge "$2" ] || die "$3: $1 has ${have} GiB free, needs $2 GiB"
  step "$3: $1 has ${have} GiB free"
}

need_bins() {  # the engine's tools this phase runs
  local b
  for b in "$@"; do
    [ -x "$b" ] || die "no $b: run --phase build first"
  done
}

# Every directory the fleet file names (and the checkpoint's): the service writes its files in them, so
# they must be there, or be creatable, before anything else runs. A mount point that is not mounted is
# the usual cause of a failure here.
#
# BIN_DIR, LOG_DIR and SERVICE_DIR matter most: the recipe's own run4.sh and service.sh make them on
# rank 0 only, then scp into them on the workers, so a worker without them fails the copy.
check_fleet_dirs() {
  local r k p bad=()
  for r in 0 1 2 3; do
    for k in BIN_DIR LOG_DIR SERVICE_DIR WEIGHTS HASHMAP ENGRAM SPILL; do
      case $k in
        # the same on every box
        BIN_DIR | LOG_DIR | SERVICE_DIR) p=$(fleet_rank "$(fleet_get "$k")" "$r") ;;
        *) p=$(dirname "$(fleet_rank "$(fleet_get "RANK${r}_$k")" "$r")") ;;
      esac
      if run_on "$r" "mkdir -p $p" 2> /dev/null; then
        step "rank $r: $k in $p"
      else
        bad+=("rank $r: $p ($k)")
      fi
    done
  done
  p=$(dirname "$(local_path "$FLEET_RING_LOCK")")
  if mkdir -p "$p" 2> /dev/null; then
    step "rank 0: ring lock in $p"
  else
    bad+=("rank 0: $p (RING_LOCK)")
  fi
  p=$(dirname "$CKPT")
  if mkdir -p "$p" 2> /dev/null; then
    step "rank 0: checkpoint in $p"
  else
    bad+=("rank 0: $p (CKPT)")
  fi

  if [ ${#bad[@]} -gt 0 ]; then
    for p in "${bad[@]}"; do warn "cannot make $p"; done
    warn "the filesystems this box has:"
    df -h -x tmpfs -x devtmpfs -x squashfs -x overlay 2> /dev/null | sed 's/^/    /' >&2 || true
    die "the fleet file names directories this box cannot make. They are all under $FLEET_BASE, the
  directory deploy-switchless.sh runs from: make that directory on every box (or run deploy-switchless.sh from one that
  exists everywhere, or set WORK=...)"
  fi
}

# BIN_DIR, LOG_DIR and SERVICE_DIR on all four boxes, quietly. The recipe's run4.sh and service.sh make
# them on rank 0 only and then scp into them on the workers, so a worker without them fails the copy
# ("scp: dest open .../bin/ring_test: No such file or directory").
ensure_ring_dirs() {
  local r d
  for r in 0 1 2 3; do
    for d in "$FLEET_BIN_DIR" "$FLEET_LOG_DIR" "$FLEET_SERVICE_DIR"; do
      run_on "$r" "mkdir -p $d" || die "rank $r: cannot make $d"
    done
  done
}

# O_DIRECT, when the filesystem supports it (the recipe's dd pipelines use it)
DIRECT=0
probe_direct() {
  local f=$1
  DIRECT=0
  if dd if="$f" of=/dev/null iflag=direct bs=1M count=1 status=none 2>/dev/null; then DIRECT=1; fi
}
dd_read() {  # file: its bytes on stdout
  if [ "$DIRECT" = 1 ]; then dd if="$1" iflag=direct bs=64M status=none; else dd if="$1" bs=64M status=none; fi
}
remote_direct() {  # rank: 1 if that box's filesystem takes O_DIRECT writes
  run_on "$1" "dd if=/dev/zero of=/tmp/.ds41deploy.ddprobe bs=1M count=1 oflag=direct status=none 2>/dev/null &&
    [ \"\$(stat -c %s /tmp/.ds41deploy.ddprobe 2>/dev/null)\" = 1048576 ] && rm -f /tmp/.ds41deploy.ddprobe" \
    > /dev/null 2>&1
}

sha_of() {  # file: its sha256 (coreutils prefixes a backslash when the name has one)
  sha256sum "$1" | cut -d' ' -f1 | sed 's/^\\//'
}

copy_verified() {  # src rank dst(as written for that box)
  local src=$1 r=$2 dst=$3 sum lsize rsize rsum oflag= ok=0
  lsize=$(stat -c %s "$src")
  if [ "$r" != 0 ]; then
    rsize=$(run_on "$r" "stat -c %s $dst 2>/dev/null || echo 0")
    if [ "$rsize" = "$lsize" ]; then
      step "rank $r: $dst is already there ($(gib "$lsize"))"
      return 0
    fi
    run_on "$r" "mkdir -p \$(dirname $dst)" || die "rank $r: cannot make $(dirname "$dst")"
  fi
  sum=$(sha_of "$src")
  step "rank $r: copying $(basename "$src") ($(gib "$lsize")) to $dst"
  probe_direct "$src"
  if remote_direct "$r"; then oflag=oflag=direct; fi
  # O_DIRECT needs aligned writes: a filesystem that refuses them gets a plain copy
  local err
  for oflag in $oflag ""; do
    if err=$(dd_read "$src" | ssh -o BatchMode=yes "${WORKERS[$r]}" \
      "dd of=$dst.part $oflag iflag=fullblock bs=64M status=none && mv -f $dst.part $dst" 2>&1); then
      ok=1
      break
    fi
    if [ -n "$oflag" ]; then
      step "rank $r: O_DIRECT refused, copying without it"
    else
      printf '%s\n' "$err" >&2
    fi
  done
  [ "$ok" = 1 ] || die "rank $r: copying to $dst failed"
  rsum=$(run_on "$r" "sha256sum $dst | cut -d' ' -f1 | sed 's/^\\\\//'")
  [ "$rsum" = "$sum" ] || die "rank $r: $dst: sha256 $rsum, expected $sum"
  step "rank $r: $dst verified"
}

# ---------------------------------------------------------------- phases

phase_fleet() {
  write_fleet
  if [ -r "$SRC/engine/server/fleet.sh" ]; then
    load_fleet
    step "rank 0 $(box_name 0) at $(hostname); workers ${WORKERS[1]} ${WORKERS[2]} ${WORKERS[3]}"
  else
    step "the fleet file is written; the clone phase validates it"
  fi
}

# The fleet file's addresses against the boxes: a ring that is not the template's, or a file whose
# addresses were not changed, is the usual reason a first run stops. Dies with the advice if they
# do not match.
check_fleet_addresses() {
  local r d ip missing=0
  for r in 0 1 2 3; do
    for d in 0 1; do
      for ip in "$(fleet_get "RANK${r}_NEXT_IP$d")" "$(fleet_get "RANK${r}_PREV_IP$d")"; do
        run_on "$r" "ip -4 -o addr show | grep -qw $ip" && continue
        warn "rank $r: no address $ip on any interface (./deploy-switchless.sh --phase net prints the commands; NET_APPLY=1 runs them)"
        missing=1
      done
    done
  done
  if [ "$missing" = 0 ]; then step "every link's address is configured"; fi

  local bad=0
  if fleet_is_template; then
    if [ "$missing" = 0 ]; then
      step "the fleet file is the template's ring, and its addresses are on the boxes"
    else
      warn "the fleet file is still the template's ring (qspark01-04, 192.168.2.102-104)"
      bad=1
    fi
  else
    step "the fleet file is not the template's ring"
    if [ "$missing" != 0 ]; then
      warn "some of its addresses are not on the boxes: check RANK<r>_NEXT_IP<d>/PREV_IP<d>"
      bad=1
    fi
  fi
  if [ "$bad" = 1 ]; then
    fleet_advice
    die "configure the fleet file's addresses, then re-run: ./deploy-switchless.sh --force --phase preflight"
  fi
}

phase_preflight() {
  load_fleet
  local c
  for c in git curl ssh sha256sum dd awk openssl; do
    command -v "$c" > /dev/null || die "no $c on this box"
  done
  [ "$(uname -m)" = aarch64 ] || warn "this box is $(uname -m), not aarch64: the engine builds for sm_121a (GB10) only"

  local h
  h=$(hostname)
  [ "$h" = "$FLEET_RANK0_HOSTNAME" ] ||
    warn "this box's hostname is '$h', the fleet file says RANK0_HOSTNAME=$FLEET_RANK0_HOSTNAME (the engine finds its rank by hostname unless --rank)"

  local r w rh
  for r in 1 2 3; do
    w=${WORKERS[$r]}
    rh=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" hostname 2> /dev/null) ||
      die "no passwordless SSH to rank $r ($w): ssh-copy-id $w"
    [ "$rh" = "$(fleet_get "RANK${r}_HOSTNAME")" ] ||
      warn "rank $r at $w says hostname '$rh', the fleet file says $(fleet_get "RANK${r}_HOSTNAME")"
    step "rank $r: $w ($rh) reachable"
  done

  # the ring's RDMA devices, and the kernel's CMA setting the recipe needs
  local d dev missing=0
  for r in 0 1 2 3; do
    for d in NEXT_DEV0 NEXT_DEV1 PREV_DEV0 PREV_DEV1; do
      dev=$(fleet_get "$d")
      run_on "$r" "[ -e /sys/class/infiniband/$dev ]" ||
        die "rank $r: no RDMA device $dev ($d in the fleet file): bash micro/ring_links.sh on that box"
    done
    run_on "$r" "grep -q kho=off /proc/cmdline" ||
      warn "rank $r: the kernel is not booted with kho=off, without which ibv_reg_mr fails (README step 1)"
  done
  step "the four RDMA devices are there on every box"

  # the links' addresses, as the fleet file gives them
  check_fleet_addresses

  local have
  have=$(avail_gib "$(dirname "$CKPT")")
  [ "$have" -ge 520 ] || warn "$(dirname "$CKPT") has ${have} GiB free: the checkpoint is 475 GB"
  step "checkpoint space: ${have} GiB free at $(dirname "$CKPT")"

  check_fleet_dirs
  step "every directory the fleet file names is there"
}

phase_net() {
  load_fleet
  # The ring's links: each RDMA device's netdev at GID_INDEX carries the fleet file's address for that
  # link. Prints what each box needs; NET_APPLY=1 runs it (sudo ip addr add / ip link set). Either way
  # it writes the same addresses as a netplan file per box, so they survive a reboot.
  local r d dev ip ndev cmds=() c
  local -a np_rank=() np_ndev=() np_ip=()
  for r in 0 1 2 3; do
    cmds=()
    for d in NEXT_DEV0 NEXT_DEV1 PREV_DEV0 PREV_DEV1; do
      dev=$(fleet_get "$d")
      case $d in
        NEXT_DEV0) ip=$(fleet_get "RANK${r}_NEXT_IP0") ;;
        NEXT_DEV1) ip=$(fleet_get "RANK${r}_NEXT_IP1") ;;
        PREV_DEV0) ip=$(fleet_get "RANK${r}_PREV_IP0") ;;
        PREV_DEV1) ip=$(fleet_get "RANK${r}_PREV_IP1") ;;
      esac
      ndev=$(run_on "$r" "cat /sys/class/infiniband/$dev/ports/1/gid_attrs/ndevs/$FLEET_GID_INDEX 2>/dev/null") ||
        die "rank $r: no $dev (or no GID $FLEET_GID_INDEX on it)"
      [ -n "$ndev" ] || die "rank $r: $dev has no netdev at GID $FLEET_GID_INDEX"
      np_rank+=("$r"); np_ndev+=("$ndev"); np_ip+=("$ip")
      if run_on "$r" "ip -4 -o addr show dev $ndev | grep -qw $ip"; then
        step "rank $r: $ndev has $ip ($dev)"
        continue
      fi
      step "rank $r: $dev ($ndev) needs $ip/24"
      cmds+=("ip addr add $ip/24 dev $ndev" "ip link set $ndev mtu 9000 up")
    done
    [ ${#cmds[@]} -gt 0 ] || continue
    if [ "$NET_APPLY" = 1 ]; then
      log "rank $r: applying ${#cmds[@]} commands"
      for c in "${cmds[@]}"; do
        run_on "$r" "sudo $c" || die "rank $r: sudo $c failed"
      done
    else
      for c in "${cmds[@]}"; do printf '    rank %s: sudo %s\n' "$r" "$c"; done
    fi
  done
  netplan_write "${np_rank[@]}" -- "${np_ndev[@]}" -- "${np_ip[@]}"
  [ "$NET_APPLY" = 1 ] || log "re-run with NET_APPLY=1 to apply these (they do not survive a reboot: netplan does)"
}

# The same addresses as a netplan file, so they survive a reboot. One file per box, written under $WORK
# (netplan/rank<r>.yaml); copy it to /etc/netplan/ on that box and run netplan apply. The netdev names
# come from the boxes themselves, so the file is right for the hardware rather than for one example.
netplan_write() {  # rank... -- netdev... -- ip...
  local -a ranks=() ndevs=() ips=()
  local a r i part=0
  for a in "$@"; do
    if [ "$a" = -- ]; then part=$((part + 1)); continue; fi
    case $part in
      0) ranks+=("$a") ;;
      1) ndevs+=("$a") ;;
      2) ips+=("$a") ;;
    esac
  done
  [ "${#ranks[@]}" -gt 0 ] || return 0
  mkdir -p "$WORK/netplan" || die "cannot make $WORK/netplan"
  for r in 0 1 2 3; do
    local out=$WORK/netplan/rank$r.yaml any=0
    {
      printf '# The ring links of rank %s (%s), from deploy-switchless.sh --phase net.\n' "$r" "$(box_name "$r")"
      printf '# Copy to /etc/netplan/60-ring.yaml on that box, then: sudo netplan apply\n'
      printf 'network:\n  version: 2\n  ethernets:\n'
      for i in "${!ranks[@]}"; do
        [ "${ranks[$i]}" = "$r" ] || continue
        any=1
        printf '    %s:\n      addresses: [%s/24]\n      mtu: 9000\n      dhcp4: false\n' \
          "${ndevs[$i]}" "${ips[$i]}"
      done
    } > "$out"
    if [ "$any" = 1 ]; then
      step "rank $r: netplan file $out"
    else
      rm -f "$out"
    fi
  done
  log "netplan files are in $WORK/netplan (one per box; copy each to /etc/netplan/60-ring.yaml there)"
}

# What to check out: the branch REPO_REF of REPO_URL, or a pull request's head, or any ref.
#   ./deploy-switchless.sh redeploy --pr 2          # the head of Picom-code/pico-dgx-ds4's PR 2
#   ./deploy-switchless.sh redeploy --ref <sha>     # any commit, branch or tag
#   REF=pr:2 ./deploy-switchless.sh deploy          # the same, for a full run
# A pull request's head is merged with REPO_REF first (the PR may be behind main), so the build has
# both the PR's changes and everything main has since. The merge is local: nothing is pushed.
#
# A fork is tested with --repo (or REPO_URL): the clone's origin becomes that repository, and a branch
# or commit of it is deployed like any other ref. A pull request's head still comes from the base
# repository (PR_REPO_URL), since that is where refs/pull/<n>/head lives.
#   ./deploy-switchless.sh redeploy --repo https://github.com/vesatikkanen/pico-dgx-ds4.git \
#       --ref frontend/stream-tool-args-v2
#
# By default this script deploys the repository it is in, at whatever that clone has checked out (or
# REPO_REF, main, when it cloned the recipe itself).
#
# --fork deploys FORK_REF (fix/sampler-dsml-stream-args) from FORK_URL instead: that branch carries
# every fix the ring needs, all four pull requests (PR #3's sampler endpoint and PR #2's DSML
# quarantine, both @ofan's; PR #7's streamed tool calls and PR #15's encoder in the service, both
# @mpfaffenberger's).
#   ./deploy-switchless.sh redeploy --fork
#
# --fork --no-stream-args deploys BASE_REF (fix/sampler-and-dsml) instead: PRs #2, #3 and #15, without
# PR #7. It also turns STREAM_TOOL_ARGS off, since that branch has no --stream-tool-args.
#   ./deploy-switchless.sh redeploy --fork --no-stream-args
REF=${REF:-}                                        # pr:N, or a commit/branch/tag; empty: REPO_REF
PR=${PR:-}                                           # the same as REF=pr:N
STREAM_ARGS=${STREAM_ARGS:-1}                        # 0: with --fork, deploy BASE_REF and turn STREAM_TOOL_ARGS off

# The ref to check out, from REF/PR/FORK/REPO_REF. When the script is in the clone and nothing was
# asked for explicitly, it is whatever that clone has checked out: the clone is the user's, and moving
# its working tree off their branch to deploy main is not this script's business.
target_ref() {
  if [ -n "$PR" ]; then printf 'pr:%s\n' "$PR"; return; fi
  if [ -n "$REF" ]; then printf '%s\n' "$REF"; return; fi
  if [ "$FORK" = 1 ]; then
    if [ "$STREAM_ARGS" = 0 ]; then printf '%s\n' "$BASE_REF"; else printf '%s\n' "$FORK_REF"; fi
    return
  fi
  if [ "$IN_REPO" = 1 ]; then
    local cur
    cur=$(git -C "$REPO" branch --show-current 2> /dev/null) || cur=
    if [ -n "$cur" ]; then printf '%s\n' "$cur"; return; fi
  fi
  printf '%s\n' "$REPO_REF"
}

# The origin to fetch that ref from: the fork with --fork, else the repository the clone was made from.
target_url() {
  if [ "$FORK" = 1 ] && [ -z "${REPO_URL_SET:-}" ]; then printf '%s\n' "$FORK_URL"; return; fi
  printf '%s\n' "$REPO_URL"
}

# The clone's origin is target_url(): a fork deployed with --fork or --repo is fetched from there, not
# from the repository the clone was first made from.
sync_origin() {
  [ -d "$REPO/.git" ] || return 0
  local want have
  want=$(target_url)
  have=$(git -C "$REPO" remote get-url origin 2> /dev/null || true)
  [ "$have" = "$want" ] && return 0
  if [ -n "$have" ]; then
    step "origin: $have -> $want"
    git -C "$REPO" remote set-url origin "$want"
  else
    git -C "$REPO" remote add origin "$want"
  fi
}

# The commit a ref names, fetching it if the clone does not have it yet. Prints the commit.
resolve_ref() {  # ref
  local ref=$1 sha=
  case $ref in
    pr:*)
      # a pull request's head lives in the base repository, as refs/pull/<n>/head
      git -C "$REPO" fetch -q "$PR_REPO_URL" "pull/${ref#pr:}/head" ||
        die "no pull request ${ref#pr:} in $PR_REPO_URL (git fetch $PR_REPO_URL pull/${ref#pr:}/head)"
      sha=$(git -C "$REPO" rev-parse FETCH_HEAD)
      ;;
    *)
      # A branch or tag is fetched first: a local branch or a remote-tracking ref left by an earlier
      # run is stale, and deploying it would build the old code while the origin has moved on. A
      # commit that is not a branch cannot be fetched by name, so it is taken from the clone.
      if git -C "$REPO" fetch -q origin "$ref" 2> /dev/null; then
        sha=$(git -C "$REPO" rev-parse --verify -q FETCH_HEAD)
      fi
      if [ -z "$sha" ]; then
        # origin/<ref> before the local branch: origin is the fork, and a local branch of the same
        # name is what an earlier run left behind.
        sha=$(git -C "$REPO" rev-parse --verify -q "origin/$ref^{commit}") ||
          sha=$(git -C "$REPO" rev-parse --verify -q "$ref^{commit}") ||
          die "no ref '$ref' in $REPO_URL"
        warn "could not fetch $ref from $REPO_URL: using $sha from the clone, which may be stale"
      fi
      ;;
  esac
  printf '%s\n' "$sha"
}

phase_clone() {
  mkdir -p "$WORK"
  local ref sha
  ref=$(target_ref)

  if [ "$IN_REPO" = 1 ]; then
    # deploy-switchless.sh is in the clone: the clone is the repo, and it is not ours to move. Only a
    # ref other than what it has checked out needs a checkout (a pull request, a commit, --fork).
    if [ "$ref" = "$(git -C "$REPO" branch --show-current 2> /dev/null)" ]; then
      step "using the clone deploy-switchless.sh is in: $REPO"
      step "$(git -C "$REPO" --no-pager log --oneline -1)"
      [ -d "$SRC" ] || die "$SRC is not there: is $REPO the recipe's clone?"
      return 0
    fi
    sync_origin
    step "fetching $REPO_REF"
    git -C "$REPO" fetch -q origin "$REPO_REF"
  else
    if [ ! -d "$REPO/.git" ]; then
      step "cloning $REPO_URL into $REPO"
      git clone -q "$REPO_URL" "$REPO"
    fi
    sync_origin
    step "fetching $REPO_REF"
    git -C "$REPO" fetch -q origin "$REPO_REF"
  fi
  sha=$(resolve_ref "$ref")

  if [ "$ref" = "$REPO_REF" ]; then
    step "checking out $REPO_REF ($sha)"
    git -C "$REPO" checkout -q -B "$REPO_REF" "$sha"
    # a clone deploy-switchless.sh is in is the user's: only a ref it checked out itself is reset
    [ "$IN_REPO" = 1 ] || git -C "$REPO" reset -q --hard "$sha"
  else
    # a pull request or another ref: merge it with REPO_REF, so the build has both
    local base branch
    base=$(git -C "$REPO" rev-parse "origin/$REPO_REF")
    branch="deploy-$(printf '%s' "$ref" | tr -c 'A-Za-z0-9._-' '-')"
    step "checking out $ref ($sha), merged with $REPO_REF ($base)"
    git -C "$REPO" checkout -q -B "$branch" "$sha"
    if ! git -C "$REPO" merge-base --is-ancestor "$base" "$sha"; then
      if git -C "$REPO" merge -q --no-edit -m "deploy-switchless.sh: merge $REPO_REF into $ref" "$base" > /dev/null 2>&1; then
        step "merged $REPO_REF into $ref"
      else
        git -C "$REPO" merge --abort 2> /dev/null || true
        die "$ref does not merge cleanly with $REPO_REF: resolve it by hand in $REPO, or deploy $REPO_REF"
      fi
    fi
  fi
  [ -d "$SRC" ] || die "$SRC is not there: is $ref the right ref?"
  step "$(git -C "$REPO" --no-pager log --oneline -1)"
}

# The recipe's commit as it is on disk now, and as the running service was built from (service.sh
# freezes the binaries and the fleet file into SERVICE_DIR/bin, and writes the commit beside them).
built_commit() {
  local f
  f=$(local_path "$FLEET_SERVICE_DIR")/bin/build.commit
  [ -r "$f" ] && cat "$f"
}

# 1 if the clone is at a different commit than the one the service was built from
recipe_changed() {
  local now built
  now=$(git -C "$REPO" rev-parse HEAD 2> /dev/null) || return 1
  built=$(built_commit)
  [ -n "$built" ] || return 0
  [ "$now" != "$built" ]
}

phase_deps() {
  local missing=()
  command -v cmake > /dev/null || missing+=(cmake)
  command -v g++ > /dev/null || missing+=(g++)
  command -v cargo > /dev/null || missing+=(cargo)
  command -v python3 > /dev/null || missing+=(python3)
  dpkg -s libibverbs-dev > /dev/null 2>&1 || missing+=(libibverbs-dev)
  [ -x /usr/local/cuda-13.0/bin/nvcc ] || missing+=(cuda-13.0)

  if [ ${#missing[@]} -gt 0 ]; then
    if [ "$INSTALL_DEPS" = 1 ]; then
      log "installing: ${missing[*]}"
      sudo apt-get update -qq
      sudo apt-get install -y build-essential cmake libibverbs-dev python3 curl git
      if ! command -v cargo > /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
      fi
      [ -x /usr/local/cuda-13.0/bin/nvcc ] ||
        die "CUDA 13.0 is not in /usr/local/cuda-13.0: install it (the engine's CMakeLists.txt uses its nvcc)"
    else
      die "missing: ${missing[*]} (re-run with INSTALL_DEPS=1, or install them by hand)"
    fi
  fi

  local cm
  cm=$(cmake --version | head -1 | awk '{print $3}')
  awk -v v="$cm" 'BEGIN{split(v,a,"."); exit !(a[1]*100+a[2] >= 328)}' ||
    die "cmake $cm is older than 3.28"
  local py
  py=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [ "$py" = 3.12 ] ||
    warn "python3 is $py: make_hashmap_stdlib.py makes the validated map with 3.12 (Unicode 15.0)"
  step "cmake $cm, $(g++ --version | head -1), $(cargo --version), python $py"
}

# The hf CLI, whichever name the installed huggingface_hub provides.
hf_cli() {
  command -v hf || command -v huggingface-cli
}

phase_checkpoint() {
  local f have=1
  for f in model.safetensors.index.json tokenizer.json inference/config.json; do
    [ -f "$CKPT/$f" ] || have=0
  done
  if [ "$have" = 1 ]; then
    local n
    n=$(find "$CKPT" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
    step "checkpoint already at $CKPT ($n shards)"
    [ "$n" -ge 1 ] || die "$CKPT has no shards"
    # The encoder imports the checkpoint's own inference/vision.py and inference/image_processor.py
    # (docs/vision.md). A checkpoint downloaded before VISION was on has neither, and they are a few
    # KB: fetch them rather than re-download 475 GB.
    [ "$VISION" = 1 ] && vision_ensure_imports
    return 0
  fi

  if ! command -v hf > /dev/null && ! command -v huggingface-cli > /dev/null; then
    log "installing huggingface_hub (the hf CLI)"
    if ! python3 -m pip --version > /dev/null 2>&1; then
      [ "$INSTALL_DEPS" = 1 ] || die "no pip: re-run with INSTALL_DEPS=1, or install python3-pip"
      sudo apt-get install -y python3-pip
    fi
    # Ubuntu 24.04 marks the system Python externally managed (PEP 668)
    python3 -m pip install --user -q -U "huggingface_hub[hf_transfer]" 2> /dev/null ||
      python3 -m pip install --user --break-system-packages -q -U "huggingface_hub[hf_transfer]" ||
      die "pip install huggingface_hub failed"
    export PATH=$HOME/.local/bin:$PATH
  fi
  local hf
  hf=$(hf_cli) || die "no hf CLI after installing huggingface_hub"

  need_space "$(dirname "$CKPT")" 520 "checkpoint"
  log "downloading $CKPT_REPO @ $CKPT_REV into $CKPT (475 GB; it resumes if interrupted)"
  # The encoder's imports are in the same repo, so they come with the same download (a few KB).
  local inc=(--include "model-*.safetensors" --include "model.safetensors.index.json"
    --include "tokenizer.json" --include "tokenizer_config.json" --include "config.json"
    --include "chat_template.jinja" --include "inference/config.json")
  [ "$VISION" = 1 ] && inc+=(--include "inference/vision.py" --include "inference/image_processor.py")
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf" download "$CKPT_REPO" --revision "$CKPT_REV" --local-dir "$CKPT" \
    "${inc[@]}" ||
    die "the checkpoint download failed"
  [ -f "$CKPT/model.safetensors.index.json" ] || die "$CKPT: no model.safetensors.index.json"
  [ -f "$CKPT/tokenizer.json" ] || die "$CKPT: no tokenizer.json"
  [ -f "$CKPT/inference/config.json" ] || die "$CKPT: no inference/config.json"
  if [ "$VISION" = 1 ]; then
    [ -f "$CKPT/inference/vision.py" ] || die "$CKPT: no inference/vision.py (the encoder imports it)"
    [ -f "$CKPT/inference/image_processor.py" ] || die "$CKPT: no inference/image_processor.py"
  fi
  step "checkpoint complete: $(find "$CKPT" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l) shards"
}

# dsfront's tokenizer, kept outside the checkpoint so that KEEP_CHECKPOINT=0 does not take it
TOKENIZER=$WORK/tokenizer.json
keep_tokenizer() {
  [ -f "$CKPT/tokenizer.json" ] || die "$CKPT: no tokenizer.json"
  cp -f "$CKPT/tokenizer.json" "$TOKENIZER"
  step "tokenizer: $TOKENIZER"
}

phase_build() {
  log "building the engine (sm_121a) and the front-end"
  cmake -S "$SRC/engine" -B "$SRC/engine/build"
  cmake --build "$SRC/engine/build" -j "$JOBS"
  (cd "$SRC/frontend" && cargo build --release -j "$JOBS")
  [ -x "$SRC/engine/build/server/dsgb10_rank" ] || die "no engine/build/server/dsgb10_rank"
  [ -x "$SRC/frontend/target/release/dsfront" ] || die "no frontend/target/release/dsfront"
  step "engine/build/server/dsgb10_rank and frontend/target/release/dsfront are built"
}

phase_weights() {
  load_fleet
  local B=$SRC/engine/build/weights r out dst rel
  need_bins "$B/dsw_convert" "$B/dsw_verify"
  mkdir -p "$OUT"
  need_space "$OUT" 90 "staging"

  log "the partition, from the checkpoint headers alone"
  for r in 0 1 2 3; do "$B/dsw_convert" --ckpt "$CKPT" --rank "$r" --plan; done

  for r in 0 1 2 3; do
    dst=$(fleet_rank "$(fleet_get "RANK${r}_WEIGHTS")" "$r")
    if [ "$r" = 0 ]; then
      # rank 0 writes its file where it belongs: no 78 GB copy across filesystems
      out=$(local_path "$dst")
      if [ -f "$out" ]; then
        step "rank 0: $out is already there"
        continue
      fi
      mkdir -p "$(dirname "$out")"
      need_space "$(dirname "$out")" 80 "rank 0's weights"
      log "rank 0: cutting $out"
      "$B/dsw_convert" --ckpt "$CKPT" --rank 0 --out "$out"
      step "rank 0: verifying every byte against the checkpoint"
      "$B/dsw_verify" --ckpt "$CKPT" --file "$out" --threads 4
      continue
    fi
    out=$OUT/rank$r.dsw
    if [ "$(run_on "$r" "stat -c %s $(remote_path "$dst") 2>/dev/null || echo 0")" != 0 ]; then
      step "rank $r: $dst is already there"
      continue
    fi
    log "rank $r: cutting $out"
    "$B/dsw_convert" --ckpt "$CKPT" --rank "$r" --out "$out"
    step "rank $r: verifying every byte against the checkpoint"
    "$B/dsw_verify" --ckpt "$CKPT" --file "$out" --threads 4
    rel=$(remote_path "$dst")
    run_on "$r" "mkdir -p \$(dirname $rel)"
    copy_verified "$out" "$r" "$rel"
    rm -f "$out"
  done
}

phase_engram() {
  load_fleet
  local P=$SRC/engine/build/engram/engram_pack r dir rel f out
  need_bins "$P"
  mkdir -p "$OUT"
  need_space "$OUT" 55 "staging"

  for r in 0 1 2 3; do
    dir=$(fleet_rank "$(fleet_get "RANK${r}_ENGRAM")" "$r")
    if [ "$r" = 0 ]; then
      out=$(local_path "$dir")
      if [ -f "$out/engram-l1-r0of4.bin" ] && [ -f "$out/engram-l14-r0of4.bin" ]; then
        step "rank 0: $out is already there"
        continue
      fi
      mkdir -p "$out"
      need_space "$out" 52 "rank 0's Engram shards"
      log "rank 0: packing its two Engram shards (50.7 GB) into $out"
      "$P" "$CKPT" "$out" 0
      step "rank 0: $out"
      continue
    fi
    rel=$(remote_path "$dir")
    if [ "$(run_on "$r" "stat -c %s $rel/engram-l14-r${r}of4.bin 2>/dev/null || echo 0")" != 0 ]; then
      step "rank $r: $dir is already there"
      continue
    fi
    log "rank $r: packing its two Engram shards (50.7 GB)"
    "$P" "$CKPT" "$OUT" "$r"
    run_on "$r" "mkdir -p $rel"
    for f in "engram-l1-r${r}of4.bin" "engram-l14-r${r}of4.bin"; do
      [ -f "$OUT/$f" ] || die "engram_pack wrote no $OUT/$f"
      copy_verified "$OUT/$f" "$r" "$rel/$f"
      rm -f "$OUT/$f"
    done
  done

  log "the Engram hash map (one for all four ranks)"
  python3 "$SRC/engine/engram/py/make_hashmap_stdlib.py" "$CKPT" "$OUT/engram_hashmap.bin"
  for r in 0 1 2 3; do
    dst=$(fleet_rank "$(fleet_get "RANK${r}_HASHMAP")" "$r")
    if [ "$r" = 0 ]; then
      mkdir -p "$(dirname "$(local_path "$dst")")"
      cp -f "$OUT/engram_hashmap.bin" "$(local_path "$dst")"
      step "rank 0: $(local_path "$dst")"
    else
      rel=$(remote_path "$dst")
      run_on "$r" "mkdir -p \$(dirname $rel)"
      copy_verified "$OUT/engram_hashmap.bin" "$r" "$rel"
    fi
  done
  rm -f "$OUT/engram_hashmap.bin"

  keep_tokenizer
  if [ "$KEEP_CHECKPOINT" = 0 ]; then
    log "deleting the checkpoint at $CKPT (KEEP_CHECKPOINT=0)"
    rm -rf "$CKPT"
  fi
}

phase_gate() {
  load_fleet
  # the fork's learned confidence gate: the same file at the same path on all four boxes
  local f=$SRC/tools/stack4/cfmlp_cml.bin rel=$FLEET_BASE/cfmlp_cml.bin
  [ -f "$f" ] || die "no $f"
  cp -f "$f" "$(local_path "$rel")"
  step "rank 0: $(local_path "$rel")"
  local r
  for r in 1 2 3; do
    if [ "$(run_on "$r" "stat -c %s $(remote_path "$rel") 2>/dev/null || echo 0")" = "$(stat -c %s "$f")" ]; then
      step "rank $r: $rel is already there"
    else
      copy_verified "$f" "$r" "$(remote_path "$rel")"
    fi
  done
}

phase_verify() {
  load_fleet
  need_bins "$SRC/engine/build/comm/ring_test"
  ensure_ring_dirs
  if (cd "$SRC" && engine/server/service.sh status > /dev/null 2>&1); then
    warn "the service is serving: ring_test waits for the ring (up to LOCK_WAIT=7200 s). Stop it first: ./deploy-switchless.sh stop"
  fi
  log "checking the ring's links and collectives (52 s; each rank prints 'passed')"
  (cd "$SRC" && engine/server/ring.sh engine/comm/run4.sh engine/build/comm/ring_test)
}

# The front-end's addresses. The recipe refuses 0.0.0.0 (HTTP stays off public interfaces), so the
# default is loopback plus this box's address on the network the workers are reached on: that is the
# one a client elsewhere uses. DSGB10_ADDR overrides it (space or comma separated IPv4 addresses).
front_addrs() {
  if [ -n "$DSGB10_ADDR" ]; then
    printf '%s\n' "$DSGB10_ADDR"
    return 0
  fi
  local a
  a=$(ip -4 -o addr show scope global 2> /dev/null |
    awk -v w="${WORKERS[1]}" '
      { split($4, c, "."); ip = $4; sub(/\/.*/, "", ip) }
      BEGIN { split(w, wc, ".") }
      # the interface that reaches the first worker: same first three octets
      wc[1] != "" && c[1] == wc[1] && c[2] == wc[2] && c[3] == wc[3] { print ip; exit }')
  if [ -z "$a" ]; then
    a=$(ip -4 -o route get "${WORKERS[1]}" 2> /dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')
  fi
  if [ -z "$a" ]; then
    warn "no address on the network that reaches ${WORKERS[1]}: serving on 127.0.0.1 only (set DSGB10_ADDR=...)"
    printf '127.0.0.1\n'
    return 0
  fi
  printf '127.0.0.1 %s\n' "$a"
}

# The settings the running service was started with (service.sh writes them to run/settings)
running_setting() {  # KEY
  local f
  f=$(local_path "$FLEET_SERVICE_DIR")/run/settings
  [ -r "$f" ] || return 1
  sed -n "s/^declare -[^ ]* $1=\"\(.*\)\"$/\1/p" "$f"
}

# The engine's and the front-end's arguments, and the addresses, as this script would start the service
# with them. serve and redeploy both use these, so a restart keeps the settings the service had.
serve_settings() {
  local addrs
  addrs=$(front_addrs)
  local eargs="--max-ctx 1048576 --slots 16 --big-slots 4"
  eargs+=" --setenv DSGB10_BUDGET=1"
  [ "$LOSSLESS" = 1 ] || eargs+=" --setenv DSGB10_OEA=4,12,9"
  eargs+=" --setenv DSGB10_DRAFT1=6 --setenv DSGB10_WIDE=64"
  eargs+=" --setenv DSGB10_WIDE_COST=24,2.2,57,1.1"
  eargs+=" --draft-head4 --draft-markov8 --setenv DSGB10_HEAD4_C1=1"
  eargs+=" --setenv DSGB10_CF_MLP=$FLEET_BASE/cfmlp_cml.bin --setenv DSGB10_CF_ALPHA=1.0"
  eargs+=" --setenv DSGB10_CERT=1"
  # DSGB10_RING_DIRECT=2 is for a switch (every rank one hop from every rank); this ring is switchless
  local fargs="--tokenizer $TOKENIZER --model-name $MODEL_NAME --encoding $ENCODING"
  [ "$STREAM_TOOL_ARGS" = 1 ] && fargs+=" --stream-tool-args"
  # No --vision-addr here: service.sh adds it when it is given the encoder's command (DSGB10_VISION_CMD)
  # and refuses it in DSGB10_FRONT_ARGS.
  printf '%s\n%s\n%s\n' "$addrs" "$eargs" "$fargs"
}

# service.sh start with this script's settings (it reads them from the environment)
service_start() {  # the encoder's command, from vision_cmd
  local addrs eargs fargs
  { read -r addrs; read -r eargs; read -r fargs; } < <(serve_settings)
  (cd "$SRC" &&
    DSGB10_KEY_FILE=$WORK/api-key DSGB10_ADDR="$addrs" DSGB10_PORT="$DSGB10_PORT" \
      DSGB10_PROD="$DSGB10_PROD" \
      DSGB10_ENGINE_ARGS="$eargs" DSGB10_FRONT_ARGS="$fargs" \
      DSGB10_VISION_CMD="$1" DSGB10_VISION_PORT="$VISION_PORT" \
      engine/server/service.sh start)
}

# ---------------------------------------------------------------- the image encoder (docs/vision.md)

# The encoder turns images into the LLM's embedding rows on rank 0's loopback. dsfront refuses every
# image part without --vision-addr, and service.sh adds that when it is given the encoder's command
# (DSGB10_VISION_CMD, PR #15): it starts the encoder before the engine, starts it again when it exits,
# stops it with the service by process group, and reports it in status. So this script only has to
# name the command.
#
# The command runs in the foreground, from the recipe's directory, and service.sh appends
# --host 127.0.0.1 --port $VISION_PORT. It is a container's client (no -d, with --init: without it the
# encoder is the container's process 1, which ignores SIGTERM and outlives the stop). The image is the
# public base of the recipe's own kit image, which already has torch, Pillow, safetensors and PyAV, so
# nothing is built here.

# The encoder imports the checkpoint's own inference/vision.py and inference/image_processor.py. A
# checkpoint downloaded before VISION was on has neither, and they are a few KB: fetch them rather
# than make the operator re-download 475 GB.
#
# Everything it prints goes to stderr: vision_cmd() is captured with $(...), so anything on stdout
# would become part of the command service.sh runs.
vision_ensure_imports() {
  local f missing=()
  for f in vision.py image_processor.py; do
    [ -f "$CKPT/inference/$f" ] || missing+=("inference/$f")
  done
  [ ${#missing[@]} -gt 0 ] || return 0
  [ -f "$CKPT/inference/config.json" ] ||
    die "$CKPT is not the checkpoint (no inference/config.json): run --phase checkpoint"
  local hf
  hf=$(hf_cli) ||
    die "$CKPT/inference is missing ${missing[*]}: install huggingface_hub (INSTALL_DEPS=1) or run --phase checkpoint"
  local inc=()
  for f in "${missing[@]}"; do inc+=(--include "$f"); done
  log "fetching the encoder's imports from $CKPT_REPO: ${missing[*]}" >&2
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf" download "$CKPT_REPO" --revision "$CKPT_REV" --local-dir "$CKPT" \
    "${inc[@]}" > /dev/null || die "could not fetch ${missing[*]} from $CKPT_REPO"
  for f in "${missing[@]}"; do
    [ -f "$CKPT/$f" ] || die "$CKPT: no $f after the download"
  done
  step "the encoder's imports are in $CKPT/inference" >&2
}

# Prints the command and nothing else: service.sh runs what this prints, so a stray line of output
# becomes part of the command (that is how "exec ==> fetching ..." happened).
#
# --memory is a cgroup limit on the container's pages, and on a GB10 the GPU's memory *is* host memory
# (unified LPDDR): a CUDA allocation is a page of the same cgroup. So --memory caps the encoder's GPU
# memory too, and a limit that looks generous for a 0.95 GiB model is not: torch's CUDA context, the
# CPU-side state dict it loads before moving it, and the ViT's first allocation all count against it.
# At 6g the encoder died in load() with "CUDA error: out of memory" on a 1 MB nn.Linear, while rank 0
# had 6.24 GiB of device memory free. 32g leaves room for the context and the load without letting a
# runaway encoder take rank 0's box.
vision_cmd() {
  [ "$VISION" = 1 ] || return 0
  command -v docker > /dev/null || die "VISION=1 needs docker (the encoder runs in $VISION_IMAGE)"
  [ -f "$SRC/tools/vision/vision_encoder.py" ] || die "no $SRC/tools/vision/vision_encoder.py (run --phase clone)"
  vision_ensure_imports
  printf 'docker run --rm --init --gpus all --net=host --memory=%s -e VISION_PRELOAD=1' "$VISION_MEM"
  printf ' -v %s:/model:ro -v %s/tools/vision:/app:ro --entrypoint python3 %s' "$CKPT" "$SRC" "$VISION_IMAGE"
  printf ' /app/vision_encoder.py --model /model --max-images %s --max-rows %s' "$VISION_MAX_IMAGES" "$VISION_MAX_ROWS"
}

# The encoder needs torch, Pillow and safetensors in the image's python3 for images, and PyAV (`av`)
# for video. service.sh restarts a failing encoder after 2, 10, 60 and 300 s and then every 15
# minutes, so a wrong image is a service that answers 503 for images for ever with only
# "exited (rc 1)" in service.log. Check once, here, where the message can say what is missing.
#
# The image is pulled here too, not left to the encoder's first start: it is 13.55 GB, and inside the
# supervised start it is invisible (status says "not listening (it loads its weights first)" while it
# is really downloading) and a failed pull counts as an encoder exit, backing off to 15 minutes.
vision_preflight() {
  [ "$VISION" = 1 ] || return 0
  command -v docker > /dev/null || die "VISION=1 needs docker (the encoder runs in $VISION_IMAGE)"
  if ! docker image inspect "$VISION_IMAGE" > /dev/null 2>&1; then
    log "pulling $VISION_IMAGE (13.55 GB, once; the encoder runs in it)"
    docker pull "$VISION_IMAGE" || die "could not pull $VISION_IMAGE (set VISION_IMAGE, or VISION=0 for text only)"
    step "pulled $VISION_IMAGE"
  fi
  local out
  # images need these three; a missing one is fatal
  if ! out=$(docker run --rm --entrypoint python3 "$VISION_IMAGE" -c \
    'import torch, PIL, safetensors' 2>&1); then
    case $out in
      *"Cannot connect to the Docker daemon"* | *"permission denied"*) return 0 ;;
    esac
    warn "the encoder's image $VISION_IMAGE cannot import torch, PIL and safetensors:"
    printf '%s\n' "$out" | tail -3 | sed 's/^/    /' >&2
    warn "set VISION_IMAGE to an image that has them, or VISION=0 to serve text only"
    return 1
  fi
  # video needs PyAV; without it images still work, so this is a warning
  docker run --rm --entrypoint python3 "$VISION_IMAGE" -c 'import av' > /dev/null 2>&1 ||
    warn "$VISION_IMAGE has no PyAV (av): images work, video clips will be refused"
  # The imports are not enough: the encoder's first CUDA allocation is what fails when the container's
  # memory limit is too small for the context (see vision_cmd). Allocate one tensor under the same
  # --memory the encoder gets, so that failure is reported here instead of as a 503 for ever.
  if ! out=$(docker run --rm --memory="$VISION_MEM" --gpus all --entrypoint python3 "$VISION_IMAGE" -c \
    'import torch; torch.empty(1024, 1024, device="cuda"); print("ok")' 2>&1); then
    case $out in
      *"Cannot connect to the Docker daemon"* | *"permission denied"* | *"could not select device"*) return 0 ;;
    esac
    warn "the encoder cannot allocate GPU memory in $VISION_IMAGE with --memory=$VISION_MEM:"
    printf '%s\n' "$out" | tail -3 | sed 's/^/    /' >&2
    warn "raise VISION_MEM (on a GB10 the GPU's memory is host memory, so --memory caps it too)"
    return 1
  fi
  return 0
}

# An older version of this script ran the encoder as a detached container of its own. service.sh owns
# the encoder now, so one left behind would hold rank 0's GPU and the port: remove it.
VISION_LABEL=fi.qumio.deploy-switchless.vision

vision_container() {  # the container that older version started, if it is still there
  command -v docker > /dev/null || return 1
  local id
  id=$(docker ps -aq --filter "label=$VISION_LABEL=1" --filter "name=^/$VISION_NAME\$" 2> /dev/null) || return 1
  [ -n "$id" ] || return 1
  printf '%s\n' "$id"
}

vision_stop() {
  local id
  id=$(vision_container) || return 0
  docker rm -f "$id" > /dev/null 2>&1 || true
  step "removed the image encoder container an older version of this script left ($VISION_NAME)"
}

phase_serve() {
  load_fleet
  ensure_ring_dirs
  need_bins "$SRC/engine/build/server/dsgb10_rank" "$SRC/frontend/target/release/dsfront"
  [ -f "$TOKENIZER" ] || keep_tokenizer
  local key=$WORK/api-key
  if [ ! -s "$key" ]; then
    (umask 077; openssl rand -hex 24 > "$key")
    step "wrote the API key to $key"
  fi
  chmod 600 "$key"

  local addrs eargs fargs vcmd
  { read -r addrs; read -r eargs; read -r fargs; } < <(serve_settings)
  local first=${addrs%% *}

  # Resolved first, in this shell: die() inside a command substitution would only end the subshell, and
  # the service would start without the encoder. Before anything is stopped, so a failure here leaves
  # a running service alone.
  vcmd=$(vision_cmd) || exit 1
  vision_preflight || exit 1

  # service.sh owns the encoder (DSGB10_VISION_CMD): it starts it before the engine and stops it with
  # the service. Only a container an older version of this script left behind is removed here.
  vision_stop

  # Idempotent: a service already serving with these settings is left alone (a restart drops the
  # requests in flight); one serving with other settings is restarted, so that the settings here win.
  local serving=0
  if service_cmd status > /dev/null 2>&1; then serving=1; fi
  if [ "$serving" = 1 ]; then
    local rp ra rf rv
    rp=$(running_setting PORT) ra=$(running_setting ADDR) rf=$(running_setting FARGS) rv=$(running_setting VCMD)
    if [ "$rp" = "$DSGB10_PORT" ] && [ "$ra" = "$addrs" ] && [ "$rf" = "$fargs" ] &&
      [ "$rv" = "$vcmd" ]; then
      step "already serving on $addrs port $DSGB10_PORT: not restarting"
    else
      log "restarting: it serves on ${ra:-?} port ${rp:-?} with '${rf:-}', these settings say $addrs port $DSGB10_PORT with '$fargs'"
      service_cmd stop
      serving=0
    fi
  fi

  if [ "$serving" = 0 ]; then
    log "starting the service on $addrs port $DSGB10_PORT (about a minute)"
    service_start "$vcmd"
    # what this build is, so that a later run can tell whether the recipe moved (recipe_changed)
    mkdir -p "$(local_path "$FLEET_SERVICE_DIR")/bin"
    git -C "$REPO" rev-parse HEAD > "$(local_path "$FLEET_SERVICE_DIR")/bin/build.commit" 2> /dev/null || true
  fi

  (umask 077; { printf 'Authorization: Bearer '; cat "$key"; } > "$WORK/api-key.header")
  local url=http://$first:$DSGB10_PORT/v1
  step "GET $url/models"
  curl -s -m 10 "$url/models" -H @"$WORK/api-key.header" || die "the front-end does not answer"
  echo
  step "a chat completion"
  curl -s -m 120 "$url/chat/completions" -H @"$WORK/api-key.header" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with one word: pong"}],"max_tokens":20}'
  echo
  local a
  for a in $addrs; do
    [ "$a" = "$first" ] && continue
    step "also on http://$a:$DSGB10_PORT/v1 (from another box)"
  done
  log "serving. ./deploy-switchless.sh status | stop | logs"
}

# ---------------------------------------------------------------- the service afterwards

service_cmd() {
  load_fleet
  ensure_ring_dirs
  (cd "$SRC" && engine/server/service.sh "$@")
}

# stop and restart take the encoder with the service: service.sh owns it (DSGB10_VISION_CMD), so its
# stop reaches the encoder's process group. Only a container an older version of this script left
# behind is removed here.
# Through service_cmd, not service.sh directly: it loads the fleet, and service.sh resolves the
# service's directory from FLEET. Without it, stop looks in the recipe's own fleet/homelab.conf, finds
# no service, and says "not running" while a supervisor is alive.
service_stop() {
  service_cmd stop || true
  vision_stop
}

# Why the engine last died. service.sh logs one line per exit, "rank 0 exited (rc N) after T s: <the
# last 3 lines of rank.0.log>", and rotates rank.0.log on every start (KEEP=5), so the evidence is one
# restart away from being gone. This collects it while it is still there: the exit line, the engine's
# own last words from the current log and the rotated copies, the kernel's OOM verdict, and the
# restart count. Read-only: it changes nothing.
#
# The exit code is the diagnosis:
#   0    a clean stop (service.sh stop, a restart, a lend): not a crash
#   134  SIGABRT: the engine called die() (a CUDA_CHECK, a failed assertion) and printed why to stderr
#   137  SIGKILL: the kernel's OOM killer, or service.sh's own kill of a rank that did not start
#   139  SIGSEGV: a segfault
#   143  SIGTERM: service.sh stopped it
crash() {
  load_fleet
  local d rc line n=0
  d=$(local_path "$FLEET_SERVICE_DIR")/logs
  [ -d "$d" ] || die "no $d (the service has never run here)"

  log "the engine's exits, oldest first (service.log)"
  # "rank 0 exited (rc N)" and "rank 0 exited (rc N) after T s: ..."; the vision encoder's are not ours
  grep -n 'rank 0 exited (rc' "$d/service.log" 2> /dev/null | tail -n 10 || true
  line=$(grep 'rank 0 exited (rc' "$d/service.log" 2> /dev/null | tail -n 1) || line=
  if [ -z "$line" ]; then
    warn "no exit recorded in $d/service.log: the engine has not died since the log began"
  else
    rc=$(printf '%s\n' "$line" | sed -n 's/.*(rc \([0-9]*\)).*/\1/p')
    printf '\n'
    case $rc in
      0) log "the last exit was rc 0: a clean stop (stop, restart or a lend), not a crash" ;;
      134) log "the last exit was rc 134 (SIGABRT): the engine called die() and printed why to stderr" ;;
      137) log "the last exit was rc 137 (SIGKILL): the OOM killer, or service.sh killing a rank that did not start" ;;
      139) log "the last exit was rc 139 (SIGSEGV): a segfault" ;;
      143) log "the last exit was rc 143 (SIGTERM): service.sh stopped it" ;;
      *) log "the last exit was rc $rc" ;;
    esac
  fi

  # The engine's own last words. rank.0.log is rotated on every start, so the crash is in the current
  # file or in one of the copies; print the newest one that has anything in it.
  printf '\n'
  log "the engine's last words (rank.0.log, then its rotated copies)"
  for f in rank.0.log rank.0.log.1 rank.0.log.2 rank.0.log.3 rank.0.log.4 rank.0.log.5; do
    [ -s "$d/$f" ] || continue
    printf '\n--- %s (last 40 lines)\n' "$d/$f"
    tail -n 40 "$d/$f"
    n=$((n + 1))
    [ "$n" -lt 2 ] || break  # the current log and the one before it: enough, and not a wall of text
  done
  [ "$n" -gt 0 ] || warn "no rank.0.log in $d"

  # The kernel's verdict, when it was the kernel. dmesg needs root; say so rather than fail.
  printf '\n'
  log "the kernel's OOM verdict (dmesg)"
  if command -v dmesg > /dev/null 2>&1; then
    if dmesg 2> /dev/null | grep -iE 'oom|killed process|out of memory' | tail -n 10 | grep -q .; then
      dmesg 2> /dev/null | grep -iE 'oom|killed process|out of memory' | tail -n 10
    elif sudo -n dmesg 2> /dev/null | grep -iE 'oom|killed process|out of memory' | tail -n 10 | grep -q .; then
      sudo -n dmesg 2> /dev/null | grep -iE 'oom|killed process|out of memory' | tail -n 10
    else
      step "nothing in dmesg (or it needs root: sudo dmesg | grep -i oom)"
    fi
  else
    step "no dmesg on this box"
  fi

  printf '\n'
  log "restarts"
  cat "$(local_path "$FLEET_SERVICE_DIR")/run/restarts" 2> /dev/null || step "no run/restarts (the service has not restarted anything)"
  printf '\n'
  log "the front-end's view of the same moment (front.log)"
  tail -n 15 "$d/front.log" 2> /dev/null || step "no front.log"
  printf '\n'
  log "a client's retries cannot cover this: it retries for ~15 s, the engine takes ~10 s to restart"
  step "the request that died is in front.log, and its correlation_id in whatever sits in front of the ring"
}

# A new version of the recipe: fetch it, rebuild, restart. The checkpoint, the weight files and the
# Engram shards are not touched (they are the model, not the code), so this is minutes, not hours.
redeploy() {
  load_fleet
  local before after built
  before=$(git -C "$REPO" rev-parse HEAD 2> /dev/null || echo none)
  built=$(built_commit)
  log "fetching $(target_ref)"
  phase_clone
  after=$(git -C "$REPO" rev-parse HEAD)
  # rebuild when the fetch moved the clone, or when the service was built from another commit (a
  # --phase clone since then, or a first run that never recorded one)
  if [ "$before" = "$after" ] && [ "$built" = "$after" ] && [ "$FORCE" = 0 ]; then
    step "the recipe is already at $after and the service was built from it: nothing to rebuild (--force rebuilds anyway)"
  else
    if [ "$before" != "$after" ]; then
      step "the recipe moved: $before -> $after"
    elif [ "$built" != "$after" ]; then
      step "the service was built from ${built:-nothing}, the clone is at $after"
    fi
    log "rebuilding"
    phase_build
  fi
  log "restarting the service"
  # Resolved before the stop: a failure here must not leave the service down.
  local vcmd
  vcmd=$(vision_cmd) || exit 1
  vision_preflight || exit 1
  service_stop
  service_start "$vcmd" || die "the restart failed: ./deploy-switchless.sh logs"
  mkdir -p "$(local_path "$FLEET_SERVICE_DIR")/bin"
  printf '%s\n' "$after" > "$(local_path "$FLEET_SERVICE_DIR")/bin/build.commit"
  printf '%s\n' "$(target_ref)" > "$(local_path "$FLEET_SERVICE_DIR")/bin/build.ref"
  log "redeployed $(target_ref) at $after. ./deploy-switchless.sh status | logs"
}

# ---------------------------------------------------------------- main

# What the service is doing right now, in a few lines (for the bare ./deploy-switchless.sh)
summary() {
  local st built p done=() todo=()
  for p in clone fleet preflight net deps checkpoint build weights engram gate verify serve; do
    if [ -f "$STATE/$p.done" ]; then done+=("$p"); else todo+=("$p"); fi
  done

  if [ -r "$FLEET" ] && [ -r "$SRC/engine/server/fleet.sh" ]; then
    load_fleet
    if (cd "$SRC" && engine/server/service.sh status > /dev/null 2>&1); then
      st=serving
    else
      st=stopped
    fi
    printf '  service:    %s' "$st"
    if [ "$st" = serving ]; then
      printf ' on %s port %s' "$(running_setting ADDR)" "$(running_setting PORT)"
    fi
    printf '\n'
    built=$(built_commit) || built=
    printf '  built from: %s\n' "${built:-nothing recorded}"
  else
    printf '  service:    not deployed yet (no fleet file)\n'
  fi

  if [ -d "$REPO/.git" ]; then
    printf '  clone:      %s\n' "$(git -C "$REPO" --no-pager log --oneline -1 2> /dev/null)"
    local bref
    bref=$(cat "$(local_path "$FLEET_SERVICE_DIR")/bin/build.ref" 2> /dev/null) || bref=
    [ -z "$bref" ] || printf '  built ref:  %s\n' "$bref"
  else
    printf '  clone:      not cloned yet\n'
  fi
  printf '  phases:     %s of 12 done' "${#done[@]}"
  [ ${#todo[@]} -eq 0 ] || printf ', to do: %s' "${todo[*]}"
  printf '\n'
}

# The bare ./deploy-switchless.sh: what this is, what it can do, and where things stand
instructions() {
  usage 0
  printf '\n'
  summary
  printf '\n'
  printf '  ./deploy-switchless.sh deploy     run every phase (the first run: hours, 475 GB of download)\n'
  printf '  ./deploy-switchless.sh start      start the service (or leave a running one alone)\n'
  printf '  ./deploy-switchless.sh stop       stop it\n'
  printf '  ./deploy-switchless.sh restart    stop, then start with this script'"'"'s settings\n'
  printf '  ./deploy-switchless.sh status     what it is doing, and the boxes\n'
  printf '  ./deploy-switchless.sh logs       the last lines of its logs\n'
  printf '  ./deploy-switchless.sh crash      why the engine last died: exit code, its last words, OOM\n'
  printf '  ./deploy-switchless.sh redeploy   a new version of the recipe: fetch, rebuild, restart\n'
  printf '  ./deploy-switchless.sh redeploy --fork   the branch with PRs #2, #3, #7 and #15\n'
  printf '  ./deploy-switchless.sh redeploy --fork --no-stream-args   ... without PR #7\n'
  printf '  ./deploy-switchless.sh redeploy --no-vision        ... without the image encoder\n'
  printf '\n'
}

main() {
  PHASES=(clone fleet preflight net deps checkpoint build weights engram gate verify serve)
  ONLY=() SKIP=() FORCE=0 CMD=

  while [ $# -gt 0 ]; do
    case $1 in
      --phase) ONLY+=("$2"); shift 2 ;;
      --skip) SKIP+=("$2"); shift 2 ;;
      --force) FORCE=1; shift ;;
      --pr) PR=$2; shift 2 ;;
      --ref) REF=$2; shift 2 ;;
      --repo) REPO_URL=$2; REPO_URL_SET=1; shift 2 ;;
      --fork) FORK=1; shift ;;
      --no-fork) FORK=0; shift ;;
      --stream-args) STREAM_ARGS=1; STREAM_TOOL_ARGS=1; shift ;;
      --no-stream-args) STREAM_ARGS=0; STREAM_TOOL_ARGS=0; shift ;;
      --vision) VISION=1; shift ;;
      --no-vision) VISION=0; shift ;;
      --list) printf '%s\n' "${PHASES[@]}"; exit 0 ;;
      -h | --help) usage 0; exit 0 ;;
      deploy | all) CMD=deploy; shift ;;
      start | stop | restart | status | logs | crash | redeploy) CMD=$1; shift ;;
      *) usage 2; exit 2 ;;
    esac
  done

  mkdir -p "$STATE"

  if [ -n "$CMD" ]; then
    case $CMD in
      deploy) ;;  # the phase loop below
      start) phase_serve; exit 0 ;;
      restart)
        load_fleet
        ensure_ring_dirs
        service_stop
        phase_serve
        exit 0
        ;;
      redeploy) redeploy; exit 0 ;;
      logs)
        service_cmd status || true
        # service.sh keeps its logs under SERVICE_DIR/logs, not LOG_DIR
        local d
        d=$(local_path "$FLEET_SERVICE_DIR")/logs
        for f in service.log front.log rank.0.log vision.log; do
          [ -f "$d/$f" ] || continue
          printf '\n--- %s\n' "$d/$f"
          tail -n 20 "$d/$f"
        done
        exit 0
        ;;
      status) service_cmd status; exit $? ;;
      crash) crash; exit 0 ;;
      stop) service_stop; exit 0 ;;
      *) service_cmd "$CMD"; exit 0 ;;
    esac
  fi

  # no command and no phase: say what this is and where things stand, change nothing
  if [ -z "$CMD" ] && [ ${#ONLY[@]} -eq 0 ] && [ ${#SKIP[@]} -eq 0 ] && [ "$FORCE" = 0 ] && [ "$REDEPLOY" = 0 ]; then
    instructions
    exit 0
  fi

  in_list() {  # x list...
    local x=$1 y
    shift
    for y in "$@"; do
      if [ "$x" = "$y" ]; then return 0; fi
    done
    return 1
  }

  for p in "${ONLY[@]}" "${SKIP[@]}"; do
    in_list "$p" "${PHASES[@]}" || die "no phase '$p' (--list)"
  done

  # REDEPLOY=1: a new version of the recipe means the clone, the build and the service are redone,
  # whatever the markers say. The model's files (checkpoint, weights, Engram) are left alone.
  if [ "$REDEPLOY" = 1 ]; then
    for p in clone build serve; do
      if [ ${#ONLY[@]} -gt 0 ] && ! in_list "$p" "${ONLY[@]}"; then continue; fi
      if in_list "$p" "${SKIP[@]}"; then continue; fi
      rm -f "$STATE/$p.done"
    done
    FORCE=1
  fi

  for p in "${PHASES[@]}"; do
    if [ ${#ONLY[@]} -gt 0 ] && ! in_list "$p" "${ONLY[@]}"; then continue; fi
    if in_list "$p" "${SKIP[@]}"; then log "$p: skipped"; continue; fi
    # A phase named with --phase runs even if its marker says it finished: asking for it is the
    # request. serve is never skipped either, because it is idempotent itself (it leaves a service
    # that already serves with these settings alone), so `deploy` always ends by making sure the
    # service is up.
    if [ "$p" != serve ] && [ ${#ONLY[@]} -eq 0 ] && [ -f "$STATE/$p.done" ] && [ "$FORCE" = 0 ]; then
      log "$p: done already ($STATE/$p.done)"
      continue
    fi
    log "$p"
    "phase_$p"
    date '+%F %T' > "$STATE/$p.done"
  done

  log "done"
}

# DEPLOY_LIB=1 sources the functions without running anything (the tests do)
[ "${DEPLOY_LIB:-0}" = 1 ] || main "$@"
