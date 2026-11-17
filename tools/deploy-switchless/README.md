# deploy-switchless.sh: the whole ring from one command

[deploy-switchless.sh](deploy-switchless.sh) takes a fresh 4-box switchless ring to a serving endpoint,
and keeps it there. It is the manual procedure in
[the recipe's README](../../ds41f-4xgx10-switchless/README.md#how-to-use) and
[operations.md](../../ds41f-4xgx10-switchless/docs/operations.md) written down once, with the order,
the checks and the resumability that a hand-run does not have.

Run it on rank 0, the box with passwordless SSH to the other three.

```bash
./tools/deploy-switchless/deploy-switchless.sh              # what this is, what it can do, where things stand
./tools/deploy-switchless/deploy-switchless.sh deploy       # every phase, in order (first run: hours, 475 GB)
./tools/deploy-switchless/deploy-switchless.sh status       # what the service is doing, and the boxes
./tools/deploy-switchless/deploy-switchless.sh logs         # the last lines of its logs
```

- [The network](#the-network): the cabling, the addresses, and how to set them up
- [The deployment](#the-deployment): the twelve phases, in order
- [Where it puts things](#where-it-puts-things)
- [What it deploys](#what-it-deploys)
- [The service afterwards](#the-service-afterwards)
- [Settings](#settings)
- [Images](#images)
- [The four pull requests](#the-four-pull-requests)
- [What it does not do](#what-it-does-not-do)

## The network

Four GB10 boxes in a **directed ring**: each box has one ConnectX-7 with two QSFP ports, port `f0`
facing the next rank and port `f1` facing the previous one. There is no switch. Rank 3's `f0` cabled
back to rank 0's `f1` closes the ring.

```
                        ┌──────────────────────────────────────────────┐
                        │                                              │
                        │  L3: 200G, 10.141.80.0/24 + 10.141.81.0/24   │
                        │                                              │
                        ▼                                              │
        ┌───────────────────────────────┐              ┌───────────────────────────────┐
        │  rank 0        qspark01       │              │  rank 3        qspark04       │
        │  ASUS Ascent GX10             │              │  NVIDIA DGX Spark             │
        │  192.168.2.101  (management)  │              │  192.168.2.104  (management)  │
        │                               │              │                               │
        │  f0 ──► 10.112.80.1/24        │              │  f0 ──► 10.141.80.1/24        │
        │         10.112.81.1/24        │              │         10.141.81.1/24        │
        │  f1 ◄── 10.141.80.2/24        │              │  f1 ◄── 10.134.80.2/24        │
        │         10.141.81.2/24        │              │         10.134.81.2/24        │
        └───────────────┬───────────────┘              └───────────────▲───────────────┘
                        │                                              │
      L0: 100G          │                                              │        L2: 200G
      10.112.80.0/24    │                                              │        10.134.80.0/24
      10.112.81.0/24    │                                              │        10.134.81.0/24
                        │                                              │
                        ▼                                              │
        ┌───────────────────────────────┐              ┌───────────────────────────────┐
        │  rank 1        qspark02       │              │  rank 2        qspark03       │
        │  ASUS Ascent GX10             │   L1: 200G   │  NVIDIA DGX Spark             │
        │  192.168.2.102  (management)  │─────────────►│  192.168.2.103  (management)  │
        │                               │ 10.123.80.0/24│                              │
        │  f0 ──► 10.123.80.1/24        │ 10.123.81.0/24│  f0 ──► 10.134.80.1/24        │
        │         10.123.81.1/24        │              │         10.134.81.1/24        │
        │  f1 ◄── 10.112.80.2/24        │              │  f1 ◄── 10.123.80.2/24        │
        │         10.112.81.2/24        │              │         10.123.81.2/24        │
        └───────────────────────────────┘              └───────────────────────────────┘
```

Read it as: **rank r's `f0` address is rank r+1's `f1` address.** Rank 0's `f0` is `10.112.80.1` and
rank 1's `f1` is `10.112.80.2` — the two ends of link L0, in the same `/24`. That is the whole rule,
and it is what the fleet file encodes.

### The addresses

| Rank | Box | Management (SSH) | `f0` → next (domain 0 / 2) | `f1` ← prev (domain 0 / 2) |
| --- | --- | --- | --- | --- |
| 0 | qspark01, ASUS Ascent GX10 | 192.168.2.101 | 10.112.80.1 / 10.112.81.1 | 10.141.80.2 / 10.141.81.2 |
| 1 | qspark02, ASUS Ascent GX10 | 192.168.2.102 | 10.123.80.1 / 10.123.81.1 | 10.112.80.2 / 10.112.81.2 |
| 2 | qspark03, NVIDIA DGX Spark | 192.168.2.103 | 10.134.80.1 / 10.134.81.1 | 10.123.80.2 / 10.123.81.2 |
| 3 | qspark04, NVIDIA DGX Spark | 192.168.2.104 | 10.141.80.1 / 10.141.81.1 | 10.134.80.2 / 10.134.81.2 |

| Link | Speed | From | To |
| --- | --- | --- | --- |
| L0 | **100G** | rank 0 `f0` | rank 1 `f1` |
| L1 | 200G | rank 1 `f0` | rank 2 `f1` |
| L2 | 200G | rank 2 `f0` | rank 3 `f1` |
| L3 | 200G | rank 3 `f0` | rank 0 `f1` |

L0 is the ring's bandwidth bottleneck (12.25 GB/s per direction); the other three are 200G. The
addresses above are this ring's. **Another ring uses its own**, and the fleet file is where they go.

### The devices

Each QSFP port appears as **two** RDMA devices, one per PCI domain, each a PCIe Gen5 x4 function with
its own netdev and its own `/24`:

| Port | PCI domain 0 | PCI domain 2 |
| --- | --- | --- |
| `f0` (next-facing) | `rocep1s0f0` | `roceP2p1s0f0` |
| `f1` (prev-facing) | `rocep1s0f1` | `roceP2p1s0f1` |

RoCE v2 is **GID index 3** on every device; netdev MTU 9000, RDMA active MTU 4096. One PCIe x4
function cannot fill a 200G port (109 Gb/s), which is why each port is split across two domains: both
together give 196 Gb/s.

### Setting it up

**1. Cable it.** Rank 0 `f0` → rank 1 `f1`, rank 1 `f0` → rank 2 `f1`, rank 2 `f0` → rank 3 `f1`,
rank 3 `f0` → rank 0 `f1`. Check what each box actually has:

```bash
ds41f-4xgx10-switchless/micro/ring_links.sh
```

**2. Give rank 0 SSH to the others**, without a password:

```bash
ssh-keygen -t ed25519          # if there is no key yet
for h in 192.168.2.102 192.168.2.103 192.168.2.104; do ssh-copy-id "$h"; done
```

**3. Write the fleet file.** `--phase fleet` writes a template with this ring's addresses; edit it for
yours. The keys that matter for the network:

```bash
NEXT_DEV0=rocep1s0f0     # every box alike: the four RDMA devices
NEXT_DEV1=roceP2p1s0f0
PREV_DEV0=rocep1s0f1
PREV_DEV1=roceP2p1s0f1
GID_INDEX=3

RANK0_HOSTNAME=qspark01  # `hostname` on the box: the engine finds its rank by it
RANK0_NEXT_IP0=10.112.80.1   # f0, domain 0
RANK0_NEXT_IP1=10.112.81.1   # f0, domain 2
RANK0_PREV_IP0=10.141.80.2   # f1, domain 0
RANK0_PREV_IP1=10.141.81.2   # f1, domain 2

RANK1_SSH=192.168.2.102  # how rank 0 reaches rank 1 (ranks 1-3 only)
```

Rank r's `NEXT_IP<d>` and rank r+1's `PREV_IP<d>` must be in the same subnet, and each address must be
on the netdev of the RDMA device the file names. Before the bootstrap each rank checks that the GID at
`GID_INDEX` of each of its four devices is the IPv4-mapped address the file gives that link, and stops
otherwise, naming the device, the GID, the key and the file.

**4. Put the addresses on the links.** `--phase net` prints what each box needs and changes nothing:

```bash
./tools/deploy-switchless/deploy-switchless.sh --phase net
```

```
  · rank 0: rocep1s0f0 (enp1s0f0np0) needs 10.112.80.1/24
    rank 0: sudo ip addr add 10.112.80.1/24 dev enp1s0f0np0
    rank 0: sudo ip link set enp1s0f0np0 mtu 9000 up
  ...
```

`NET_APPLY=1` runs them. They do **not** survive a reboot, so put them in netplan once they work:

```yaml
# /etc/netplan/60-ring.yaml on rank 0, for one link; one stanza per netdev
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [10.112.80.1/24]
      mtu: 9000
      dhcp4: false
```

```bash
sudo netplan apply
```

**5. Check it.** `--phase preflight` checks SSH, disk, memory, the GPU and the tools; `--phase verify`
runs the ring test, which is the real proof that the links work.

## The deployment

Twelve phases, in this order. Each is idempotent and leaves a marker in `$WORK/.state`, so a run that
stops can be repeated and picks up where it left off.

| Phase | What it does |
| --- | --- |
| `clone` | clones this repository (or uses the clone the script is in) and checks out the ref to deploy |
| `fleet` | writes the fleet file: the four boxes, their cabling, their addresses, their directories |
| `preflight` | checks the boxes: SSH, disk, memory, the GPU, the tools the build needs |
| `net` | prints the link addresses to configure (adds them only with `NET_APPLY=1`) |
| `deps` | installs the build dependencies (`INSTALL_DEPS=1`; needs sudo) |
| `checkpoint` | downloads the checkpoint (475 GB, resumes), including the encoder's `inference/` imports |
| `build` | builds the engine (`sm_121a`) and the front-end |
| `weights` | cuts the four per-rank weight files |
| `engram` | packs the Engram shards |
| `gate` | copies the confidence-gate weights to all four boxes |
| `verify` | runs the ring test |
| `serve` | starts the service, and checks that it answers |

`--phase NAME` runs one phase (repeatable, and it runs even if it finished before); `--skip NAME` runs
everything but those; `--force` redoes a phase that finished. `--list` prints them.

A first run is hours, most of it the 475 GB download. Disk, per box: rank 0 needs the checkpoint
(475 GB) + one rank's files at a time (78 GB) + its own weights (78 GB) + its Engram shards (51 GB) +
64 GiB of spill; ranks 1-3 need 78 + 51 + 64 GiB.

## Where it puts things

Everything it creates lives in one directory, `$WORK`, and nothing outside it is touched:

| The script is at | `$WORK` |
| --- | --- |
| `<repo>/tools/deploy-switchless/deploy-switchless.sh` (where this repo keeps it) | `<repo>/ds41deploy` |
| `<repo>/tools/deploy-switchless.sh` | `<repo>/ds41deploy` |
| `<repo>/deploy-switchless.sh` | `<repo>/ds41deploy` |
| anywhere else | the script's own directory |

In the first three cases `$WORK` carries a `.gitignore` (`*`), so `git status` stays clean and a pull
never touches the checkpoint, the weights or the logs. `REPO=` and `WORK=` override both.

`$WORK` holds the clone, the fleet file, the checkpoint, the weight files, the Engram shards, the
spill, the API key and the logs. The fleet file's paths are written as `~/...` when `$WORK` is under
the home directory, so every box uses its own home.

## What it deploys

By default, the repository the script is in, at whatever that clone has **checked out**. That is what
a reader of this repo expects, and it means the script never moves your working tree off your branch.

```bash
git checkout main
./tools/deploy-switchless/deploy-switchless.sh redeploy             # this repo at main
./tools/deploy-switchless/deploy-switchless.sh redeploy --pr 2      # a pull request's head, merged with main
./tools/deploy-switchless/deploy-switchless.sh redeploy --ref <sha> # any commit, branch or tag
./tools/deploy-switchless/deploy-switchless.sh redeploy --repo URL --ref <branch>   # a fork's branch
```

Outside the clone (the script cloned the recipe itself), the default is `REPO_REF` (`main`), since
there is no working tree of yours to respect.

A pull request's head lives in the base repository as `refs/pull/<n>/head`, so `--pr` needs no fork
named. A PR's head is usually behind `main`, so the script merges `main` into it on a local branch
`deploy-<ref>` (nothing is pushed); a conflict stops it with a message.

`--fork` deploys `FORK_REF` from `FORK_URL` instead: the branch that carries the four pull requests
this ring needs (see [below](#the-four-pull-requests)). `--fork --no-stream-args` deploys `BASE_REF`,
the same without PR #7.

## The service afterwards

```bash
./tools/deploy-switchless/deploy-switchless.sh start      # start it (or leave a running one alone)
./tools/deploy-switchless/deploy-switchless.sh stop       # stop it
./tools/deploy-switchless/deploy-switchless.sh restart    # stop, then start with this script's settings
./tools/deploy-switchless/deploy-switchless.sh status     # what it is doing, and the boxes
./tools/deploy-switchless/deploy-switchless.sh logs       # the last lines of service, rank 0, front-end, encoder
```

`start` is idempotent: a service already serving with these settings is left alone, because a restart
drops the requests in flight. One serving with *other* settings is restarted, so the settings here win.

The script never patches the recipe. It drives the recipe's own scripts (`fleet.sh`, `service.sh`,
`ring.sh`), so an upstream change is picked up by re-running `--phase clone`.

## Settings

Every setting is an environment variable, and the ones with a flag have one.

| Setting | Default | What it does |
| --- | --- | --- |
| `REPO_REF` | `main` | the ref to deploy when the clone has no branch checked out (or the script cloned it) |
| `FORK` (`--fork`) | 0 | deploy `FORK_REF` from `FORK_URL` instead |
| `FORK_REF` | `fix/sampler-dsml-stream-args` | the branch `--fork` deploys |
| `BASE_REF` | `fix/sampler-and-dsml` | the branch `--fork --no-stream-args` deploys |
| `REF` (`--ref`) | | a commit, branch or tag to deploy instead |
| `PR` (`--pr`) | | a pull request's head to deploy instead |
| `REPO_URL` (`--repo`) | this repo | the clone's origin |
| `WORK` | see above | where everything this script creates goes |
| `CKPT` | `$WORK/DeepSeek-V4.1-Flash-1017` | the checkpoint's directory |
| `FLEET` | `$WORK/fleet.conf` | the fleet file the recipe's scripts read |
| `FLEET_SRC` | | copy this fleet file instead of writing the template |
| `JOBS` | 4 | cmake/cargo parallelism |
| `KEEP_CHECKPOINT` | 1 | `0` deletes the checkpoint once the weight files are cut |
| `LOSSLESS` | 0 | `1` drops `DSGB10_OEA` (the lossy expert re-route) |
| `DSGB10_PORT` | 8888 | the front-end's port (`DSGB10_PROD` follows it) |
| `DSGB10_ADDR` | loopback + this box's LAN address | the front-end's addresses |
| `STREAM_TOOL_ARGS` | 1 | `--stream-tool-args`: a stream's tool calls in pieces |
| `VISION` (`--vision` / `--no-vision`) | 1 | run the image encoder, with `service.sh` owning it |
| `VISION_IMAGE` | `lmsysorg/sglang:dev-dsv41` | the image the encoder runs in |
| `VISION_MEM` | `32g` | the encoder container's memory limit (see below) |
| `VISION_PORT` | 18950 | the encoder's port on rank 0's loopback |
| `VISION_MAX_IMAGES` / `VISION_MAX_ROWS` | 100 / 40000 | the encoder's `--max-images` / `--max-rows` |
| `INSTALL_DEPS` | 0 | `1` apt-gets the build dependencies (needs sudo) |
| `NET_APPLY` | 0 | `1` lets `--phase net` add the link addresses (needs sudo) |

Anything that needs sudo or deletes data is opt-in: `INSTALL_DEPS=1`, `NET_APPLY=1`,
`KEEP_CHECKPOINT=0`.

## Images

`VISION=1` (the default) gives `service.sh` the encoder's command (`DSGB10_VISION_CMD`), so the
service starts the encoder before the engine, restarts it when it exits, stops it with the service and
reports it in `status`. Without it dsfront refuses every image part with
`400 image, audio and video input are not supported`, which a gateway in front of it reports as a 502.
[vision.md](../../ds41f-4xgx10-switchless/docs/vision.md) has the encoder itself.

The encoder runs in a container of `VISION_IMAGE`, the public base of the kit image, so nothing is
built here. Two things about it are worth knowing:

- **`VISION_MEM` caps the encoder's GPU memory.** On a GB10 the GPU's memory *is* host memory (unified
  LPDDR), so a CUDA allocation is a page of the container's cgroup and `--memory` limits it. A limit
  that looks generous for a 0.95 GiB model is not: torch's CUDA context, the CPU-side state dict it
  loads before moving it, and the ViT's first allocation all count. At `6g` the encoder dies in
  `load()` with `CUDA error: out of memory` on a 1 MB `nn.Linear`.
- **The image is pulled before the service starts**, not inside the encoder's supervised start: it is
  13.55 GB, and inside the start it is invisible (`status` says "not listening (it loads its weights
  first)" while it is really downloading) and a failed pull counts as an encoder exit, backing off to
  15 minutes.

`vision_preflight()` checks both before anything is stopped: it pulls the image, imports `torch`,
`PIL` and `safetensors` (fatal if missing, with the module named), warns if PyAV is absent (video
needs it, images do not), and allocates a tensor under the same `--memory` the encoder gets.

## The four pull requests

`--fork` deploys a branch that merges four pull requests on top of `main`. Each is a separate fix, and
each is worth having on its own:

| PR | What it fixes |
| --- | --- |
| #3 | the sampler's open-interval endpoint: random Chinese and Cyrillic characters in prose, reasoning and tool arguments |
| #2 | DSML quarantine: `｜DSML｜` markup leaking into the answer text |
| #7 | a stream's tool calls in pieces, instead of one chunk once the call is complete |
| #15 | the service owns the image encoder (`DSGB10_VISION_CMD`) |

PR #7 conflicts with PR #2 in `encoding.rs` (both rewrite the same parser), so the branch is a
deliberate merge, not a rebase.

## Testing it without a ring

`DEPLOY_LIB=1 source tools/deploy-switchless/deploy-switchless.sh` loads the functions without running
`main`, so helpers and phases can be exercised with stub `ssh`/`hostname` binaries on a box that is not
a DGX.

## What it does not do

- It does not configure the links' addresses unless `NET_APPLY=1`, and it does not put them in netplan,
  so they do not survive a reboot until you do.
- It does not install the build dependencies unless `INSTALL_DEPS=1`.
- It does not tune the ring: the engine arguments are the recipe's, and `LOSSLESS=1` is the only
  variation it offers.
- It does not push anything. A PR's head is merged locally, on a branch named `deploy-<ref>`.
