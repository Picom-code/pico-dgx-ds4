# deploy-switchless.sh: the whole ring from one command

[tools/deploy-switchless.sh](../../tools/deploy-switchless.sh) takes a fresh 4-box switchless ring to a
serving endpoint, and keeps it there. It is the manual procedure in
[README.md](../README.md#how-to-use) and [operations.md](operations.md) written down once, with the
order, the checks and the resumability that a hand-run does not have.

Run it on rank 0, the box with passwordless SSH to the other three.

```bash
./tools/deploy-switchless.sh              # what this is, what it can do, and where things stand
./tools/deploy-switchless.sh deploy       # every phase, in order (the first run: hours, 475 GB)
./tools/deploy-switchless.sh status       # what the service is doing, and the boxes
./tools/deploy-switchless.sh logs         # the last lines of its logs
```

## What it does

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

## Where it puts things

Everything it creates lives in one directory, `$WORK`, and nothing outside it is touched:

| The script is at | `$WORK` |
| --- | --- |
| `<repo>/tools/deploy-switchless.sh` (where this repo keeps it) | `<repo>/ds41deploy` |
| `<repo>/deploy-switchless.sh` | `<repo>/ds41deploy` |
| anywhere else | the script's own directory |

In the first two cases `$WORK` carries a `.gitignore` (`*`), so `git status` stays clean and a pull
never touches the checkpoint, the weights or the logs. `REPO=` and `WORK=` override both.

`$WORK` holds the clone, the fleet file, the checkpoint, the weight files, the Engram shards, the
spill, the API key and the logs. The fleet file's paths are written as `~/...` when `$WORK` is under
the home directory, so every box uses its own home.

## What it deploys

By default, the repository the script is in, at `REPO_REF` (`main`). That is what a reader of this
repo expects.

```bash
./tools/deploy-switchless.sh redeploy             # this repo at main: fetch, rebuild, restart
./tools/deploy-switchless.sh redeploy --pr 2      # a pull request's head, merged with main
./tools/deploy-switchless.sh redeploy --ref <sha> # any commit, branch or tag
./tools/deploy-switchless.sh redeploy --repo URL --ref <branch>   # a fork's branch
```

A pull request's head lives in the base repository as `refs/pull/<n>/head`, so `--pr` needs no fork
named. A PR's head is usually behind `main`, so the script merges `main` into it on a local branch
`deploy-<ref>` (nothing is pushed); a conflict stops it with a message.

`--fork` deploys `FORK_REF` from `FORK_URL` instead: the branch that carries the four pull requests
this ring needs (see [below](#the-four-pull-requests)). `--fork --no-stream-args` deploys `BASE_REF`,
the same without PR #7.

## The service afterwards

```bash
./tools/deploy-switchless.sh start      # start it (or leave a running one alone)
./tools/deploy-switchless.sh stop       # stop it
./tools/deploy-switchless.sh restart    # stop, then start with this script's settings
./tools/deploy-switchless.sh status     # what it is doing, and the boxes
./tools/deploy-switchless.sh logs       # the last lines of service, rank 0, front-end and encoder logs
```

`start` is idempotent: a service already serving with these settings is left alone, because a restart
drops the requests in flight. One serving with *other* settings is restarted, so the settings here win.

The script never patches the recipe. It drives the recipe's own scripts (`fleet.sh`, `service.sh`,
`ring.sh`), so an upstream change is picked up by re-running `--phase clone`.

## Settings

Every setting is an environment variable, and the ones with a flag have one.

| Setting | Default | What it does |
| --- | --- | --- |
| `REPO_REF` | `main` | the ref to deploy from the repository the script is in |
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
[docs/vision.md](vision.md) has the encoder itself.

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

`DEPLOY_LIB=1 source tools/deploy-switchless.sh` loads the functions without running `main`, so
helpers and phases can be exercised with stub `ssh`/`hostname` binaries on a box that is not a DGX.

## What it does not do

- It does not configure the links' addresses unless `NET_APPLY=1`, and it does not put them in netplan,
  so they do not survive a reboot until you do.
- It does not install the build dependencies unless `INSTALL_DEPS=1`.
- It does not tune the ring: the engine arguments are the recipe's, and `LOSSLESS=1` is the only
  variation it offers.
- It does not push anything. A PR's head is merged locally, on a branch named `deploy-<ref>`.
