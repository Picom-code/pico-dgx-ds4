# Builds the engine `dsgb10_rank` and the front-end `dsfront` for GB10 (sm_121a).
#
# The versions below are the ones README.md's "What the build and the ring need"
# lists. The base image is CUDA 13.0.3 on Ubuntu 24.04, the distribution the
# README's rank 0 runs. README.md line 128 states CUDA 13.0; 13.0.3 is a patch
# release of the same minor version, and it is the image this fleet holds.
#
# THE BUILD READS NO NETWORK. Every package, the Rust toolchain and every crate
# comes from build/cache/, which build/fetch-dependencies.sh fills on a host
# with network. build/dependencies.txt pins each file by version and sha256.
# The build is therefore run as:
#
#   docker build --network none -t dsgb10-build ds41f-4xgx10-switchless/
#
# A build that reaches the network fails, because there is no network.
#
# The image carries the build toolchain, not the weights. Mount the weights, the
# Engram shards and the fleet file at run time; see README.md step 2.

FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 AS toolchain

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}

# ---- the pinned dependencies, copied in before anything is installed -------
# The cache holds 57 .deb files, the Rust 1.97.1 tarball and 52 .crate files.
# It is copied as one layer so a rebuild reuses it.
COPY build/cache/ /tmp/cache/

# ---- the apt packages, installed from the local .deb files -----------------
# `dpkg` is given the absolute path of every .deb, so it reads no package index
# and resolves nothing over the network.
#
# The install runs in rounds because dpkg checks a Pre-Depends against the
# packages already unpacked, and a glob order puts `python3-minimal` before
# `python3.12-minimal`. Each round unpacks every file that has not succeeded
# yet and then configures what it can; a file that still fails waits for the
# next round. The loop stops when a round leaves nothing pending, and the
# final `dpkg --configure -a` is the check: it exits non-zero if any package
# is left unconfigured. On the 2026-11-27 cache this takes 3 rounds.
# build-essential is a metapackage the base image already satisfies; it is
# named in dependencies.txt so the list matches.
RUN set -eux; \
    pending="$(ls /tmp/cache/*.deb)"; \
    round=0; \
    while [ -n "$pending" ]; do \
        round=$((round + 1)); \
        [ "$round" -le 10 ] || { echo "dpkg still fails after 10 rounds: $pending" >&2; exit 1; }; \
        failed=""; \
        for f in $pending; do \
            dpkg --unpack "$f" >/dev/null 2>&1 || failed="$failed $f"; \
        done; \
        dpkg --configure -a >/dev/null 2>&1 || true; \
        pending="$(echo $failed)"; \
        echo "round $round: $(echo $pending | wc -w) files pending"; \
    done; \
    dpkg --configure -a; \
    rm -rf /var/lib/apt/lists/* /tmp/cache

# ---- the Rust toolchain, from the pinned tarball ---------------------------
# The tarball is the official 1.97.1 release for aarch64-unknown-linux-gnu.
# `install.sh --prefix` places it without rustup and without a network read.
# The sha256 is checked here as well as in the fetch script, so a cache that
# was edited after the fetch cannot enter the image.
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=${CARGO_HOME}/bin:${PATH}
COPY build/cache/rust-1.97.1-aarch64-unknown-linux-gnu.tar.xz /tmp/rust.tar.xz
RUN set -eux; \
    echo "9a7a2c336b4787f1b72f6bab7c35d5b7af2fd03cbd39b4fc721466a70d402a7d  /tmp/rust.tar.xz" \
        | sha256sum -c -; \
    mkdir -p /tmp/rust; \
    tar -xJf /tmp/rust.tar.xz -C /tmp/rust --strip-components=1; \
    /tmp/rust/install.sh --prefix=/usr/local --disable-ldconfig; \
    rm -rf /tmp/rust /tmp/rust.tar.xz; \
    cargo --version; \
    rustc --version

# The stage `toolchain` ends here, so the cargo step can be built and
# tested on its own:
#   docker build --network none --target toolchain -t dsgb10-toolchain .
WORKDIR /src

# ---- the engine: C++20 + CUDA, targeting compute_121a/sm_121a only ---------
# fleet/homelab.conf is copied too: engine/CMakeLists.txt line 34 reads
# ../fleet/homelab.conf and installs it beside each binary as fleet.conf.
COPY engine/ engine/
COPY fleet/ fleet/
RUN cmake -S engine -B engine/build && cmake --build engine/build -j"$(nproc)"

# ---- the front-end: Rust, edition 2021, from the local registry -------------
# `cargo vendor` is not used: it needs the crates.io index to resolve the lock
# file, and the build has no network.
#
# A local registry is a directory of `.crate` files plus one index file per
# crate name. Cargo reads it directly, so no index is fetched:
#
#   /tmp/registry/index/<name>   one JSON line per version
#   /tmp/registry/<name>-<ver>.crate
#
# The index is generated from frontend/Cargo.lock by build/make-local-registry.py,
# so each line carries the checksum and the dependency list the lock file pins.
# The script reads the lock file only; it opens no socket.
COPY build/cache/ /tmp/crates/
COPY build/make-local-registry.py /tmp/make-local-registry.py
COPY frontend/ frontend/
RUN set -eux; \
    python3 /tmp/make-local-registry.py \
        --lock /src/frontend/Cargo.lock \
        --crates /tmp/crates \
        --out /tmp/registry; \
    cd /src/frontend; \
    mkdir -p .cargo; \
    printf '[source.crates-io]\nreplace-with = "local-registry"\n\n[source.local-registry]\nlocal-registry = "/tmp/registry"\n\n[net]\noffline = true\n' \
        > .cargo/config.toml; \
    cargo build --release --offline --locked -j"$(nproc)"

# ---- the binaries, in one place -------------------------------------------
# The 2 paths the goal compares by sha256. `service.sh` copies from the same
# relative paths, so the image and a host build agree.
RUN set -eux; \
    mkdir -p /out; \
    cp engine/build/server/dsgb10_rank /out/dsgb10_rank; \
    cp frontend/target/release/dsfront /out/dsfront; \
    sha256sum /out/dsgb10_rank /out/dsfront

ENV DSGB10_BUILD_DIR=/src
CMD ["/bin/bash"]