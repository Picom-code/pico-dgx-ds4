# Builds the engine `dsgb10_rank` and the front-end `dsfront` for GB10 (sm_121a).
#
# The versions below are the ones README.md's "What the build and the ring need"
# lists. The base image is CUDA 13.0.3 on Ubuntu 24.04, the distribution the
# README's rank 0 runs. README.md line 128 states CUDA 13.0; 13.0.3 is a patch
# release of the same minor version, and it is the image this fleet holds.
#
# The image carries the build toolchain, not the weights. Mount the weights, the
# Engram shards and the fleet file at run time; see README.md step 2.

FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}

# CMake 3.28 or later, g++ 13.3 (C++20), libibverbs-dev (rdma-core), and the
# Rust toolchain for frontend/.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libibverbs-dev \
        libnuma-dev \
        pkg-config \
        curl \
        ca-certificates \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Rust: the README names cargo 1.97.1. rustup installs that exact toolchain
# rather than whatever the distribution ships.
# The download is checked before it is run: `curl | sh` takes its exit status
# from sh, so a failed download leaves the layer green with no cargo on PATH.
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=${CARGO_HOME}/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf -o /tmp/rustup.sh https://sh.rustup.rs \
    && sh /tmp/rustup.sh -y --no-modify-path --default-toolchain 1.97.1 --profile minimal \
    && rm /tmp/rustup.sh \
    && cargo --version

WORKDIR /src

# The engine: C++20 + CUDA, targeting compute_121a/sm_121a only.
# fleet/homelab.conf is copied too: engine/CMakeLists.txt line 34 reads
# ../fleet/homelab.conf and installs it beside each binary as fleet.conf.
COPY engine/ engine/
COPY fleet/ fleet/
RUN cmake -S engine -B engine/build && cmake --build engine/build -j"$(nproc)"

# The front-end: Rust, edition 2021.
COPY frontend/ frontend/
RUN cd frontend && cargo build --release -j"$(nproc)"

# The scripts under engine/server/ run the binaries and reach ranks 1-3 by SSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client \
        iproute2 \
        iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# The binaries land here, the same relative paths service.sh copies from.
ENV DSGB10_BUILD_DIR=/src
CMD ["/bin/bash"]