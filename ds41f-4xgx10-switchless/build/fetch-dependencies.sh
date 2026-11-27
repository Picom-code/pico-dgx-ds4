#!/usr/bin/env bash
# fetch-dependencies.sh - download every pinned dependency into build/cache/
# and check each file against the sha256 in build/dependencies.txt.
#
# This is the only step of the build that needs network. It runs on a host
# with network; `docker build --network none` then reads build/cache/ alone.
#
# Usage:
#   ./build/fetch-dependencies.sh            # fetch what is missing, verify all
#   ./build/fetch-dependencies.sh --verify   # verify only, fetch nothing
#
# Exit 0 when every file in build/cache/ matches its manifest sha256.
# Exit 1 and name the first mismatch when a file does not match.
# Exit 2 when the manifest is missing or a record is malformed.
#
# The base image is not downloaded here. It is already on the build host, and
# the registry mirror returned 000 on 2026-11-26, so the script checks that
# the local image carries the pinned digest and stops if it does not.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/dependencies.txt"
CACHE="$HERE/cache"
VERIFY_ONLY=0

# The Rust release host, and the date of the 1.97.1 release the channel file
# names. The tarball path carries the date, so the date is pinned too.
RUST_BASE="https://static.rust-lang.org/dist/2026-07-16"
RUSTUP_INIT_URL="https://sh.rustup.rs"
CRATE_BASE="https://static.crates.io/crates"

for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit "${2:-1}"; }

[[ -f "$MANIFEST" ]] || die "the manifest does not exist: $MANIFEST" 2
mkdir -p "$CACHE"

fetch_file() {
  local url="$1" out="$2"
  curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$out.part" "$url" 2>/dev/null \
    && mv "$out.part" "$out"
}

check_sha() {
  local file="$1" want="$2" got
  got="$(sha256sum "$CACHE/$file" | cut -d' ' -f1)"
  [[ "$got" == "$want" ]] || die "sha256 mismatch on $file: got $got, the manifest pins $want"
}

# ---- the base image --------------------------------------------------------
# The image is not fetched: it is already on the host, and the digest proves
# it is the image the manifest names.
check_image() {
  local name version sha found
  read -r name version sha <<< "$1"
  if ! command -v docker >/dev/null 2>&1; then
    log "image $name: docker is not on PATH, skipped"
    return 0
  fi
  found="$(docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}}' 2>/dev/null \
           | awk -v n="$name" '$1 == n {print $2; exit}')"
  [[ -n "$found" ]] || die "the base image $name is not on this host; load it before the offline build"
  [[ "$found" == "$sha" ]] || die "the base image $name has digest $found, the manifest pins $sha"
  log "image $name: digest matches $sha"
}

# ---- the .deb files --------------------------------------------------------
# The pool URL of each .deb is pinned in the manifest, so the script never
# guesses a path. The URL was read from the base image with
# `apt-get install --download-only --print-uris`.
fetch_debs() {
  local name version sha url file
  while read -r name version sha url; do
    [[ -z "$name" ]] && continue
    if [[ "$sha" == "-" ]]; then
      log "apt $name $version: no file, skipped"
      continue
    fi
    [[ -n "$url" ]] || die "the manifest pins no URL for apt package $name $version" 2
    file="${url##*/}"
    if [[ ! -f "$CACHE/$file" ]]; then
      [[ $VERIFY_ONLY -eq 1 ]] && die "the cache holds no $file and --verify was given"
      fetch_file "$url" "$CACHE/$file" || die "the download of $url failed"
    fi
    check_sha "$file" "$sha"
    log "apt $name $version: sha256 matches"
  done
}

# ---- the Rust toolchain ----------------------------------------------------
# The manifest record is `rustup-toolchain <version> <target> <sha256>`, so the
# file name is `rust-<version>-<target>.tar.xz`, the name the release host uses.
fetch_rust() {
  local kind version target sha file url
  while read -r kind version target sha; do
    [[ -z "$kind" ]] && continue
    if [[ "$kind" == "rustup-toolchain" ]]; then
      file="rust-$version-$target.tar.xz"
      url="$RUST_BASE/$file"
    else
      file="rustup-init.sh"
      url="$RUSTUP_INIT_URL"
    fi
    if [[ ! -f "$CACHE/$file" ]]; then
      [[ $VERIFY_ONLY -eq 1 ]] && die "the cache holds no $file and --verify was given"
      log "fetching $file"
      fetch_file "$url" "$CACHE/$file" || die "the download of $url failed"
    fi
    check_sha "$file" "$sha"
    log "$kind $version: sha256 matches"
  done
}

# ---- the crates ------------------------------------------------------------
# A crate is fetched from the static crates.io CDN, which serves the exact
# .crate file the lock file's checksum describes. The checksum in Cargo.lock is
# the sha256 of that file, so the same check applies.
fetch_crates() {
  local name version sha file url
  while read -r name version sha; do
    [[ -z "$name" ]] && continue
    if [[ "$sha" == "local-package" ]]; then
      log "crate $name $version: local package, no file"
      continue
    fi
    file="$name-$version.crate"
    url="$CRATE_BASE/$name/$file"
    if [[ ! -f "$CACHE/$file" ]]; then
      [[ $VERIFY_ONLY -eq 1 ]] && die "the cache holds no $file and --verify was given"
      fetch_file "$url" "$CACHE/$file" || die "the download of $url failed"
    fi
    check_sha "$file" "$sha"
    log "crate $name $version: sha256 matches"
  done
}

# ---- the final count -------------------------------------------------------
# Every file in the cache must be named by the manifest, and every manifest
# record that names a file must be in the cache. A file in the cache that the
# manifest does not name is a defect: it would enter the image unverified.
count_cache() {
  local manifest_files cache_files missing extra n
  manifest_files="$(awk '
    $1 == "count" || $1 == "image" { next }
    $4 == "-" || $4 == "local-package" { next }
    $1 == "crate" { print $2 "-" $3 ".crate"; next }
    $1 == "rustup-toolchain" { print "rust-" $2 "-" $3 ".tar.xz"; next }
    $1 == "rustup-init" { print "rustup-init.sh"; next }
    $1 == "apt-direct" || $1 == "apt-transitive" { n = split($5, a, "/"); print a[n]; next }
  ' "$MANIFEST" | sort)"
  cache_files="$(cd "$CACHE" && ls -1 2>/dev/null | grep -v '\.part$' | sort)"

  missing="$(comm -23 <(printf '%s\n' "$manifest_files") <(printf '%s\n' "$cache_files"))"
  extra="$(comm -13 <(printf '%s\n' "$manifest_files") <(printf '%s\n' "$cache_files"))"

  if [[ -n "$missing" ]]; then
    printf '%s\n' "$missing" | sed 's/^/  missing: /' >&2
    die "the cache is missing files the manifest names"
  fi
  if [[ -n "$extra" ]]; then
    printf '%s\n' "$extra" | sed 's/^/  unlisted: /' >&2
    die "the cache holds files the manifest does not name"
  fi

  n="$(printf '%s\n' "$cache_files" | grep -c . || true)"
  log "OK: $n files in $CACHE, every sha256 matches the manifest"
}

# ---- run -------------------------------------------------------------------
IMAGE_RECORD="$(awk '$1 == "image" { print $2, $3, $4; exit }' "$MANIFEST")"
[[ -n "$IMAGE_RECORD" ]] || die "the manifest names no base image" 2

check_image "$IMAGE_RECORD"
awk '$1 == "apt-direct" || $1 == "apt-transitive" { print $2, $3, $4, $5 }' "$MANIFEST" | fetch_debs
awk '$1 == "rustup-toolchain" || $1 == "rustup-init" { print $1, $2, $3, $4 }' "$MANIFEST" | fetch_rust
awk '$1 == "crate" { print $2, $3, $4 }' "$MANIFEST" | fetch_crates
count_cache

exit 0