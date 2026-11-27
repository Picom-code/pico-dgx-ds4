#!/usr/bin/env python3
"""Write a cargo local registry from frontend/Cargo.lock and build/cache/.

A cargo local registry is 2 things in one directory:

  <out>/index/<hash path>/<name>   one JSON line per published version
  <out>/<name>-<version>.crate     the crate file itself

Cargo reads that directory without a network and without the crates.io index,
so `cargo build --offline --locked` resolves the lock file from it.

2 details decide whether cargo accepts the registry, and both cost a build
when they are wrong:

1. The index path follows the crates.io git-index layout, not a flat name:
   a 1-character name sits at `index/1/<name>`, a 2-character name at
   `index/2/<name>`, and a longer name at `index/<first two>/<next two>/<name>`.
   A flat `index/<name>` file is not found, and cargo reports
   "no matching package named ...".

2. The index line carries the crate's real `features` and `deps`, read from
   the `Cargo.toml` inside the `.crate` file. An empty `features` map makes
   cargo report "package `dsfront` depends on `axum` with feature `http1` but
   `axum` does not have that feature".

The script reads the lock file and the `.crate` archives. It opens no socket.

Usage:
  make-local-registry.py --lock frontend/Cargo.lock --crates build/cache --out /tmp/registry

Exit 0 when every package with a checksum has its .crate file and the index is
written. Exit 1 naming the first package whose .crate file is missing.
"""

import argparse
import json
import os
import re
import sys
import tarfile


def parse_lock(path):
    """Return the packages of a Cargo.lock file as a list of dicts."""
    text = open(path, encoding="utf-8").read()
    packages = []
    for block in text.split("[[package]]")[1:]:
        name = re.search(r'^name = "(.+)"', block, re.M)
        version = re.search(r'^version = "(.+)"', block, re.M)
        if not name or not version:
            continue
        source = re.search(r'^source = "(.+)"', block, re.M)
        checksum = re.search(r'^checksum = "(.+)"', block, re.M)
        deps = []
        dep_block = re.search(r'^dependencies = \[(.*?)\]', block, re.M | re.S)
        if dep_block:
            deps = re.findall(r'"([^"]+)"', dep_block.group(1))
        packages.append({
            "name": name.group(1),
            "version": version.group(1),
            "source": source.group(1) if source else None,
            "checksum": checksum.group(1) if checksum else None,
            "deps": deps,
        })
    return packages


def index_relpath(name):
    """Return the path of a crate's index file under <registry>/index/."""
    lowered = name.lower()
    if len(lowered) == 1:
        return os.path.join("1", lowered)
    if len(lowered) == 2:
        return os.path.join("2", lowered)
    if len(lowered) == 3:
        return os.path.join("3", lowered[0], lowered)
    return os.path.join(lowered[0:2], lowered[2:4], lowered)


def read_crate_toml(crate_path):
    """Return the text of the Cargo.toml inside a .crate archive."""
    with tarfile.open(crate_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.endswith("/Cargo.toml") and member.name.count("/") == 1:
                return archive.extractfile(member).read().decode("utf-8")
    raise SystemExit("the archive holds no Cargo.toml: %s" % crate_path)


def parse_toml_subset(text):
    """Parse the parts of a crate's Cargo.toml the index needs.

    The file is the normalized form cargo writes when it publishes, so the
    shapes are few: `[features]` with arrays of strings, and
    `[dependencies.<name>]` / `[dependencies.<name>.target...]` tables with
    `version`, `optional`, `default-features`, `features` and `package`.

    A full TOML parser is not used because the build image has no third-party
    Python package and the standard library has no TOML reader before 3.11.
    """
    features = {}
    deps = []
    section = None
    current = None
    pending_key = None
    pending_value = []

    def flush():
        nonlocal pending_key, pending_value
        if pending_key is None:
            return
        value = " ".join(pending_value).strip()
        if section == "features":
            features[pending_key] = re.findall(r'"([^"]*)"', value)
        elif current is not None:
            current[pending_key] = parse_scalar_or_list(value)
        pending_key = None
        pending_value = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            flush()
            section = line[1:-1].strip()
            table = dependency_table_name(section)
            if table is not None:
                current = {}
                deps.append((table, current))
            else:
                current = None
            continue
        if section is None:
            continue
        if pending_key is not None:
            # A multi-line array continues on a line that holds no `=`, so the
            # continuation is collected before the `=` test below. Reading the
            # test first drops every member and leaves the feature empty.
            pending_value.append(line)
            if line.endswith("]"):
                flush()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().strip('"')
        value = value.strip()
        if value.startswith("[") and not value.endswith("]"):
            flush()
            pending_key = key
            pending_value = [value]
            continue
        if section == "features":
            # `rt = []` is a feature with no member, and it must still be
            # listed: another feature may name it, and cargo rejects an index
            # line whose feature names something that is neither a dependency
            # nor a feature.
            features[key] = re.findall(r'"([^"]*)"', value)
        elif current is not None:
            current[key] = parse_scalar_or_list(value)
    flush()
    return features, deps


def dependency_table_name(section):
    """Return the dependency name of a Cargo.toml table header, or None.

    A crate's Cargo.toml names a dependency in 3 header shapes:

      [dependencies.foo]                                  plain
      [dependencies.foo.target.'cfg(unix)']               target-specific
      [target.'cfg(unix)'.dependencies.foo]               target-specific

    All 3 describe the same dependency `foo`, and the index needs one entry
    per dependency, so the name is returned and the target is dropped.

    A `dev-dependencies` table returns None. The index of a published crate
    carries no dev-dependency, because a crate that depends on it never builds
    its tests. Including one makes cargo look for a package the lock file does
    not hold, with the message "no matching package named `anyhow` found ...
    required by package `axum v0.8.9`".

    A header that names no dependency returns None.
    """
    parts = split_table_header(section)
    if not parts:
        return None
    if parts[0] in ("dependencies", "build-dependencies"):
        return parts[1] if len(parts) > 1 else None
    if parts[0] == "target":
        for index, part in enumerate(parts):
            if part in ("dependencies", "build-dependencies"):
                return parts[index + 1] if index + 1 < len(parts) else None
    return None


def split_table_header(section):
    """Split a table header on the dots that sit outside quotes.

    A header such as `target.'cfg(all(unix, target_os = "linux"))'.dependencies.foo`
    holds dots inside a quoted key, so a plain `split(".")` cuts the key apart.
    """
    parts = []
    current = []
    quote = None
    for char in section:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == ".":
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def parse_scalar_or_list(value):
    """Return a TOML scalar as a Python value, or a list of strings."""
    value = value.strip()
    if value.startswith("["):
        return re.findall(r'"([^"]*)"', value)
    if value.startswith('"'):
        return value.strip('"')
    if value in ("true", "false"):
        return value == "true"
    return value


def dep_entries(lock_packages, crate_deps):
    """Turn a crate's Cargo.toml dependency tables into index `deps` entries.

    Every table becomes an entry, including an optional dependency the lock
    file does not list. Cargo rejects the whole index line when a feature
    names `dep:<name>` and `<name>` is not in `deps`, with the message
    "feature `json` includes `dep:axum-macros`, but `axum-macros` is not
    listed as a dependency". The lock file only lists what the build enables,
    so the version requirement comes from the crate's own Cargo.toml.
    """
    by_name = {}
    for pkg in lock_packages:
        by_name.setdefault(pkg["name"], []).append(pkg)

    entries = []
    seen = set()
    for table_name, fields in crate_deps:
        if not isinstance(fields, dict):
            continue
        # The dependency is named by its table, and `package = "futures"`
        # renames the crate it resolves to. A feature names the table name,
        # as in `compat = ["futures_01"]`, so the index entry keeps the table
        # name and carries the real crate name in `package`.
        dep_name = table_name
        real_name = fields.get("package", table_name)
        if not isinstance(real_name, str):
            real_name = table_name
        if dep_name in seen:
            continue
        seen.add(dep_name)
        req = fields.get("version", "*")
        if not isinstance(req, str):
            req = "*"
        feat = fields.get("features", [])
        if not isinstance(feat, list):
            feat = []
        default_features = fields.get("default-features", True)
        if not isinstance(default_features, bool):
            default_features = True
        optional = fields.get("optional", False)
        if not isinstance(optional, bool):
            optional = False
        entries.append({
            "name": dep_name,
            "req": req,
            "features": feat,
            "optional": optional,
            "default_features": default_features,
            "target": None,
            "kind": "normal",
            "registry": None,
            "package": real_name if real_name != dep_name else None,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, help="path to frontend/Cargo.lock")
    parser.add_argument("--crates", required=True, help="directory holding the .crate files")
    parser.add_argument("--out", required=True, help="directory to write the registry into")
    args = parser.parse_args()

    packages = parse_lock(args.lock)
    if not packages:
        raise SystemExit("the lock file lists no package: %s" % args.lock)

    index_dir = os.path.join(args.out, "index")
    os.makedirs(index_dir, exist_ok=True)

    written = 0
    for pkg in packages:
        if pkg["checksum"] is None:
            # The local package (dsfront) is built from the source tree, so it
            # has no .crate file and no index line.
            continue
        crate_file = "%s-%s.crate" % (pkg["name"], pkg["version"])
        crate_path = os.path.join(args.crates, crate_file)
        if not os.path.isfile(crate_path):
            raise SystemExit("the cache holds no %s" % crate_file)

        features, crate_deps = parse_toml_subset(read_crate_toml(crate_path))
        entries = dep_entries(packages, crate_deps)

        # Cargo rejects an index line whose optional dependency no feature
        # names, with the message "optional dependency `tracing` is not
        # included in any feature". The crate's own Cargo.toml satisfies this
        # through `dep:<name>` entries, but a feature list this script reads
        # only in part can miss one, so each unnamed optional dependency gets
        # a feature of its own name. The feature is never enabled by the build
        # unless the crate names it, so it changes no resolution.
        dep_names = set(entry["name"] for entry in entries)
        feature_names = set(features)
        for name in list(features):
            kept = []
            for value in features[name]:
                if value.startswith("dep:"):
                    kept.append(value)
                    continue
                # A member such as `hyper?/http1` names the dependency
                # `hyper` and asks for its feature `http1` only when `hyper`
                # is already enabled. The `?` marks that condition, so the
                # head is read without it. Reading the head with the `?` left
                # matches no dependency, and the member was dropped, which
                # built `hyper-util` without `http1` and failed the axum build
                # with "could not find `auto` in `conn`".
                head = value.split("/", 1)[0].rstrip("?")
                if head in dep_names or head in feature_names:
                    kept.append(value)
                    continue
                # The member names neither a dependency nor a feature, so it
                # names a dev-dependency, as in `test = ["syn-test-suite/all-features"]`.
                # The index carries no dev-dependency, so cargo rejects the
                # line with "feature `test` includes `syn-test-suite/all-features`,
                # but `syn-test-suite` is not a dependency". Dropping the
                # member keeps the feature usable and changes no resolution,
                # because no crate can enable a dev-dependency feature.
            features[name] = kept

        named = set()
        for values in features.values():
            for value in values:
                if value.startswith("dep:"):
                    named.add(value[len("dep:"):])
        for entry in entries:
            if entry["optional"] and entry["name"] not in named:
                features.setdefault(entry["name"], ["dep:" + entry["name"]])

        line = {
            "name": pkg["name"],
            "vers": pkg["version"],
            "deps": entries,
            "cksum": pkg["checksum"],
            "features": features,
            "yanked": False,
            "links": None,
        }
        index_path = os.path.join(index_dir, index_relpath(pkg["name"]))
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")

        # Cargo reads the .crate file from the registry root, not from the
        # cache, so each file is linked into place.
        dest = os.path.join(args.out, crate_file)
        if not os.path.exists(dest):
            os.link(crate_path, dest)
        written += 1

    print("wrote %d index lines and %d .crate files into %s" % (written, written, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())