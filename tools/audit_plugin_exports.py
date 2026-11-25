#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Audit a libspatialblend shared library for the public sb_* exports.

The expected names are the functions declared with SB_EXPORT or SB_OBS_EXPORT
in spatialblend/api/c. Platform tools:

  ELF (.so)     nm -D --defined-only
  Mach-O (.dylib) nm -gU
  COFF (.dll)   dumpbin /exports

Linker section markers (__start_* / __stop_*) are ignored. Any other exported
symbol outside sb_* fails the audit, which is what the version script is there
to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


EXPORT_MACROS = ("SB_EXPORT", "SB_OBS_EXPORT")
LINKER_BOUNDARY = re.compile(r"^__(?:start|stop)_")
DUMPBIN_EXPORT = re.compile(
    r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8,}\s+(\S+)"
)


def expected_symbols(headers_dir: Path) -> set[str]:
    """Return sb_* function names declared for export."""
    names: set[str] = set()
    headers = sorted(headers_dir.glob("*.h"))
    if not headers:
        raise SystemExit(f"no headers in {headers_dir}")
    for path in headers:
        text = _strip_comments(path.read_text(encoding="utf-8"))
        kept = [
            line
            for line in text.splitlines()
            if not line.lstrip().startswith("#")
        ]
        text = "\n".join(kept)
        for macro in EXPORT_MACROS:
            for match in re.finditer(rf"\b{macro}\b", text):
                decl = text[match.end() :]
                semi = decl.find(";")
                if semi < 0:
                    continue
                names.update(re.findall(r"\b(sb_[A-Za-z0-9_]+)\s*\(", decl[:semi]))
    if not names:
        raise SystemExit(f"no exported sb_* declarations in {headers_dir}")
    return names


def normalize_export(name: str) -> str:
    """Drop a GNU symbol version and the Mach-O underscore prefix."""
    name = name.split("@@", 1)[0]
    if name.startswith("_sb_"):
        return name[1:]
    return name


# The node itself: "VERS_0.0.1" (GNU nm) or "VERS_0.0.1@@VERS_0.0.1" (llvm-nm appends its own version). The
# "@@"/"@" form is accepted only when both halves name the same version.
VERSION_NODE = re.compile(r"^(VERS_[0-9A-Za-z_.]*)(?:@@?\1)?$")


def parse_nm(text: str) -> set[str]:
    """Return the names of defined, externally visible symbols from nm output.

    Not exports, so skipped:
      U            undefined (imports)
      b d t r ...  lowercase types are STB_LOCAL. Newer GNU ld (2.42) copies thread-local statics of the static deps
                   (grpc, absl, protobuf) into .dynsym as LOCAL entries to carry TLS relocations; the dynamic linker
                   cannot bind or dlsym them. w, v and u are lowercase but weak/unique-global, so kept.
      A VERS_*     the version-definition node of the version script (llvm-nm prints it as
                   VERS_x@@VERS_x).
    """
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # "address TYPE name" or "TYPE name". Undefined symbols are type U.
        if len(parts) >= 3 and len(parts[-2]) == 1 and parts[-2].isalpha():
            kind, name = parts[-2], parts[-1]
            if kind.upper() == "U":
                continue
            if kind.islower() and kind not in "wvu":
                continue
            if kind == "A" and VERSION_NODE.match(name):
                continue
            names.add(name)
        elif len(parts[-1]) == 1 and parts[-1].isalpha():
            continue
    return names


def parse_dumpbin(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        match = DUMPBIN_EXPORT.match(line)
        if match:
            names.add(match.group(1))
    return names


def audit_symbols(
    expected: set[str], exported: set[str]
) -> tuple[list[str], list[str]]:
    """Return (missing expected names, unexpected non-sb_* names)."""
    normalized = {normalize_export(name) for name in exported}
    missing = sorted(expected - normalized)
    foreign = sorted(
        name
        for name in normalized
        if not name.startswith("sb_") and not LINKER_BOUNDARY.match(name)
    )
    return missing, foreign


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(library: Path, checksum_path: Path) -> str:
    digest = sha256_file(library)
    checksum_path.write_text(f"{digest}  {library.name}\n", encoding="utf-8")
    return digest


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*?$", "", text, flags=re.M)


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"missing tool: {command[0]}") from exc
    if completed.returncode != 0:
        raise SystemExit(
            f"{' '.join(command)} failed ({completed.returncode}): {completed.stderr}"
        )
    return completed.stdout


def list_exports(
    library: Path, dumpbin: str | None, nm: str | None = None
) -> set[str]:
    suffix = library.suffix.lower()
    if suffix == ".dll":
        tool = dumpbin or "dumpbin"
        return parse_dumpbin(_run([tool, "/exports", str(library)]))
    if suffix == ".dylib":
        tool = nm or "nm"
        return parse_nm(_run([tool, "-gU", str(library)]))
    if suffix == ".so":
        # Android arm64 is audited with the NDK llvm-nm, not the host binutils nm.
        tool = nm or "nm"
        return parse_nm(_run([tool, "-D", "--defined-only", str(library)]))
    raise SystemExit(f"unsupported library type: {library}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--headers",
        type=Path,
        default=Path("spatialblend/api/c"),
        help="Directory of the C API headers",
    )
    parser.add_argument("--dumpbin", default=None, help="Path to dumpbin.exe")
    parser.add_argument(
        "--nm",
        default=None,
        help="nm or llvm-nm used for .so and .dylib exports",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--checksum",
        type=Path,
        default=None,
        help="Write a sha256sum sidecar for this library",
    )
    args = parser.parse_args(argv)

    library = args.library
    if not library.is_file():
        print(f"library not found: {library}", file=sys.stderr)
        return 1

    expected = expected_symbols(args.headers)
    exported = list_exports(library, args.dumpbin, args.nm)
    missing, foreign = audit_symbols(expected, exported)
    normalized = {normalize_export(name) for name in exported}
    sb_count = sum(1 for name in normalized if name.startswith("sb_"))
    digest = sha256_file(library)
    summary = (
        f"sha256={digest} exports={len(normalized)} os={sb_count} "
        f"expected={len(expected)} missing={len(missing)} other={len(foreign)}"
    )
    print(summary)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        lines = [summary]
        if missing:
            lines.append("missing:")
            lines.extend(f"  {name}" for name in missing)
        if foreign:
            lines.append("foreign:")
            lines.extend(f"  {name}" for name in foreign)
        args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing or foreign:
        for name in missing[:40]:
            print(f"missing {name}", file=sys.stderr)
        for name in foreign[:40]:
            print(f"foreign {name}", file=sys.stderr)
        return 1
    if args.checksum:
        write_sha256sums(library, args.checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
