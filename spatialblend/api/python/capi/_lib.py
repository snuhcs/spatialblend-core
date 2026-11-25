# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Shared-library discovery and loading for spatialblend_capi.

Discovery order (first match wins):
  1. ``SPATIALBLEND_LIB`` env var (or the old name ``SPATIALBLEND_V2_LIB``, kept
     for 1.0) -- an explicit path to the shared library file (not a
     directory). Set this to point at a bazel-bin build output, e.g.::

         export SPATIALBLEND_LIB=/path/to/bazel-bin/spatialblend/api/c/libspatialblend.dylib

  2. The packaged data directory next to this module
     (``capi/_data/<libname>``), populated when this package is installed
     from the ``spatialblend_capi`` wheel (see spatialblend/api/python/BUILD).
  3. A plain system search: ``ctypes.util.find_library`` plus a short list
     of conventional install locations (virtualenv/conda lib dirs,
     /usr/local/lib, the bazel-bin tree relative to this checkout).

This module has no dependency on the Python version used to build
libspatialblend -- it is a plain C shared library loaded via ``ctypes.CDLL``,
so any CPython interpreter (2.7 is not supported only because this package
uses f-strings) can load the exact same binary artifact.
"""
import ctypes
import ctypes.util
import os
import platform
import sys

_ENV_VARS = ("SPATIALBLEND_LIB", "SPATIALBLEND_V2_LIB")

_LIB_BASENAME_BY_SYSTEM = {
    "Linux": "libspatialblend.so",
    "Darwin": "libspatialblend.dylib",
    "Windows": "spatialblend.dll",
}


class LibraryNotFoundError(RuntimeError):
    """Raised when libspatialblend cannot be located or loaded."""


def _candidate_basenames():
    system = platform.system()
    if system == "Windows":
        return ["spatialblend.dll", "libspatialblend.dll"]
    if system == "Darwin":
        return ["libspatialblend.dylib", "libspatialblend_d.dylib"]
    # Linux and anything else that behaves like it (BSDs, etc).
    return ["libspatialblend.so", "libspatialblend_d.so"]


def _packaged_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")


def _repo_bazel_bin_candidates():
    """Best-effort guesses for a dev checkout that never ran the wheel build.

    Walks up from this file looking for `bazel-bin/spatialblend/api/c` (the
    libspatialblend output under the spatialblend-core repo root), so
    `python -c "import capi"` works straight out of a bazel build without
    having to set SPATIALBLEND_LIB by hand.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parts = here.split(os.sep)
    candidates = []
    for i in range(len(parts), 0, -1):
        root = os.sep.join(parts[:i])
        bazel_bin = os.path.join(root, "bazel-bin", "spatialblend", "api", "c")
        if os.path.isdir(bazel_bin):
            candidates.append(bazel_bin)
    return candidates


def _system_search_dirs():
    dirs = []
    for env_var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
        value = os.environ.get(env_var)
        if not value:
            continue
        dirs.extend(value.split(os.pathsep))
    dirs.extend([
        "/usr/local/lib",
        "/opt/homebrew/lib",
        sys.prefix and os.path.join(sys.prefix, "lib") or "",
    ])
    return [d for d in dirs if d]


def find_library_path():
    """Return the filesystem path to libspatialblend, or raise.

    See module docstring for the discovery order.
    """
    # 1. Explicit override.
    for env_var in _ENV_VARS:
        env_path = os.environ.get(env_var)
        if env_path:
            if os.path.isfile(env_path):
                return env_path
            raise LibraryNotFoundError(
                f"{env_var}={env_path!r} does not point at an existing file")

    names = _candidate_basenames()

    # 2. Packaged data directory (wheel install).
    data_dir = _packaged_data_dir()
    for name in names:
        candidate = os.path.join(data_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # 3. System / dev-checkout search.
    for directory in _repo_bazel_bin_candidates() + _system_search_dirs():
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate

    # 4. ctypes.util.find_library as a last resort (relies on the platform's
    #    own dynamic-linker search path / install_name conventions).
    found = ctypes.util.find_library("spatialblend")
    if found:
        return found

    raise LibraryNotFoundError(
        "Could not locate libspatialblend (looked for "
        f"{names!r}). Set {_ENV_VARS[0]} to an explicit path, e.g.\n"
        f"  export {_ENV_VARS[0]}=/path/to/bazel-bin/spatialblend/api/c/"
        + names[0])


_lib_cache = None
_lib_path_cache = None


def get_lib():
    """Load (once) and return the libspatialblend ctypes.CDLL handle."""
    global _lib_cache, _lib_path_cache
    if _lib_cache is not None:
        return _lib_cache
    path = find_library_path()
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise LibraryNotFoundError(
            f"Found libspatialblend at {path!r} but ctypes.CDLL() failed to "
            f"load it: {exc}") from exc
    _lib_cache = lib
    _lib_path_cache = path
    return lib


def loaded_library_path():
    """Path of the currently-loaded library, or None if not loaded yet."""
    return _lib_path_cache


def reset_for_testing():
    """Test-only hook: forget the cached CDLL so a different lib can load."""
    global _lib_cache, _lib_path_cache
    _lib_cache = None
    _lib_path_cache = None
