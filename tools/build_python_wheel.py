# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

from util import *
import fnmatch
import shutil

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_host_platform():
    platform = get_platform()
    if platform == "linux":
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "windows":
        return "windows"
    else:
        raise ValueError(f"Unsupported platform: {platform}")


# (bazel target, bazel-bin output dir relative to WORKSPACE_DIR, wheel
# filename glob) for each buildable wheel. Kept as a dict so callers can
# request any combination (e.g. --pybind --capi builds both in one run).
WHEEL_SPECS = {
    "pybind": (
        "//spatialblend/api/python:spatialblend_wheel",
        "bazel-bin/spatialblend/api/python",
        "spatialblend-*.whl",
    ),
    "capi": (
        "//spatialblend/api/python:spatialblend_capi_wheel",
        "bazel-bin/spatialblend/api/python",
        "spatialblend_capi-*.whl",
    ),
}


def build_one(which, args, output_dir):
    target, bin_subdir, name_glob = WHEEL_SPECS[which]
    output_bin_dir = os.path.join(WORKSPACE_DIR, bin_subdir)

    print(f"Building wheel {target} with bazel")
    cmd = make_cmd(
        args.docker,
        args.build,
        args.debug,
        get_host_platform(),
        target,
        strip_never=getattr(args, "strip_never", False),
    )
    run_cmd(cmd)

    # look for the wheel that Bazel produced
    matches = fnmatch.filter(
        os.listdir(util.canon_path(output_bin_dir)), name_glob)
    if not matches:
        raise FileNotFoundError(
            f"No wheel matching pattern {name_glob} found in "
            f"{util.canon_path(output_bin_dir)}.\n"
            "• Did the Bazel build succeed?\n"
            "• Is the output_base you supplied to Bazel the same one this "
            "script is looking in?\n"
            "• Did the version/platform pattern in the glob need updating?")
    package_name = matches[0]

    package_path = canon_path(os.path.join(output_bin_dir, package_name))
    output_path = canon_path(os.path.join(output_dir, package_name))
    print(f"Copying {package_path} to {output_path}")
    shutil.copy2(package_path, output_path)

    if args.install:
        print(f"Installing {output_path}")
        run_cmd(f"python -m pip install --force-reinstall {output_path}")

    return output_path


if __name__ == "__main__":
    parser = get_argument_parser("Build Python wheel")
    parser.add_argument("--output", type=str,
                        default="output", help="Output directory")
    parser.add_argument("-i", "--install", action="store_true",
                        help="Install the wheel")
    parser.add_argument("--pybind", action="store_true",
                        help="Build the spatialblend pybind wheel "
                        "(cp310 C extension; this is the default when no "
                        "wheel-selecting flag is given)")
    parser.add_argument("--capi", action="store_true",
                        help="Build the spatialblend_capi wheel (pure "
                        "Python + ctypes, python_tag py3/abi none -- "
                        "importable from any CPython 3.6+ interpreter, "
                        "e.g. a consumer pinned to an older Python, "
                        "without needing a matching pybind ABI "
                        "build)")
    args = parser.parse_args()

    # set cwd to file directory
    os.chdir(os.path.dirname(__file__))

    output_dir = os.path.abspath(os.path.join(WORKSPACE_DIR, args.output))
    print(f"Output directory: {output_dir}")

    # clean output directory
    if not os.path.exists(output_dir):
        print("Output directory does not exist, creating it")
        os.makedirs(output_dir, exist_ok=True)

    # Which wheel(s) to build: with no flag, only the spatialblend pybind wheel.
    # --pybind and --capi can each be passed (together or apart) to build one
    # or both wheels in a single invocation.
    requested = []
    if args.pybind:
        requested.append("pybind")
    if args.capi:
        requested.append("capi")
    if not requested:
        requested = ["pybind"]

    built = []
    for which in requested:
        built.append(build_one(which, args, output_dir))

    print("Built wheel(s):")
    for path in built:
        print(f"  {path}")
