# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

import builtins
from genericpath import isfile
import os
import argparse
import platform as plf
import shlex
import shutil
import subprocess
import multiprocessing

# ANSI color codes
BLUE = '\033[94m'
RESET = '\033[0m'
PREFIX = f"{BLUE}SPATIALBLEND:{RESET} "

# Save the original print function
original_print = builtins.print


def custom_print(*args, **kwargs):
    """Custom print function that adds the SpatialBlend prefix to all prints."""
    original_print(PREFIX, *args, **kwargs)


# Override the default print function
builtins.print = custom_print

PLATFORM = {
    "linux": "linux",
    "darwin": "macos",
    "macos": "macos",  # Allow both darwin and macos as keys
    "visionos": "visionos",
    "android": "android",
    "windows": "windows"
}


def is_windows():
    return plf.system() == "Windows"


def is_linux():
    return plf.system() == "Linux"


def is_macos():
    return plf.system() == "Darwin"


def is_cygwin():
    return plf.system().startswith("CYGWIN_NT")


def get_platform():
    return plf.system().lower()


def canon_path(path):
    if is_windows() or is_cygwin():
        return path.replace('\\', '/')
    return path


def get_output_base_path(docker: bool, debug: bool, platform: str):
    # Get base directory for bazel output
    if docker:
        home_dir = "/root"
    elif is_windows():
        home_dir = "C:/"
    else:
        home_dir = os.path.expanduser("~")
    build_type = "d" if debug else "o"

    # Create a unique path for each platform/build combination
    if platform == "host":
        platform_name = PLATFORM[get_platform()]
    else:
        platform_name = PLATFORM[platform]

    return canon_path(os.path.join(home_dir, ".cache", "bazel",
                                   f"sb_{platform_name}_{build_type}"))


def get_bazel_options(
        debug: bool,
        platform: str,
        strip_never: bool = False):
    opt = "--copt=-DSPATIALBLEND_BUILD_DLL "
    opt += "-c " + ("dbg" if debug else "opt") + " "
    # Allow forcing strip=never even in release builds
    opt += "--strip " + ("never" if (debug or strip_never) else "always") + " "
    if platform != "host" and platform != "linux":
        opt += f"--config {PLATFORM[platform]} "
    if platform == "windows":
        opt += "--features=-compiler_param_file --features=-archive_param_file "
    return opt


def _maybe_fix_python_executable(args):
    """If the command starts with 'python' but that binary is missing, try 'python3'.

    This provides a systematic fallback for environments where only 'python3'
    exists on PATH.
    """
    if not args:
        return args

    # Only rewrite when explicitly using 'python'
    if args[0] != "python":
        return args

    # If 'python' actually exists, keep as-is.
    if shutil.which("python"):
        return args

    # Fall back to 'python3' when available.
    if shutil.which("python3"):
        args[0] = "python3"
    return args


def run_cmd(cmd):
    args = shlex.split(cmd)
    args = _maybe_fix_python_executable(args)
    subprocess.call(args, cwd=os.getcwd())


def run_cmd_with_output(cmd):
    """Run command and capture output, returning lines as list."""
    args = shlex.split(cmd)
    args = _maybe_fix_python_executable(args)
    try:
        result = subprocess.run(args, cwd=os.getcwd(), capture_output=True, text=True)
        # Combine stdout and stderr
        output = result.stdout + result.stderr
        return output.splitlines() if output else []
    except Exception as e:
        return [f"Error running command: {e}"]


def copy(src, dst, overwrite=False):
    if not os.path.exists(src):
        print(f"Error: Source file/directory does not exist: {src}")
        return False

    if not os.path.exists(dst):
        subprocess.call(['mkdir', '-p', f'{os.path.normpath(dst)}'])

    # append filename to dst directory
    dst = canon_path(os.path.join(dst, os.path.basename(src)))

    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore_dangling_symlinks=True,
                            symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return True
    except FileNotFoundError:
        print(f"Error: Could not find source file: {src}")
        return False
    except Exception as e:
        print(f"Error copying {src} to {dst}: {str(e)}")
        return False


def make_cmd(
    docker: bool,
    build_only: bool,
    debug: bool,
    platform: str,
    target: str,
    strip_never: bool = False,
):
    cmd = "bazel" + " "
    cmd += "--output_base " + \
        get_output_base_path(docker, debug, platform) + " "
    cmd += ("build" if build_only else "test") + " "
    cmd += get_bazel_options(debug, platform, strip_never)
    cmd += target
    return cmd


def get_dst_path(base_dir, target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(base_dir, target_platform, build)
    return canon_path(path)


def clean_bazel_cache(docker: bool, debug: bool, platforms: list, docker_hash=None):
    """Reset Bazel cache (output bases) to fix stale repo symlinks (e.g. after NDK path change)."""
    bases_to_clean = set()
    for platform in platforms:
        if platform == "host":
            platform = PLATFORM.get(plf.system().lower(), "linux")
        bases_to_clean.add(get_output_base_path(docker, debug, platform))

    if docker:
        from .docker import run_cmd_docker, get_container_hash
        hash_id = docker_hash or (get_container_hash()[0] if get_container_hash() else None)
        if not hash_id:
            print("Warning: No Docker container found, cannot clean cache inside container")
            return
        paths = " ".join(bases_to_clean)
        run_cmd_docker(f"rm -rf {paths} 2>/dev/null; echo Cleared Bazel cache", hash_id)
    else:
        import shutil
        for base in bases_to_clean:
            if os.path.exists(base):
                try:
                    shutil.rmtree(base)
                    print(f"Cleared Bazel cache: {base}")
                except Exception as e:
                    print(f"Warning: Could not remove {base}: {e}")


def get_argument_parser(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-all', action="store_true", default=False,
                        help='Build all platforms (default = false)')
    parser.add_argument('-a', '--android', action="store_true", default=False,
                        help='Build android (default = false)')
    parser.add_argument('-v', '--visionos', action="store_true", default=False,
                        help='Build visionos (default = false)')
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android ')
    parser.add_argument('-b', '--build', action="store_true", default=True,
                        help='Build only, only affects to local (default = true)')
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries for android from docker (assuming that the current docker engine has devcontainer built with a /.devcontainer')
    parser.add_argument('-dh', '--docker_hash', type=str, default=None,
                        help='Docker container ID/hash to use for building. If not specified, will auto-detect the first spatialblend-core container.')
    parser.add_argument('-sn', '--strip-never', action="store_true", default=False,
                        help='Force Bazel --strip=never even for release builds')
    parser.add_argument('--verbose', action="store_true", default=True,
                        help='Show build output in terminal (default: quiet, output saved to files)')
    parser.add_argument('--clean-cache', action="store_true", default=False,
                        help='Reset Bazel cache before building (use after NDK path change)')
    return parser
