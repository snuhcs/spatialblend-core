# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

import os
import tarfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import *

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "..")
TARGET_NAME = "libspatialblend_c"
TARGET = f"//spatialblend/api/c:{TARGET_NAME}"
TARGET_PKG_PATH = f"bazel-bin/spatialblend/api/c/{TARGET_NAME}.tar"
TARGET_PKG_LOCAL_PATH = os.path.abspath(
    os.path.join(WORKSPACE_DIR, TARGET_PKG_PATH))


def safe_extract_tar(tar_path, extract_path):
    """Safely extract tar file with robust error handling."""

    def force_remove_directory(dir_path):
        """Force remove a directory and its contents."""
        if not os.path.exists(dir_path):
            return

        try:
            import platform
            if platform.system() == "Windows":
                # Prefer fast built-in removal to avoid external tool hangs
                try:
                    import subprocess
                    subprocess.run(
                        ["cmd", "/c", "rmdir", "/S", "/Q", dir_path],
                        capture_output=True,
                        check=False,
                        timeout=20,
                    )
                    if not os.path.exists(dir_path):
                        return
                except Exception:
                    pass

            # Robust fallback: clear read-only attributes and retry removal
            def _on_rm_error(func, path, exc_info):
                try:
                    import stat
                    if platform.system() == "Windows":
                        try:
                            os.chmod(path, stat.S_IWRITE)
                        except Exception:
                            pass
                    os.chmod(path, 0o777)
                    func(path)
                except Exception:
                    pass

            shutil.rmtree(dir_path, onerror=_on_rm_error)
        except Exception:
            pass

    try:
        # Proactively clear the destination directory before extraction
        if os.path.exists(extract_path):
            print(f"Cleaning existing directory: {extract_path}")
            # Remove everything except the tar file
            for item in os.listdir(extract_path):
                item_path = os.path.join(extract_path, item)
                if os.path.isfile(item_path) and item.endswith('.tar'):
                    continue  # Preserve tar file
                try:
                    if os.path.isdir(item_path):
                        force_remove_directory(item_path)
                    else:
                        # Windows: Remove read-only attribute first
                        import platform
                        if platform.system() == "Windows":
                            try:
                                import stat
                                current_attrs = os.stat(
                                    item_path).st_file_attributes
                                os.chmod(item_path, stat.S_IWRITE)
                            except:
                                pass
                        os.chmod(item_path, 0o666)
                        os.remove(item_path)
                except:
                    pass

        # Create fresh directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Extract directly to target directory with proper permissions
        with tarfile.open(tar_path, "r") as tar:
            # Extract files one by one to handle locked files gracefully
            for member in tar.getmembers():
                try:
                    # Windows: Remove existing file/directory before extraction
                    import platform
                    if platform.system() == "Windows":
                        member_path = os.path.join(
                            extract_path, member.name)
                        if os.path.exists(member_path):
                            try:
                                if member.isdir():
                                    force_remove_directory(member_path)
                                else:
                                    # Remove read-only attribute first
                                    try:
                                        import stat
                                        current_attrs = os.stat(
                                            member_path).st_file_attributes
                                        os.chmod(
                                            member_path, stat.S_IWRITE)
                                    except:
                                        pass
                                    os.chmod(member_path, 0o666)
                                    os.remove(member_path)
                            except Exception as e:
                                print(
                                    f"Warning: Could not remove existing {member.name}: {e}")
                                continue

                    # Set permissions before extraction (Python 3.10 compatible)
                    member.mode = 0o666
                    tar.extract(member, extract_path)
                    # Set permissions after extraction
                    member_path = os.path.join(extract_path, member.name)
                    if member.isdir():
                        try:
                            os.chmod(member_path, 0o777)
                        except:
                            pass
                    else:
                        try:
                            # Windows: Ensure file is writable
                            import platform
                            if platform.system() == "Windows":
                                try:
                                    import stat
                                    current_attrs = os.stat(
                                        member_path).st_file_attributes
                                    os.chmod(
                                        member_path, stat.S_IWRITE | stat.S_IREAD)
                                except:
                                    pass
                            os.chmod(member_path, 0o666)
                        except:
                            pass
                except PermissionError as e:
                    print(f"Warning: Could not extract {member.name}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error extracting {member.name}: {e}")
                    continue

        return True

    except PermissionError as e:
        print(f"Permission error during extraction: {e}")
        print("Possible solutions:")
        print("1. Close any applications that might be using the files")
        print("2. Run the script as Administrator")
        print("3. Check if antivirus is blocking the operation")
        print("4. Try extracting to a different directory")
        print("5. On Windows: Right-click and 'Run as Administrator'")
        return False
    except Exception as e:
        print(f"Unexpected error during extraction: {e}")
        return False


def safe_copy(src_path, dst_path):
    """Safely copy file with robust error handling."""
    print(f"Attempting to copy {src_path} to {dst_path}")

    if not os.path.exists(src_path):
        print(f"Source file does not exist: {src_path}")
        return False

    try:
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        # Try to remove existing file if it exists
        if os.path.exists(dst_path):
            try:
                os.chmod(dst_path, 0o666)
                os.remove(dst_path)
            except (PermissionError, OSError) as e:
                print(
                    f"Warning: Could not remove existing file {dst_path}: {e}")

        shutil.copy2(src_path, dst_path)
        print(f"Successfully copied {src_path} to {dst_path}")
        return True
    except PermissionError as e:
        print(f"Permission error copying {src_path} to {dst_path}: {e}")
        print("Possible solutions:")
        print("1. Close any applications that might be using the files")
        print("2. Run the script as Administrator")
        print("3. Check if antivirus is blocking the operation")
        print("4. Try copying to a different directory")
        return False
    except Exception as e:
        print(f"Unexpected error during copy: {e}")
        return False


def move_pdbs_into_lib(output_root_dir):
    """On Windows, move top-level .pdb files into the lib directory."""
    try:
        import platform
        if platform.system() != "Windows":
            return

        if not os.path.isdir(output_root_dir):
            return

        lib_dir = os.path.join(output_root_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)

        for item in os.listdir(output_root_dir):
            if not item.lower().endswith(".pdb"):
                continue
            src = os.path.join(output_root_dir, item)
            # Only move files at the root level
            if not os.path.isfile(src):
                continue
            dst = os.path.join(lib_dir, item)
            try:
                # Ensure writable before move
                try:
                    os.chmod(src, 0o666)
                except Exception:
                    pass
                # If destination exists, remove it first
                if os.path.exists(dst):
                    try:
                        os.chmod(dst, 0o666)
                        os.remove(dst)
                    except Exception:
                        pass
                shutil.move(src, dst)
                print(f"Moved PDB to lib: {item}")
            except Exception as e:
                print(f"Warning: Could not move PDB {item} to lib: {e}")
    except Exception:
        # Don't fail the build if this housekeeping step has issues
        pass


def get_target_compilation_mode(args: argparse.Namespace):
    target_compilation_modes = []

    if args.debug:
        target_compilation_modes.append(True)
    else:
        target_compilation_modes.append(False)

    if args.all:
        target_compilation_modes = [True, False]

    return target_compilation_modes


def get_target_platform(args: argparse.Namespace):
    def get_target_platform_internal(docker: bool, android: bool, visionos: bool):
        platform = get_platform()
        
        # Base platforms to build
        platforms = []
        
        # Only include host platform when not using Docker and not building only android
        # When -a is used, build only android (exclude host)
        if not docker and not android:
            platforms.append("host")
        
        # Add Docker platforms if requested
        if docker:
            platforms.extend(["linux", "android"])
        else:
            # Add platform-specific targets when not using Docker
            if platform == "linux":
                platforms.extend(["android"] if android else [])
            elif platform == "darwin":
                platforms.extend(["android"] if android else [])
                platforms.extend(["visionos"] if visionos else [])
            # Windows only builds for host (no additional platforms)
        
        return platforms

    if args.all:
        return get_target_platform_internal(args.docker, True, True)
    else:
        return get_target_platform_internal(args.docker, args.android, args.visionos)


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


def is_error_line(line):
    """Check if a line contains error indicators."""
    line_lower = line.lower()
    error_indicators = [
        'error:', 'error ', 'failed', 'failure', 'fatal:', 'fatal ',
        'exception:', 'exception ', 'traceback', 'abort', 'aborted',
        'cannot', 'unable to', 'not found', 'no such file',
        'permission denied', 'access denied', 'timeout', 'killed',
        'segmentation fault', 'core dumped', 'undefined reference',
        'undefined symbol', 'linker error', 'compilation failed',
        'build failed', 'test failed', 'assertion failed'
    ]
    return any(indicator in line_lower for indicator in error_indicators)


def display_error_lines(output_lines, platform, build_type):
    """Display error lines from build output. Returns count of error lines."""
    error_lines = [line for line in output_lines if is_error_line(line)]
    if error_lines:
        print(f"\n⚠️  Errors detected in {platform} {build_type} build:")
        for line in error_lines:
            print(f"   {line}")
        print()  # Add spacing after errors
    return len(error_lines)


def save_build_output(output_lines, output_dir, platform, build_type, command):
    """Save build output to a text file next to the platform directory."""
    import datetime
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filename
    output_filename = f"build_output_{platform}_{build_type}_{timestamp}.txt"
    output_path = os.path.join(os.path.dirname(output_dir), output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Build Output for {platform} ({build_type})\n")
            f.write(f"Command: {command}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            for line in output_lines:
                f.write(line + "\n")
        
        print(f"Build output saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save build output: {e}")


def build_single_target(args, target_platform, target_compilation_mode, output_dir, verbose, docker_hash=None):
    """Build a single target platform/compilation mode combination."""
    # Thread-safe print function
    print_lock = threading.Lock()
    
    def safe_print(*print_args, **kwargs):
        with print_lock:
            print(*print_args, **kwargs)
    
    try:
        cmd = make_cmd(
            args.docker,
            args.build,
            target_compilation_mode,
            target_platform,
            TARGET,
            strip_never=getattr(args, "strip_never", False),
        )

        if target_platform == "":
            target_platform = get_platform()

        if target_platform == "host":
            target_platform = get_host_platform()

        build_type = "debug" if target_compilation_mode else "release"
        
        if verbose:
            safe_print(f"Running {target_platform} {build_type}... ({cmd})")
        else:
            safe_print(f"Building {target_platform} {build_type} (output will be saved to file)...")

        # Capture output for logging
        output_lines = []
        error_count = 0
        
        if args.docker:
            try:
                if verbose:
                    # Show output in terminal and capture it
                    run_cmd_docker(cmd, docker_hash)
                    output_lines = ["Build completed (output shown in terminal)"]
                else:
                    # Capture output silently
                    output_lines = run_cmd_docker_with_output(cmd, docker_hash)
                    # Show error lines even in quiet mode
                    error_count = display_error_lines(output_lines, target_platform, build_type)
            except ValueError as e:
                safe_print(f"Error: {e}")
                return {"success": False, "platform": target_platform, "build_type": build_type, "error": str(e)}
        else:
            if verbose:
                # Show output in terminal and capture it
                run_cmd(cmd)
                output_lines = ["Build completed (output shown in terminal)"]
            else:
                # Capture output silently
                output_lines = run_cmd_with_output(cmd)
                # Show error lines even in quiet mode
                error_count = display_error_lines(output_lines, target_platform, build_type)

        os.makedirs(os.path.join(output_dir, target_platform), exist_ok=True)

        # copy and extract tar to output directory
        if args.docker:
            try:
                copy_docker(TARGET_PKG_PATH, os.path.join(
                    output_dir, target_platform), docker_hash)
            except ValueError as e:
                safe_print(f"Error: {e}")
                return {"success": False, "platform": target_platform, "build_type": build_type, "error": str(e)}
        else:
            # Use safe_copy instead of regular copy
            src_path = TARGET_PKG_LOCAL_PATH
            dst_path = os.path.join(
                output_dir, target_platform, f"{TARGET_NAME}.tar")

            if not safe_copy(src_path, dst_path):
                safe_print(f"Failed to copy {src_path} to {dst_path}")
                return {"success": False, "platform": target_platform, "build_type": build_type, "error": "Copy failed"}

        safe_print(f"Copying {TARGET_PKG_LOCAL_PATH} to {os.path.join(output_dir, target_platform)}")

        tar_path = os.path.join(output_dir, target_platform, f"{TARGET_NAME}.tar")

        # Verify the tar file exists before extraction
        if not os.path.exists(tar_path):
            safe_print(f"Tar file does not exist: {tar_path}")
            return {"success": False, "platform": target_platform, "build_type": build_type, "error": "Tar file missing"}

        # Extract directly to final location with error handling
        final_path = os.path.join(output_dir, target_platform)

        # Extract with error handling
        if not safe_extract_tar(tar_path, final_path):
            safe_print(f"Failed to extract {tar_path}")
            return {"success": False, "platform": target_platform, "build_type": build_type, "error": "Extraction failed"}

        # Housekeeping: ensure Windows PDBs live under lib/
        try:
            move_pdbs_into_lib(final_path)
        except Exception:
            pass

        safe_print(f"Successfully built and extracted {target_platform} {build_type} to: {final_path}")

        # Save build output to text file
        save_build_output(output_lines, final_path, target_platform, build_type, cmd)

        # Change file permissions before removing
        if os.path.exists(tar_path):
            try:
                os.chmod(tar_path, 0o777)  # Give full permissions
                os.remove(tar_path)
            except PermissionError:
                safe_print(f"Warning: Could not remove {tar_path} - file may be in use")

        return {
            "success": True, 
            "platform": target_platform, 
            "build_type": build_type, 
            "error_count": error_count
        }
        
    except Exception as e:
        safe_print(f"Unexpected error building {target_platform} {build_type}: {e}")
        return {"success": False, "platform": target_platform, "build_type": build_type, "error": str(e)}


if __name__ == "__main__":
    parser = get_argument_parser("Build C API")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory"
    )
    args = parser.parse_args()

    # set cwd to file directory
    os.chdir(os.path.dirname(__file__))

    output_dir = os.path.abspath(os.path.join(WORKSPACE_DIR, args.output))
    print(f"Output directory: {output_dir}")

    # clean output directory
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist, creating it")
        os.makedirs(output_dir, exist_ok=True)

    target_compilation_modes = get_target_compilation_mode(args)
    target_platforms = get_target_platform(args)
    print(f"Building for {target_platforms} with {target_compilation_modes}")

    if getattr(args, "clean_cache", False):
        print("Resetting Bazel cache (--clean-cache)...")
        for debug in target_compilation_modes:
            clean_bazel_cache(args.docker, debug, target_platforms, args.docker_hash)

    # Create build tasks
    build_tasks = []
    for target_platform in target_platforms:
        for target_compilation_mode in target_compilation_modes:
            build_tasks.append((target_platform, target_compilation_mode))

    print(f"Starting {len(build_tasks)} build tasks...")

    # Track errors across all builds
    total_errors = 0
    builds_with_errors = []
    successful_builds = []

    # Separate Docker and host builds to avoid contention
    docker_tasks = [(p, c) for p, c in build_tasks if p in ["linux", "android"]]
    host_tasks = [(p, c) for p, c in build_tasks if p not in ["linux", "android"]]
    
    def run_builds_sequential(tasks, is_docker):
        """Run builds sequentially for a given task list."""
        if not tasks:
            return
            
        for target_platform, target_compilation_mode in tasks:
            # Only turn off verbose when docker mode is on, otherwise each build is anyway done in host side one by one
            effective_verbose = args.verbose if is_docker else True
            try:
                result = build_single_target(
                    args, 
                    target_platform, 
                    target_compilation_mode, 
                    output_dir, 
                    effective_verbose, 
                    args.docker_hash
                )
                if result["success"]:
                    successful_builds.append(f"{result['platform']} {result['build_type']}")
                    if result.get("error_count", 0) > 0:
                        total_errors += result["error_count"]
                        builds_with_errors.append(f"{result['platform']} {result['build_type']}")
                else:
                    builds_with_errors.append(f"{result['platform']} {result['build_type']}")
                    print(f"❌ Build failed for {result['platform']} {result['build_type']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                builds_with_errors.append(f"{target_platform} {target_compilation_mode}")
                print(f"❌ Unexpected error in {target_platform} {target_compilation_mode}: {e}")
    
    # Run Docker builds first (if any)
    if docker_tasks:
        print(f"Running {len(docker_tasks)} Docker builds sequentially...")
        run_builds_sequential(docker_tasks, is_docker=True)
    
    # Then run host builds (if any)
    if host_tasks:
        print(f"Running {len(host_tasks)} host builds sequentially...")
        run_builds_sequential(host_tasks, is_docker=False)

    print(f"\nCompleted {len(successful_builds)} successful builds out of {len(build_tasks)} total tasks.")

    # Print summary
    print("\n" + "=" * 60)
    if total_errors == 0:
        print("✅ All builds completed successfully with no errors!")
    else:
        print(f"⚠️  Build completed with {total_errors} total errors across {len(builds_with_errors)} builds:")
        for build in builds_with_errors:
            print(f"   - {build}")
        print("\nCheck the saved output files for detailed error information.")
    print("=" * 60)
