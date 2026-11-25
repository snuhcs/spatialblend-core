#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""
Android ADB Runner - build and run any Bazel binary on device via ADB.

Works for tests (cc_test), benchmarks (cc_binary), and other executables.

Usage:
    # Build and run any target (test, benchmark, etc.)
    python run_android_test.py --target //path/to:target

    # Run pre-built binary
    python run_android_test.py --binary path/to/binary

    # Pass args to the binary (e.g., a gtest filter)
    python run_android_test.py --target //spatialblend/space:geometry_mediation_benchmark_test -- '--gtest_filter=GeometryMediationBenchmark.*'

    # Emulator, device, filter, etc.
    python run_android_test.py --avd Pixel_4_API_30 --target //path:test
    python run_android_test.py --device SERIAL --binary path/to/test
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def print_info(msg):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def print_emu(msg):
    print(f"{Colors.BLUE}[EMU]{Colors.NC} {msg}")


def run_cmd(cmd, check=True, capture_output=False, **kwargs):
    """Run a command and return the result."""
    if not capture_output:
        print_info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True, **kwargs)
    if check and result.returncode != 0:
        if capture_output:
            print_error(f"Command failed: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def find_adb():
    """Find ADB executable."""
    adb = os.environ.get('ADB')
    if adb and os.path.isfile(adb):
        return adb

    adb = shutil.which('adb')
    if adb:
        return adb

    # Check common locations
    android_home = os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_HOME')
    if android_home:
        adb_path = Path(android_home) / 'platform-tools' / 'adb'
        if adb_path.exists():
            return str(adb_path)

    raise RuntimeError("ADB not found. Set ADB or install Android SDK platform-tools.")


def find_emulator():
    """Find emulator executable."""
    emulator = shutil.which('emulator')
    if emulator:
        return emulator

    # Check common locations
    android_home = os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_HOME')
    if android_home:
        emulator_path = Path(android_home) / 'emulator' / 'emulator'
        if emulator_path.exists():
            return str(emulator_path)

    return None


def get_adb_cmd(adb, device=None):
    """Get ADB command with optional device specification."""
    cmd = [adb]
    if device:
        cmd.extend(['-s', device])
    return cmd


def list_avds(emulator):
    """List available AVDs."""
    if not emulator:
        print_error("Emulator not found. Please install Android SDK emulator.")
        return []

    result = subprocess.run([emulator, '-list-avds'], capture_output=True, text=True)
    avds = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    return avds


def list_devices(adb):
    """List connected devices."""
    result = subprocess.run([adb, 'devices', '-l'], capture_output=True, text=True)
    return result.stdout


def is_device_connected(adb):
    """Check if any device is connected."""
    result = subprocess.run([adb, 'devices'], capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().split('\n')[1:] if l.strip() and 'device' in l]
    return len(lines) > 0


def get_emulator_serial(adb, avd_name):
    """Get emulator serial by AVD name."""
    result = subprocess.run([adb, 'devices'], capture_output=True, text=True)
    for line in result.stdout.strip().split('\n')[1:]:
        if line.startswith('emulator-'):
            serial = line.split()[0]
            try:
                emu_result = subprocess.run(
                    [adb, '-s', serial, 'emu', 'avd', 'name'],
                    capture_output=True, text=True, timeout=5
                )
                emu_avd = emu_result.stdout.strip().split('\n')[0].strip()
                if emu_avd == avd_name:
                    return serial
            except (subprocess.TimeoutExpired, IndexError):
                continue
    return None


class EmulatorManager:
    """Manages Android emulator lifecycle."""

    def __init__(self, adb, emulator_path):
        self.adb = adb
        self.emulator_path = emulator_path
        self.process = None
        self.serial = None
        self.started_by_us = False

    def start(self, avd_name, headless=False, boot_timeout=120):
        """Start an emulator if not already running."""
        if not self.emulator_path:
            raise RuntimeError("Emulator not found. Please install Android SDK emulator.")

        # Check available AVDs
        avds = list_avds(self.emulator_path)
        if avd_name not in avds:
            print_error(f"AVD '{avd_name}' not found.")
            print_info("Available AVDs:")
            for avd in avds:
                print(f"  - {avd}")
            raise RuntimeError(f"AVD '{avd_name}' not found")

        # Check if already running
        serial = get_emulator_serial(self.adb, avd_name)
        if serial:
            print_emu(f"Emulator '{avd_name}' is already running (serial: {serial})")
            self.serial = serial
            return serial

        print_emu(f"Starting emulator: {avd_name}")

        # Build emulator command
        cmd = [self.emulator_path, '-avd', avd_name, '-no-snapshot-save']
        if headless:
            cmd.extend(['-no-window', '-no-audio', '-gpu', 'swiftshader_indirect'])

        # Start emulator
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.started_by_us = True

        print_emu(f"Emulator started with PID: {self.process.pid}")
        print_emu(f"Waiting for emulator to boot (timeout: {boot_timeout}s)...")

        # Wait for emulator to appear
        start_time = time.time()
        while time.time() - start_time < boot_timeout:
            time.sleep(2)

            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print_error("Emulator process terminated unexpectedly")
                print_error(f"stderr: {stderr.decode()}")
                raise RuntimeError("Emulator failed to start")

            # Try to get serial
            serial = get_emulator_serial(self.adb, avd_name)
            if serial:
                self.serial = serial
                print_emu(f"Emulator appeared as: {serial}")
                break

            print(".", end="", flush=True)

        print()

        if not self.serial:
            # Fallback: use first emulator found
            result = subprocess.run([self.adb, 'devices'], capture_output=True, text=True)
            for line in result.stdout.strip().split('\n')[1:]:
                if line.startswith('emulator-'):
                    self.serial = line.split()[0]
                    break

        if not self.serial:
            raise RuntimeError("Emulator did not start within timeout")

        # Wait for boot to complete
        print_emu("Waiting for boot to complete...")
        boot_start = time.time()
        while time.time() - boot_start < boot_timeout:
            try:
                result = subprocess.run(
                    [self.adb, '-s', self.serial, 'shell', 'getprop', 'sys.boot_completed'],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip() == '1':
                    print_emu("Emulator boot completed!")
                    return self.serial
            except subprocess.TimeoutExpired:
                pass

            time.sleep(2)
            print(".", end="", flush=True)

        print()
        print_warn("Boot completion check timed out, but continuing anyway...")
        return self.serial

    def stop(self):
        """Stop the emulator if we started it."""
        if self.started_by_us and self.process:
            print_emu(f"Stopping emulator (PID: {self.process.pid})...")

            # Try graceful shutdown via ADB
            if self.serial:
                try:
                    subprocess.run(
                        [self.adb, '-s', self.serial, 'emu', 'kill'],
                        timeout=10
                    )
                    time.sleep(2)
                except (subprocess.TimeoutExpired, Exception):
                    pass

            # Force kill if still running
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()


def check_device(adb_cmd):
    """Check if an Android device is connected."""
    result = run_cmd(adb_cmd + ['get-state'], check=False, capture_output=True)
    if result.returncode != 0:
        return False

    # Get device info
    model = run_cmd(adb_cmd + ['shell', 'getprop', 'ro.product.model'],
                    capture_output=True).stdout.strip()
    api = run_cmd(adb_cmd + ['shell', 'getprop', 'ro.build.version.sdk'],
                  capture_output=True).stdout.strip()

    # Check if emulator
    is_emulator = False
    try:
        qemu = run_cmd(adb_cmd + ['shell', 'getprop', 'ro.kernel.qemu'],
                       capture_output=True, check=False).stdout.strip()
        is_emulator = qemu == '1'
    except Exception:
        pass

    device_type = "emulator" if is_emulator else "device"
    print_info(f"Connected to: {model} (API {api}) [{device_type}]")
    return True


def setup_device(adb_cmd, test_dir):
    """Create test directory on device."""
    print_info(f"Setting up test directory: {test_dir}")
    run_cmd(adb_cmd + ['shell', f'mkdir -p {test_dir}'], check=False)


def push_binary(adb_cmd, binary_path, test_dir):
    """Push test binary to device."""
    binary_name = os.path.basename(binary_path)

    if not os.path.isfile(binary_path):
        print_error(f"Test binary not found: {binary_path}")
        sys.exit(1)

    print_info(f"Pushing test binary: {binary_name}")
    remote_path = f"{test_dir}/{binary_name}"
    run_cmd(adb_cmd + ['push', binary_path, remote_path])
    run_cmd(adb_cmd + ['shell', f'chmod +x {remote_path}'])

    return remote_path


def push_libraries(adb_cmd, binary_path, test_dir):
    """Push shared libraries from the same directory."""
    binary_dir = os.path.dirname(binary_path)

    for lib in Path(binary_dir).glob('*.so*'):
        if lib.is_file():
            print_info(f"Pushing library: {lib.name}")
            run_cmd(adb_cmd + ['push', str(lib), f"{test_dir}/{lib.name}"])


def run_test(adb_cmd, remote_binary, test_dir, args=None, timeout=300):
    """Run test on device."""
    binary_name = os.path.basename(remote_binary)
    test_args = ' '.join(args) if args else ''

    print_info(f"Running test: {binary_name} {test_args}")
    print()
    print("=" * 50)
    print("Test Output")
    print("=" * 50)
    print()

    cmd = f"cd {test_dir} && LD_LIBRARY_PATH={test_dir}:$LD_LIBRARY_PATH ./{binary_name} {test_args}"
    result = run_cmd(adb_cmd + ['shell', cmd], check=False)

    print()
    print("=" * 50)
    return result.returncode


def cleanup(adb_cmd, test_dir, binary_name, keep=False):
    """Clean up test files from device."""
    if not keep:
        print_info("Cleaning up test files...")
        run_cmd(adb_cmd + ['shell', f'rm -f {test_dir}/{binary_name}'], check=False)


def build_android_target(target):
    """Build a Bazel target for Android."""
    bazel = shutil.which('bazel') or shutil.which('bazelisk')
    if not bazel:
        print_error("Bazel not found. Please install Bazel or Bazelisk.")
        sys.exit(1)

    print_info(f"Building target: {target}")
    run_cmd([bazel, 'build', '--config=android', target])

    # Determine the output path
    target_path = target.lstrip('/')
    if ':' in target_path:
        package, name = target_path.rsplit(':', 1)
    else:
        package = target_path
        name = os.path.basename(package)

    binary_path = f"bazel-bin/{package}/{name}"
    if not os.path.exists(binary_path):
        binary_path = f"bazel-bin/{name}"

    if not os.path.exists(binary_path):
        print_error(f"Built binary not found at expected path: {binary_path}")
        print_info("Try using --binary flag with the actual path")
        sys.exit(1)

    return binary_path


def main():
    parser = argparse.ArgumentParser(
        description="Run tests on Android devices or emulators via ADB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run a pre-built test binary
    %(prog)s --binary bazel-bin/spatialblend/space/geometry_mediation_benchmark_test

    # Build and run a test target
    %(prog)s --target //spatialblend/space:geometry_mediation_benchmark_test

    # Run on emulator (starts if not running)
    %(prog)s --avd Pixel_4_API_30 --binary path/to/test

    # Run headless emulator (for CI)
    %(prog)s --avd Pixel_4_API_30 --headless --binary path/to/test

    # List available AVDs
    %(prog)s --list-avds

    # List connected devices
    %(prog)s --list-devices
        """
    )

    parser.add_argument('--binary', '-b',
                        help='Path to pre-built test binary')
    parser.add_argument('--target', '-t',
                        help='Bazel target to build and run (e.g., //path:test)')
    parser.add_argument('--device', '-d',
                        help='Target device serial (for multiple devices)')
    parser.add_argument('--avd', '-a',
                        help='AVD name to start if no device connected')
    parser.add_argument('--headless', action='store_true',
                        help='Run emulator in headless mode (no window)')
    parser.add_argument('--test-dir', default='/data/local/tmp/spatialblend_tests',
                        help='Directory on device for test files')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Test timeout in seconds (default: 300)')
    parser.add_argument('--boot-timeout', type=int, default=120,
                        help='Emulator boot timeout in seconds (default: 120)')
    parser.add_argument('--keep', action='store_true',
                        help='Keep binary on device after test')
    parser.add_argument('--filter', '-f',
                        help='Google Test filter pattern')
    parser.add_argument('--list-avds', action='store_true',
                        help='List available AVDs and exit')
    parser.add_argument('--list-devices', action='store_true',
                        help='List connected devices and exit')
    parser.add_argument('args', nargs='*',
                        help='Additional arguments to pass to the test')

    args = parser.parse_args()

    # Remote ADB server (e.g., when forwarding via tools/adb/setup.sh)
    os.environ.setdefault('ADB_SERVER_SOCKET', 'tcp:127.0.0.1:5037')

    # Find tools
    adb = find_adb()
    emulator = find_emulator()

    # Handle list commands
    if args.list_avds:
        print_info("Available Android Virtual Devices (AVDs):")
        avds = list_avds(emulator)
        if avds:
            for avd in avds:
                print(f"  - {avd}")
        else:
            print("  No AVDs found. Create one with: avdmanager create avd")
        sys.exit(0)

    if args.list_devices:
        print_info("Connected Android devices:")
        print(list_devices(adb))
        sys.exit(0)

    if not args.binary and not args.target:
        parser.error("Either --binary or --target must be specified")

    emulator_manager = None
    device = args.device

    try:
        # Start emulator if needed
        if args.avd and not is_device_connected(adb):
            if not emulator:
                print_error("Emulator not found. Please install Android SDK emulator.")
                sys.exit(1)

            emulator_manager = EmulatorManager(adb, emulator)
            device = emulator_manager.start(args.avd, args.headless, args.boot_timeout)

        adb_cmd = get_adb_cmd(adb, device)

        # Check device
        if not check_device(adb_cmd):
            print_error("No Android device connected or device not authorized.")
            print()
            print(list_devices(adb))
            print()
            if not args.avd:
                print_info("Tip: Use --avd to automatically start an emulator")
                print_info("     List AVDs with: --list-avds")
            sys.exit(1)

        # Build if needed
        if args.target:
            binary_path = build_android_target(args.target)
        else:
            binary_path = args.binary

        # Setup device
        setup_device(adb_cmd, args.test_dir)

        # Push files
        remote_binary = push_binary(adb_cmd, binary_path, args.test_dir)
        push_libraries(adb_cmd, binary_path, args.test_dir)

        # Build test arguments
        test_args = list(args.args) if args.args else []
        if args.filter:
            test_args.append(f'--gtest_filter={args.filter}')

        # Run test
        exit_code = run_test(adb_cmd, remote_binary, args.test_dir,
                             test_args, args.timeout)

        # Cleanup
        cleanup(adb_cmd, args.test_dir, os.path.basename(binary_path), args.keep)

        # Report result
        print()
        if exit_code == 0:
            print_info("Test PASSED")
        else:
            print_error(f"Test FAILED (exit code: {exit_code})")

    finally:
        # Stop emulator if we started it
        if emulator_manager:
            emulator_manager.stop()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
