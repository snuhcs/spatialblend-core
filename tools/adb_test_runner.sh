#!/bin/bash
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

# ADB Test Runner for Bazel Android Tests
#
# This script pushes a native test binary to an Android device/emulator and executes it.
# Usage: adb_test_runner.sh <test_binary> [test_args...]
#
# Environment variables:
#   ADB_SERVER_SOCKET - Remote ADB server (e.g., tcp:127.0.0.1:5037 when forwarding via tools/adb/setup.sh)
#   ADB_DEVICE       - Target device serial (optional)
#   ADB_AVD          - AVD name to explicitly use/start (optional)
#   ADB_TEST_DIR     - Directory on device to push tests (default: /data/local/tmp/spatialblend_tests)
#   ADB_TIMEOUT      - Test timeout in seconds (default: 300)
#   ADB_KEEP_BINARY  - Keep binary on device after test (default: false)
#   ADB_EMULATOR_OPTS - Additional emulator options (optional)

set -e

# Remote ADB server (e.g., when forwarding via tools/adb/setup.sh)
ADB_SERVER_SOCKET="${ADB_SERVER_SOCKET:-tcp:127.0.0.1:5037}"
export ADB_SERVER_SOCKET

# Configuration
ADB="${ADB:-adb}"
EMULATOR="${EMULATOR:-emulator}"
ADB_TEST_DIR="${ADB_TEST_DIR:-/data/local/tmp/spatialblend_tests}"
ADB_TIMEOUT="${ADB_TIMEOUT:-300}"
ADB_KEEP_BINARY="${ADB_KEEP_BINARY:-false}"
ADB_AVD="${ADB_AVD:-}"
ADB_EMULATOR_OPTS="${ADB_EMULATOR_OPTS:-}"
EMULATOR_BOOT_TIMEOUT="${EMULATOR_BOOT_TIMEOUT:-120}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track if we started the emulator
EMULATOR_STARTED=false
EMULATOR_PID=""

# Print with color
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_blue() {
    echo -e "${BLUE}[EMU]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 <test_binary> [test_args...]"
    echo "       $0 --list-avds"
    echo "       $0 --list-devices"
    echo ""
    echo "Environment variables:"
    echo "  ADB_DEVICE         - Target device serial (optional)"
    echo "  ADB_AVD            - AVD name to explicitly use/start"
    echo "  ADB_TEST_DIR       - Directory on device (default: /data/local/tmp/spatialblend_tests)"
    echo "  ADB_TIMEOUT        - Test timeout in seconds (default: 300)"
    echo "  ADB_KEEP_BINARY    - Keep binary on device after test (default: false)"
    echo "  ADB_EMULATOR_OPTS  - Additional emulator options (e.g., '-no-window -no-audio')"
    echo "  EMULATOR_BOOT_TIMEOUT - Emulator boot timeout in seconds (default: 120)"
    echo ""
    echo "Examples:"
    echo "  # Run on connected device"
    echo "  $0 ./my_test"
    echo ""
    echo "  # Run on specific emulator AVD"
    echo "  ADB_AVD=Pixel_4_API_30 $0 ./my_test"
    echo ""
    echo "  # Run headless emulator"
    echo "  ADB_AVD=Pixel_4_API_30 ADB_EMULATOR_OPTS='-no-window -no-audio' $0 ./my_test"
    exit 1
}

# List available AVDs
list_avds() {
    print_info "Available Android Virtual Devices (AVDs):"
    echo ""
    if command -v "$EMULATOR" &> /dev/null; then
        $EMULATOR -list-avds 2>/dev/null || echo "No AVDs found. Create one with: avdmanager create avd"
    else
        print_error "Emulator not found. Set ANDROID_HOME or add emulator to PATH."
        exit 1
    fi
    exit 0
}

# List connected devices
list_devices() {
    print_info "Connected Android devices:"
    echo ""
    $ADB devices -l
    exit 0
}

# Check if any device is connected
is_device_connected() {
    local device_count=$($ADB devices 2>/dev/null | grep -E "device$" | wc -l)
    [ "$device_count" -gt 0 ]
}

# Get list of connected devices
get_devices() {
    $ADB devices 2>/dev/null | grep -E "device$" | cut -f1
}

# Get list of physical devices (non-emulators)
get_physical_devices() {
    get_devices | grep -v "^emulator-"
}

# Get list of emulators
get_emulators() {
    get_devices | grep "^emulator-"
}

# Count connected devices
count_devices() {
    get_devices | wc -l | tr -d ' '
}

# Select best device (prefer physical over emulator)
select_best_device() {
    local physical=$(get_physical_devices | head -1)
    if [ -n "$physical" ]; then
        echo "$physical"
    else
        get_emulators | head -1
    fi
}

# Get emulator serial by AVD name
get_emulator_serial() {
    local avd_name="$1"
    # Emulators typically use emulator-5554, emulator-5556, etc.
    for serial in $($ADB devices | grep "emulator-" | cut -f1); do
        local emu_avd=$($ADB -s "$serial" emu avd name 2>/dev/null | head -1 | tr -d '\r')
        if [ "$emu_avd" = "$avd_name" ]; then
            echo "$serial"
            return 0
        fi
    done
    return 1
}

# Start emulator
start_emulator() {
    local avd_name="$1"

    # Check if emulator command exists
    if ! command -v "$EMULATOR" &> /dev/null; then
        # Try to find emulator in ANDROID_HOME
        if [ -n "$ANDROID_HOME" ] && [ -x "$ANDROID_HOME/emulator/emulator" ]; then
            EMULATOR="$ANDROID_HOME/emulator/emulator"
        elif [ -n "$ANDROID_SDK_HOME" ] && [ -x "$ANDROID_SDK_HOME/emulator/emulator" ]; then
            EMULATOR="$ANDROID_SDK_HOME/emulator/emulator"
        else
            print_error "Emulator not found. Please install Android SDK emulator."
            exit 1
        fi
    fi

    # Check if AVD exists
    if ! $EMULATOR -list-avds 2>/dev/null | grep -q "^${avd_name}$"; then
        print_error "AVD '$avd_name' not found."
        print_info "Available AVDs:"
        $EMULATOR -list-avds 2>/dev/null
        exit 1
    fi

    # Check if emulator for this AVD is already running
    if serial=$(get_emulator_serial "$avd_name"); then
        print_blue "Emulator '$avd_name' is already running (serial: $serial)"
        ADB_DEVICE="$serial"
        return 0
    fi

    print_blue "Starting emulator: $avd_name"

    # Build emulator command
    local emu_cmd="$EMULATOR -avd $avd_name -no-snapshot-save"
    if [ -n "$ADB_EMULATOR_OPTS" ]; then
        emu_cmd="$emu_cmd $ADB_EMULATOR_OPTS"
    fi

    # Start emulator in background
    $emu_cmd &
    EMULATOR_PID=$!
    EMULATOR_STARTED=true

    print_blue "Emulator started with PID: $EMULATOR_PID"
    print_blue "Waiting for emulator to boot (timeout: ${EMULATOR_BOOT_TIMEOUT}s)..."

    # Wait for emulator to appear in adb devices
    local elapsed=0
    while [ $elapsed -lt $EMULATOR_BOOT_TIMEOUT ]; do
        sleep 2
        elapsed=$((elapsed + 2))

        # Check if emulator process is still running
        if ! kill -0 $EMULATOR_PID 2>/dev/null; then
            print_error "Emulator process terminated unexpectedly"
            exit 1
        fi

        # Try to get the serial
        if serial=$(get_emulator_serial "$avd_name"); then
            ADB_DEVICE="$serial"
            print_blue "Emulator appeared as: $serial"
            break
        fi

        echo -n "."
    done
    echo ""

    if [ -z "$ADB_DEVICE" ]; then
        # Fallback: use first emulator found
        ADB_DEVICE=$($ADB devices | grep "emulator-" | head -1 | cut -f1)
    fi

    if [ -z "$ADB_DEVICE" ]; then
        print_error "Emulator did not start within timeout"
        exit 1
    fi

    # Wait for boot to complete
    print_blue "Waiting for boot to complete..."
    local boot_elapsed=0
    while [ $boot_elapsed -lt $EMULATOR_BOOT_TIMEOUT ]; do
        local boot_completed=$($ADB -s "$ADB_DEVICE" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
        if [ "$boot_completed" = "1" ]; then
            print_blue "Emulator boot completed!"
            return 0
        fi
        sleep 2
        boot_elapsed=$((boot_elapsed + 2))
        echo -n "."
    done
    echo ""

    print_warn "Boot completion check timed out, but continuing anyway..."
}

# Stop emulator if we started it
stop_emulator() {
    if [ "$EMULATOR_STARTED" = "true" ] && [ -n "$EMULATOR_PID" ]; then
        print_blue "Stopping emulator (PID: $EMULATOR_PID)..."
        if [ -n "$ADB_DEVICE" ]; then
            $ADB -s "$ADB_DEVICE" emu kill 2>/dev/null || true
        fi
        # Give it a moment to shut down gracefully
        sleep 2
        if kill -0 $EMULATOR_PID 2>/dev/null; then
            kill $EMULATOR_PID 2>/dev/null || true
        fi
    fi
}

# Check prerequisites
check_prerequisites() {
    # Check ADB is available
    if ! command -v "$ADB" &> /dev/null; then
        print_error "ADB not found. Please install Android SDK platform-tools."
        exit 1
    fi

    # If AVD is specified, treat it as explicit emulator selection.
    # This works even when physical devices are already connected.
    if [ -n "$ADB_AVD" ] && [ -z "$ADB_DEVICE" ]; then
        start_emulator "$ADB_AVD"
    fi

    # Handle device selection
    local device_count=$(count_devices)

    if [ "$device_count" -eq 0 ]; then
        print_error "No Android device connected."
        echo ""
        if [ -z "$ADB_AVD" ]; then
            print_info "Tip: Set ADB_AVD to automatically start an emulator"
            print_info "     List AVDs with: $0 --list-avds"
        fi
        exit 1
    elif [ -z "$ADB_DEVICE" ]; then
        if [ "$device_count" -gt 1 ]; then
            print_error "Multiple devices connected. Please select one explicitly."
            echo ""
            echo "Available devices:"
            $ADB devices -l
            echo ""
            print_info "Use one of:"
            print_info "  ADB_DEVICE=<serial> bazel run //<pkg>:<target>_adb"
            print_info "  ADB_AVD=<avd_name> bazel run //<pkg>:<target>_adb"
            exit 1
        fi
        # Auto-select only when there is exactly one connected target.
        # Preference order remains: physical device, then emulator.
        ADB_DEVICE=$(select_best_device)
    fi

    # Build ADB command with device selection
    ADB_CMD="$ADB"
    if [ -n "$ADB_DEVICE" ]; then
        ADB_CMD="$ADB -s $ADB_DEVICE"
    fi

    # Check device is accessible
    if ! $ADB_CMD get-state &> /dev/null; then
        print_error "Device not accessible or not authorized: $ADB_DEVICE"
        echo ""
        echo "Available devices:"
        $ADB devices -l
        exit 1
    fi

    # Get device info
    DEVICE_MODEL=$($ADB_CMD shell getprop ro.product.model 2>/dev/null | tr -d '\r')
    DEVICE_API=$($ADB_CMD shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')
    DEVICE_TYPE="device"
    if [[ "$ADB_DEVICE" == emulator-* ]] || [[ "$($ADB_CMD shell getprop ro.kernel.qemu 2>/dev/null | tr -d '\r')" == "1" ]]; then
        DEVICE_TYPE="emulator"
    fi
    print_info "Connected to: $DEVICE_MODEL (API $DEVICE_API) [$DEVICE_TYPE]"
}

# Create test directory on device
setup_device() {
    print_info "Setting up test directory: $ADB_TEST_DIR"
    $ADB_CMD shell "mkdir -p $ADB_TEST_DIR" 2>/dev/null || true
}

# Push test binary to device
push_binary() {
    local binary_path="$1"
    local binary_name=$(basename "$binary_path")

    if [ ! -f "$binary_path" ]; then
        print_error "Test binary not found: $binary_path"
        exit 1
    fi

    print_info "Pushing test binary: $binary_name"
    $ADB_CMD push "$binary_path" "$ADB_TEST_DIR/$binary_name" > /dev/null

    # Make executable
    $ADB_CMD shell "chmod +x $ADB_TEST_DIR/$binary_name"

    echo "$ADB_TEST_DIR/$binary_name"
}

# Push shared libraries if needed
push_libraries() {
    local binary_path="$1"
    local binary_dir=$(dirname "$binary_path")

    # Check for runfiles _solib directory (Bazel-built binaries)
    local solib_dir=""
    for arch in arm64-v8a armeabi-v7a x86_64 x86; do
        if [ -d "$RUNFILES_DIR/_main/_solib_$arch" ]; then
            solib_dir="$RUNFILES_DIR/_main/_solib_$arch"
            break
        fi
    done

    if [ -n "$solib_dir" ] && [ -d "$solib_dir" ]; then
        print_info "Pushing shared libraries from runfiles..."
        # Create a temp directory, copy libs following symlinks, then push
        local temp_libs=$(mktemp -d)
        cp -L "$solib_dir"/*.so "$temp_libs/" 2>/dev/null || true

        # Also copy NDK's libc++_shared.so if available
        local libcxx=""
        if [ -n "$ANDROID_NDK_HOME" ]; then
            libcxx=$(find "$ANDROID_NDK_HOME" -path "*/aarch64-linux-android/libc++_shared.so" 2>/dev/null | head -1)
        elif [ -n "$ANDROID_HOME" ]; then
            libcxx=$(find "$ANDROID_HOME/ndk" -path "*/aarch64-linux-android/libc++_shared.so" 2>/dev/null | head -1)
        fi
        if [ -n "$libcxx" ] && [ -f "$libcxx" ]; then
            cp "$libcxx" "$temp_libs/"
        fi

        local lib_count=$(ls -1 "$temp_libs"/*.so 2>/dev/null | wc -l | tr -d ' ')
        if [ "$lib_count" -gt 0 ]; then
            $ADB_CMD push "$temp_libs/." "$ADB_TEST_DIR/" > /dev/null
            print_info "Pushed $lib_count shared libraries"
        fi
        rm -rf "$temp_libs"
    else
        # Fallback: push any .so files in the same directory
        for lib in "$binary_dir"/*.so "$binary_dir"/*.so.*; do
            if [ -f "$lib" ]; then
                local lib_name=$(basename "$lib")
                print_info "Pushing library: $lib_name"
                $ADB_CMD push "$lib" "$ADB_TEST_DIR/$lib_name" > /dev/null
            fi
        done
    fi
}

# Run test on device
run_test() {
    local remote_binary="$1"
    shift
    local test_args="$@"

    local binary_name=$(basename "$remote_binary")
    print_info "Running test: $binary_name $test_args"

    # Set library path and LD_PRELOAD for libc++_shared.so
    local preload=""
    if $ADB_CMD shell "[ -f $ADB_TEST_DIR/libc++_shared.so ]" 2>/dev/null; then
        preload="LD_PRELOAD=$ADB_TEST_DIR/libc++_shared.so"
    fi

    local cmd="cd $ADB_TEST_DIR && $preload LD_LIBRARY_PATH=$ADB_TEST_DIR:\$LD_LIBRARY_PATH ./$binary_name $test_args"

    # Run with timeout
    local exit_code=0
    $ADB_CMD shell "$cmd" || exit_code=$?

    return $exit_code
}

# Cleanup
cleanup() {
    if [ "$ADB_KEEP_BINARY" != "true" ]; then
        local binary_name=$(basename "$TEST_BINARY")
        print_info "Cleaning up test files..."
        $ADB_CMD shell "rm -f $ADB_TEST_DIR/$binary_name" 2>/dev/null || true
    fi

    # Stop emulator if we started it
    stop_emulator
}

# Main
main() {
    if [ $# -lt 1 ]; then
        usage
    fi

    # Handle special commands
    case "$1" in
        --list-avds|-la)
            list_avds
            ;;
        --list-devices|-ld)
            list_devices
            ;;
        --help|-h)
            usage
            ;;
    esac

    TEST_BINARY="$1"
    shift
    TEST_ARGS="$@"

    # Compute RUNFILES_DIR if not set (for library discovery)
    if [ -z "$RUNFILES_DIR" ]; then
        # Try to find runfiles directory from binary path
        local binary_dir=$(dirname "$TEST_BINARY")
        # Check if we're inside a runfiles tree
        if [[ "$binary_dir" == *".runfiles/"* ]]; then
            RUNFILES_DIR="${binary_dir%%/_main/*}"
        elif [ -d "${TEST_BINARY}.runfiles" ]; then
            RUNFILES_DIR="${TEST_BINARY}.runfiles"
        fi
    fi

    # Setup trap for cleanup
    trap cleanup EXIT

    check_prerequisites
    setup_device

    REMOTE_BINARY=$(push_binary "$TEST_BINARY")
    push_libraries "$TEST_BINARY"

    echo ""
    echo "=========================================="
    echo "Running test on Android device/emulator"
    echo "=========================================="
    echo ""

    run_test "$REMOTE_BINARY" $TEST_ARGS
    exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        print_info "Test PASSED"
    else
        print_error "Test FAILED (exit code: $exit_code)"
    fi

    exit $exit_code
}

main "$@"
