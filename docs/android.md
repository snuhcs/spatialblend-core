# Android Testing Guide

> **Note:** the example targets `//spatialblend/net/util:network_android_test(_adb)` and
> `//spatialblend/benchmark:geometry_benchmark` no longer exist (they went with the v1 code); the `adb_test`
> rule and the workflow below are unchanged.

This file is the canonical Android/ADB documentation for this repository.
Keep Android target naming, device/emulator selection behavior, environment variables, and troubleshooting details here.

This guide covers how to run native C++ tests on Android devices and emulators using Bazel and ADB (Android Debug Bridge).

## Prerequisites

### Required Tools

1. **Android SDK Platform Tools** (includes ADB)
   - Install via Android Studio SDK Manager, or
   - Download from [Android Developer](https://developer.android.com/studio/releases/platform-tools)
   - Ensure `adb` is in your PATH or set `ANDROID_HOME`/`ANDROID_SDK_HOME`

2. **Android Emulator** (for emulator testing)
   - Install via Android Studio SDK Manager
   - Ensure `emulator` is in your PATH or set `ANDROID_HOME`/`ANDROID_SDK_HOME`

3. **Android NDK** (for building Android binaries)
   - NDK r25c (25.2.9519653), the NDK the released library is built with (rules_android_ndk 0.1.3 needs r25b or newer;
     r27d builds showed device regressions)
   - API level 28, NDK API level 21
   - Configured in Bazel via `--config=android`

### Device Setup

For physical devices:
- Enable Developer Options on your Android device
- Enable USB Debugging
- Connect device via USB
- Authorize the computer when prompted

For emulators:
- Create an AVD (Android Virtual Device) using Android Studio or `avdmanager`
- Ensure the emulator is listed: `emulator -list-avds`

## Quick Start

### ADB Target Discovery

```bash
bazel query 'attr(name, ".*_adb$", //...)'
```

`*_adb` targets are auto-generated for `spatialblend_cc_test(...)` and `spatialblend_cc_binary(...)`.

### Target Selection Behavior (`*_adb`)

- If exactly one Android target is connected, it is selected automatically.
- If multiple targets are connected, the runner requires explicit selection.
- Set `ADB_DEVICE=<serial>` to pick a specific connected device/emulator.
- Set `ADB_AVD=<name>` to explicitly select/start that emulator.

### Running Tests on a Connected Device

```bash
# Build test for Android
bazel build //spatialblend/net/util:network_android_test_adb

# Run test on connected device
bazel run //spatialblend/net/util:network_android_test_adb

# Run with test filter
bazel run //spatialblend/net/util:network_android_test_adb -- --gtest_filter=*GetIPv4*
```

### Running Tests on an Emulator

```bash
# List available AVDs
./tools/adb_test_runner.sh --list-avds

# Run test on emulator (auto-starts if not running)
ADB_AVD=Pixel_4_API_30 bazel run //spatialblend/net/util:network_android_test_adb

# Run headless emulator (for CI/automation)
ADB_AVD=Pixel_4_API_30 ADB_EMULATOR_OPTS='-no-window -no-audio' bazel run //spatialblend/net/util:network_android_test_adb
```

### Running Any Bazel Binary/Test Without `_adb` Target

```bash
# Wrapper target that builds with --config=android and runs via ADB
bazel run //tools:run_on_android -- --target //spatialblend/net/util:network_android_test
```

## Adding ADB Tests to Your BUILD File

### Option 1: Wrap an Existing Test

```python
load("//spatialblend:adb_test.bzl", "adb_test")

# Your existing test
cc_test(
    name = "my_test",
    srcs = ["my_test.cc"],
    deps = ["@googletest//:gtest_main"],
)

# Create ADB test wrapper
adb_test(
    name = "my_test_adb",
    test = ":my_test",
    tags = ["android", "adb", "manual"],
)
```

### Option 2: Use Combined Macro

The `spatialblend_adb_test` macro creates both a standard `cc_test` and an `adb_test`:

```python
load("//spatialblend:adb_test.bzl", "spatialblend_adb_test")

spatialblend_adb_test(
    name = "my_test",
    srcs = ["my_test.cc"],
    deps = ["@googletest//:gtest_main"],
    tags = ["android", "adb"],
)
```

This creates:
- `:my_test` - Standard test that runs on host
- `:my_test_adb` - ADB test that runs on Android device

### Option 3: Test Suite

Create a suite of ADB tests:

```python
load("//spatialblend:adb_test.bzl", "adb_test_suite")

adb_test_suite(
    name = "all_android_tests",
    tests = [
        ":test1",
        ":test2",
        ":test3",
    ],
    tags = ["android", "adb"],
)
```

## Environment Variables

### Device Selection

- `ADB_DEVICE`: Target device serial (required if multiple devices connected)
  ```bash
  ADB_DEVICE=emulator-5554 bazel run //path:test_adb
  ```

### Test Configuration

- `ADB_TEST_DIR`: Directory on device for test binaries (default: `/data/local/tmp/spatialblend_tests`)
- `ADB_TIMEOUT`: Test timeout in seconds (default: 300)
- `ADB_KEEP_BINARY`: Keep binary on device after test (default: false)
  ```bash
  ADB_KEEP_BINARY=true bazel run //path:test_adb
  ```

### Emulator Configuration

- `ADB_AVD`: AVD name to explicitly use/start
  ```bash
  ADB_AVD=Pixel_4_API_30 bazel run //path:test_adb
  ```

- `ADB_EMULATOR_OPTS`: Additional emulator options
  ```bash
  ADB_EMULATOR_OPTS='-no-window -no-audio' bazel run //path:test_adb
  ```

- `EMULATOR_BOOT_TIMEOUT`: Emulator boot timeout in seconds (default: 120)
  ```bash
  EMULATOR_BOOT_TIMEOUT=180 bazel run //path:test_adb
  ```

## Using Standalone Scripts

### Python Test Runner

The `run_android_test.py` script provides a standalone way to run tests:

```bash
# Run a pre-built test binary
python tools/run_android_test.py --binary bazel-bin/spatialblend/geometry/path/test

# Build and run a test
python tools/run_android_test.py --target //spatialblend/net/util:network_android_test

# Run on emulator (starts if not running)
python tools/run_android_test.py --avd Pixel_4_API_30 --binary path/to/test

# Run headless emulator in CI
python tools/run_android_test.py --avd Pixel_4_API_30 --headless --binary path/to/test

# Run with specific device
python tools/run_android_test.py --device SERIAL --binary path/to/test

# Run with test filter
python tools/run_android_test.py --binary path/to/test --filter '*TestName*'
```

### Shell Test Runner

The `adb_test_runner.sh` script can be used directly:

```bash
# List available AVDs
./tools/adb_test_runner.sh --list-avds

# List connected devices
./tools/adb_test_runner.sh --list-devices

# Run test on connected device
./tools/adb_test_runner.sh bazel-bin/spatialblend/geometry/path/test

# Run on emulator
ADB_AVD=Pixel_4_API_30 ./tools/adb_test_runner.sh bazel-bin/spatialblend/geometry/path/test

# Run headless emulator
ADB_AVD=Pixel_4_API_30 ADB_EMULATOR_OPTS='-no-window -no-audio' ./tools/adb_test_runner.sh bazel-bin/spatialblend/geometry/path/test
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Android Tests
  env:
    ADB_AVD: Pixel_4_API_30
    ADB_EMULATOR_OPTS: '-no-window -no-audio'
    EMULATOR_BOOT_TIMEOUT: 180
  run: |
    bazel test --config=android //spatialblend/... --test_tag_filters=adb
```

### Headless Emulator Setup

For CI environments, run emulators in headless mode:

```bash
# Start emulator in background (headless)
ADB_AVD=Pixel_4_API_30 ADB_EMULATOR_OPTS='-no-window -no-audio' \
  bazel run //path:test_adb
```

## Troubleshooting

### No Device Found

```bash
# Check connected devices
adb devices

# List available AVDs
./tools/adb_test_runner.sh --list-avds

# Start emulator manually
emulator -avd Pixel_4_API_30 &
```

### Device Not Authorized

1. Check device screen for authorization prompt
2. Enable "Always allow from this computer"
3. Verify: `adb devices` shows "device" (not "unauthorized")

### Emulator Not Starting

1. Verify AVD exists: `emulator -list-avds`
2. Check emulator logs: `~/.android/avd/<AVD_NAME>.avd/emulator-*.log`
3. Increase boot timeout: `EMULATOR_BOOT_TIMEOUT=300`

### Test Binary Not Found

1. Ensure test is built: `bazel build --config=android //path:test`
2. Check binary exists: `ls bazel-bin/path/test`
3. Verify Android wrapper target: `bazel build //path:test_adb`

### Permission Denied on Device

The test runner automatically sets executable permissions. If issues persist:

```bash
adb shell chmod +x /data/local/tmp/spatialblend_tests/test_binary
```

## Test Tags

ADB tests should use these tags:

- `android`: Indicates Android-specific test
- `adb`: Indicates test runs via ADB
- `manual`: Prevents test from running in standard test suite (optional)

Example:
```python
adb_test(
    name = "my_test_adb",
    test = ":my_test",
    tags = ["android", "adb", "manual"],
)
```

## Best Practices

1. **Use `spatialblend_adb_test` macro** for new tests to get both host and Android versions
2. **Tag tests appropriately** with `android` and `adb` tags
3. **Use emulators for CI** with headless mode (`-no-window -no-audio`)
4. **Set timeouts appropriately** for long-running tests
5. **Clean up binaries** after tests (unless debugging, set `ADB_KEEP_BINARY=false`)
6. **Test on multiple API levels** to ensure compatibility

## Examples

See `spatialblend/net/util/BUILD` for a complete example of ADB test integration.
