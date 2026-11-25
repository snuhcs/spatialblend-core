# Build and release scripts

Moved from the top-level README.

## 1. C API Build Script (`tools/build_c_api.py`)
Builds the C API library for various platforms and configurations.

```bash
python tools/build_c_api.py [options]
```

### Options:
- `-all`: Build for all supported platforms and configurations
- `-a, --android`: Build for Android platform
- `-v, --visionos`: Build for VisionOS platform (only available on macOS)
- `-d, --debug`: Build in debug mode (default is release mode)
- `-r, --rebuild`: Re-build test target (only affects Android builds)
- `-b, --build`: Build only, skip tests (only affects local builds, default: true)
- `-o, --output`: Specify output directory (default: "output")
- `-docker`: Use Docker for cross-compilation (requires devcontainer setup)
- `-dh, --docker_hash`: Specify Docker container hash for builds

### Examples:
```bash
# Build for current platform in release mode
python tools/build_c_api.py

# Build for all platforms in both debug and release modes
python tools/build_c_api.py -all

# Build for Android with debug symbols
python tools/build_c_api.py -a -d

# Build using Docker for cross-compilation
python tools/build_c_api.py -docker
```

## 2. Python Wheel Build Script (`tools/build_python_wheel.py`)
Builds Python wheel packages for distribution.

```bash
python tools/build_python_wheel.py [options]
```

### Options:
- `--output`: Specify output directory (default: "output")
- `-i, --install`: Install the wheel after building
- `--pybind`: Build the `spatialblend` wheel (pybind11, CPython 3.10; the default when no wheel is named)
- `--capi`: Build the `spatialblend_capi` wheel (pure Python + ctypes, any CPython 3.6+); pass both flags to build both

### Examples:
```bash
# Build Python wheel
python tools/build_python_wheel.py

# Build and install Python wheel
python tools/build_python_wheel.py --install

# Build to custom output directory
python tools/build_python_wheel.py --output dist/
```

## 3. Release Publishing Script (`tools/publish_release.py`)
Creates and publishes releases with git tags.

```bash
python tools/publish_release.py [options]
```

### Options:
- `--version`: Version string (must start with 'v', default: "v0.0.1")
- `--caption`: Optional caption for the tag

### Examples:
```bash
# Create a basic release
python tools/publish_release.py --version v1.0.0

# Create a release with description
python tools/publish_release.py --version v1.0.0 --caption "Major release with new features"
```

## C API development
### Local Build
```bash
python tools/build_c_api.py
```

### Cross-platform Build
```bash
# For all platforms
python tools/build_c_api.py -all

# For specific platforms
python tools/build_c_api.py -a -v  # Android and VisionOS
```

### Output Structure
Build artifacts are placed in the output directory (default: `output/`) with this structure:
```
output/
├── linux/       # one folder per platform: lib/ and include/
├── macos/
├── android/
└── visionos/    # macOS hosts only
```
A debug build (`-d`) goes to the same platform folder as a release build and replaces it.

## Python API development

### Build Python Wheel Locally
```bash
python tools/build_python_wheel.py
```

### Alternative: Direct Bazel Build
```bash
bazel build //spatialblend/api/python:spatialblend_wheel
```

### Install Wheel
```bash
# Install from build script output
python tools/build_python_wheel.py --install

# Or install manually from bazel output
pip install bazel-bin/spatialblend/api/python/spatialblend-<version>-py3-none-any.whl
```

