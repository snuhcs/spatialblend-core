# Contributing to SpatialBlend Core

Build and test commands, the conventions every change follows, and the layout of the core.

## Build & Test Commands

```bash
# Run all tests
bazel test //spatialblend/...

# Run specific test with verbose output
bazel test //spatialblend/geometry/path:astar_path_finder_test --test_output=all

# Filter specific test case
bazel test //spatialblend/model/actor:actor_test --test_filter=ActorTest.RemoveAnchor

# Build specific target
bazel build //spatialblend/space

# Build C API
bazel build //spatialblend/api/c:c_shared

# Build Python wheel
bazel build //spatialblend/api/python:spatialblend_wheel

# Format code
clang-format -i spatialblend/module/file.cc

# Query dependencies
bazel query "deps(//spatialblend/space)"
bazel query "rdeps(//spatialblend/..., //spatialblend/model/actor:actor)"
```

**Build configurations:**
- Release (default): `bazel build -c opt`
- Debug: `bazel build --config=debug`
- Android: `bazel build --config=android`
- VisionOS (macOS only): `bazel build --config=visionos`

**Testing discipline:** Run individual tests during development. Only run `bazel test //spatialblend/...` before committing. Do not relax test expectations - only change them when there is a logical flaw.

## Critical Convention: Z-UP Coordinate System

**This is the most important convention in the codebase.**

```
2D Coordinates: X (right), Y (forward) → points: (x, y)
3D Coordinates: X (right), Y (forward), Z (UP) → points: (x, y, z)
Rotations: Yaw (Z-axis), Pitch (Y-axis), Roll (X-axis) → order: Z × Y × X
```

**State Types** (`spatialblend/base/common/state_type.h`):
- `kPosition2D`: [x, y] - horizontal position
- `kPosition3D`: [x, y, z] - z is height/elevation
- `kRotationYaw`: [yaw] - around the Z axis
- `kRotation3D`: [yaw, pitch, roll]

State vector dimensions must match the `StateType` (`GetStateTypeDimension`).

## Code Style

- **C++17** with Google C++ style guide (enforced via `.clang-format`)
- **Files**: `snake_case.{cc,h}`, tests: `snake_case_test.cc`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `kPascalCase`
- **Private members**: trailing underscore (`member_name_`)
- **Header guards**: `SPATIALBLEND_<PATH>_<FILE>_H_`
- **Namespaces**: `spatialblend` for main code, `spatialblend::test` for tests (the protobuf packages keep their `spatialblend_v2` wire names, aliased as `spatialblend::proto` and `spatialblend::room`)

## Key Architecture

The layers depend only on the layers above them (`python3 tools/check_layering.py`): `base` →
`geometry` → `model` → `solver` → `space` → `net` → `api`, with `sim` (simulator, harness, scenario suite)
on top. v1 (Entity, AffordanceVolume, GeometryCache, the state planners) was removed.

### Space and spaces
- `Space` (`space/space.h`) holds one participant's room: its element tree, its actors, the
  relations between them, and the solver that realizes a remote source's relations here.
- `LocalContext` / `SpaceHub` (`space/`) connect spaces in-process; `RemoteHub` / `HubReceiver`
  (`net/`) connect them over gRPC (or the tunnel, `net/tunnel_link.h`).

### Actors and elements
- `Actor` (`model/actor/actor.h`): an id, the root transform and named `SubjectAnchor`s (head,
  hands, bones), each optionally with a world-pose override.
- Elements (`model/element/`): the room and its objects as an `ElementTree`; each element has
  spatial definitions (navigable polygon, polytope, surface), sub-regions, an affordance pose and
  a scope extension.

### Relations and realization
- Relation kernels (`model/relation/`): Facing, Spacing and AffordancePose, with grounding rules
  and scope resolution; `solver/relation_solver.h` solves them for the receiver.
- The realization pipeline (shared interaction hierarchy → relation extraction →
  heterogeneity-aware realization) is described in the paper (see the README's Citation).

## Important Files

- `spatialblend/space/space.h` - Space management and callbacks
- `spatialblend/solver/relation_solver.h` - the receiver's relation solve
- `spatialblend/model/actor/actor.h` - actors and their anchors
- `spatialblend/model/element/element_tree.h` - the room's element tree
- `spatialblend/model/relation/relation_kernel.h` - relation kernels
- `spatialblend/api/c/spatialblend.h` - C API with coordinate system docs

## Build Scripts

```bash
# C API build
python tools/build_c_api.py          # Local release
python tools/build_c_api.py -d       # Debug
python tools/build_c_api.py -a       # Android
python tools/build_c_api.py -v       # VisionOS (macOS only)
python tools/build_c_api.py -all     # All platforms

# Python wheel
python tools/build_python_wheel.py --install

# Release
python tools/publish_release.py --version v1.0.0
```

## Error Handling

- Use `absl::Status`/`absl::StatusOr` for expected failures
- Reserve exceptions for programming errors; report expected failures as status values
- Prefer smart pointers (`std::shared_ptr`, `std::unique_ptr`) over raw pointers

## Test Tags

- `exclusive`: Tests requiring exclusive execution (network, concurrency tests)
- `manual`: Tests not run in standard suite (long-running, interactive)
- `android`: Tests designed for Android platform
- `adb`: Tests that run on Android device via ADB

## Android Device Testing (ADB)

All detailed Android/ADB instructions live in `docs/android.md`.

No ADB test target is defined at present (the v1 `network_android_test_adb` was removed with
the v1 network code); `bazel/adb_test.bzl` provides `adb_test` for adding one, and
`docs/android.md` describes running it.

## Commits and pull requests

- Prefix the subject with the module: `[space] fix: ...`, `[geometry] feat: ...`; keep each commit to one change.
- `bazel test //spatialblend/...` passes before a pull request (there is no test CI: run it locally).
- Never commit credentials, host names or local paths.
