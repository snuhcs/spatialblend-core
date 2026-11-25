# SpatialBlend Core

The native runtime of [SpatialBlend](https://github.com/snuhcs/spatialblend) (UIST '26): `libspatialblend`, a C++ library with
a C ABI and Python bindings. It reproduces a remote person's behaviour in another room as relations (whom they face, how
far they stand from something, how they use a piece of furniture) rather than raw coordinates, and realizes those
relations under the receiving room's own layout. The Unity app
[snuhcs/spatialblend-unity](https://github.com/snuhcs/spatialblend-unity) uses it on Meta Quest and desktop.

[Project page](https://github.com/snuhcs/spatialblend) · [Paper](https://doi.org/10.1145/3830398.3830690) ·
[Video](https://www.youtube.com/watch?v=3TWBkZOWY_4) · [Unity app](https://github.com/snuhcs/spatialblend-unity) ·
[Object scanning](https://github.com/snuhcs/spatialblend-reconstruction)

## How it works

```
 source room                                                receiver room
 room geometry ─┐                                       ┌─ room geometry
 tracked body  ─┴─> relations ──── network ────> solve ─┴─> pose proposals
                    (Facing, Spacing,                         (the app drives
                     AffordancePose)                           the avatar)
```

Each participant's room is a **space**: an element tree (floor, walls, furniture, with their regions and affordances)
and the actors in it. On the source side the core reads a person's pose as relations; on the receiving side the solver
finds the pose that best keeps those relations in the receiver's room and proposes it to the application. Spaces
connect in-process or over gRPC.

## Get the library

**Prebuilt:** the [v1.0.0 release](https://github.com/snuhcs/spatialblend-core/releases/tag/v1.0.0) has `libspatialblend` for
Android arm64, macOS arm64, Windows x64 and Linux x64, the C headers and `SHA256SUMS`.

**From source:** requires [Bazel](https://github.com/bazelbuild/bazelisk) 8.4.2 (pinned in `.bazelversion`) and a JDK;
for Android, NDK r25c in `ANDROID_NDK_HOME`. The development container (`.devcontainer/`) has the full toolchain.

```bash
bazel test //spatialblend/...                             # all tests
python tools/build_c_api.py                            # libspatialblend for this host -> output/<platform>/
python tools/build_c_api.py -a                         # Android arm64
bazel build //spatialblend/api/python:spatialblend_wheel     # Python wheel (pybind11)
bazel build //spatialblend/api/python:spatialblend_capi_wheel  # Python wheel (ctypes)
```

Linux hosts build Linux x64 and Android; macOS hosts build macOS arm64, Android and visionOS; Windows hosts build
Windows x64.

## Use it

- **C API:** start from [`spatialblend/api/c/spatialblend.h`](spatialblend/api/c/spatialblend.h); its comments document each
  `sb_*` function.
- **Python:** the `spatialblend` wheel (pybind11, CPython 3.10; `import spatialblend.api.python`) or the `spatialblend_capi`
  wheel (ctypes over `libspatialblend`, any Python 3; `import spatialblend.api.python.capi`).
  [`spatialblend/api/python/examples/preset_visualization.py`](spatialblend/api/python/examples/preset_visualization.py)
  runs two rooms with a simulated person and renders the result.
- **Traces:** `sb_set_trace` writes a JSON Lines trace of each solve
  ([format](spatialblend/api/c/trace.format.md)); [`tools/scenario-viewer/`](tools/scenario-viewer/) views scenario traces
  in a browser.

## Documentation

- [docs/build-scripts.md](docs/build-scripts.md): every build option and the output layout
- [docs/android.md](docs/android.md): running the tests on an Android device
- [docs/MIGRATION-1.0.md](docs/MIGRATION-1.0.md): upgrading code written against a pre-1.0 core
- [CONTRIBUTING.md](CONTRIBUTING.md): conventions, architecture and the package layering

## License

MIT, Copyright (c) 2024-2026 Seoul National University ([LICENSE](LICENSE)). Third-party material is listed in
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Please cite the SpatialBlend paper ([doi:10.1145/3830398.3830690](https://doi.org/10.1145/3830398.3830690)); the BibTeX
is on the [project page](https://github.com/snuhcs/spatialblend#citation).

## Contact

Jingyu Lee (<jingyu.lee@hcs.snu.ac.kr>), Human-Centered Computer Systems Lab, Seoul National University.

## Contributors

- Jingyu Lee (Seoul National University)
- Dongho Han (Seoul National University)
- Jinmyeong Kim (Seoul National University)
