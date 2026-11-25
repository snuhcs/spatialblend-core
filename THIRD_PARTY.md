# Third-party software

What SpatialBlend Core uses and under which terms. Dependencies are fetched by Bazel at build time (`MODULE.bazel`); none of
them is vendored except the files listed in `NOTICE`. This is a summary for contributors, not legal advice.

## Linked into `libspatialblend` (C API) and the Python extension

| Dependency | Version (MODULE.bazel) | Licence |
|---|---|---|
| Abseil (`abseil-cpp`) | 20250127.0 | Apache-2.0 |
| gRPC (`grpc`) | 1.72.0-pre1 | Apache-2.0 |
| Protocol Buffers (`protobuf`) | 30.0 | BSD-3-Clause |
| c-ares | 1.19.1.bcr.1 | MIT |
| Eigen | 3.4.0.bcr.3 | MPL-2.0 (header-only, unmodified; source: https://gitlab.com/libeigen/eigen) |
| Boost (geometry, qvm, thread and their dependencies) | 1.83.0.bcr.2 | BSL-1.0 |
| CDT (Constrained Delaunay Triangulation) | 1.4.4 | MPL-2.0 (header-only, unmodified; source: https://github.com/artem-ogre/CDT) |
| nlohmann/json | 3.11.3.bcr.1 | MIT |
| zlib, RE2, BoringSSL and the other gRPC transitive dependencies | as resolved in `MODULE.bazel.lock` | zlib, BSD-3-Clause, OpenSSL/ISC (BoringSSL) |
| pybind11 (Python bindings only) | 2.12.0 via pybind11_bazel | BSD-3-Clause |

## Build and test only (not in a binary)

| Dependency | Licence |
|---|---|
| GoogleTest 1.15.2 | BSD-3-Clause |
| three.js 0.160.0 (scenario viewer, loaded from a CDN) | MIT |
| IBM Plex Sans and IBM Plex Mono (scenario viewer fonts, loaded at runtime from Google Fonts by `index.html`; not distributed) | OFL-1.1 |
| rules_python, rules_pkg, rules_android_ndk, platforms, pybind11_bazel | Apache-2.0 |
| hedron compile-commands extractor | see its repository (editor tooling only) |
| Python packages in `third_party/python/requirements.txt` (numpy, matplotlib, ...) | BSD / PSF / MIT-style |

## Data

See `NOTICE` items 3 and 4 (LocoVR-derived presets; offsets measured on the SMPL / SMPL-X body models).
