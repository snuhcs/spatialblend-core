# Naming and migration to 1.0
The "v2" label is gone from paths, targets and artifacts: `spatialblend_v2/` is `spatialblend/`, `libspatialblend_v2` is `libspatialblend`, the wheels
are `spatialblend` and `spatialblend_capi`. The v1 tree no longer exists here. The v1 comparison oracle is not part of
this repository.

These names still say v2, **on purpose**, and stay through 1.0:
- the protobuf packages `spatialblend_v2.proto` and `spatialblend_v2.room.v1` (and the generated C++ namespaces `spatialblend_v2::proto`
  and `spatialblend_v2::room`, which C++ code reaches through the aliases `spatialblend::proto` and `spatialblend::room`): they are on the wire
  (gRPC method paths), and APKs built against the old names must interoperate;
- the `sb_*_v2` layout/type suffixes of the C ABI and the exported `sb_*` symbols themselves;
- the `SB_V2_*` and `SPATIALBLEND_V2_*` environment variables and build macros.

Shared base: the files v2 production still compiled from the v1 tree live in the matching `spatialblend/` packages (util, common, event, entity,
geometry, path, policy, state_planner, interface, math, network/grpc, network/util, data). Five files collided with a v2 file of the same path but different
code; they keep their code and got a `base_` name (`common/base_trace.*`, `common/base_state_type.h`, `geometry/base_visibility_graph.*`,
`path/base_discrete_path.*`, `path/base_path_finder.h`). All moved base headers use `SPATIALBLEND_BASE_*` include guards (v2 headers use `SPATIALBLEND_*`).

C++ namespace: `spatialblend_v2` is now `spatialblend`, merged with the shared base namespace. Base types whose names the v2 code also
defines were renamed `Base<Name>` (`BasePolygon2D`, `BaseBound2D`, `BaseConvexCellPartition`, `internal::BaseEdgeData`,
`BaseNormalizedPolygonPosition`, `BasePolygonCoordinateMapper`, `BaseVisibilityGraph`, `BasePathFinder`, `BaseAStarPathFinder`,
`BaseDiscretePath`, `BaseStateType`); `common/common.h` takes `space_id_t` and `kInvalidSpaceId` from `common/types.h`. The C ABI is unchanged.

Layered layout: the packages are grouped by layer, and a layer depends only on the layers before it
(`python3 tools/check_layering.py`): `spatialblend/base/` (common, math, util, interface, policy), `spatialblend/geometry/` (primitives,
`path/`, `context/`, `element/`), `spatialblend/model/` (actor, entity, event, relation, link, state_planner), `spatialblend/solver/` (ex
`space/`, with `builder/` for the C++ handles and space builder that were in `simulator/`), `spatialblend/net/` (ex `network/`, with
`session/` and `media/`), `spatialblend/api/c/` (ex `c/` and `specs/`), `spatialblend/api/python/` (ex `python/`) and `spatialblend/sim/` (ex
`simulator/`, `harness/`, `data/`). `script/` is `tools/`, `config/` is `bazel/`, `android/README.md` is `docs/android.md`.
- Include paths and Bazel labels follow the directories (`spatialblend/space/space.h` is `spatialblend/solver/space.h`,
  `//spatialblend/c:c_shared` is `//spatialblend/api/c:c_shared`, `//spatialblend/space` is `//spatialblend/solver:space`).
- The Python import path follows too: `import spatialblend.python` is `import spatialblend.api.python`, and
  `spatialblend.python.capi` is `spatialblend.api.python.capi`. The wheel names are unchanged.
- The C ABI (`libspatialblend`, the `sb_*` exports, the headers installed as `include/spatialblend/*.h`) and the wire format are unchanged.
  The trace's `ev=meta` `spec` field names `spatialblend/api/c/trace.format.md`.

Space/solver split: the Space runtime has its own layer again, as in v1, and `solver/` keeps the solve. A layer
still depends only on the layers before it: `base` < `geometry` (primitives < `path/` < `context/`) < `model` (`element/`,
`actor/`, `link/`, `embodiment/` < `relation/` < `realization/`) < `solver` < `space` < `net` < `api/c` < `api/python` < `sim`.
- `spatialblend/geometry/element/` is `spatialblend/model/element/`.
- `spatialblend/space/` holds what was the rest of `solver/`: `Space` (`space.h` and the `.cc` files that implement its members),
  `SpaceHub`, `LocalContext`, bindings, phases, ports, receiver approach and source evidence; `builder/` and `testing/` (with
  `ideal_driver.h`) moved with it. `//spatialblend/solver:space` is `//spatialblend/space`.
- `solver/` keeps the relation solve and evaluation, the trust region, the scope pursuit, the realization policy and weights,
  the referent resolver and `affordance_pose_sd`.
- `model/embodiment/` (embodiment definitions), `model/realization/` (receiver input, remote updates, containment transfer)
  and `base/common/seqlock_box.h` hold what both the solver and the space use.
- The scope case catalog is `spatialblend/space/testing/SCOPE_CASES.md`.
- The C ABI is unchanged.
