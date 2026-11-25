// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/sim/harness/unity_twin.cc
//
// See unity_twin.h for the overall design. This file drives a ScenarioModel
// end-to-end through the stable C API (spatialblend/api/c/spatialblend.h) only.

#include "spatialblend/sim/harness/unity_twin.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/space/testing/test_driver.h"

namespace spatialblend {
namespace harness {

namespace {

// ---------------------------------------------------------------------------
// Small math / geometry helpers
// ---------------------------------------------------------------------------

double WrapAngle(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;
  return a;
}

// Row-major 4x4 transform (Z-UP) encoding a translate + yaw-only rotation, as
// required by sb_set_element_transform's documented layout.
void YawZTransform(double x, double y, double z, double yaw, double* m) {
  double c = std::cos(yaw), s = std::sin(yaw);
  m[0] = c;  m[1] = -s; m[2] = 0;  m[3] = x;
  m[4] = s;  m[5] = c;  m[6] = 0;  m[7] = y;
  m[8] = 0;  m[9] = 0;  m[10] = 1; m[11] = z;
  m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
}

bool PointInPolygon(double px, double py, const std::vector<Vec2>& poly) {
  if (poly.size() < 3) return false;
  bool inside = false;
  size_t n = poly.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    double xi = poly[i].x, yi = poly[i].y;
    double xj = poly[j].x, yj = poly[j].y;
    bool intersect = ((yi > py) != (yj > py)) &&
                     (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  if (inside) return true;
  // The closed walkable set includes its boundary. A projection can land a
  // rounding error outside; ObstacleFreeSpace::InWalkable counts that as
  // walkable (1e-9). The same tolerance here, so a corner pose is contained.
  constexpr double kBoundaryTol = 1e-9;
  const double tol2 = kBoundaryTol * kBoundaryTol;
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const double vx = poly[i].x - poly[j].x;
    const double vy = poly[i].y - poly[j].y;
    const double len2 = vx * vx + vy * vy;
    double t = 0.0;
    if (len2 > 0.0) {
      t = ((px - poly[j].x) * vx + (py - poly[j].y) * vy) / len2;
      if (t < 0.0) t = 0.0;
      else if (t > 1.0) t = 1.0;
    }
    const double dx = px - (poly[j].x + t * vx);
    const double dy = py - (poly[j].y + t * vy);
    if (dx * dx + dy * dy <= tol2) return true;
  }
  return false;
}

std::vector<Vec2> BoxWorldCorners(double ex, double ey, double eyaw,
                                  double sx, double sy) {
  double hx = sx * 0.5, hy = sy * 0.5;
  std::vector<Vec2> local = {{-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}};
  double c = std::cos(eyaw), s = std::sin(eyaw);
  std::vector<Vec2> world;
  world.reserve(4);
  for (auto& p : local) {
    world.push_back({ex + c * p.x - s * p.y, ey + s * p.x + c * p.y});
  }
  return world;
}

sb_kernel_type_t KernelFromName(const std::string& k) {
  if (k == "facing") return SB_KERNEL_FACING;
  if (k == "affordance_pose") return SB_KERNEL_AFFORDANCE_POSE;
  return SB_KERNEL_SPACING;
}

// Pick a free loopback TCP port for Transport::kGrpcTwoSessions'
// sb_host_space call. The C API's sb_host_space(space, 0) can ask the OS to
// auto-assign a port, but exposes no way to read back which port it bound
// (sb_try_wait/sb_get_error are stubs -- see spatialblend.h), and the
// harness's sb_connect_space call needs that exact port. So the harness
// probes for a free port itself the same way any test harness would: bind a
// throwaway socket to loopback:0 (or to `preferred` if non-zero), read back
// the OS-assigned port via getsockname, then close the socket and hand that
// port number to sb_host_space. This has the usual bind-probe TOCTOU
// race (another process could grab the port before sb_host_space binds it)
// but that risk is inherent to any "probe then use" port picker and is the
// same tradeoff network_test.cc's "port 0" pattern makes one level down.
// Returns 0 on failure.
int32_t PickFreePort(int32_t preferred) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(preferred > 0 ? preferred : 0));
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return 0;
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len) != 0) {
    close(fd);
    return 0;
  }
  int32_t port = ntohs(bound.sin_port);
  close(fd);
  return port;
}

// Bounded polling helper: waits until `counter` exceeds `prior_value` or
// `timeout_ms` elapses. Used instead of network_test.cc's fixed 300-600ms
// sleeps so both in-process (single-session, synchronous) and real-network
// (gRPC loopback, genuinely async) scenarios settle as fast as the actual
// work allows, while still bounding worst case.
bool WaitForCountAbove(std::atomic<int>& counter, int prior_value,
                       int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (counter.load(std::memory_order_relaxed) <= prior_value) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

absl::Status WaitForExactSequenceApplied(
    sb_space_t receiver_space, sb_space_t source_space, sb_actor_t source_actor,
    int64_t expected_sequence, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  int64_t observed = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    observed = sb_get_last_applied_remote_sequence(
        receiver_space, source_space, source_actor);
    if (observed == expected_sequence) return absl::OkStatus();
    if (observed > expected_sequence) {
      return absl::FailedPreconditionError(
          "receiver applied sequence " + std::to_string(observed) +
          " ahead of expected source sequence " +
          std::to_string(expected_sequence));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  observed = sb_get_last_applied_remote_sequence(receiver_space, source_space,
                                                   source_actor);
  return absl::DeadlineExceededError(
      "timed out waiting for receiver sequence: source_space=" +
      std::to_string(source_space) + " source_actor=" +
      std::to_string(source_actor) + " expected=" +
      std::to_string(expected_sequence) + " actual=" +
      std::to_string(observed));
}

// ---------------------------------------------------------------------------
// Per-space bookkeeping. Non-movable (address is used as callback user_data),
// so callers keep these in std::vector<std::unique_ptr<SpaceCtx>>.
// ---------------------------------------------------------------------------
struct SpaceGeom {
  std::vector<Vec2> walkable;  // world frame
  std::unordered_map<std::string, Vec2> pos_by_tag;
  std::unordered_map<std::string, double> yaw_by_tag;
  std::unordered_map<std::string, std::vector<Vec2>> footprint_by_tag;
  std::unordered_map<int64_t, std::string> tag_by_shared_id;
};

struct AddedElem {
  sb_element_t local = 0;
  int64_t shared_id = 0;
  std::string tag;
};

struct SpaceCtx {
  std::string scenario_space_id;
  sb_space_t space = 0;
  sb_element_t room = 0;
  std::unordered_map<std::string, sb_element_t> elem_by_tag;
  std::unordered_map<int64_t, sb_element_t> elem_by_shared_id;
  // Reverse of elem_by_shared_id: this SPACE's own authoritative shared_id
  // for a local element handle it built itself (via BuildElements /
  // BuildSourceActor). Used to translate a mismatched-id share/resolve
  // (identity.mismatched_ids link_to_existing) back to the LOCAL shared_id
  // -- see source_shared_to_local_shared below.
  std::unordered_map<sb_element_t, int64_t> shared_id_by_elem;
  // Populated only for a PEER/target ctx during the share/resolve loop:
  // source-space shared_id -> this space's own local shared_id for the
  // element that shared_id resolved to. Identity in the common (matching-
  // ids) case; only differs under identity.mismatched_ids link_to_existing,
  // where the incoming relation payload's referent_shared_id is still the
  // SOURCE's id (space.cc's OnRemoteUpdate deliberately does not rewrite
  // it -- see CornerRepro.MismatchedIdsLinkerTwoFurniture, which likewise
  // must translate via the linker rather than trust the raw shared_id).
  std::unordered_map<int64_t, int64_t> source_shared_to_local_shared;
  SpaceGeom geom;

  std::atomic<int> element_added_count{0};
  std::atomic<int> target_events{0};

  std::mutex added_mu;
  std::vector<AddedElem> added_log;

  // Per-referent relation phase, keyed by referent_shared_id, updated by
  // OnRelationEventCb (sb_register_on_relation_event). This is the ONLY
  // reliable source of "which referent is this actor currently committed
  // to" for a PEER/target-space avatar: sb_get_relation_states() pulls from
  // Space::bindings_, which is populated only by an explicit sb_bind() call
  // -- and by design (see the RunScenario rationale comment) target avatars
  // are never explicitly bound, matching CApiE2ETest.SourceToTargetCallback's
  // precedent. EmitRelationTransitions (space.cc) fires this callback from
  // BOTH the local-actor path and Space::OnRemoteUpdate, independent of
  // bindings_, so it works uniformly for driven actors and peer avatars.
  std::mutex phase_mu;
  std::map<int64_t, sb_relation_phase_t> phase_by_referent;

  SpaceCtx() = default;
  SpaceCtx(const SpaceCtx&) = delete;
  SpaceCtx& operator=(const SpaceCtx&) = delete;
};

void OnElementAddedCb(sb_space_t /*space*/, sb_element_t local_element,
                      int64_t shared_id, const char* tag,
                      const void* /*user_data*/, int64_t /*user_data_size*/,
                      void* cb_user_data) {
  auto* ctx = static_cast<SpaceCtx*>(cb_user_data);
  {
    std::lock_guard<std::mutex> lock(ctx->added_mu);
    ctx->added_log.push_back({local_element, shared_id, tag ? tag : ""});
  }
  ctx->element_added_count.fetch_add(1, std::memory_order_relaxed);
}

void OnTargetsCb(sb_space_t /*space*/, sb_actor_t /*actor*/,
                 const sb_realization_target_t* /*targets*/,
                 int64_t /*num_targets*/, void* user_data) {
  auto* ctx = static_cast<SpaceCtx*>(user_data);
  ctx->target_events.fetch_add(1, std::memory_order_relaxed);
}

void OnRelationEventCb(const sb_relation_event_t* event, void* user_data) {
  auto* ctx = static_cast<SpaceCtx*>(user_data);
  int64_t shared_id = event->referent_shared_id;
  auto it = ctx->source_shared_to_local_shared.find(shared_id);
  if (it != ctx->source_shared_to_local_shared.end()) shared_id = it->second;
  std::lock_guard<std::mutex> lock(ctx->phase_mu);
  ctx->phase_by_referent[shared_id] = event->to_phase;
}

// Best current referent for `ctx`'s most-recently-observed phase snapshot:
// the referent at the HIGHEST lifecycle phase (contact > driving > active >
// dormant), ties broken by lowest shared_id for determinism. Returns
// {false, 0} if no referent has ever reached SB_PHASE_ACTIVE or above.
std::pair<bool, int64_t> BestCommittedReferent(SpaceCtx& ctx) {
  std::lock_guard<std::mutex> lock(ctx.phase_mu);
  bool found = false;
  int64_t best_id = 0;
  sb_relation_phase_t best_phase = SB_PHASE_DORMANT;
  for (const auto& [shared_id, phase] : ctx.phase_by_referent) {
    if (phase == SB_PHASE_DORMANT) continue;
    if (!found || phase > best_phase ||
        (phase == best_phase && shared_id < best_id)) {
      found = true;
      best_phase = phase;
      best_id = shared_id;
    }
  }
  return {found, best_id};
}

// ---------------------------------------------------------------------------
// Element construction. Per scenario.h's documented convention, an element's
// "transform.pos/yaw" is already WORLD-ABSOLUTE within its space -- "parent"
// only determines element PARENT-CHILD wiring (needed so sb_add_element gets
// the right parent handle), not pose composition. sb_set_element_transform
// sets a LOCAL transform, so we still convert: local = inverse(parent_world)
// * child_world, using each element's own already-known world pose.
// ---------------------------------------------------------------------------
// A scene mutator the core rejected (sb_scene_status_t) is a fixture error: the
// containment layout must be nested and non-overlapping (explicit status naming
// both elements), never silently dropped.
absl::Status SceneOk(SpaceCtx& ctx, sb_scene_status_t status, const char* what) {
  if (status == SB_SCENE_OK) return absl::OkStatus();
  char message[512] = {};
  sb_get_scene_status(ctx.space, message, sizeof message);
  return absl::InvalidArgumentError(std::string(what) + " rejected: " + message);
}

absl::Status BuildElements(const ScenarioSpace& sp, SpaceCtx& ctx) {
  std::vector<double> room_pts;
  room_pts.reserve(sp.walkable.size() * 2);
  for (auto& v : sp.walkable) {
    room_pts.push_back(v.x);
    room_pts.push_back(v.y);
  }
  ctx.room = sb_add_element(ctx.space, 0);
  if (!sp.walkable.empty()) {
    sb_set_element_polygon(ctx.space, ctx.room, SB_GEOMETRY_NAVIGABLE,
                           room_pts.data(),
                           static_cast<int64_t>(sp.walkable.size()));
  }
  ctx.geom.walkable = sp.walkable;

  ctx.elem_by_tag["room"] = ctx.room;
  std::unordered_map<std::string, Vec2> world_pos = {{"room", {0.0, 0.0}}};
  std::unordered_map<std::string, double> world_yaw = {{"room", 0.0}};
  std::unordered_map<std::string, double> world_z = {{"room", 0.0}};

  std::vector<const ScenarioElement*> pending;
  pending.reserve(sp.elements.size());
  for (auto& e : sp.elements) pending.push_back(&e);

  while (!pending.empty()) {
    bool progressed = false;
    for (size_t i = 0; i < pending.size();) {
      const ScenarioElement* e = pending[i];
      std::string parent = e->parent.empty() ? "room" : e->parent;
      auto parent_it = ctx.elem_by_tag.find(parent);
      if (parent_it == ctx.elem_by_tag.end()) {
        ++i;
        continue;
      }
      Vec2 parent_pos = world_pos.at(parent);
      double parent_yaw = world_yaw.at(parent);
      double parent_z = world_z.at(parent);

      double cwx = e->transform.pos.x, cwy = e->transform.pos.y,
             cwz = e->transform.pos.z;
      double cwyaw = e->transform.yaw;

      double dx = cwx - parent_pos.x, dy = cwy - parent_pos.y;
      double c = std::cos(-parent_yaw), s = std::sin(-parent_yaw);
      double lx = c * dx - s * dy;
      double ly = s * dx + c * dy;
      double lz = cwz - parent_z;
      double lyaw = cwyaw - parent_yaw;

      sb_element_t elem = sb_add_element(ctx.space, parent_it->second);
      if (e->shared_id != 0) {
        sb_set_element_shared_id(ctx.space, elem, e->shared_id);
      }
      if (!e->tag.empty()) {
        sb_set_element_tag(ctx.space, elem, e->tag.c_str());
      }
      double m[16];
      YawZTransform(lx, ly, lz, lyaw, m);
      if (auto st = SceneOk(ctx, sb_set_element_transform(ctx.space, elem, m),
                            "sb_set_element_transform");
          !st.ok())
        return st;

      if (e->has_footprint) {
        const auto& fp = e->footprint;
        if (fp.kind == "box") {
          // The collider params are in the element's LOCAL frame (the derived
          // envelope footprint is element-local: c/spatialblend.h frame rules).
          double params[6] = {0, 0, fp.size.z * 0.5, fp.size.x, fp.size.y,
                              fp.size.z};
          if (auto st = SceneOk(ctx,
                                sb_set_element_collider(ctx.space, elem,
                                                        SB_COLLIDER_BOX, params, 6),
                                "box footprint");
              !st.ok())
            return st;
        } else if (fp.kind == "sphere") {
          double params[4] = {0, 0, fp.radius, fp.radius};
          if (auto st = SceneOk(ctx,
                                sb_set_element_collider(ctx.space, elem,
                                                        SB_COLLIDER_SPHERE, params, 4),
                                "sphere footprint");
              !st.ok())
            return st;
        } else if (fp.kind == "capsule") {
          double params[6] = {0, 0, fp.height * 0.5, fp.radius, fp.height,
                              static_cast<double>(fp.axis)};
          if (auto st = SceneOk(ctx,
                                sb_set_element_collider(ctx.space, elem,
                                                        SB_COLLIDER_CAPSULE, params, 6),
                                "capsule footprint");
              !st.ok())
            return st;
        } else if (fp.kind == "polygon") {
          std::vector<double> pts;
          pts.reserve(fp.points.size() * 2);
          for (auto& p : fp.points) {
            pts.push_back(p.x);
            pts.push_back(p.y);
          }
          if (auto st = SceneOk(
                  ctx,
                  sb_set_element_polygon(ctx.space, elem, SB_GEOMETRY_CONTAINMENT,
                                         pts.data(),
                                         static_cast<int64_t>(fp.points.size())),
                  "polygon footprint");
              !st.ok())
            return st;
        }
      }

      for (auto& sr : e->sub_regions) {
        std::vector<double> pts;
        pts.reserve(sr.points.size() * 2);
        for (auto& p : sr.points) {
          pts.push_back(p.x);
          pts.push_back(p.y);
        }
        double xform[16];
        const double* xform_ptr = nullptr;
        if (sr.height != 0.0) {
          YawZTransform(0, 0, sr.height, 0.0, xform);
          xform_ptr = xform;
        }
        sb_add_sub_region(ctx.space, elem, sr.name.c_str(), sr.kind.c_str(),
                          pts.data(), static_cast<int32_t>(sr.points.size()),
                          xform_ptr);
      }

      if (!e->tag.empty()) {
        ctx.elem_by_tag[e->tag] = elem;
        world_pos[e->tag] = {cwx, cwy};
        world_yaw[e->tag] = cwyaw;
        world_z[e->tag] = cwz;
        ctx.geom.pos_by_tag[e->tag] = {cwx, cwy};
        ctx.geom.yaw_by_tag[e->tag] = cwyaw;
        if (e->has_footprint && e->footprint.kind == "box") {
          ctx.geom.footprint_by_tag[e->tag] =
              BoxWorldCorners(cwx, cwy, cwyaw, e->footprint.size.x,
                             e->footprint.size.y);
        }
      }
      if (e->shared_id != 0) {
        ctx.elem_by_shared_id[e->shared_id] = elem;
        ctx.shared_id_by_elem[elem] = e->shared_id;
        ctx.geom.tag_by_shared_id[e->shared_id] = e->tag;
      }

      pending.erase(pending.begin() + static_cast<long>(i));
      progressed = true;
    }
    if (!progressed) {
      return absl::InvalidArgumentError(
          "BuildElements: unresolved element parent chain (missing parent "
          "tag or cycle) in space '" +
          sp.id + "'");
    }
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Actor construction (source side): creates the actor, its anchors, a
// grounding leaf element (child of room) carrying the actor's shared_id, and
// its bindings. Initial source propagation is deliberately performed later,
// after cross-space clone actors are attached, so both transports observe the
// same initial payload/target lifecycle.
// ---------------------------------------------------------------------------
sb_actor_t BuildSourceActor(const ScenarioActor& a, SpaceCtx& home) {
  sb_actor_t act = sb_create_actor(home.space);
  for (auto& an : a.anchors) {
    sb_add_anchor(home.space, act, an.name.c_str(), an.offset.x, an.offset.y,
                 an.offset.z, nullptr);
  }
  sb_element_t elem = sb_add_element(home.space, home.room);
  if (a.shared_id != 0) {
    sb_set_element_shared_id(home.space, elem, a.shared_id);
  }
  double m[16];
  YawZTransform(a.start_x, a.start_y, 0.0, a.start_yaw, m);
  sb_set_element_transform(home.space, elem, m);
  sb_set_actor_element(home.space, act, elem);

  for (auto& b : a.bindings) {
    sb_kernel_type_t k = KernelFromName(b.kernel);
    if (k == SB_KERNEL_AFFORDANCE_POSE) {
      sb_bind_affordance(home.space, act, b.anchor.c_str(),
                         b.has_tag_filter ? b.tag_filter.c_str() : nullptr);
    } else {
      sb_bind(home.space, act, b.anchor.c_str(), k);
    }
  }

  if (a.shared_id != 0) {
    home.elem_by_shared_id[a.shared_id] = elem;
    home.shared_id_by_elem[elem] = a.shared_id;
  }
  return act;
}

// ---------------------------------------------------------------------------
// Deterministic seeded trajectory generator (behavior.script -> 30Hz frames).
// See unity_twin.h for the documented op set / simplifications.
// ---------------------------------------------------------------------------
struct Pose {
  double x = 0.0, y = 0.0, yaw = 0.0;
};

std::vector<Pose> RenderScript(const ScenarioBehavior& behavior, double start_x,
                               double start_y, double start_yaw,
                               const SpaceGeom& geom, double dt,
                               uint64_t seed) {
  std::vector<Pose> frames;
  double x = start_x, y = start_y, yaw = start_yaw;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> ndist(0.0, 1.0);
  std::uniform_real_distribution<double> udist(-1.0, 1.0);
  std::uniform_real_distribution<double> udist01(0.0, 1.0);

  for (const auto& op : behavior.script) {
    double duration = op.raw.value("duration", 1.0);
    int n = std::max(1, static_cast<int>(std::llround(duration / dt)));
    const std::string& kind = op.op;

    if (kind == "approach" || kind == "move") {
      double tx = x, ty = y;
      double target_yaw = yaw;
      if (kind == "approach") {
        std::string tag = op.raw.at("tag").get<std::string>();
        double distance = op.raw.value("distance", 0.5);
        auto it = geom.pos_by_tag.find(tag);
        Vec2 center = it != geom.pos_by_tag.end() ? it->second : Vec2{x, y};
        double dx = x - center.x, dy = y - center.y;
        double r = std::hypot(dx, dy);
        double dirx = (r > 1e-9) ? dx / r : 0.0;
        double diry = (r > 1e-9) ? dy / r : -1.0;
        tx = center.x + dirx * distance;
        ty = center.y + diry * distance;
        target_yaw = std::atan2(center.y - ty, center.x - tx);
      } else {  // move
        const auto& tgt = op.raw.at("target");
        tx = tgt.at(0).get<double>();
        ty = tgt.at(1).get<double>();
        if (std::hypot(tx - x, ty - y) > 1e-9) {
          target_yaw = std::atan2(ty - y, tx - x);
        }
      }
      double sx = x, sy = y, syaw = yaw;
      double yaw_delta = WrapAngle(target_yaw - syaw);
      for (int i = 1; i <= n; ++i) {
        double t = static_cast<double>(i) / n;
        frames.push_back({sx + t * (tx - sx), sy + t * (ty - sy),
                          syaw + t * yaw_delta});
      }
      x = tx;
      y = ty;
      yaw = target_yaw;
    } else if (kind == "look_at") {
      std::string tag = op.raw.at("tag").get<std::string>();
      auto it = geom.pos_by_tag.find(tag);
      Vec2 center = it != geom.pos_by_tag.end() ? it->second : Vec2{x, y};
      double target_yaw = std::atan2(center.y - y, center.x - x);
      double syaw = yaw;
      double yaw_delta = WrapAngle(target_yaw - syaw);
      for (int i = 1; i <= n; ++i) {
        double t = static_cast<double>(i) / n;
        frames.push_back({x, y, syaw + t * yaw_delta});
      }
      yaw = target_yaw;
    } else if (kind == "wander") {
      double radius = op.raw.value("radius", 0.3);
      double ox = x, oy = y;
      for (int i = 0; i < n; ++i) {
        double ang = udist(rng) * M_PI;
        double rad = udist01(rng) * radius;
        x = ox + rad * std::cos(ang);
        y = oy + rad * std::sin(ang);
        frames.push_back({x, y, yaw});
      }
    } else {
      // reach | sit | stand | wait | release: hold position/yaw, optional
      // seeded Gaussian jitter. No hand-IK affordance solve (documented
      // simplification -- root/chest pose only).
      double jitter_sigma = op.raw.value("jitter_sigma", 0.0);
      double bx = x, by = y;
      for (int i = 0; i < n; ++i) {
        double fx = bx, fy = by;
        if (jitter_sigma > 0.0) {
          fx += ndist(rng) * jitter_sigma;
          fy += ndist(rng) * jitter_sigma;
        }
        frames.push_back({fx, fy, yaw});
      }
    }
  }
  return frames;
}

// ---------------------------------------------------------------------------
// Harness-level network delay queue: implements latency_ms/jitter_ms/
// drop_rate/reorder by gating WHEN the harness calls sb_update_actor_state on
// the source actor, since sb_update_actor_state is fire-and-forget with no
// gRPC-level delay-injection hook exposed by the C API (sb_try_wait is a
// no-op stub for its handle). This is a
// harness-level approximation, not true wire-level packet injection.
// ---------------------------------------------------------------------------
class NetworkQueue {
 public:
  NetworkQueue(double latency_ms, double jitter_ms, double drop_rate,
              bool reorder, double dt, uint64_t seed)
      : latency_ms_(latency_ms),
        jitter_ms_(jitter_ms),
        drop_rate_(drop_rate),
        reorder_(reorder),
        dt_(dt > 0 ? dt : 1.0 / 30.0),
        rng_(seed) {}

  // Feed this frame's freshly-scripted pose in; returns the (possibly empty)
  // set of poses that should actually be applied to the source actor this
  // frame, in the order they should be applied.
  std::vector<Pose> Step(int frame, const Pose& p) {
    std::normal_distribution<double> jitter(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    bool dropped = drop_rate_ > 0.0 && unif(rng_) < drop_rate_;
    if (!dropped) {
      double lat = latency_ms_;
      if (jitter_ms_ > 0.0) lat += jitter(rng_) * jitter_ms_;
      if (lat < 0.0) lat = 0.0;
      int delay_frames =
          static_cast<int>(std::llround((lat / 1000.0) / dt_));
      pending_.push_back({frame + delay_frames, p});
      if (reorder_ && pending_.size() >= 2) {
        std::swap(pending_[pending_.size() - 1], pending_[pending_.size() - 2]);
      }
    }
    std::vector<Pose> ready;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (it->release_frame <= frame) {
        ready.push_back(it->pose);
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
    return ready;
  }

 private:
  struct Item {
    int release_frame;
    Pose pose;
  };
  double latency_ms_, jitter_ms_, drop_rate_;
  bool reorder_;
  double dt_;
  std::mt19937_64 rng_;
  std::vector<Item> pending_;
};

void ApplyCoreConfig(const nlohmann::json& core,
                     std::vector<std::unique_ptr<SpaceCtx>>& ctxs,
                     double dt) {
  for (auto& c : ctxs) {
    sb_set_update_interval(c->space, dt);
    if (!core.is_object()) continue;
    if (core.contains("referent_resolve")) {
      std::string v = core.at("referent_resolve").get<std::string>();
      sb_set_referent_resolve_mode(
          c->space, v == "sharedid" ? SB_REFERENT_RESOLVE_SHAREDID
                                    : SB_REFERENT_RESOLVE_LINKER);
    }
    if (core.contains("weight_smoothing")) {
      sb_set_weight_smoothing(c->space,
                              core.at("weight_smoothing").get<double>());
    }
    if (core.contains("min_move")) {
      sb_set_min_move(c->space, core.at("min_move").get<double>());
    }
    if (core.contains("min_yaw")) {
      sb_set_min_yaw(c->space, core.at("min_yaw").get<double>());
    }
    if (core.contains("policy")) {
      std::string v = core.at("policy").get<std::string>();
      sb_realization_policy_t p = SB_REALIZATION_POLICY_PROGRESSIVE;
      if (v == "uniform") {
        p = SB_REALIZATION_POLICY_UNIFORM;
      } else if (v == "greedy") {
        p = SB_REALIZATION_POLICY_GREEDY;
      }
      sb_set_realization_policy(c->space, p);
    }
  }
}

}  // namespace

absl::StatusOr<RunResult> RunScenario(const ScenarioModel& model,
                                      const RunOptions& opts) {
  if (model.spaces.empty()) {
    return absl::InvalidArgumentError("scenario has no spaces");
  }
  if (model.actors.empty()) {
    return absl::InvalidArgumentError("scenario has no actors");
  }

  // Transport::kInProcess: all scenario spaces share ONE sb_session_t (hence
  // one shared SpaceHub -- see LocalContext::AddSpace, which wires every
  // space it creates to its session's single hub_), and NO
  // sb_host_space/sb_connect_space call is ever made. This is a deliberate
  // DEVIATION from the literal "host + connect on a free loopback port via
  // sb_host_space/sb_connect_space" approach, discovered and
  // documented while debugging R1:
  //
  //  - Historically, sb_host_space()/sb_connect_space() each overwrote that
  //    ONE session hub's single-slot callbacks (SetElementEventCallback /
  //    SetRelationUpdateCallback) to point at their own HubReceiver /
  //    RemoteHub. With 2+ scenario spaces in one session, the last
  //    host/connect call clobbered the other's registration -- see
  //    spatialblend/api/c/spatialblend_c.cc's HubRole guard, which now
  //    makes a second host/connect call on the same session an explicit
  //    error (sb_host_space/sb_connect_space return 0) instead of silently
  //    clobbering. "One session = one hub role" is the resulting API
  //    contract: a single sb_session_t may EITHER host EITHER connect, never
  //    both, and never twice. That is exactly why Transport::kGrpcTwoSessions
  //    below gives every scenario space (host and peers alike) its OWN
  //    sb_session_t.
  //  - Separately, sb_connect_space's bootstrap step (RemoteHub::OnConnect)
  //    clones the HOST's *entire* element tree verbatim onto the client
  //    using the host's own sequential element ids (HubReceiver::
  //    BuildBootstrapPayload walks the full tree, not just shared/propagate-
  //    flagged elements). That assumes both sides start as the same blank
  //    scene (a real "presence sync" client) -- it would actively conflict
  //    with this harness's scenario shape, where source and target each
  //    already have their OWN independently-authored geometry (that's the
  //    entire point of the cross-space regression corpus), UNLESS the
  //    host/connect call happens before either side's geometry is built (see
  //    Transport::kGrpcTwoSessions below, which does exactly that).
  //  - The actual regression precedent (spatialblend/net/cross_space_test.cc)
  //    and the C API's own precedent integration test (CApiE2ETest.
  //    SourceToTargetCallback) both use exactly this single-session/shared-hub
  //    pattern -- Space::ShareElement/ResolveSharedElement (sb_share_element /
  //    sb_resolve_shared_element at the C-API level) propagate correctly
  //    through it with NO sb_host_space/sb_connect_space call at all, and a
  //    peer's avatar receives realization targets for a source actor with no
  //    explicit binding on the peer side, purely from the shared hub relaying
  //    RemoteRelationUpdate payloads that the peer's Space::OnRemoteUpdate
  //    resolves using its own local geometry.
  //
  // Transport::kGrpcTwoSessions: the source space gets its own sb_session_t
  // and is hosted (sb_host_space) on a harness-probed free loopback port;
  // every OTHER scenario space gets its own separate sb_session_t and
  // connects to that port (sb_connect_space). Element sharing / relation
  // propagation between them now travels over the real gRPC wire
  // (HubReceiver <-> RemoteHub/SubscribeStreamCall), not an in-process
  // SpaceHub::Propagate call.
  // latency_ms/jitter_ms/drop_rate/reorder are still faithfully modeled --
  // via the harness-level NetworkQueue gating WHEN sb_update_actor_state is
  // called below -- entirely independently of this transport choice (the C
  // API exposes no wire-level delay-injection hook to do otherwise).
  std::string source_id;
  if (model.has_identity && !model.identity.share_from.empty()) {
    source_id = model.identity.share_from;
  } else {
    for (auto& sp : model.spaces) {
      if (sp.role == "source") {
        source_id = sp.id;
        break;
      }
    }
    if (source_id.empty()) source_id = model.spaces.front().id;
  }
  {
    bool source_declared = false;
    for (const auto& sp : model.spaces) {
      if (sp.id == source_id) {
        source_declared = true;
        break;
      }
    }
    if (!source_declared) {
      return absl::InvalidArgumentError("identity.share_from '" + source_id +
                                        "' does not name a declared space");
    }
  }

  bool use_network = model.spaces.size() > 1 && model.has_identity;
  bool split_sessions =
      use_network && opts.transport == Transport::kGrpcTwoSessions;

  // Every sb_session_t created for this run, in creation order (source/host
  // first when split_sessions, otherwise the single shared session). Torn
  // down in REVERSE order so a split run destroys peer/client sessions
  // before the host session -- the host's HubReceiver keeps live gRPC
  // streams to every connected peer, so tearing down a peer first lets that
  // peer's SubscribeStreamCall unwind while the host side is still alive,
  // rather than the host vanishing out from under an active stream.
  std::vector<sb_session_t> sessions;
  struct SessionGuard {
    std::vector<sb_session_t>* sessions;
    ~SessionGuard() {
      for (auto it = sessions->rbegin(); it != sessions->rend(); ++it) {
        sb_destroy_session(*it);
      }
    }
  } session_guard{&sessions};

  std::vector<std::unique_ptr<SpaceCtx>> ctxs;
  std::unordered_map<std::string, SpaceCtx*> ctx_by_id;
  if (split_sessions) {
    sb_session_t host_session = sb_create_session();
    sessions.push_back(host_session);
    auto host_ctx = std::make_unique<SpaceCtx>();
    host_ctx->scenario_space_id = source_id;
    host_ctx->space = sb_add_space(host_session);
    ctx_by_id[source_id] = host_ctx.get();
    ctxs.push_back(std::move(host_ctx));

    for (const auto& sp : model.spaces) {
      if (sp.id == source_id) continue;
      sb_session_t peer_session = sb_create_session();
      sessions.push_back(peer_session);
      auto ctx = std::make_unique<SpaceCtx>();
      ctx->scenario_space_id = sp.id;
      ctx->space = sb_add_space(peer_session);
      ctx_by_id[sp.id] = ctx.get();
      ctxs.push_back(std::move(ctx));
    }
  } else {
    sb_session_t session = sb_create_session();
    sessions.push_back(session);
    for (const auto& sp : model.spaces) {
      auto ctx = std::make_unique<SpaceCtx>();
      ctx->scenario_space_id = sp.id;
      ctx->space = sb_add_space(session);
      ctx_by_id[sp.id] = ctx.get();
      ctxs.push_back(std::move(ctx));
    }
  }

  ApplyCoreConfig(model.core, ctxs, opts.dt);

  double v_max = 1.4, w_max = 2.0;
  if (model.core.is_object() && model.core.contains("controller")) {
    const auto& ctl = model.core.at("controller");
    if (ctl.contains("v_max")) v_max = ctl.at("v_max").get<double>();
    if (ctl.contains("w_max")) w_max = ctl.at("w_max").get<double>();
  }

  std::string trace_path;
  if (!opts.run_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(opts.run_dir, ec);
    trace_path = opts.run_dir + "/trace.jsonl";
    std::filesystem::remove(trace_path, ec);  // SetPath appends; start clean.
    // sb_set_trace is genuinely per-space (TraceSink::ForSpace) --
    // binding only ctxs.front()'s space (the earlier call, back when every
    // space shared one process-wide sink regardless of which handle was
    // passed) would leave every OTHER scenario space's tracing on the
    // process-wide default, i.e. untraced by default. Bind every space in
    // the run to the SAME path so trace.jsonl keeps carrying every space's
    // lines in one file, exactly as before -- now via N independent
    // per-space sinks that all happen to open the same file for append.
    for (auto& c : ctxs) {
      sb_set_trace(c->space, trace_path.c_str());
    }
  }

  if (split_sessions) {
    // Host + connect BEFORE any element/actor is built (see the transport
    // rationale comment above) so both sides are still genuinely blank and
    // RemoteHub::OnConnect's bootstrap-clone step is a harmless no-op.
    int32_t port = PickFreePort(opts.preferred_port);
    if (port <= 0) {
      return absl::InternalError(
          "kGrpcTwoSessions: failed to probe a free loopback port");
    }
    SpaceCtx* host_ctx = ctx_by_id.at(source_id);
    sb_handle_t host_handle = sb_host_space(host_ctx->space, port);
    if (host_handle == 0) {
      return absl::InternalError("sb_host_space failed on source space '" +
                                 source_id + "'");
    }
    if (!sb_try_wait(host_handle, 2000)) {
      return absl::InternalError(
          "sb_host_space did not become ready within timeout");
    }
    for (const auto& sp : model.spaces) {
      if (sp.id == source_id) continue;
      SpaceCtx* peer_ctx = ctx_by_id.at(sp.id);
      sb_handle_t connect_handle =
          sb_connect_space(peer_ctx->space, "127.0.0.1", port);
      if (connect_handle == 0) {
        return absl::InternalError("sb_connect_space failed on space '" +
                                   sp.id + "'");
      }
      if (!sb_try_wait(connect_handle, 5000)) {
        return absl::InternalError(
            "sb_connect_space did not become ready within timeout for "
            "space '" +
            sp.id + "'");
      }
    }
    // Give the client-side gRPC completion-queue background thread
    // (GrpcClientBase::HandleRpcs, spawned inside sb_connect_space's
    // RemoteHub::Create) a moment to settle its Subscribe stream before any
    // element traffic starts -- sb_try_wait only confirms the synchronous
    // bootstrap RPC completed, not that the async stream's first Proceed()
    // has run.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (auto& c : ctxs) {
    sb_register_on_element_added(c->space, &OnElementAddedCb, c.get());
    sb_register_on_targets(c->space, &OnTargetsCb, c.get());
    sb_register_on_relation_event(c->space, &OnRelationEventCb, c.get());
    // When set, overrides the scenario's own JSON "core.referent_resolve"
    // (already applied per-space by ApplyCoreConfig above) uniformly across
    // every space -- lets harness_test.cc's A/B test force one mode without
    // needing two copies of a fixture. Space::OnRemoteUpdate is what
    // actually reads this mode (see space.cc), so it only has an
    // observable effect on a space that ends up on the RECEIVING end of a
    // remote relation update.
    if (opts.override_referent_resolve_mode) {
      sb_set_referent_resolve_mode(c->space, opts.referent_resolve_mode);
    }
  }

  for (const auto& sp : model.spaces) {
    absl::Status st = BuildElements(sp, *ctx_by_id.at(sp.id));
    if (!st.ok()) return st;
  }

  std::unordered_map<std::string, sb_actor_t> src_actor_by_id;
  for (const auto& a : model.actors) {
    auto it = ctx_by_id.find(a.space);
    if (it == ctx_by_id.end()) {
      return absl::InvalidArgumentError("actor '" + a.id +
                                        "' names unknown space '" + a.space +
                                        "'");
    }
    src_actor_by_id[a.id] = BuildSourceActor(a, *it->second);
  }

  // Share/resolve: furniture elements marked propagate + every source-space
  // actor's own grounding element (identity.expect always carries an entry
  // for the actor's shared_id too, per the confirmed fixture shape).
  std::map<std::pair<std::string, int64_t>, sb_element_t> peer_resolved;
  if (use_network) {
    SpaceCtx* src_ctx = ctx_by_id.at(source_id);
    const ScenarioSpace* src_scn = model.FindSpace(source_id);
    if (src_scn == nullptr) {
      return absl::InternalError("internal: source space vanished");
    }

    struct ShareItem {
      int64_t shared_id;
      sb_element_t local;
      std::string tag;
      bool actor = false;  // the grounding element of a source actor
    };
    std::vector<ShareItem> items;
    for (const auto& e : src_scn->elements) {
      if (e.propagate && e.shared_id != 0) {
        items.push_back({e.shared_id, src_ctx->elem_by_shared_id.at(e.shared_id),
                         e.tag});
      }
    }
    for (const auto& a : model.actors) {
      if (a.space == source_id && a.shared_id != 0) {
        items.push_back(
            {a.shared_id, src_ctx->elem_by_shared_id.at(a.shared_id), "", true});
      }
    }

    for (const auto& item : items) {
      std::unordered_map<std::string, int> prior_counts;
      for (const auto& kv : ctx_by_id) {
        if (kv.second != src_ctx) prior_counts[kv.first] = kv.second->element_added_count.load();
      }
      sb_share_element(src_ctx->space, item.local, nullptr, 0);
      for (auto& kv : ctx_by_id) {
        if (kv.second == src_ctx) continue;
        SpaceCtx* peer = kv.second;
        if (!WaitForCountAbove(peer->element_added_count,
                               prior_counts.at(kv.first), 2000)) {
          return absl::InternalError(
              "timed out waiting for on_element_added (shared_id " +
              std::to_string(item.shared_id) + ") on space '" + kv.first +
              "'");
        }
        sb_element_t received = 0;
        std::string recv_tag;
        {
          std::lock_guard<std::mutex> lock(peer->added_mu);
          for (auto rit = peer->added_log.rbegin();
               rit != peer->added_log.rend(); ++rit) {
            if (rit->shared_id == item.shared_id) {
              received = rit->local;
              recv_tag = rit->tag;
              break;
            }
          }
        }
        if (received == 0) {
          return absl::InternalError(
              "share/resolve: no on_element_added entry found for shared_id " +
              std::to_string(item.shared_id) + " on space '" + kv.first + "'");
        }

        std::string resolution = "add_as_new";
        if (model.has_identity) {
          for (const auto& ex : model.identity.expect) {
            if (ex.shared_id == item.shared_id) {
              resolution = ex.resolution;
              break;
            }
          }
        }

        sb_element_t resolved = received;
        if (item.actor) {
          // The twin plays the application: it instantiates the clone avatar's
          // grounding element INSIDE the receiver's room (a received element is
          // an unparented temp), so the avatar inherits the room's walkable
          // scope and its proposals are clamped to it.
          resolved = sb_add_element(peer->space, peer->room);
          if (resolved == 0) {
            return absl::InternalError("could not create the clone avatar element in space '" + kv.first + "'");
          }
        } else if (resolution == "link_to_existing") {
          // Tag-based lookup (NOT shared_id-based): identity.mismatched_ids
          // scenarios deliberately give the pre-existing target-side element
          // a DIFFERENT shared_id than the one carried by the shared
          // payload -- the two spaces' corresponding real-world objects are
          // tied together only by matching tag strings in that case.
          std::string lookup_tag = !item.tag.empty() ? item.tag : recv_tag;
          auto tag_it = peer->elem_by_tag.find(lookup_tag);
          if (tag_it == peer->elem_by_tag.end()) {
            return absl::InternalError(
                "link_to_existing: no pre-existing element tagged '" +
                lookup_tag + "' in space '" + kv.first + "'");
          }
          resolved = tag_it->second;
        }
        sb_resolve_shared_element(peer->space, received, resolved);
        peer->elem_by_shared_id[item.shared_id] = resolved;
        if (!item.tag.empty()) peer->elem_by_tag[item.tag] = resolved;
        peer_resolved[{kv.first, item.shared_id}] = resolved;

        // Record the source-shared-id -> local-shared-id translation for
        // BestCommittedReferent (see SpaceCtx::source_shared_to_local_shared).
        // Default (add_as_new / matching-ids case): identity, since `resolved`
        // is just the freshly-shared temp with the SAME shared_id. Overridden
        // below when `resolved` is a pre-existing local element with its own
        // shared_id (the mismatched-ids link_to_existing case).
        int64_t local_shared_id = item.shared_id;
        auto rev_it = peer->shared_id_by_elem.find(resolved);
        if (rev_it != peer->shared_id_by_elem.end()) {
          local_shared_id = rev_it->second;
        }
        peer->source_shared_to_local_shared[item.shared_id] = local_shared_id;
      }
    }
  }

  // Target-side "clone" avatars: one per (peer space, source actor) pair
  // whose grounding element was just resolved above. Bindings are NOT set
  // here -- they propagate automatically over the wire from the source
  // actor's sb_bind/sb_bind_affordance calls (confirmed by
  // CApiE2ETest.SourceToTargetCallback, which never calls sb_bind on its
  // target-space "avatar" yet still receives realization targets for it).
  std::map<std::pair<std::string, std::string>, sb_actor_t> target_avatar;
  if (use_network) {
    for (const auto& a : model.actors) {
      if (a.space != source_id || a.shared_id == 0) continue;
      for (auto& kv : ctx_by_id) {
        if (kv.first == source_id) continue;
        auto it = peer_resolved.find({kv.first, a.shared_id});
        if (it == peer_resolved.end()) continue;
        sb_actor_t avatar = sb_create_actor(kv.second->space);
        for (auto& an : a.anchors) {
          sb_add_anchor(kv.second->space, avatar, an.name.c_str(),
                       an.offset.x, an.offset.y, an.offset.z, nullptr);
        }
        sb_set_actor_element(kv.second->space, avatar, it->second);
        // The twin plays the application: it owns each clone's embodied
        // pose, starting at the origin, and publishes it into the sampling
        // port (core keeps no pose and falls back to none).
        {
          sb_anchor_pose_t start{};
          start.struct_size = sizeof(start);
          start.orientation.struct_size = sizeof(sb_orientation_t);
          start.epoch = 1;
          sb_publish_anchor_pose(kv.second->space, avatar, 0, &start, 0.0, 1);
        }
        target_avatar[{kv.first, a.id}] = avatar;
      }
    }
  }

  // Now that every peer clone actor is attached to its resolved element, emit
  // every declared actor's start pose in both transports. This includes
  // unscripted secondary actors, preserves their declared initial state, and
  // prevents in-process callbacks from observing a different setup lifecycle
  // from gRPC. For split sessions, receipt is exact-sequence proof; the
  // in-process callback is synchronous through the shared hub.
  for (const auto& actor : model.actors) {
    auto source_it = src_actor_by_id.find(actor.id);
    if (source_it == src_actor_by_id.end()) continue;
    SpaceCtx* actor_source = ctx_by_id.at(actor.space);
    const int64_t before = sb_get_last_emitted_update_sequence(
        actor_source->space, source_it->second);
    sb_update_actor_state(actor_source->space, source_it->second,
                          actor.start_x, actor.start_y, actor.start_yaw);
    const int64_t emitted = sb_get_last_emitted_update_sequence(
        actor_source->space, source_it->second);

    // Only actors cloned out of the designated source space have a receiver
    // in this harness. If an actor has no relation payload, the native source
    // correctly keeps its deadband sequence unchanged and there is no wire
    // receipt to await.
    if (!split_sessions || actor.space != source_id || emitted <= before) {
      continue;
    }
    for (auto& peer : ctx_by_id) {
      if (peer.first == source_id) continue;
      auto avatar_it = target_avatar.find({peer.first, actor.id});
      if (avatar_it == target_avatar.end()) continue;
      absl::Status receipt = WaitForExactSequenceApplied(
          peer.second->space, actor_source->space, source_it->second,
          emitted, 1000);
      if (!receipt.ok()) {
        return absl::DeadlineExceededError(
            "initial actor '" + actor.id + "' receipt: " +
            std::string(receipt.message().data(), receipt.message().size()));
      }
      sb_poll_realization_targets(peer.second->space, avatar_it->second);
    }
  }

  // Driven actor: by fixture convention, the FIRST declared actor is the one
  // whose motion is rendered from the top-level "behavior" script; any other
  // declared actor holds its start pose for the whole run (the setup path
  // explicitly propagated that initial pose after clone construction).
  const ScenarioActor& driven = model.actors.front();
  SpaceCtx* src_ctx = ctx_by_id.at(driven.space);
  sb_actor_t src_actor = src_actor_by_id.at(driven.id);

  sb_space_t measure_space = src_ctx->space;
  sb_actor_t measure_actor = src_actor;
  SpaceGeom* measure_geom = &src_ctx->geom;
  SpaceCtx* measure_ctx = src_ctx;
  SpaceCtx* target_ctx_for_wait = nullptr;
  for (auto& kv : ctx_by_id) {
    if (kv.first == driven.space) continue;
    auto it = target_avatar.find({kv.first, driven.id});
    if (it != target_avatar.end()) {
      measure_space = kv.second->space;
      measure_actor = it->second;
      measure_geom = &kv.second->geom;
      measure_ctx = kv.second;
      target_ctx_for_wait = kv.second;
      break;
    }
  }
  // The measured clone's driver: velocity- and turn-bounded, stepping toward
  // each proposal and publishing the new pose (core paces nothing).
  driver::Pose body;
  uint64_t body_sequence = 1;  // the initial publish above used sequence 1
  driver::Bounds driver_bounds;
  driver_bounds.max_linear_speed = v_max;
  driver_bounds.max_angular_speed = w_max;

  std::vector<Pose> script_frames =
      RenderScript(model.behavior, driven.start_x, driven.start_y,
                  driven.start_yaw, src_ctx->geom, opts.dt, opts.seed);
  int total_script_frames = static_cast<int>(script_frames.size());
  if (total_script_frames == 0) {
    script_frames.push_back({driven.start_x, driven.start_y, driven.start_yaw});
    total_script_frames = 1;
  }

  double latency_ms = model.has_network ? model.network.latency_ms : 0.0;
  double jitter_ms = model.has_network ? model.network.jitter_ms : 0.0;
  double drop_rate = model.has_network ? model.network.drop_rate : 0.0;
  bool reorder = model.has_network && model.network.reorder;
  if (opts.override_latency) latency_ms = opts.override_latency_ms;
  NetworkQueue netq(latency_ms, jitter_ms, drop_rate, reorder, opts.dt,
                    opts.seed);

  RunResult result;
  int total_frames = std::min(opts.max_frames, total_script_frames);
  bool have_prev = false;
  double prev_x = 0, prev_y = 0, prev_yaw = 0;
  int64_t last_source_sequence =
      sb_get_last_emitted_update_sequence(src_ctx->space, src_actor);

  for (int frame = 0; frame < total_frames; ++frame) {
    const Pose& p = script_frames[frame];
    auto ready = netq.Step(frame, p);
    for (const auto& rp : ready) {
      int prior =
          target_ctx_for_wait ? target_ctx_for_wait->target_events.load() : 0;
      const int64_t before_sequence =
          sb_get_last_emitted_update_sequence(src_ctx->space, src_actor);
      sb_update_actor_state(src_ctx->space, src_actor, rp.x, rp.y, rp.yaw);
      const int64_t emitted_sequence =
          sb_get_last_emitted_update_sequence(src_ctx->space, src_actor);
      last_source_sequence = emitted_sequence;
      if (target_ctx_for_wait != nullptr) {
        if (split_sessions) {
          // Polling an already-known target is not proof that THIS source
          // state arrived: sb_poll_realization_targets intentionally replays
          // the current snapshot. Only wait when the source actually emitted
          // a new wire sequence, then require the receiver to have accepted
          // that exact sequence before one post-receipt poll/tick.
          if (emitted_sequence > before_sequence) {
            absl::Status receipt = WaitForExactSequenceApplied(
                measure_space, src_ctx->space, src_actor, emitted_sequence,
                1000);
            if (!receipt.ok()) {
              return absl::DeadlineExceededError(
                  "frame " + std::to_string(frame) + ": " +
                  std::string(receipt.message().data(), receipt.message().size()));
            }
            sb_poll_realization_targets(measure_space, measure_actor);
          }
        } else {
          WaitForCountAbove(target_ctx_for_wait->target_events, prior, 500);
        }
      }
    }

    sb_actor_state_t src_state = sb_get_actor_state(src_ctx->space, src_actor);

    sb_actor_state_t out{};
    {
      sb_proposal_set_t set{};
      set.struct_size = sizeof(set);
      sb_anchor_proposal_t proposal{};  // the root anchor's proposal is the first
      proposal.struct_size = sizeof(proposal);
      if (sb_read_proposals(measure_space, measure_actor, &set, &proposal, 1) == 0 &&
          set.anchor_count >= 1 &&
          (proposal.status == SB_PROPOSAL_EXACT ||
           proposal.status == SB_PROPOSAL_INEXACT)) {
        driver::Pose target;
        target.position = Eigen::Vector2d(proposal.x, proposal.y);
        target.yaw = proposal.orientation.yaw;
        double course_xy[2 * 64];
        std::vector<Eigen::Vector2d> course;
        const int32_t n = sb_query_course(
            measure_space, measure_actor, body.position.x(), body.position.y(),
            target.position.x(), target.position.y(), course_xy, 64);
        for (int32_t i = 0; i < n && i < 64; ++i)
          course.emplace_back(course_xy[2 * i], course_xy[2 * i + 1]);
        if (n > 64) course.clear();
        body = driver::Step(body, target, course, driver_bounds, opts.dt);
        sb_anchor_pose_t pose{};
        pose.struct_size = sizeof(pose);
        pose.x = body.position.x();
        pose.y = body.position.y();
        pose.orientation.struct_size = sizeof(sb_orientation_t);
        pose.orientation.yaw = body.yaw;
        pose.epoch = 1;
        sb_publish_anchor_pose(measure_space, measure_actor, 0, &pose, 0.0,
                               ++body_sequence);
      }
    }
    out.position_x = body.position.x();
    out.position_y = body.position.y();
    out.yaw = body.yaw;

    // See BestCommittedReferent's doc comment: sb_get_relation_states() would
    // return empty here for a peer/target avatar (it reads Space::bindings_,
    // populated only by an explicit sb_bind() we deliberately never issue on
    // the target side). The relation-lifecycle-phase callback fires from
    // Space::OnRemoteUpdate/local-update alike, so it works for both.
    auto [has_committed, best_shared] = BestCommittedReferent(*measure_ctx);
    double best_prox = 0.0;

    bool in_walkable = PointInPolygon(out.position_x, out.position_y,
                                      measure_geom->walkable);
    bool in_footprint = false;
    if (has_committed) {
      auto tag_it = measure_geom->tag_by_shared_id.find(best_shared);
      if (tag_it != measure_geom->tag_by_shared_id.end()) {
        auto fp_it = measure_geom->footprint_by_tag.find(tag_it->second);
        if (fp_it != measure_geom->footprint_by_tag.end()) {
          in_footprint =
              PointInPolygon(out.position_x, out.position_y, fp_it->second);
        }
      }
    }
    bool containment_violation = !(in_walkable || in_footprint);

    bool speed_violation = false, yaw_rate_violation = false;
    if (have_prev) {
      double dist = std::hypot(out.position_x - prev_x, out.position_y - prev_y);
      double speed = dist / opts.dt;
      if (speed > v_max + 1e-6) speed_violation = true;
      double dyaw = std::abs(WrapAngle(out.yaw - prev_yaw));
      double yaw_rate = dyaw / opts.dt;
      if (yaw_rate > w_max + 1e-6) yaw_rate_violation = true;
    }
    prev_x = out.position_x;
    prev_y = out.position_y;
    prev_yaw = out.yaw;
    have_prev = true;

    FrameRecord fr;
    fr.frame = frame;
    fr.t = frame * opts.dt;
    fr.src_x = src_state.position_x;
    fr.src_y = src_state.position_y;
    fr.src_yaw = src_state.yaw;
    fr.tgt_x = out.position_x;
    fr.tgt_y = out.position_y;
    fr.tgt_yaw = out.yaw;
    fr.has_committed_referent = has_committed;
    fr.committed_referent_shared_id = best_shared;
    fr.committed_proximity = has_committed ? best_prox : 0.0;
    fr.containment_violation = containment_violation;
    fr.speed_violation = speed_violation;
    fr.yaw_rate_violation = yaw_rate_violation;
    fr.source_emitted_sequence = last_source_sequence;
    fr.target_applied_sequence = split_sessions
        ? sb_get_last_applied_remote_sequence(measure_space, src_ctx->space,
                                              src_actor)
        : last_source_sequence;
    result.frames.push_back(fr);

    if (containment_violation) result.containment_violations++;
    if (speed_violation) result.speed_violations++;
    if (yaw_rate_violation) result.yaw_rate_violations++;
    if (result.arrival_frame < 0 && frame == total_script_frames - 1) {
      result.arrival_frame = frame;
    }
  }

  // Network/subscribe delivery: the frame loop above runs with no real
  // per-frame wall-clock delay -- this is a deterministic offline harness
  // that calls sb_update_actor_state as fast as the CPU allows -- but Grpc
  // delivery (split_sessions) is genuinely asynchronous: HubReceiver enqueues
  // each relation update on the host's SubscribeReceiverCall write_queue_,
  // and a background completion-queue thread drains it one Write() at a
  // time. Returning immediately here would let RunScenario's SessionGuard
  // (below, at function exit) tear every session down right away, which
  // races that drain: SubscribeReceiverCall::Close() (invoked from
  // ~RemoteSpace() during session teardown) explicitly DISCARDS whatever is
  // still queued -- an update that hasn't been written+read yet by then is
  // silently lost, not merely delayed. Confirmed empirically: a scenario
  // that propagates ~90 updates could see as few as 1-2 actually reach
  // Space::OnRemoteUpdate on the peer without this settle. Bounded (never
  // hangs): poll for new target_events (the same signal the per-frame loop
  // above already uses) and treat a short quiet window as "caught up".
  if (split_sessions && target_ctx_for_wait != nullptr) {
    constexpr int kQuietPollsNeeded = 20;  // ~20ms of no new events
    const auto settle_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    int quiet_polls = 0;
    int last_count = target_ctx_for_wait->target_events.load();
    while (quiet_polls < kQuietPollsNeeded &&
           std::chrono::steady_clock::now() < settle_deadline) {
      sb_poll_realization_targets(measure_space, measure_actor);
      int cur = target_ctx_for_wait->target_events.load();
      if (cur == last_count) {
        ++quiet_polls;
      } else {
        quiet_polls = 0;
        last_count = cur;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  if (!result.frames.empty()) {
    result.final_has_committed_referent = result.frames.back().has_committed_referent;
    result.final_committed_referent_shared_id =
        result.frames.back().committed_referent_shared_id;
  }
  result.trace_path = trace_path;
  return result;
}

}  // namespace harness
}  // namespace spatialblend
