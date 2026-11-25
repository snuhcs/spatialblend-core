// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "spatialblend/model/actor/subject_anchor.h"
#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"
#include "spatialblend/api/c/spatialblend_receiver_approach_context.h"
#include "spatialblend/api/c/spatialblend_receiver_approach_feasibility.h"
#include "spatialblend/api/c/spatialblend_receiver_reevaluation.h"
#include "spatialblend/api/c/spatialblend_planar_support.h"
#include "spatialblend/api/c/spatialblend_containment_transfer.h"
#include "spatialblend/api/c/spatialblend_source_evidence.h"
#include "spatialblend/api/c/spatialblend_observed_collaborator.h"
#include "spatialblend/api/c/spatialblend_observed_source_setup.h"
#include "spatialblend/base/common/trace.h"
#include "spatialblend/model/element/envelope.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/geometry/polygon3d.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/model/element/sub_region.h"
#include "spatialblend/model/element/surface_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/net/hub_receiver.h"
#include "spatialblend/net/remote_hub.h"
#include "spatialblend/net/tunnel_link.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/geometry/planar_frame_chart.h"
#include "spatialblend/geometry/radial_surface_chart.h"
#include "spatialblend/space/space.h"

#ifdef SPATIALBLEND_V2_C_TESTING
#include "spatialblend/api/c/receiver_approach_feasibility_test_api.h"
#include "spatialblend/api/c/scope_extension_test_api.h"
#endif

using spatialblend::GeometryRole;
using spatialblend::RelationKernelType;

#ifdef SPATIALBLEND_V2_C_TESTING
namespace spatialblend {
// This friend exists only in the test-only duplicate C target.  It constructs
// an otherwise ordinary LocalContext whose spaces read one controlled native
// receiver clock; the shipping C target has neither this symbol nor the API.
class CDeterministicSessionFactory {
 public:
  static std::unique_ptr<LocalContext> Create(
      std::function<int64_t()> receiver_input_clock) {
    auto context = std::unique_ptr<LocalContext>(new LocalContext());
    context->receiver_input_clock_ = std::move(receiver_input_clock);
    return context;
  }
};
}  // namespace spatialblend
#endif

// Sentinel handle value for synchronous (already-done) operations.
static constexpr sb_handle_t kSyncDoneHandle = -1;

constexpr uint32_t kObservedCollaboratorMaxFrames = 64;
constexpr uint32_t kObservedCollaboratorMaxObservationsPerFrame = 128;
constexpr uint32_t kObservedCollaboratorMaxPublications = 64;
constexpr uint32_t kObservedCollaboratorMaxAnchorNameBytes = 128;
constexpr int64_t kObservedCollaboratorMaxAgeNs = 1000000000LL;

// A SessionContext owns exactly ONE SpaceHub (see
// LocalContext::GetHub()), and that hub stores its element/relation
// callbacks in single-slot (replace-semantics) std::function members --
// see space/space_hub.h's SetElementEventCallback/SetRelationUpdateCallback.
// sb_host_space() and sb_connect_space() each install their OWN callback
// pair onto that one hub. Originally, calling sb_host_space() on one space
// in a session and then sb_connect_space() on ANOTHER space in the SAME
// session silently clobbered whichever callback pair was installed first --
// e.g. hosting space A then connecting space B in the same session left the
// hub's callbacks pointing at B's RemoteHub, so the host's own
// HubReceiver::PropagateToRemoteSpaces/PropagateElementEventToRemoteSpaces
// path (installed by sb_host_space) silently stopped firing, breaking
// forwarding to any OTHER already-connected remote client, and (depending on
// destruction order at session teardown) could leave a dangling
// shared_ptr capture referencing an already-torn-down receiver/remote_hub.
// Fixed by giving each SessionContext a HubRole latch: the first successful
// sb_host_space() or sb_connect_space() call claims the session's hub for
// that role, and any subsequent sb_host_space()/sb_connect_space() call on
// the SAME session (regardless of which space it targets, and regardless of
// whether it repeats the same role) now fails fast (returns 0) instead of
// silently overwriting the hub's callbacks. This makes "one sb_session_t
// may EITHER host EITHER connect, never both, and never twice" an explicit,
// documented part of the C API's contract rather than an unstated landmine.
// A real multi-space-per-process client that needs BOTH a host role and a
// client role must use two separate sb_session_t (sb_create_session())
// instances, one hub each -- exactly what
// spatialblend/sim/harness/unity_twin.cc's Transport::kGrpcTwoSessions does.
enum class HubRole { kNone, kHost, kClient, kObservationFrames };

struct SessionContext {
  std::unique_ptr<spatialblend::LocalContext> context;
#ifdef SPATIALBLEND_V2_C_TESTING
  std::shared_ptr<std::atomic<int64_t>> test_receiver_input_clock;
#endif
  // See the HubRole comment above. Guarded by g_sessions_mutex (all
  // reads/writes happen while holding it, matching how `context` itself is
  // only ever touched under that lock via GetSession()).
  HubRole hub_role = HubRole::kNone;
  bool rich_retiring = false;
  struct SpaceInfo {
    spatialblend::Space* space;
    std::atomic<int64_t> next_element_id{1};
    // Callback registrations are mutable while hub threads may dispatch.
    // Always copy under this mutex, then invoke after releasing it.
    std::mutex callback_mutex;
    sb_on_targets_t on_targets_callback = nullptr;
    void* on_targets_user_data = nullptr;
    sb_on_rich_realization_t on_rich_callback = nullptr;
    void* on_rich_user_data = nullptr;
    sb_on_element_added_t on_element_added_callback = nullptr;
    void* on_element_added_user_data = nullptr;
    sb_on_element_removed_t on_element_removed_callback = nullptr;
    void* on_element_removed_user_data = nullptr;
    sb_on_link_removed_t on_link_removed_callback = nullptr;
    void* on_link_removed_user_data = nullptr;
    sb_on_relation_event_t on_relation_event_callback = nullptr;
    void* on_relation_event_user_data = nullptr;
    size_t last_received_count = 0;
    // Networking
    std::shared_ptr<spatialblend::HubReceiver> receiver;  // host mode
    std::shared_ptr<spatialblend::RemoteHub> remote_hub;  // client mode
  };
  std::unordered_map<sb_space_t, std::unique_ptr<SpaceInfo>> spaces;
};

struct SupportCorrespondenceRegistration {
  sb_session_t session;
  std::vector<spatialblend::ElementLinker::OwnedLink> installed;
  bool strict_owned = false;
};

struct ObservedSourceSetupRegistration {
  enum class State { kInstalling, kActive, kRetired };
  sb_session_t session = 0;
  sb_space_t space = 0;
  std::vector<spatialblend::Space::ObservedSourceSetupActorResult> actors;
  std::vector<spatialblend::Space::ObservedSourceSetupPointResult> points;
  State state = State::kInstalling;
};

static std::mutex g_sessions_mutex;
// Core relation callbacks run under the Space solve lock. Admission from one
// must fail before either trying that nonrecursive mutex or a registry lock.
static thread_local bool g_inside_relation_callback = false;
static std::unordered_map<sb_session_t, std::unique_ptr<SessionContext>>
    g_sessions;
static std::atomic<int64_t> g_next_session_id{1};
static std::atomic<int64_t> g_next_space_id{1};
static std::atomic<int64_t> g_next_support_registration_id{1};
static std::atomic<int64_t> g_next_observed_source_setup_id{1};
// Caller holds g_sessions_mutex. Never wrap to zero or recycle an old lease.
// Reserving before installation may leave a gap, but exhaustion cannot leak links.
static int64_t ReserveSupportRegistrationId() {
  const auto id = g_next_support_registration_id.load();
  if (id <= 0 || id == INT64_MAX) return 0;
  g_next_support_registration_id.store(id + 1);
  return id;
}
static std::atomic<uint64_t> g_next_rich_lifetime{1};
static std::unordered_map<sb_support_correspondence_registration_t,
                          SupportCorrespondenceRegistration>
    g_support_registrations;
static std::unordered_map<sb_observed_source_setup_t,
                          ObservedSourceSetupRegistration>
    g_observed_source_setups;

// Global space handle -> Space* map for functions that only take sb_space_t.
static std::mutex g_spaces_mutex;
static std::unordered_map<sb_space_t, SessionContext::SpaceInfo*> g_spaces;

// Reverse map: space handle -> session handle (for finding sibling spaces).
static std::unordered_map<sb_space_t, sb_session_t> g_space_to_session;

struct FrozenRichSnapshot {
  spatialblend::ReceiverInputStamp receiver_input;
  sb_rich_realization_header_t header{};
  std::vector<sb_rich_realization_channel_t> channels;
  std::vector<sb_rich_realization_contributor_t> contributors;
  std::vector<uint8_t> utf8;
};

struct RichDelivery {
  std::shared_ptr<const FrozenRichSnapshot> snapshot;
  sb_on_rich_realization_t callback = nullptr;
  void* user_data = nullptr;
  bool terminal = false;
};
struct RichDeliveryState {
  std::mutex mutex;
  bool draining = false;
  bool terminal_enqueued = false;
  // Fence at enqueue, not merely at Core freeze: concurrent dispatchers can
  // otherwise freeze rev1/rev2 in order but enqueue rev2/rev1 out of order.
  uint64_t accepted_revision = 0;
  std::vector<RichDelivery> pending;
};

struct RichActivation {
  sb_space_t space = 0;
  sb_actor_t actor = 0;
  bool active = false;
  uint64_t last_polled_revision = 0;
  std::shared_ptr<const FrozenRichSnapshot> snapshot;
  std::shared_ptr<RichDeliveryState> delivery;
};

static std::mutex g_rich_mutex;
static std::unordered_map<sb_rich_realization_token_t, RichActivation>
    g_rich_activations;

constexpr uint32_t kRichMaxChannels = 1024;
constexpr uint32_t kRichMaxContributors = 8192;
constexpr uint32_t kRichMaxUtf8Bytes = 1024 * 1024;
constexpr size_t kRichMaxActivations = 4096;
// DSO/process lifetime nonce, not a clock kind.  random_device is the native
// platform entropy source; the nonzero fallback merely preserves fencing on a
// platform where it fails.  Clock id 1 names steady_clock, while this nonce
// changes after a process/DSO lifetime and must be calibrated again.
static uint64_t RichClockEpoch() {
  static const uint64_t epoch = [] {
    uint64_t value = 0;
    try {
      std::random_device random;
      value = (static_cast<uint64_t>(random()) << 32) ^ random();
    } catch (...) {
      // Use the nonzero lifetime fallback below if platform entropy setup
      // fails; do not make loading the native boundary throw.
    }
    if (value == 0) {
      value = static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
      if (value == 0) value = 1;
    }
    return value;
  }();
  return epoch;
}

static void RetireRichActivationsForActor(sb_space_t space, sb_actor_t actor);
static void RetireAndReleaseRichActivationsForSession(sb_session_t session);

constexpr uint32_t kSpatialBlendV2AbiVersion = 4u;
constexpr uint32_t kSpatialBlendV2AbiMinorVersion = 14u;

static SessionContext* GetSession(sb_session_t id) {
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(id);
  return it != g_sessions.end() ? it->second.get() : nullptr;
}

// The SessionContext owning `space`, or nullptr. Lock order is g_sessions_mutex
// before g_spaces_mutex (sb_destroy_session and sb_cleanup_all hold the first
// while taking the second), so this reads the session id, RELEASES
// g_spaces_mutex, and only then takes g_sessions_mutex: no path holds both in
// the other order.
static SessionContext* SessionContextOfSpace(sb_space_t space) {
  sb_session_t session = 0;
  {
    std::lock_guard<std::mutex> lock(g_spaces_mutex);
    auto sit = g_space_to_session.find(space);
    if (sit == g_space_to_session.end()) return nullptr;
    session = sit->second;
  }
  std::lock_guard<std::mutex> slock(g_sessions_mutex);
  auto it = g_sessions.find(session);
  return it == g_sessions.end() ? nullptr : it->second.get();
}

static SessionContext::SpaceInfo* GetSpace(sb_space_t id) {
  std::lock_guard<std::mutex> lock(g_spaces_mutex);
  auto it = g_spaces.find(id);
  return it != g_spaces.end() ? it->second : nullptr;
}

static RelationKernelType ToKernelType(sb_kernel_type_t type) {
  switch (type) {
    case SB_KERNEL_SPACING:
      return RelationKernelType::kAvatarSpacing;
    case SB_KERNEL_FACING:
      return RelationKernelType::kAvatarFacing;
    case SB_KERNEL_AFFORDANCE_POSE:
      return RelationKernelType::kAvatarAffordancePose;
    default:
      return RelationKernelType::kAvatarSpacing;
  }
}

static sb_kernel_type_t FromKernelType(RelationKernelType type) {
  switch (type) {
    case RelationKernelType::kAvatarSpacing:
      return SB_KERNEL_SPACING;
    case RelationKernelType::kAvatarFacing:
      return SB_KERNEL_FACING;
    case RelationKernelType::kAvatarAffordancePose:
      return SB_KERNEL_AFFORDANCE_POSE;
    default:
      return SB_KERNEL_SPACING;
  }
}

static GeometryRole ToGeometryRole(sb_geometry_role_t role) {
  switch (role) {
    case SB_GEOMETRY_NAVIGABLE:
      return GeometryRole::kNavigable;
    case SB_GEOMETRY_CONTAINMENT:
      return GeometryRole::kContainment;
    case SB_GEOMETRY_OBSTACLE:
      return GeometryRole::kObstacle;
    case SB_GEOMETRY_SURFACE:
      return GeometryRole::kSurface;
    case SB_GEOMETRY_INTERACTION_REGION:
      return GeometryRole::kInteractionRegion;
    case SB_GEOMETRY_UNSPECIFIED:
    default:
      return GeometryRole::kUnspecified;
  }
}

// ABI2 buffers are caller-owned trusted memory with the declared extent.
// Numeric/count preflight cannot validate an arbitrary pointer's allocation.
constexpr int64_t kMaxSceneVertices = 65536;
constexpr int64_t kMaxSceneFaces = 65536;
constexpr int64_t kMaxSceneFaceIndices = 1048576;
static bool ValidSceneRole(sb_geometry_role_t role) {
  return role == SB_GEOMETRY_UNSPECIFIED || role == SB_GEOMETRY_NAVIGABLE ||
         role == SB_GEOMETRY_CONTAINMENT || role == SB_GEOMETRY_OBSTACLE ||
         role == SB_GEOMETRY_SURFACE || role == SB_GEOMETRY_INTERACTION_REGION;
}
static bool FiniteSceneBuffer(const double* values, int64_t count) {
  if (!values) return false;
  for (int64_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

static int32_t DispatchRealizationTargets(sb_space_t space_handle,
                                          SessionContext::SpaceInfo* info,
                                          sb_actor_t actor) {
  if (!info || !info->space) return 0;
  sb_on_targets_t callback = nullptr;
  void* callback_user_data = nullptr;
  {
    std::lock_guard<std::mutex> lock(info->callback_mutex);
    callback = info->on_targets_callback;
    callback_user_data = info->on_targets_user_data;
  }
  if (!callback) return 0;

  const auto* result = info->space->GetRealizationTargets(
      static_cast<spatialblend::entity_id_t>(actor));
  if (!result) return 0;  // no known snapshot for this actor
  if (result->targets.empty()) {
    // An empty known snapshot is an authoritative release, not an absence of
    // information. Notify consumers so they can clear previously latched rig
    // targets; nullptr is valid whenever num_targets is zero.
    callback(space_handle, actor, nullptr, 0, callback_user_data);
    return 1;
  }

  // Keep anchor_name strings alive for the duration of the callback.
  std::vector<std::string> anchor_names;
  anchor_names.reserve(result->targets.size());
  std::vector<sb_realization_target_t> c_targets;
  c_targets.reserve(result->targets.size());

  for (const auto& target : result->targets) {
    anchor_names.push_back(target.anchor_name);
  }

  for (size_t i = 0; i < result->targets.size(); ++i) {
    const auto& target = result->targets[i];
    sb_realization_target_t c_target;
    c_target.arrival_time = target.arrival_time;
    c_target.anchor_name = anchor_names[i].c_str();
    c_target.position_x = target.position.x();
    c_target.position_y = target.position.y();
    c_target.position_z = target.z;
    c_target.has_z = target.has_z ? 1 : 0;
    c_target.yaw = target.yaw;
    c_target.kernel_type = FromKernelType(target.kernel_type);
    c_target.referent_id = static_cast<int64_t>(target.referent_id);
    c_target.weight = target.weight;
    c_targets.push_back(c_target);
  }

  callback(space_handle, actor, c_targets.data(),
           static_cast<int64_t>(c_targets.size()), callback_user_data);
  return 1;
}

// Tunneled core link (ABI 4.5); defined in tunnel_c.inc.
static void RetireTunnelsForSession(sb_session_t session);
static void RetireAllTunnels();
static int32_t TunnelAsyncTryWait(sb_handle_t handle, int32_t timeout_ms);
static const char* TunnelAsyncError(sb_handle_t handle);

// ----------------------------------------------------------------------------
// Session management
// ----------------------------------------------------------------------------

sb_session_t sb_create_session() {
  auto ctx = std::make_unique<SessionContext>();
  ctx->context = spatialblend::LocalContext::Create();
  if (!ctx->context) {
    return 0;
  }
  sb_session_t id = g_next_session_id.fetch_add(1);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  g_sessions[id] = std::move(ctx);
  return id;
}

void sb_destroy_session(sb_session_t session) {
  // Tunneled links first: a join in progress is waiting on its link, and it
  // must be over (joined) before this function takes the session registry lock
  // that the join's failure path also takes.
  RetireTunnelsForSession(session);
  {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    auto it = g_sessions.find(session);
    if (it != g_sessions.end()) it->second->rich_retiring = true;
  }
  // Rich retirement must happen while the old SpaceInfo identity and callback
  // slot still exist.  It is deliberately before this function nulls callbacks
  // and releases the LocalContext, and callbacks run outside all registry
  // locks.
  RetireAndReleaseRichActivationsForSession(session);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(session);
  if (it != g_sessions.end()) {
    // Registrations are session-local capabilities. Forget them before their
    // hub disappears so later disposal is inert rather than dereferencing a
    // stale session or recreating a mapping.
    for (auto reg_it = g_support_registrations.begin();
         reg_it != g_support_registrations.end();) {
      if (reg_it->second.session == session) {
        reg_it = g_support_registrations.erase(reg_it);
      } else {
        ++reg_it;
      }
    }
    for (auto setup_it = g_observed_source_setups.begin();
         setup_it != g_observed_source_setups.end();) {
      if (setup_it->second.session == session) {
        setup_it = g_observed_source_setups.erase(setup_it);
      } else {
        ++setup_it;
      }
    }
    // Null the session hub's callback slots FIRST, before touching any
    // remote_hub/receiver they may reference. The callbacks are invoked
    // from gRPC completion-queue threads (HubReceiver::OnUpdateActorState /
    // OnClientShareElement run on the host's server CQ thread and call into
    // Space -> hub_->Propagate()/PropagateElementEvent(), which read these
    // slots) concurrently with this teardown. SpaceHub::Propagate takes its
    // own copy of the callback under lock before invoking it, so an
    // in-flight call is unaffected by this clear — this only guarantees no
    // *new* invocation is dispatched to an about-to-be-destroyed remote_hub
    // or receiver.
    if (it->second->context) {
      it->second->context->GetHub()->ClearCallbacks();
    }
    // Null every space-level app callback slot BEFORE the network
    // teardown below. RemoteHub::PrepareDisconnect()/HubSender::Disconnect()
    // legitimately BLOCK (a bounded drain) while an in-flight
    // SubscribeStreamCall::Proceed() -- running concurrently on that
    // RemoteHub's own CQ thread -- finishes delivering an already-dequeued
    // relation/element update via Space::OnRemoteUpdate()/OnElementEvent(),
    // which in turn invokes whatever the app registered via
    // sb_register_on_relation_event/on_targets/on_element_added. That
    // registered callback may reach into caller-owned state (e.g. a test
    // harness's own per-space bookkeeping) that the CALLER is free to tear
    // down independently of when this drain completes -- observed as a
    // "mutex lock failed: Invalid argument" abort when a harness's own
    // callback fired into already-destroyed harness state mid-drain. Nulling
    // these slots first (SpaceInfo itself is still alive -- it's only erased
    // from the session below, well after this loop) turns any such
    // still-in-flight dispatch into a safe no-op regardless of what order the
    // CALLER's own local state gets destroyed in. Requires
    // sb_register_on_relation_event's lambda to re-read the callback through
    // `info` rather than capturing it by value (see that function).
    for (auto& [space_id, info] : it->second->spaces) {
      std::lock_guard<std::mutex> callback_lock(info->callback_mutex);
      info->on_targets_callback = nullptr;
      info->on_rich_callback = nullptr;
      info->on_rich_user_data = nullptr;
      info->on_element_added_callback = nullptr;
      info->on_element_removed_callback = nullptr;
      info->on_link_removed_callback = nullptr;
      info->on_relation_event_callback = nullptr;
    }
    // Stop network connections before destroying spaces. PrepareDisconnect
    // cancels the Subscribe stream, drains its completion queue and joins
    // the CQ thread (see remote_hub.cc / hub_sender_call.cc) before this
    // returns, so by the time remote_hub/receiver are reset here no
    // networking thread can still be touching this session's spaces.
    for (auto& [space_id, info] : it->second->spaces) {
      if (info->remote_hub) {
        (void)info->remote_hub->PrepareDisconnect();
        info->remote_hub.reset();
      }
      if (info->receiver) {
        // Call Stop() explicitly HERE, while this SessionContext's own
        // shared_ptr<HubReceiver> still holds a live strong reference,
        // rather than relying solely on plain .reset() to trigger
        // ~HubReceiver() (which calls Stop() itself). A HubReceiver call
        // object's Proceed() (e.g. DisconnectFromHubReceiverCall, processing
        // the peer's disconnect notification that a preceding
        // PrepareDisconnect() above just sent) obtains its own *temporary*
        // shared_ptr via call_ctx_->receiver.lock() -- if THIS reset() below
        // happened to be the one racing to drop the refcount to zero
        // concurrently with that lock(), the CQ thread's temporary could
        // instead end up holding the LAST reference, so ~HubReceiver()
        // (hence Stop(), hence GrpcServerBase::Stop()'s
        // `thread.join()` on server_threads_) would run ON that very CQ
        // thread -- joining itself
        // ("std::system_error: thread::join failed: Resource deadlock
        // avoided") or hitting torn-down state mid-callback ("mutex lock
        // failed: Invalid argument"), both observed via harness_test's
        // gRPC-two-session transport. Calling Stop() explicitly here keeps
        // this shared_ptr's own reference alive for the ENTIRE duration of
        // the shutdown/join sequence below (blocking THIS thread, not the CQ
        // thread, until HandleRpcs() has fully drained and returned), so
        // Stop()'s internal join always runs on this thread. Stop() is
        // idempotent (guarded by shutdown_.exchange()), so the later
        // ~HubReceiver() triggered by .reset() (whichever thread that ends
        // up being, if any reference is still outstanding) is then a safe
        // no-op. See RemoteHub::~RemoteHub() (remote_hub.cc) for the
        // sibling fix on the sender side.
        info->receiver->Stop();
        info->receiver.reset();
      }
    }
    // Remove all space entries from the global map.
    std::lock_guard<std::mutex> slock(g_spaces_mutex);
    for (auto& [space_id, info] : it->second->spaces) {
      g_spaces.erase(space_id);
      g_space_to_session.erase(space_id);
    }
    g_sessions.erase(it);
  }
}

// ----------------------------------------------------------------------------
// Space management
// ----------------------------------------------------------------------------

sb_space_t sb_add_space(sb_session_t session) {
  auto* ctx = GetSession(session);
  if (!ctx) {
    return 0;
  }
  auto* space = ctx->context->AddSpace();
  if (!space) {
    return 0;
  }
  // Connect the space to its session hub so ShareElement() and
  // UpdateActorState() actually propagate cross-space. Without this the space's
  // hub_ is null and both silently no-op (no element/actor-state propagation).
  space->SetHub(ctx->context->GetHub());
  sb_space_t id = g_next_space_id.fetch_add(1);
  // LocalContext assigns space_id from a per-context counter starting at 1, so
  // spaces created in different sessions collide (both get id=1). That breaks
  // host/client routing — PropagateElementEvent skips spaces whose id ==
  // source_space_id, so a client sharing the host's id is treated as the source
  // and never receives events. Override with the globally-unique handle.
  space->SetSpaceId(static_cast<spatialblend::space_id_t>(id));
  auto info = std::make_unique<SessionContext::SpaceInfo>();
  info->space = space;
  auto* info_ptr = info.get();
  // Forward peer-shared elements (received via the hub) to the C callback
  // registered with sb_register_on_element_added(). Read the callback pointer
  // at invocation time so registration can happen after sb_add_space().
  space->SetOnElementAdded(
      [info_ptr, id](spatialblend::element_id_t local_id,
                     spatialblend::shared_element_id_t shared_id,
                     const std::string& tag, const std::string& user_data) {
        sb_on_element_added_t callback = nullptr;
        void* callback_user_data = nullptr;
        {
          std::lock_guard<std::mutex> lock(info_ptr->callback_mutex);
          callback = info_ptr->on_element_added_callback;
          callback_user_data = info_ptr->on_element_added_user_data;
        }
        if (callback) {
          callback(id, static_cast<sb_element_t>(local_id),
                   static_cast<int64_t>(shared_id), tag.c_str(),
                   user_data.empty() ? nullptr : user_data.data(),
                   static_cast<int64_t>(user_data.size()), callback_user_data);
        }
      });
  space->SetOnElementRemoved(
      [info_ptr, id](spatialblend::element_id_t local_id) {
        sb_on_element_removed_t callback = nullptr;
        void* callback_user_data = nullptr;
        {
          std::lock_guard<std::mutex> lock(info_ptr->callback_mutex);
          callback = info_ptr->on_element_removed_callback;
          callback_user_data = info_ptr->on_element_removed_user_data;
        }
        if (callback) {
          callback(id, static_cast<sb_element_t>(local_id), callback_user_data);
        }
      });
  space->SetOnLinkRemoved(
      [info_ptr, id](spatialblend::element_id_t local_id,
                     spatialblend::space_id_t peer_space,
                     spatialblend::Space::LinkRemovedReason reason) {
        sb_on_link_removed_t callback = nullptr;
        void* callback_user_data = nullptr;
        {
          std::lock_guard<std::mutex> lock(info_ptr->callback_mutex);
          callback = info_ptr->on_link_removed_callback;
          callback_user_data = info_ptr->on_link_removed_user_data;
        }
        if (callback) {
          callback(id, static_cast<sb_element_t>(local_id),
                   static_cast<int64_t>(peer_space),
                   reason == spatialblend::Space::LinkRemovedReason::kPeerUnshared
                       ? SB_LINK_REMOVED_PEER_UNSHARED
                       : SB_LINK_REMOVED_PEER_UNLINKED,
                   callback_user_data);
        }
      });
  ctx->spaces[id] = std::move(info);
  {
    std::lock_guard<std::mutex> lock(g_spaces_mutex);
    g_spaces[id] = info_ptr;
    g_space_to_session[id] = session;
  }
  return id;
}

// `tunneled` hosts with no listening socket (ABI 4.5): peers attach through
// sb_accept_tunnel and `port` is ignored.
static sb_handle_t HostSpaceImpl(sb_space_t space, int32_t port, bool tunneled) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;

  // Get the session's SpaceHub for the HubReceiver.
  SessionContext* ctx = SessionContextOfSpace(space);
  if (!ctx || !ctx->context) return 0;

  // One session = one hub role. Claim the "host" role before doing any
  // work; a session that already hosted or connected refuses a second
  // host/connect call instead of clobbering the hub's single-slot
  // callbacks (see the HubRole comment above SessionContext).
  {
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    if (ctx->hub_role != HubRole::kNone) return 0;
    ctx->hub_role = HubRole::kHost;
  }

  info->receiver = std::make_shared<spatialblend::HubReceiver>(
      info->space, ctx->context->GetHub());
  // A joining peer's id comes from the same counter as every sb_space_t handle
  // in this process, so it can never equal a space created here later.
  info->receiver->SetSpaceIdAllocator([]() {
    return static_cast<spatialblend::space_id_t>(g_next_space_id.fetch_add(1));
  });
  auto status = tunneled ? info->receiver->StartTunneled()
                         : info->receiver->Start(port);
  if (!status.ok()) {
    info->receiver.reset();
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    ctx->hub_role = HubRole::kNone;
    return 0;
  }

  auto receiver = info->receiver;
  ctx->context->GetHub()->SetRelationUpdateCallback(
      [receiver](const spatialblend::RemoteRelationUpdate& update) {
        if (receiver) receiver->PropagateToRemoteSpaces(update);
      });
  ctx->context->GetHub()->SetElementEventCallback(
      [receiver](const spatialblend::ElementEvent& event,
                 spatialblend::space_id_t source_space_id) {
        if (receiver) {
          receiver->PropagateElementEventToRemoteSpaces(event, source_space_id);
        }
      });

  return kSyncDoneHandle;
}

sb_handle_t sb_host_space(sb_space_t space, int32_t port) {
  return HostSpaceImpl(space, port, /*tunneled=*/false);
}

// Forwards this client's element events and avatar movement up to the host
// (which applies and relays them). Shared by the TCP and the tunneled join.
static void WireClientHub(SessionContext* ctx,
                          std::shared_ptr<spatialblend::RemoteHub> hub) {
  if (!ctx || !ctx->context || !hub) return;
  // The callbacks own the hub: tearing the join down (sb_tunnel_close,
  // sb_destroy_session) clears the callbacks, and a call already running keeps
  // the hub alive instead of reading info->remote_hub while it is reset.
  ctx->context->GetHub()->SetElementEventCallback(
      [hub](const spatialblend::ElementEvent& event,
            spatialblend::space_id_t /*source_space_id*/) {
        if (hub->IsConnected()) hub->SendElementEvent(event);
      });
  // Wire the session hub's relation-update callback so that this client's
  // avatar movement is forwarded up to the host (which applies + relays it),
  // driving the client's avatar clone on the host display.
  ctx->context->GetHub()->SetRelationUpdateCallback(
      [hub](const spatialblend::RemoteRelationUpdate& update) {
        if (hub->IsConnected()) hub->SendRelationUpdate(update);
      });
}

sb_handle_t sb_connect_space(sb_space_t space, const char* host, int32_t port) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !host) return 0;

  // One session = one hub role. Claim the "client" role before doing
  // any network work; a session that already hosted or connected refuses a
  // second host/connect call instead of clobbering the hub's single-slot
  // callbacks (see the HubRole comment above SessionContext). Resolve the
  // session up front (rather than after RemoteHub::Create, as before) so we
  // never open a connection only to discard it because the role was
  // unavailable.
  SessionContext* ctx = SessionContextOfSpace(space);
  if (!ctx || !ctx->context) return 0;
  {
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    if (ctx->hub_role != HubRole::kNone) return 0;
    ctx->hub_role = HubRole::kClient;
  }

  info->remote_hub = spatialblend::RemoteHub::Create(info->space, host, port,
                                                     info->space->GetSpaceId());
  if (!info->remote_hub || !info->remote_hub->IsConnected()) {
    info->remote_hub.reset();
    std::lock_guard<std::mutex> slock(g_sessions_mutex);
    ctx->hub_role = HubRole::kNone;
    return 0;
  }

  WireClientHub(ctx, info->remote_hub);

  return kSyncDoneHandle;
}

#include "spatialblend/api/c/tunnel_c.inc"

// ----------------------------------------------------------------------------
// Async handle utilities
// ----------------------------------------------------------------------------

int32_t sb_try_wait(sb_handle_t handle, int32_t timeout_ms) {
  // Sync-done sentinel is always complete.
  if (handle == kSyncDoneHandle) return 1;
  return TunnelAsyncTryWait(handle, timeout_ms);
}

const char* sb_get_error(sb_handle_t handle) { return TunnelAsyncError(handle); }

// ----------------------------------------------------------------------------
// Element management
// ----------------------------------------------------------------------------

sb_element_t sb_add_element(sb_space_t space, sb_element_t parent) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return 0;
  }
  // A joiner's bootstrap copies the host's element ids verbatim (HubSender::ApplyBootstrap), so the next id of this
  // space may already be taken: never hand out an id in use (the Space's own received elements skip them the same way).
  sb_element_t elem_id = info->next_element_id.fetch_add(1);
  while (info->space->HasElement(elem_id)) elem_id = info->next_element_id.fetch_add(1);
  if (!info->space->AddElement(
          elem_id, parent == 0 ? spatialblend::kInvalidElementId : parent)) {
    return 0;
  }
  return elem_id;
}

namespace {
// The status of a scene mutator that returned `ok`: the Space records the
// containment-layout violation (if any) it rejected the change for.
sb_scene_status_t SceneStatusOf(spatialblend::Space* s, bool ok) {
  if (ok) return SB_SCENE_OK;
  const auto err = s->LastSceneError();
  return err.code == spatialblend::SceneErrorCode::kOk
             ? SB_SCENE_INVALID_ARGUMENT
             : static_cast<sb_scene_status_t>(err.code);
}
}  // namespace

sb_scene_status_t sb_get_scene_status(sb_space_t space, char* message,
                                      int32_t capacity) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return SB_SCENE_INVALID_ARGUMENT;
  const auto err = info->space->LastSceneError();
  if (message && capacity > 0) {
    std::snprintf(message, static_cast<size_t>(capacity), "%s",
                  err.message.c_str());
  }
  return static_cast<sb_scene_status_t>(err.code);
}

sb_scene_status_t sb_set_element_transform(sb_space_t space, sb_element_t elem,
                                           const double* matrix_4x4) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !matrix_4x4) {
    return SB_SCENE_INVALID_ARGUMENT;
  }
  // Row-major 4x4 input -> Eigen::Matrix4d (column-major by default).
  Eigen::Matrix4d mat;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mat(r, c) = matrix_4x4[r * 4 + c];
    }
  }
  return SceneStatusOf(info->space,
                       info->space->SetElementTransform(elem, mat));
}

void sb_set_element_shared_id(sb_space_t space, sb_element_t elem,
                              int64_t shared_id) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  info->space->SetElementSharedId(elem, shared_id);
}

void sb_set_element_tag(sb_space_t space, sb_element_t elem, const char* tag) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !tag) {
    return;
  }
  info->space->SetElementTag(elem, tag);
}

int32_t sb_set_element_affordance_pose(sb_space_t space, sb_element_t elem, const char* subject_anchor,
                                       int32_t support_kind, const char* part) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  if (!subject_anchor) return info->space->SetElementAffordancePose(elem, nullptr) ? 0 : -1;
  if (support_kind != 0 && support_kind != 1) return -1;
  spatialblend::AffordancePoseTemplate t;
  t.subject_anchor = subject_anchor;
  t.support = support_kind == 0 ? spatialblend::AffordancePoseTemplate::Support::kPart
                                : spatialblend::AffordancePoseTemplate::Support::kActorRoot;
  if (support_kind == 0) {
    if (!part || !*part) return -1;
    t.part = part;
  }
  return info->space->SetElementAffordancePose(elem, &t) ? 0 : -1;
}

sb_scene_status_t sb_set_element_polygon(sb_space_t space, sb_element_t elem,
                                         sb_geometry_role_t role,
                                         const double* points_xy,
                                         int64_t num_points) {
  auto* info = GetSpace(space);
  if (!info || !info->space || num_points < 3 ||
      num_points > kMaxSceneVertices || !ValidSceneRole(role) ||
      !FiniteSceneBuffer(points_xy, num_points * 2)) {
    return SB_SCENE_INVALID_ARGUMENT;
  }
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(num_points);
  for (int64_t i = 0; i < num_points; ++i) {
    pts.emplace_back(points_xy[i * 2], points_xy[i * 2 + 1]);
  }
  auto polygon = std::make_shared<spatialblend::Polygon2D>(pts);
  auto definition = std::make_shared<spatialblend::NavigableSpatialDefinition>(
      polygon, nullptr, 0.0, ToGeometryRole(role));
  // A NAVIGABLE polygon is world-frame (free space, obstacle sync; for the root
  // room that IS its local frame); every other role is in the element's frame.
  if (role != SB_GEOMETRY_NAVIGABLE) definition->SetElementLocal(true);
  return SceneStatusOf(
      info->space, info->space->SetElementSpatialDefinition(elem, definition));
}

sb_scene_status_t sb_set_element_box(sb_space_t space, sb_element_t elem,
                                     sb_geometry_role_t role, double min_x,
                                     double min_y, double min_z, double max_x,
                                     double max_y, double max_z) {
  auto* info = GetSpace(space);
  const double bounds[] = {min_x, min_y, min_z, max_x, max_y, max_z};
  if (!info || !info->space || !ValidSceneRole(role) ||
      !FiniteSceneBuffer(bounds, 6) || min_x > max_x || min_y > max_y ||
      min_z > max_z) {
    return SB_SCENE_INVALID_ARGUMENT;
  }
  return SceneStatusOf(
      info->space,
      info->space->SetElementSpatialDefinition(
          elem, std::make_shared<spatialblend::PolytopeSpatialDefinition>(
                    Eigen::Vector3d(min_x, min_y, min_z),
                    Eigen::Vector3d(max_x, max_y, max_z), ToGeometryRole(role))));
}

sb_scene_status_t sb_set_element_collider(sb_space_t space, sb_element_t elem,
                                          sb_collider_type_t type,
                                          const double* params,
                                          int64_t num_params) {
  auto* info = GetSpace(space);
  const int64_t expected = type == SB_COLLIDER_SPHERE ? 4 : 6;
  if (!info || !info->space ||
      (type != SB_COLLIDER_BOX && type != SB_COLLIDER_SPHERE &&
       type != SB_COLLIDER_CAPSULE) || num_params != expected ||
      !FiniteSceneBuffer(params, num_params) ||
      (type == SB_COLLIDER_CAPSULE && params[5] != 0 && params[5] != 1 &&
       params[5] != 2)) {
    return SB_SCENE_INVALID_ARGUMENT;
  }
  // Auto-derive only: never clobber an explicit boundary. An affordance/polygon
  // set via sb_set_element_polygon (before or after) takes precedence over the
  // envelope.
  spatialblend::ColliderType core_type;
  switch (type) {
    case SB_COLLIDER_SPHERE:
      core_type = spatialblend::ColliderType::kSphere;
      break;
    case SB_COLLIDER_CAPSULE:
      core_type = spatialblend::ColliderType::kCapsule;
      break;
    case SB_COLLIDER_BOX:
    default:
      core_type = spatialblend::ColliderType::kBox;
      break;
  }
  std::vector<double> p(params, params + num_params);
  auto envelope = spatialblend::MakeEnvelope(core_type, p);
  if (!envelope) return SB_SCENE_INVALID_ARGUMENT;
  // The params are in the element's frame, so the footprint is element-local.
  if (auto nav = std::dynamic_pointer_cast<spatialblend::NavigableSpatialDefinition>(
          envelope))
    nav->SetElementLocal(true);
  return SceneStatusOf(
      info->space, info->space->SetElementSpatialDefinition(elem, envelope, true));
}

sb_scene_status_t sb_set_element_surface(sb_space_t space, sb_element_t elem,
                                         const double* points_xyz,
                                         int32_t num_points,
                                         sb_geometry_role_t role) {
  auto* info = GetSpace(space);
  if (!info || !info->space || num_points < 3 ||
      num_points > kMaxSceneVertices || !ValidSceneRole(role) ||
      !FiniteSceneBuffer(points_xyz, static_cast<int64_t>(num_points) * 3)) {
    return SB_SCENE_INVALID_ARGUMENT;
  }
  std::vector<Eigen::Vector3d> vertices;
  vertices.reserve(num_points);
  for (int32_t i = 0; i < num_points; ++i) {
    vertices.emplace_back(points_xyz[i * 3], points_xyz[i * 3 + 1],
                          points_xyz[i * 3 + 2]);
  }
  auto polygon = std::make_shared<spatialblend::Polygon3D>(vertices);
  auto def = std::make_shared<spatialblend::SurfaceSpatialDefinition>(polygon);
  def->SetGeometryRole(ToGeometryRole(role));
  return SceneStatusOf(info->space,
                       info->space->SetElementSpatialDefinition(elem, def));
}

void sb_set_element_polytope(sb_space_t space, sb_element_t elem,
                             const double* vertices_xyz, int32_t num_vertices,
                             const int32_t* face_indices,
                             const int32_t* face_sizes, int32_t num_faces,
                             sb_geometry_role_t role) {
  auto* info = GetSpace(space);
  if (!info || !info->space || num_vertices < 4 ||
      num_vertices > kMaxSceneVertices || !ValidSceneRole(role) ||
      !FiniteSceneBuffer(vertices_xyz, static_cast<int64_t>(num_vertices) * 3) ||
      !face_indices || !face_sizes || num_faces < 1 ||
      num_faces > kMaxSceneFaces) {
    return;
  }
  int64_t index_count = 0;
  for (int32_t f = 0; f < num_faces; ++f) {
    if (face_sizes[f] < 3 || face_sizes[f] > num_vertices) return;
    index_count += face_sizes[f];
    if (index_count > kMaxSceneFaceIndices) return;
  }
  for (int64_t i = 0; i < index_count; ++i)
    if (face_indices[i] < 0 || face_indices[i] >= num_vertices) return;
  std::vector<Eigen::Vector3d> vertices;
  vertices.reserve(num_vertices);
  for (int32_t i = 0; i < num_vertices; ++i) {
    vertices.emplace_back(vertices_xyz[i * 3], vertices_xyz[i * 3 + 1],
                          vertices_xyz[i * 3 + 2]);
  }
  std::vector<std::vector<int>> faces;
  faces.reserve(num_faces);
  int32_t cursor = 0;
  for (int32_t f = 0; f < num_faces; ++f) {
    std::vector<int> face;
    face.reserve(face_sizes[f]);
    for (int32_t k = 0; k < face_sizes[f]; ++k) {
      face.push_back(face_indices[cursor++]);
    }
    faces.push_back(std::move(face));
  }
  info->space->SetElementSpatialDefinition(elem,
      std::make_shared<spatialblend::PolytopeSpatialDefinition>(
          std::move(vertices), std::move(faces), ToGeometryRole(role)));
}

void sb_add_sub_region(sb_space_t space, sb_element_t elem, const char* name,
                       const char* tag, const double* points_xy,
                       int32_t num_points, const double* transform16) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !name || num_points < 3 ||
      num_points > kMaxSceneVertices ||
      !FiniteSceneBuffer(points_xy, static_cast<int64_t>(num_points) * 2)) {
    return;
  }
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(num_points);
  for (int32_t i = 0; i < num_points; ++i) {
    pts.emplace_back(points_xy[i * 2], points_xy[i * 2 + 1]);
  }
  spatialblend::SubRegion region;
  region.name = name;
  region.tag = tag ? tag : "";
  region.boundary = std::make_shared<spatialblend::Polygon2D>(pts);
  if (transform16) {
    // Row-major 4x4 → Eigen (column-major storage).
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        region.local_transform(r, c) = transform16[r * 4 + c];
      }
    }
  }
  info->space->AddElementSubRegion(elem, region);
}

int32_t sb_set_sub_region_chart(sb_space_t space, sb_element_t elem, const char* part_name, int32_t chart) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !part_name) return -1;
  auto kind = spatialblend::SurfaceChartKindFromInt(chart);
  if (!kind.ok()) return -1;
  switch (info->space->SetElementSubRegionChart(elem, part_name, *kind)) {
    case spatialblend::SurfaceChartStatus::kOk: return 0;
    case spatialblend::SurfaceChartStatus::kNotARectangle: return -2;
    default: return -1;
  }
}

int32_t sb_part_chart_encode(sb_space_t space, sb_element_t elem, const char* part_name, const double* world_xyz,
                             sb_part_chart_coord_t* out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !part_name || !world_xyz || !out || out->struct_size != sizeof(sb_part_chart_coord_t))
    return -1;
  spatialblend::Space::PartFrame frame;
  if (!info->space->GetElementPartFrame(elem, part_name, &frame)) return -1;
  const Eigen::Vector4d world(world_xyz[0], world_xyz[1], world_xyz[2], 1.0);
  const Eigen::Matrix4d to_world = frame.part_to_world;
  if (!world.allFinite() || !std::isfinite(std::abs(to_world.determinant())) || std::abs(to_world.determinant()) < 1e-12)
    return -1;
  const Eigen::Vector4d local = to_world.inverse() * world;
  if (!local.allFinite()) return -1;
  const Eigen::Vector2d xy(local.x(), local.y());
  sb_part_chart_coord_t c{};
  c.struct_size = sizeof(sb_part_chart_coord_t);
  c.chart = static_cast<int32_t>(frame.chart);
  c.normal_m = local.z();
  if (spatialblend::IsFrameChart(frame.chart)) {
    auto f = spatialblend::PlanarFrameChart::Encode(xy, local.z(), *frame.boundary);
    if (!f.ok()) return -2;
    c.u = f->u;
    c.v = f->v;
    c.src_width_m = f->src_width_m;
    c.src_height_m = f->src_height_m;
  } else {
    auto r = spatialblend::RadialSurfaceChart::Encode(xy, *frame.boundary, Eigen::Vector2d(0, 1));
    if (!r.ok()) return -2;
    c.angle = r->angle;
    c.radial_ratio = r->radial_ratio;
  }
  *out = c;
  return 0;
}

int32_t sb_part_chart_decode(sb_space_t space, sb_element_t elem, const char* part_name,
                             const sb_part_chart_coord_t* in, double* out_world_xyz) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !part_name || !in || !out_world_xyz || in->struct_size != sizeof(sb_part_chart_coord_t))
    return -1;
  spatialblend::Space::PartFrame frame;
  if (!info->space->GetElementPartFrame(elem, part_name, &frame)) return -1;
  auto kind = spatialblend::SurfaceChartKindFromInt(in->chart);
  if (!kind.ok() || *kind != frame.chart) return -2;
  if (!std::isfinite(in->normal_m)) return -3;
  Eigen::Vector3d local;
  if (spatialblend::IsFrameChart(*kind)) {
    if (!std::isfinite(in->u) || !std::isfinite(in->v) || !(in->src_width_m > 0.0) || !(in->src_height_m > 0.0) ||
        !std::isfinite(in->src_width_m) || !std::isfinite(in->src_height_m))
      return -3;
    spatialblend::FrameCoord f;
    f.u = in->u;
    f.v = in->v;
    f.normal_m = in->normal_m;
    f.src_width_m = in->src_width_m;
    f.src_height_m = in->src_height_m;
    auto d = spatialblend::PlanarFrameChart::Decode(f, *frame.boundary, *kind);
    if (!d.ok()) return -2;
    local = *d;
  } else {
    if (!std::isfinite(in->angle) || !std::isfinite(in->radial_ratio) || in->radial_ratio < 0.0 || in->radial_ratio > 1.0)
      return -3;
    spatialblend::NormalizedPolygonPosition n;
    n.strategy = spatialblend::NormalizedPolygonPosition::Strategy::kRadial;
    n.angle = in->angle;
    n.radial_ratio = in->radial_ratio;
    auto d = spatialblend::RadialSurfaceChart::Decode(n, *frame.boundary, Eigen::Vector2d(0, 1));
    if (!d.ok()) return -2;
    local = Eigen::Vector3d(d->x(), d->y(), in->normal_m);
  }
  const Eigen::Vector3d world = (frame.part_to_world * local.homogeneous()).head<3>();
  if (!world.allFinite()) return -2;
  out_world_xyz[0] = world.x();
  out_world_xyz[1] = world.y();
  out_world_xyz[2] = world.z();
  return 0;
}

sb_element_t sb_find_element_by_shared_id(sb_space_t space, int64_t shared_id) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return 0;
  }
  const auto id = info->space->FindElementBySharedId(shared_id);
  return id == spatialblend::kInvalidElementId ? 0 : id;
}

void sb_add_element_anchor(sb_space_t space, sb_element_t elem,
                           const char* name, const double* matrix_4x4) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !name || !matrix_4x4) {
    return;
  }
  // Input is col-major Eigen layout: map directly.
  Eigen::Matrix4d mat;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      mat(r, c) = matrix_4x4[c * 4 + r];
    }
  }
  info->space->AddElementAnchor(elem, name, mat);
}

void sb_share_element(sb_space_t space, sb_element_t elem,
                      const void* user_data, int64_t user_data_size) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  std::string payload;
  if (user_data && user_data_size > 0) {
    payload.assign(static_cast<const char*>(user_data),
                   static_cast<size_t>(user_data_size));
  }
  info->space->ShareElement(static_cast<spatialblend::element_id_t>(elem),
                            payload);
}

void sb_unshare_element(sb_space_t space, sb_element_t elem) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  info->space->UnshareElement(static_cast<spatialblend::element_id_t>(elem));
}

int32_t sb_unlink(sb_space_t space, sb_element_t elem) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  return info->space->UnlinkElement(static_cast<spatialblend::element_id_t>(elem));
}

sb_support_correspondence_registration_t sb_register_support_correspondences(
    sb_session_t session, const sb_support_correspondence_t* correspondences,
    int64_t count) {
  if (!correspondences || count <= 0 || g_inside_relation_callback) return 0;
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto session_it = g_sessions.find(session);
  if (session_it == g_sessions.end() || !session_it->second->context) return 0;
  SessionContext* context = session_it->second.get();
  std::vector<
      std::pair<spatialblend::element_handle_t, spatialblend::element_handle_t>>
      links;
  links.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const auto& correspondence = correspondences[i];
    if (correspondence.source_element == 0 ||
        correspondence.target_element == 0) {
      return 0;
    }
    auto source_it = context->spaces.find(correspondence.source_space);
    auto target_it = context->spaces.find(correspondence.target_space);
    if (source_it == context->spaces.end() ||
        target_it == context->spaces.end() || !source_it->second->space ||
        !target_it->second->space ||
        !source_it->second->space->TryHasElementOutsideCallback(
            correspondence.source_element) ||
        !target_it->second->space->TryHasElementOutsideCallback(
            correspondence.target_element)) {
      return 0;
    }
    links.push_back(
        {{correspondence.source_space, correspondence.source_element},
         {correspondence.target_space, correspondence.target_element}});
  }
  // An exact pair already leased by another registration is rejected rather
  // than returned as a misleading borrowed lease.
  for (const auto& [id, existing] : g_support_registrations) {
    for (const auto& owned : existing.installed) {
      for (const auto& requested : links) {
        if (owned.source == requested.first &&
            owned.target == requested.second) {
          return 0;
        }
      }
    }
  }
  const auto id = ReserveSupportRegistrationId();
  if (!id) return 0;
  SupportCorrespondenceRegistration registration{session, {}};
  if (!context->context->GetHub()->GetLinker().InstallLinksIfUnconflicted(
          links, &registration.installed)) {
    return 0;
  }
  g_support_registrations.emplace(id, std::move(registration));
  return id;
}

void sb_dispose_support_correspondences(
    sb_support_correspondence_registration_t registration) {
  if (registration == 0) return;
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto registration_it = g_support_registrations.find(registration);
  if (registration_it == g_support_registrations.end()) return;
  auto session_it = g_sessions.find(registration_it->second.session);
  if (session_it != g_sessions.end() && session_it->second->context) {
    auto& linker = session_it->second->context->GetHub()->GetLinker();
    for (const auto& link : registration_it->second.installed) {
      linker.RemoveOwnedLink(link);
    }
  }
  g_support_registrations.erase(registration_it);
}

#ifdef SPATIALBLEND_V2_C_TESTING
sb_session_t sb_test_create_session_with_receiver_input_clock(int64_t now_ns) {
  auto clock = std::make_shared<std::atomic<int64_t>>(now_ns);
  auto ctx = std::make_unique<SessionContext>();
  ctx->context = spatialblend::CDeterministicSessionFactory::Create(
      [clock] { return clock->load(std::memory_order_relaxed); });
  if (!ctx->context) return 0;
  ctx->test_receiver_input_clock = std::move(clock);
  const sb_session_t id = g_next_session_id.fetch_add(1);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  g_sessions[id] = std::move(ctx);
  return id;
}

int32_t sb_test_set_receiver_input_clock(sb_session_t session, int64_t now_ns) {
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  const auto found = g_sessions.find(session);
  if (found == g_sessions.end() || !found->second->test_receiver_input_clock)
    return 0;
  found->second->test_receiver_input_clock->store(now_ns,
                                                    std::memory_order_relaxed);
  return 1;
}

int32_t sb_test_wait_for_scope_extension(sb_space_t space) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  return info->space->WaitForScopeExtension() ? 1 : 0;
}
#endif

#include "spatialblend/api/c/owned_correspondence_c.inc"

int32_t sb_remove_local_element(sb_space_t space, sb_element_t elem) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  return info->space->RemoveLocalElementSubtree(
             static_cast<spatialblend::element_id_t>(elem))
             ? 1
             : 0;
}

sb_scene_status_t sb_reparent_element(sb_space_t space, sb_element_t elem,
                                      sb_element_t new_parent,
                                      const double* matrix_4x4) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return SB_SCENE_INVALID_ARGUMENT;
  Eigen::Matrix4d mat;
  if (matrix_4x4) {
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) mat(r, c) = matrix_4x4[r * 4 + c];
  }
  return SceneStatusOf(
      info->space,
      info->space->ReparentElement(static_cast<spatialblend::element_id_t>(elem),
                                   static_cast<spatialblend::element_id_t>(new_parent),
                                   matrix_4x4 ? &mat : nullptr));
}

void sb_resolve_shared_element(sb_space_t space, sb_element_t received,
                               sb_element_t resolved) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  info->space->ResolveSharedElement(
      static_cast<spatialblend::element_id_t>(received),
      static_cast<spatialblend::element_id_t>(resolved));
}

// ----------------------------------------------------------------------------
// Actor management
// ----------------------------------------------------------------------------

sb_actor_t sb_create_actor(sb_space_t space) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return 0;
  }
  std::vector<spatialblend::SubjectAnchor> anchors;
  return info->space->CreateActor(anchors);
}

static bool IsValidKernelType(sb_kernel_type_t type) {
  return type == SB_KERNEL_SPACING || type == SB_KERNEL_FACING ||
         type == SB_KERNEL_AFFORDANCE_POSE;
}

int32_t sb_destroy_actor(sb_space_t space, sb_actor_t actor) {
  auto* info = GetSpace(space);
  if (!info || !info->space || actor == 0) return 0;
  // Preserve the old actor/space identity in a terminal rich snapshot before
  // Core erases its actor state.
  RetireRichActivationsForActor(space, actor);
  return info->space->DestroyActor(static_cast<spatialblend::entity_id_t>(actor))
             ? 1
             : 0;
}

void sb_add_anchor(sb_space_t space, sb_actor_t actor, const char* name,
                   double x, double y, double z, const char* parent) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !name) {
    return;
  }
  info->space->AddActorAnchors(actor, {
      spatialblend::SubjectAnchor(name, Eigen::Vector3d(x, y, z),
                                  {spatialblend::StateType::kPosition2D,
                                   spatialblend::StateType::kRotationYaw},
                                  parent ? parent : "")});
}

int32_t sb_add_anchor_with_position_dimension(
    sb_space_t space, sb_actor_t actor, const char* name, double x, double y,
    double z, const char* parent, int32_t position_dimension) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !name || name[0] == '\0' || !std::isfinite(x) ||
      !std::isfinite(y) || !std::isfinite(z) ||
      (position_dimension != 2 && position_dimension != 3)) {
    return 0;
  }
  const auto position = position_dimension == 3
                            ? spatialblend::StateType::kPosition3D
                            : spatialblend::StateType::kPosition2D;
  return info->space->AddActorAnchors(actor, {spatialblend::SubjectAnchor(
      name, Eigen::Vector3d(x, y, z),
      {position, spatialblend::StateType::kRotationYaw}, parent ? parent : "")},
      true) ? 1 : 0;
}

static uint32_t AppendRichUtf8(const std::string& value,
                               std::vector<uint8_t>* bytes) {
  const uint32_t offset = static_cast<uint32_t>(bytes->size());
  bytes->insert(bytes->end(), value.begin(), value.end());
  return offset;
}

static std::shared_ptr<const FrozenRichSnapshot> FreezeRichSnapshot(
    const spatialblend::ActorRealizationSnapshot& source,
    sb_rich_realization_token_t lifetime, sb_space_t receiver_space,
    sb_actor_t receiver_actor) {
  size_t contributor_count = 0;
  for (const auto& channel : source.channels)
    contributor_count += channel.contributors.size();
  auto frozen = std::make_shared<FrozenRichSnapshot>();
  frozen->receiver_input = source.receiver_input;
  auto& h = frozen->header;
  h.producer_lifetime = lifetime;
  h.revision = source.revision;
  h.monotonic_clock_id = SB_RICH_REALIZATION_CLOCK_ID;
  h.monotonic_clock_epoch = RichClockEpoch();
  h.receiver_ingress_monotonic_ns =
      static_cast<uint64_t>(source.receiver_ingress_monotonic_ns);
  h.receiver_solve_monotonic_ns =
      static_cast<uint64_t>(source.receiver_solve_monotonic_ns);
  h.source_epoch = source.source_epoch;
  h.source_sequence = source.source_sequence;
  h.source_observation_sequence = source.source_observation_sequence;
  h.source_actor_space = static_cast<int64_t>(source.source_space_id);
  h.source_actor_element = static_cast<int64_t>(source.source_actor_element_id);
  h.relay_source_space = static_cast<int64_t>(source.relay_source_space_id);
  h.receiver_space = receiver_space;
  h.receiver_actor = receiver_actor;
  h.source_time_seconds = source.source_time_seconds;
  h.source_clock_seconds = source.source_clock_seconds;
  h.source_time_known = source.source_time_known ? 1u : 0u;
  h.source_clock_known = source.source_clock_known ? 1u : 0u;
  h.source_observation_sequence_known =
      source.source_observation_sequence_known ? 1u : 0u;
  h.source_timing_mode =
      source.source_timing_mode == spatialblend::ActorRealizationSnapshot::
                                       SourceTimingMode::kTimestamped
          ? SB_RICH_SOURCE_TIMESTAMPED_OBSERVATION
      : source.source_timing_mode ==
              spatialblend::ActorRealizationSnapshot::SourceTimingMode::kReset
          ? SB_RICH_SOURCE_OBSERVATION_RESET
          : SB_RICH_SOURCE_LEGACY_UNKNOWN;
  h.status = source.solve_status == spatialblend::SolverStatus::kFailed
                 ? SB_RICH_SNAPSHOT_FAILED
             : source.channels.empty() ? SB_RICH_SNAPSHOT_EMPTY
                                       : SB_RICH_SNAPSHOT_COMPLETE;
  h.policy_revision = source.policy_revision;
  h.tree_revision = source.tree_revision;
  h.counterpart_revision = source.counterpart_revision;
  h.support_revision = source.support_revision;
  h.known_revision_mask = source.known_revision_mask;
  h.unknown_revision_mask = source.unknown_revision_mask;
  if (source.channels.size() > kRichMaxChannels ||
      contributor_count > kRichMaxContributors) {
    // A limit violation replaces the prior result with a fresh, explicit
    // failed/release transaction.  No prefix is delivered or retained.
    h.status = SB_RICH_SNAPSHOT_FAILED;
    h.failure_code = SB_RICH_FAILURE_LIMIT_EXCEEDED;
    return frozen;
  }
  frozen->channels.reserve(source.channels.size());
  frozen->contributors.reserve(contributor_count);
  for (const auto& channel : source.channels) {
    sb_rich_realization_channel_t row{};
    row.anchor_offset = AppendRichUtf8(channel.anchor_name, &frozen->utf8);
    row.anchor_length = static_cast<uint32_t>(channel.anchor_name.size());
    row.contributor_offset = static_cast<uint32_t>(frozen->contributors.size());
    row.contributor_count = static_cast<uint32_t>(channel.contributors.size());
    row.family = static_cast<int32_t>(channel.family);
    row.solver_status = static_cast<int32_t>(channel.status);
    row.geometry = static_cast<int32_t>(channel.geometry);
    row.goal_kind = static_cast<int32_t>(channel.goal_kind);
    row.has_goal = channel.has_goal ? 1u : 0u;
    row.has_z = channel.target.has_z ? 1u : 0u;
    row.has_requested_position =
        channel.target.has_requested_position ? 1u : 0u;
    row.reach_projected = channel.reach_projected ? 1u : 0u;
    row.referent_id = static_cast<int64_t>(channel.target.referent_id);
    row.arrival_seconds = channel.target.arrival_time;
    row.position_x = channel.target.position.x();
    row.position_y = channel.target.position.y();
    row.position_z = channel.target.z;
    row.yaw = channel.target.yaw;
    row.weight = channel.target.weight;
    row.reach_displacement_m = channel.reach_displacement_m;
    row.requested_position_x = channel.target.requested_position.x();
    row.requested_position_y = channel.target.requested_position.y();
    frozen->channels.push_back(row);
    for (const auto& contributor : channel.contributors) {
      sb_rich_realization_contributor_t c{};
      c.referent_id = static_cast<int64_t>(contributor.referent_id);
      c.source_space = static_cast<int64_t>(contributor.source_space_id);
      c.source_element = static_cast<int64_t>(contributor.source_element_id);
      c.resolved_local_element =
          static_cast<int64_t>(contributor.resolved_local_element_id);
      c.binding_revision = contributor.binding_revision;
      c.snapshot_ordinal = contributor.snapshot_ordinal;
      static_assert(
          static_cast<int>(
              spatialblend::RealizationDisposition::kAwaitingEngagement) ==
              SB_RICH_DISPOSITION_AWAITING_ENGAGEMENT,
          "rich disposition mirrors RealizationDisposition");
      c.disposition = static_cast<int32_t>(contributor.disposition);
      c.grounding_mode = static_cast<int32_t>(contributor.grounding_mode);
      c.has_mapped_support = contributor.has_mapped_support ? 1u : 0u;
      c.chart_version_known = contributor.chart_version_known ? 1u : 0u;
      c.binding_revision_known = contributor.binding_revision_known ? 1u : 0u;
      c.reason_offset = AppendRichUtf8(contributor.reason, &frozen->utf8);
      c.reason_length = static_cast<uint32_t>(contributor.reason.size());
      c.support_identifier_offset =
          AppendRichUtf8(contributor.support_identifier, &frozen->utf8);
      c.support_identifier_length =
          static_cast<uint32_t>(contributor.support_identifier.size());
      c.applied_weight = contributor.applied_weight;
      c.mapped_support_x = contributor.mapped_support.x();
      c.mapped_support_y = contributor.mapped_support.y();
      c.mapped_support_z = contributor.mapped_support.z();
      for (int r = 0; r != 4; ++r)
        for (int col = 0; col != 4; ++col)
          c.support_plane_to_world_row_major[4 * r + col] =
              contributor.support_plane_to_world(r, col);
      frozen->contributors.push_back(c);
    }
  }
  if (frozen->utf8.size() > kRichMaxUtf8Bytes) {
    frozen->channels.clear();
    frozen->contributors.clear();
    frozen->utf8.clear();
    frozen->header.status = SB_RICH_SNAPSHOT_FAILED;
    frozen->header.failure_code = SB_RICH_FAILURE_LIMIT_EXCEEDED;
  }
  return frozen;
}

static sb_rich_realization_snapshot_t RichCallbackView(
    const FrozenRichSnapshot& frozen) {
  sb_rich_realization_snapshot_t view{};
  view.header = frozen.header;
  view.channels = frozen.channels.empty() ? nullptr : frozen.channels.data();
  view.contributors =
      frozen.contributors.empty() ? nullptr : frozen.contributors.data();
  view.utf8 = frozen.utf8.empty() ? nullptr : frozen.utf8.data();
  view.channel_count = static_cast<uint32_t>(frozen.channels.size());
  view.contributor_count = static_cast<uint32_t>(frozen.contributors.size());
  view.utf8_bytes = static_cast<uint32_t>(frozen.utf8.size());
  return view;
}

// A single drainer owns callback execution but holds no Space, registry, or
// callback-registration mutex. Reentrant retirement queues terminal delivery
// after the current complete callback; terminal is never overwritten.
static void EnqueueRichDelivery(const std::shared_ptr<RichDeliveryState>& state,
                                RichDelivery delivery) {
  bool drain = false;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->terminal_enqueued && !delivery.terminal) return;
    if (!delivery.terminal) {
      if (delivery.snapshot->header.revision <= state->accepted_revision)
        return;
      state->accepted_revision = delivery.snapshot->header.revision;
    }
    if (delivery.terminal) {
      state->terminal_enqueued = true;
      state->accepted_revision = std::max(state->accepted_revision,
                                          delivery.snapshot->header.revision);
    }
    if (!delivery.terminal && !state->pending.empty() &&
        !state->pending.back().terminal) {
      state->pending.back() = std::move(delivery);  // newest complete only.
    } else {
      state->pending.push_back(std::move(delivery));
    }
    if (!state->draining) {
      state->draining = true;
      drain = true;
    }
  }
  if (!drain) return;
  for (;;) {
    RichDelivery next;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->pending.empty()) {
        state->draining = false;
        return;
      }
      next = std::move(state->pending.front());
      state->pending.erase(state->pending.begin());
    }
    if (next.callback) {
      const sb_rich_realization_snapshot_t view =
          RichCallbackView(*next.snapshot);
      next.callback(&view, next.user_data);
    }
  }
}

// Freeze under Space's mutex through its copy API, then release every native
// lock before invoking application code.  Rich delivery is independent from
// the legacy target callback registration.
static int32_t DispatchRichRealization(sb_space_t space_handle,
                                       SessionContext::SpaceInfo* info,
                                       sb_actor_t actor,
                                       sb_rich_realization_token_t expected_token = 0) {
  if (!info || !info->space) return 0;
  spatialblend::ActorRealizationSnapshot core_snapshot;
  if (!info->space->CopyActorRealizationSnapshot(actor, &core_snapshot))
    return 0;
  sb_on_rich_realization_t callback = nullptr;
  void* callback_user_data = nullptr;
  {
    std::lock_guard<std::mutex> lock(info->callback_mutex);
    callback = info->on_rich_callback;
    callback_user_data = info->on_rich_user_data;
  }
  std::vector<std::pair<std::shared_ptr<RichDeliveryState>,
                        std::shared_ptr<const FrozenRichSnapshot>>>
      deliver;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    for (auto& [token, activation] : g_rich_activations) {
      if ((expected_token && token != expected_token) ||
          !activation.active || activation.space != space_handle ||
          activation.actor != actor)
        continue;
      if (activation.snapshot &&
          activation.snapshot->header.revision >= core_snapshot.revision)
        continue;  // duplicate/reordered peer delivery never rewinds a token.
      auto frozen =
          FreezeRichSnapshot(core_snapshot, token, space_handle, actor);
      activation.snapshot = frozen;
      if (callback)
        deliver.emplace_back(activation.delivery, std::move(frozen));
    }
  }
  for (const auto& [state, frozen] : deliver)
    EnqueueRichDelivery(state, {frozen, callback, callback_user_data, false});
  return deliver.empty() ? 0 : 1;
}

struct PeerSnapshot {
  sb_space_t handle;
  SessionContext::SpaceInfo* info;
  uint64_t received_count;
};

static std::vector<PeerSnapshot> SnapshotPeerReceipts(sb_session_t session,
                                                      sb_space_t source) {
  std::vector<PeerSnapshot> peers;
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(session);
  if (it == g_sessions.end()) return peers;
  for (auto& [handle, info] : it->second->spaces) {
    if (handle != source) {
      peers.push_back(
          {handle, info.get(), info->space->GetReceivedUpdateCount()});
    }
  }
  return peers;
}

static void DispatchNewPeerTargets(const std::vector<PeerSnapshot>& peers) {
  for (const PeerSnapshot& peer : peers) {
    const auto snapshot = peer.info->space->GetReceivedUpdatesSnapshot();
    if (snapshot.count <= peer.received_count) continue;
    const size_t retained = static_cast<size_t>(std::min<uint64_t>(
        snapshot.count - peer.received_count, snapshot.updates.size()));
    for (size_t i = snapshot.updates.size() - retained;
         i < snapshot.updates.size(); ++i) {
      const auto actor =
          peer.info->space->FindActorForRemoteUpdate(snapshot.updates[i]);
      if (actor != spatialblend::kInvalidEntityId) {
        DispatchRealizationTargets(peer.handle, peer.info,
                                   static_cast<sb_actor_t>(actor));
        DispatchRichRealization(peer.handle, peer.info,
                                static_cast<sb_actor_t>(actor));
      }
    }
  }
}

static sb_session_t SessionForSpace(sb_space_t space) {
  std::lock_guard<std::mutex> lock(g_spaces_mutex);
  auto it = g_space_to_session.find(space);
  return it == g_space_to_session.end() ? 0 : it->second;
}

static bool ObservationFramesEnabled(sb_space_t space) {
  const sb_session_t session = SessionForSpace(space);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(session);
  return it != g_sessions.end() &&
         it->second->hub_role == HubRole::kObservationFrames;
}

void sb_add_standard_anchors(sb_space_t space, sb_actor_t actor) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  info->space->AddActorAnchors(actor, spatialblend::StandardAnchors());
}

void sb_set_anchor_transform(sb_space_t space, sb_actor_t actor,
                             const char* anchor_name, double x, double y,
                             double z, double yaw) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name) {
    return;
  }
  info->space->SetAnchorWorldPose(static_cast<spatialblend::entity_id_t>(actor),
                                  anchor_name, Eigen::Vector3d(x, y, z), yaw);
}

void sb_clear_anchor_transform(sb_space_t space, sb_actor_t actor,
                               const char* anchor_name) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name) {
    return;
  }
  info->space->ClearAnchorWorldPose(static_cast<spatialblend::entity_id_t>(actor),
                                    anchor_name);
}

int32_t sb_set_anchor_active(sb_space_t space, sb_actor_t actor,
                             const char* anchor_name, int32_t active) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name) return 0;
  return info->space->SetAnchorActive(
             static_cast<spatialblend::entity_id_t>(actor), anchor_name,
             active != 0)
             ? 1
             : 0;
}

void sb_set_actor_element(sb_space_t space, sb_actor_t actor,
                          sb_element_t elem) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return;
  }
  info->space->SetActorElement(
      actor, elem == 0 ? spatialblend::kInvalidElementId : elem);
}

sb_actor_t sb_find_actor_by_shared_id(sb_space_t space, int64_t shared_id) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return 0;
  }
  const auto id = info->space->FindActorBySharedId(shared_id);
  return id == spatialblend::kInvalidEntityId ? 0 : id;
}

sb_handle_t sb_update_actor_state(sb_space_t space, sb_actor_t actor, double x,
                                  double y, double yaw) {
  // Look up space info and owning session without holding the session lock
  // during the update (UpdateActorState may be expensive).
  const sb_session_t session_id = SessionForSpace(space);
  auto* info = GetSpace(space);
  if (!info || !info->space) return kSyncDoneHandle;

  // Snapshot pre-update received counts for all other spaces in the session.
  const auto peers = SnapshotPeerReceipts(session_id, space);

  // Perform the state update. In LocalContext, this synchronously propagates
  // through SpaceHub to all other spaces in the session.
  info->space->UpdateActorState(actor, Eigen::Vector2d(x, y), yaw);
  // NOTE: we deliberately do NOT also send the raw position to the host via
  // SendUpdateActorState. UpdateActorState() above already propagates this
  // actor's RELATION update to the host (relation_update_callback_ ->
  // SendRelationUpdate, wired in sb_connect_space), and the host RECONSTRUCTS
  // the avatar's position from those relations (realization). Sending the raw
  // source-space coordinates too made the host overwrite the clone's
  // realized position with the *source's* frame, fighting the realized pose
  // each tick -> visible jitter. Relations are the single source of truth for
  // the remote avatar's placement.

  DispatchNewPeerTargets(peers);

  return kSyncDoneHandle;
}

sb_handle_t sb_update_actor_pose(sb_space_t space, sb_actor_t actor, double x,
                                 double y,
                                 const sb_orientation_t* orientation) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return kSyncDoneHandle;
  if (!orientation || orientation->struct_size != sizeof(sb_orientation_t) ||
      orientation->type < SB_ORIENTATION_YAW ||
      orientation->type > SB_ORIENTATION_YAW_PITCH_ROLL ||
      !std::isfinite(orientation->yaw) || !std::isfinite(orientation->pitch) ||
      !std::isfinite(orientation->roll))
    return kSyncDoneHandle;
  const sb_session_t session_id = SessionForSpace(space);
  const auto peers = SnapshotPeerReceipts(session_id, space);
  spatialblend::SourceOrientation o;
  o.type = orientation->type;
  o.pitch = orientation->type >= SB_ORIENTATION_YAW_PITCH ? orientation->pitch : 0.0;
  o.roll = orientation->type >= SB_ORIENTATION_YAW_PITCH_ROLL ? orientation->roll : 0.0;
  info->space->UpdateActorState(actor, Eigen::Vector2d(x, y), orientation->yaw, o);
  DispatchNewPeerTargets(peers);
  return kSyncDoneHandle;
}

int32_t sb_enable_actor_observation_frames(sb_session_t session) {
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(session);
  if (it == g_sessions.end()) return 0;
  if (it->second->hub_role == HubRole::kObservationFrames) return 1;
  if (it->second->hub_role != HubRole::kNone) return 0;
  it->second->hub_role = HubRole::kObservationFrames;
  return 1;
}

uint64_t sb_begin_actor_observation_epoch(sb_space_t space, sb_actor_t actor,
                                          const char* root_anchor_name) {
  auto* info = GetSpace(space);
  if (!info || !root_anchor_name || !ObservationFramesEnabled(space)) return 0;
  const auto peers = SnapshotPeerReceipts(SessionForSpace(space), space);
  uint64_t epoch = 0;
  if (!info->space->BeginActorObservationEpoch(actor, root_anchor_name, &epoch))
    return 0;
  DispatchNewPeerTargets(peers);
  return epoch;
}

int32_t sb_apply_actor_observation_frame(
    sb_space_t space, sb_actor_t actor, uint64_t epoch, int64_t frame_sequence,
    double time_seconds, const sb_actor_observation_t* observations,
    int32_t count) {
  auto* info = GetSpace(space);
  if (!info || !ObservationFramesEnabled(space) || count < 0 ||
      (count > 0 && !observations))
    return 0;
  std::vector<spatialblend::Space::ActorObservation> staged;
  staged.reserve(count);
  for (int32_t i = 0; i < count; ++i) {
    const auto& value = observations[i];
    if (!value.anchor_name || !std::isfinite(value.position_x) ||
        !std::isfinite(value.position_y) || !std::isfinite(value.position_z) ||
        !std::isfinite(value.yaw))
      return 0;
    staged.push_back(
        {value.anchor_name,
         Eigen::Vector3d(value.position_x, value.position_y, value.position_z),
         value.yaw});
  }
  const auto peers = SnapshotPeerReceipts(SessionForSpace(space), space);
  if (!info->space->ApplyActorObservationFrame(actor, epoch, frame_sequence,
                                               time_seconds, staged))
    return 0;
  DispatchNewPeerTargets(peers);
  return 1;
}

uint32_t sb_get_observed_collaborator_feature(
    sb_observed_collaborator_feature_t* out_feature, uint32_t out_size) {
  const sb_observed_collaborator_feature_t feature{
      sizeof(feature),
      SB_OBSERVED_COLLABORATOR_FEATURE_VERSION,
      sizeof(sb_observed_collaborator_frame_t),
      sizeof(sb_actor_observation_t),
      sizeof(sb_observed_collaborator_publication_t),
      kObservedCollaboratorMaxFrames,
      kObservedCollaboratorMaxObservationsPerFrame,
      kObservedCollaboratorMaxPublications,
      kObservedCollaboratorMaxAnchorNameBytes,
      kObservedCollaboratorMaxAgeNs,
      1,  // Core Z-up.
      0};
  if (out_feature && out_size >= sizeof(feature)) *out_feature = feature;
  return sizeof(feature);
}

uint32_t sb_get_observed_collaborator_field_count(uint32_t layout) {
  switch (layout) {
    case SB_OBSERVED_COLLABORATOR_LAYOUT_FEATURE:
      return 12;
    case SB_OBSERVED_COLLABORATOR_LAYOUT_FRAME:
      return 9;
    case SB_OBSERVED_COLLABORATOR_LAYOUT_PUBLICATION:
      return 6;
    default:
      return 0;
  }
}

uint32_t sb_get_observed_collaborator_field_offset(uint32_t layout,
                                                    uint32_t field) {
  const uint32_t feature[] = {
      offsetof(sb_observed_collaborator_feature_t, descriptor_size),
      offsetof(sb_observed_collaborator_feature_t, feature_version),
      offsetof(sb_observed_collaborator_feature_t, frame_size),
      offsetof(sb_observed_collaborator_feature_t, observation_size),
      offsetof(sb_observed_collaborator_feature_t, publication_size),
      offsetof(sb_observed_collaborator_feature_t, max_frames),
      offsetof(sb_observed_collaborator_feature_t, max_observations_per_frame),
      offsetof(sb_observed_collaborator_feature_t, max_publications),
      offsetof(sb_observed_collaborator_feature_t, max_anchor_name_bytes),
      offsetof(sb_observed_collaborator_feature_t, max_age_ns),
      offsetof(sb_observed_collaborator_feature_t, coordinate_convention),
      offsetof(sb_observed_collaborator_feature_t, reserved)};
  const uint32_t frame[] = {
      offsetof(sb_observed_collaborator_frame_t, struct_size),
      offsetof(sb_observed_collaborator_frame_t, feature_version),
      offsetof(sb_observed_collaborator_frame_t, actor),
      offsetof(sb_observed_collaborator_frame_t, epoch),
      offsetof(sb_observed_collaborator_frame_t, sequence),
      offsetof(sb_observed_collaborator_frame_t, time_seconds),
      offsetof(sb_observed_collaborator_frame_t, observations),
      offsetof(sb_observed_collaborator_frame_t, observation_count),
      offsetof(sb_observed_collaborator_frame_t, observation_size)};
  const uint32_t publication[] = {
      offsetof(sb_observed_collaborator_publication_t, struct_size),
      offsetof(sb_observed_collaborator_publication_t, feature_version),
      offsetof(sb_observed_collaborator_publication_t, element),
      offsetof(sb_observed_collaborator_publication_t, lease),
      offsetof(sb_observed_collaborator_publication_t, observed_at_ns),
      offsetof(sb_observed_collaborator_publication_t, valid_until_ns)};
  const uint32_t* offsets = nullptr;
  uint32_t count = 0;
  switch (layout) {
    case SB_OBSERVED_COLLABORATOR_LAYOUT_FEATURE:
      offsets = feature; count = sizeof(feature) / sizeof(feature[0]); break;
    case SB_OBSERVED_COLLABORATOR_LAYOUT_FRAME:
      offsets = frame; count = sizeof(frame) / sizeof(frame[0]); break;
    case SB_OBSERVED_COLLABORATOR_LAYOUT_PUBLICATION:
      offsets = publication; count = sizeof(publication) / sizeof(publication[0]); break;
    default:
      return UINT32_MAX;
  }
  return field < count ? offsets[field] : UINT32_MAX;
}

static bool IsBoundedObservedCollaboratorName(const char* value) {
  if (!value) return false;
  for (uint32_t index = 0; index <= kObservedCollaboratorMaxAnchorNameBytes;
       ++index) {
    if (value[index] == '\0') return index != 0;
  }
  return false;
}

uint32_t sb_get_observed_source_setup_feature(
    sb_observed_source_setup_feature_t* out_feature, uint32_t out_size) {
  const sb_observed_source_setup_feature_t feature{
      sizeof(feature),
      SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION,
      sizeof(sb_observed_source_setup_participant_t),
      sizeof(sb_observed_source_setup_point_t),
      sizeof(sb_observed_source_setup_actor_result_t),
      sizeof(sb_observed_source_setup_point_result_t),
      kObservedCollaboratorMaxFrames,
      kObservedCollaboratorMaxPublications,
      kObservedCollaboratorMaxAnchorNameBytes,
      0,
      kObservedCollaboratorMaxAgeNs,
      1,  // Core Z-up.
      0};
  if (out_feature && out_size >= sizeof(feature)) *out_feature = feature;
  return sizeof(feature);
}

uint32_t sb_get_observed_source_setup_field_count(uint32_t layout) {
  switch (layout) {
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_FEATURE:
      return 13;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_PARTICIPANT:
      return 6;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT:
      return 7;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_ACTOR_RESULT:
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT_RESULT:
      return 2;
    default:
      return 0;
  }
}

uint32_t sb_get_observed_source_setup_field_offset(uint32_t layout,
                                                    uint32_t field) {
  const uint32_t feature[] = {
      offsetof(sb_observed_source_setup_feature_t, descriptor_size),
      offsetof(sb_observed_source_setup_feature_t, feature_version),
      offsetof(sb_observed_source_setup_feature_t, participant_size),
      offsetof(sb_observed_source_setup_feature_t, point_size),
      offsetof(sb_observed_source_setup_feature_t, actor_result_size),
      offsetof(sb_observed_source_setup_feature_t, point_result_size),
      offsetof(sb_observed_source_setup_feature_t, max_participants),
      offsetof(sb_observed_source_setup_feature_t, max_points),
      offsetof(sb_observed_source_setup_feature_t, max_anchor_name_bytes),
      offsetof(sb_observed_source_setup_feature_t, reserved),
      offsetof(sb_observed_source_setup_feature_t, max_age_ns),
      offsetof(sb_observed_source_setup_feature_t, coordinate_convention),
      offsetof(sb_observed_source_setup_feature_t, reserved2)};
  const uint32_t participant[] = {
      offsetof(sb_observed_source_setup_participant_t, struct_size),
      offsetof(sb_observed_source_setup_participant_t, feature_version),
      offsetof(sb_observed_source_setup_participant_t, space),
      offsetof(sb_observed_source_setup_participant_t, actor),
      offsetof(sb_observed_source_setup_participant_t, root_anchor_name),
      offsetof(sb_observed_source_setup_participant_t, expected_prior_epoch)};
  const uint32_t point[] = {
      offsetof(sb_observed_source_setup_point_t, struct_size),
      offsetof(sb_observed_source_setup_point_t, feature_version),
      offsetof(sb_observed_source_setup_point_t, space),
      offsetof(sb_observed_source_setup_point_t, element),
      offsetof(sb_observed_source_setup_point_t, producer),
      offsetof(sb_observed_source_setup_point_t, anchor_name),
      offsetof(sb_observed_source_setup_point_t, maximum_age_ns)};
  const uint32_t actor_result[] = {
      offsetof(sb_observed_source_setup_actor_result_t, actor),
      offsetof(sb_observed_source_setup_actor_result_t, epoch)};
  const uint32_t point_result[] = {
      offsetof(sb_observed_source_setup_point_result_t, element),
      offsetof(sb_observed_source_setup_point_result_t, lease)};
  const uint32_t* offsets = nullptr;
  uint32_t count = 0;
  switch (layout) {
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_FEATURE:
      offsets = feature; count = sizeof(feature) / sizeof(feature[0]); break;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_PARTICIPANT:
      offsets = participant; count = sizeof(participant) / sizeof(participant[0]); break;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT:
      offsets = point; count = sizeof(point) / sizeof(point[0]); break;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_ACTOR_RESULT:
      offsets = actor_result; count = sizeof(actor_result) / sizeof(actor_result[0]); break;
    case SB_OBSERVED_SOURCE_SETUP_LAYOUT_POINT_RESULT:
      offsets = point_result; count = sizeof(point_result) / sizeof(point_result[0]); break;
    default:
      return UINT32_MAX;
  }
  return field < count ? offsets[field] : UINT32_MAX;
}

int32_t sb_begin_observed_source_setup(
    sb_space_t space,
    const sb_observed_source_setup_participant_t* participants,
    uint32_t participant_count,
    const sb_observed_source_setup_point_t* points, uint32_t point_count,
    sb_observed_source_setup_t* out_setup,
    sb_observed_source_setup_actor_result_t* out_actors,
    uint32_t out_actor_capacity,
    sb_observed_source_setup_point_result_t* out_points,
    uint32_t out_point_capacity) {
  // Relation callbacks execute under Space's solve lock. Observed-collaborator mutation from
  // that callback would otherwise either re-enter the lock or publish behind
  // the callback's in-flight relation snapshot.
  if (g_inside_relation_callback) return 0;
  auto* info = GetSpace(space);
  if (!info || !ObservationFramesEnabled(space) || !participants || !points ||
      !out_setup || !out_actors || !out_points || participant_count == 0 ||
      point_count == 0 || participant_count > kObservedCollaboratorMaxFrames ||
      point_count > kObservedCollaboratorMaxPublications ||
      out_actor_capacity < participant_count || out_point_capacity < point_count)
    return 0;

  std::vector<spatialblend::Space::ObservedSourceSetupParticipant>
      staged_participants;
  std::vector<spatialblend::Space::ObservedSourceSetupPoint> staged_points;
  staged_participants.reserve(participant_count);
  staged_points.reserve(point_count);
  for (uint32_t index = 0; index < participant_count; ++index) {
    const auto& participant = participants[index];
    if (participant.struct_size != sizeof(participant) ||
        participant.feature_version != SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION ||
        participant.space != space || participant.actor == 0 ||
        !IsBoundedObservedCollaboratorName(participant.root_anchor_name))
      return 0;
    staged_participants.push_back(
        {participant.actor, participant.root_anchor_name,
         participant.expected_prior_epoch});
  }
  for (uint32_t index = 0; index < point_count; ++index) {
    const auto& point = points[index];
    if (point.struct_size != sizeof(point) ||
        point.feature_version != SB_OBSERVED_SOURCE_SETUP_FEATURE_VERSION ||
        point.space != space || point.element == 0 || point.producer == 0 ||
        !IsBoundedObservedCollaboratorName(point.anchor_name) ||
        point.maximum_age_ns <= 0 ||
        point.maximum_age_ns > kObservedCollaboratorMaxAgeNs)
      return 0;
    staged_points.push_back(
        {point.element, point.producer, point.anchor_name, point.maximum_age_ns});
  }

  // Resolve the space before taking the session registry lock. Holding the
  // latter through the short no-upcall installation pins `info` without ever
  // acquiring g_spaces_mutex underneath it.
  const sb_session_t session = SessionForSpace(space);
  sb_observed_source_setup_t setup = 0;
  std::vector<spatialblend::Space::ObservedSourceSetupActorResult> actor_results;
  std::vector<spatialblend::Space::ObservedSourceSetupPointResult> point_results;
  {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    const auto session_it = g_sessions.find(session);
    if (session_it == g_sessions.end() ||
        session_it->second->hub_role != HubRole::kObservationFrames ||
        !info->space)
      return 0;
    setup = g_next_observed_source_setup_id.fetch_add(1);
    if (setup <= 0 || setup == INT64_MAX) return 0;
    g_observed_source_setups.emplace(
        setup, ObservedSourceSetupRegistration{session, space, {}, {},
                                                ObservedSourceSetupRegistration::State::kInstalling});
    // Session teardown cannot invalidate this SpaceInfo until the exact Core
    // setup result is published or the installing registry entry is removed.
    if (!info->space->BeginObservedSourceSetup(staged_participants, staged_points,
                                                &actor_results, &point_results)) {
      g_observed_source_setups.erase(setup);
      return 0;
    }
    const auto registration = g_observed_source_setups.find(setup);
    if (registration == g_observed_source_setups.end()) return 0;
    registration->second.actors = actor_results;
    registration->second.points = point_results;
    registration->second.state = ObservedSourceSetupRegistration::State::kActive;
  }
  // Result writes are deliberately last: malformed setup never creates a
  // prefix result visible to the caller.
  for (uint32_t index = 0; index < participant_count; ++index)
    out_actors[index] = {actor_results[index].actor, actor_results[index].epoch};
  for (uint32_t index = 0; index < point_count; ++index)
    out_points[index] = {point_results[index].element, point_results[index].lease};
  *out_setup = setup;
  return 1;
}

int32_t sb_retire_observed_source_setup(sb_space_t space,
                                        sb_observed_source_setup_t setup) {
  if (g_inside_relation_callback) return 0;
  if (!setup) return 0;
  // Get the mapping before the registry lock. Once the session lock is held,
  // session teardown cannot invalidate the resulting SpaceInfo.
  auto* info = GetSpace(space);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  const auto registration = g_observed_source_setups.find(setup);
  if (registration == g_observed_source_setups.end() ||
      registration->second.space != space)
    return 0;
  if (registration->second.state == ObservedSourceSetupRegistration::State::kRetired)
    return 1;
  if (registration->second.state != ObservedSourceSetupRegistration::State::kActive ||
      !info || !info->space ||
      g_sessions.find(registration->second.session) == g_sessions.end() ||
      !info->space->RetireObservedSourceSetup(registration->second.actors,
                                              registration->second.points))
    return 0;
  registration->second.state = ObservedSourceSetupRegistration::State::kRetired;
  return 1;
}

int32_t sb_release_observed_source_setup(sb_observed_source_setup_t setup) {
  if (g_inside_relation_callback) return 0;
  if (!setup) return 0;
  sb_space_t space = 0;
  {
    // Do not nest the registry and space-map locks. A second registry lookup
    // below makes a concurrent retire/release/destroy transition harmless.
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    const auto registration = g_observed_source_setups.find(setup);
    if (registration == g_observed_source_setups.end()) return 0;
    space = registration->second.space;
  }
  auto* info = GetSpace(space);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  const auto registration = g_observed_source_setups.find(setup);
  if (registration == g_observed_source_setups.end()) return 0;
  if (registration->second.state == ObservedSourceSetupRegistration::State::kRetired) {
    g_observed_source_setups.erase(registration);
    return 1;
  }
  if (registration->second.state != ObservedSourceSetupRegistration::State::kActive)
    return 0;
  // Session destroy normally erases registrations under this lock. If a
  // caller observes a missing mapping while a registration remains, that
  // setup is already terminal: only its registry capability is discarded.
  if (!info || !info->space ||
      g_sessions.find(registration->second.session) == g_sessions.end()) {
    g_observed_source_setups.erase(registration);
    return 1;
  }
  const auto state = info->space->GetObservedSourceSetupReleaseState(
      registration->second.actors, registration->second.points);
  if (state == spatialblend::Space::ObservedSourceSetupReleaseState::kUnadmitted)
    return 0;
  // For admitted and stale/terminal setups this is capability cleanup only;
  // it never retires Core state or mutates an admitted epoch/lease.
  g_observed_source_setups.erase(registration);
  return 1;
}

int32_t sb_begin_observed_source_anchor_point(
    sb_space_t space, sb_element_t element, sb_actor_t producer,
    const char* anchor_name, int64_t maximum_age_ns, uint64_t* out_lease) {
  if (out_lease) *out_lease = 0;
  if (g_inside_relation_callback) return 0;
  auto* info = GetSpace(space);
  if (!info || !out_lease || !ObservationFramesEnabled(space) ||
      !IsBoundedObservedCollaboratorName(anchor_name) || maximum_age_ns <= 0 ||
      maximum_age_ns > kObservedCollaboratorMaxAgeNs)
    return 0;
  return info->space->BeginObservedFacingPoint(
             element, producer, anchor_name,
             spatialblend::Space::ObservedFacingPointKind::kSourceAnchor,
             maximum_age_ns, out_lease)
             ? 1
             : 0;
}

int32_t sb_retire_observed_source_anchor_point(
    sb_space_t space, sb_element_t element, uint64_t lease) {
  if (g_inside_relation_callback) return 0;
  auto* info = GetSpace(space);
  if (!info || !ObservationFramesEnabled(space) || !lease) return 0;
  return info->space->RetireObservedFacingPoint(element, lease) ? 1 : 0;
}

int32_t sb_apply_observed_collaborator_batch(
    sb_space_t space, const sb_observed_collaborator_frame_t* frames,
    uint32_t frame_count,
    const sb_observed_collaborator_publication_t* publications,
    uint32_t publication_count) {
  if (g_inside_relation_callback) return 0;
  auto* info = GetSpace(space);
  if (!info || !ObservationFramesEnabled(space) || !frames || frame_count == 0 ||
      frame_count > kObservedCollaboratorMaxFrames ||
      publication_count > kObservedCollaboratorMaxPublications ||
      (publication_count != 0 && !publications))
    return 0;

  std::vector<spatialblend::Space::ActorObservationFrame> staged_frames;
  staged_frames.reserve(frame_count);
  for (uint32_t index = 0; index < frame_count; ++index) {
    const auto& frame = frames[index];
    if (frame.struct_size != sizeof(frame) ||
        frame.feature_version != SB_OBSERVED_COLLABORATOR_FEATURE_VERSION ||
        frame.actor == 0 || frame.sequence < 0 || !std::isfinite(frame.time_seconds) ||
        frame.time_seconds < 0 ||
        frame.observation_count > kObservedCollaboratorMaxObservationsPerFrame ||
        frame.observation_size != sizeof(sb_actor_observation_t) ||
        (frame.observation_count != 0 && !frame.observations))
      return 0;
    std::vector<spatialblend::Space::ActorObservation> observations;
    observations.reserve(frame.observation_count);
    for (uint32_t observation_index = 0;
         observation_index < frame.observation_count; ++observation_index) {
      const auto& observation = frame.observations[observation_index];
      if (!IsBoundedObservedCollaboratorName(observation.anchor_name) ||
          !std::isfinite(observation.position_x) ||
          !std::isfinite(observation.position_y) ||
          !std::isfinite(observation.position_z) || !std::isfinite(observation.yaw))
        return 0;
      observations.push_back({observation.anchor_name,
                              Eigen::Vector3d(observation.position_x,
                                              observation.position_y,
                                              observation.position_z),
                              observation.yaw});
    }
    staged_frames.push_back(
        {frame.actor, frame.epoch, frame.sequence, frame.time_seconds,
         std::move(observations)});
  }

  std::vector<spatialblend::Space::ObservedFacingPointPublication>
      staged_publications;
  staged_publications.reserve(publication_count);
  for (uint32_t index = 0; index < publication_count; ++index) {
    const auto& publication = publications[index];
    if (publication.struct_size != sizeof(publication) ||
        publication.feature_version != SB_OBSERVED_COLLABORATOR_FEATURE_VERSION ||
        publication.element == 0 || publication.lease == 0 ||
        publication.observed_at_ns <= 0 ||
        publication.valid_until_ns <= publication.observed_at_ns ||
        publication.valid_until_ns - publication.observed_at_ns >
            kObservedCollaboratorMaxAgeNs)
      return 0;
    staged_publications.push_back({publication.element, publication.lease,
                                   publication.observed_at_ns,
                                   publication.valid_until_ns});
  }
  const auto peers = SnapshotPeerReceipts(SessionForSpace(space), space);
  if (!info->space->ApplyActorObservationBatch(staged_frames,
                                                staged_publications))
    return 0;
  DispatchNewPeerTargets(peers);
  return 1;
}

int32_t sb_clear_actor_observation_anchor(sb_space_t space, sb_actor_t actor,
                                          uint64_t epoch,
                                          const char* anchor_name) {
  auto* info = GetSpace(space);
  if (!info || !anchor_name || !ObservationFramesEnabled(space)) return 0;
  const auto peers = SnapshotPeerReceipts(SessionForSpace(space), space);
  if (!info->space->ClearActorObservationAnchor(actor, epoch, anchor_name))
    return 0;
  DispatchNewPeerTargets(peers);
  return 1;
}

int32_t sb_get_actor_observation_clock(sb_space_t space, sb_actor_t actor,
                                       uint64_t* epoch, int64_t* frame_sequence,
                                       double* time_seconds,
                                       int32_t* has_sample) {
  if (!epoch || !frame_sequence || !time_seconds || !has_sample) return 0;
  auto* info = GetSpace(space);
  uint64_t e;
  int64_t s;
  double t;
  bool has;
  if (!info || !ObservationFramesEnabled(space) ||
      !info->space->GetActorObservationClock(actor, &e, &s, &t, &has))
    return 0;
  *epoch = e;
  *frame_sequence = s;
  *time_seconds = t;
  *has_sample = has ? 1 : 0;
  return 1;
}

int64_t sb_get_last_emitted_update_sequence(sb_space_t space,
                                            sb_actor_t actor) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  return info->space->GetLastEmittedSequence(actor);
}

int64_t sb_get_last_applied_remote_sequence(sb_space_t receiver_space,
                                            sb_space_t source_space,
                                            sb_actor_t source_actor) {
  auto* receiver = GetSpace(receiver_space);
  auto* source = GetSpace(source_space);
  if (!receiver || !receiver->space || !source || !source->space) return 0;

  spatialblend::space_id_t source_space_id = spatialblend::kInvalidSpaceId;
  spatialblend::shared_element_id_t source_actor_shared_id =
      spatialblend::kInvalidSharedElementId;
  if (!source->space->GetActorWireIdentity(source_actor, &source_space_id,
                                           &source_actor_shared_id)) {
    return 0;
  }
  return receiver->space->GetLastAppliedRemoteSequence(source_space_id,
                                                       source_actor_shared_id);
}

sb_actor_state_t sb_get_actor_state(sb_space_t space, sb_actor_t actor) {
  auto* info = GetSpace(space);
  if (!info || !info->space) {
    return {0.0, 0.0, 0.0};
  }
  Eigen::Vector2d position;
  double yaw = 0.0;
  if (!info->space->CopyActorPose(actor, &position, &yaw)) {
    return {0.0, 0.0, 0.0};
  }
  return {position.x(), position.y(), yaw};
}

int32_t sb_get_anchor_id(sb_space_t space, sb_actor_t actor,
                         const char* anchor_name, uint32_t* id) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name || !id) return -1;
  return info->space->AnchorIdFor(actor, anchor_name, id) ? 0 : -1;
}

int32_t sb_get_anchor_name(sb_space_t space, sb_actor_t actor,
                           uint32_t anchor_id, char* buf, int32_t cap) {
  auto* info = GetSpace(space);
  std::string name;
  if (!info || !info->space || !buf || cap < 1 ||
      !info->space->AnchorNameOf(actor, anchor_id, &name))
    return -1;
  const size_t n = std::min<size_t>(name.size(), static_cast<size_t>(cap - 1));
  std::memcpy(buf, name.data(), n);
  buf[n] = '\0';
  return static_cast<int32_t>(name.size());
}

int32_t sb_publish_anchor_pose(sb_space_t space, sb_actor_t actor,
                               uint32_t anchor_id, const sb_anchor_pose_t* pose,
                               double timestamp_s, uint64_t sequence) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !pose ||
      pose->struct_size != sizeof(sb_anchor_pose_t) ||
      pose->orientation.struct_size != sizeof(sb_orientation_t) ||
      pose->orientation.type < SB_ORIENTATION_YAW ||
      pose->orientation.type > SB_ORIENTATION_YAW_PITCH_ROLL)
    return -1;
  spatialblend::EmbodiedPoseInput in;
  in.x = pose->x;
  in.y = pose->y;
  in.z = pose->z;
  in.yaw = pose->orientation.yaw;
  in.pitch = pose->orientation.pitch;
  in.roll = pose->orientation.roll;
  in.orientation_type = pose->orientation.type;
  in.epoch = pose->epoch;
  in.flags = pose->flags;
  switch (info->space->PublishAnchorPose(actor, anchor_id, in, timestamp_s,
                                         sequence)) {
    case spatialblend::EmbodiedPublishResult::kOk:
      return 0;
    case spatialblend::EmbodiedPublishResult::kStale:
      return -2;
    case spatialblend::EmbodiedPublishResult::kRefused:
      return -3;
    case spatialblend::EmbodiedPublishResult::kInvalid:
      break;
  }
  return -1;
}

namespace {

bool ReadBonePose(const sb_bone_pose_t* in, spatialblend::RigidTransform* out) {
  if (!in || in->struct_size != sizeof(sb_bone_pose_t)) return false;
  for (double v : in->position)
    if (!std::isfinite(v)) return false;
  for (double v : in->rotation)
    if (!std::isfinite(v)) return false;
  Eigen::Quaterniond q(in->rotation[3], in->rotation[0], in->rotation[1], in->rotation[2]);
  if (q.norm() == 0.0) return false;
  q.normalize();
  out->position = Eigen::Vector3d(in->position[0], in->position[1], in->position[2]);
  out->rotation = q;
  return true;
}

void WriteBonePose(const spatialblend::RigidTransform& in, sb_bone_pose_t* out) {
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(sb_bone_pose_t);
  out->position[0] = in.position.x();
  out->position[1] = in.position.y();
  out->position[2] = in.position.z();
  out->rotation[0] = in.rotation.x();
  out->rotation[1] = in.rotation.y();
  out->rotation[2] = in.rotation.z();
  out->rotation[3] = in.rotation.w();
}

void CopyError(const std::string& text, char* error, int32_t cap) {
  if (!error || cap < 1) return;
  const size_t n = std::min<size_t>(text.size(), static_cast<size_t>(cap - 1));
  std::memcpy(error, text.data(), n);
  error[n] = '\0';
}

}  // namespace

int32_t sb_register_embodiment_json(sb_space_t space, sb_actor_t actor, const char* json, int32_t json_len,
                                    char* error, int32_t error_cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !json || json_len < 1) return -1;
  spatialblend::Embodiment embodiment;
  std::string why;
  if (!spatialblend::Embodiment::FromJson(std::string(json, static_cast<size_t>(json_len)), &embodiment, &why)) {
    CopyError(why, error, error_cap);
    return -2;
  }
  std::string conflict;
  switch (info->space->RegisterEmbodiment(static_cast<spatialblend::entity_id_t>(actor), embodiment, &conflict)) {
    case spatialblend::Space::RegisterEmbodimentResult::kBindingConflict:
      CopyError("an existing binding is not among the anchor's declared families: " + conflict, error, error_cap);
      return -3;
    case spatialblend::Space::RegisterEmbodimentResult::kOk:
      return 0;
    case spatialblend::Space::RegisterEmbodimentResult::kRefused:
      CopyError("actor already has an embodiment, a bone-bound anchor already has a published port, or the pose is input-owned",
                error, error_cap);
      return -3;
    case spatialblend::Space::RegisterEmbodimentResult::kInvalid:
      break;
  }
  CopyError("unknown actor", error, error_cap);
  return -1;
}

int32_t sb_get_embodiment_json(sb_space_t space, sb_actor_t actor, char* buf, int32_t cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  if (!e) {
    if (buf && cap > 0) buf[0] = '\0';
    return 0;
  }
  const std::string text = e->ToJson();
  if (buf && cap > 0) {
    const size_t n = std::min<size_t>(text.size(), static_cast<size_t>(cap - 1));
    std::memcpy(buf, text.data(), n);
    buf[n] = '\0';
  }
  return static_cast<int32_t>(text.size());
}

int32_t sb_get_bone_id(sb_space_t space, sb_actor_t actor, const char* name) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !name) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  return e ? e->BoneId(name) : -1;
}

int32_t sb_publish_bone_pose(sb_space_t space, sb_actor_t actor, int32_t bone_id, const sb_bone_pose_t* pose,
                             double timestamp_s, uint64_t sequence) {
  auto* info = GetSpace(space);
  spatialblend::RigidTransform rigid;
  if (!info || !info->space || !ReadBonePose(pose, &rigid)) return -1;
  spatialblend::BonePoseInput in;
  for (int i = 0; i < 3; ++i) in.p[i] = pose->position[i];
  in.q[0] = pose->rotation[0];
  in.q[1] = pose->rotation[1];
  in.q[2] = pose->rotation[2];
  in.q[3] = pose->rotation[3];
  in.epoch = pose->epoch;
  in.flags = pose->flags;
  switch (info->space->PublishBonePose(static_cast<spatialblend::entity_id_t>(actor), bone_id, in, timestamp_s,
                                       sequence)) {
    case spatialblend::EmbodiedPublishResult::kOk:
      return 0;
    case spatialblend::EmbodiedPublishResult::kStale:
      return -2;
    case spatialblend::EmbodiedPublishResult::kRefused:
      return -3;
    case spatialblend::EmbodiedPublishResult::kInvalid:
      break;
  }
  return -1;
}

int32_t sb_apply_actor_bone_frame(sb_space_t space, sb_actor_t actor, uint64_t epoch, int64_t frame_sequence,
                                  double time_seconds, const sb_named_bone_pose_t* bones, int32_t bone_count,
                                  const sb_actor_observation_t* observations, int32_t observation_count) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !ObservationFramesEnabled(space) || bone_count < 0 || observation_count < 0 ||
      (bone_count > 0 && !bones) || (observation_count > 0 && !observations))
    return 0;
  const spatialblend::entity_id_t id = static_cast<spatialblend::entity_id_t>(actor);
  const auto embodiment = info->space->EmbodimentOf(id);
  if (!embodiment) return 0;
  std::vector<std::pair<std::string, spatialblend::RigidTransform>> named;
  named.reserve(bone_count);
  for (int32_t i = 0; i < bone_count; ++i) {
    spatialblend::RigidTransform rigid;
    if (!bones[i].bone_name || !ReadBonePose(&bones[i].pose, &rigid)) return 0;
    named.emplace_back(bones[i].bone_name, rigid);
  }
  std::vector<spatialblend::Space::ActorObservation> staged;
  if (!info->space->ComposeBoneObservations(id, named, &staged)) return 0;
  for (int32_t i = 0; i < observation_count; ++i) {
    const auto& value = observations[i];
    if (!value.anchor_name || !std::isfinite(value.position_x) || !std::isfinite(value.position_y) ||
        !std::isfinite(value.position_z) || !std::isfinite(value.yaw))
      return 0;
    // A bone-bound anchor is derived from its bone: stating it as well would be two sources for one fact.
    const auto* declared = embodiment->FindAnchor(value.anchor_name);
    if (declared && declared->bone >= 0) return 0;
    staged.push_back({value.anchor_name, Eigen::Vector3d(value.position_x, value.position_y, value.position_z),
                      value.yaw});
  }
  const auto peers = SnapshotPeerReceipts(SessionForSpace(space), space);
  if (!info->space->ApplyActorObservationFrame(id, epoch, frame_sequence, time_seconds, staged)) return 0;
  DispatchNewPeerTargets(peers);
  return 1;
}

int32_t sb_set_bone_anchor_transforms(sb_space_t space, sb_actor_t actor, const sb_named_bone_pose_t* bones,
                                      int32_t bone_count) {
  auto* info = GetSpace(space);
  if (!info || !info->space || bone_count < 0 || (bone_count > 0 && !bones)) return -1;
  const spatialblend::entity_id_t id = static_cast<spatialblend::entity_id_t>(actor);
  if (!info->space->EmbodimentOf(id)) return -1;
  std::vector<std::pair<std::string, spatialblend::RigidTransform>> named;
  named.reserve(bone_count);
  for (int32_t i = 0; i < bone_count; ++i) {
    spatialblend::RigidTransform rigid;
    if (!bones[i].bone_name || !ReadBonePose(&bones[i].pose, &rigid)) return -1;
    named.emplace_back(bones[i].bone_name, rigid);
  }
  std::vector<spatialblend::Space::ActorObservation> composed;
  if (!info->space->ComposeBoneObservations(id, named, &composed)) return -1;
  int32_t set = 0;
  for (const auto& o : composed)
    if (info->space->SetBoneAnchorWorldPose(id, o.anchor_name, o.position, o.yaw, o.orientation_type, o.pitch, o.roll))
      ++set;
  return set;
}

int32_t sb_get_chain_joint_count(sb_space_t space, sb_actor_t actor) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  return e && e->chain.present ? static_cast<int32_t>(e->joints.size()) : 0;
}

int32_t sb_get_chain_joint(sb_space_t space, sb_actor_t actor, int32_t index, sb_chain_joint_t* out, char* name,
                           int32_t name_cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out || out->struct_size != sizeof(sb_chain_joint_t)) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  if (!e || !e->chain.present || index < 0 || index >= static_cast<int32_t>(e->joints.size())) return -1;
  const spatialblend::ChainJoint& cj = e->chain.joints[index];
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(sb_chain_joint_t);
  out->parent = e->joints[index].parent;
  for (int i = 0; i < 3; ++i) out->rest_position[i] = cj.rest.position[i];
  out->rest_rotation[0] = cj.rest.rotation.x();
  out->rest_rotation[1] = cj.rest.rotation.y();
  out->rest_rotation[2] = cj.rest.rotation.z();
  out->rest_rotation[3] = cj.rest.rotation.w();
  out->segment_length_m = spatialblend::SegmentLength(*e, index);
  for (int a = 0; a < 3; ++a) {
    if (cj.has_lo[a]) {
      out->limit_mask |= 1u << (2 * a);
      out->limit_min[a] = cj.lo[a];
    }
    if (cj.has_hi[a]) {
      out->limit_mask |= 1u << (2 * a + 1);
      out->limit_max[a] = cj.hi[a];
    }
  }
  if (name && name_cap > 0) CopyError(e->joints[index].name, name, name_cap);
  return 0;
}

int32_t sb_convert_pose_basis(const double* basis, int32_t to_canonical, const sb_bone_pose_t* in,
                              sb_bone_pose_t* out) {
  spatialblend::AxisBasis b;
  spatialblend::RigidTransform rigid;
  if (!basis || !out || !ReadBonePose(in, &rigid) || !spatialblend::AxisBasis::FromRows(basis, &b, nullptr))
    return -1;
  const spatialblend::RigidTransform converted =
      to_canonical ? spatialblend::ToCanonical(b, rigid) : spatialblend::FromCanonical(b, rigid);
  WriteBonePose(converted, out);
  out->flags = in->flags;
  out->epoch = in->epoch;
  return 0;
}

int32_t sb_embodiment_anchor_pose_from_bone(sb_space_t space, sb_actor_t actor, const char* anchor_name,
                                            const sb_bone_pose_t* bone, sb_bone_pose_t* anchor_out) {
  auto* info = GetSpace(space);
  spatialblend::RigidTransform rigid;
  if (!info || !info->space || !anchor_name || !anchor_out || !ReadBonePose(bone, &rigid)) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  const auto* declared = e ? e->FindAnchor(anchor_name) : nullptr;
  if (!declared || declared->bone < 0) return -1;
  WriteBonePose(spatialblend::ComposeAnchor(*declared, rigid), anchor_out);
  return 0;
}

int32_t sb_embodiment_bone_pose_from_anchor(sb_space_t space, sb_actor_t actor, const char* anchor_name,
                                            const sb_bone_pose_t* anchor, sb_bone_pose_t* bone_out) {
  auto* info = GetSpace(space);
  spatialblend::RigidTransform rigid;
  if (!info || !info->space || !anchor_name || !bone_out || !ReadBonePose(anchor, &rigid)) return -1;
  const auto e = info->space->EmbodimentOf(static_cast<spatialblend::entity_id_t>(actor));
  const auto* declared = e ? e->FindAnchor(anchor_name) : nullptr;
  if (!declared || declared->bone < 0) return -1;
  WriteBonePose(spatialblend::BoneFromAnchor(*declared, rigid), bone_out);
  return 0;
}

namespace {

void FillAnchorProposal(const spatialblend::AnchorProposal& a,
                        sb_anchor_proposal_t* c) {
  std::memset(c, 0, sizeof(*c));
  c->struct_size = sizeof(sb_anchor_proposal_t);
  c->anchor_id = a.anchor_id;
  c->status = static_cast<int32_t>(a.status);
  c->cause = a.cause;
  c->flags = 0;
  if (!a.yaw_specified()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_YAW_UNSPECIFIED;
  if (a.engaged()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_ENGAGED;
  if (a.sampled()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_SAMPLED;
  if (a.arrived()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_ARRIVED;
  if (!a.position_specified())
    c->flags |= SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED;
  if (a.transfer()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_TRANSFER;
  if (a.inner_solve()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_INNER;
  if (a.has_facing_yaw()) c->flags |= SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW;
  c->x = a.x;
  c->y = a.y;
  c->z = a.z;
  c->orientation.struct_size = sizeof(sb_orientation_t);
  c->orientation.type = a.orientation_type;
  c->orientation.yaw = a.yaw;
  c->orientation.pitch = a.pitch;
  c->orientation.roll = a.roll;
  c->err_spacing_target = a.err_spacing_target;
  c->err_spacing_sample = a.err_spacing_sample;
  c->err_facing_target = a.err_facing_target;
  c->err_facing_sample = a.err_facing_sample;
  c->err_pose_target = a.err_pose_target;
  c->err_pose_sample = a.err_pose_sample;
  c->err_scale_m = a.err_scale_m;
  c->referent_shared_id = a.referent_shared_id;
}

}  // namespace

int32_t sb_read_proposals(sb_space_t space, sb_actor_t actor,
                          sb_proposal_set_t* set, sb_anchor_proposal_t* out,
                          int32_t cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !set ||
      set->struct_size != sizeof(sb_proposal_set_t) || cap < 0 ||
      (cap > 0 && !out))
    return -1;
  for (int32_t i = 0; i < cap; ++i)
    if (out[i].struct_size != sizeof(sb_anchor_proposal_t)) return -1;
  spatialblend::ReceiverProposal p;
  if (!info->space->ReadProposal(actor, &p)) return -1;
  const auto all = p.All();
  sb_proposal_set_t c{};
  c.struct_size = sizeof(sb_proposal_set_t);
  c.total_anchor_count = static_cast<uint32_t>(all.size());
  c.anchor_count = static_cast<uint32_t>(std::min<size_t>(all.size(), cap));
  c.sample_valid = p.sample_valid ? 1 : 0;
  c.revision = p.revision;
  c.source_sequence = p.source_sequence;
  c.sample_sequence = p.sample_sequence;
  c.sample_epoch = p.sample_epoch;
  c.sample_timestamp_s = p.sample_timestamp_s;
  c.sample_age_s = p.sample_age_s;
  for (uint32_t i = 0; i < c.anchor_count; ++i)
    FillAnchorProposal(*all[i], &out[i]);
  *set = c;
  return 0;
}

int32_t sb_read_anchor_solve(sb_space_t space, sb_actor_t actor, uint32_t anchor_id, sb_anchor_solve_t* out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out || out->struct_size != sizeof(sb_anchor_solve_t)) return -1;
  spatialblend::ReceiverProposal p;
  if (!info->space->ReadProposal(actor, &p)) return -1;
  const spatialblend::AnchorProposal* a = p.Find(anchor_id);
  if (!a) return 0;
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(sb_anchor_solve_t);
  out->anchor_id = anchor_id;
  if (a->inner_solve()) out->flags |= SB_ANCHOR_PROPOSAL_FLAG_INNER;
  if (a->has_facing_yaw()) out->flags |= SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW;
  out->iterations = a->inner.iterations;
  out->object_shared_id = a->inner.object_shared_id;
  out->sd_scale_m = a->inner.scale_m;
  out->objective_j = a->inner.j;
  out->facing_yaw = a->facing_yaw;
  std::strncpy(out->driving_pose_tag, a->inner.driving_tag.c_str(), sizeof(out->driving_pose_tag) - 1);
  return 1;
}

int32_t sb_read_anchor_turn(sb_space_t space, sb_actor_t actor, uint32_t anchor_id, sb_anchor_turn_t* out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out || out->struct_size != sizeof(sb_anchor_turn_t)) return -1;
  spatialblend::ReceiverProposal p;
  if (!info->space->ReadProposal(actor, &p)) return -1;
  const spatialblend::AnchorProposal* a = p.Find(anchor_id);
  if (!a) return 0;
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(sb_anchor_turn_t);
  out->anchor_id = anchor_id;
  sb_anchor_proposal_t c;
  FillAnchorProposal(*a, &c);  // the same SB_ANCHOR_PROPOSAL_FLAG_* mapping as sb_read_proposals
  out->flags = c.flags;
  out->yaw = a->yaw;
  out->yaw_turn = a->yaw_turn;
  return 1;
}

int32_t sb_read_anchor_chart(sb_space_t space, sb_actor_t actor, uint32_t anchor_id, sb_anchor_chart_t* out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out || out->struct_size != sizeof(sb_anchor_chart_t)) return -1;
  spatialblend::ReceiverProposal p;
  if (!info->space->ReadProposal(actor, &p)) return -1;
  const spatialblend::AnchorProposal* a = p.Find(anchor_id);
  if (!a || !a->has_chart) return 0;
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(sb_anchor_chart_t);
  out->anchor_id = anchor_id;
  out->flags = a->chart_verified ? SB_ANCHOR_CHART_FLAG_VERIFIED : 0u;
  out->leaf_shared_id = a->chart_leaf_shared_id;
  out->x = a->chart_x;
  out->y = a->chart_y;
  out->z = a->chart_z;
  out->orientation.struct_size = sizeof(sb_orientation_t);
  out->orientation.type = a->chart_orientation_type;
  out->orientation.yaw = a->chart_yaw;
  out->orientation.pitch = a->chart_pitch;
  out->orientation.roll = a->chart_roll;
  out->revision = p.revision;
  return 1;
}

int32_t sb_raise_source_exception(sb_space_t space, sb_actor_t source_actor,
                                  uint32_t exception_id, const void* payload,
                                  uint32_t payload_size) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return -1;
  return info->space->RaiseSourceException(source_actor, exception_id, payload,
                                           payload_size)
             ? 0
             : -1;
}

int32_t sb_set_target_event_delivery(sb_space_t space, sb_actor_t actor,
                                     int32_t enabled) {
  auto* info = GetSpace(space);
  if (!info || !info->space || (enabled != 0 && enabled != 1)) return -1;
  return info->space->SetTargetEventDelivery(actor, enabled != 0) ? 0 : -1;
}

int32_t sb_poll_actor_event(sb_space_t space, sb_actor_t actor,
                            sb_actor_event_t* out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out ||
      out->struct_size != sizeof(sb_actor_event_t))
    return -1;
  spatialblend::Space::ActorEvent e;
  const int result = info->space->PollActorEventAfterSolve(actor, &e);
  if (result != 1) return result;
  sb_actor_event_t c{};
  c.struct_size = sizeof(sb_actor_event_t);
  c.kind = static_cast<int32_t>(e.kind);
  c.revision = e.revision;
  c.source_sequence = e.source_sequence;
  c.source_shared_id = e.source_shared_id;
  c.exception_id = e.exception_id;
  c.payload_size =
      static_cast<uint32_t>(std::min<size_t>(e.payload.size(), SB_EXCEPTION_PAYLOAD_MAX));
  if (c.payload_size > 0) std::memcpy(c.payload, e.payload.data(), c.payload_size);
  if (e.kind == spatialblend::Space::ActorEvent::Kind::kPositionRequest ||
      e.kind == spatialblend::Space::ActorEvent::Kind::kYawRequest ||
      e.kind == spatialblend::Space::ActorEvent::Kind::kTargetRelease) {
    const sb_target_request_payload_t request = {
        sizeof(sb_target_request_payload_t),
        e.channels,
        e.sample_epoch,
        e.sample_sequence,
        e.sample_yaw,
        e.yaw_turn,
        e.proposal_flags,
        e.proposal_cause,
        e.facing_yaw,
        static_cast<int32_t>(e.proposal_status),
        e.request_flags};
    static_assert(sizeof(request) == 64 &&
                  sizeof(request) <= SB_EXCEPTION_PAYLOAD_MAX);
    c.payload_size = sizeof(request);
    std::memcpy(c.payload, &request, sizeof(request));
  }
  c.interaction_kind = static_cast<int32_t>(e.interaction_kind);
  c.anchor_id = e.anchor_id;
  c.x = e.x;
  c.y = e.y;
  c.z = e.z;
  c.orientation.struct_size = sizeof(sb_orientation_t);
  c.orientation.type = e.orientation_type;
  c.orientation.yaw = e.yaw;
  c.orientation.pitch = e.pitch;
  c.orientation.roll = e.roll;
  *out = c;
  return 1;
}

int32_t sb_query_course(sb_space_t space, sb_actor_t actor, double from_x,
                        double from_y, double to_x, double to_y,
                        double* xy_out, int32_t cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !xy_out || cap < 2 ||
      !info->space->HasActor(actor))
    return -1;
  const auto course = info->space->QueryCourse(
      actor, Eigen::Vector2d(from_x, from_y), Eigen::Vector2d(to_x, to_y));
  const int32_t n = static_cast<int32_t>(course.size());
  for (int32_t i = 0; i < n && i < cap; ++i) {
    xy_out[2 * i] = course[i].x();
    xy_out[2 * i + 1] = course[i].y();
  }
  return n;
}

int32_t sb_query_nearest_clear_point(sb_space_t space, sb_actor_t actor, double x, double y, double* xy_out) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !xy_out || !std::isfinite(x) || !std::isfinite(y) || !info->space->HasActor(actor))
    return -1;
  const auto p = info->space->QueryNearestClearPoint(actor, Eigen::Vector2d(x, y), {}, {});
  if (!p) return 0;
  xy_out[0] = p->x();
  xy_out[1] = p->y();
  return 1;
}

int32_t sb_set_body_radius(sb_space_t space, double radius_m) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !std::isfinite(radius_m) || radius_m < 0.0)
    return -1;
  info->space->SetBodyRadius(radius_m);
  return 0;
}

// ----------------------------------------------------------------------------
// Binding
// ----------------------------------------------------------------------------

int32_t sb_bind(sb_space_t space, sb_actor_t actor, const char* anchor_name,
                sb_kernel_type_t kernel_type) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name || !IsValidKernelType(kernel_type))
    return 0;
  return info->space->BindOwned(static_cast<spatialblend::entity_id_t>(actor),
                                anchor_name, ToKernelType(kernel_type))
             ? 1
             : 0;
}

int32_t sb_bind_affordance(sb_space_t space, sb_actor_t actor,
                           const char* anchor_name, const char* tag) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name) return 0;
  return info->space->BindOwned(
             static_cast<spatialblend::entity_id_t>(actor), anchor_name,
             spatialblend::RelationKernelType::kAvatarAffordancePose,
             spatialblend::kInvalidElementId, tag ? std::string(tag) : std::string())
             ? 1
             : 0;
}

// ----------------------------------------------------------------------------
// Realization target callback
// ----------------------------------------------------------------------------

void sb_register_on_targets(sb_space_t space, sb_on_targets_t callback,
                            void* user_data) {
  auto* info = GetSpace(space);
  if (!info) return;
  std::lock_guard<std::mutex> lock(info->callback_mutex);
  info->on_targets_callback = callback;
  info->on_targets_user_data = user_data;
}

void sb_register_on_element_added(sb_space_t space,
                                  sb_on_element_added_t callback,
                                  void* user_data) {
  auto* info = GetSpace(space);
  if (!info) return;
  std::lock_guard<std::mutex> lock(info->callback_mutex);
  info->on_element_added_callback = callback;
  info->on_element_added_user_data = user_data;
}
void sb_register_on_relation_event(sb_space_t space,
                                   sb_on_relation_event_t callback,
                                   void* user_data) {
  auto* info = GetSpace(space);
  // Registering takes the space lock, which is held while a relation event
  // callback runs: from inside one it would deadlock.
  if (g_inside_relation_callback || !info || !info->space) return;
  info->on_relation_event_callback = callback;
  info->on_relation_event_user_data = user_data;
  // Capture `info` (the SpaceInfo*, which outlives this registration --
  // it's only destroyed at the end of sb_destroy_session/sb_cleanup_all,
  // well after this Space is torn down) and re-read
  // info->on_relation_event_callback/_user_data at INVOCATION time, rather
  // than capturing `callback`/`user_data` by value here. Space::OnRemoteUpdate
  // (called from SubscribeStreamCall::Proceed() on the RemoteHub's own CQ
  // thread when a relation update arrives from the host) can invoke this
  // registered callback WHILE sb_destroy_session's teardown for this same
  // session is still in flight on another thread -- RemoteHub::Disconnect()
  // legitimately waits (bounded drain) for exactly this in-flight call to
  // finish, so it is NOT itself a bug for this lambda to still fire during
  // that window. The bug this fixes: capturing `callback`/`user_data` BY
  // VALUE meant the ORIGINAL, possibly-now-dangling callback/user_data (e.g.
  // a harness fixture's own per-space state, freed by the caller's local
  // destruction order once its own scope exits, independent of when this
  // network drain completes) would still be invoked regardless of whether
  // the space had already been logically torn down -- observed as a "mutex
  // lock failed: Invalid argument" abort inside harness_test's own
  // OnRelationEventCb, called from exactly this path. Re-reading through
  // `info` lets sb_destroy_session/sb_cleanup_all null out
  // on_relation_event_callback FIRST (see below) so any such late-arriving
  // event becomes a safe no-op instead of reaching into torn-down state. This
  // mirrors the existing, already-safe on_element_added_callback pattern in
  // sb_add_space() above, which re-reads through `info_ptr` the same way.
  info->space->RegisterOnRelationEvent(
      [info](const spatialblend::RelationEvent& ev) {
        sb_on_relation_event_t cb = info->on_relation_event_callback;
        if (!cb) return;
        void* ud = info->on_relation_event_user_data;
        sb_relation_event_t c_ev;
        c_ev.actor_shared_id = ev.actor_shared_id;
        // anchor_name is valid only for the duration of this callback.
        c_ev.anchor_name = ev.anchor_name.c_str();
        c_ev.referent_shared_id = ev.referent_shared_id;
        // C++ RelationKernelType is Facing=0, Spacing=1. The C API is the
        // reverse (SB_KERNEL_SPACING=0, SB_KERNEL_FACING=1). A raw cast
        // labels a Spacing event as Facing.
        c_ev.kernel_type = FromKernelType(ev.kernel_type);
        c_ev.from_phase = static_cast<sb_relation_phase_t>(ev.from);
        c_ev.to_phase = static_cast<sb_relation_phase_t>(ev.to);
        c_ev.timestamp = ev.timestamp;
        struct CallbackScope {
          bool previous = g_inside_relation_callback;
          CallbackScope() { g_inside_relation_callback = true; }
          ~CallbackScope() { g_inside_relation_callback = previous; }
        } callback_scope;
        cb(&c_ev, ud);
      });
}

int32_t sb_poll_realization_targets(sb_space_t space, sb_actor_t actor) {
  auto* info = GetSpace(space);
  return DispatchRealizationTargets(space, info, actor);
}

uint64_t sb_read_rich_monotonic_ns(void) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

double sb_read_rich_monotonic_seconds(void) {
  return static_cast<double>(sb_read_rich_monotonic_ns()) / 1e9;
}

uint32_t sb_get_rich_realization_feature(sb_rich_realization_feature_t* feature,
                                         uint32_t feature_size) {
  const uint32_t required = sizeof(sb_rich_realization_feature_t);
  if (!feature || feature_size < required) return required;
  sb_rich_realization_feature_t value{};
  value.descriptor_size = required;
  value.feature_version = SB_RICH_REALIZATION_FEATURE_VERSION;
  value.snapshot_size = sizeof(sb_rich_realization_snapshot_t);
  value.channel_size = sizeof(sb_rich_realization_channel_t);
  value.contributor_size = sizeof(sb_rich_realization_contributor_t);
  value.copy_size = sizeof(sb_rich_realization_copy_t);
  value.snapshot_channels_offset =
      offsetof(sb_rich_realization_snapshot_t, channels);
  value.snapshot_contributors_offset =
      offsetof(sb_rich_realization_snapshot_t, contributors);
  value.snapshot_utf8_offset = offsetof(sb_rich_realization_snapshot_t, utf8);
  value.snapshot_channel_count_offset =
      offsetof(sb_rich_realization_snapshot_t, channel_count);
  value.snapshot_contributor_count_offset =
      offsetof(sb_rich_realization_snapshot_t, contributor_count);
  value.snapshot_utf8_bytes_offset =
      offsetof(sb_rich_realization_snapshot_t, utf8_bytes);
  value.copy_channels_offset = offsetof(sb_rich_realization_copy_t, channels);
  value.copy_contributors_offset =
      offsetof(sb_rich_realization_copy_t, contributors);
  value.copy_utf8_offset = offsetof(sb_rich_realization_copy_t, utf8);
  value.copy_header_offset = offsetof(sb_rich_realization_copy_t, header);
  value.channel_anchor_offset_offset =
      offsetof(sb_rich_realization_channel_t, anchor_offset);
  value.channel_contributor_offset_offset =
      offsetof(sb_rich_realization_channel_t, contributor_offset);
  value.channel_has_z_offset = offsetof(sb_rich_realization_channel_t, has_z);
  value.channel_requested_position_offset =
      offsetof(sb_rich_realization_channel_t, requested_position_x);
  value.contributor_reason_offset_offset =
      offsetof(sb_rich_realization_contributor_t, reason_offset);
  value.contributor_support_identifier_offset =
      offsetof(sb_rich_realization_contributor_t, support_identifier_offset);
  value.contributor_support_plane_to_world_offset = offsetof(
      sb_rich_realization_contributor_t, support_plane_to_world_row_major);
  value.max_channels = kRichMaxChannels;
  value.max_contributors = kRichMaxContributors;
  value.max_utf8_bytes = kRichMaxUtf8Bytes;
  value.coordinate_convention = 1;
  value.monotonic_clock_id = SB_RICH_REALIZATION_CLOCK_ID;
  value.monotonic_clock_epoch = RichClockEpoch();
  *feature = value;
  return required;
}

uint32_t sb_get_rich_realization_field_count(uint32_t struct_kind) {
  switch (struct_kind) {
    case SB_RICH_LAYOUT_FEATURE:
      return 29;
    case SB_RICH_LAYOUT_HEADER:
      return 28;
    case SB_RICH_LAYOUT_CHANNEL:
      return 22;
    case SB_RICH_LAYOUT_CONTRIBUTOR:
      return 20;
    case SB_RICH_LAYOUT_SNAPSHOT:
      return 8;
    case SB_RICH_LAYOUT_COPY:
      return 12;
    default:
      return 0;
  }
}

uint32_t sb_get_rich_realization_field_offset(uint32_t kind, uint32_t i) {
#define RICH_OFF(T, ...) \
  {                      \
    __VA_ARGS__          \
  }
  // Keep each list in declaration order.  Array fields appear once: the
  // feature's size + this complete table is the layout contract, including
  // every pointer and every fixed array's starting offset.
  static const uint32_t feature[] = RICH_OFF(
      sb_rich_realization_feature_t,
      offsetof(sb_rich_realization_feature_t, descriptor_size),
      offsetof(sb_rich_realization_feature_t, feature_version),
      offsetof(sb_rich_realization_feature_t, snapshot_size),
      offsetof(sb_rich_realization_feature_t, channel_size),
      offsetof(sb_rich_realization_feature_t, contributor_size),
      offsetof(sb_rich_realization_feature_t, copy_size),
      offsetof(sb_rich_realization_feature_t, snapshot_channels_offset),
      offsetof(sb_rich_realization_feature_t, snapshot_contributors_offset),
      offsetof(sb_rich_realization_feature_t, snapshot_utf8_offset),
      offsetof(sb_rich_realization_feature_t, snapshot_channel_count_offset),
      offsetof(sb_rich_realization_feature_t,
               snapshot_contributor_count_offset),
      offsetof(sb_rich_realization_feature_t, snapshot_utf8_bytes_offset),
      offsetof(sb_rich_realization_feature_t, copy_channels_offset),
      offsetof(sb_rich_realization_feature_t, copy_contributors_offset),
      offsetof(sb_rich_realization_feature_t, copy_utf8_offset),
      offsetof(sb_rich_realization_feature_t, copy_header_offset),
      offsetof(sb_rich_realization_feature_t, channel_anchor_offset_offset),
      offsetof(sb_rich_realization_feature_t,
               channel_contributor_offset_offset),
      offsetof(sb_rich_realization_feature_t, channel_has_z_offset),
      offsetof(sb_rich_realization_feature_t,
               channel_requested_position_offset),
      offsetof(sb_rich_realization_feature_t, contributor_reason_offset_offset),
      offsetof(sb_rich_realization_feature_t,
               contributor_support_identifier_offset),
      offsetof(sb_rich_realization_feature_t,
               contributor_support_plane_to_world_offset),
      offsetof(sb_rich_realization_feature_t, max_channels),
      offsetof(sb_rich_realization_feature_t, max_contributors),
      offsetof(sb_rich_realization_feature_t, max_utf8_bytes),
      offsetof(sb_rich_realization_feature_t, coordinate_convention),
      offsetof(sb_rich_realization_feature_t, monotonic_clock_id),
      offsetof(sb_rich_realization_feature_t, monotonic_clock_epoch));
  static const uint32_t header[] = RICH_OFF(
      sb_rich_realization_header_t,
      offsetof(sb_rich_realization_header_t, producer_lifetime),
      offsetof(sb_rich_realization_header_t, revision),
      offsetof(sb_rich_realization_header_t, monotonic_clock_id),
      offsetof(sb_rich_realization_header_t, monotonic_clock_epoch),
      offsetof(sb_rich_realization_header_t, receiver_ingress_monotonic_ns),
      offsetof(sb_rich_realization_header_t, receiver_solve_monotonic_ns),
      offsetof(sb_rich_realization_header_t, source_epoch),
      offsetof(sb_rich_realization_header_t, source_sequence),
      offsetof(sb_rich_realization_header_t, source_observation_sequence),
      offsetof(sb_rich_realization_header_t, source_actor_space),
      offsetof(sb_rich_realization_header_t, source_actor_element),
      offsetof(sb_rich_realization_header_t, relay_source_space),
      offsetof(sb_rich_realization_header_t, receiver_space),
      offsetof(sb_rich_realization_header_t, receiver_actor),
      offsetof(sb_rich_realization_header_t, policy_revision),
      offsetof(sb_rich_realization_header_t, tree_revision),
      offsetof(sb_rich_realization_header_t, counterpart_revision),
      offsetof(sb_rich_realization_header_t, support_revision),
      offsetof(sb_rich_realization_header_t, source_time_seconds),
      offsetof(sb_rich_realization_header_t, source_clock_seconds),
      offsetof(sb_rich_realization_header_t, status),
      offsetof(sb_rich_realization_header_t, source_timing_mode),
      offsetof(sb_rich_realization_header_t, source_time_known),
      offsetof(sb_rich_realization_header_t, source_clock_known),
      offsetof(sb_rich_realization_header_t, source_observation_sequence_known),
      offsetof(sb_rich_realization_header_t, known_revision_mask),
      offsetof(sb_rich_realization_header_t, unknown_revision_mask),
      offsetof(sb_rich_realization_header_t, failure_code));
  static const uint32_t channel[] =
      RICH_OFF(sb_rich_realization_channel_t,
               offsetof(sb_rich_realization_channel_t, anchor_offset),
               offsetof(sb_rich_realization_channel_t, anchor_length),
               offsetof(sb_rich_realization_channel_t, contributor_offset),
               offsetof(sb_rich_realization_channel_t, contributor_count),
               offsetof(sb_rich_realization_channel_t, family),
               offsetof(sb_rich_realization_channel_t, solver_status),
               offsetof(sb_rich_realization_channel_t, geometry),
               offsetof(sb_rich_realization_channel_t, goal_kind),
               offsetof(sb_rich_realization_channel_t, has_goal),
               offsetof(sb_rich_realization_channel_t, has_z),
               offsetof(sb_rich_realization_channel_t, has_requested_position),
               offsetof(sb_rich_realization_channel_t, reach_projected),
               offsetof(sb_rich_realization_channel_t, referent_id),
               offsetof(sb_rich_realization_channel_t, arrival_seconds),
               offsetof(sb_rich_realization_channel_t, position_x),
               offsetof(sb_rich_realization_channel_t, position_y),
               offsetof(sb_rich_realization_channel_t, position_z),
               offsetof(sb_rich_realization_channel_t, yaw),
               offsetof(sb_rich_realization_channel_t, weight),
               offsetof(sb_rich_realization_channel_t, reach_displacement_m),
               offsetof(sb_rich_realization_channel_t, requested_position_x),
               offsetof(sb_rich_realization_channel_t, requested_position_y));
  static const uint32_t contributor[] = RICH_OFF(
      sb_rich_realization_contributor_t,
      offsetof(sb_rich_realization_contributor_t, referent_id),
      offsetof(sb_rich_realization_contributor_t, source_space),
      offsetof(sb_rich_realization_contributor_t, source_element),
      offsetof(sb_rich_realization_contributor_t, resolved_local_element),
      offsetof(sb_rich_realization_contributor_t, binding_revision),
      offsetof(sb_rich_realization_contributor_t, snapshot_ordinal),
      offsetof(sb_rich_realization_contributor_t, disposition),
      offsetof(sb_rich_realization_contributor_t, grounding_mode),
      offsetof(sb_rich_realization_contributor_t, has_mapped_support),
      offsetof(sb_rich_realization_contributor_t, chart_version_known),
      offsetof(sb_rich_realization_contributor_t, binding_revision_known),
      offsetof(sb_rich_realization_contributor_t, reason_offset),
      offsetof(sb_rich_realization_contributor_t, reason_length),
      offsetof(sb_rich_realization_contributor_t, support_identifier_offset),
      offsetof(sb_rich_realization_contributor_t, support_identifier_length),
      offsetof(sb_rich_realization_contributor_t, applied_weight),
      offsetof(sb_rich_realization_contributor_t, mapped_support_x),
      offsetof(sb_rich_realization_contributor_t, mapped_support_y),
      offsetof(sb_rich_realization_contributor_t, mapped_support_z),
      offsetof(sb_rich_realization_contributor_t,
               support_plane_to_world_row_major));
  static const uint32_t snapshot[] =
      RICH_OFF(sb_rich_realization_snapshot_t,
               offsetof(sb_rich_realization_snapshot_t, header),
               offsetof(sb_rich_realization_snapshot_t, channels),
               offsetof(sb_rich_realization_snapshot_t, contributors),
               offsetof(sb_rich_realization_snapshot_t, utf8),
               offsetof(sb_rich_realization_snapshot_t, channel_count),
               offsetof(sb_rich_realization_snapshot_t, contributor_count),
               offsetof(sb_rich_realization_snapshot_t, utf8_bytes),
               offsetof(sb_rich_realization_snapshot_t, reserved));
  static const uint32_t copy[] =
      RICH_OFF(sb_rich_realization_copy_t,
               offsetof(sb_rich_realization_copy_t, struct_size),
               offsetof(sb_rich_realization_copy_t, channel_capacity),
               offsetof(sb_rich_realization_copy_t, contributor_capacity),
               offsetof(sb_rich_realization_copy_t, utf8_capacity),
               offsetof(sb_rich_realization_copy_t, channels),
               offsetof(sb_rich_realization_copy_t, contributors),
               offsetof(sb_rich_realization_copy_t, utf8),
               offsetof(sb_rich_realization_copy_t, header),
               offsetof(sb_rich_realization_copy_t, required_channels),
               offsetof(sb_rich_realization_copy_t, required_contributors),
               offsetof(sb_rich_realization_copy_t, required_utf8_bytes),
               offsetof(sb_rich_realization_copy_t, reserved));
#undef RICH_OFF
  const uint32_t* values = nullptr;
  uint32_t count = 0;
  switch (kind) {
    case SB_RICH_LAYOUT_FEATURE:
      values = feature;
      count = 29;
      break;
    case SB_RICH_LAYOUT_HEADER:
      values = header;
      count = 28;
      break;
    case SB_RICH_LAYOUT_CHANNEL:
      values = channel;
      count = 22;
      break;
    case SB_RICH_LAYOUT_CONTRIBUTOR:
      values = contributor;
      count = 20;
      break;
    case SB_RICH_LAYOUT_SNAPSHOT:
      values = snapshot;
      count = 8;
      break;
    case SB_RICH_LAYOUT_COPY:
      values = copy;
      count = 12;
      break;
    default:
      return UINT32_MAX;
  }
  return i < count ? values[i] : UINT32_MAX;
}

uint32_t sb_get_rich_realization_validity_feature(
    sb_rich_realization_validity_feature_t* feature, uint32_t feature_size) {
  const uint32_t required = sizeof(sb_rich_realization_validity_feature_t);
  if (!feature || feature_size < required) return required;
  sb_rich_realization_validity_feature_t value{};
  value.descriptor_size = required;
  value.feature_version = SB_RICH_REALIZATION_VALIDITY_FEATURE_VERSION;
  value.validity_size = sizeof(sb_rich_realization_validity_t);
  value.supported_domain_mask = SB_RICH_VALIDITY_DOMAIN_ALL;
  value.validity_producer_lifetime_offset =
      offsetof(sb_rich_realization_validity_t, producer_lifetime);
  value.validity_revision_offset =
      offsetof(sb_rich_realization_validity_t, revision);
  value.validity_captured_known_mask_offset =
      offsetof(sb_rich_realization_validity_t, captured_known_mask);
  value.validity_current_known_mask_offset =
      offsetof(sb_rich_realization_validity_t, current_known_mask);
  value.validity_changed_mask_offset =
      offsetof(sb_rich_realization_validity_t, changed_mask);
  value.validity_unchecked_mask_offset =
      offsetof(sb_rich_realization_validity_t, unchecked_mask);
  value.validity_valid_offset = offsetof(sb_rich_realization_validity_t, valid);
  *feature = value;
  return required;
}

uint32_t sb_get_rich_realization_validity_field_count(uint32_t struct_kind) {
  switch (struct_kind) {
    case SB_RICH_VALIDITY_LAYOUT_FEATURE:
      return 12;
    case SB_RICH_VALIDITY_LAYOUT_RESULT:
      return 16;
    default:
      return 0;
  }
}

uint32_t sb_get_rich_realization_validity_field_offset(uint32_t kind,
                                                        uint32_t i) {
#define RICH_VALIDITY_OFF(...) { __VA_ARGS__ }
  static const uint32_t feature[] = RICH_VALIDITY_OFF(
      offsetof(sb_rich_realization_validity_feature_t, descriptor_size),
      offsetof(sb_rich_realization_validity_feature_t, feature_version),
      offsetof(sb_rich_realization_validity_feature_t, validity_size),
      offsetof(sb_rich_realization_validity_feature_t, supported_domain_mask),
      offsetof(sb_rich_realization_validity_feature_t,
               validity_producer_lifetime_offset),
      offsetof(sb_rich_realization_validity_feature_t, validity_revision_offset),
      offsetof(sb_rich_realization_validity_feature_t,
               validity_captured_known_mask_offset),
      offsetof(sb_rich_realization_validity_feature_t,
               validity_current_known_mask_offset),
      offsetof(sb_rich_realization_validity_feature_t, validity_changed_mask_offset),
      offsetof(sb_rich_realization_validity_feature_t,
               validity_unchecked_mask_offset),
      offsetof(sb_rich_realization_validity_feature_t, validity_valid_offset),
      offsetof(sb_rich_realization_validity_feature_t, reserved));
  static const uint32_t result[] = RICH_VALIDITY_OFF(
      offsetof(sb_rich_realization_validity_t, producer_lifetime),
      offsetof(sb_rich_realization_validity_t, revision),
      offsetof(sb_rich_realization_validity_t, captured_policy_revision),
      offsetof(sb_rich_realization_validity_t, captured_tree_revision),
      offsetof(sb_rich_realization_validity_t, captured_counterpart_revision),
      offsetof(sb_rich_realization_validity_t, captured_support_revision),
      offsetof(sb_rich_realization_validity_t, current_policy_revision),
      offsetof(sb_rich_realization_validity_t, current_tree_revision),
      offsetof(sb_rich_realization_validity_t, current_counterpart_revision),
      offsetof(sb_rich_realization_validity_t, current_support_revision),
      offsetof(sb_rich_realization_validity_t, requested_domain_mask),
      offsetof(sb_rich_realization_validity_t, captured_known_mask),
      offsetof(sb_rich_realization_validity_t, current_known_mask),
      offsetof(sb_rich_realization_validity_t, changed_mask),
      offsetof(sb_rich_realization_validity_t, unchecked_mask),
      offsetof(sb_rich_realization_validity_t, valid));
#undef RICH_VALIDITY_OFF
  const uint32_t* fields = nullptr;
  uint32_t count = 0;
  if (kind == SB_RICH_VALIDITY_LAYOUT_FEATURE) {
    fields = feature;
    count = 12;
  } else if (kind == SB_RICH_VALIDITY_LAYOUT_RESULT) {
    fields = result;
    count = 16;
  } else {
    return UINT32_MAX;
  }
  return i < count ? fields[i] : UINT32_MAX;
}

sb_rich_realization_token_t sb_activate_rich_realization(
    sb_space_t space, sb_actor_t actor, uint32_t requested_feature_version) {
  if (requested_feature_version != SB_RICH_REALIZATION_FEATURE_VERSION)
    return 0;
  auto* info = GetSpace(space);
  if (!info || !info->space || actor == 0 || !info->space->HasActor(actor))
    return 0;
  const sb_session_t session = SessionForSpace(space);
  {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    const auto session_it = g_sessions.find(session);
    if (session_it == g_sessions.end() || session_it->second->rich_retiring)
      return 0;
  }
  const sb_rich_realization_token_t token = g_next_rich_lifetime.fetch_add(1);
  if (token == 0) return 0;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    if (g_rich_activations.size() >= kRichMaxActivations) return 0;
    // One active native lifetime per receiver actor.  This prevents a retiring
    // sibling token from disabling/capturing a newly activated actor state.
    for (const auto& [_, active] : g_rich_activations) {
      if (active.space == space && active.actor == actor) return 0;
    }
    g_rich_activations.emplace(
        token, RichActivation{space, actor, true, 0, nullptr,
                              std::make_shared<RichDeliveryState>()});
  }
  // Actor-scoped: this cannot alter legacy-facing behavior for a sibling actor.
  info->space->SetActorRichRealizationEnabled(actor, true);
  return token;
}

void sb_register_on_rich_realization(sb_space_t space,
                                     sb_on_rich_realization_t callback,
                                     void* user_data) {
  auto* info = GetSpace(space);
  if (!info) return;
  std::lock_guard<std::mutex> lock(info->callback_mutex);
  info->on_rich_callback = callback;
  info->on_rich_user_data = user_data;
}

int32_t sb_poll_rich_realization(sb_rich_realization_token_t token,
                                 sb_rich_realization_copy_t* out) {
  if (!out || out->struct_size != sizeof(sb_rich_realization_copy_t))
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  RichActivation activation_copy;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    activation_copy = it->second;
  }
  // Poll is a real transport-independent pull boundary: it refreshes from a
  // mutex-protected Core copy even when a transport did not use the legacy C
  // peer-dispatch seam.  It never invokes callbacks.
  if (activation_copy.active) {
    if (auto* info = GetSpace(activation_copy.space)) {
      spatialblend::ActorRealizationSnapshot core_snapshot;
      if (info->space && info->space->CopyActorRealizationSnapshot(
                             activation_copy.actor, &core_snapshot)) {
        std::lock_guard<std::mutex> lock(g_rich_mutex);
        auto it = g_rich_activations.find(token);
        if (it != g_rich_activations.end() && it->second.active &&
            (!it->second.snapshot ||
             core_snapshot.revision > it->second.snapshot->header.revision)) {
          it->second.snapshot =
              FreezeRichSnapshot(core_snapshot, token, activation_copy.space,
                                 activation_copy.actor);
        }
      }
    }
  }
  std::shared_ptr<const FrozenRichSnapshot> snapshot;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    snapshot = it->second.snapshot;
    if (!snapshot ||
        snapshot->header.revision == it->second.last_polled_revision)
      return SB_RICH_RESULT_NO_NEW_SNAPSHOT;
  }
  out->required_channels = static_cast<uint32_t>(snapshot->channels.size());
  out->required_contributors =
      static_cast<uint32_t>(snapshot->contributors.size());
  out->required_utf8_bytes = static_cast<uint32_t>(snapshot->utf8.size());
  if (out->channel_capacity < out->required_channels ||
      out->contributor_capacity < out->required_contributors ||
      out->utf8_capacity < out->required_utf8_bytes ||
      (out->required_channels && !out->channels) ||
      (out->required_contributors && !out->contributors) ||
      (out->required_utf8_bytes && !out->utf8)) {
    return SB_RICH_RESULT_BUFFER_TOO_SMALL;
  }
  if (out->required_channels)
    std::memcpy(out->channels, snapshot->channels.data(),
                out->required_channels * sizeof(*out->channels));
  if (out->required_contributors)
    std::memcpy(out->contributors, snapshot->contributors.data(),
                out->required_contributors * sizeof(*out->contributors));
  if (out->required_utf8_bytes)
    std::memcpy(out->utf8, snapshot->utf8.data(), out->required_utf8_bytes);
  out->header = snapshot->header;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    auto it = g_rich_activations.find(token);
    if (it != g_rich_activations.end() && it->second.snapshot == snapshot)
      it->second.last_polled_revision = snapshot->header.revision;
  }
  return SB_RICH_RESULT_OK;
}

int32_t sb_get_rich_realization_validity(
    sb_rich_realization_token_t token, uint64_t revision,
    uint32_t requested_domain_mask, sb_rich_realization_validity_t* out,
    uint32_t out_size) {
  if (!out || out_size < sizeof(*out) || requested_domain_mask == 0 ||
      (requested_domain_mask & ~SB_RICH_VALIDITY_DOMAIN_ALL) != 0) {
    return SB_RICH_RESULT_INVALID_ARGUMENT;
  }
  *out = {};
  RichActivation activation;
  std::shared_ptr<const FrozenRichSnapshot> snapshot;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    if (!it->second.active) return SB_RICH_RESULT_UNAVAILABLE;
    snapshot = it->second.snapshot;
    if (!snapshot || snapshot->header.revision != revision)
      return SB_RICH_RESULT_STALE_SNAPSHOT;
    activation = it->second;
  }
  auto* info = GetSpace(activation.space);
  if (!info || !info->space) return SB_RICH_RESULT_STALE_SNAPSHOT;
  spatialblend::ActorRealizationValidity core;
  if (!info->space->GetActorRealizationValidity(activation.actor, revision,
                                                requested_domain_mask, &core)) {
    return SB_RICH_RESULT_STALE_SNAPSHOT;
  }
  // A newer solve, retirement, or reactivation fence must win over this
  // current-state read.  Do not report current validity for another revision.
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    const auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end()) return SB_RICH_RESULT_UNKNOWN_TOKEN;
    if (!it->second.active || it->second.snapshot != snapshot ||
        it->second.snapshot->header.revision != revision) {
      return SB_RICH_RESULT_STALE_SNAPSHOT;
    }
  }
  const auto& h = snapshot->header;
  out->producer_lifetime = token;
  out->revision = revision;
  out->captured_policy_revision = h.policy_revision;
  out->captured_tree_revision = h.tree_revision;
  out->captured_counterpart_revision = h.counterpart_revision;
  out->captured_support_revision = h.support_revision;
  out->current_policy_revision = core.policy_revision;
  out->current_tree_revision = core.tree_revision;
  out->current_counterpart_revision = core.counterpart_revision;
  out->current_support_revision = core.support_revision;
  out->requested_domain_mask = requested_domain_mask;
  out->captured_known_mask = core.captured_known_mask;
  out->current_known_mask = core.current_known_mask;
  out->changed_mask = core.changed_mask;
  out->unchecked_mask = core.unchecked_mask |
                        SB_RICH_VALIDITY_DOMAIN_RECEIVER_INPUT;
  const uint32_t unavailable = requested_domain_mask &
      (out->unchecked_mask | ~out->captured_known_mask |
       ~out->current_known_mask);
  out->valid = unavailable == 0 &&
                       (out->changed_mask & requested_domain_mask) == 0
                   ? 1u
                   : 0u;
  return unavailable == 0 ? SB_RICH_RESULT_OK : SB_RICH_RESULT_UNAVAILABLE;
}

#include "spatialblend/api/c/receiver_input_c.inc"
#include "spatialblend/api/c/receiver_reevaluation_c.inc"
#include "spatialblend/api/c/planar_support_c.inc"
#include "spatialblend/api/c/containment_transfer_c.inc"
#include "spatialblend/api/c/source_evidence_c.inc"
#include "spatialblend/api/c/receiver_approach_context_c.inc"
#include "spatialblend/api/c/receiver_approach_feasibility_c.inc"

void sb_retire_rich_realization(sb_rich_realization_token_t token) {
  std::shared_ptr<FrozenRichSnapshot> terminal;
  sb_on_rich_realization_t callback = nullptr;
  void* user_data = nullptr;
  sb_space_t space = 0;
  sb_actor_t actor = 0;
  std::shared_ptr<RichDeliveryState> delivery;
  bool another_active_for_actor = false;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    auto it = g_rich_activations.find(token);
    if (it == g_rich_activations.end() || !it->second.active) return;
    RichActivation& activation = it->second;
    space = activation.space;
    actor = activation.actor;
    delivery = activation.delivery;
    terminal = std::make_shared<FrozenRichSnapshot>();
    terminal->header.producer_lifetime = token;
    terminal->header.revision =
        activation.snapshot ? activation.snapshot->header.revision + 1 : 1;
    terminal->header.monotonic_clock_id = SB_RICH_REALIZATION_CLOCK_ID;
    terminal->header.monotonic_clock_epoch = RichClockEpoch();
    terminal->header.receiver_space = space;
    terminal->header.receiver_actor = actor;
    // A local retirement has no source-side causal actor or relay.  Preserve
    // the C ABI's explicit invalid-handle sentinel rather than the
    // zero-initialized value, which managed consumers correctly interpret as
    // a known numeric identity.
    terminal->header.source_actor_space =
        static_cast<int64_t>(spatialblend::kInvalidSpaceId);
    terminal->header.source_actor_element =
        static_cast<int64_t>(spatialblend::kInvalidElementId);
    terminal->header.relay_source_space =
        static_cast<int64_t>(spatialblend::kInvalidSpaceId);
    terminal->header.status = SB_RICH_SNAPSHOT_EMPTY;
    terminal->header.source_timing_mode = SB_RICH_SOURCE_LEGACY_UNKNOWN;
    terminal->header.unknown_revision_mask =
        SB_RICH_REALIZATION_UNKNOWN_POLICY_REVISION |
        SB_RICH_REALIZATION_UNKNOWN_TREE_REVISION |
        SB_RICH_REALIZATION_UNKNOWN_COUNTERPART_REVISION |
        SB_RICH_REALIZATION_UNKNOWN_SUPPORT_REVISION;
    activation.snapshot = terminal;
    activation.active = false;
    for (const auto& [other_token, other] : g_rich_activations) {
      if (other_token != token && other.active && other.space == space &&
          other.actor == actor) {
        another_active_for_actor = true;
        break;
      }
    }
  }
  if (auto* info = GetSpace(space)) {
    // Capture registration after registry release; callback never runs under
    // either registry or Space mutex and may freely re-enter this API.
    {
      std::lock_guard<std::mutex> lock(info->callback_mutex);
      callback = info->on_rich_callback;
      user_data = info->on_rich_user_data;
    }
    if (!another_active_for_actor)
      info->space->SetActorRichRealizationEnabled(actor, false);
  }
  if (callback) {
    EnqueueRichDelivery(delivery, {terminal, callback, user_data, true});
  }
}

int32_t sb_release_rich_realization(sb_rich_realization_token_t token) {
  std::lock_guard<std::mutex> lock(g_rich_mutex);
  const auto it = g_rich_activations.find(token);
  if (it == g_rich_activations.end() || it->second.active) return 0;
  g_rich_activations.erase(it);
  return 1;
}

static void RetireRichActivationsForActor(sb_space_t space, sb_actor_t actor) {
  std::vector<sb_rich_realization_token_t> tokens;
  {
    std::lock_guard<std::mutex> lock(g_rich_mutex);
    for (const auto& [token, activation] : g_rich_activations) {
      if (activation.active && activation.space == space &&
          activation.actor == actor)
        tokens.push_back(token);
    }
  }
  for (const auto token : tokens) sb_retire_rich_realization(token);
}

static void RetireAndReleaseRichActivationsForSession(sb_session_t session) {
  std::vector<sb_rich_realization_token_t> tokens;
  {
    std::lock_guard<std::mutex> rich_lock(g_rich_mutex);
    std::lock_guard<std::mutex> spaces_lock(g_spaces_mutex);
    for (const auto& [token, activation] : g_rich_activations) {
      const auto owner = g_space_to_session.find(activation.space);
      if (owner != g_space_to_session.end() && owner->second == session)
        tokens.push_back(token);
    }
  }
  for (const auto token : tokens) {
    sb_retire_rich_realization(token);
    sb_release_rich_realization(token);
  }
}

static sb_normalized_strategy_t ToNormalizedStrategy(
    spatialblend::NormalizedPolygonPosition::Strategy strategy) {
  switch (strategy) {
    case spatialblend::NormalizedPolygonPosition::Strategy::kMVC:
      return SB_NORMALIZED_MVC;
    case spatialblend::NormalizedPolygonPosition::Strategy::kRadial:
    default:
      return SB_NORMALIZED_RADIAL;
  }
}

int32_t sb_get_relation_states(sb_space_t space, sb_actor_t actor,
                               sb_relation_state_t* out, int32_t cap) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !out || cap <= 0) return 0;

  // GetRelationStates takes the space lock. A relation-event callback runs
  // with that lock already held by this thread, so it reads through the
  // callback variant instead of deadlocking.
  std::vector<spatialblend::RelationState> states =
      g_inside_relation_callback
          ? info->space->GetRelationStatesFromCallback(
                static_cast<spatialblend::entity_id_t>(actor))
          : info->space->GetRelationStates(
                static_cast<spatialblend::entity_id_t>(actor));

  // anchor_name pointers must stay valid after this function returns (the
  // caller reads out[i].anchor_name after the call, not during a callback),
  // so keep the backing strings alive in thread-local storage until the next
  // call on this thread — documented in sb_get_relation_states' header
  // comment.
  static thread_local std::vector<std::string> anchor_name_storage;
  anchor_name_storage.clear();
  anchor_name_storage.reserve(states.size());
  for (const auto& s : states) {
    anchor_name_storage.push_back(s.anchor_name);
  }

  int32_t count = static_cast<int32_t>(states.size());
  int32_t to_write = count < cap ? count : cap;
  for (int32_t i = 0; i < to_write; ++i) {
    const auto& s = states[i];
    sb_relation_state_t& c = out[i];
    c.anchor_name = anchor_name_storage[i].c_str();
    c.proximity = s.proximity;
    c.contact_point_world_x = s.contact_point_world.x();
    c.contact_point_world_y = s.contact_point_world.y();
    c.contact_point_world_z = s.contact_point_world.z();
    c.normalized_angle = s.contact_normalized.angle;
    c.normalized_radial_ratio = s.contact_normalized.radial_ratio;
    c.normalized_height_ratio = s.contact_normalized.height_ratio;
    c.referent_element_id = static_cast<int64_t>(s.referent_element_id);
    c.referent_shared_id = static_cast<int64_t>(s.referent_shared_id);
    c.kernel_type = FromKernelType(s.kernel_type);
    c.normalized_strategy = ToNormalizedStrategy(s.contact_normalized.strategy);
    c.has_contact = s.has_contact ? 1 : 0;
    c.active = s.active ? 1 : 0;
  }
  return count;
}

// ----------------------------------------------------------------------------
// Update interval
// ----------------------------------------------------------------------------

void sb_set_update_interval(sb_space_t space, double seconds) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  info->space->SetUpdateInterval(seconds);
}

// ----------------------------------------------------------------------------
// Global cleanup
// ----------------------------------------------------------------------------

void sb_cleanup_all() {
  // Tunneled links first, for every session, as sb_destroy_session does for
  // one: a join in progress must be over before any registry lock is taken, and
  // no joinable thread may outlive this call.
  RetireAllTunnels();
  std::vector<sb_session_t> retiring_sessions;
  {
    std::lock_guard<std::mutex> mark_lock(g_sessions_mutex);
    for (auto& [session_id, ctx] : g_sessions) {
      if (!ctx) continue;
      ctx->rich_retiring = true;
      retiring_sessions.push_back(session_id);
    }
  }
  // Same rich terminal ordering as individual destroy, before broad cleanup
  // nulls callback slots and drops the space registry.
  for (const auto session : retiring_sessions)
    RetireAndReleaseRichActivationsForSession(session);
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  // Mirror sb_destroy_session's ordering per session (clear hub callbacks,
  // then PrepareDisconnect/reset network objects) rather than relying on
  // implicit destructor cascade: SessionContext's member declaration order
  // (context before spaces) happens to make plain destruction safe too, but
  // going through PrepareDisconnect() also sends the graceful
  // DisconnectFromHub RPC and keeps this path's behavior identical to
  // sb_destroy_session instead of two subtly different teardown sequences.
  for (auto& [session_id, ctx] : g_sessions) {
    if (!ctx) continue;
    if (ctx->context) {
      ctx->context->GetHub()->ClearCallbacks();
    }
    // See the matching comment in sb_destroy_session() above.
    for (auto& [space_id, info] : ctx->spaces) {
      std::lock_guard<std::mutex> callback_lock(info->callback_mutex);
      info->on_targets_callback = nullptr;
      info->on_rich_callback = nullptr;
      info->on_rich_user_data = nullptr;
      info->on_element_added_callback = nullptr;
      info->on_element_removed_callback = nullptr;
      info->on_link_removed_callback = nullptr;
      info->on_relation_event_callback = nullptr;
    }
    for (auto& [space_id, info] : ctx->spaces) {
      if (info->remote_hub) {
        (void)info->remote_hub->PrepareDisconnect();
        info->remote_hub.reset();
      }
      if (info->receiver) {
        // See the matching comment in sb_destroy_session() above --
        // call Stop() explicitly before dropping this reference so any
        // GrpcServerBase::Stop() self-join/torn-down-state race against a
        // HubReceiver call's own call_ctx_->receiver.lock() temporary always
        // resolves on THIS thread instead of a CQ thread.
        info->receiver->Stop();
        info->receiver.reset();
      }
    }
  }
  {
    std::lock_guard<std::mutex> slock(g_spaces_mutex);
    g_spaces.clear();
    g_space_to_session.clear();
  }
  g_sessions.clear();
  g_support_registrations.clear();
  g_observed_source_setups.clear();
}

// ----------------------------------------------------------------------------
// Runtime configuration
// ----------------------------------------------------------------------------

void sb_set_referent_resolve_mode(sb_space_t space,
                                  sb_referent_resolve_mode_t mode) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  info->space->SetReferentResolveMode(
      mode == SB_REFERENT_RESOLVE_SHAREDID
          ? spatialblend::ReferentResolveMode::kSharedId
      : mode == SB_REFERENT_RESOLVE_LINKER_ONLY
          ? spatialblend::ReferentResolveMode::kLinkerOnly
          : spatialblend::ReferentResolveMode::kLinker);
}

void sb_register_on_element_removed(sb_space_t space,
                                    sb_on_element_removed_t callback,
                                    void* user_data) {
  auto* info = GetSpace(space);
  if (!info) return;
  std::lock_guard<std::mutex> lock(info->callback_mutex);
  info->on_element_removed_callback = callback;
  info->on_element_removed_user_data = user_data;
}

void sb_register_on_link_removed(sb_space_t space,
                                 sb_on_link_removed_t callback,
                                 void* user_data) {
  auto* info = GetSpace(space);
  if (!info) return;
  std::lock_guard<std::mutex> lock(info->callback_mutex);
  info->on_link_removed_callback = callback;
  info->on_link_removed_user_data = user_data;
}

void sb_set_weight_smoothing(sb_space_t space, double alpha) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  info->space->SetWeightSmoothing(alpha);
}

void sb_set_min_move(sb_space_t space, double meters) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  info->space->SetMinMove(meters);
}

void sb_set_min_yaw(sb_space_t space, double radians) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  info->space->SetMinYaw(radians);
}


void sb_set_anchor_reach(sb_space_t space, const char* anchor_name,
                         double radius) {
  auto* info = GetSpace(space);
  if (!info || !info->space || !anchor_name) return;
  info->space->SetAnchorReach(anchor_name, radius);
}

int32_t sb_set_relation_phase_config(sb_space_t space,
                                     sb_kernel_type_t kernel_type,
                                     const sb_relation_phase_config_t* config,
                                     sb_relation_phase_config_t* out_effective) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return 0;
  if (kernel_type != SB_KERNEL_SPACING && kernel_type != SB_KERNEL_FACING &&
      kernel_type != SB_KERNEL_AFFORDANCE_POSE) {
    return 0;
  }
  const RelationKernelType family = ToKernelType(kernel_type);
  int32_t ok = 1;
  if (config && config->struct_size < sizeof(sb_relation_phase_config_t)) {
    ok = 0;
  } else if (config) {
    spatialblend::RelationPhaseConfig c;
    c.driving_weight = config->driving_weight;
    c.driving_release = config->driving_release;
    c.contact_proximity = config->contact_proximity;
    c.contact_release = config->contact_release;
    ok = info->space->SetRelationPhaseConfig(family, c).ok() ? 1 : 0;
  }
  if (out_effective) {
    // Honor the caller's struct_size: an older/smaller caller gets only the
    // prefix its buffer holds (never an overrun), and struct_size reports the
    // bytes written. A size too small for the header writes nothing.
    const uint32_t caller_size = out_effective->struct_size;
    if (caller_size < offsetof(sb_relation_phase_config_t, driving_weight)) {
      return 0;
    }
    const spatialblend::RelationPhaseConfig c =
        info->space->GetRelationPhaseConfig(family);
    sb_relation_phase_config_t full{};
    sb_relation_phase_config_t* const out = &full;
    out->driving_weight = c.driving_weight;
    out->driving_release = c.driving_release;
    out->contact_proximity = c.contact_proximity;
    out->contact_release = c.contact_release;
    const size_t n =
        std::min<size_t>(caller_size, sizeof(sb_relation_phase_config_t));
    full.struct_size = static_cast<uint32_t>(n);
    std::memcpy(out_effective, &full, n);
  }
  return ok;
}

void sb_set_realization_policy(sb_space_t space,
                               sb_realization_policy_t policy) {
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  spatialblend::RealizationPolicyType type;
  switch (policy) {
    case SB_REALIZATION_POLICY_UNIFORM:
      type = spatialblend::RealizationPolicyType::kUniform;
      break;
    case SB_REALIZATION_POLICY_GREEDY:
      type = spatialblend::RealizationPolicyType::kGreedy;
      break;
    case SB_REALIZATION_POLICY_PROGRESSIVE:
    default:
      type = spatialblend::RealizationPolicyType::kProgressive;
      break;
  }
  info->space->SetRealizationPolicy(type);
}

void sb_set_trace(sb_space_t space, const char* path) {
  // Binds THIS space's own space_id to its OWN TraceSink (see
  // trace.format.md and trace.h's ForSpace doc). Two spaces in
  // one process can be routed to two different trace files by calling
  // this once per space with different paths; calling it with the SAME path
  // for every space reproduces the old "one shared trace.jsonl" behavior. A
  // space that never gets its own sb_set_trace call keeps falling back to
  // the process-wide env-driven default (TraceSink::Instance()), unchanged.
  auto* info = GetSpace(space);
  if (!info || !info->space) return;
  spatialblend::space_id_t space_id = info->space->GetSpaceId();
  if (path == nullptr) {
    spatialblend::TraceSink::DisableForSpace(space_id);
  } else {
    spatialblend::TraceSink::SetPathForSpace(space_id, path);
    // Every trace a caller binds starts with a self-describing meta line
    // for this space (no "t": it precedes the space's timeline), and the
    // space re-emits its ev=policy on the next traced update.
    auto& sink = spatialblend::TraceSink::ForSpace(space_id);
    if (sink.Active()) {
      nlohmann::ordered_json ev;
      ev["ev"] = "meta";
      ev["space"] = space_id;
      ev["format"] = spatialblend::kTraceFormatVersion;
      ev["spec"] = "spatialblend/api/c/trace.format.md";
      ev["frame"] = "z_up";
      sink.Emit(ev);
    }
    info->space->OnTraceRebound();
  }
}

uint32_t sb_abi_version() { return kSpatialBlendV2AbiVersion; }
uint32_t sb_abi_minor_version() { return kSpatialBlendV2AbiMinorVersion; }

uint32_t sb_get_v2_abi_layout(sb_v2_abi_layout_t* layout,
                              uint32_t layout_size) {
  const uint32_t required = static_cast<uint32_t>(sizeof(sb_v2_abi_layout_t));
  if (!layout || layout_size < required) {
    return required;
  }
  layout->descriptor_size = required;
  layout->abi_version = kSpatialBlendV2AbiVersion;
  layout->abi_minor = kSpatialBlendV2AbiMinorVersion;
  layout->realization_target_size =
      static_cast<uint32_t>(sizeof(sb_realization_target_t));
  layout->pointer_size = static_cast<uint32_t>(sizeof(void*));
  layout->kernel_type_size = static_cast<uint32_t>(sizeof(sb_kernel_type_t));
  layout->arrival_time_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, arrival_time));
  layout->anchor_name_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, anchor_name));
  layout->position_x_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_x));
  layout->position_y_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_y));
  layout->position_z_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, position_z));
  layout->yaw_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, yaw));
  layout->weight_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, weight));
  layout->has_z_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, has_z));
  layout->kernel_type_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, kernel_type));
  layout->referent_id_offset =
      static_cast<uint32_t>(offsetof(sb_realization_target_t, referent_id));
  return required;
}

size_t sb_sizeof(sb_struct_id_t struct_id) {
  switch (struct_id) {
    case SB_STRUCT_ACTOR_STATE:
      return sizeof(sb_actor_state_t);
    case SB_STRUCT_REALIZATION_TARGET:
      return sizeof(sb_realization_target_t);
    case SB_STRUCT_RELATION_EVENT:
      return sizeof(sb_relation_event_t);
    case SB_STRUCT_RELATION_STATE:
      return sizeof(sb_relation_state_t);
    case SB_STRUCT_RELATION_STATE_V2:
      return sizeof(sb_relation_state_v2_t);
    case SB_STRUCT_SCOPE_RECORD:
      return sizeof(sb_scope_record_t);
    case SB_STRUCT_SCOPE_CANDIDATE:
      return sizeof(sb_scope_candidate_t);
    case SB_STRUCT_REALIZATION_POLICY_PARAMS:
      return sizeof(sb_realization_policy_params_t);
    case SB_STRUCT_REALIZATION_WEIGHT:
      return sizeof(sb_realization_weight_t);
    case SB_STRUCT_REALIZATION_CHANNEL:
      return sizeof(sb_realization_channel_t);
    case SB_STRUCT_SOLVER_DIAGNOSTICS:
      return sizeof(sb_solver_diagnostics_t);
    case SB_STRUCT_SOLVER_GOAL:
      return sizeof(sb_solver_goal_t);
    case SB_STRUCT_RELATION_PAYLOAD:
      return sizeof(sb_relation_payload_t);
    case SB_STRUCT_RELATION_QUERY:
      return sizeof(sb_relation_query_t);
    case SB_STRUCT_RELATION_EVALUATION:
      return sizeof(sb_relation_evaluation_t);
    case SB_STRUCT_REALIZATION_TARGET_V2:
      return sizeof(sb_realization_target_v2_t);
    case SB_STRUCT_V2_ABI_LAYOUT:
      return sizeof(sb_v2_abi_layout_t);
    case SB_STRUCT_RELATION_PHASE_CONFIG:
      return sizeof(sb_relation_phase_config_t);
    case SB_STRUCT_TARGET_RELATION_EVALUATION:
      return sizeof(sb_target_relation_evaluation_t);
    case SB_STRUCT_BONE_POSE:
      return sizeof(sb_bone_pose_t);
    case SB_STRUCT_NAMED_BONE_POSE:
      return sizeof(sb_named_bone_pose_t);
    case SB_STRUCT_CHAIN_JOINT:
      return sizeof(sb_chain_joint_t);
    case SB_STRUCT_PART_CHART_COORD:
      return sizeof(sb_part_chart_coord_t);
    case SB_STRUCT_ANCHOR_SOLVE:
      return sizeof(sb_anchor_solve_t);
    case SB_STRUCT_ANCHOR_TURN:
      return sizeof(sb_anchor_turn_t);
    case SB_STRUCT_ANCHOR_CHART:
      return sizeof(sb_anchor_chart_t);
    case SB_STRUCT_TARGET_REQUEST_PAYLOAD:
      return sizeof(sb_target_request_payload_t);
    case SB_STRUCT_ANCHOR_POSE:
      return sizeof(sb_anchor_pose_t);
    case SB_STRUCT_ANCHOR_PROPOSAL:
      return sizeof(sb_anchor_proposal_t);
    case SB_STRUCT_PROPOSAL_SET:
      return sizeof(sb_proposal_set_t);
    case SB_STRUCT_ACTOR_EVENT:
      return sizeof(sb_actor_event_t);
    case SB_STRUCT_ORIENTATION:
      return sizeof(sb_orientation_t);
  }
  return 0;
}

#include "spatialblend/api/c/observability_c.inc"
