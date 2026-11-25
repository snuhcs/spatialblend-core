// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_SPACE_H_
#define SPATIALBLEND_SPACE_SPACE_H_

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "spatialblend/base/common/coalescing_worker.h"
#include "spatialblend/base/common/common.h"
#include "spatialblend/model/actor/actor.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/containment_layout.h"
#include "spatialblend/model/element/scope_extension.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/geometry/planar_frame_chart.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/grounding_rule.h"
#include "spatialblend/model/relation/kernel_factory.h"
#include "spatialblend/model/actor/actor_motion_tracker.h"
#include "spatialblend/space/bone_port.h"
#include "spatialblend/space/embodied_port.h"
#include "spatialblend/solver/affordance_pose_sd.h"
#include "spatialblend/model/embodiment/embodiment.h"
#include "spatialblend/space/interaction_latency.h"
#include "spatialblend/space/receiver_proposal.h"
#include "spatialblend/base/common/seqlock_box.h"
#include "spatialblend/solver/realization_target.h"
#include "spatialblend/space/binding.h"
#include "spatialblend/solver/realization_policy.h"
#include "spatialblend/solver/realization_target.h"
#include "spatialblend/solver/referent_resolver.h"
#include "spatialblend/space/relation_event.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/space/relation_phase_tracker.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/space/relation_state.h"
#include "spatialblend/space/source_interaction_evidence.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/model/realization/receiver_input.h"
#include "spatialblend/space/receiver_approach_context.h"
#include "spatialblend/space/receiver_approach_feasibility.h"

namespace spatialblend {

class SpaceHub;

// Runtime config: how OnRemoteUpdate resolves an incoming payload's
// referent to a local element. kLinker (default) uses the ElementLinker
// cross-space identity then falls back to shared-id lookup; kSharedId forces
// legacy shared-id-only lookup. kLinkerOnly accepts only an explicit linker
// mapping, so unrelated local geometry with coincident dataset IDs remains in
// the scene but cannot become a relation referent. Mirrors
// SB_V2_REFERENT_RESOLVE ("linker" | "sharedid" | "linker_only").
enum class SurfaceChartStatus { kOk = 0, kUnknownPart = -1, kNotARectangle = -2 };

enum class ReferentResolveMode { kLinker, kSharedId, kLinkerOnly };

// Bits shared with the additive rich C boundary.  These describe the four
// historical snapshot domains; receiver-current input is deliberately a
// separate validity-only domain because Stage A cannot certify it yet.
enum RealizationRevisionDomain : uint32_t {
  kRealizationPolicyRevision = 1u << 0,
  kRealizationTreeRevision = 1u << 1,
  kRealizationCounterpartRevision = 1u << 2,
  kRealizationSupportRevision = 1u << 3,
};

struct ActorReceiverInputValidity {
  uint64_t revision = 0;
  ReceiverInputStamp captured;
  ReceiverInputStamp current;
  bool changed = false;
  bool valid = false;
};

struct ActorRealizationValidity {
  uint64_t revision = 0;
  uint64_t policy_revision = 0;
  uint64_t tree_revision = 0;
  uint64_t counterpart_revision = 0;
  uint64_t support_revision = 0;
  uint32_t captured_known_mask = 0;
  uint32_t current_known_mask = 0;
  uint32_t changed_mask = 0;
  uint32_t unchecked_mask = 0;
};

// Manages actors, bindings, and cross-space relation solving.
//
// Receiver contract (one owner per state):
//   1. OnRemoteUpdate() receives remote relation payloads, on whatever thread
//      delivers them.
//   2. It SAMPLES the receiver's embodied pose from the actor's port
//      (PublishEmbodiedPose is written only by the driver) and solves the
//      relation-preserving root target, continuing from the sample.
//   3. The result is published as the actor's ReceiverProposal (ReadProposal).
//   The driver moves the body within its own kinetic bounds and publishes the
//   new pose back. Core keeps no receiver pose, paces nothing, never writes a
//   receiver's transform; with no sample the proposal says so explicitly.
//
// Important invariants:
//   - received_updates_ is capped at 100 entries to prevent unbounded growth.
// One position goal of a solve (a decoded approach point and its softmax weight),
// keyed by the relation: the inputs of S0, kept for the next solve's attribution.
struct ProposalGoal {
  shared_element_id_t referent = kInvalidSharedElementId;
  int kernel = -1;
  Eigen::Vector2d point = Eigen::Vector2d::Zero();
  double weight = 0.0;
  int64_t decode_branch = 0;
};

// The yaw inputs of a solve: a Facing relation's weight, the conditioning of its
// bearing and the oriented bearing (the terms of the circular mean).
struct ProposalYawGoal {
  shared_element_id_t referent = kInvalidSharedElementId;
  int kernel = -1;
  double weight = 0.0, conditioning = 0.0, bearing = 0.0;
  // Support::look_at_vertex of the Facing's payload (-1: the gaze hits the referent).
  int32_t look_at_vertex = -1;
  // Support::look_at_boundary_crossed of the Facing's payload.
  bool boundary_crossed = false;
};

class Space {
 public:
  // The shipped source-side movement gate: v1's RuntimePolicy::minimal_state_change_threshold,
  // 5e-2, which v1 applies to every state in the state's OWN unit (ShouldProcessStateUpdate). A position is in metres, so the gate is 5 cm; a yaw state is in
  // DEGREES, so v1's yaw gate is 0.05 degrees. v2 carries
  // yaw in radians and ships the same ANGLE: MIN_YAW follows v1.
  static constexpr double kDefaultMinMoveM = 5e-2;
  static constexpr double kDefaultMinYawRad = 5e-2 * M_PI / 180.0;

  Space();
  // Offline deterministic receiver-input tests only. The C facade always
  // constructs the default native steady-clock Space; injected time is not
  // exported as the native C clock identity.
  explicit Space(std::function<int64_t()> receiver_input_clock);

  // Non-copyable (contains unique_ptr and move-only members).
  Space(const Space&) = delete;
  Space& operator=(const Space&) = delete;

  // Legacy borrowed views: even the const tree exposes mutable nested geometry.
  // Calling either permanently makes tree/support validity uncertifiable for
  // this Space. These views remain single-threaded compatibility APIs; they
  // cannot be used concurrently with solving. Prefer the owned APIs below.
  ElementTree* GetElementTree();
  const ElementTree* GetElementTree() const;

  // Receiver-owned scene input. Writers commit under the solve lock; geometry
  // and sub-region polygons are deep-copied, never retained by alias. Caller
  // must not concurrently modify input objects while a call copies them.
  // Unsupported SpatialDefinition subclasses are rejected, not sliced.
  bool AddElement(element_id_t id, element_id_t parent = kInvalidElementId);
  // Moves `id` (with its subtree, identity, shared id and links) under `parent`. A new parent-local transform given with
  // the move is applied atomically, so the footprint the layout check sees is the one the element will have; without it
  // the element keeps its local transform (now read under the new parent). A change that would overlap a sibling, or leave a
  // Containment child outside its parent, is not registered and the element is put back (LastSceneError says why).
  bool ReparentElement(element_id_t id, element_id_t parent,
                       const Eigen::Matrix4d* local_transform = nullptr);
  bool SetElementTransform(element_id_t id, const Eigen::Matrix4d& transform);
  // Duplicates are accepted (legacy contract); FindElementBySharedId /
  // referent resolution then follow ElementTree's declared rule (smallest
  // element id holding the shared id).
  bool SetElementSharedId(element_id_t id, shared_element_id_t shared_id);
  bool SetElementTag(element_id_t id, const std::string& tag);
  // ABI 4.11: the affordance pose an interaction-role child carries for its parent object (element/affordance_pose.h);
  // nullptr clears it. False for an unknown element.
  bool SetElementAffordancePose(element_id_t id, const AffordancePoseTemplate* pose);
  bool SetElementSpatialDefinition(
      element_id_t id, const std::shared_ptr<SpatialDefinition>& definition,
      bool only_if_absent = false);
  // Scene validation (element/containment_layout.h): a change that makes sibling
  // Containment/Navigable elements overlap, or a child stick out of its parent,
  // is NOT registered; the mutator returns false and this holds the explicit
  // status (code + a message naming both elements). It describes the most recent
  // scene mutator call (kOk after a successful one).
  struct SceneError {
    SceneErrorCode code = SceneErrorCode::kOk;
    std::string message;
  };
  SceneError LastSceneError() const;

  // --- Scope extension (element/scope_extension.h) ---
  // A region child that sticks out of its region parent extends the parent's
  // scope, at any distance (bridged across the gap). Admission is synchronous: a
  // change the layout rules reject (an extension or bridge overlapping a footprint
  // outside its line) is not applied, with LastSceneError. The union is built
  // on a background worker, coalesced (a burst of moves builds once more, from the
  // latest state), and installed atomically under the space lock: until then every
  // consumer reads the last complete scope. Actor elements never extend a parent.
  //
  // Bumped by every install that changed an effective region.
  uint64_t GetScopeExtensionRevision() const;
  // Waits until every rebuild requested so far is installed (true), or the timeout.
  // Must not be called from inside a callback that holds this space's lock.
  bool WaitForScopeExtension(std::chrono::milliseconds timeout = std::chrono::seconds(5));
  // How many rebuilds ran (the coalescing is observable through it).
  uint64_t GetScopeExtensionBuildCount() const;
  // Test seam: called on the worker between the capture and the computation, without
  // the space lock (a test blocks in it to observe the scope during a rebuild, or
  // throws from it to fail the build).
  void SetScopeExtensionBuildHookForTesting(std::function<void()> hook);
  // How many times the obstacle sync rebuilt a free space on a caller's thread (a
  // navigable's obstacle set changed): an installed extension must cause none.
  uint64_t GetObstacleSyncRebuildCountForTesting() const;
  bool AddElementSubRegion(element_id_t id, const SubRegion& region);
  // Surface charts of a named part (planar_frame_chart.h). Set: kUnknownPart for an unknown element or part,
  // kNotARectangle for a frame chart on a boundary that is not an axis-aligned rectangle (nothing changes).
  SurfaceChartStatus SetElementSubRegionChart(element_id_t id, const std::string& name, SurfaceChartKind chart);
  // The part's frame in the world, its boundary (part-local) and its chart, for encoding/decoding a point of the part.
  struct PartFrame {
    Eigen::Matrix4d part_to_world = Eigen::Matrix4d::Identity();
    std::shared_ptr<Polygon2D> boundary;
    SurfaceChartKind chart = SurfaceChartKind::kRadial;
  };
  bool GetElementPartFrame(element_id_t id, const std::string& name, PartFrame* out) const;
  bool AddElementAnchor(element_id_t id, const std::string& name,
                        const Eigen::Matrix4d& transform);
  bool SetActorElement(spatialblend::entity_id_t actor_id, element_id_t id);
  bool HasElement(element_id_t id) const;
  // Nonblocking preflight for callers already holding an outer registry lock.
  // False includes contention. Like other Space lock APIs, MUST NOT be called
  // from a callback running inside this Space's solve lock.
  bool TryHasElementOutsideCallback(element_id_t id) const;
  element_id_t FindElementBySharedId(shared_element_id_t shared_id) const;
  spatialblend::entity_id_t FindActorBySharedId(shared_element_id_t shared_id) const;
  struct ElementState {
    element_id_t id = kInvalidElementId;
    element_id_t parent = kInvalidElementId;
    shared_element_id_t shared_id = kInvalidSharedElementId;
    spatialblend::entity_id_t grounding_actor = spatialblend::kInvalidEntityId;
    std::string tag;
    Eigen::Matrix4d local_transform = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity();
    std::vector<element_id_t> children;
    std::unordered_map<std::string, Eigen::Matrix4d> anchors;
    // Detached deep copy; mutating this value never changes the Space.
    std::shared_ptr<SpatialDefinition> definition;
  };
  bool CopyElementState(element_id_t id, ElementState* out) const;

  // Independently supplied, local-only point referents. Never actor-owned
  // elements or a fabricated Surface/volume. SourceAnchor verifies an exact
  // current native observation; ReceiverCounterpart admits explicitly mapped
  // numeric input and never reads the producer's realized Actor transform.
  enum class ObservedFacingPointKind { kSourceAnchor, kReceiverCounterpart };
  bool BeginObservedFacingPoint(element_id_t element,
                                 spatialblend::entity_id_t producer,
                                 const std::string& anchor,
                                 ObservedFacingPointKind kind, int64_t maximum_age_ns,
                                 uint64_t* lease, element_handle_t source_point = {},
                                 uint64_t source_point_lease = 0);
  bool CommitObservedFacingPoint(element_id_t element, uint64_t lease,
                                  uint64_t source_epoch, int64_t source_sequence,
                                  const Eigen::Vector3d& world_point,
                                  int64_t observed_at_ns, int64_t valid_until_ns);
  struct ObservedFacingPointUpdate {
    element_id_t element;
    uint64_t lease;
    uint64_t source_epoch;
    int64_t source_sequence;
    Eigen::Vector3d world_point;
    int64_t observed_at_ns;
    int64_t valid_until_ns;
  };
  // All-or-nothing point publication, with no subject evaluation or replay.
  bool CommitObservedFacingPoints(
      const std::vector<ObservedFacingPointUpdate>& updates);
  bool ClearObservedFacingPoint(element_id_t element, uint64_t lease);
  bool RetireObservedFacingPoint(element_id_t element, uint64_t lease);
  // Explicit idle-time withdrawal, using the same native monotonic clock as
  // input admission. Never re-solves held source input or creates a new sample.
  void PruneObservedFacingPoints();

  // Detached numeric geometry, copied with the scene revisions under the solve
  // lock. Whole Surface/Navigable boundaries only; not a contact certificate.
  struct PlanarSupportState {
    uint64_t tree_revision = 0;
    uint64_t support_revision = 0;
    uint32_t definition_kind = 0;  // 1: Surface, 2: Navigable.
    GeometryRole role = GeometryRole::kUnspecified;
    Eigen::Matrix4d plane_to_world = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector2d> boundary;
  };
  enum class PlanarSupportReadResult { kOk, kUnavailable, kStale, kBusy };
  // Nonblocking, including under an outer lifetime registry lock. The copied
  // chart XY points map through plane_to_world, never element transform again.
  PlanarSupportReadResult TryCopyPlanarSupport(
      element_id_t id, uint64_t expected_tree_revision,
      uint64_t expected_support_revision, PlanarSupportState* out) const;

  // Actor management
  spatialblend::entity_id_t CreateActor(const std::vector<SubjectAnchor>& anchors);
  // Removes one actor and every actor-keyed binding, solver, controller, and
  // propagation cache. Returns false when the actor is already absent.
  bool DestroyActor(spatialblend::entity_id_t actor_id);
  // Legacy borrowed actor views also permanently taint scene certification:
  // mutable Actor::SetElementId bypasses receiver identity tracking.
  Actor* GetActor(spatialblend::entity_id_t actor_id);
  const Actor* GetActor(spatialblend::entity_id_t actor_id) const;
  bool HasActor(spatialblend::entity_id_t actor_id) const;
  bool AddActorAnchors(spatialblend::entity_id_t actor_id,
                       const std::vector<SubjectAnchor>& anchors,
                       bool reject_duplicates = false);
  // Embodied-pose sampling port (embodied_port.h). PublishEmbodiedPose is the
  // DRIVER's call: it never takes space_mutex_ and never waits for a solve (a
  // seqlock write; the only lock it takes is the port's own writer mutex, which
  // only the driver contends), the only writer of the actor's port.
  // SampleEmbodiedPose / ReadProposal are lock-free seqlock reads usable from
  // any thread (the mailbox map lookup is a hash lookup under a short mutex that
  // is never held across a solve); ReadProposal returns the proposal of the
  // latest solve (one consistent snapshot).
  EmbodiedPublishResult PublishEmbodiedPose(spatialblend::entity_id_t actor_id,
                                            const EmbodiedPoseInput& pose,
                                            double timestamp_s,
                                            uint64_t sequence);
  // Receiver-input frames forward their root pose through this internal path
  // (the public one refuses an input-owned actor).
  EmbodiedPublishResult PublishReceiverInputPose(
      spatialblend::entity_id_t actor_id, const EmbodiedPoseInput& pose,
      double timestamp_s, uint64_t sequence);
  bool SampleEmbodiedPose(spatialblend::entity_id_t actor_id,
                          EmbodiedSample* out) const;
  bool ReadProposal(spatialblend::entity_id_t actor_id,
                    ReceiverProposal* out) const;
  // Proposal-ready notification for in-process drivers (the harness, tests):
  // invoked after every published proposal, on the solving thread WITH the
  // space lock held. The handler may call only the non-blocking port calls
  // (PublishEmbodiedPose, SampleEmbodiedPose, ReadProposal) -- never another
  // Space method. A driver that polls ReadProposal needs no observer.
  using ProposalObserver = std::function<void(spatialblend::entity_id_t,
                                              const ReceiverProposal&)>;
  uint64_t AddProposalObserver(ProposalObserver observer);
  void RemoveProposalObserver(uint64_t id);
  // ---- ABI 4: anchors, per-anchor ports, events
  // Anchor id 0 is the actor's root anchor, whatever its name; every other
  // anchor is interned on first use. AnchorIdFor interns, so an app can obtain
  // the id of "pelvis" before it publishes a pose for it. False for an unknown
  // actor. Ids are stable for the actor's lifetime and never reused.
  static constexpr uint32_t kRootAnchorId = 0;
  bool AnchorIdFor(spatialblend::entity_id_t actor_id, const std::string& name,
                   uint32_t* id);
  bool AnchorNameOf(spatialblend::entity_id_t actor_id, uint32_t anchor_id,
                    std::string* name) const;
  // The DRIVER's call for any anchor: the only writer of that anchor's port.
  // Same contract as PublishEmbodiedPose (which is the root anchor's form).
  EmbodiedPublishResult PublishAnchorPose(spatialblend::entity_id_t actor_id,
                                          uint32_t anchor_id,
                                          const EmbodiedPoseInput& pose,
                                          double timestamp_s, uint64_t sequence);
  bool SampleAnchorPose(spatialblend::entity_id_t actor_id, uint32_t anchor_id,
                        EmbodiedSample* out) const;

  // ---- ABI 4.7: embodiment registration ----
  // Registers the actor's skeleton and the anchors it declares against it. Once per actor. The driver then
  // publishes BONES (PublishBonePose); a registered, bone-bound anchor has no port of its own: its sample is
  // the composition of its bone's latest sample with the registered offset, so the offset exists in exactly
  // one place. Anchors not in the registration (and the actor root) behave exactly as before.
  enum class RegisterEmbodimentResult { kOk, kInvalid, kRefused, kBindingConflict };
  // On kBindingConflict: the first existing binding the declared families would forbid ("anchor@family").
  RegisterEmbodimentResult RegisterEmbodiment(spatialblend::entity_id_t actor_id, const Embodiment& embodiment,
                                              std::string* conflict);
  RegisterEmbodimentResult RegisterEmbodiment(spatialblend::entity_id_t actor_id, const Embodiment& embodiment);
  std::shared_ptr<const Embodiment> EmbodimentOf(spatialblend::entity_id_t actor_id) const;
  // The driver's call for one bone (the only writer of that bone's port; same contract as PublishAnchorPose).
  EmbodiedPublishResult PublishBonePose(spatialblend::entity_id_t actor_id, int bone_id, const BonePoseInput& pose,
                                        double timestamp_s, uint64_t sequence);
  bool SampleBonePose(spatialblend::entity_id_t actor_id, int bone_id, BoneSample* out) const;

  // A source-app exception (docs ABI 4 section 5): the source raises an
  // app-specific exception with an opaque payload (a button next to a whiteboard
  // ...). It rides the next update this actor propagates, which it forces out, and
  // every receiving space realizes it as an SB_ACTOR_EVENT_SOURCE_EXCEPTION. Core
  // never interprets the payload. False for an unknown actor or a payload above
  // kMaxExceptionPayloadBytes.
  static constexpr size_t kMaxExceptionPayloadBytes = 256;
  bool RaiseSourceException(spatialblend::entity_id_t actor_id,
                            uint32_t exception_id, const void* payload,
                            size_t payload_size);

  // Events for a receiver actor, polled in order. The latency policy
  // (interaction_latency.h) raises kTeleportRequest; a source exception raises
  // kSourceException.
  struct ActorEvent {
    enum class Kind : int32_t {
      kSourceException = 1,
      kTeleportRequest = 2,
      kPositionRequest = 3,
      kYawRequest = 4,
      kTargetRelease = 5
    };
    Kind kind = Kind::kSourceException;
    uint64_t revision = 0;  // the proposal-set revision it was raised with
    int64_t source_sequence = 0;
    int64_t source_shared_id = 0;
    uint32_t exception_id = 0;
    std::vector<uint8_t> payload;
    // kTeleportRequest: the interaction, the anchor to place and its target pose.
    InteractionKind interaction_kind = InteractionKind::kSpacingContact;
    uint32_t anchor_id = 0;
    double x = 0.0, y = 0.0, z = 0.0;
    double yaw = 0.0, pitch = 0.0, roll = 0.0;
    int32_t orientation_type = 0;
    // ABI 4.13 typed target events. channels: position=1, yaw=2; releases may
    // carry both.
    uint32_t channels = 0;
    uint64_t sample_epoch = 0, sample_sequence = 0;
    double sample_yaw = 0.0, yaw_turn = 0.0;
    uint32_t proposal_flags = 0, proposal_cause = 0;
    double facing_yaw = 0.0;
    ProposalStatus proposal_status = ProposalStatus::kNone;
    uint32_t request_flags =
        0;  // bit 0: absolute placement, never ordinary motion
    bool target_delivery = false;  // native admission provenance: this revision
                                   // has immutable typed targets
  };
  // Opt-in; legacy readers keep their lifecycle-only event stream. Enabling
  // (also on reload) releases old targets and starts a fresh placed answer on
  // the next source evaluation. Disabling affects future typed deliveries only.
  bool SetTargetEventDelivery(spatialblend::entity_id_t actor_id, bool enabled);
  // Callback-safe nonblocking queue pop; unlike the C polling boundary, this
  // does not fence in-flight solves.
  bool PollActorEvent(spatialblend::entity_id_t actor_id, ActorEvent* out);
  // C polling boundary: takes the solve lock before popping. 1 event, 0 true
  // empty, -1 invalid actor/output. Not callable from a synchronous proposal
  // observer. After publishing one new sample per anchor, draining to 0 fences
  // every solve that could have sampled the previous sample; future ordinary
  // requests use the new sample.
  int PollActorEventAfterSolve(spatialblend::entity_id_t actor_id,
                               ActorEvent* out);
  // The per-interaction maximum-latency policy. The compiled default is v1's
  // interaction timeout, 5000 ms for both kinds (DefaultInteractionLatencyPolicy).
  // C++ only: it is core's policy and is not exported through the C ABI.
  void SetInteractionLatencyPolicy(const InteractionLatencyPolicy& policy);
  InteractionLatencyPolicy GetInteractionLatencyPolicy() const;
  // The actor's pose as the element tree shows it: a source actor's own pose,
  // or, for a receiver avatar, the driver's latest sample (derived mirror).
  bool CopyActorPose(spatialblend::entity_id_t actor_id, Eigen::Vector2d* position,
                     double* yaw);
  // Stateless course query in the actor's obstacle-aware free space (empty =
  // no traversable course). Reads no pose.
  // The same query in the free space under the exemptions a solve declared
  // (SolverResult::free_exempt / free_ring_exempt): a footprint the source engages is
  // open to the body, so a course may end inside it. Not part of the C ABI; a test
  // double that stands in for the avatar controller uses it.
  // Whether `point` is in the actor's free space F under the exemptions a solve declared. The
  // harness asks it to judge an answer against the set the solver optimised over. Not part of
  // the C ABI.
  bool IsFreeUnder(spatialblend::entity_id_t actor_id, const Eigen::Vector2d& point,
                   const std::vector<int64_t>& exempt,
                   const std::vector<int64_t>& ring_exempt);
  std::vector<Eigen::Vector2d> QueryCourseUnder(
      spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
      const Eigen::Vector2d& to, const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt);
  // DRIVER SERVICE (see QueryCourse): where a driver that PLACES its body (a first placement, a teleport on a source
  // exception) puts it so it is clear of the footprints: the nearest point of the free space of a body of this actor's
  // radius. The proposal is never moved by it.
  std::optional<Eigen::Vector2d> QueryNearestClearPoint(spatialblend::entity_id_t actor_id, const Eigen::Vector2d& point,
                                                        const std::vector<int64_t>& exempt,
                                                        const std::vector<int64_t>& ring_exempt);
  // Every route the course query chooses among, with its first waypoint, length, segment count and
  // arithmetic error (ObstacleFreeSpace::GeodesicRoutes), in the free space of QueryCourseUnder; and the
  // exact cone of directions in which that free space continues from a point (TangentConeAt). The harness
  // builds its directional derivatives from them. Not part of the C ABI.
  std::vector<ObstacleFreeSpace::RoutePiece> QueryRoutesUnder(
      spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from, const Eigen::Vector2d& to,
      const std::vector<int64_t>& exempt, const std::vector<int64_t>& ring_exempt, double slack);
  // The same in the free space GetGeodesicDistance measures (point connectivity around the raw footprints, no
  // exemptions): the length of a spacing relation's distance.
  std::vector<ObstacleFreeSpace::RoutePiece> QueryRoutes(spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
                                                         const Eigen::Vector2d& to, double slack);
  // The routes to the NEAREST point of a boundary (its rings, in the world), one piece per boundary edge and route
  // (ObstacleFreeSpace::ContactRoutes), in the free space of QueryRoutes: the pieces of a relation whose contact is the nearest
  // point of a polygon (a container's border, a referent's boundary).
  std::vector<ObstacleFreeSpace::RoutePiece> QueryContactRoutes(
      spatialblend::entity_id_t actor_id, const Eigen::Vector2d& from,
      const std::vector<std::vector<Eigen::Vector2d>>& rings, double slack);
  ObstacleFreeSpace::FeasibleCone QueryConeUnder(
      spatialblend::entity_id_t actor_id, const Eigen::Vector2d& point, const std::vector<int64_t>& exempt,
      const std::vector<int64_t>& ring_exempt, double tol);
  // DRIVER SERVICES: the course services below take the actor's registered body radius (0 when
  // none) so a DRIVER can keep its body clear of the furniture. They are for the driver to call; the core's proposal
  // and solve never apply a body radius (their free space F has clearance 0).
  std::vector<Eigen::Vector2d> QueryCourse(spatialblend::entity_id_t actor_id,
                                           const Eigen::Vector2d& from,
                                           const Eigen::Vector2d& to);

  // Dedicated external receiver input. Begin permanently opts this actor out
  // of generated/local-offset pose fallback. All configured Actor anchors must
  // appear once per frame, including explicit unavailable rows. The declared
  // root must be planar, unparented and zero-offset. Legacy controllers never
  // actuate an opted receiver; body admission is separate. Rig/calibration
  // revisions are caller identities, not proof of physical calibration.
  bool BeginReceiverInputEpoch(spatialblend::entity_id_t actor_id,
                               const std::string& root_anchor,
                               uint64_t rig_revision,
                               uint64_t calibration_revision, uint64_t* epoch);
  bool ApplyReceiverInputFrame(
      spatialblend::entity_id_t actor_id, uint64_t epoch, uint64_t sequence,
      int64_t observed_at_ns, int64_t valid_until_ns,
      const std::vector<ReceiverAnchorInput>& anchors);
  bool ClearReceiverInputFrame(spatialblend::entity_id_t actor_id, uint64_t epoch);
  enum class ReceiverReevaluationResult {
    kEvaluated,
    kNoNewInput,
    kUnavailable,
    kStaleInput,
  };
  // Re-solve the last admitted source packet against an exact fresh external
  // receiver frame. Never receives/replays source data or renews its ingress
  // timestamp. This operation only produces rich geometric goals.
  ReceiverReevaluationResult ReevaluateReceiverRealization(
      spatialblend::entity_id_t actor_id, uint64_t expected_epoch,
      uint64_t expected_sequence);
  bool GetActorReceiverInputValidity(spatialblend::entity_id_t actor_id,
                                     uint64_t realization_revision,
                                     ActorReceiverInputValidity* out) const;

  // Native-owner approach-input boundary.  These APIs do not read scene
  // support, source motion, Unity state, or actor display transforms.  The
  // caller supplies already-resolved local geometry/destination; Space fences
  // it against one exact external receiver input frame.
  ReceiverApproachContextStatus RegisterReceiverApproachContext(
      const ReceiverApproachContextRegistration& registration,
      uint64_t* out_context_revision = nullptr);
  ReceiverApproachContextStatus TryCopyReceiverApproachContext(
      spatialblend::entity_id_t receiver_actor,
      uint64_t expected_context_revision, ReceiverApproachContext* out) const;
  // Approach feasibility consumes only the immutable approach context.  It
  // performs receiver-local 2D route feasibility, never reads source motion or actuator state.
  ReceiverApproachFeasibilityStatus TryComputeReceiverApproachFeasibility(
      spatialblend::entity_id_t receiver_actor, uint64_t context_revision,
      ReceiverApproachFeasibilitySnapshot* out) const;
  ReceiverApproachFeasibilityStatus TryCopyReceiverApproachFeasibilitySnapshot(
      spatialblend::entity_id_t receiver_actor, uint64_t calculation_revision,
      ReceiverApproachFeasibilitySnapshot* out) const;

  // Bind API (replaces AddRelation)
  Binding* Bind(spatialblend::entity_id_t actor_id, const std::string& anchor_name,
                RelationKernelType kernel_type,
                const std::string& tag_filter = "");
  // Checked explicit local subtree bind. Unlike Bind(), scope is a hard
  // candidate boundary and must name a live local element.
  Binding* BindScoped(spatialblend::entity_id_t actor_id,
                      const std::string& anchor_name,
                      RelationKernelType kernel_type, element_id_t scope_root,
                      const std::string& tag_filter = "");

  const std::vector<Binding>& GetBindings(
      spatialblend::entity_id_t actor_id) const;

  // Owned binding creation for consumers that need certifiable source
  // evidence. The legacy pointer-returning Bind APIs remain supported but
  // permanently mark that actor's bindings as externally mutable.
  bool BindOwned(spatialblend::entity_id_t actor_id, const std::string& anchor,
                 RelationKernelType kernel, element_id_t scope = kInvalidElementId,
                 const std::string& tag_filter = "");
  // Explicit opt-in, for future frames only. Reset/retirement ends this mode;
  // enabling cannot turn a held input into a new observation.
  bool EnableSourceInteractionEvidence(spatialblend::entity_id_t actor_id,
                                        uint64_t epoch, bool enabled);
  enum class SourceEvidenceReadResult { kOk, kUnavailable, kStale, kBusy };
  // Nonblocking form for callers pinning an outer session/lifetime registry.
  SourceEvidenceReadResult TryEnableSourceInteractionEvidence(
      spatialblend::entity_id_t actor_id, uint64_t epoch, bool enabled);
  // Exact-frame, current-context copy of an already captured value. Does not
  // evaluate relations or join mutable scene data onto an earlier frame.
  SourceEvidenceReadResult TryCopySourceInteractionEvidence(
      spatialblend::entity_id_t actor_id, uint64_t epoch, int64_t sequence,
      int64_t maximum_age_ns, SourceInteractionEvidence* out) const;

  // State update — resolves scope, evaluates, auto-propagates
  void UpdateActorState(spatialblend::entity_id_t actor_id,
                        const Eigen::Vector2d& position, double yaw);
  // The same with an orientation beyond yaw (ABI 3.1); pitch and roll are carried
  // to the receiver's proposal as a pass-through (never solved).
  void UpdateActorState(spatialblend::entity_id_t actor_id,
                        const Eigen::Vector2d& position, double yaw,
                        const SourceOrientation& orientation);
  struct ActorObservation {
    std::string anchor_name;
    Eigen::Vector3d position;
    double yaw;
    // Set only by ComposeBoneObservations: this pose was composed from a bone by the registration. A bone-bound
    // anchor of a registered actor is refused from any other source.
    bool from_bones = false;
    // The anchor's orientation beyond yaw when it is known (a composed bone pose: SB_ORIENTATION_YAW_PITCH_ROLL).
    int32_t orientation_type = 0;
    double pitch = 0.0, roll = 0.0;
  };
  // SOURCE side: the world poses of the bones the app tracked, by name. The registered, bone-bound anchors
  // whose bone is present are composed (position and yaw) and returned as observations; no state is touched.
  // False for an actor without an embodiment or a bone name the skeleton lacks.
  bool ComposeBoneObservations(spatialblend::entity_id_t actor_id,
                               const std::vector<std::pair<std::string, RigidTransform>>& bones,
                               std::vector<ActorObservation>* out) const;
  bool BeginActorObservationEpoch(spatialblend::entity_id_t actor_id,
                                  const std::string& root_anchor,
                                  uint64_t* epoch);
  // Fresh source-only setup admission.  This is intentionally narrower than
  // an observation batch: it opens epochs and observed point leases, but does
  // not admit a sample, evaluate a subject, invoke a callback, or propagate.
  struct ObservedSourceSetupParticipant {
    spatialblend::entity_id_t actor;
    std::string root_anchor;
    uint64_t expected_prior_epoch = 0;
  };
  struct ObservedSourceSetupPoint {
    element_id_t element;
    spatialblend::entity_id_t producer;
    std::string anchor;
    int64_t maximum_age_ns;
  };
  struct ObservedSourceSetupActorResult {
    spatialblend::entity_id_t actor;
    uint64_t epoch;
  };
  struct ObservedSourceSetupPointResult {
    element_id_t element;
    uint64_t lease;
  };
  bool BeginObservedSourceSetup(
      const std::vector<ObservedSourceSetupParticipant>& participants,
      const std::vector<ObservedSourceSetupPoint>& points,
      std::vector<ObservedSourceSetupActorResult>* actor_results,
      std::vector<ObservedSourceSetupPointResult>* point_results);
  // Cancellation is truthful only before any participant frame or point
  // publication.  It preserves a per-actor epoch fence so a cancelled epoch
  // can never be recreated and mistaken for the old capability.
  bool RetireObservedSourceSetup(
      const std::vector<ObservedSourceSetupActorResult>& actors,
      const std::vector<ObservedSourceSetupPointResult>& points);
  // Classifies an exact setup capability without changing Core state. The C
  // registry discards its cancellation capability only after evidence was
  // admitted or another terminal transition made it stale. An untouched setup
  // remains cancellable.
  enum class ObservedSourceSetupReleaseState {
    kUnadmitted,
    kAdmitted,
    kTerminal,
  };
  ObservedSourceSetupReleaseState GetObservedSourceSetupReleaseState(
      const std::vector<ObservedSourceSetupActorResult>& actors,
      const std::vector<ObservedSourceSetupPointResult>& points) const;
  bool ApplyActorObservationFrame(
      spatialblend::entity_id_t actor_id, uint64_t epoch, int64_t sequence,
      double time_seconds, const std::vector<ActorObservation>& observations);
  struct ActorObservationFrame {
    spatialblend::entity_id_t actor;
    uint64_t epoch;
    int64_t sequence;
    double time_seconds;
    std::vector<ActorObservation> observations;
  };
  struct ObservedFacingPointPublication {
    element_id_t element;
    uint64_t lease;
    int64_t observed_at_ns;
    int64_t valid_until_ns;
  };
  // Validate every fresh frame and source point before mutating anything.
  // Admit all anchors, leaf poses and selected points before evaluating any
  // subject. A publication derives its value only from its producer's frame.
  // Missing/unpublished producer anchors withdraw their previous point.
  bool ApplyActorObservationBatch(
      const std::vector<ActorObservationFrame>& frames,
      const std::vector<ObservedFacingPointPublication>& publications);
  bool ClearActorObservationAnchor(spatialblend::entity_id_t actor_id,
                                   uint64_t epoch,
                                   const std::string& anchor_name);
  bool GetActorObservationClock(spatialblend::entity_id_t actor_id,
                                uint64_t* epoch, int64_t* sequence,
                                double* time, bool* has_sample) const;

  // Space identity
  space_id_t GetSpaceId() const;
  void SetSpaceId(space_id_t id);

  // Update interval (used to set step_duration on outgoing
  // RemoteRelationUpdate)
  void SetUpdateInterval(double seconds);

  // The relation solver's configuration (the per-solve evaluation budget, the safety cap). Set before the first update; the
  // default Config is what a Space is built with, and the budget is on in it.
  void SetRelationSolverConfig(const RelationSolver::Config& config);

  // This space's monotonic local clock (seconds) -- see space_clock_
  // below for what advances it. Exposed so tests (and any consumer that
  // wants "how far has this space's own timeline progressed") don't need to
  // scrape it back out of emitted trace lines.
  double GetSpaceClock() const { return space_clock_; }

  // Hub integration
  void SetHub(SpaceHub* hub);

  // Receiving side
  void OnRemoteUpdate(const RemoteRelationUpdate& update);

  // Resolves the receiver-local actor for an incoming update using the same
  // source-handle linker path as OnRemoteUpdate (with legacy shared-id
  // fallback only when no link resolves). Returns kInvalidEntityId if absent.
  spatialblend::entity_id_t FindActorForRemoteUpdate(
      const RemoteRelationUpdate& update) const;

  // Legacy diagnostic view. Callers that may race a receiver must use
  // GetReceivedUpdatesSnapshot() rather than retaining this vector reference.
  const std::vector<RemoteRelationUpdate>& GetReceivedUpdates() const;
  struct ReceivedUpdatesSnapshot {
    uint64_t count = 0;
    std::vector<RemoteRelationUpdate> updates;
  };
  // Atomically snapshots the monotonic observation count and its bounded
  // retained update ring. Intended for callback dispatch; no callback runs
  // while the space lock is held.
  ReceivedUpdatesSnapshot GetReceivedUpdatesSnapshot() const;
  // Monotonic observation count independent of the bounded diagnostic ring.
  uint64_t GetReceivedUpdateCount() const;

  // Read-only wire-progress diagnostics. An emitted sequence advances only
  // when UpdateActorState actually propagates (not on deadband-held ticks).
  // A receiver records the exact source sequence only after OnRemoteUpdate
  // has accepted it past duplicate suppression.
  int64_t GetLastEmittedSequence(spatialblend::entity_id_t actor_id) const;
  int64_t GetLastAppliedRemoteSequence(
      space_id_t source_space_id,
      shared_element_id_t source_actor_shared_id) const;

  // Returns the wire identity UpdateActorState places in RemoteRelationUpdate
  // for this local actor. `space_id` and `actor_id` here are core identities,
  // not C handles. False means the actor has no element identity to publish.
  bool GetActorWireIdentity(spatialblend::entity_id_t actor_id,
                            space_id_t* source_space_id,
                            shared_element_id_t* source_actor_shared_id) const;

  // Per-anchor world-pose override ("anchor liveness"):
  // SubjectAnchor is a fixed local offset from
  // the actor root, fine for the simulator but wrong for a real tracked hand
  // (Quest). When set, grounding (the anchor_pos computation in
  // UpdateActorState) uses this WORLD pose directly instead of deriving the
  // anchor's position/yaw from actor_transform * local_offset every frame;
  // clearing restores the offset-derived behavior. No-op if the actor or
  // anchor doesn't exist.
  //
  // The "root" anchor is a partial exception: root_pos (used for scope
  // resolution / containment / IsActive checks) is ALWAYS driven by the
  // live actor transform set via UpdateActorState, never by this override,
  // even if an override happens to be stored under the name "root" — only a
  // binding whose anchor_name is literally "root" would ever read it.
  // A bone-bound anchor of a registered actor is derived from its bone: this call leaves it unchanged (use
  // SetBoneAnchorWorldPose with a pose ComposeBoneObservations returned).
  void SetAnchorWorldPose(spatialblend::entity_id_t actor_id,
                          const std::string& anchor_name,
                          const Eigen::Vector3d& position, double yaw);
  // The harness feed of a composed anchor: only for a registered, bone-bound anchor. False otherwise.
  bool SetBoneAnchorWorldPose(spatialblend::entity_id_t actor_id, const std::string& anchor_name,
                              const Eigen::Vector3d& position, double yaw, int32_t orientation_type = 0,
                              double pitch = 0.0, double roll = 0.0);
  void SetAnchorWorldPoseImpl(spatialblend::entity_id_t actor_id, const std::string& anchor_name,
                              const Eigen::Vector3d& position, double yaw, int32_t orientation_type = 0,
                              double pitch = 0.0, double roll = 0.0);
  void ClearAnchorWorldPose(spatialblend::entity_id_t actor_id,
                            const std::string& anchor_name);
  // Marks an anchor present/absent in authoritative tracked observations.
  // Anchors start active for legacy callers. Any availability transition is a
  // propagation-dirty event even if the actor root remains stationary.
  bool SetAnchorActive(spatialblend::entity_id_t actor_id,
                       const std::string& anchor_name, bool active);

  // Relation-state query: a read-only snapshot of every currently
  // grounded (anchor, referent) relation for this actor, taken directly from
  // each Binding's latest_matches (populated by UpdateActorState, and on a
  // received actor by OnRemoteUpdate) — i.e. the exact numbers last
  // propagated, never recomputed. Empty if the actor has no bindings or none
  // matched this frame.
  std::vector<RelationState> GetRelationStates(
      spatialblend::entity_id_t actor_id) const;
  // Same, for a caller that already holds the space lock -- i.e. code running
  // inside a relation-event callback, which the space invokes under its lock.
  std::vector<RelationState> GetRelationStatesFromCallback(
      spatialblend::entity_id_t actor_id) const;

  // Per-family relation activation thresholds (RelationPhaseConfig).
  // Applies to this space's source-side phases -- which gate what reaches
  // the receiver -- and to its receiver-side lifecycle events. An
  // invalid config (ValidateRelationPhaseConfig) is rejected and the previous
  // one kept. Defaults: DefaultRelationPhaseConfig.
  absl::Status SetRelationPhaseConfig(RelationKernelType family,
                                      const RelationPhaseConfig& config);
  RelationPhaseConfig GetRelationPhaseConfig(RelationKernelType family) const;

  // The body's floor-footprint radius belongs to the DRIVER: the core's free space F is the
  // walkable minus the obstacle footprints, with no body-radius erosion, and the core never errors on an unset radius.
  // The radius is PER ACTOR, registered by the actor's embodiment (RegisterEmbodiment,
  // `body.floor_footprint_radius_m`) so avatars of different sizes coexist in one space; SetBodyRadius is the
  // space-wide radius of the actors that register none. GetBodyRadius(actor) is the radius that actor's driver
  // keeps clear (NaN when none); QueryCourse and QueryCourseUnder, the driver's course services, keep a body of
  // that radius clear of the footprints (0 when none is registered).
  void SetBodyRadius(double radius_m);
  double GetBodyRadius() const;
  double GetBodyRadius(spatialblend::entity_id_t actor_id) const;

  const RealizationResult* GetRealizationTargets(
      spatialblend::entity_id_t actor_id) const;

  // Native opt-in.  Legacy realization continues to project Facing into
  // root yaw.  When enabled, facing remains a separately addressable channel
  // and is solved at the named receiver anchor pose.  This whole-space switch
  // is C++ only; the C ABI (spatialblend_rich.h) uses the actor-scoped one below.
  void SetRichRealizationEnabled(bool enabled);
  // Actor-scoped activation.  Unlike the whole-space fixture switch above,
  // this never changes realization behavior for other actors in the space.
  void SetActorRichRealizationEnabled(spatialblend::entity_id_t actor_id,
                                      bool enabled);
  // Thread-safe value copy for C ABI consumers.  Do not retain the native pointer
  // view across a receiver update.
  bool CopyActorRealizationSnapshot(spatialblend::entity_id_t actor_id,
                                    ActorRealizationSnapshot* out) const;
  // Compares an immutable historical solve with receiver state without
  // producing a new solve/snapshot.  `revision` fences a caller's frozen
  // transaction; false means the actor has no such current snapshot.
  bool GetActorRealizationValidity(spatialblend::entity_id_t actor_id,
                                   uint64_t revision,
                                   uint32_t requested_domains,
                                   ActorRealizationValidity* out);
  const ActorRealizationSnapshot* GetActorRealizationSnapshot(
      spatialblend::entity_id_t actor_id) const;

  // Temporal EMA factor for the realization weights (see OnRealizationTargets):
  // alpha in (0,1], smaller = smoother/laggier, 1 = no smoothing. Defaults from
  // SB_V2_WEIGHT_SMOOTHING (else 0.25). Exposed for tests/tuning.
  void SetWeightSmoothing(double alpha);

  // Movement-gate deadband (see the propagation gate in UpdateActorState):
  // min_move_m in metres, min_yaw_rad in radians. Values < 0 reset to the
  // value resolved at construction from SB_V2_MIN_MOVE/SB_V2_MIN_YAW (or their
  // v1 defaults, 5 cm / 0.05 deg); the environment is not re-read. Set explicitly
  // per-instance here instead of via process environment variables — tests
  // that need to disable the gate (e.g. to converge a progressive target over
  // repeated same-pose updates) MUST use this instead of setenv(), since an
  // env var mutation is process-global and leaks into every Space created
  // later in the same test binary.
  void SetMovementGate(double min_move_m, double min_yaw_rad);

  // Element lifecycle — share/unshare elements across spaces via hub.
  // user_data is an opaque application payload (e.g. serialized entity)
  // propagated verbatim to peer spaces and surfaced via the on-element-added
  // hook so they can reconstruct the shared object (OnShare).
  void ShareElement(element_id_t elem_id, const std::string& user_data = "");
  void UnshareElement(element_id_t elem_id);
  // Deletes a local-only, non-root subtree and its actor/solver state. Refuses
  // received, shared, or linked geometry; this does not propagate an unshare.
  bool RemoveLocalElementSubtree(element_id_t elem_id);
  void OnElementEvent(const ElementEvent& event);

  // Late join: the host reports that the element `element_id` (already in this
  // tree, created from the bootstrap) is a live share whose ADDED event is
  // `event`. Treats it as that event received on a live share, but onto the
  // existing element instead of a new one: records its source for
  // ResolveSharedElement and fires the on-element-added hook with the payload.
  // No-op for an unknown element or a source already received or linked.
  void AdoptBootstrapShare(element_id_t element_id, const ElementEvent& event);

  // Host side: the local element that stands for the share whose source is
  // `source` (this space's own shared element, a received share pending or
  // resolved as a new replica, or the local element it was linked to).
  // kInvalidElementId if there is none.
  element_id_t LocalElementForSource(const element_handle_t& source) const;

  // Invoked when a peer-shared element is added to this space's tree via
  // OnElementEvent(kAdded). Args: local element id, shared id, tag, user_data.
  using OnElementAddedFn = std::function<void(
      element_id_t, shared_element_id_t, const std::string& /*tag*/,
      const std::string& /*user_data*/)>;
  void SetOnElementAdded(OnElementAddedFn fn);

  // Invoked after a source-owned received replica has actually been removed.
  // The callback runs without space_mutex_ held and may re-enter Space.
  using OnElementRemovedFn = std::function<void(element_id_t)>;
  void SetOnElementRemoved(OnElementRemovedFn fn);

  // Removes the cross-space link(s) whose receiver-side counterpart is
  // `elem_id` in this space, and tells each linked source's space (through the
  // hub) so it can release its side. The element is NOT deleted: a
  // LinkToExisting target stays exactly as it was, an AddAsNew replica stops
  // being source-owned and becomes a plain local element (a later unshare of
  // the source no longer removes it). The source stays shared; other receivers'
  // links are untouched. Relations from a retired source never resolve onto
  // `elem_id` again, not even through a shared id, until the source is shared
  // and resolved anew. Returns the number of links removed (0: the element is
  // unknown, is not the counterpart of a remote source, or is a sharer's own
  // source element -- use UnshareElement for that).
  int UnlinkElement(element_id_t elem_id);

  // Why a link this space held was removed by the peer.
  enum class LinkRemovedReason {
    // The sharer unshared the source this space had linked a local element to.
    // The local element survives (a LinkToExisting target); an AddAsNew replica
    // is removed and reported through OnElementRemovedFn instead.
    kPeerUnshared = 1,
    // A receiver unlinked a share this space owns; the source stays shared.
    kPeerUnlinked = 2,
  };
  // Invoked, without space_mutex_ held, when the peer removed a link whose end
  // in this space is `local_element` (which still exists). `peer_space` is the
  // sharer's space for kPeerUnshared and the receiver's for kPeerUnlinked.
  using OnLinkRemovedFn = std::function<void(
      element_id_t /*local_element*/, space_id_t /*peer_space*/,
      LinkRemovedReason)>;
  void SetOnLinkRemoved(OnLinkRemovedFn fn);

  // Called by the app after receiving an on_element_added callback to resolve
  // a shared element:
  //   - received_id == resolved_id (AddAsNew): keep the temp element, link
  //     source -> received.
  //   - received_id != resolved_id (LinkToExisting): remove the temp element,
  //     link source -> resolved.
  void ResolveSharedElement(element_id_t received_id, element_id_t resolved_id);
  // Relation lifecycle events. Registers a single
  // callback (replaces any previous) invoked for each relation phase
  // transition as actors are updated (source side) or remote updates are
  // applied (target side).
  void RegisterOnRelationEvent(std::function<void(const RelationEvent&)> cb);

  // --- Containment transfer (space/containment_transfer.h) -----------------
  // Opt-in per Containment element: declares the region for replay. A transfer
  // LEAF is a region declared on BOTH sides whose shapes are equivalent in the
  // two spaces (congruent under a rigid transform; whether it carries an object
  // does not matter). While an actor's root is inside a leaf, the leaf is the
  // actor's whole scope and the empty-scope mapping is the transfer; the
  // receiver's root proposal says so with kAnchorFlagTransfer, recomputed on
  // every update. Returns false for an unknown element. The element must carry
  // a Containment-role Navigable polygon or a box/collider. With any other
  // definition (none, a surface) the root is never contained in the element and
  // the flag has no effect.
  //
  // The leaf's shape is NOT checked here: shape equivalence is part of the receiver's
  // per-update transfer flag, so a late counterpart, a
  // geometry edit or a counterpart in another process is judged on the update it appears.
  bool SetElementContainmentTransfer(element_id_t id, bool enabled);
  bool IsElementContainmentTransfer(element_id_t id) const;
  // The source-side transfer fact: the shared id
  // of the transfer leaf the SOURCE actor's root was inside at its latest update
  // (kInvalidSharedElementId when outside any). It is what that update computed; nothing
  // else is stored for it. False for an unknown actor.
  bool CopySourceTransferLeaf(spatialblend::entity_id_t actor_id,
                              shared_element_id_t* leaf_shared_id) const;

  // Referent resolution extension point. The default resolves an
  // incoming payload's shared referent id by exact match; installing a custom
  // resolver enables semantic matching / virtual-object materialization.
  void SetReferentResolver(std::shared_ptr<ReferentResolver> resolver);
  const ReferentResolver& GetReferentResolver() const;  // never null

  // Runtime config setters. Each pre-empts the corresponding env-var lazy
  // default (SB_V2_REFERENT_RESOLVE / SB_V2_REALIZATION_POLICY / SB_V2_MIN_MOVE
  // / SB_V2_MIN_YAW) so callers that never touch env (e.g. Unity via the C API)
  // get the same behavior programmatically.
  void SetReferentResolveMode(ReferentResolveMode mode);
  ReferentResolveMode GetReferentResolveMode() const {
    return referent_resolve_mode_;
  }
  void SetRealizationPolicy(RealizationPolicyType type);
  RealizationPolicyType GetRealizationPolicy() const {
    return realization_policy_type_;
  }
  // Movement-gate deadbands (see min_propagation_move_/min_propagation_yaw_
  // below). Values are clamped to >= 0.
  void SetMinMove(double meters);
  void SetMinYaw(double radians);

  // Horizontal reach radius (metres) for a named anchor. An
  // engaged affordance-anchor target is the mapped source contact and is
  // never moved by reach (a clamped point is not the contact); a contact
  // farther than this from the solved root is flagged on the target
  // (RealizationTarget::beyond_reach / reach_excess_m). Applies to any anchor
  // name; anchors without an explicit radius fall back to the default
  // (0.9 m, hand-scale).
  void SetAnchorReach(const std::string& anchor_name, double radius);

  // ---------------------------------------------------------------------
  // Observability and injection. Every getter below takes the space lock and returns a VALUE copy
  // of what the last evaluation recorded -- never a recomputation -- and must
  // not be called from inside a relation-event callback (the space lock is
  // held there). None of them marks scene inputs as escaped.
  //
  // Recording costs allocations, so it is OPT-IN: it runs while the space is
  // traced, or after EnableObservabilityCapture(true). The observability
  // getters enable capture themselves, so the first read after construction
  // may be empty and every evaluation after it is recorded.
  // api/c/trace.format.md documents the matching trace events.
  // ---------------------------------------------------------------------
  void EnableObservabilityCapture(bool enabled) {
    observability_capture_.store(enabled);
  }
  bool ObservabilityCaptureEnabled() const {
    return observability_capture_.load();
  }

  // Declared policy parameters. The policy FAMILY stays owned by
  // SetRealizationPolicy (config.type is ignored here); returns false and
  // changes nothing when either config is invalid
  // (IsValidRealizationPolicyConfig / IsValidImminenceConfig). A change of
  // the receiver schedule bumps the policy revision (the rich snapshot
  // validity domain); a change of the source-only imminence constants bumps
  // only the imminence revision. Either re-emits ev=policy.
  bool SetRealizationPolicyParams(const RealizationPolicyConfig& policy,
                                  const ImminenceConfig& imminence);
  // Effective values, with `type` = the resolved policy family.
  RealizationPolicyConfig GetRealizationPolicyParams() const;
  ImminenceConfig GetImminenceParams() const;

  // One entry per binding of `actor_id`, in binding order.
  struct ScopeObservation {
    size_t binding_index = 0;
    std::string anchor_name;
    std::string tag_filter;
    RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
    ScopeRecord record;
  };
  std::vector<ScopeObservation> GetScopeRecords(
      spatialblend::entity_id_t actor_id) const;

  // The receiver policy's inputs and outputs for the latest
  // evaluation of `actor_id` (payloads = the resolved policy input,
  // weights[i] <-> payloads[i]).
  struct PolicyObservation {
    bool valid = false;
    double time_seconds = 0.0;
    RealizationPolicyConfig config;
    std::vector<RelationPayload> payloads;
    std::vector<double> weights;
    std::vector<RealizationChannelDiagnostics> channels;
  };
  bool CopyPolicyObservation(spatialblend::entity_id_t actor_id,
                             PolicyObservation* out) const;

  // The root solve of the latest evaluation (per-goal witness points
  // before blending, unconstrained blend, feasibility snap).
  // goal_payload_index[k] maps solve.goals[k].payload_index (an index into
  // the ROOT payload subset) back to PolicyObservation::payloads.
  struct SolverObservation {
    bool valid = false;
    double time_seconds = 0.0;
    SolverResult solve;
    std::vector<size_t> goal_payload_index;
    // The scene-level answer before the containment blend: the clamped S0 and the
    // Facing yaw. `solve` carries the blended pose (what the proposal is made of).
    Eigen::Vector2d scene_position = Eigen::Vector2d::Zero();
    double scene_yaw = 0.0;
  };
  bool CopySolverObservation(spatialblend::entity_id_t actor_id,
                             SolverObservation* out) const;

  // Per-actor continuation counters. The continuation SEED is not kept here:
  // it is the embodied pose sampled from the actor's port on every solve;
  // core keeps no previous-target pose.
  struct SolverWarmState {
    // Root solves recorded / solves whose search actually ran the embodied
    // seed as a restart (SolverResult::objective_warm_start_used).
    uint64_t solves = 0;
    uint64_t warm_started_solves = 0;
  };
  bool CopySolverWarmState(spatialblend::entity_id_t actor_id,
                           SolverWarmState* out) const;

  // Drive the receiver of `actor_id` with a fixed relation snapshot, as
  // if it had arrived from a peer at `time_seconds` (legacy timing), without
  // running a source extractor. Referents resolve through this space's
  // normal resolution (shared id; no linker handle is attached). Returns
  // false for an unknown actor or a non-finite time. Advances the space
  // clock to max(clock, time_seconds).
  bool InjectRelationPayloads(spatialblend::entity_id_t actor_id,
                              std::vector<RelationPayload> payloads,
                              double time_seconds);

  // Pure Extract_p for one (kernel, anchor pose, referent): the same
  // GroundingRule::ComputeSupport + RelationKernel::Encode the binding
  // pipeline runs, with the same normalizing navigable: the deepest
  // containment element under `scope_root` containing the anchor, then the
  // nearest Navigable ancestor of it. With explicit_scope the walk stops at
  // `scope_root` (sb_bind_scoped bindings); without it the walk continues to
  // the tree root, and `scope_root` must be what an owner-derived binding
  // uses: the actor element's nearest containment ancestor.
  // scope_root == kInvalidElementId evaluates without a navigable
  // (Euclidean, unnormalized).
  struct RelationEvaluation {
    Support support;
    GroundedRelation grounded;
    element_id_t deepest_containing = kInvalidElementId;
    element_id_t navigable_element = kInvalidElementId;
  };
  bool EvaluateRelation(RelationKernelType kernel_type,
                        const Eigen::Vector3d& anchor_position,
                        double anchor_yaw, element_id_t referent,
                        element_id_t scope_root, bool explicit_scope,
                        const std::vector<StateType>& dof,
                        RelationEvaluation* out) const;

  // Target-side Eval_p of a received payload at a candidate
  // anchor pose (EvaluateTargetRelation, relation_evaluator.h), resolving
  // the referent like OnRemoteUpdate's shared-id path (no linker handle).
  // `normalizing_scope` != kInvalidElementId overrides the normalizing
  // context. Locks the space; false only for a non-finite pose.
  void ResolveReceiverIdentitiesLocked(RelationPayload* payload) const;
  // True when `source` was linked, unshared or unlinked on this space and has
  // not been resolved anew since: its relations must not resolve through the
  // shared-id fallback onto whatever element once stood for it.
  bool SourceRetiredLocked(space_id_t space, element_id_t element) const;
  // Drops the retained receiver evaluations driven by this source actor.
  void ClearRetainedEvaluationsForSourceLocked(const element_handle_t& source,
                                               shared_element_id_t shared_id);
  bool EvaluateTargetRelation(const RelationPayload& payload,
                              const TargetAnchorPose& anchor,
                              element_id_t normalizing_scope,
                              TargetRelationEvaluation* out) const;

  // World transform of a tree element (composition of ancestors'
  // local transforms) and its parent. False for an unknown element.
  bool GetElementWorldTransform(element_id_t elem_id,
                                Eigen::Matrix4d* out) const;
  bool GetElementParent(element_id_t elem_id, element_id_t* parent) const;
  // Locked value copy of GetRealizationTargets() (same validity rules).
  bool CopyRealizationResult(spatialblend::entity_id_t actor_id,
                             RealizationResult* out) const;

  // Forget which policy revision this space already traced, so the next
  // traced update re-emits ev=policy (called when the trace sink is rebound).
  void OnTraceRebound();


 private:
  // Internal borrowed access. Callers serialize mutation using space_mutex_.
  Actor* FindActorLocked(spatialblend::entity_id_t actor_id);
  const Actor* FindActorLocked(spatialblend::entity_id_t actor_id) const;
  Binding* CreateBindingLocked(spatialblend::entity_id_t actor_id,
                                const std::string& anchor,
                                RelationKernelType kernel,
                                element_id_t scope,
                                const std::string& tag_filter);
  void CaptureSourceInteractionEvidenceLocked(spatialblend::entity_id_t actor_id);
  bool EnableSourceInteractionEvidenceLocked(spatialblend::entity_id_t actor_id,
                                              uint64_t epoch, bool enabled);
  void InvalidateSourceEvidencePhaseLocked(spatialblend::entity_id_t actor_id);
  // Feed one relation observation into the phase tracker and dispatch any
  // resulting transition events to the registered callback.
  void EmitRelationTransitions(
      const RelationEvent& identity, double proximity, double weight,
      double timestamp, const std::string& owner = "");
  void DispatchRelationEvents(const std::vector<RelationEvent>& events,
                              double proximity);
  struct SourceRelationObservation {
    element_id_t referent = kInvalidElementId;
    RelationEvent identity;
    double proximity = 0.0;
    double weight = 0.0;
  };
  // Retain only phases whose exact source relation is present in this complete
  // evaluation. Missing observations are withdrawal, not low-proximity samples.
  void ReconcileSourceRelationPhasesLocked(
      spatialblend::entity_id_t actor_id,
      const std::vector<SourceRelationObservation>& current, double timestamp);
  // Facing radius kind: true for a person -- another actor's observed
  // facing point, an actor's element, or a part under an actor's element.
  bool IsActorReferentLocked(
      element_id_t element_id,
      const std::unordered_set<element_id_t>& actor_element_ids) const;
  // Internal-only phase owner. It deliberately does not reuse the public
  // shared actor id, which may collide across local actors.
  std::string PhaseOwner(spatialblend::entity_id_t actor_id) const;

  // Caller holds space_mutex_. Legacy callers retain call-timed behavior;
  // timestamped observations provide their own source time and elapsed delta.
  // A send staged under space_mutex_ and delivered after it is released.
  // SpaceHub::Propagate dispatches synchronously into the receiving space's
  // OnRemoteUpdate, which takes that space's own mutex; propagating while
  // holding ours let two spaces take each other's mutex in opposite order.
  struct PendingPropagation {
    bool has_send = false;
    RemoteRelationUpdate update;
    SpaceHub* hub = nullptr;
    space_id_t space_id = kInvalidSpaceId;
    double clock = 0.0;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double yaw = 0.0;
  };
  // Delivers staged sends in order, each followed by the `ev=src` trace that
  // describes it. Must be called with space_mutex_ released and
  // propagation_mutex_ held.
  static void DeliverPropagations(const std::vector<PendingPropagation>& pending);

  // `pending` collects sends instead of them being dispatched inline; the
  // caller delivers after releasing space_mutex_. A batch entry point appends
  // one per frame, so this is a vector rather than a single slot.
  void EvaluateActorStateLocked(spatialblend::entity_id_t actor_id,
                                const Eigen::Vector2d& position, double yaw,
                                bool timestamped_observation,
                                double source_time, double elapsed_seconds,
                                std::vector<PendingPropagation>* pending,
                                bool source_observation_sequence_known = false);
  void ResetObservationStateLocked(spatialblend::entity_id_t actor_id);
  // Containment transfer helpers (space.cc, caller holds space_mutex_).
  // Source: this tick's observation for the root pose; nullopt when the root
  // is not contained in a flagged element. Updates actor_containment_.
  std::optional<ContainmentObservation> UpdateSourceContainmentLocked(
      spatialblend::entity_id_t actor_id, const Eigen::Vector2d& position,
      bool* transitioned);
  // Stores `next` as the actor's last observed contained identity. Returns true
  // when the contained flag or the containing element changed (the source sends
  // that tick even when every relation is dormant, so the receiver learns it).
  bool SetActorContainmentLocked(spatialblend::entity_id_t actor_id,
                                 const ActorContainment& next);
  // Observation reset or new epoch: the actor has no contained root any more.
  void ClearActorContainmentLocked(spatialblend::entity_id_t actor_id);
  // Resolves environment-derived policy once, at construction.
  void ResolveBootstrapPolicy();

  // Stable actor identity for relation events/phase-tracker keys: shared id
  // when the actor's element is shared, otherwise the local element id
  // (single-space setups). Factored out of UpdateActorState so
  // GetRelationStates can look up the SAME phase-tracker key that was used
  // when the phase was last recorded (see EmitRelationTransitions call
  // sites) — must not diverge, or "active" would silently read the wrong
  // (or an always-absent) phase entry.
  shared_element_id_t GetActorSharedIdForEvents(
      spatialblend::entity_id_t actor_id) const;

  // Appends per-anchor affordance-pose targets (hysteresis-gated, reach-
  // clamped around the already-computed root) to `realization.targets`,
  // which must already contain the root target as targets[0].
  // `root_position`/`root_yaw` are the
  // JUST-computed root pose for THIS frame (used both as the reach-clamp
  // center and as SolveActorTargets' internal facing-evaluation point).
  // `payloads`/`weights` are the FULL, unfiltered per-frame vectors (this
  // function does its own affordance-only grouping via SolveActorTargets).
  void AppendAnchorTargets(spatialblend::entity_id_t actor_id,
                           const Eigen::Vector2d& root_position,
                           double root_yaw,
                           const std::vector<RelationPayload>& payloads,
                           const std::vector<double>& weights,
                           const NavigableSpatialDefinition* navigable_def,
                           RealizationResult& realization);

  void AppendFacingTargets(spatialblend::entity_id_t actor_id,
                           const std::vector<RelationPayload>& payloads,
                           const std::vector<double>& weights,
                           const NavigableSpatialDefinition* navigable_def,
                           RealizationResult& realization);
  struct SceneInputStamp {
    uint64_t tree = 0;
    uint64_t support = 0;
    bool known = false;
  };
  SceneInputStamp CaptureSceneInputStampLocked() const;
  struct PreparedActorObservation {
    Eigen::Vector2d position;
    double yaw;
    double elapsed;
  };
  bool ValidateActorObservationFrameLocked(
      const ActorObservationFrame& frame) const;
  PreparedActorObservation CommitActorObservationFrameLocked(
      const ActorObservationFrame& frame, int64_t receipt);
  bool ApplyActorObservationBatchWithReceipt(
      std::vector<ActorObservationFrame> frames,
      std::vector<ObservedFacingPointPublication> publications,
      int64_t receipt);
  struct PreparedObservedFacingPoint {
    ObservedFacingPointUpdate update;
    Eigen::Matrix4d local_transform;
    bool continuous = false;
  };
  bool PrepareObservedFacingPointLocked(
      const ObservedFacingPointUpdate& update, int64_t now,
      PreparedObservedFacingPoint* prepared,
      const ActorObservationFrame* prospective_frame = nullptr,
      int64_t admission_receipt = 0) const;
  void CommitObservedFacingPointLocked(
      const PreparedObservedFacingPoint& prepared);
  struct ObservedFacingPoint {
    ObservedFacingPointKind kind = ObservedFacingPointKind::kSourceAnchor;
    spatialblend::entity_id_t producer = spatialblend::kInvalidEntityId;
    element_id_t producer_element = kInvalidElementId;
    shared_element_id_t logical_actor = kInvalidSharedElementId;
    shared_element_id_t referent_event_id = kInvalidSharedElementId;
    std::string anchor;
    uint64_t lease = 0;
    uint64_t revision = 0;
    element_handle_t source_point;
    uint64_t source_point_lease = 0;
    int64_t maximum_age_ns = 0;
    uint64_t source_epoch = 0;
    int64_t source_sequence = -1;
    int64_t observed_at_ns = 0;
    int64_t valid_until_ns = 0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Matrix4d committed_world = Eigen::Matrix4d::Identity();
    bool active = false;
    bool retired = false;
  };
  struct ObservedFacingDependency {
    element_id_t element = kInvalidElementId;
    uint64_t lease = 0;
    uint64_t revision = 0;
    int64_t valid_until_ns = 0;
    // Receiver phase uses incoming public IDs, which need not equal local IDs.
    RelationEvent phase_identity;
  };
  bool IsObservedFacingPointLiveLocked(element_id_t element, int64_t now) const;
  bool IsObservedFacingCandidateLocked(element_id_t element,
                                       spatialblend::entity_id_t actor,
                                       RelationKernelType kernel,
                                       int64_t now) const;
  bool AreObservedFacingDependenciesLiveLocked(
      spatialblend::entity_id_t actor, int64_t now) const;
  void PruneObservedFacingPointsLocked();
  void InvalidateObservedFacingConsumersLocked(element_id_t element,
                                               bool preserve_history = false);
  void WithdrawObservedFacingPointsBelowLocked(element_id_t ancestor);
  std::unordered_map<element_id_t, ObservedFacingPoint> observed_facing_points_;
  std::unordered_map<spatialblend::entity_id_t,
                     std::vector<ObservedFacingDependency>> observed_facing_dependencies_;
  void AdvanceSceneInputsLocked(bool tree, bool support, bool geometry = true);
  // `actor_pose`: the move is an ACTOR's own pose (the actor-grounded leaf
  // element follows the actor): it advances the scene-input stamps but not the
  // geometry revision, which is what SCENE_CHANGE keys to.
  // Records the status of the most recent scene mutator (space_mutex_ held).
  void RecordSceneErrorLocked(SceneErrorCode code, std::string message);
  bool SetElementTransformLocked(element_id_t id,
                                 const Eigen::Matrix4d& transform,
                                 bool actor_pose = false);
  ReceiverInputStamp CurrentReceiverInputLocked(
      spatialblend::entity_id_t actor_id) const;
  void InvalidateReceiverInputLocked(spatialblend::entity_id_t actor_id);
  ReceiverApproachContextStatus ValidateReceiverApproachContextLocked(
      const ReceiverApproachContext& context) const;
  // `publish_released = false`: the caller publishes the one proposal of this
  // solve itself (a NONE published first would read as "released" to a driver).
  void ClearReceiverTargetsLocked(spatialblend::entity_id_t actor_id,
                                   bool preserve_history = false,
                                   bool publish_released = true);
  void EvaluateRemoteRelationLocked(const RemoteRelationUpdate& update,
                                    spatialblend::entity_id_t actor_id,
                                    int64_t original_ingress_ns,
                                    bool receiver_reevaluation,
                                    uint64_t source_actor_resolution_revision = 0,
                                    bool carries_containment = true);
  // Copy accepted remote payloads into the received actor's bindings so a
  // query sees the same R and referent the source sent. The public phase
  // stays in the phase tracker (EmitRelationTransitions). One row per
  // referent. Each delivery replaces every binding's rows, including a
  // binding the snapshot does not mention, so an empty snapshot, a
  // deactivated anchor, or an unshared referent leaves no stale row.
  void RecordRemoteBindingMatchesLocked(
      spatialblend::entity_id_t actor_id,
      const std::vector<RelationPayload>& payloads);
  void CaptureRealizationSnapshot(
      spatialblend::entity_id_t actor_id,
      const std::vector<RelationPayload>& incoming_payloads,
      const std::vector<int>& resolved_index,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const RealizationResult& realization,
      const SceneInputStamp& scene_inputs,
      const RemoteRelationUpdate* source_update = nullptr,
      int64_t receiver_ingress_monotonic_ns = 0,
      bool counterpart_revision_known = false,
      uint64_t counterpart_revision = 0,
      const ReceiverInputStamp& receiver_input = {});

  void ResolveRealizationPolicyLocked();
  void ResolveReferentResolveModeLocked();
  void ObserveCounterpartGraphLocked(uint64_t graph_revision);
  bool CaptureCounterpartRevisionLocked(uint64_t graph_revision_before,
                                        uint64_t graph_revision_after,
                                        uint64_t* revision);
  bool CurrentCounterpartRevisionLocked(uint64_t* revision);
  static bool IsValidRealizationPolicy(RealizationPolicyType type);

  // Horizontal reach radius (metres) for an anchor name; falls back to
  // default_anchor_reach_ when not explicitly set via SetAnchorReach.
  double GetAnchorReach(const std::string& anchor_name) const;
  // Clears state keyed by elements about to be removed from a receiver-owned
  // replica subtree. Actors rooted there become unassigned; their bindings
  // and cached controller/realization state cannot outlive the tree nodes.
  void CleanupRemovedElements(const std::vector<element_id_t>& removed_ids);

  // Reparent direct children before removing a source-owned replica. Children
  // represent independently authored/local or independently shared nodes and
  // keep their world transforms while their container disappears.
  void PreserveIndependentChildren(element_id_t removed_root);

  spatialblend::entity_id_t FindActorForRemoteUpdateLocked(
      const RemoteRelationUpdate& update) const;

  std::function<int64_t()> receiver_input_clock_ = ReceiverInputNowNs;
  std::unordered_map<spatialblend::entity_id_t, ReceiverInputState>
      receiver_inputs_;
  // Latest registration per actor.  Replacements retain a monotonic revision
  // so a consumer cannot accidentally treat a newer context as its old one.
  std::unordered_map<spatialblend::entity_id_t, ReceiverApproachContext>
      receiver_approach_contexts_;
  uint64_t next_receiver_approach_context_revision_ = 1;
  mutable std::mutex receiver_approach_feasibility_mutex_;
  mutable std::unordered_map<spatialblend::entity_id_t,
                             ReceiverApproachFeasibilitySnapshot>
      receiver_approach_feasibility_snapshots_;
  mutable uint64_t next_receiver_approach_feasibility_revision_ = 1;
  std::unordered_map<spatialblend::entity_id_t, ReceiverInputStamp>
      receiver_solve_inputs_;
  struct RetainedReceiverEvaluation {
    RemoteRelationUpdate update;
    int64_t original_ingress_ns = 0;
    element_id_t receiver_element = kInvalidElementId;
    // A revoked explicit actor link must not fall back to coincident shared ID.
    bool requires_actor_link = false;
    // Separate from actionable-target storage: an empty/failed evaluation
    // also consumes this input and cannot generate unlimited rich revisions.
    ReceiverInputStamp last_evaluated_input;
  };
  std::unordered_map<spatialblend::entity_id_t, RetainedReceiverEvaluation>
      retained_receiver_evaluations_;
  // Core controllers never actuate the external-input path.
  ElementTree element_tree_;
  uint64_t tree_revision_ = 1;
  uint64_t support_revision_ = 1;
  // Advances with every receiver scene edit EXCEPT an actor's own pose (the avatar
  // mirror and the actor-grounded leaf move every frame): the stamp the proposal's
  // SCENE_CHANGE cause is keyed to. Spatial-definition, structure, tag, anchor and
  // non-actor element-transform edits advance it.
  uint64_t geometry_revision_ = 1;
  // Permanent: a borrowed reference can outlive any later configuration reset.
  mutable std::atomic<bool> scene_inputs_escaped_{false};
  std::unordered_map<spatialblend::entity_id_t, std::unique_ptr<Actor>> actors_;
  std::unordered_map<spatialblend::entity_id_t, std::vector<Binding>> bindings_;
  spatialblend::entity_id_t next_actor_id_ = 1;
  space_id_t space_id_ = kInvalidSpaceId;
  SpaceHub* hub_ = nullptr;
  double update_interval_ = 0.033;  // default 30fps
  RelationSolver solver_;
  std::vector<RemoteRelationUpdate> received_updates_;
  uint64_t received_update_count_ = 0;

  std::unordered_map<spatialblend::entity_id_t, ActorMotionTracker>
      motion_trackers_;
  // Per-actor mailbox, reachable without space_mutex_: the driver's embodied
  // port (the driver is its only writer) and the latest proposal (written by
  // the solve, read by the driver), both seqlocks: reads take no lock and never
  // block the writer. mailboxes_mutex_ guards only the map (a hash lookup or an
  // insert/erase), is never held across a solve, a publish or a read of a box.
  struct ActorMailbox {
    EmbodiedPort port;  // the ROOT anchor's port (anchor id 0)
    // Interned anchors and their ports. Index = anchor id; entry 0 is the root.
    // Guarded by anchors_mutex (never held across a solve).
    std::mutex anchors_mutex;
    std::vector<std::string> anchor_names;
    std::vector<std::unique_ptr<EmbodiedPort>> anchor_ports;
    // ABI 4.7: the registered embodiment (set once, immutable) and one port per bone. Guarded by anchors_mutex
    // for the pointer/vector; the ports are lock-free.
    std::shared_ptr<const Embodiment> embodiment;
    std::vector<std::unique_ptr<BonePort>> bone_ports;
    ActorMailbox() {
      anchor_names.emplace_back("root");
      anchor_ports.emplace_back(nullptr);
    }
    // Events for the app, polled in order.
    std::mutex events_mutex;
    std::deque<ActorEvent> events;
    std::atomic<bool> target_events_enabled{false};
    // The actor's embodied pose comes from a dedicated receiver-input stream
    // (BeginReceiverInputEpoch): the public port refuses direct publishes so
    // the stamped frame stays the single owner of the pose.
    std::atomic<bool> input_owned{false};
    ProposalBox proposal;
  };
  std::shared_ptr<ActorMailbox> MailboxFor(spatialblend::entity_id_t actor_id) const;
  mutable std::mutex mailboxes_mutex_;
  std::unordered_map<spatialblend::entity_id_t, std::shared_ptr<ActorMailbox>>
      mailboxes_;
  // Attribution cache for the next proposal: derived from the source stream and
  // the sample it was computed from, keyed to the sample epoch and dropped on a
  // source reset or a driver epoch change. Holds no pose.
  // ONE STATE PER ANCHOR (the anchor is a parameter of the relation definition, so the cause
  // bits are one rule for every anchor, the root included): the previous proposal of one anchor as the causes compare
  // it, and the facts of this proposal (AttributeAnchorCauses, anchor_proposals.cc).
  struct AnchorCauseState {
    bool has_prev = false;
    uint64_t sample_epoch = 0;
    uint64_t teleport_sequence = 0;  // last TELEPORTED sample already attributed
    std::vector<std::pair<shared_element_id_t, int>> signature;
    shared_element_id_t driving_referent = kInvalidSharedElementId;
    int driving_kernel = -1;
    std::string driving_tag;  // the driving pose of a pose relation ("" = none)
    bool stuck = false;       // the previous proposal proposed nothing (NONE or NO_SAMPLE)
    bool engaged = false;
    bool contained = false;
    shared_element_id_t contain_scope = kInvalidSharedElementId;  // the contained frame
    uint64_t geometry_revision = 0;
    int facing_sign = 0;
    bool facing_tie = false;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    bool has_position = false;
    int64_t clamp_branch = 0;
    uint64_t clamp_exempt_hash = 0;
    // The previous solve's decoded approach points and softmax weights (position goals), by (referent, kernel), and its
    // yaw inputs.
    std::vector<ProposalGoal> goals;
    std::vector<ProposalYawGoal> yaw_goals;
  };
  struct AnchorCauseFacts {
    bool sampled = false;
    uint64_t sample_epoch = 0, sample_sequence = 0;
    bool teleported = false;
    std::vector<std::pair<shared_element_id_t, int>> signature;
    shared_element_id_t driving_referent = kInvalidSharedElementId;
    int driving_kernel = -1;
    std::string driving_tag;
    bool stuck = false;
    bool engaged = false;
    bool contained = false;
    shared_element_id_t contain_scope = kInvalidSharedElementId;
    bool exception = false;
    // From the relation solve, when the anchor's relations were solved by it.
    bool solved = false;
    std::vector<ProposalGoal> goals;
    std::vector<ProposalYawGoal> yaw_goals;
    std::vector<SolverGoalWitness> witnesses;
    int facing_sign = 0;
    bool facing_tie = false;
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    int64_t clamp_branch = 0;
    uint64_t clamp_exempt_hash = 0;
    bool progress = false, chart_snap = false, no_request = false;
  };
  // One anchor's Spacing / Facing relations solved in the room's scope from its sample (the root's and every other
  // anchor's: one path). Applies the leaf-relative yaw. The caller holds space_mutex_.
  SolverResult SolveRoomScopeLocked(const EmbodiedSample& sample, const std::vector<RelationPayload>& payloads,
                                    const std::vector<double>& weights, const NavigableSpatialDefinition* navigable_def,
                                    bool placed, bool cold, spatialblend::entity_id_t actor_id = 0,
                                    uint32_t anchor_id = 0);
  // The anchor's previous turn direction: the sign of the last non-zero yaw turn the
  // receiver itself PROPOSED on a continuation (the requested turn, yaw_turn, which is above pi the long way round), 0
  // before it has turned. Sample noise, a
  // teleport and a placed answer do not set it.
  struct YawTrack {
    int turn_sign = 0;
  };
  std::map<std::pair<spatialblend::entity_id_t, uint32_t>, YawTrack> yaw_track_;
  // The containment relation of one anchor (containment transfer is a relation, of any
  // anchor): a Spacing relation with weight whose referent is a containment-transfer leaf here and whose source anchor
  // is inside it (support.has_interior). Null when there is none; `leaf` receives the counterpart.
  const RelationPayload* ContainmentRelationLocked(const std::vector<RelationPayload>& payloads,
                                                   const std::vector<double>& weights, const Element** leaf) const;
  // The leaf-relative yaw of an anchor inside a transfer leaf: the receiver's leaf yaw plus the
  // yaw the source anchor holds relative to its own leaf (the Spacing kernel's encoded relative yaw), as v1's
  // inside-volume snap carries the yaw relative to the volume. False when the anchor has no containment relation.
  bool LeafRelativeYawLocked(const std::vector<RelationPayload>& payloads, const std::vector<double>& weights,
                             double* yaw) const;
  // ABI 4.14: the replay through the chart of one anchor, a separate output on its proposal
  // (AnchorProposal::has_chart and chart_*): the source's AnchorChartObservation of this anchor mapped through this
  // space's counterpart of the leaf, when that counterpart is declared a transfer leaf. `is_root`: the receiver anchor is
  // the actor's root (matched by a source anchor named "root" or unnamed, or `name`).
  void ApplyAnchorChartLocked(const RemoteRelationUpdate& update, const std::string& name, bool is_root,
                              AnchorProposal* ap) const;
  // The facts of a relation solve of one anchor's relations (`payloads` / `weights` as passed to the solve).
  static AnchorCauseFacts SolveCauseFacts(const SolverResult& r, const std::vector<RelationPayload>& payloads,
                                          const std::vector<double>& weights);
  // The cause bits of one anchor's proposal against its previous one; `commit` stores this proposal as the previous.
  static uint32_t AttributeAnchorCauses(AnchorCauseState* state, const AnchorCauseFacts& now, uint64_t geometry_revision,
                                        bool commit);
  struct ProposalAttribution {
    std::map<uint32_t, AnchorCauseState> anchors;  // by anchor id, the root (0) included
    bool AnyPrev() const {
      for (const auto& [id, st] : anchors)
        if (st.has_prev) return true;
      return false;
    }
  };
  std::unordered_map<spatialblend::entity_id_t, ProposalAttribution>
      proposal_attribution_;
  uint64_t proposal_revision_ = 0;
  // Source side: an exception raised by the app rides the next propagated update.
  struct PendingException {
    uint32_t id = 0;
    std::vector<uint8_t> payload;
  };
  std::unordered_map<spatialblend::entity_id_t, PendingException>
      pending_source_exception_;
  // Per receiver actor: the latency timers of the interactions its sources hold.
  std::unordered_map<spatialblend::entity_id_t, InteractionLatencyTracker>
      latency_trackers_;
  InteractionLatencyPolicy latency_policy_ = DefaultInteractionLatencyPolicy();
  // Select the retained target before evaluating diagnostics or child scopes.
  // This does not change the current sample used as the solve's origin.
  void RetainIdlePositionLocked(spatialblend::entity_id_t actor_id,
                                AnchorProposal* anchor);
  void PublishProposalLocked(spatialblend::entity_id_t actor_id,
                             ReceiverProposal proposal,
                             std::vector<ActorEvent> events = {});
  // ABI 4 helpers (anchor_proposals.cc). The caller holds space_mutex_.
  static uint32_t InternAnchor(ActorMailbox& box, const std::string& name,
                               const std::string& root_name);
  std::string RootAnchorNameLocked(spatialblend::entity_id_t actor_id) const;
  // The sample of one anchor of a receiver actor: its port, or the stamped
  // input frame of an external receiver. False (and an invalid sample) when none.
  bool SampleAnchorLocked(spatialblend::entity_id_t actor_id,
                          const std::string& name, const std::string& root_name,
                          EmbodiedSample* out) const;
  // Store the matching snapshot before appending its complete publication
  // batch; observers run after this unlocks.
  void AppendActorEvents(spatialblend::entity_id_t actor_id,
                         std::vector<ActorEvent> events);
  // Appends the proposal of every relation-bearing anchor other than the root to `set->anchors`. `placed`: the actor's
  // first proposal or a source exception (every anchor is placed then). `root_cause` is the root proposal's cause,
  // which an anchor whose scope is instantiated over the root's proposal carries (the scope order).
  void BuildAnchorProposalsLocked(
      spatialblend::entity_id_t actor_id, const RemoteRelationUpdate& update,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const RealizationResult& realization,
      const NavigableSpatialDefinition* navigable_def,
      const std::string& root_name, bool receiver_reevaluation, bool placed,
      uint32_t root_cause, ReceiverProposal* set);
  // Steps the latency timers of the interactions the source holds and queues a
  // teleport request for each one that expired without the target arriving.
  void EvaluateInteractionLatencyLocked(
      spatialblend::entity_id_t actor_id, const RemoteRelationUpdate& update,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const std::string& root_name,
      const ReceiverProposal& set, uint64_t revision,
      std::vector<ActorEvent>* events);
  void SyncEmbodiedMirrorsLocked();
  // The body of every person in this space, for the Facing sightline (OcclusionScene's body_tops): an actor's
  // element with no declared geometry is the actor's body, whose top is its highest anchor above the element.
  std::unordered_map<element_id_t, double> ActorBodyTopsLocked() const;
  // Actors that have received a remote update: only these are receiver
  // avatars whose element-tree mirror follows the driver's sample.
  std::unordered_set<spatialblend::entity_id_t> receiving_actors_;
  uint64_t next_proposal_observer_id_ = 1;
  std::vector<std::pair<uint64_t, ProposalObserver>> proposal_observers_;
  // (epoch, sequence) of the sample last mirrored into the element tree.
  std::unordered_map<spatialblend::entity_id_t, std::pair<uint64_t, uint64_t>>
      mirrored_sample_;
  // Obstacle clearance for root feasibility/paths (SetBodyRadius).
  // Unset (NaN) unless sb_set_body_radius sets the space-wide radius of actors that register none. The core does not
  // use it (its free space has clearance 0); it serves the driver (GetBodyRadius, the driver course queries).
  double body_radius_ = std::numeric_limits<double>::quiet_NaN();
  // The radius each actor's embodiment registered (RegisterEmbodiment); an actor not in the map uses body_radius_.
  std::unordered_map<spatialblend::entity_id_t, double> actor_body_radius_;
  double BodyRadiusOfLocked(spatialblend::entity_id_t actor_id) const;
  // The clearance a DRIVER course keeps for this actor's body: its registered radius, 0 when none is registered.
  double DriverClearanceLocked(spatialblend::entity_id_t actor_id) const;
  // Pushes every kObstacle element's world footprint into the navigable
  // definitions it overlaps (no-op when unchanged). Called at the start of
  // source evaluation and receiver solve.
  void SyncNavigableObstaclesLocked();
  const NavigableSpatialDefinition* FindActorNavigableLocked(
      spatialblend::entity_id_t actor_id) const;
  // Obstacle-sync cache (item: rebuild only on change).
  bool obstacle_sync_valid_ = false;
  uint64_t obstacle_sync_tree_ = 0;
  uint64_t obstacle_sync_support_ = 0;
  std::vector<double> obstacle_fingerprint_;
  std::unordered_map<spatialblend::entity_id_t, RealizationResult>
      realization_results_;
  // Whether the actor's latest solve was a PLACED answer (a first proposal or a source exception): the placement reads
  // every relation whatever its weight, so the realization snapshot then reports a relation
  // without a valid count as included, not disabled.
  std::map<spatialblend::entity_id_t, bool> placed_solve_;
  bool rich_realization_enabled_ = false;
  std::unordered_set<spatialblend::entity_id_t> rich_realization_actors_;
  std::unordered_map<spatialblend::entity_id_t, ActorRealizationSnapshot>
      realization_snapshots_;
  // Previous realization weights per actor, keyed by (referent_shared_id,
  // kernel_type), for temporal EMA smoothing. The realized target is a weighted
  // blend of (possibly distant) supports, so an unsmoothed frame-to-frame
  // weight swing sweeps the target across the gap → visible oscillation. Keyed
  // (not positional) because the active payload set/order varies between
  // frames.
  std::unordered_map<spatialblend::entity_id_t, std::vector<double>>
      previous_weights_;
  // EMA factor for the above; <0 means "not yet resolved from env" (lazy).
  double weight_smoothing_alpha_ = -1.0;

  // Per-anchor-name horizontal reach radius (metres); see SetAnchorReach.
  std::unordered_map<std::string, double> anchor_reach_radii_ = {
      {"hand", 0.9}, {"pelvis", 0.35}};
  double default_anchor_reach_ = 0.9;
  // Receiver actor's own source time (see EvaluateRemoteRelationLocked).
  // One clock per receiver actor epoch: its source is latched by the epoch's first update.
  struct ReceiverActorClock {
    enum class Source { kUnset, kSampleTime, kSourceClock, kSteps };
    Source source = Source::kUnset;
    bool timestamped = false;
    uint64_t epoch = 0;
    double seconds = 0.0;
  };
  std::unordered_map<spatialblend::entity_id_t, ReceiverActorClock> receiver_actor_seconds_;
  // Source-update cadence per receiver actor, describing the update
  // being evaluated. An anchor's entry qualification restarts when it missed
  // a source update, on a delivery-sequence gap, on a change of clock domain
  // (timestamped <-> legacy, or epoch), on a time regression, and on any
  // interval longer than half the stillness window that the sender did not
  // assert quiet (quiet_since_last_send on a legacy update with a
  // contiguous, known sequence).
  struct ReceiverCadence {
    uint64_t updates = 0;         // source updates evaluated
    int64_t last_sequence = 0;    // last delivery sequence (0 = none seen)
    bool sequence_known = false;  // this update carries a sequence (> 0)
    bool sequence_gap = false;    // this update skipped a delivery sequence
    bool timestamped = false;     // this update is a timestamped observation
    uint64_t epoch = 0;           // its observation epoch (0 = legacy)
    bool quiet = false;           // its quiet_since_last_send assertion
  };
  std::unordered_map<spatialblend::entity_id_t, ReceiverCadence>
      receiver_actor_updates_;
  // Per-update time step assumed for a delivery carrying no time source.
  static constexpr double kClocklessUpdateSeconds = 1.0 / 30.0;
  double ActorSecondsLocked(spatialblend::entity_id_t actor_id) const;
  static bool InvalidPolicyWeights(const std::vector<double>& weights);
  // Effective receiver policy configuration generation.  This only changes
  // when a solve-affecting effective option changes, never per solve.
  uint64_t policy_revision_ = 1;
  // Next Binding::revision (space-wide, never reused, 0 = none).
  uint64_t next_binding_revision_ = 1;
  // Declared policy parameters (production defaults: γ = 2, a recorded
  // deviation; see realization_policy.h) and source imminence constants.
  RealizationPolicyConfig realization_policy_config_ =
      ProductionRealizationPolicyConfig();
  ImminenceConfig imminence_config_;
  // Observation (latest evaluation per actor; see the public getters).
  std::unordered_map<spatialblend::entity_id_t, PolicyObservation>
      policy_observations_;
  std::unordered_map<spatialblend::entity_id_t, SolverObservation>
      solver_observations_;
  // Continuation counters (see SolverWarmState): independent of tracing.
  std::unordered_map<spatialblend::entity_id_t, SolverWarmState>
      solver_warm_states_;

  // Source-only imminence constants generation (not a rich validity domain).
  uint64_t imminence_revision_ = 1;
  // (policy, imminence) revisions last written as ev=policy (0 = never).
  uint64_t traced_policy_revision_ = 0;
  uint64_t traced_imminence_revision_ = 0;
  void EmitPolicyTraceIfChangedLocked(double t);
  // Opt-in observation capture (see EnableObservabilityCapture).
  std::atomic<bool> observability_capture_{false};
  bool ObservabilityCaptureActiveLocked() const;
  int64_t SharedOrLocalIdLocked(element_id_t id) const;
  std::vector<RelationState> GetRelationStatesLocked(
      spatialblend::entity_id_t actor_id) const;

  // Heartbeat counter per actor (frames since last propagation)
  std::unordered_map<spatialblend::entity_id_t, int> heartbeat_counters_;
  // Every source tick since this actor's last send was skipped only by
  // the movement deadband (RemoteRelationUpdate::quiet_since_last_send).
  // Cleared at the start of each tick and restored only by a deadband skip,
  // so a tick that never reaches the gate cannot be counted as quiet.
  std::unordered_map<spatialblend::entity_id_t, bool> quiet_since_send_;
  // A tracked-anchor pose or availability transition must bypass root-pose
  // deadband once, so stationary hands cannot leave remote targets stale.
  std::unordered_set<spatialblend::entity_id_t> anchor_observation_dirty_;

  // Movement gate (mirrors v1 Space::ShouldProcessStateUpdate /
  // runtime_policy.minimal_state_change_threshold). The actor pose we last
  // actually propagated; a relation update is sent only when the actor moves
  // beyond a deadband from this pose (or on heartbeat). A stationary source's
  // mm-scale jitter therefore produces ZERO propagation — the clone holds its
  // pose by construction instead of re-grounding and propagating the jitter.
  std::unordered_map<spatialblend::entity_id_t, std::pair<Eigen::Vector2d, double>>
      last_propagated_pose_;
  // Actors whose last propagated snapshot carried payloads: the next frame with
  // no relation in scope sends one empty snapshot so the receiver releases.
  std::unordered_set<spatialblend::entity_id_t> nonempty_snapshot_sent_;
  // Deadbands for the movement gate, resolved at construction (see
  // ResolveBootstrapPolicy); <0 only after SetMovementGate asks for a reset.
  // Defaults are v1's gate (kDefaultMinMoveM / kDefaultMinYawRad); env SB_V2_MIN_MOVE / SB_V2_MIN_YAW.
  double min_propagation_move_ = -1.0;  // metres
  double min_propagation_yaw_ = -1.0;   // radians
  // The construction-time resolution of the two deadbands above. SetMovementGate
  // assigns without clamping, so a caller can still pass a negative to mean
  // "reset to default"; that reset reads these instead of the environment, so
  // no update path calls getenv.
  double bootstrap_min_propagation_move_ = kDefaultMinMoveM;
  double bootstrap_min_propagation_yaw_ = kDefaultMinYawRad;

  // OnRemoteUpdate referent resolution mode. Default kLinker is
  // the unconditional behavior (ElementLinker Translate always
  // attempted, falling back to shared-id lookup when unlinked). Lazily
  // resolved once from SB_V2_REFERENT_RESOLVE unless pre-empted by
  // SetReferentResolveMode.
  ReferentResolveMode referent_resolve_mode_ = ReferentResolveMode::kLinker;
  bool referent_resolve_mode_env_resolved_ = false;

  // Realization policy family. Default kProgressive matches
  // the default RealizationPolicyConfig. Lazily resolved once from
  // SB_V2_REALIZATION_POLICY ("progressive"|"uniform"|"greedy") unless
  // pre-empted by SetRealizationPolicy.
  RealizationPolicyType realization_policy_type_ =
      RealizationPolicyType::kProgressive;
  bool realization_policy_env_resolved_ = false;

  // A receiver-local counterpart generation observes one linked hub graph at
  // a time.  A custom resolver is intentionally uncertifiable unless a later
  // resolver API supplies its own explicit version.
  uint64_t counterpart_revision_ = 1;
  SpaceHub* observed_counterpart_hub_ = nullptr;
  uint64_t observed_linker_graph_revision_ = 0;
  bool has_observed_linker_graph_revision_ = false;
  bool has_custom_referent_resolver_ = false;

  // Next element ID for elements created via OnElementEvent
  element_id_t next_element_id_ = 10000;

  // Tracks temp received element id -> source handle for resolve-on-share.
  // Populated in OnElementEvent
  // kAdded; consumed and erased in ResolveSharedElement.
  std::unordered_map<element_id_t, element_handle_t> pending_share_source_;

  // The room: the first element this space added itself with no parent
  // (Space::AddElement). A received element whose parent is absent goes under
  // it. Not ElementTree::GetRoot(), which every parentless add overwrites.
  element_id_t room_element_id_ = kInvalidElementId;
  const Element* RoomLocked() const { return element_tree_.GetElement(room_element_id_); }
  // Creates the received element under its parent, else under the room.
  // Appends the element it creates, with its event, for on_element_added.
  void AddReceivedElementLocked(const ElementEvent& event,
                                std::vector<std::pair<element_id_t, ElementEvent>>* added);

  // Source -> locally materialized AddAsNew replica. LinkToExisting entries
  // deliberately never enter this map: unsharing a remote source must not
  // remove locally authored furniture merely because it is linked.
  std::map<element_handle_t, element_id_t> remote_owned_replicas_;

  // Sources (the sharer's handle) whose link on this space ended by an unshare
  // or an unlink and which have not been resolved onto an element since. See
  // SourceRetiredLocked. Only sources this space had received or linked enter.
  std::set<element_handle_t> retired_sources_;

  // Locally authored elements explicitly shared through this Space. Kept
  // separate from shared_id: a dataset ID alone is not cross-space ownership.
  std::unordered_set<element_id_t> locally_shared_elements_;

  // Hook invoked when a peer-shared element is added locally (see
  // SetOnElementAdded).
  OnElementAddedFn on_element_added_;
  OnElementRemovedFn on_element_removed_;
  OnLinkRemovedFn on_link_removed_;
  // Relation lifecycle tracking.
  RelationPhaseTracker phase_tracker_;
  mutable std::unordered_set<spatialblend::entity_id_t> borrowed_binding_actors_;
  std::unordered_set<spatialblend::entity_id_t> source_evidence_enabled_;
  // Opted-in source phases cannot inherit legacy/generated/remote samples.
  // Cleared at the next fully validated source batch, without callbacks.
  std::unordered_set<spatialblend::entity_id_t> source_evidence_phase_dirty_;
  std::unordered_map<spatialblend::entity_id_t, SourceInteractionEvidence>
      source_evidence_;
  // Frozen public identities survive referent removal/shared-id changes. Keep
  // local referent handles too, so a recreated element cannot borrow a phase.
  std::unordered_map<spatialblend::entity_id_t,
                     std::vector<SourceRelationObservation>>
      source_relations_;
  std::function<void(const RelationEvent&)> on_relation_event_;
  // Containment transfer. `containment_transfer_used_` turns true the first
  // time any element is flagged; until then no containment code runs.
  bool containment_transfer_used_ = false;
  std::unordered_map<spatialblend::entity_id_t, ActorContainment>
      actor_containment_;
  // Referent resolution. Lazily defaulted; never null once queried.
  mutable std::shared_ptr<ReferentResolver> referent_resolver_;
  // Monotonic space-local clock (seconds), advanced by update_interval_ each
  // UpdateActorState — the same time source used for target arrival timing.
  // ALSO advanced inside OnRemoteUpdate (by the incoming update's own
  // step_duration, falling back to update_interval_) so a receiving space
  // whose only actor is a remotely-driven clone -- never locally calling
  // UpdateActorState at all -- has a real, monotonically-advancing clock
  // instead of one frozen at 0.0 forever (every "share"/"payload"/"target"/
  // "tick" trace line it emits, and every RelationEvent.timestamp the phase
  // tracker records, used to read "t":0.0 for the entire run on such a
  // space). Each call site advances the SAME clock for the SAME reason
  // (space-local elapsed time as of the event just processed), so a space
  // that is both locally driven AND a remote-update target (e.g. two
  // actors, one local, one a peer's clone, sharing a Space) simply
  // accumulates one tick per event from either source, matching the
  // pre-existing UpdateActorState-only semantics for a purely-driven space.
  double space_clock_ = 0.0;
  // Legacy per-actor elapsed input time. This removes cross-actor timing leaks,
  // but remains call-driven until a future authoritative timestamp API exists.
  std::unordered_map<spatialblend::entity_id_t, double> actor_update_seconds_;
  struct ObservationClock {
    uint64_t epoch = 0;
    int64_t sequence = -1;
    double time = 0;
    int64_t receipt_monotonic_ns = 0;
    // Legacy setters can change Actor overrides without a new source sample.
    // Only atomic frame admission can restore observed-point eligibility.
    bool anchors_from_committed_frame = false;
    bool has_sample = false;
    std::string root;
  };
  std::unordered_map<spatialblend::entity_id_t, ObservationClock>
      observation_clocks_;
  // Epoch values remain fenced after a setup cancellation.  The actor has no
  // active clock then, but a later epoch must not ABA-reuse the cancelled ID.
  std::unordered_map<spatialblend::entity_id_t, uint64_t>
      retired_observation_epoch_fences_;
  // Exact active epochs opened by BeginObservedSourceSetup.  Their opaque
  // setup token owns retirement, so another setup cannot silently clobber one.
  std::unordered_map<spatialblend::entity_id_t, uint64_t>
      observed_source_setup_epochs_;
  // Last propagated authoritative timestamp per actor. Presence is separate
  // so a first sample at exactly zero is not mistaken for unset.
  std::unordered_map<spatialblend::entity_id_t, double>
      last_observation_propagated_time_;
  std::unordered_set<spatialblend::entity_id_t> has_observation_propagated_time_;
  // Receiver observation fences key a complete source handle. The shared-id
  // fallback has an invalid element slot, keeping it separate from handles.
  using ObservationRemoteKey =
      std::tuple<space_id_t, element_id_t, shared_element_id_t>;
  std::map<ObservationRemoteKey, uint64_t> last_remote_observation_epoch_;
  std::map<ObservationRemoteKey, int64_t> last_remote_observation_sequence_;
  std::map<ObservationRemoteKey, double> last_remote_observation_time_;
  std::set<ObservationRemoteKey> has_remote_observation_time_;

  // Per-(source_space_id, actor_shared_id) last-applied
  // RemoteRelationUpdate::sequence, so OnRemoteUpdate can drop an exact
  // replay of an update it already processed (see remote_update.h's
  // `sequence` field doc for the observed duplicate-delivery bug this
  // guards against). Absent key / value 0 means "nothing applied yet".
  std::map<std::pair<space_id_t, shared_element_id_t>, int64_t>
      last_remote_sequence_;

  // Per-(source_space_id, actor_shared_id) last-seen
  // RemoteRelationUpdate::source_clock, so OnRemoteUpdate can advance this
  // space's own space_clock_ by the REAL elapsed source-side time (source_
  // clock - last_source_clock_[key]) instead of the constant step_duration
  // (see remote_update.h's `source_clock` field doc for why step_duration
  // alone under-runs real time across a heartbeat gap).
  // Absent key / value 0.0 means "no genuine source_clock observed yet for
  // this key" -- the very first delivery, or every delivery from a caller
  // that never populates source_clock (0.0 default, e.g. a hand-built
  // update in an older unit test) -- and OnRemoteUpdate falls back to
  // step_duration in that case. Only
  // updated when the incoming source_clock is strictly greater than the
  // stored value, so a stale/out-of-order delivery (non-monotonic
  // source_clock) can never regress this map -- OnRemoteUpdate also falls
  // back to step_duration for that delivery instead of computing a
  // negative/zero delta.
  std::map<std::pair<space_id_t, shared_element_id_t>, double>
      last_source_clock_;

  // Per-actor counter used to stamp RemoteRelationUpdate::sequence right
  // before this space (as SOURCE) actually propagates an update via
  // hub_->Propagate() -- i.e. only real, sent updates consume a sequence
  // number, so gaps from a movement-gated/heartbeat-skipped tick are
  // harmless (the dedupe check only cares about strict monotonicity, not
  // contiguity).
  std::unordered_map<spatialblend::entity_id_t, int64_t> next_propagate_sequence_;

  // Thread safety
  mutable std::mutex space_mutex_;
  SceneError last_scene_error_;  // guarded by space_mutex_
  // The orientation the source last set (pass-through); guarded by space_mutex_.
  std::unordered_map<spatialblend::entity_id_t, SourceOrientation> source_orientation_;
  std::unordered_map<spatialblend::entity_id_t, SourceOrientation> last_propagated_orientation_;
  // Orders deliveries made after space_mutex_ is released. Acquired while
  // space_mutex_ is still held and released after the send, so the lock order
  // is always space_mutex_ then propagation_mutex_ and concurrent callers
  // deliver in the order they stamped sequence numbers. The receiving path
  // takes only its own space_mutex_ and never this one, so no cycle exists.
  mutable std::mutex propagation_mutex_;

  static const std::vector<Binding> kEmptyBindings;

  // Scope extension (GetScopeExtensionRevision). `scope_extension_watch_` (guarded by
  // space_mutex_): some region extends or was admitted to extend its parent, so a
  // geometry change requests a rebuild (AdvanceSceneInputsLocked). With no
  // extension anywhere, nothing is requested and nothing changes.
  void RequestScopeExtensionLocked();
  void RunScopeExtensionBuild();
  bool scope_extension_watch_ = false;
  uint64_t obstacle_sync_rebuilds_ = 0;  // guarded by space_mutex_
  bool installing_scope_extension_ = false;
  std::function<void()> scope_extension_build_hook_;
  // Worker-thread only (RunScopeExtensionBuild): the definitions built for unchanged inputs.
  ScopeExtensionCache scope_extension_cache_;
  std::atomic<uint64_t> scope_extension_revision_{0};
  // Declared LAST: destroyed (its thread joined) before every member the task uses.
  std::unique_ptr<spatialblend::CoalescingWorker> scope_extension_worker_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_SPACE_H_
