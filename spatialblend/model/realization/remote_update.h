// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_REMOTE_UPDATE_H_
#define SPATIALBLEND_SPACE_REMOTE_UPDATE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "spatialblend/base/common/types.h"
#include "spatialblend/model/realization/containment_transfer.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/model/relation/support.h"

namespace spatialblend {

// The ONE contact epsilon of the core: v1's math::Constants::kEpsilon = 1e-6 (v1 applies it to degrees for an
// angle). A metric
// distance below it is contact (the count is 1, and engagement.h's contact and engagement facts); a progress below it
// is no motion toward the referent.
constexpr double kContactEpsilonM = 1e-6;

// v1's per-update count of a distance relation, with d_old the source's distance at its
// previous sent pose and d_new at this one, both the relation's own distance in its scope:
//   d_old < eps                       contact or dwell: 1 (one update; v1 keeps the target moving toward the furniture);
//   progress = d_old - d_new < eps    no approach (static OR receding): infinite, no valid count (v1 steps = inf);
//   otherwise                         d_old / progress, the updates the source needs to close its distance.
// The receiver closes 1/n of its own distance per update (RatioTargetDistance); only a positive finite count is a
// request (RatioRequest, HasValidUpdateCount; v1 FilterToValidSteps). An unknown distance is no count.
inline double SourceUpdateCount(double d_old, double d_new) {
  if (!(d_old >= 0.0) || !(d_new >= 0.0)) return std::numeric_limits<double>::infinity();
  if (d_old < kContactEpsilonM) return 1.0;
  const double progress = d_old - d_new;
  if (progress < kContactEpsilonM) return std::numeric_limits<double>::infinity();
  return d_old / progress;
}

// v1 Angular1D count for a Facing relation, counted in angle: the bearing offset
// phi = bearing - yaw is zero when the source looks at the referent. With the same epsilon as every count
// (kContactEpsilonM, applied as v1 applies it, to degrees):
//   the source is co-located with the look-at point (its distance below eps): 1;
//   aligned, |phi| below eps degrees: 1;
//   aligned NOW, |phi_new| below eps degrees: 1 (the paper's formula is the reference, which v1
//   approximates: the contact rule applied to Facing; a source that
//   looks at its look-at point is in contact with the relation, count 1, before any ratio. The previous offset is the
//   same-point one (the current look-at contact from the previous sent pose), which a contact that slides with the
//   gaze leaves non-zero: without this test the count of a source that stays aligned read a ratio near but not at 1);
//   no rotation at all: infinite, no valid count;
//   otherwise the rotation still to go in the direction of this update's rotation (the short way when it closes on
//   zero, the long way when it opens) divided by the absolute rotation of this update.
// The sign of the source's offset rotation, offset_old -> offset_new the short way (the rotation FacingUpdateCount
// reads), 0 when there is none.
inline int FacingRotationSign(double offset_old, double offset_new) {
  if (!std::isfinite(offset_old) || !std::isfinite(offset_new)) return 0;
  const double rotation = std::remainder(offset_new - offset_old, 2.0 * M_PI);
  return (rotation > 0.0) - (rotation < 0.0);
}

// Whether FacingUpdateCount counted the long way round (the offset opened): a rotation away from alignment, not from an
// aligned or co-located source.
inline bool FacingOpening(double offset_old, double offset_new, double distance_old = 1.0) {
  if (!std::isfinite(offset_old) || !std::isfinite(offset_new)) return false;
  if (std::isfinite(distance_old) && distance_old >= 0.0 && distance_old < kContactEpsilonM) return false;
  const double old_w = std::remainder(offset_old, 2.0 * M_PI);
  if (std::fabs(old_w) * (180.0 / M_PI) < kContactEpsilonM) return false;
  if (std::fabs(std::remainder(offset_new, 2.0 * M_PI)) * (180.0 / M_PI) < kContactEpsilonM) return false;  // aligned now
  const double rotation = std::remainder(offset_new - offset_old, 2.0 * M_PI);
  if (rotation == 0.0) return false;
  return (rotation > 0.0) == (old_w > 0.0);
}

inline double FacingUpdateCount(double offset_old, double offset_new, double distance_old = 1.0) {
  if (!std::isfinite(offset_old) || !std::isfinite(offset_new))
    return std::numeric_limits<double>::infinity();
  if (std::isfinite(distance_old) && distance_old >= 0.0 && distance_old < kContactEpsilonM) return 1.0;
  const double old_w = std::remainder(offset_old, 2.0 * M_PI);
  if (std::fabs(old_w) * (180.0 / M_PI) < kContactEpsilonM) return 1.0;
  if (std::fabs(std::remainder(offset_new, 2.0 * M_PI)) * (180.0 / M_PI) < kContactEpsilonM) return 1.0;  // aligned now
  const double rotation = std::remainder(offset_new - offset_old, 2.0 * M_PI);
  if (rotation == 0.0) return std::numeric_limits<double>::infinity();
  const bool closing = (rotation > 0.0) != (old_w > 0.0);
  const double remaining = closing ? std::fabs(old_w) : 2.0 * M_PI - std::fabs(old_w);
  return remaining / std::fabs(rotation);
}

// Over the network (network/proto_util) the timing mode, source_time,
// source_epoch and quiet_since_last_send travel as optional fields (absent =
// legacy / false); the caller's observation-frame sequence does not.
// Separately, legacy projection omits a disabled native payload rather than
// silently re-enabling it; consumers must not treat that projection as a rich
// realization snapshot.
enum class RelationUpdateTiming {
  kLegacy,
  kTimestampedObservation,
  kObservationReset,
};

struct SourceOrientation {
  int32_t type = 0;  // sb_orientation_type_t
  double pitch = 0.0;
  double roll = 0.0;
  bool operator==(const SourceOrientation& o) const {
    return type == o.type && pitch == o.pitch && roll == o.roll;
  }
  bool operator!=(const SourceOrientation& o) const { return !(*this == o); }
};

struct RelationPayload {
  shared_element_id_t referent_shared_id = kInvalidSharedElementId;
  std::string subject_anchor_name;
  GroundedRelation grounded_relation;
  Support support;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  // ABI 4.11: an INNER relation of the subject anchor to one affordance pose of the referent object
  // (room -> object SD -> affordance poses): the pose's tag ("stand", "seated", ...). The referent is the OBJECT; the
  // receiver finds its own instance of the pose with the same tag in its counterpart object (correspondence by equal
  // tags). Empty for every other relation. The relation is Spacing's: grounded_relation.support_distance_m is the
  // distance to the pose's nearest point and source_scope_diameter_m the SD's scale D.
  std::string affordance_pose_tag;
  // Explicit producer-side disable: retained in the snapshot as a disabled
  // contributor but receives no policy mass and releases a fully disabled
  // channel.  This is not inferred from proximity or satisfaction.
  bool enabled = true;

  // In-process observed-point capability. Never serialized by the legacy
  // network projection: those peers have no point lifetime/clock contract.
  // A receiver requires an independently live point with this logical owner;
  // neither a coincident shared ID nor a static object can substitute for it.
  bool observed_facing_point = false;
  shared_element_id_t observed_logical_actor = kInvalidSharedElementId;
  element_handle_t observed_source_point;
  uint64_t observed_source_point_lease = 0;
  int64_t observed_valid_until_ns = 0;

  // Estimated time to satisfaction in seconds (from SatisfactionMetric).
  // Lower = more imminent (closer to interaction-defining).
  // 0.0 = already satisfied; large values = far from engagement.
  // Computed source-side from kernel-specific distance / actor velocity.
  double estimated_time = 0.0;

  // v1 update count (SourceUpdateCount). The
  // source's per-update progress fraction n = d_old / (d_old − d_new). The
  // receiver closes 1/n of its own remaining distance. 1: d_old below the contact
  // epsilon (contact, dwell, or leaving contact). +inf: progress below the
  // epsilon (static or receding). Only a positive finite count is a valid intent
  // (RatioRequest); a payload that does not carry one (legacy wire) is no valid
  // intent.
  double update_count = std::numeric_limits<double>::infinity();
  // Facing only: the sign of the source's offset rotation this update (+1 / -1; 0 = none or unknown, a sender that
  // predates the field). With the offset's sign it says whether the offset is closing (the count the short way) or
  // opening (v1's count the long way round); the receiver turns in the same direction
  // (the receiver reproduces the source's motion and keeps its turning direction).
  int facing_rotation_sign = 0;
  // Facing only: the source's offset OPENED this update (FacingOpening: v1 counted the long way round). Read against the
  // source's OLD offset, as the count is; the new offset alone cannot tell it (a source that just closed to alignment
  // has a new offset of either sign).
  bool facing_opening = false;

  // Linker fields: source-side identity of the referent element.
  // Set by the SOURCE space when building the payload so the RECEIVER can
  // resolve the referent via ElementLinker instead of shared-id lookup.
  // Both are kInvalid by default; receiver uses them in linker and
  // linker_only modes (see space.cc OnRemoteUpdate).
  space_id_t referent_source_space_id = kInvalidSpaceId;
  element_id_t referent_source_element_id = kInvalidElementId;

  // Filled by the RECEIVER under the linker flag: the local element_id
  // resolved from (referent_source_space_id, referent_source_element_id).
  // kInvalidElementId when the linker has no link for this source handle,
  // or when shared-id mode is selected. linker_only rejects that payload;
  // legacy linker mode falls back to GetElementBySharedId unchanged.
  element_id_t resolved_local_element_id = kInvalidElementId;
  // Receiver-derived on every delivery (never trusted from a sender, never on
  // the wire): the referent's source was linked here and the link ended, and it
  // has not been resolved anew. It has no counterpart, not even by shared id.
  bool referent_retired = false;

  // The SOURCE element whose navigable geometry normalized R
  // (the anchor's deepest containing navigable), so the receiver
  // denormalizes / evaluates over its counterpart rather than over whatever
  // happens to enclose its own referent. Same identity scheme as the
  // referent: linker handle first, shared id second. Invalid = unknown
  // (legacy payload or no scope at encode time).
  shared_element_id_t scope_shared_id = kInvalidSharedElementId;
  space_id_t scope_source_space_id = kInvalidSpaceId;
  element_id_t scope_source_element_id = kInvalidElementId;
  // Filled by the RECEIVER: the local counterpart of the source scope, after
  // applying its referent-resolve mode (scope_resolved_by_receiver = true).
  element_id_t resolved_scope_element_id = kInvalidElementId;
  bool scope_resolved_by_receiver = false;

  // The SOURCE binding this relation was extracted under: its ordinal
  // on the source actor and its revision (Binding::revision). Known only
  // when the source can certify the binding definition (bound through the
  // owned C API path, never exposed as a mutable pointer); receivers carry
  // it unchanged into the rich contributor's binding_revision.
  uint64_t source_binding_ordinal = UINT64_MAX;
  uint64_t binding_revision = 0;
  bool binding_revision_known = false;
  // The subject anchor's orientation beyond yaw on the source (pitch and roll are
  // per-anchor, a pass-through of that anchor's proposal; the root's is the actor's sb_update_actor_pose orientation).
  // Absent: a sender that predates it.
  bool has_subject_orientation = false;
  SourceOrientation subject_orientation;
};

// The source's orientation beyond yaw (ABI 3.1, sb_orientation_t): carried as a
// pass-through, never solved.
// An inner affordance-pose relation (see RelationPayload::affordance_pose_tag).
inline bool IsInnerPosePayload(const RelationPayload& p) {
  return p.kernel_type == RelationKernelType::kAvatarAffordancePose && !p.affordance_pose_tag.empty();
}


// The exception id the first update of a source actor carries (a join).
constexpr uint32_t kExceptionJoin = 1;

struct RemoteRelationUpdate {
  RelationUpdateTiming timing = RelationUpdateTiming::kLegacy;
  // Meaningful for kTimestampedObservation and kObservationReset. Epoch zero
  // is never a valid observation epoch.
  uint64_t source_epoch = 0;
  // A source sample time, including a valid zero. It is not overloaded with
  // source_clock (the legacy accumulated local clock).
  double source_time = 0.0;
  // Original caller-supplied observation-frame sequence for a timestamped
  // sample.  This is deliberately distinct from `sequence`, which numbers
  // relation deliveries and is also consumed by observation resets.  Unknown
  // is represented explicitly so legacy and control updates do not invent a
  // sample identity.
  int64_t source_observation_sequence = 0;
  bool source_observation_sequence_known = false;
  space_id_t source_space_id = kInvalidSpaceId;
  shared_element_id_t actor_shared_id = kInvalidSharedElementId;
  // Source-side identity of the actor element, so the receiver can resolve the
  // local clone via the ElementLinker (the link recorded at share time) instead
  // of matching actor_shared_id. The link IS the cross-space identity; the
  // actor_shared_id remains as a fallback for not-yet-linked / legacy paths.
  space_id_t actor_source_space_id = kInvalidSpaceId;
  element_id_t actor_source_element_id = kInvalidElementId;
  std::vector<RelationPayload> payloads;
  // A source-app EXCEPTION (ABI 4): the source app raised an app-specific
  // exception (a teleport, a button next to a whiteboard, ...) with an opaque
  // payload. Core never interprets it; every receiving space realizes it as an
  // SB_ACTOR_EVENT_SOURCE_EXCEPTION and solves the absolute relation. The first
  // update a source actor propagates carries the join exception
  // (kExceptionJoin, empty payload). Absent on the wire = no exception.
  bool has_exception = false;
  uint32_t exception_id = 0;
  std::vector<uint8_t> exception_payload;
  // The source's orientation beyond yaw (ABI 3.1): a PASS-THROUGH to the receiver
  // proposal. Core never solves it and no relation uses it. type is
  // sb_orientation_type_t (0 yaw only, 1 yaw+pitch, 2 yaw+pitch+roll); absent on
  // the wire = yaw only.
  int32_t source_orientation_type = 0;
  double source_pitch = 0.0;
  double source_roll = 0.0;
  // Source update interval in seconds.
  double step_duration = 0.033;

  // Per-(source space, source actor) monotonically increasing relation
  // delivery counter.  It is positive for timestamped observations AND
  // observation resets; it is not the caller's observation-frame sequence.
  // assigned once by Space::UpdateActorState right before hub_->Propagate(),
  // never touched by any relay hop (HubReceiver::OnClientRelationUpdate /
  // SpaceHub::Propagate forward the same struct unchanged). Lets a receiver
  // (Space::OnRemoteUpdate) recognize an exact-replay delivery of an update
  // it has already applied -- observed ~1/10 runs under Transport::
  // kGrpcTwoSessions, an extra duplicate OnRemoteUpdate call for the very
  // first delivered update right after sb_connect_space's bootstrap/
  // Subscribe-stream-settle window (see unity_twin.h's Transport doc comment
  // for the investigation). 0 is the sentinel for "not assigned" (a hand-
  // built update, e.g. in a unit test) and always bypasses the dedupe check.
  int64_t sequence = 0;

  // The SOURCE space's own Space::space_clock_ at the moment this
  // update was propagated: the source ACTOR's own elapsed time (its ticks, or its timestamps), never the source space's
  // diagnostic clock, which also counts what that space receives (otherwise two peers' stamps feed each other and grow
  // geometrically) (stamped in Space::UpdateActorState, right
  // alongside `sequence`, immediately before hub_->Propagate()). This is
  // real elapsed source-side time -- unlike `step_duration` (always the
  // constant update_interval_, regardless of how many frames were actually
  // skipped by the movement-gate deadband before this delivery), the gap
  // between two successive source_clock values for the same (source_space_
  // id, actor_shared_id) key is the TRUE real-time gap between those two
  // deliveries, whether that gap was one frame (genuine per-frame motion)
  // or a full kHeartbeatInterval (~1.0s, a heartbeat-forced resend after the
  // source held still). Space::OnRemoteUpdate uses this delta -- not
  // step_duration -- to advance the receiver's own space_clock_, so a peer
  // space's "t" (every ev=target/payload/relation/tick trace line, every
  // RelationEvent.timestamp) tracks real elapsed time instead of under-
  // running it by delivery-count * update_interval_ across heartbeat gaps.
  // 0.0 (default) is the "unset" sentinel -- a hand-built update (unit test,
  // legacy caller) that never populates this field -- and always makes
  // OnRemoteUpdate fall back to step_duration, exactly like the earlier
  // behavior. step_duration itself is kept unchanged for backward
  // compatibility (some callers/tests still only set it).
  double source_clock = 0.0;

  // The SENDER asserts that every source tick since its previous send
  // of this actor was skipped only because the actor stayed within the
  // movement deadband (with relations to send), and that this send is the
  // heartbeat alone (nothing moved or changed on its tick): the interval
  // this update closes is a movement-gated heartbeat gap, over which the
  // source was still. Any other skip (no payloads, a dormant scope, a missing anchor, a
  // tick that never reached the gate) and every timestamped or reset update
  // leave it false. Default/absent = false: a receiver credits a long
  // interval as stillness only on this explicit assertion.
  bool quiet_since_last_send = false;

  // Containment transfer: the source root's pose in the frame of the flagged
  // Containment element it is contained in. Optional on the wire; absent means
  // "root not contained" (and is what a peer that predates the field decodes
  // to). Never gates the scene-level payloads, which keep being sent so a
  // receiver without the object falls back to them.
  std::optional<ContainmentObservation> containment;

  // ABI 4.14: each active bound anchor's pose in the transfer leaf that contains it (AnchorChartObservation). Empty when
  // no anchor is inside a transfer leaf; an old peer sends none.
  std::vector<AnchorChartObservation> anchor_charts;
};

struct ElementEvent {
  // kUnlinked: a receiver dropped its link to the shared element identified by
  // source_space_id/source_element_id (the sharer's handle) without the sharer
  // unsharing it. peer_space_id names the receiver. Only the sharer's space acts
  // on it.
  enum Type { kAdded, kRemoved, kUnlinked };
  Type type;
  shared_element_id_t shared_id;
  shared_element_id_t parent_shared_id = kInvalidSharedElementId;
  std::string tag;
  space_id_t source_space_id = kInvalidSpaceId;
  // Opaque application payload (e.g. serialized entity) propagated verbatim so
  // the receiving space can reconstruct the shared object (OnShare).
  std::string user_data;
  // Source-side element id of the shared element (set by ShareElement so the
  // receiving space can build source_handle for ElementLinker).
  element_id_t source_element_id = kInvalidElementId;
  // kUnlinked only: the receiver space that removed its link.
  space_id_t peer_space_id = kInvalidSpaceId;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_REMOTE_UPDATE_H_
