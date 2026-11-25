// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/receiver_proposal.h
//
// What core PROPOSES for a receiver (ABI 4).
//
// A proposal SET is computed on every source-triggered solve from the source
// record and the poses SAMPLED from the driver's per-anchor ports at that moment.
// It holds ONE pose proposal per relation-bearing anchor (root, pelvis, head, ...):
// the relation-preserving pose plus the attribution the driver needs (why it
// moved, status, cause bits) and the identity of the samples it was computed from.
// Containment, seat and placement are not side channels: they are the proposals of
// the anchors that carry them, and cause bits. Core does no pacing, slewing,
// fading, rate limiting or teleport smoothing: the driver applies the proposals
// within its own kinetic bounds and publishes the new poses back through the ports.
#ifndef SPATIALBLEND_SPACE_RECEIVER_PROPOSAL_H_
#define SPATIALBLEND_SPACE_RECEIVER_PROPOSAL_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "spatialblend/space/embodied_port.h"

namespace spatialblend {

enum class ProposalStatus : int32_t {
  // Nothing proposed: no source update has been solved for this actor yet, or
  // the source released every relation (the driver keeps its body).
  kNone = 0,
  // The solve could not run: the driver has not published an embodied pose
  // (or the published one is invalid). Core never substitutes a pose of its
  // own. The target fields are zero and must not be used.
  kNoSample = 1,
  // The target is an exact minimiser of the weighted relation objective.
  kExact = 2,
  // The target is the best minimiser found, with a residual above epsilon
  // (relations that cannot all hold at once).
  kInexact = 3,
  // 4 (HOLD) is RETIRED with the holds: there is
  // no "infeasible, stay" status. Every referent the receiver cannot map drops;
  // with none left the proposal is NONE. The number is not reused.
  // The solve hit its loop guard before a step was rejected at maximum
  // damping and before the gradient reached the solver tolerance. Not an
  // answer: the pose fields are the last trial, for diagnosis only.
  kNotConverged = 5,
};

// Cause bitmask: why THIS proposal differs from the previous one. The live target of every anchor is the PURSUED
// target (the placed destination, the chart image of the source pose, only for an explicit one: the first proposal, a
// source exception), so PROGRESS, CHART_SNAP and NO_REQUEST are set; the human speed limit is the driver's. With no
// request core publishes no new set (event model); a set published for another reason that carries NO_REQUEST holds
// the anchor's retained pending target, not its sample.
// Mirrors SB_PROPOSAL_CAUSE_*.
constexpr uint32_t kCauseFirstProposal = 1u << 0;     // none before (or after a reset)
constexpr uint32_t kCauseWeightSwitch = 1u << 1;      // the driving (max-weight) relation changed, or the weights moved S0 beyond the points' own motion
constexpr uint32_t kCauseReferentSetChange = 1u << 2; // the set of grounded referents changed
constexpr uint32_t kCauseSourceException = 1u << 4;   // the source app raised an exception (teleport, ...); absolute solve
constexpr uint32_t kCauseContainEnter = 1u << 5;      // the actor entered a containment-transfer element
constexpr uint32_t kCauseContainExit = 1u << 6;
constexpr uint32_t kCauseSeatEngage = 1u << 7;        // an AffordancePose anchor touched its surface
constexpr uint32_t kCauseSeatRelease = 1u << 8;
constexpr uint32_t kCauseEmbodiedReset = 1u << 9;     // the driver's epoch changed or it declared a teleport
constexpr uint32_t kCauseFacingRootSwitch = 1u << 10;  // the Facing +-theta sign root changed, or a Facing's look-at contact swapped silhouette vertex (the least-angle rule's own discontinuity, Q9 option (a))
constexpr uint32_t kCauseStatusChange = 1u << 11;  // nothing proposed (NONE / NO_SAMPLE) <-> a target
constexpr uint32_t kCauseSceneChange = 1u << 12;   // receiver support revision changed
constexpr uint32_t kCauseClampBranch = 1u << 13;   // the clamp's NearestFree switched constraint branch
constexpr uint32_t kCauseDecodeBranch = 1u << 14;  // a referent's approach-point decode changed rule
constexpr uint32_t kCauseProgress = 1u << 15;  // a spacing term closed d(sample)/n of its gap
constexpr uint32_t kCauseChartSnap = 1u << 16;  // the source is inside its referent or at contact and the engaged relations carry all the weight: the proposal is their chart pose, the minimiser of J (with weight elsewhere the engaged referent is one term of J, not a snap)
constexpr uint32_t kCauseFacingTie = 1u << 18;  // the Facing bearings' circular resultant is within the error interval of its computation (an exact or numerically unresolved tie of opposing bearings): the yaw is the middle nearest the embodied yaw, and it may differ from the neighbouring updates' by the tie alone; or the path between this proposal and the previous one passes a look-at point within the data's precision (the bearing to it has a pole there), and the mean of the bearings may differ from the previous update's by the pass alone
// ABI 4.11: an anchor's inner solve (its object's affordance poses) answered on another branch of a weighted term's
// distance (the nearest feature of that pose's support changed) with the same driving pose: the continuation crossed
// a fold of the objective. Mirrors SB_PROPOSAL_CAUSE_FOLD.
constexpr uint32_t kCauseFold = 1u << 19;
constexpr uint32_t kCauseNoRequest = 1u << 17;  // no relation issues a request (every update count is infinite). Event model: core publishes no new set for it (the revision does not advance) and the driver continues its pending target; a set published for another reason that carries this bit holds the anchor's RETAINED pending target (Space::RetainIdlePositionLocked), not its sample

// Flags of one anchor proposal (mirror SB_ANCHOR_PROPOSAL_FLAG_*).
constexpr uint32_t kAnchorFlagYawUnspecified = 1u << 0;
constexpr uint32_t kAnchorFlagEngaged = 1u << 1;  // the relation's geometric condition holds
constexpr uint32_t kAnchorFlagSampled = 1u << 2;  // this anchor's own sample was published
// The target app has ARRIVED at this anchor's relation: the SAMPLED anchor already
// satisfies it (root: the Spacing contact the source holds is satisfied at the sampled
// pose, its error in metres within the solver's declared tolerance, and with none held
// every positive-weight root relation is; AffordancePose: the sampled anchor touches
// the surface; Facing: the SIGNED angular error at the sample is within v1's angle,
// relation_evaluator.h kFacingArrivalRad). This is what stops the latency timer.
constexpr uint32_t kAnchorFlagArrived = 1u << 3;
// Core publishes NO position target for this anchor (ABI 4.1): the receiver
// holds, as v1 sent nothing when no relation issued
// a request (a static, receding or non-closing source, n = infinity). The position
// fields echo the sample and are not a target. Analogous to kAnchorFlagYawUnspecified.
constexpr uint32_t kAnchorFlagPositionUnspecified = 1u << 4;
// ABI 4.3, root anchor only: this update's root
// proposal IS the transfer mapping. Set exactly when the source root is inside a
// transfer leaf (the source-side point test) and this space has that leaf's
// counterpart (same shared id, declared a transfer leaf). Recomputed on every
// update; nothing is stored for it. CONTAIN_ENTER / CONTAIN_EXIT remain the cause
// bits of the jump update.
constexpr uint32_t kAnchorFlagTransfer = 1u << 5;
// ABI 4.11: this anchor's proposal is its INNER solve, the anchor's position among its object's
// affordance poses inside the object's SD (the source anchor is in the SD and not in contact). Position only; the yaw
// is unspecified. Mirrors SB_ANCHOR_PROPOSAL_FLAG_INNER.
constexpr uint32_t kAnchorFlagInner = 1u << 6;
// ABI 4.11: `facing_yaw` is valid: the anchor is engaged (seated) or in its inner solve, and the root has a Facing
// yaw; the head and upper body follow it. Mirrors SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW.
constexpr uint32_t kAnchorFlagFacingYaw = 1u << 7;

// The inner level of one anchor's proposal (ABI 4.11, sb_anchor_solve_t).
struct InnerSolveInfo {
  bool valid = false;
  int64_t object_shared_id = 0;  // the object whose SD holds the poses
  double scale_m = 0.0;          // D, the receiver SD's scale
  double j = 0.0;                // J at the answer
  int iterations = 0;
  std::string driving_tag;       // the pose of the largest weight
  std::vector<std::pair<std::string, int>> branch;  // (pose tag, nearest feature) of the weighted terms
  bool carries_root = false;  // an actor-root pose (`stand`) is a term: the anchor moves with the proposed root
  bool no_request = false;    // no inner relation has a valid count: the solve answers at the sample; the published anchor keeps its retained pending target
  bool face_holds = false;    // a face of the SD region volume V holds the answer (the descent is blocked there)
};

// One anchor's pose proposal. Root is anchor id 0.
struct AnchorProposal {
  uint32_t anchor_id = 0;
  std::string anchor_name;
  ProposalStatus status = ProposalStatus::kNone;
  uint32_t cause = 0;
  uint32_t flags = 0;  // kAnchorFlag*

  // The proposed pose (receiver space, Z-up). `z` is the anchor's own height:
  // the sampled height for the planar root, the contact height for an
  // AffordancePose anchor. yaw is the solved yaw; pitch and roll are the
  // source's PASS-THROUGH (ABI 3.1 semantics), 0 when the type does not carry them.
  double x = 0.0, y = 0.0, z = 0.0;
  double yaw = 0.0;
  double pitch = 0.0, roll = 0.0;
  int32_t orientation_type = 0;  // sb_orientation_type_t of the source's feed

  // Relation error |R_tgt - R_src|, max over the used relations of a family
  // (R units; multiply by err_scale_m for metres), at the target and at the
  // SAMPLED pose (the relation error the body has right now). -1 = none.
  double err_spacing_target = -1.0, err_spacing_sample = -1.0;
  double err_facing_target = -1.0, err_facing_sample = -1.0;
  // AffordancePose: the same for the anchor's pose-fit R.
  double err_pose_target = -1.0, err_pose_sample = -1.0;
  double err_scale_m = 0.0;  // scope diameter D of the heaviest spacing term

  // The driving referent of this anchor's solve (0 = none).
  int64_t referent_shared_id = 0;

  // ABI 4.11: the root's Facing yaw for a seated / inner anchor (kAnchorFlagFacingYaw), and the inner solve.
  double facing_yaw = 0.0;
  // ABI 4.12: the signed, unwrapped yaw turn from the anchor's sampled yaw that `yaw` is (radians; yaw = sampled yaw +
  // yaw_turn modulo 2 pi). Pursuing on a Facing it is the turn the relations request, which may exceed pi (reproducing
  // a source turning away the long way round); arriving, the shortest arc; the shortest arc when
  // placed (first proposal, source exception), for an engaged pose or a
  // leaf-relative yaw; 0 for a hold, an unspecified yaw, or an anchor with no referent.
  double yaw_turn = 0.0;
  InnerSolveInfo inner;

  // ABI 4.14: the replay through the chart, a separate output beside the Spacing position and the
  // Facing yaw (the driver composes them; core does not choose). Present when the source sent this anchor's pose in a
  // transfer leaf (AnchorChartObservation) and this space has the leaf's counterpart declared a transfer leaf: the
  // source anchor's own pose mapped through the counterpart (receiver space, Z-up), with the anchor's own pitch / roll
  // pass-through (orientation_type / pitch / roll above).
  // chart_verified: the source's footprint of the leaf rode this update and is congruent with the counterpart's (the
  // root transfer flag's test); otherwise the declaration alone.
  bool has_chart = false;
  bool chart_verified = false;
  int64_t chart_leaf_shared_id = 0;
  double chart_x = 0.0, chart_y = 0.0, chart_z = 0.0;
  double chart_yaw = 0.0, chart_pitch = 0.0, chart_roll = 0.0;
  int32_t chart_orientation_type = 0;

  // Native-only request metadata, captured at the solve, never reconstructed
  // from a later port read. The C proposal snapshot keeps its layout/flag
  // meanings; ABI 4.13 target events carry these independent channels and
  // sample origin.
  bool position_active = false, yaw_active = false;
  bool position_requested = false, yaw_requested = false;
  bool placement = false;  // actual absolute solve, not inferred from
                           // FIRST_PROPOSAL attribution
  EmbodiedSample request_sample;

  bool yaw_specified() const { return (flags & kAnchorFlagYawUnspecified) == 0; }
  bool engaged() const { return (flags & kAnchorFlagEngaged) != 0; }
  bool sampled() const { return (flags & kAnchorFlagSampled) != 0; }
  bool arrived() const { return (flags & kAnchorFlagArrived) != 0; }
  bool position_specified() const {
    return (flags & kAnchorFlagPositionUnspecified) == 0;
  }
  bool transfer() const { return (flags & kAnchorFlagTransfer) != 0; }
  bool inner_solve() const { return (flags & kAnchorFlagInner) != 0; }
  bool has_facing_yaw() const { return (flags & kAnchorFlagFacingYaw) != 0; }
};

// The proposal set of one solve. The ROOT anchor's proposal is the base class
// (anchor id 0), so the root is read as `p.x`, `p.status`, `p.cause`; every
// other relation-bearing anchor is in `anchors`, in anchor-id order.
struct ReceiverProposal : AnchorProposal {
  uint64_t revision = 0;  // increments with every published set

  // The source update this answers (RemoteRelationUpdate::sequence; 0 = a
  // hand-built update) and the root sample it was computed from.
  int64_t source_sequence = 0;
  int64_t source_shared_id =
      0;  // native target-event identity (not an extra C proposal field)
  bool sample_valid = false;
  uint64_t sample_sequence = 0;
  uint64_t sample_epoch = 0;
  double sample_timestamp_s = 0.0;  // the driver's own timestamp, echoed
  // Age of the sample when the solve ran, on core's steady clock (time since
  // the driver's Publish call). Large = the driver is not publishing.
  double sample_age_s = 0.0;

  // The non-root relation-bearing anchors (pelvis, head, ...).
  std::vector<AnchorProposal> anchors;

  // Every anchor proposal, root first.
  std::vector<const AnchorProposal*> All() const {
    std::vector<const AnchorProposal*> out;
    out.push_back(this);
    for (const auto& a : anchors) out.push_back(&a);
    return out;
  }
  const AnchorProposal* Find(uint32_t id) const {
    if (id == 0) return this;
    for (const auto& a : anchors)
      if (a.anchor_id == id) return &a;
    return nullptr;
  }
  // The absolute solve (first proposal or a source exception): the proposals
  // are the relation's own target, not a continuation; the app realizes them as
  // its exception policy dictates (teleport, ...).
  bool IsAbsolute() const {
    return (cause & (kCauseFirstProposal | kCauseSourceException)) != 0 &&
           (status == ProposalStatus::kExact || status == ProposalStatus::kInexact);
  }
};

// The latest set of an actor, readable from any thread. The set is immutable
// once stored (a snapshot); the lock is held for a pointer copy only, never
// across a solve.
class ProposalBox {
 public:
  void Store(std::shared_ptr<const ReceiverProposal> p) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = std::move(p);
  }
  // False until the first Store.
  bool Load(ReceiverProposal* out) const {
    std::shared_ptr<const ReceiverProposal> p;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      p = value_;
    }
    if (!p) return false;
    *out = *p;
    return true;
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const ReceiverProposal> value_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RECEIVER_PROPOSAL_H_
