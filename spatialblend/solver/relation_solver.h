// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/relation_solver.h
#ifndef SPATIALBLEND_SPACE_RELATION_SOLVER_H_
#define SPATIALBLEND_SPACE_RELATION_SOLVER_H_

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "spatialblend/base/common/types.h"
#include "spatialblend/model/element/element_tree.h"
#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/relation/relation_kernel.h"
#include "spatialblend/solver/engagement.h"
#include "spatialblend/solver/referent_resolver.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/model/realization/receiver_input.h"

namespace spatialblend {

enum class SolverStatus {
  kConverged,
  kFallback,
  kFailed,  // an invalid policy weight vector only
  // The iteration guard fired before a step was rejected at maximum damping
  // and before the gradient reached the solver tolerance. Not a minimiser.
  kNotConverged,
  // The per-solve evaluation budget (RelationSolver::Config::solve_budget_evals) ran out before the stop rule was met. The result
  // is the best feasible incumbent (the lowest-J feasible iterate of this solve, or the warm start when nothing improved) and
  // IS published. It is not converged and never counts as converged (SolverResult::budget_hit).
  kBudgetHit,
};

// The per-frame budget of the Space component (2.5 ms of one thread's
// CPU time), expressed as a count of RELATION EVALUATIONS (the solver's own counter, SolverResult::objective_evaluations), because a count is the same on every host
// and under every load where thread CPU time is not (measured: two identical slow runs gave 229,721 and 162,705 budget hits
// in a CPU-time experiment).
//   kP1SolveBudgetEvals = floor(kP1SolveBudgetUs / c), c = the median thread-CPU cost of one counted relation evaluation.
// WHAT IS COUNTED, exactly: one unit per EvaluatePreparedTargetRelation call made on a candidate position, i.e. one per position term per
// evaluation of J (eval_errors), plus the one evaluation per term at the sample. NOT counted: the derivatives (term_deriv, the finite
// differences of moving contacts), the route and contact-route listings chi and the tie models are built from, the free-space membership and
// walkability tests, and the trust-region subproblem. c is the thread-CPU time of the WHOLE solve divided by the counted units, so it
// includes the uncounted work in proportion to what the measured solves did; a solve that spends relatively more on uncounted work than the
// measured ones takes more CPU time than 2500 us at 729 units.
// PROVISIONAL, XEON-DERIVED, PENDING QUEST: c was measured on a Xeon host, not on Quest 3, and the whole per-frame budget is assigned to
// the solve until that budget is allocated per component. Re-derive it when Quest is measured.
// Measurement (Xeon host, thread CPU time of every Solve() divided by its objective_evaluations, quick level whole and 8 of
// the 64 slow shards: 5,173,389 solves, 86,792,734 evaluations): c = 3.4252 us, the median over evaluations (each evaluation counts
// once, so a solve counts as often as it has evaluations; slow alone 3.397 us, quick alone 4.175 us); the median over solves is 5.67 us
// (the median solve has 4 evaluations and a fixed overhead). 2500 / 3.4252 = 729.8, so the cap is 729.
// NOT ABOVE THE PER-SOLVE MAXIMUM: the most evaluations one solve used in that measurement is 12,160 (about 46 ms of thread CPU time),
// 16.7 times the cap. The cap is reported as derived and not changed to fit; 9,181 of the 8,957,848 solves measured used 729 evaluations or more.
constexpr double kP1SolveBudgetUs = 2500.0;
constexpr int kP1SolveBudgetEvals = 729;

// Full transform state: [x, y, z, yaw, pitch, roll]
// Index layout constants
constexpr int kStateX = 0;
constexpr int kStateY = 1;
constexpr int kStateZ = 2;
constexpr int kStateYaw = 3;
constexpr int kStatePitch = 4;
constexpr int kStateRoll = 5;
constexpr int kStateSize = 6;

struct SolverGoalWitness {
  size_t payload_index = 0;  // index into the payloads passed to Solve()
  Eigen::Vector2d witness = Eigen::Vector2d::Zero();
  double weight = 0.0;
  bool position_goal = false;
  bool yaw_goal = false;
  // Observation of the yaw decode (set only on a yaw goal): the conditioning c of
  // the bearing and the oriented bearing (bearing to the witness minus the source's
  // offset) it entered the circular mean with. Reported so a harness can recompute
  // the yaw independently; the solver never reads them.
  // Which rule of the stand-off decode produced the witness (see DecodeTargetPoint).
  int64_t decode_branch = 0;
  // True when the decoded look-at point lies inside the receiver's body clearance: that
  // Facing has no defined bearing and is left out of the yaw.
  bool yaw_excluded = false;
  // The weight multiplier the bearing entered the circular mean with: the conditioning c, or
  // 0 when the Facing is excluded (yaw_excluded) -- so sum(w * c * (cos, sin)(bearing)) over
  // the yaw goals IS the solver's mean.
  double yaw_conditioning = 0.0;
  double yaw_bearing = 0.0;
};

// THE RATIO CONTINUATION of one relation, shared by every anchor (the root's pursuit terms and an anchor's inner
// affordance-pose terms: the pursued target is the live rule whatever the anchor). v1's per-update
// count n = d_old / (d_old - d_new) of the source (SourceUpdateCount) asks the pursuer to close 1/n of ITS OWN distance
// d0 to the relation's mapped contact in this update: the distance it is to have afterwards is d0 (1 - 1/n), never
// below 0 (n = 1: arrive). A count that is not a positive finite number (a static source, a receding one, a payload
// without a count) issues no request.
inline bool RatioRequest(double update_count) { return std::isfinite(update_count) && update_count > 0.0; }
inline double RatioTargetDistance(double d0, double update_count) {
  double d_star = d0 * (1.0 - 1.0 / update_count);
  if (!(d_star > 0.0)) d_star = 0.0;
  return d_star;
}

// Feasibility of the solved root position.
enum class FeasibilityOutcome {
  kUnconstrained,   // no navigable definition: nothing to check against
  kFeasible,        // the emitted pose is in F
  kProjected,       // the embodied sample was in the walkable but not in F: the solve
                    // started from its nearest pose in F
  kUnreachable,     // retained: a free pose the body cannot be connected to
  kNoFreeSpace,     // the navigable region is empty: kFailed
  kSampleOutside,   // no free pose answers the relation: the answer is the sample (not a hold, D7)
  kSearchFoundNone, // a projection found no free candidate; F is not empty
};

// What the solver did with each payload.
enum class SolverRelationDisposition {
  kUsed,             // entered the position and/or yaw solve
  kDisabled,         // zero policy weight
  kUnmapped,         // no receiver counterpart for the referent
  kGroundingFailed,  // the source support cannot be reproduced on it
  kUnsupported,      // unknown relation family
};
struct SolverRelationOutcome {
  size_t payload_index = 0;  // index into the payloads passed to Solve()
  SolverRelationDisposition disposition = SolverRelationDisposition::kUsed;
  std::string reason;
};

struct SolverResult {
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  double z = 0.0;
  double yaw = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
  double total_residual = 0.0;
  // Every position relation dropped (unmapped or ungroundable on the receiver):
  // nothing to map. Position and yaw echo the sample; the proposal is NONE (D7).
  bool no_grounded_referent = false;
  // The solve's evaluation budget ran out (status == kBudgetHit). The position is the incumbent: the lowest-J feasible iterate
  // seen in this solve, or the warm start when nothing improved (`budget_incumbent_is_warm_start`). Observation: the iteration the
  // check fired at, and J at the incumbent.
  bool budget_hit = false;
  bool budget_incumbent_is_warm_start = false;
  int budget_hit_iteration = -1;
  double incumbent_objective = 0.0;
  int iterations = 0;
  SolverStatus status = SolverStatus::kConverged;
  // Feasibility of `position` (the free-space projection).
  FeasibilityOutcome feasibility = FeasibilityOutcome::kUnconstrained;
  // The pose the position solve was posed from: the embodied sample, or its nearest pose in F
  // when the sample was in the walkable but not in F (feasibility == kProjected). The
  // proposal's continuation (the ratio step) starts here, not at the raw sample.
  // Observation only: nothing reads it.
  Eigen::Vector2d solve_origin = Eigen::Vector2d::Zero();
  // The engagement exemptions of the F this solve admitted against.
  std::vector<int64_t> free_exempt;
  std::vector<int64_t> free_ring_exempt;

  // Paper objective (SpatialBlend App. A eq. weighted):
  // Σ_p w_p (R^tgt_p(S) - R^src_p)^2, R^tgt_p evaluated with the SAME
  // per-family evaluator relation_evaluator.cc uses for target evaluation
  // (EvaluateTargetRelation), over the relations that entered the position
  // and/or yaw goal sets above (AffordancePose anchor poses are solved
  // separately by SolveActorTargets and are not part of this objective).
  // `objective_before` is the value at the goal-point centroid initializer
  // (the earlier solve this replaces); `objective_after` is the value of
  // whatever state was actually emitted -- the refined state when it
  // strictly improved on the initializer, else the initializer itself
  // (so objective_after <= objective_before always, and == when
  // `objective_fallback` is set). `objective_iterations` is the Gauss-Newton
  // iteration count (0 when nothing was evaluable, or the gradient was
  // already within tolerance). A solve that hits the loop guard instead is
  // `kNotConverged`. `objective_relation_count` is
  // the number of position/yaw-goal relations that were evaluable by
  // Eval_p at the initializer (a relation Eval_p cannot ground -- e.g. an
  // unmapped or ungroundable receiver counterpart -- does not enter the
  // sum, matching Eval_p's own disposition semantics).
  double objective_before = 0.0;
  double objective_after = 0.0;
  int objective_iterations = 0;
  bool objective_fallback = false;
  int objective_relation_count = 0;
  // The number of evaluate() calls the search actually used
  // (bounded by Config::objective_max_evaluations, never wall-clock) and how
  // many deterministic restarts (S0, the warm start, each decoded goal
  // point) it tried before returning the best one found.
  int objective_evaluations = 0;
  int objective_restarts = 0;
  // True when a caller-supplied SolverWarmStart seed was actually run as a
  // restart of the position and/or yaw search (not merely passed in): a seed
  // that duplicates the initializer, or is infeasible, is not "used".
  bool objective_warm_start_used = false;
  // Sign of the signed Facing offset theta_tgt of the heaviest well-conditioned
  // Facing term at the answered yaw: +1 / -1, or 0 when theta is exactly 0 (no deadband)
  // or there is no Facing term. A change of this sign between
  // two solves is the +-theta root switch: attributed, never smoothed.
  int facing_sign = 0;
  // The circular resultant of the Facing bearings is within the error interval of its own computation, so it has no
  // direction the arithmetic can name (an exact or numerically unresolved tie of opposing bearings). The yaw is then the
  // deterministic continuation, the middle nearest the embodied yaw (see the yaw stage of SolveImpl). `facing_resultant`
  // is |resultant| / (sum of the weights), `facing_resultant_bound` the interval it is judged against, the same units.
  bool facing_tie = false;
  double facing_resultant = 0.0, facing_resultant_bound = 0.0;
  // The yaw is a ratio continuation (some Facing stepped 1/n of its own offset, n > 1): its exactness is
  // judged against the stepped targets, as a position's against d*. `yaw_step_residual` is the largest angle between the
  // answered yaw and a relation's own step target (0 for one Facing; a weighted mean of several misses some).
  bool yaw_stepped = false;
  double yaw_step_residual = 0.0;
  // Reported only: the largest turn a stepping Facing requests this update (|its offset's move| / n, radians; 0 when
  // none steps). The output is an absolute yaw, so a requested turn above pi is carried as another turn (the shorter
  // one the other way) until the signed-turn output of the Unity integration.
  double yaw_requested_turn_max = 0.0;
  // ABI 4.12: the signed, unwrapped yaw turn from the current yaw that the answer requests (radians; congruent to yaw
  // modulo 2 pi): the shortest arc to the yaw, except when a weighted stepping Facing itself requests pi or more (a
  // source turning away the long way round): then the branch is the weighted mean of the stepping Facings' own branches
  // relative to the answer, every arriving Facing counting branch 0, and a non-zero branch is taken only when that mean
  // is beyond one half (a mean of exactly +-0.5 is branch 0). Placed (first proposal, source exception): the shortest
  // arc; a hold, 0; 0 when the yaw is unspecified.
  double yaw_turn = 0.0;
  // False when no Facing relation entered the solve: the yaw is the held yaw, not an
  // answer (the driver faces its direction of motion).
  bool yaw_specified = true;
  // The initializer S0 the position search started from (the decode after the
  // feasibility clamp), valid when has_objective_seed. `position -
  // objective_seed_position` is what the search added.
  bool has_objective_seed = false;
  // The answer is a minimiser: its exact J <= epsilon (Config::
  // objective_improve_epsilon_m)^2 (false when nothing was evaluable).
  bool objective_exact = false;
  // The position answer is a placement at the approach pose (the first
  // sample, or a declared discontinuity), not a step from the embodied sample.
  bool absolute_placement = false;
  // The position answer is the chart pose of a source that is inside its referent or
  // at contact (v1's inside-volume snap), not a pursuit step. It is answered only while
  // the engaged relations carry all the weight, where it is the minimiser of J; with
  // weight on any other relation the engaged referent is one term of J.
  bool chart_snap = false;
  // No relation issued a request: every spacing update count is infinite, so the
  // position is the embodied sample (v1 plans nothing for such a relation).
  bool no_request = false;
  // Native request presence is independent of the legacy held-pose/yaw flags.
  // An active relation may make no new request; the driver's pending target
  // then survives. Set only by the terms actually admitted by this solve.
  bool position_active = false, yaw_active = false;
  bool position_requested = false, yaw_requested = false;
  // True when the Gauss-Newton loop ended by one of its stop rules (no damped step lowering J before the step fell
  // below the data's precision, or a step below the data's precision) and not by the safety cap.
  bool objective_stationary = false;
  // Why the multi-relation Gauss-Newton loop ended (observation only): no damped step lowering J before the step
  // fell below the data's precision (with the last step tried an active-set step on the boundary of F, or not); a
  // step below the data's precision; or the cap.
  enum class GnStop { kNone, kNoDampedStep, kNoDampedStepOnBoundary, kPrecision, kCap, kBudget };
  GnStop gn_stop = GnStop::kNone;
  // Iterates projected onto F by more than the free space's boundary hair (observation only).
  int projection_excursions = 0;
  // Route lists asked of the free space (GeodesicRoutes / ContactRoutes: a walk of the route grid each) and retractions that
  // reached their bound of 60 chord steps, over the solve (observation only: an abandoned chord keeps the best point J has judged and is not
  // a failure of the solve; the harness reports the frames that had one, n_retract_cap_frames).
  int route_list_queries = 0;
  int retract_cap_hits = 0;
  // Descents along the direction chi measured that were tried (U1 and U2 held with U3 failing, or the radius collapsed above the
  // threshold) and that moved the iterate (observation only).
  int chi_descent_tries = 0;
  int chi_descent_taken = 0;
  // Terms whose distance had no analytic derivative and were differenced (observation only; see term_deriv).
  int fd_terms = 0;
  Eigen::Vector2d objective_seed_position = Eigen::Vector2d::Zero();
  // Echo of SolverWarmStart::cold: the source declared a discontinuity (a
  // teleport or reset), so the solve ran without continuing from the seed.
  bool source_discontinuity = false;

  // Observation (never read by the solver itself):
  std::vector<SolverGoalWitness> goals;
  // Weighted blend of the position witnesses before any feasibility clamp.
  Eigen::Vector2d unconstrained_position = Eigen::Vector2d::Zero();
  bool has_unconstrained_position = false;
  // True when the feasibility clamp (F(t) projection) moved the candidate at
  // least once; feasibility_snap_m = |position - unconstrained_position|.
  bool feasibility_snapped = false;
  // The constraint branch of the clamp (ObstacleFreeSpace::NearestFree); 0 when S0 was free.
  int64_t clamp_branch = 0;
  // A hash of the footprints the free space exempts for this solve (the referents the
  // source stands on or touches): the clamp's constraint SET. It changes when the
  // source engages or releases a referent.
  uint64_t clamp_exempt_hash = 0;
  double feasibility_snap_m = 0.0;

  // Spacing terms the ratio objective admitted (finite n). The closure from
  // the embodied sample is sample_distance_m / update_count. Recorded so a
  // caller can recompute that quotient; the solver does not assert a step.
  struct RatioIntent {
    double sample_distance_m = -1.0;
    double update_count = 0.0;
    double weight = 0.0;
    // Single relation: the arc length the receiver walked along its smoothed
    // course, L/n (v1 CalculateDistanceToMove). Negative when the proposal is
    // not a one-relation course step.
    double step_m = -1.0;
    // Index of the payload (into the payloads passed to Solve) this term answers.
    size_t payload_index = 0;
    // An object relation the source is engaged on measures the geodesic length of the course from the
    // pose to `goal` (the mapped contact) instead of the relation's own distance; recorded so an
    // observer evaluates the same objective.
    bool engaged_goal = false;
    Eigen::Vector2d goal = Eigen::Vector2d::Zero();
    // The inputs of an object relation's one-relation step, so an observer can
    // recompute it: the geodesic course from the sample to the mapped contact, the
    // mapped entering direction (when the leg has one), and whether the course was
    // smoothed into the Bezier course (else the discrete course was walked).
    // (course is recorded for every one-relation step; course_relation says whether the arc
    // is the course's own L/n, an object relation, or min(d(sample)/n, L), a border relation.)
    std::vector<Eigen::Vector2d> course;
    bool course_relation = false;
    bool smoothed = false;
    bool has_entering = false;
    Eigen::Vector2d entering = Eigen::Vector2d::Zero();
  };
  std::vector<RatioIntent> ratio_intents;

  // One entry per payload. A POSITION relation that existed and every one
  // dropped (unmapped / ungroundable) sets no_grounded_referent: nothing to map
  // (D7), position and yaw echo the sample.
  std::vector<SolverRelationOutcome> relations;
};

// A single solved target with temporal arrival metadata. For a newly supplied
// target schedule, arrival_time is a remaining-duration hint in seconds;
// consumers that pace motion must consume it across ticks rather than restart
// it on every tick.
struct RealizationTarget {
  double arrival_time = 0.0;
  std::string anchor_name;
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  // Composed support before any reach admission.  Present only for an
  // actual clamp operation; `position` is then the admitted target.
  Eigen::Vector2d requested_position = Eigen::Vector2d::Zero();
  bool has_requested_position = false;
  // World height of the target. Only anchor (body-part) targets carry it — an
  // hand target on a board or a pelvis target on a seat is meaningless
  // without Z. Root locomotion targets keep has_z=false (ground-projected).
  double z = 0.0;
  bool has_z = false;
  double yaw = 0.0;
  RelationKernelType kernel_type = RelationKernelType::kAvatarFacing;
  shared_element_id_t referent_id = kInvalidSharedElementId;
  double weight = 0.0;
  // Reach diagnostic for an engaged body-anchor target: the contact
  // target is never moved (a moved target is not the contact), but when it
  // lies beyond the anchor's reach from the solved root this is flagged with
  // the excess horizontal distance in metres.
  bool beyond_reach = false;
  double reach_excess_m = 0.0;
  // An AffordancePose target that is the exact mapped support of the
  // single relation the source anchor is engaged on (not a blend).
  bool engaged_support = false;
};

enum class AffordanceGroundingMode {
  // A named target sub-region supplied both the boundary and local plane.
  kExplicitSubRegion,
  // The target's whole Surface/Navigable boundary supplied its native plane.
  kWholePlanarBoundary,
  // RETIRED: never produced. A geometryless referent is now a
  // grounding failure rather than a target at its origin. Kept so the
  // enumerator values seen by native consumers do not shift.
  kLegacyCenterFallback,
};

// Collection of per-bucket solved targets.
struct RealizationResult {
  std::vector<RealizationTarget> targets;
  SolverStatus status = SolverStatus::kConverged;
  // AffordancePose anchors whose gate is open but whose source anchor
  // is not engaged on a support: no target by design (the body drives them).
  std::vector<std::string> awaiting_engagement;
  // Per-payload outcomes of the root position and facing solves
  // (payload_index into the payloads given to SolveActorTargets).
  std::vector<SolverRelationOutcome> relations;
};

// Native-only observation of the live realization producer.  This is
// deliberately not a C ABI type: the rich C ABI owns versioning, lifetime and
// callback negotiation.  It keeps exact support mappings distinct from the
// composed channel target so a blend cannot be mistaken for a contact assertion.
enum class RealizationGeometry {
  kNotApplicable,
  kExactSupport,
  kBlendedSupport,
  kReachProjected,
  kLegacyCenterNonContact,
  kUnsupported,
};

enum class RealizationGoalKind {
  kNone,
  kRootPosition,
  kFacingDirection,
  kSupportPose,
  kSoftGeometricPose,
};

enum class RealizationDisposition {
  kIncluded,
  kDisabled,
  kUnmapped,
  kGroundingFailed,
  kUnsupported,
  // The relation is admitted but the source anchor is not (yet) on its
  // support, so the anchor follows the body. A normal state, not a failure.
  kAwaitingEngagement,
};

struct RealizationContributor {
  shared_element_id_t referent_id = kInvalidSharedElementId;
  space_id_t source_space_id = kInvalidSpaceId;
  element_id_t source_element_id = kInvalidElementId;
  element_id_t resolved_local_element_id = kInvalidElementId;
  uint32_t snapshot_ordinal = 0;
  double applied_weight = 0.0;
  bool has_mapped_support = false;
  Eigen::Vector3d mapped_support = Eigen::Vector3d::Zero();
  AffordanceGroundingMode grounding_mode =
      AffordanceGroundingMode::kExplicitSubRegion;
  RealizationDisposition disposition = RealizationDisposition::kUnmapped;
  std::string reason;
  Eigen::Matrix4d support_plane_to_world = Eigen::Matrix4d::Identity();
  std::string support_identifier;
  bool chart_version_known = false;
  uint64_t binding_revision = 0;
  bool binding_revision_known = false;
};

struct RealizationChannel {
  std::string anchor_name;
  RelationKernelType family = RelationKernelType::kAvatarFacing;
  SolverStatus status = SolverStatus::kConverged;
  bool has_goal = false;
  RealizationGeometry geometry = RealizationGeometry::kUnsupported;
  RealizationGoalKind goal_kind = RealizationGoalKind::kNone;
  RealizationTarget target;
  bool reach_projected = false;
  double reach_displacement_m = 0.0;
  std::vector<RealizationContributor> contributors;
};

struct ActorRealizationSnapshot {
  uint64_t revision = 0;
  // Internal sidecar: never added to the frozen rich-v1 header/masks.
  ReceiverInputStamp receiver_input;
  SolverStatus solve_status = SolverStatus::kConverged;
  // Causal metadata is captured while Space::OnRemoteUpdate holds its
  // solve mutex.  It deliberately lives beside (rather than replacing) the
  // geometry snapshot so native consumers can freeze one transaction.
  enum class SourceTimingMode { kUnknownLegacy, kTimestamped, kReset };
  SourceTimingMode source_timing_mode = SourceTimingMode::kUnknownLegacy;
  uint64_t source_epoch = 0;
  // Relation-delivery counter, distinct from the original observation sample
  // sequence below. Resets consume delivery sequences too.
  int64_t source_sequence = 0;
  int64_t source_observation_sequence = 0;
  bool source_observation_sequence_known = false;
  double source_time_seconds = 0.0;
  bool source_time_known = false;
  double source_clock_seconds = 0.0;
  bool source_clock_known = false;
  space_id_t source_space_id = kInvalidSpaceId;
  space_id_t relay_source_space_id = kInvalidSpaceId;
  element_id_t source_actor_element_id = kInvalidElementId;
  // CLOCK_MONOTONIC-like process-local steady-clock readings.  Zero means no
  // receiver ingress exists (e.g. a local teardown), never a fabricated time.
  int64_t receiver_ingress_monotonic_ns = 0;
  int64_t receiver_solve_monotonic_ns = 0;
  // Revision domains are intentionally independent of the per-solve revision.
  // A zero value is never interpreted without the corresponding known bit.
  uint64_t policy_revision = 0;
  uint64_t tree_revision = 0;
  uint64_t counterpart_revision = 0;
  uint64_t support_revision = 0;
  uint32_t known_revision_mask = 0;
  uint32_t unknown_revision_mask = 0;
  std::vector<RealizationChannel> channels;
};

struct GroundedAffordanceTarget {
  element_id_t local_element_id = kInvalidElementId;
  shared_element_id_t referent_id = kInvalidSharedElementId;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  AffordanceGroundingMode mode = AffordanceGroundingMode::kExplicitSubRegion;
  // Captured at mapping time; unavailable source fields remain explicitly
  // unknown rather than being inferred from object/world yaw.
  Eigen::Matrix4d support_plane_to_world = Eigen::Matrix4d::Identity();
  std::string support_identifier;
  bool chart_version_known = false;
  // The source binding revision carried by the payload.
  uint64_t binding_revision = 0;
  bool binding_revision_known = false;
};

// ---------------------------------------------------------------------------
// Body-anchor engagement is GEOMETRIC (engagement.h):
// an AffordancePose relation is engaged while the source anchor touches its support
// surface (IsAffordanceContact). There is no distance threshold, no stillness, no
// dwell and no hysteresis, and no seat authority over the root's yaw: the anchor's
// own proposal carries the seat pose, and the driver / IK realizes it. While not
// engaged the anchor has NO support target and follows the actor's body.
//
// Source support distance of an AffordancePose payload in metres; +infinity when
// unknown.
double AffordanceSupportDistance(const RelationPayload& payload);

// Circular interpolation from `from` to `to` by alpha in [0,1] along the
// shorter arc.
double BlendYaw(double from, double to, double alpha);


// Facing weight. The footprint fade, the 0.05/0.25 m distance fade, and the
// away-R fade are removed: core reports the bearing. A Facing is left out of
// the yaw only when its look-at lies inside the body clearance, or when the
// scope gate has already dropped it as occluded. Whether to act is the
// driver's decision.
double FacingBearingConditioning(const Element& referent,
                                 const ElementTree& tree,
                                 const Eigen::Vector2d& position,
                                 double distance_to_look_at,
                                 double yaw_offset);

// Clamps `point` to lie within `radius` (horizontal, XY-only) of `center`.
// Used to keep affordance-anchor targets (hand, pelvis) from being pulled
// further from the solved root than the anchor's physical reach allows —
// e.g. a hand target on a large tabletop should not exceed arm's length from
// the root even if the decoded support point is further away. Returns the
// clamped point; `point` is left unchanged if already within radius. Pure
// function, no state.
Eigen::Vector2d ClampToReach(const Eigen::Vector2d& point,
                             const Eigen::Vector2d& center, double radius);

// Normalizing-scope rule: the receiver navigable corresponding to the
// scope that normalized the payload's R at the source -- the receiver-resolved
// counterpart of the payload's source scope (payload.resolved_scope_element_id;
// a payload no receiver resolved is matched by the scope's shared id), else
// the nearest navigable enclosing the receiver referent's parent (the
// structural correspondent). A receiver-resolved payload never falls back to
// shared-id lookup here (linker-only mode). nullptr when none.
// `scope_element` receives its element id.
const NavigableSpatialDefinition* CorrespondingScope(
    const RelationPayload& payload, const ElementTree& target_tree,
    element_id_t* scope_element = nullptr);

// The source's own heading expressed on the receiver -- for the
// DRIVING (highest-weight) enabled Spacing payload, the receiver referent yaw
// + the source's yaw relative to that referent (carried in the Spacing
// encoding). Not averaged across referents: the receiver may orient them
// differently, and near-opposite headings would cancel.
// It follows the source's turns and is the facing fallback when every
// bearing is conditioned out. nullopt when no Spacing payload carries a yaw.
std::optional<double> ReferentRelativeHeading(
    const std::vector<RelationPayload>& payloads,
    const std::vector<double>& weights, const ElementTree& target_tree,
    const ReferentResolver* resolver = nullptr);

// Normalizing context of a Spacing payload -- the ONE rule shared by the
// decode that chooses S (DecodeTargetPoint) and by Eval_p
// (EvaluateTargetRelation), keyed on what the source actually did:
//   kScoped       source_scope_diameter_m > 0: R = 1 - d/D over the source
//                 scope; the receiver uses CorrespondingScope (the resolved
//                 counterpart, else the structural correspondent).
//   kUnscoped     no source scope but a metric support distance: R = 1/(1+d)
//                 (the kernel's scope-less definition) on both sides.
//   kLegacyScoped neither field (a sender predating both): R = 1 - d/D over
//                 the structural correspondent, as that sender normalized.
// Presence, not value, decides: support_distance_m < 0 means absent (the wire
// field is proto3 `optional`, so a legacy 0.0 is absent, not "inside").
struct SpacingNormalization {
  enum class Mode { kScoped, kUnscoped, kLegacyScoped };
  Mode mode = Mode::kUnscoped;
  const NavigableSpatialDefinition* scope = nullptr;
  element_id_t scope_element = kInvalidElementId;
  double diameter_m = 0.0;
};
SpacingNormalization ResolveSpacingNormalization(const RelationPayload& payload,
                                                 const ElementTree& target_tree);

// The source support of a Spacing/Facing payload decoded onto the receiver
// referent (world XY): the contact chart mapped onto the target boundary, or
// the referent origin when the SOURCE support was the origin (no contact).
// Errors when the source contact cannot be reproduced on the target.
absl::StatusOr<Eigen::Vector2d> DecodeMappedSupportPoint(
    const RelationPayload& payload, const Element& target_referent,
    const ElementTree& target_tree);

// Proposes the receiver actor's position and yaw from the source's relations.
//
// Each kernel declares which state components are dynamic (position, yaw).
// The solver decodes a goal point per payload (the source contact read on the
// receiver support through the one chart). Position and yaw are solved
// separately:
//   - Position follows v1: the first sample and a
//     declared discontinuity are placed at the approach pose; a source inside
//     its referent or at contact snaps the receiver to the chart pose; otherwise
//     each relation's mapped contact is pursued from the EMBODIED sample by 1/n
//     of the course (one relation) or by v1 StepSync (several). An infinite
//     update count issues no request.
//   - Yaw: weighted circular mean of bearings to target points.
//
// The solver is policy-transparent: it receives pre-computed weights from a
// RealizationPolicy.
// Continuation: the position step starts from the EMBODIED pose the driver
// published, not from core's previous answer. Space samples the port and passes
// the sample here on every solve; a direct Solve() call with no seed (a test)
// leaves it empty.
struct SolverWarmStart {
  // The EMBODIED pose the driver last published (sampled by Space from the
  // port): the continuation seed of the position tracking pass and the
  // reference of the Facing sign rule. It is an input of the call, never
  // core's previous answer.
  bool has_position = false;
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  bool has_yaw = false;
  double yaw = 0.0;
  // The source declared a discontinuity (teleport / reset): no continuation.
  bool cold = false;
  // First sample, or a declared discontinuity. The position stage places the
  // body at the approach pose: the source offset through the same-id support
  // chart. A later update leaves this false and pursues each relation's mapped
  // contact from the embodied sample.
  bool absolute = false;
  // Approach points the caller already decoded through a chart of its own (payload index
  // -> point), used instead of the decode: the containment transfer's mapped support is the
  // canonical chart of a frame element, so the S0 pipeline takes the contained pose for
  // that element's own referent instead of a second chart of the same point.
  std::vector<std::pair<size_t, Eigen::Vector2d>> decode_overrides;
  // The receiver's previous turn direction: the sign (+1, -1) of the last non-zero step of its sampled yaw, 0 when it
  // has not turned. It breaks the Facing yaw's ties (the direction it was already turning wins).
  int turn_sign = 0;
};

class RelationSolver {
 public:
  struct Config {
    // The multi-relation position step iterates to
    // convergence. This is a safety cap on a loop that never meets a stop
    // rule, not a convergence budget; reaching it is NOT_CONVERGED.
    int gn_safety_cap = 10000;

    // A safety net, not a stop rule: the relation evaluations one Solve() may use (counted as kP1SolveBudgetEvals above says: per position term per evaluation of J; derivatives and route listings are not counted) (kP1SolveBudgetEvals is P-1). On by default: the product
    // path and the scenario harness run with it. 0 or below disables it (a test of the unbudgeted answer). Checked at the outer
    // iteration boundary and inside the inner loops that can run long (the chord retraction, the descent along chi, the radius
    // bisection, the face and tie candidates). On a hit the solve returns its best feasible incumbent with status kBudgetHit; it is
    // never reported as converged. One inner step is not interruptible, so a hit overshoots the cap by at most one step's evaluations.
    int solve_budget_evals = kP1SolveBudgetEvals;

    // Exactness tolerance, metres of Spacing. A channel is exact when
    // |R_tgt - R_src| * K <= this, with K the metres-per-score of R
    // (the scope diameter D for R = 1 - d/D, and 1/R^2 at the sample for
    // the unscoped R = 1/(1+d)). A channel with no metre scale uses this
    // value directly as an R tolerance. Zero evaluable relations is not
    // exact. The same 1 mm is the continuation's improvement scale.
    //
    // Trust radius of one proposal, derived, not fitted:
    //   q_p = R_src,p - R_tgt,p(sample),  b = ||q||_2,  rho = K * b.
    // K is that same metres-per-score (Spacing is Lipschitz with constant K).
    // A zero residual has rho 0: the proposal is the sample. There is no
    // extra cap. Yaw is the circular mean, not a step in this radius.
    double objective_improve_epsilon_m = 1e-3;
  };

  RelationSolver() = default;
  explicit RelationSolver(const Config& config) : config_(config) {}
  const Config& config() const { return config_; }

  // Solve with externally-provided weights (policy-transparent).
  // weights[i] corresponds to payloads[i]. If empty, uniform weights are used.
  // The position is constrained to the navigable's FREE space (walkable,
  // >= body radius from obstacles). The only exception is the footprint
  // of a referent the source root is ON (spacing R_src = 1), so a seat
  // being sat on -- even one mapped outside the walkable polygon -- is not
  // snapped away, while an approached-but-unengaged seat never pulls the
  // root onto itself. An infeasible outcome (no free space, or a target not
  // connected to the current root) is reported in SolverResult::feasibility;
  // there is no hold (D7). `resolver` plugs in referent-id ->
  // local-element resolution (default: exact shared-id match).
  SolverResult Solve(const Eigen::Vector2d& current_position,
                     double current_yaw,
                     const std::vector<RelationPayload>& payloads,
                     const std::vector<double>& weights,
                     const ElementTree& target_tree,
                     const NavigableSpatialDefinition* navigable_def = nullptr,
                     const ReferentResolver* resolver = nullptr,
                     std::optional<double> heading = std::nullopt,
                     const SolverWarmStart& warm_start = {}) const;
  // `heading`: facing fallback when every bearing is conditioned out;
  // default: ReferentRelativeHeading over `payloads`.
  // `warm_start`: see SolverWarmStart.

  // Step-synchronized progressive solve for controlled avatars. Each position
  // relation is converted into a next-step target by reducing its distance by
  // d_current / estimated_steps, then those step targets are blended with the
  // provided annealed weights.
  SolverResult SolveProgressive(
      const Eigen::Vector2d& current_position, double current_yaw,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const ElementTree& target_tree,
      const NavigableSpatialDefinition* navigable_def = nullptr,
      const ReferentResolver* resolver = nullptr) const;

  // Convenience overload: uniform weights (for unit tests / no-policy usage).
  SolverResult Solve(const Eigen::Vector2d& current_position,
                     double current_yaw,
                     const std::vector<RelationPayload>& payloads,
                     const ElementTree& target_tree,
                     const NavigableSpatialDefinition* navigable_def = nullptr,
                     const ReferentResolver* resolver = nullptr) const;

  // Two-stage realizer. Stage A: payloads are
  // grouped by (kernel family, subject anchor) and each group is solved
  // INDEPENDENTLY for its own state component — spacing → root position,
  // facing → yaw, affordance → one world pose per body anchor. No group ever
  // sees another family's payloads, so a hand/pelvis relation can never drag
  // root (and vice versa). Stage B composes them in dependency order: root
  // position first, then yaw with facing bearings evaluated AT the composed
  // position, then per-anchor poses.
  //
  // Returns targets[0] = root (anchor_name "root", position+yaw, held at the
  // current pose when nothing drives it), followed by one target per
  // affordance anchor group (world position, Z, yaw; has_z=true).
  // weights[i] corresponds to payloads[i]; empty = uniform. The solver is
  // IDEMPOTENT: it emits the committed target; pacing toward it is the
  // controller's job (velocity-bounded follower), not a per-frame step here.
  RealizationResult SolveActorTargets(
      const Eigen::Vector2d& current_position, double current_yaw,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const ElementTree& target_tree,
      const NavigableSpatialDefinition* navigable_def = nullptr,
      const SolverWarmStart& warm_start = {}) const;

  // Grounds an affordance payload to a target position and yaw.  This first
  // contract accepts finite radial charts (angle in [-pi, pi], ratio in [0,1])
  // over a named sub-region or whole planar Surface/Navigable definition.
  // A referent without support geometry returns FailedPrecondition (it is
  // never a target at its origin).  Missing identities/patches (NotFound),
  // stale local IDs, malformed charts, and ambiguous unannotated volumes
  // return a non-OK status; MVC charts return Unimplemented.
  absl::StatusOr<GroundedAffordanceTarget> GroundAffordanceTarget(
      const RelationPayload& payload, const ElementTree& target_tree) const;

  // Groups payloads by arrival_time bucket (floor(arrival_time / bucket_size))
  // and solves each bucket independently, producing one RealizationTarget per
  // bucket. arrival_time = payload.estimated_time (already in seconds). Target
  // metadata is taken from the first payload in each bucket.
  RealizationResult SolveTargets(
      const Eigen::Vector2d& current_position, double current_yaw,
      const std::vector<RelationPayload>& payloads,
      const std::vector<double>& weights, const ElementTree& target_tree,
      double step_duration,
      const NavigableSpatialDefinition* navigable_def = nullptr,
      double bucket_size = 0.5) const;

 private:
  Config config_;
};

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RELATION_SOLVER_H_
