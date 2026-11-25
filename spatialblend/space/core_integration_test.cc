// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// The core's cross-space integration suite, modelled on v1's (parameterized cross-space scenarios, the contact
// guarantee, step-based weighting). It is the acceptance harness of every core module.
//
// A source actor walks in its space; a receiver in another space, whose body is moved EXACTLY to every proposal
// (IdealDriver: no kinematic limit), pursues the relations it receives. The receiver goes to the counterpart of the SOURCE's contact point, the chart
// image of it on its own referent (the mapped contact c), and the ratio comes from the source's count. Every update of
// every scenario is checked against that rule, not only the end:
//   * ratio: with exactly one request (a relation with weight > 0 and a valid count n), the receiver's distance to the
//     relation's mapped contact closes 1/n of itself: g_new = g_old (1 - 1/n), g the receiver's geodesic to c (the
//     evaluator's contact_distance_m), within kContactEpsilonM;
//   * hold: with no request the body does not move (within kContactEpsilonM);
//   * no teleport: only the first proposal (the join) is absolute;
//   * arrival: the source arrives when its distance to the target referent is below kContactEpsilonM; the receiver
//     arrives when its distance to the MAPPED contact (the chart image of the source's contact point) is below
//     kContactEpsilonM, not when it touches the counterpart anywhere. Both must arrive, on the same update.
// With several requests at once the relations are minimised together and no single ratio is exact: the worst
// per-relation ratio deviation is reported, and the solve must converge (n_not_converged == 0).
// After the join every scenario moves the receiver body elsewhere (a driver-side move, not a core teleport), so the
// join placement does not do the pursuit's work.
//
// Scenarios vary room shape (other sizes and aspect, an L-shaped non-convex receiver room), object layout and size
// (identical, scaled, moved, rotated, a non-box prism, an extra obstacle in the receiver's way), the approach (straight
// from four directions, diagonal, curved, to the far side, stop-and-go, receding then re-approaching, switching target,
// passing by a referent, two referents together, jitter around the contact epsilon), speed, policy (Progressive,
// Uniform, Greedy) and kernel (Spacing, Facing). Sitting, standing up and leaving an object come with the inside-object
// module (D): they are DISABLED placeholders below.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "spatialblend/model/element/navigable_spatial_definition.h"
#include "spatialblend/model/element/polytope_spatial_definition.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/space/testing/ideal_driver.h"
#include "spatialblend/space/local_context.h"
#include "spatialblend/solver/relation_evaluator.h"
#include "spatialblend/solver/relation_solver.h"
#include "spatialblend/model/realization/remote_update.h"
#include "spatialblend/space/space.h"
#include "spatialblend/space/testing/local_minimum.h"

namespace spatialblend {
namespace {

using Vec2 = Eigen::Vector2d;
constexpr double kDt = 1.0 / 30.0;  // the update interval the source sends at
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---- geometry ------------------------------------------------------------------------------------------------------

struct Obj {
  Vec2 c;
  Vec2 half;
  double yaw = 0.0;       // rotation about z (radians)
  bool prism = false;     // a triangular prism inside the half extents instead of a box
  bool referent = true;   // false: an obstacle without a shared id (never a referent)
};
struct RoomSpec {
  std::string name;
  std::vector<Vec2> walkable;
  std::vector<Obj> objs;  // referent k carries shared id 300 + k on both sides
};

RoomSpec Rect(const std::string& name, double w, double h, std::vector<Obj> objs) {
  return {name, {{0, 0}, {w, 0}, {w, h}, {0, h}}, std::move(objs)};
}
// An L: a w x h rectangle with its upper-right quadrant (x > cx, y > cy) removed.
RoomSpec LRoom(const std::string& name, double w, double h, double cx, double cy, std::vector<Obj> objs) {
  return {name, {{0, 0}, {w, 0}, {w, cy}, {cx, cy}, {cx, h}, {0, h}}, std::move(objs)};
}
Obj Box(double x, double y, double hx, double hy, double yaw = 0.0) { return {{x, y}, {hx, hy}, yaw}; }
Obj Scaled(Obj o, double sx, double sy) {
  o.c = Vec2(o.c.x() * sx, o.c.y() * sy);
  o.half = Vec2(o.half.x() * sx, o.half.y() * sy);
  return o;
}
std::vector<Obj> ScaledAll(const std::vector<Obj>& v, double sx, double sy) {
  std::vector<Obj> out;
  for (const auto& o : v) out.push_back(Scaled(o, sx, sy));
  return out;
}

void Build(Space* space, const RoomSpec& spec) {
  auto* tree = space->GetElementTree();
  ASSERT_TRUE(tree->AddElement(1).ok());
  tree->GetElement(1)->SetSharedId(100);
  tree->GetElement(1)->SetSpatialDefinition(
      std::make_shared<NavigableSpatialDefinition>(std::make_shared<Polygon2D>(spec.walkable)));
  int shared = 300;
  for (size_t k = 0; k < spec.objs.size(); ++k) {
    const Obj& o = spec.objs[k];
    const element_id_t id = static_cast<element_id_t>(10 + k);
    ASSERT_TRUE(tree->AddElement(id, 1).ok());
    if (o.referent) tree->GetElement(id)->SetSharedId(static_cast<shared_element_id_t>(shared++));
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t(0, 0) = std::cos(o.yaw);
    t(0, 1) = -std::sin(o.yaw);
    t(1, 0) = std::sin(o.yaw);
    t(1, 1) = std::cos(o.yaw);
    t(0, 3) = o.c.x();
    t(1, 3) = o.c.y();
    tree->GetElement(id)->SetLocalTransform(t);
    if (o.prism) {
      // A triangle (-hx, -hy), (hx, -hy), (0, hy), extruded to 0.8 m.
      const double hx = o.half.x(), hy = o.half.y();
      std::vector<Eigen::Vector3d> v = {{-hx, -hy, 0}, {hx, -hy, 0}, {0, hy, 0},
                                        {-hx, -hy, 0.8}, {hx, -hy, 0.8}, {0, hy, 0.8}};
      std::vector<std::vector<int>> f = {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
      tree->GetElement(id)->SetSpatialDefinition(
          std::make_shared<PolytopeSpatialDefinition>(std::move(v), std::move(f), GeometryRole::kObstacle));
    } else {
      tree->GetElement(id)->SetSpatialDefinition(std::make_shared<PolytopeSpatialDefinition>(
          Eigen::Vector3d(-o.half.x(), -o.half.y(), 0.0), Eigen::Vector3d(o.half.x(), o.half.y(), 0.8),
          GeometryRole::kObstacle));
    }
  }
}

// ---- source paths --------------------------------------------------------------------------------------------------

struct Step {
  Vec2 pos;
  double yaw = 0.0;
};
using Path = std::vector<Step>;

// Straight from the path's last point to b at `speed` (the first call starts at a).
void Line(Path* p, const Vec2& a, const Vec2& b, double speed, double yaw = 0.0) {
  if (p->empty()) p->push_back({a, yaw});
  const Vec2 from = p->back().pos;
  const int n = std::max(1, static_cast<int>(std::ceil((b - from).norm() / (speed * kDt))));
  for (int i = 1; i <= n; ++i) p->push_back({from + (b - from) * (double(i) / n), yaw});
}
void Pause(Path* p, int updates) {
  const Step last = p->back();
  for (int i = 0; i < updates; ++i) p->push_back(last);
}
// A quadratic Bezier from the last point through control q to b, sampled at roughly `speed`.
void Curve(Path* p, const Vec2& q, const Vec2& b, double speed) {
  const Vec2 a = p->back().pos;
  double len = 0.0;
  Vec2 prev = a;
  for (int i = 1; i <= 200; ++i) {
    const double t = i / 200.0;
    const Vec2 x = (1 - t) * (1 - t) * a + 2 * (1 - t) * t * q + t * t * b;
    len += (x - prev).norm();
    prev = x;
  }
  const int n = std::max(1, static_cast<int>(std::ceil(len / (speed * kDt))));
  for (int i = 1; i <= n; ++i) {
    const double t = double(i) / n;
    p->push_back({(1 - t) * (1 - t) * a + 2 * (1 - t) * t * q + t * t * b, 0.0});
  }
}

// ---- scenarios -----------------------------------------------------------------------------------------------------

enum class Kernel { kSpacing, kFacing };
// The Facing yaw follows the ratio continuation: each request turns the receiver 1/n of its own remaining offset,
// which differs from v1's placed yaw.
constexpr bool kFacingYawRatioPending = false;

struct Scenario {
  std::string name;
  RoomSpec src, rcv;
  int target = 0;           // the target referent's index among the REFERENTS (shared id 300 + target)
  Path path;
  Vec2 rcv_after_join = Vec2(kNaN, kNaN);  // where the receiver body is moved after the join
  double rcv_yaw_after_join = 0.0;         // and the yaw it is given there
  RealizationPolicyType policy = RealizationPolicyType::kProgressive;
  Kernel kernel = Kernel::kSpacing;
  bool several_requests = false;  // two referents approached together: their requests are checked jointly
  // The receiver room is the source's scaled by `similarity` (objects too) about the origin: the mapped contact is then
  // known in closed form, s times the source's contact point (NaN: not a similarity, not checked).
  double similarity = kNaN;
  // The receiver starts as the source's copy (in a similar room: s times its position, the same yaw) and is never moved:
  // it stays the copy on every update (receiver = s source, receiver yaw = source yaw).
  bool identity = false;
  int join_at = 0;        // the receiver joins at this update (its space connects then)
  int teleport_at = -1;   // the source declares a teleport (a source exception) on this update of its path
  bool blocked = false;   // F does not contain the mapped contact: the receiver can never arrive, and holds in F
  std::string anchor = "root";  // the source anchor the relations are bound to ("hand": a non-root anchor)
  // Two Facing referents whose targets are opposite on the receiver's side (no single yaw serves both): the receiver
  // has no arrival; its yaw must not alternate direction between updates (ties keep the turn).
  bool opposing_tie = false;
  // The source's look-at contact switches to another silhouette vertex on the path (it turns through the antipode of its
  // look-at): the scenario must exercise a switch, and the copy is asserted through it.
  bool look_switch = false;
  // The receiver's body turns at most this far per update (radians; NaN: the ideal driver, no bound). A test-only
  // rate-capped driver: the body moves to every proposal's position and turns toward its yaw along the short arc within
  // the cap, so the body lags a source that turns faster. The per-update ratio is then checked on the PROPOSAL (the step
  // from the lagging body), the copy is not asserted, and the body must never turn opposite to the source.
  double yaw_rate_cap = kNaN;
  // ABI 4.12: the capped driver turns by the proposal's signed turn (yaw_turn), not the short arc to its yaw; on
  // one-request Facing updates the body's turn must go the requested way (the rule's -move / n), also above pi.
  bool turn_driver = false;
};

struct Outcome {
  int updates = 0;
  int src_arrival = -1, rcv_arrival = -1;
  int single = 0, multi = 0, holds = 0;   // updates with one request, several, none
  double worst_ratio = 0.0;               // worst |g_new - g_old (1 - 1/n)| over single-request updates
  double worst_multi = 0.0;               // worst per-relation deviation over several-request updates (reported)
  int multi_not_minimum = 0;              // several-request answers that are not a local minimum of J (LocalMinimumOfJ)
  double worst_hold = 0.0;                // worst body move over no-request updates
  int unexpected_absolute = 0;            // absolute proposals other than the join and a declared teleport
  int not_converged = 0;
  int missing_target = 0;                 // updates whose proposal carries no target
  int facing_single = 0, facing_multi = 0, facing_holds = 0;
  double worst_facing = 0.0;              // Facing, one request: worst |o_new - o_old (1 - 1/n)|, degrees
  double worst_facing_hold = 0.0;         // Facing, no request: worst yaw change, degrees
  double worst_arrival_g = 0.0;           // the geodesic g the evaluator reports with the body AT the mapped contact
  double worst_closed_form = 0.0;         // worst |c - s c_src| (similarity rooms)
  int closed_form_checked = 0;
  double worst_identity = 0.0;            // identity scenarios: worst |receiver - source|
  double worst_identity_yaw_deg = 0.0;    // identity Facing scenarios: worst |receiver yaw - source yaw|, degrees
  int look_switches = 0;                  // updates whose source look-at point switched to another silhouette vertex
  int outside_f = 0;                      // updates whose body is not in the solve's F
  int turn_flips = 0;                     // consecutive non-zero yaw steps of opposite sign while pursuing in a tie
  int facing_ties = 0;                    // updates whose circular mean was a tie (SolverResult::facing_tie)
  // ABI 4.12: proposals whose signed turn is not congruent to their yaw (from the sampled yaw), and one-request Facing
  // updates of a receiver that does not translate whose signed turn is not the requested one (-move / n: its sign is
  // the chosen direction, its size may exceed pi).
  int turn_not_congruent = 0, turn_not_requested = 0, turns_over_pi = 0;
  int turn_driver_against = 0;  // turn_driver: one-request updates whose body turned against the requested turn
  int arrival_over_pi = 0;      // turn_driver: arrival updates (n <= 1) whose turn in the source's direction exceeds pi
};

double Deg(double rad) { return std::fabs(std::remainder(rad, 2.0 * M_PI)) * (180.0 / M_PI); }

// The nearest point of an axis-aligned box object (yaw 0, not a prism) to p, in the room's frame.
std::optional<Vec2> BoxNearest(const Obj& o, const Vec2& p) {
  if (o.prism || o.yaw != 0.0) return std::nullopt;
  return p.cwiseMax(o.c - o.half).cwiseMin(o.c + o.half);
}
const Obj* ReferentObj(const RoomSpec& r, shared_element_id_t id) {
  int shared = 300;
  for (const Obj& o : r.objs)
    if (o.referent && shared++ == static_cast<int>(id)) return &o;
  return nullptr;
}

// Test-only rate-capped driver (Scenario::yaw_rate_cap): the IdealDriver's follower, with the yaw turning toward the
// proposal's along the short arc by at most `cap` per proposal (an unspecified yaw holds). With `use_turn` (ABI 4.12,
// Scenario::turn_driver) it turns by the proposal's signed turn (AnchorProposal::yaw_turn, from the sampled yaw) instead
// of the short arc to its yaw, within the same cap: the requested direction, also above pi.
class CappedYawDriver {
 public:
  CappedYawDriver(Space* space, spatialblend::entity_id_t actor, const Vec2& start, double yaw, double cap,
                  bool use_turn = false)
      : space_(space), actor_(actor) {
    EmbodiedPoseInput in;
    in.x = start.x();
    in.y = start.y();
    in.yaw = yaw;
    in.epoch = 1;
    space->PublishEmbodiedPose(actor, in, 0.0, NextPortSequence(space, actor));
    observer_ = space->AddProposalObserver([space, actor, cap, use_turn](spatialblend::entity_id_t id,
                                                                     const ReceiverProposal& p) {
      if (id != actor) return;
      if (!(p.IsAbsolute() || p.status == ProposalStatus::kExact || p.status == ProposalStatus::kInexact)) return;
      EmbodiedSample body;
      space->SampleEmbodiedPose(actor, &body);
      EmbodiedPoseInput next;
      next.epoch = 1;
      next.x = p.x;
      next.y = p.y;
      if (p.IsAbsolute()) {
        next.yaw = p.yaw;
        next.flags = kEmbodiedTeleported;
      } else {
        const double turn = use_turn ? p.yaw_turn : std::remainder(p.yaw - body.yaw, 2.0 * M_PI);
        next.yaw = body.yaw + std::clamp(turn, -cap, cap);
      }
      space->PublishEmbodiedPose(actor, next, 0.0, NextPortSequence(space, actor));
    });
  }
  ~CappedYawDriver() { space_->RemoveProposalObserver(observer_); }
  CappedYawDriver(const CappedYawDriver&) = delete;
  CappedYawDriver& operator=(const CappedYawDriver&) = delete;
  // Place the body (a driver-side move, as IdealDriver::Publish).
  void Publish(double x, double y, double yaw) {
    EmbodiedPoseInput in;
    in.x = x;
    in.y = y;
    in.yaw = yaw;
    in.epoch = 1;
    space_->PublishEmbodiedPose(actor_, in, 0.0, NextPortSequence(space_, actor_));
  }

 private:
  Space* space_;
  spatialblend::entity_id_t actor_;
  uint64_t observer_ = 0;
};

struct Receiver {
  Space* space = nullptr;
  spatialblend::entity_id_t avatar = 0;
  uint32_t anchor_id = 0;  // the bound anchor (0: the root)
  std::unique_ptr<IdealDriver> root;
  std::unique_ptr<CappedYawDriver> capped;  // instead of `root` when the scenario caps the yaw rate
  uint64_t seq = 1000;
  // The bound anchor's sampled pose.
  Eigen::Vector3d Pos() const {
    EmbodiedSample e;
    space->SampleAnchorPose(avatar, anchor_id, &e);
    return Eigen::Vector3d(e.x, e.y, e.z);
  }
  double Yaw() const {
    EmbodiedSample e;
    space->SampleAnchorPose(avatar, anchor_id, &e);
    return e.yaw;
  }
};

Outcome RunScenario(const Scenario& s) {
  Outcome out;
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Build(src, s.src);
  const bool hand = s.anchor != "root";
  const std::vector<SubjectAnchor> anchors =
      hand ? std::vector<SubjectAnchor>{SubjectAnchor("root", Eigen::Vector3d(0, 0, 0)),
                                        SubjectAnchor("hand", Eigen::Vector3d(0, 0, 1.0))}
           : std::vector<SubjectAnchor>{SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  const auto actor = src->CreateActor(anchors);
  src->GetActor(actor)->SetElementId(1);
  src->Bind(actor, s.anchor,
            s.kernel == Kernel::kSpacing ? RelationKernelType::kAvatarSpacing : RelationKernelType::kAvatarFacing);
  src->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  Receiver r;
  const auto connect = [&](const Step& at) {
    r.space = ctx->AddSpace();
    Build(r.space, s.rcv);
    r.space->SetRealizationPolicy(s.policy);
    r.space->EnableObservabilityCapture(true);
    r.avatar = r.space->CreateActor(anchors);
    r.space->GetActor(r.avatar)->SetElementId(1);
    r.space->SetHub(ctx->GetHub());
    if (std::isfinite(s.yaw_rate_cap))
      r.capped = std::make_unique<CappedYawDriver>(r.space, r.avatar, at.pos, at.yaw, s.yaw_rate_cap, s.turn_driver);
    else
      r.root = std::make_unique<IdealDriver>(r.space, r.avatar, at.pos, at.yaw);
    if (hand) {
      r.space->AnchorIdFor(r.avatar, "hand", &r.anchor_id);
      EmbodiedPoseInput in;
      in.x = at.pos.x();
      in.y = at.pos.y();
      in.z = 1.0;
      in.yaw = at.yaw;
      in.epoch = 1;
      r.space->PublishAnchorPose(r.avatar, r.anchor_id, in, 0.0, ++r.seq);
    }
  };
  if (s.join_at == 0) {
    Step first = s.path.front();
    if (s.identity && std::isfinite(s.similarity)) first.pos = s.similarity * first.pos;
    connect(first);
  }
  const shared_element_id_t target_id = static_cast<shared_element_id_t>(300 + s.target);
  int last_turn = 0;
  uint64_t last_revision = 0;  // the revision of the last proposal checked for congruence
  std::optional<int> prev_vertex;  // the source's look-at vertex (target Facing) on the previous update

  for (size_t k = 0; k < s.path.size(); ++k) {
    const Step& step = s.path[k];
    if (static_cast<int>(k) == s.teleport_at) src->RaiseSourceException(actor, 2, nullptr, 0);
    if (hand) src->SetAnchorWorldPose(actor, "hand", Eigen::Vector3d(step.pos.x(), step.pos.y(), 1.0), step.yaw);
    if (!r.space && static_cast<int>(k) == s.join_at) connect(step);
    const bool joined_now = r.space && static_cast<int>(k) == s.join_at;
    const Eigen::Vector3d before3 = r.space ? r.Pos() : Eigen::Vector3d::Zero();
    const double yaw_before = r.space ? r.Yaw() : 0.0;
    src->UpdateActorState(actor, step.pos, step.yaw);
    if (!r.space) continue;
    ReceiverProposal p;
    if (!r.space->ReadProposal(r.avatar, &p) ||
        !(p.status == ProposalStatus::kExact || p.status == ProposalStatus::kInexact))
      ++out.missing_target;
    if (hand) {
      // The driver realizes the hand's proposal at once (no kinematic limit), as IdealDriver does the root's.
      for (const auto& a : p.anchors)
        if (a.anchor_name == "hand" &&
            (a.status == ProposalStatus::kExact || a.status == ProposalStatus::kInexact)) {
          EmbodiedPoseInput in;
          in.x = a.x;
          in.y = a.y;
          in.z = a.z;
          in.yaw = a.yaw_specified() ? a.yaw : yaw_before;
          in.epoch = 1;
          r.space->PublishAnchorPose(r.avatar, r.anchor_id, in, 0.0, ++r.seq);
        }
    }
    const bool declared = static_cast<int>(k) == s.teleport_at;
    if (p.IsAbsolute() && !joined_now && !declared) ++out.unexpected_absolute;
    // Only a proposal published on this update is from this sample. An update that publishes none (a source pose
    // unchanged from the last, as at a pause or a turnaround) leaves the previous proposal in place, already realized by
    // the driver: its yaw_turn is from the sample before, not from yaw_before.
    const bool fresh = p.revision != last_revision;
    last_revision = p.revision;
    if (r.space && fresh && p.yaw_specified() &&
        std::fabs(std::remainder(yaw_before + p.yaw_turn - p.yaw, 2.0 * M_PI)) > 1e-9)
      ++out.turn_not_congruent;
    const Eigen::Vector3d after3 = r.Pos();
    const Vec2 before = before3.head<2>(), after = after3.head<2>();
    // With a rate-capped body the rule is checked on the proposal (the step from the lagging body).
    const bool capped = std::isfinite(s.yaw_rate_cap);
    const double body_yaw_after = r.Yaw();
    const double yaw_after = capped && p.yaw_specified() ? p.yaw : body_yaw_after;
    Space::PolicyObservation po;
    Space::SolverObservation so;
    const bool have_po = r.space->CopyPolicyObservation(r.avatar, &po) && po.valid;
    const bool have_so = r.space->CopySolverObservation(r.avatar, &so) && so.valid;
    if (have_so && so.solve.status == SolverStatus::kNotConverged) ++out.not_converged;
    if (have_so && so.solve.facing_tie) ++out.facing_ties;
    // While the circular mean is a tie, consecutive non-zero yaw steps must keep one sign (a tie keeps the receiver's
    // turn direction); the record restarts when the tie ends.
    // The tie at n = 1: only while pursuing (a Facing request with n > 1) does a tie keep the turn; the
    // arrival update (every request at n = 1) takes the nearer middle, which may turn back.
    bool pursuing = false;
    if (have_po)
      for (size_t i = 0; i < po.payloads.size(); ++i)
        if (po.payloads[i].kernel_type == RelationKernelType::kAvatarFacing && i < po.weights.size() &&
            po.weights[i] > 0.0 && RatioRequest(po.payloads[i].update_count) && po.payloads[i].update_count != 1.0)
          pursuing = true;
    if (s.kernel == Kernel::kFacing && !joined_now) {
      if (!(have_so && so.solve.facing_tie) || !pursuing) {
        last_turn = 0;
      } else {
        const double step_yaw = std::remainder(yaw_after - yaw_before, 2.0 * M_PI);
        const int turn = (step_yaw > 0.0) - (step_yaw < 0.0);
        if (turn != 0) {
          if (last_turn != 0 && turn != last_turn) ++out.turn_flips;
          last_turn = turn;
        }
      }
    }
    if (have_so && !hand &&
        !r.space->IsFreeUnder(r.avatar, after, so.solve.free_exempt, so.solve.free_ring_exempt))
      ++out.outside_f;
    if (s.kernel == Kernel::kFacing) {
      const RelationPayload* tp = nullptr;
      if (have_po)
        for (const auto& q : po.payloads)
          if (q.kernel_type == RelationKernelType::kAvatarFacing && q.referent_shared_id == target_id) tp = &q;
      if (tp) {
        // A switch: the look-at moves to another silhouette vertex (Support::look_at_vertex; -1 while the gaze hits
        // the referent, where the look-at moves continuously with the gaze and the copy stays exact).
        const int vertex = tp->support.look_at_vertex;
        if (prev_vertex && vertex != *prev_vertex && vertex >= 0 && *prev_vertex >= 0) ++out.look_switches;
        prev_vertex = vertex;
      }
    }
    if (s.identity) {
      // The receiver is the source's copy: in a similar room, s times its position (the closed form), and its yaw.
      const double sim = std::isfinite(s.similarity) ? s.similarity : 1.0;
      out.worst_identity = std::max(out.worst_identity, (after - sim * step.pos).norm());
      if (s.kernel == Kernel::kFacing) {
        // The copy holds through a look-at SWITCH (a silhouette vertex change): the source counts the rotation toward
        // the current look-at contact from its previous sent pose (one point at two poses), so the receiver's step to
        // the new point is the source's own. Switches are counted to show the path exercises one; the copy is the
        // receiver's yaw against the source's, independent of the solver's own exactness report.
        out.worst_identity_yaw_deg = std::max(out.worst_identity_yaw_deg, Deg(yaw_after - step.yaw));
      }
    }
    // Join and declared teleport are placements, not continuations: the per-update rules start after them.
    if (!have_po || joined_now || declared) {
      if (static_cast<int>(k) == s.join_at && std::isfinite(s.rcv_after_join.x()))
        r.capped ? r.capped->Publish(s.rcv_after_join.x(), s.rcv_after_join.y(), s.rcv_yaw_after_join)
                 : r.root->Publish(s.rcv_after_join.x(), s.rcv_after_join.y(), s.rcv_yaw_after_join);
      ++out.updates;
      continue;
    }
    const ElementTree& rtree = *r.space->GetElementTree();
    int requests = 0, facing_requests = 0;
    std::vector<double> errs, facing_errs, facing_turns;
    for (size_t i = 0; i < po.payloads.size(); ++i) {
      const RelationPayload& pay = po.payloads[i];
      if ((pay.subject_anchor_name.empty() ? std::string("root") : pay.subject_anchor_name) != s.anchor) continue;
      const double w = i < po.weights.size() ? po.weights[i] : 0.0;
      const bool request = w > 0.0 && RatioRequest(pay.update_count);
      const auto prep = PrepareTargetRelation(pay, rtree);
      if (!prep.ok()) continue;
      const auto ev_old = EvaluatePreparedTargetRelation(prep, {before3, yaw_before}, rtree, false, false);
      const auto ev_new = EvaluatePreparedTargetRelation(prep, {after3, yaw_after}, rtree, false, false);
      if (!ev_old.ok() || !ev_new.ok()) continue;
      if (pay.kernel_type == RelationKernelType::kAvatarFacing) {
        if (ev_new.relation.encoded_state.size() == 0 || ev_old.relation.encoded_state.size() == 0) continue;
        // The relation's target yaw: the bearing to its mapped look-at point c minus the source's offset (the yaw that
        // reproduces the source's relative orientation). In a similar room c is s times the source's look-at point,
        // computed here from the payload (closed form); otherwise it is the solver's reported witness.
        const double o_src = pay.grounded_relation.encoded_state.size() > 0 ? pay.grounded_relation.encoded_state(0) : 0.0;
        std::optional<Vec2> look;
        if (std::isfinite(s.similarity) && pay.support.has_contact)
          if (const Obj* o = ReferentObj(s.src, pay.referent_shared_id); o && o->yaw == 0.0)
            look = s.similarity * (o->c + pay.support.contact_point_local.head<2>());
        if (!look && have_so)
          for (const auto& g : so.solve.goals)
            if (g.yaw_goal && g.payload_index < so.goal_payload_index.size() &&
                so.goal_payload_index[g.payload_index] == i)
              look = g.witness;
        if (!look) continue;
        // The receiver's remaining offset is its angle to the mapped look-at point; a request closes 1/n of it (the
        // position's rule in angle). n = 1 places the yaw at bearing - source offset (the source aligned
        // within the contact epsilon in degrees), so the closed offset is then the source's own.
        const double bearing = std::atan2(look->y() - after.y(), look->x() - after.x());
        const double e_old = std::remainder(bearing - yaw_before, 2.0 * M_PI);
        const double e_new = std::remainder(bearing - yaw_after, 2.0 * M_PI);
        if (request) {
          ++facing_requests;
          // n <= 1: arrival, the source's offset. Otherwise the receiver's offset turns by 1/n of its remaining turn:
          // the angle it travels in the source's rotation direction s back to 0, on the 2 pi branch nearest the
          // source's own remaining turn, R_now n / (n - 1) from the source's current offset (any
          // branch, so a receiver past its look-at turns back the short way).
          const int sgn = pay.facing_rotation_sign;
          const double n = pay.update_count;
          const auto travel = [](double off, int dir) {
            return std::fmod(std::fmod(-dir * off, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
          };
          double move = -e_old;
          if (sgn != 0) {
            const double r_src = travel(o_src, sgn) * n / (n - 1.0);
            const double d0 = travel(e_old, sgn);
            move = sgn * (d0 + 2.0 * M_PI * std::round((r_src - d0) / (2.0 * M_PI)));
          }
          const double expected = !(n > 1.0) ? o_src : e_old + move / n;
          facing_errs.push_back(Deg(e_new - expected));
          facing_turns.push_back(n > 1.0 ? -move / n : std::numeric_limits<double>::quiet_NaN());
        }
        if (pay.referent_shared_id == target_id) {
          if (out.src_arrival < 0 && Deg(o_src) < kContactEpsilonM) out.src_arrival = static_cast<int>(k);
          if (out.rcv_arrival < 0 && Deg(e_new) < kContactEpsilonM && Deg(o_src) < kContactEpsilonM)
            out.rcv_arrival = static_cast<int>(k);
        }
        continue;
      }
      const double g_old = ev_old.relation.contact_distance_m, g_new = ev_new.relation.contact_distance_m;
      if (request) {
        ++requests;
        errs.push_back(std::fabs(g_new - g_old * (1.0 - 1.0 / pay.update_count)));
      }
      const Vec2 c = ev_new.support.contact_point_world.head<2>();
      // The mapped contact in closed form: in a similar room it is s times the source's contact point (the nearest point
      // of the source's box to the source anchor), computed here from the scene, not by the core.
      if (std::isfinite(s.similarity)) {
        if (const Obj* o = ReferentObj(s.src, pay.referent_shared_id))
          if (const auto cs = BoxNearest(*o, step.pos)) {
            out.worst_closed_form = std::max(out.worst_closed_form, (c - s.similarity * *cs).norm());
            ++out.closed_form_checked;
          }
      }
      if (pay.referent_shared_id == target_id) {
        if (out.src_arrival < 0 && pay.grounded_relation.support_distance_m >= 0.0 &&
            pay.grounded_relation.support_distance_m < kContactEpsilonM)
          out.src_arrival = static_cast<int>(k);
        // The receiver's contact point is the chart image of the source's: the body AT the mapped contact c.
        if ((after - c).norm() < kContactEpsilonM) out.worst_arrival_g = std::max(out.worst_arrival_g, g_new);
        if (out.rcv_arrival < 0 && (after - c).norm() < kContactEpsilonM) out.rcv_arrival = static_cast<int>(k);
      }
    }
    if (s.kernel == Kernel::kSpacing) {
      if (requests == 0) {
        ++out.holds;
        out.worst_hold = std::max(out.worst_hold, (after - before).norm());
      } else if (requests == 1) {
        ++out.single;
        out.worst_ratio = std::max(out.worst_ratio, errs[0]);
      } else {
        ++out.multi;
        for (double e : errs) out.worst_multi = std::max(out.worst_multi, e);
        // Several requests: the answer must be a local minimum of J = sum w (g - d*)^2 over the solve's F, judged by
        // the harness's own oracle (scn::LocalMinimumOfJ; a ridge stop is the basin's answer, as the runner counts).
        if (have_so && !hand) {
          const scn::MinKind mk = scn::LocalMinimumOfJ(r.space, po, so, so.solve.position, [&](const Vec2& q) {
            return r.space->IsFreeUnder(r.avatar, q, so.solve.free_exempt, so.solve.free_ring_exempt);
          });
          if (mk == scn::MinKind::kNotMinimum) ++out.multi_not_minimum;
        }
      }
    } else {
      if (facing_requests == 0) {
        ++out.facing_holds;
        out.worst_facing_hold = std::max(out.worst_facing_hold, Deg(yaw_after - yaw_before));
      } else if (facing_requests == 1) {
        ++out.facing_single;
        out.worst_facing = std::max(out.worst_facing, facing_errs[0]);
        // On arrival the receiver turns the shortest arc (not the source's turn direction): an arrival turn above pi is counted, and must stay 0.
        if (s.turn_driver && !std::isfinite(facing_turns[0]) && std::fabs(p.yaw_turn) > M_PI) ++out.arrival_over_pi;
        if (std::isfinite(facing_turns[0]) && (after - before).norm() == 0.0) {
          if (std::fabs(facing_turns[0]) > M_PI) ++out.turns_over_pi;
          if (s.turn_driver) {
            const double body_step = body_yaw_after - yaw_before;  // unwrapped: the turn the driver realized
            if (body_step * facing_turns[0] < 0.0) ++out.turn_driver_against;
          }
          if (std::fabs(p.yaw_turn - facing_turns[0]) * (180.0 / M_PI) >= kContactEpsilonM) ++out.turn_not_requested;
        }
      } else {
        ++out.facing_multi;
      }
    }
    ++out.updates;
  }
  return out;
}

// ---- the scenario matrix -------------------------------------------------------------------------------------------

std::string Tag(double v) { return "_v" + std::to_string(static_cast<int>(std::lround(v * 10))); }

std::vector<Scenario> Scenarios() {
  std::vector<Scenario> out;
  const Obj table = Box(6.5, 4.5, 0.5, 0.5);
  const Obj sofa = Box(2.5, 8.0, 1.0, 0.4);
  const Obj shelf = Box(8.5, 1.5, 0.3, 0.8);
  const std::vector<Obj> three = {table, sofa, shelf};
  const auto same = [](const RoomSpec& r) { return r; };
  // Receiver rooms for the table scenarios: identical, scaled 1.3 x 0.9, and an L whose notch (x > 5, y > 6) leaves the
  // table in the lower arm.
  struct Layout {
    const char* name;
    RoomSpec rcv;
    Vec2 scale;     // where a source point maps for the displaced start
    double similarity;  // the receiver room is the source's scaled by this (NaN: not a similarity)
  };
  const std::vector<Layout> layouts = {
      {"identical", Rect("10x10", 10, 10, three), {1.0, 1.0}, 1.0},
      {"scaled", Rect("13x9", 13, 9, ScaledAll(three, 1.3, 0.9)), {1.3, 0.9}, kNaN},
      // The same referents in an L-shaped room: the referents' charts are the same, so the mapped contact is the source's.
      {"L", LRoom("L10", 10, 10, 5.0, 6.0, {table, Box(2.5, 8.0, 1.0, 0.4), shelf}), {1.0, 1.0}, 1.0},
      {"uscaled", Rect("12x12", 12, 12, ScaledAll(three, 1.2, 1.2)), {1.2, 1.2}, 1.2},
  };
  (void)same;
  const RoomSpec src3 = Rect("10x10", 10, 10, three);

  // 1. Straight approaches to the table from four sides of the room, three layouts; the receiver is moved 1.5 m
  //    sideways after the join.
  struct Dir {
    const char* name;
    Vec2 start, goal, shift;
  };
  const Dir dirs[] = {{"W", {1.5, 4.5}, {6.0, 4.5}, {0.0, -1.5}},
                      {"S", {6.5, 0.6}, {6.5, 4.0}, {-1.5, 0.0}},
                      {"E", {9.5, 4.5}, {7.0, 4.5}, {0.0, 1.0}},
                      {"diagonal_SW", {2.0, 1.0}, {6.0, 4.0}, {0.0, 1.5}}};
  for (const Layout& l : layouts)
    for (const Dir& d : dirs) {
      if (std::string(l.name) == "L" && std::string(d.name) == "E") continue;  // the L's notch is east of the table
      Scenario sc;
      sc.name = std::string("straight_") + d.name + "_" + l.name;
      sc.src = src3;
      sc.rcv = l.rcv;
      sc.target = 0;
      Line(&sc.path, d.start, d.goal, 1.4);
      Pause(&sc.path, 15);
      sc.rcv_after_join = d.start.cwiseProduct(l.scale) + d.shift;
      sc.similarity = l.similarity;
      out.push_back(sc);
    }
  // 2. Speeds.
  for (double v : {0.7, 2.8}) {
    Scenario sc;
    sc.name = "straight_W_scaled" + Tag(v);
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, v);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(1.5 * 1.3, 4.5 * 0.9 - 1.5);
    out.push_back(sc);
  }
  // 3. A curved approach.
  {
    Scenario sc;
    sc.name = "curved_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    sc.path.push_back({{1.5, 1.0}, 0.0});
    Curve(&sc.path, {2.0, 6.5}, {6.0, 4.7}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(3.0, 2.0);
    out.push_back(sc);
  }
  // 4. The far side: the source goes round the table's south side to its east face; the receiver starts west of it.
  for (const Layout& l : {layouts[0], layouts[1]}) {
    Scenario sc;
    sc.name = std::string("far_side_") + l.name;
    sc.src = src3;
    sc.rcv = l.rcv;
    Line(&sc.path, {1.5, 4.5}, {6.5, 2.8}, 1.4);
    Line(&sc.path, {6.5, 2.8}, {8.0, 3.2}, 1.4);
    Line(&sc.path, {8.0, 3.2}, {7.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(1.2 * l.scale.x(), 4.5 * l.scale.y());
    sc.similarity = l.similarity;
    out.push_back(sc);
  }
  // 5. Stop-and-go.
  {
    Scenario sc;
    sc.name = "stop_and_go_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    Line(&sc.path, {1.5, 4.5}, {3.5, 4.5}, 1.4);
    Pause(&sc.path, 20);
    Line(&sc.path, {3.5, 4.5}, {5.0, 4.5}, 1.4);
    Pause(&sc.path, 10);
    Line(&sc.path, {5.0, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    out.push_back(sc);
  }
  // 6. Receding, then approaching again.
  {
    Scenario sc;
    sc.name = "recede_reapproach_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    Line(&sc.path, {1.5, 4.5}, {4.5, 4.5}, 1.4);
    Line(&sc.path, {4.5, 4.5}, {3.0, 4.0}, 1.4);
    Line(&sc.path, {3.0, 4.0}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    out.push_back(sc);
  }
  // 7. Switching target mid-walk: toward the table, then to the sofa.
  {
    Scenario sc;
    sc.name = "switch_target_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    sc.target = 1;
    Line(&sc.path, {2.0, 2.0}, {4.5, 4.0}, 1.4);
    Line(&sc.path, {4.5, 4.0}, {2.5, 7.6}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(3.5, 1.5);
    out.push_back(sc);
  }
  // 8. Passing by the table (0.5 m south of it) on the way to the shelf's north face.
  {
    Scenario sc;
    sc.name = "pass_by_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    sc.target = 2;
    Line(&sc.path, {1.0, 3.5}, {8.5, 3.5}, 1.4);
    Line(&sc.path, {8.5, 3.5}, {8.5, 2.3}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(1.5, 2.0);
    out.push_back(sc);
  }
  // 9. Two referents approached together, non-identical rooms: the source walks up between two boxes to touch one.
  {
    const std::vector<Obj> pair = {Box(4.0, 5.0, 0.4, 0.4), Box(6.0, 5.0, 0.4, 0.4)};
    for (double v : {0.7, 1.4}) {
      Scenario sc;
      sc.name = "two_together_scaled" + Tag(v);
      sc.src = Rect("10x10", 10, 10, pair);
      sc.rcv = Rect("12x9", 12, 9, ScaledAll(pair, 1.2, 0.9));
      sc.target = 0;
      Line(&sc.path, {5.0, 1.0}, {4.4, 4.6}, v);
      Pause(&sc.path, 15);
      sc.rcv_after_join = Vec2(7.5, 1.5);
      sc.several_requests = true;
      out.push_back(sc);
    }
  }
  // 10. An obstacle in the receiver's way (not a referent: no shared id).
  {
    Scenario sc;
    sc.name = "obstacle_in_way";
    sc.src = src3;
    sc.rcv = Rect("10x10", 10, 10, {table, sofa, shelf, {{4.0, 4.5}, {0.3, 1.5}, 0.0, false, false}});
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 4.5);
    out.push_back(sc);
  }
  // 11. A rotated referent and a non-box (prism) referent.
  {
    Scenario sc;
    sc.name = "rotated_table_scaled";
    sc.src = Rect("10x10", 10, 10, {Box(6.5, 4.5, 0.5, 0.3, 0.5)});
    sc.rcv = Rect("13x9", 13, 9, {Box(6.5 * 1.3, 4.5 * 0.9, 0.65, 0.27, 0.5)});
    Line(&sc.path, {1.5, 4.5}, {6.5 + 0.5 * std::cos(0.5 + M_PI) + 0.0, 4.5 + 0.5 * std::sin(0.5 + M_PI)}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    out.push_back(sc);
  }
  {
    Obj prism = Box(6.5, 4.5, 0.6, 0.6);
    prism.prism = true;
    Scenario sc;
    sc.name = "prism_scaled";
    sc.src = Rect("10x10", 10, 10, {prism});
    sc.rcv = Rect("13x9", 13, 9, {Scaled(prism, 1.3, 0.9)});
    Line(&sc.path, {6.5, 0.8}, {6.5, 3.9}, 1.4);  // to the triangle's base (y = 3.9)
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(6.0, 1.5);
    out.push_back(sc);
  }
  // 12. Jitter around the contact epsilon at the end of an approach.
  {
    Scenario sc;
    sc.name = "jitter_at_contact_scaled";
    sc.src = src3;
    sc.rcv = layouts[1].rcv;
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    for (double off : {0.5, 2.0, 0.5, 5.0, 0.0, 1.5, 0.2, 3.0, 0.0})
      sc.path.push_back({{6.0 - off * kContactEpsilonM, 4.5}, 0.0});
    Pause(&sc.path, 10);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    out.push_back(sc);
  }
  // 13. Uniform and Greedy through the suite (straight, switch target, two together).
  for (const auto pol : {RealizationPolicyType::kUniform, RealizationPolicyType::kGreedy}) {
    const std::string pn = pol == RealizationPolicyType::kUniform ? "_uniform" : "_greedy";
    for (const Scenario& base : std::vector<Scenario>(out)) {
      if (base.name == "straight_W_scaled" || base.name == "switch_target_scaled" ||
          base.name == "two_together_scaled_v14" || base.name == "far_side_scaled") {
        Scenario sc = base;
        sc.name += pn;
        sc.policy = pol;
        out.push_back(sc);
      }
    }
  }
  // 14. Facing, one referent in the room (nothing else in view): the source stands west of the table and turns to face
  //     it; the receiver, moved after the join, must close 1/n of its own offset every update (pending: 7b) and be
  //     aligned on the source's update.
  {
    Scenario sc;
    sc.name = "facing_turn_uscaled";
    sc.rcv_yaw_after_join = -1.0;  // the receiver faces away (south-east), not the table
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = Rect("12x12", 12, 12, {Scaled(table, 1.2, 1.2)});
    sc.similarity = 1.2;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);
    for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (1.0 - i / 30.0)});  // turning to yaw 0: the table
    Pause(&sc.path, 10);
    sc.rcv_after_join = Vec2(2.0, 2.0);
    out.push_back(sc);
  }
  // 14b. Opposing targets: on the source's side two boxes lie in one direction (east, one behind the other), on the
  //      receiver's side they are east and west of it, so the two Facings ask for opposite yaws with equal weights (the
  //      source turns towards both alike): the circular mean ties. The receiver must keep one turn direction.
  {
    Scenario sc;
    sc.name = "facing_opposing_tie";
    sc.src = Rect("10x10", 10, 10, {Box(4.0, 5.0, 0.3, 0.3), Box(7.0, 5.0, 0.3, 0.3)});
    sc.rcv = Rect("10x10", 10, 10, {Box(8.0, 5.0, 0.3, 0.3), Box(2.0, 5.0, 0.3, 0.3)});
    sc.kernel = Kernel::kFacing;
    const Vec2 at(1.0, 5.0);  // free on both sides (the receiver joins here before it is moved between its boxes)
    for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (1.0 - i / 30.0)});
    Pause(&sc.path, 10);
    sc.rcv_after_join = Vec2(5.0, 5.0);
    sc.opposing_tie = true;
    out.push_back(sc);
  }
  // 14c. Facing identity: identical rooms, the receiver starts with the source's pose and is never moved: its yaw is the
  //      source's on every update while the source turns to face the table (the ratio in angle keeps the copy).
  {
    Scenario sc;
    sc.name = "facing_identity";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = sc.src;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);
    for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (1.0 - i / 30.0)});
    Pause(&sc.path, 10);
    sc.identity = true;
    sc.similarity = 1.0;
    out.push_back(sc);
  }
  // 14d. The source turns AWAY from its look-at (its offset opens: v1 counts the long way round) and spins in place: a
  //      receiver that starts as its copy turns the same way and stays its copy.
  for (const bool spin : {false, true}) {
    Scenario sc;
    sc.name = spin ? "facing_identity_spin" : "facing_identity_away";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = sc.src;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);
    if (spin) {
      for (int i = 0; i <= 90; ++i) sc.path.push_back({at, std::remainder(0.2 + 0.1 * i, 2.0 * M_PI)});
    } else {
      for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (i / 30.0)});  // from facing the table, turning away
      for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (1.0 - i / 30.0)});  // and back
    }
    Pause(&sc.path, 10);
    sc.identity = true;
    sc.similarity = 1.0;
    sc.look_switch = spin;
    out.push_back(sc);
  }
  // 14d'. The source, facing away from the table, turns THROUGH the antipode of its look-at and back: the look-at
  //       switches between the table's two silhouette vertices each time (at the heading where their deviations tie),
  //       the offset opens on one side of the switch and closes on the other, and the receiver stays the copy. It then
  //       turns to face the table.
  {
    Scenario sc;
    sc.name = "facing_identity_antipode";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = sc.src;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);  // the table's centre is at bearing 0: its antipode is the heading pi
    for (int i = 0; i <= 12; ++i) sc.path.push_back({at, std::remainder(M_PI - 0.65 + 0.1 * i, 2.0 * M_PI)});
    for (int i = 0; i <= 12; ++i) sc.path.push_back({at, std::remainder(M_PI + 0.55 - 0.1 * i, 2.0 * M_PI)});
    // then turns to face the table (the gaze hits it within about 0.12 rad of bearing 0): the source arrives.
    for (int i = 1; i <= 24; ++i) sc.path.push_back({at, M_PI - 0.65 - 0.1 * i});
    Pause(&sc.path, 10);
    sc.identity = true;
    sc.similarity = 1.0;
    sc.look_switch = true;
    out.push_back(sc);
  }
  // 14d''. The receiver's body turns at most 2.5 degrees per update (the harness driver's 150 deg/s at 60 Hz) while the
  //        source spins in place at 3.82 degrees per update (229 deg/s, scenario_cli turn_in_place at 2.8 m/s) for one
  //        and a half turns: the body lags, the proposal steps from it, and the body must never turn opposite to the
  //        source (an opening source with a lagging receiver still on the other side of its look-at must not turn
  //        the receiver backwards, 180 degrees).
  {
    Scenario sc;
    sc.name = "facing_turn_in_place_capped";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = sc.src;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);
    const double rate = 229.0 / 60.0 * M_PI / 180.0;
    for (int i = 0; i <= 141; ++i) sc.path.push_back({at, std::remainder(0.2 + rate * i, 2.0 * M_PI)});
    Pause(&sc.path, 60);
    sc.yaw_rate_cap = 2.5 * M_PI / 180.0;
    sc.look_switch = true;
    out.push_back(sc);
  }
  // 14d'''. ABI 4.12: a requested turn above pi. Identical rooms, Facing only, one referent. After the join the receiver
  //         faces 0.5 rad while the source faces -1.0 and turns to -0.45 (its look-at is the table's corner at bearing
  //         -0.124: offset 0.876 -> 0.326, n = 1.59): the receiver's offset (-0.624) turns the source's way (s = -1) the
  //         long way round, 2 pi - 0.624, by 1/n: a requested turn of about +3.56 rad. A driver that turns by the proposal's signed turn (yaw_turn) turns
  //         counter-clockwise; one that took the short arc to the absolute yaw (or a yaw_turn wrapped to it) would turn
  //         clockwise, and turn_driver_against counts it.
  {
    Scenario sc;
    sc.name = "facing_turn_over_pi";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = sc.src;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 4.5);  // the table's centre is at bearing 0
    sc.path = {{at, -1.0}, {at, -0.45}, {at, 0.0}};
    Pause(&sc.path, 10);
    sc.rcv_after_join = at;
    sc.rcv_yaw_after_join = 0.5;
    sc.yaw_rate_cap = 1.0;  // the body lags: it arrives past its target and turns back the short way
    sc.turn_driver = true;
    out.push_back(sc);
  }
  // 14e. Copies in a uniformly scaled room (closed form: receiver = 1.2 source, the same yaw), Spacing with three
  //      referents and Facing with three referents in view (several Facing requests on one update).
  {
    Scenario sc;
    sc.name = "identity_three_uscaled";
    sc.src = src3;
    sc.rcv = layouts[3].rcv;
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.identity = true;
    sc.similarity = 1.2;
    out.push_back(sc);
  }
  {
    Scenario sc;
    sc.name = "facing_identity_three_uscaled";
    sc.src = src3;
    sc.rcv = layouts[3].rcv;
    sc.kernel = Kernel::kFacing;
    const Vec2 at(4.5, 5.0);
    for (int i = 0; i <= 40; ++i) sc.path.push_back({at, -0.6 + 1.2 * (i / 40.0)});
    for (int i = 0; i <= 40; ++i) sc.path.push_back({at, 0.6 - 1.2 * (i / 40.0)});
    Pause(&sc.path, 10);
    sc.identity = true;
    sc.similarity = 1.2;
    out.push_back(sc);
  }
  // 14f. A tie of the circular mean while the yaw turns: two Facings whose targets are opposite on the receiver's side
  //      at every update (on the source's side the two boxes lie on one ray, the far one twice the near one's size and
  //      distance, so their offsets are equal; on the receiver's side they lie east and west of it). The yaw turns, and
  //      its direction must not flip while the tie holds.
  {
    Scenario sc;
    sc.name = "facing_tie_turning";
    sc.src = Rect("14x10", 14, 10, {Box(5.0, 5.0, 0.3, 0.3), Box(8.0, 5.0, 0.6, 0.6)});
    // The west box is turned half a turn, so its mapped look-at (the same corner of its own frame) lies point-symmetric
    // to the east one's about the receiver: the two targets are exactly opposite.
    sc.rcv = Rect("14x10", 14, 10, {Box(10.0, 5.0, 0.3, 0.3), Box(1.0, 5.0, 0.6, 0.6, M_PI)});
    sc.kernel = Kernel::kFacing;
    const Vec2 at(2.0, 5.0);
    for (int i = 0; i <= 30; ++i) sc.path.push_back({at, 1.2 * (1.0 - i / 30.0)});
    Pause(&sc.path, 10);
    sc.rcv_after_join = Vec2(7.0, 5.0);
    sc.rcv_yaw_after_join = 0.7;
    sc.opposing_tie = true;
    out.push_back(sc);
  }
  // 15. Identity: identical rooms, the receiver starts where the source does and is never moved: receiver = source on
  //     every update, one referent and three.
  for (const auto& objs : {std::vector<Obj>{table}, three}) {
    Scenario sc;
    sc.name = objs.size() == 1 ? "identity_one" : "identity_three";
    sc.src = Rect("10x10", 10, 10, objs);
    sc.rcv = sc.src;
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.identity = true;
    sc.similarity = 1.0;
    out.push_back(sc);
  }
  // 16. The receiver joins mid-walk (its space connects on update 40), then is moved.
  {
    Scenario sc;
    sc.name = "join_mid_walk_uscaled";
    sc.src = src3;
    sc.rcv = layouts[3].rcv;
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.join_at = 40;
    sc.rcv_after_join = Vec2(3.0, 3.0);
    sc.similarity = 1.2;
    out.push_back(sc);
  }
  // 17. The source teleports mid-run (a declared source exception) and approaches the table from the other side.
  {
    Scenario sc;
    sc.name = "source_teleport_uscaled";
    sc.src = src3;
    sc.rcv = layouts[3].rcv;
    Line(&sc.path, {1.5, 4.5}, {3.5, 4.5}, 1.4);
    sc.teleport_at = static_cast<int>(sc.path.size());
    sc.path.push_back({{9.5, 4.5}, 0.0});  // the jump, declared
    Line(&sc.path, {9.5, 4.5}, {7.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    sc.similarity = 1.2;
    out.push_back(sc);
  }
  // 18. F blocks the mapped contact: an obstacle (not a referent) covers the receiver table's west face, where the
  //     source's contact maps. The receiver can never arrive; it stays in F.
  {
    Scenario sc;
    sc.name = "blocked_contact";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = Rect("10x10", 10, 10, {table, {{5.8, 4.5}, {0.3, 0.8}, 0.0, false, false}});
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.rcv_after_join = Vec2(2.0, 2.5);
    sc.blocked = true;
    out.push_back(sc);
  }
  // 19. A non-root anchor: Spacing bound to the hand (1 m up), approaching the table; the receiver's hand proposal is
  //     realized at once by the driver.
  {
    Scenario sc;
    sc.name = "hand_anchor_uscaled";
    sc.src = Rect("10x10", 10, 10, {table});
    sc.rcv = Rect("12x12", 12, 12, {Scaled(table, 1.2, 1.2)});
    sc.anchor = "hand";
    Line(&sc.path, {1.5, 4.5}, {6.0, 4.5}, 1.4);
    Pause(&sc.path, 15);
    sc.similarity = 1.2;
    out.push_back(sc);
  }
  return out;
}

std::string ParamName(const ::testing::TestParamInfo<Scenario>& info) { return info.param.name; }
void PrintTo(const Scenario& s, std::ostream* os) { *os << s.name; }

class CoreIntegration : public ::testing::TestWithParam<Scenario> {};

TEST_P(CoreIntegration, ReceiverFollowsTheSourceContactEveryUpdate) {
  const Scenario& s = GetParam();
  const Outcome o = RunScenario(s);
  EXPECT_EQ(o.missing_target, 0) << "every update carries a target";
  EXPECT_EQ(o.unexpected_absolute, 0) << "only the join and a declared teleport are absolute";
  EXPECT_EQ(o.outside_f, 0) << "the body stays in the solve's F";
  ASSERT_GE(o.src_arrival, 0) << "the source never arrives";
  // Every scenario, the opposing ties included (they return early below).
  EXPECT_EQ(o.turn_not_congruent, 0) << "ABI 4.12: sampled yaw + signed turn = yaw (mod 2 pi)";
  if (s.opposing_tie) {
    EXPECT_EQ(o.turn_flips, 0) << "a tie keeps the receiver's turn direction: it never alternates";
    EXPECT_GT(o.facing_single + o.facing_multi, 0);
    EXPECT_GT(o.facing_ties, 1) << "the scenario holds a tie over several updates";
    return;  // no yaw serves both opposite targets: there is no arrival to compare
  }
  if (s.blocked) {
    EXPECT_LT(o.rcv_arrival, 0) << "F does not contain the mapped contact: the receiver cannot be on it";
  } else {
    ASSERT_GE(o.rcv_arrival, 0) << "the receiver never reaches the chart image of the source's contact point";
    EXPECT_EQ(o.rcv_arrival - o.src_arrival, 0) << "source update " << o.src_arrival << ", receiver " << o.rcv_arrival;
  }
  if (s.kernel == Kernel::kSpacing) {
    // With F blocking the mapped contact the ratio's target can lie outside F: the answer is the nearest point of F,
    // not d*. The ratio is then reported, not asserted.
    if (!s.blocked) EXPECT_LT(o.worst_ratio, kContactEpsilonM) << "g_new = g_old (1 - 1/n) on every one-request update";
    EXPECT_LT(o.worst_hold, kContactEpsilonM) << "no request: the body holds";
    EXPECT_GT(o.single + o.multi, 0) << "the scenario exercises the ratio";
    EXPECT_EQ(o.multi_not_minimum, 0) << "several requests: the answer is a local minimum of J";
  } else {
    if (!kFacingYawRatioPending)
      EXPECT_LT(o.worst_facing, kContactEpsilonM) << "the receiver's offset closes 1/n of itself (degrees)";
    EXPECT_LT(o.worst_facing_hold, kContactEpsilonM) << "no Facing request: the yaw holds (degrees)";
    EXPECT_GT(o.facing_single, 0) << "the scenario exercises the Facing ratio";
  }
  if (std::isfinite(s.similarity) && s.kernel == Kernel::kSpacing) {
    EXPECT_GT(o.closed_form_checked, 0);
    EXPECT_LT(o.worst_closed_form, kContactEpsilonM) << "the mapped contact is s times the source's contact point";
  }
  if (s.identity) EXPECT_LT(o.worst_identity, kContactEpsilonM) << "identical rooms: the receiver is the source";
  if (s.identity && s.kernel == Kernel::kFacing)
    EXPECT_LT(o.worst_identity_yaw_deg, kContactEpsilonM) << "identical rooms: the receiver's yaw is the source's";
  if (s.look_switch) EXPECT_GT(o.look_switches, 0) << "the path turns the source's look-at through a vertex switch";
  if (s.turn_driver) {
    EXPECT_EQ(o.turns_over_pi, 0) << "a body past its look-at turns back the short way, never above pi";
    EXPECT_EQ(o.turn_driver_against, 0) << "a driver turning by yaw_turn turns the requested way, also above pi";
    EXPECT_EQ(o.arrival_over_pi, 0) << "an arrival turns the shortest arc, never more than pi";
  }
  if (s.kernel == Kernel::kFacing)
    EXPECT_EQ(o.turn_not_requested, 0) << "ABI 4.12: the signed turn is the requested one (its sign the chosen direction)";
  if (s.several_requests) {
    // Greedy is one-hot (v1 NearestFirst): it never requests two relations at once.
    if (s.policy != RealizationPolicyType::kGreedy) EXPECT_GT(o.multi, 0) << "two referents are requested together";
    EXPECT_EQ(o.not_converged, 0);
  }
}

INSTANTIATE_TEST_SUITE_P(CrossSpace, CoreIntegration, ::testing::ValuesIn(Scenarios()), ParamName);

// The table of every scenario (printed; the per-scenario test above is the check).
// The turn direction a tie keeps is the direction the receiver last TURNED, its requested turn
// (yaw_turn), not the shortest arc to its last yaw. After a long-way turn the two point opposite ways, and the receiver
// took the middle behind it: a half turn back on the next stepping tie, and again on the one after.
// The facing_tie_turning rooms (two referents whose targets are opposite on the receiver's side: a tie at every
// update).
// The updates are hand-fed: a real update of the source, then that update cut to one referent and made a long-way
// stepping request (as ASignedTurnAbovePiKeepsItsDirection: n = 1.5, the source opening the long way round), then the
// real tie again.
TEST(CoreIntegrationTurnSign, ATieAfterALongWayTurnKeepsTheTurnRequested) {
  auto ctx = LocalContext::Create();
  Space* src = ctx->AddSpace();
  Build(src, Rect("14x10", 14, 10, {Box(5.0, 5.0, 0.3, 0.3), Box(8.0, 5.0, 0.6, 0.6)}));
  const std::vector<SubjectAnchor> anchors{SubjectAnchor("root", Eigen::Vector3d(0, 0, 0))};
  const auto actor = src->CreateActor(anchors);
  src->GetActor(actor)->SetElementId(1);
  src->Bind(actor, "root", RelationKernelType::kAvatarFacing);
  src->SetHub(ctx->GetHub());
  src->SetMovementGate(0.0, 0.0);
  Space* rcv = ctx->AddSpace();
  Build(rcv, Rect("14x10", 14, 10, {Box(10.0, 5.0, 0.3, 0.3), Box(1.0, 5.0, 0.6, 0.6, M_PI)}));
  rcv->EnableObservabilityCapture(true);
  const auto avatar = rcv->CreateActor(anchors);
  rcv->GetActor(avatar)->SetElementId(1);
  rcv->SetHub(ctx->GetHub());
  IdealDriver driver(rcv, avatar, Vec2(2.0, 5.0), 1.2);
  // The source turns towards its look-ats (both on one ray, equal offsets): the join, then continuations.
  for (int i = 0; i <= 3; ++i) src->UpdateActorState(actor, Vec2(2.0, 5.0), 1.2 - 0.04 * i);
  driver.Publish(7.0, 5.0, 0.7);  // between the two counterparts: their targets are opposite
  const std::vector<RemoteRelationUpdate>& got = rcv->GetReceivedUpdates();
  ASSERT_FALSE(got.empty());
  const RemoteRelationUpdate tie = got.back();
  ASSERT_EQ(tie.payloads.size(), 2u);
  for (const auto& pl : tie.payloads) ASSERT_GT(pl.update_count, 1.0) << "the tie is a stepping (pursuing) one";
  int64_t seq = tie.sequence;
  struct Answer {
    ReceiverProposal p;
    bool tie = false;
  };
  const auto feed = [&](RemoteRelationUpdate u) {
    u.sequence = ++seq;
    rcv->OnRemoteUpdate(u);
    Answer a;
    a.p = driver.proposal();
    Space::SolverObservation so;
    a.tie = rcv->CopySolverObservation(avatar, &so) && so.valid && so.solve.facing_tie;
    return a;
  };
  feed(tie);
  RemoteRelationUpdate long_way = tie;
  long_way.payloads.resize(1);
  long_way.payloads[0].update_count = 1.5;
  long_way.payloads[0].facing_rotation_sign = +1;
  long_way.payloads[0].grounded_relation.encoded_state(0) = -1.67;
  const double yaw_before_long = driver.yaw();
  const Answer turned = feed(long_way);
  ASSERT_TRUE(turned.p.yaw_specified());
  ASSERT_FALSE(turned.p.IsAbsolute());
  ASSERT_LT(turned.p.yaw_turn, -M_PI) << "a long-way turn";
  ASSERT_GT(std::remainder(turned.p.yaw - yaw_before_long, 2.0 * M_PI), 0.0)
      << "its shortest arc points the other way";
  for (int k = 0; k < 2; ++k) {
    const Answer next = feed(tie);
    ASSERT_TRUE(next.p.yaw_specified());
    ASSERT_TRUE(next.tie) << "update " << k;
    EXPECT_LT(next.p.yaw_turn, 0.0) << "the tie keeps the direction the receiver turned, update " << k;
    EXPECT_LT(std::fabs(next.p.yaw_turn), M_PI / 2.0) << "no half turn back, update " << k;
  }
}

TEST(CoreIntegrationSummary, ArrivalAndRatioTable) {
  int at_zero = 0, never = 0, total = 0;
  std::fprintf(stderr,
               "[Arrival] %-30s %5s %5s %5s | %4s %4s %4s | %9s %9s %9s %9s | %3s %3s %3s | %9s %9s | %2s\n",
               "scenario", "src", "rcv", "diff", "1req", "2req", "hold", "ratio", "multi", "hold", "g_at_c", "nm", "nc",
               "abs", "closed", "identity", "nF");
  for (const Scenario& s : Scenarios()) {
    const Outcome o = RunScenario(s);
    ++total;
    const bool both = o.src_arrival >= 0 && o.rcv_arrival >= 0;
    if (!both) ++never;
    if (both && o.rcv_arrival == o.src_arrival) ++at_zero;
    char diff[16];
    if (both) std::snprintf(diff, sizeof diff, "%+d", o.rcv_arrival - o.src_arrival);
    else std::snprintf(diff, sizeof diff, "NEVER");
    const bool f = s.kernel == Kernel::kFacing;
    // Spacing: metres; Facing: degrees (1req/2req/hold count Facing updates).
    std::fprintf(stderr,
                 "[Arrival] %-30s %5d %5d %5s | %4d %4d %4d | %9.2e %9.2e %9.2e %9.2e | %3d %3d %3d | %9.2e %9.2e | "
                 "%2d%s\n",
                 s.name.c_str(), o.src_arrival, o.rcv_arrival, diff, f ? o.facing_single : o.single,
                 f ? o.facing_multi : o.multi, f ? o.facing_holds : o.holds, f ? o.worst_facing : o.worst_ratio,
                 o.worst_multi, f ? o.worst_facing_hold : o.worst_hold, o.worst_arrival_g, o.multi_not_minimum,
                 o.not_converged, o.unexpected_absolute, o.worst_closed_form, o.worst_identity, o.outside_f,
                 f ? ("  (facing, degrees; ties " + std::to_string(o.facing_ties) + ", turn flips " +
                      std::to_string(o.turn_flips) + ", look-at switches " + std::to_string(o.look_switches) + ")")
                         .c_str()
                   : "");
  }
  std::fprintf(stderr, "[Arrival] %d scenarios: %d at 0, %d with a side that never arrives\n", total, at_zero, never);
  SUCCEED();
}

// ---- later modules (the inside-object rule) ------------------------------------------------------------------------
TEST(CoreIntegrationLater, DISABLED_SitDownOnTheSameUpdate) {
  GTEST_SKIP() << "Inside-object rule: the source sits on a chair; the receiver's pelvis must be seated on "
                  "the same update. Written when the inside-object relations replace the inner pose level.";
}
TEST(CoreIntegrationLater, DISABLED_StandUpOnTheSameUpdates) {
  GTEST_SKIP() << "Inside-object rule: the vertical stand-up must follow the same updates on both sides.";
}
TEST(CoreIntegrationLater, DISABLED_ExitOnTheSameSide) {
  GTEST_SKIP() << "Inside-object rule: leaving a chair to its left/right/front exits on the same side (the outer support).";
}

}  // namespace
}  // namespace spatialblend
