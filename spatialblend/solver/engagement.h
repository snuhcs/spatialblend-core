// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/solver/engagement.h
#ifndef SPATIALBLEND_SPACE_ENGAGEMENT_H_
#define SPATIALBLEND_SPACE_ENGAGEMENT_H_

#include <cmath>
#include <limits>

#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {

// Contact is a metric distance below kContactEpsilonM (remote_update.h), the one contact epsilon of the core: v1's
// kEpsilon, the same that decides the count of 1 at contact (SourceUpdateCount). R == 1 of a sender that predates the
// metric distance is read with the same epsilon on R.

// Engagement is a GEOMETRIC fact about a relation: seated means the
// relation-bearing anchor touches the sittable surface. There
// is no radius, no stillness, no dwell and no hysteresis; a threshold on a metric
// distance would also change F (the free space) with the threshold, and there is
// one F.
//
//   Spacing (root): the source root is inside the referent's footprint
//                   (support.has_interior), or touches its edge (support
//                   distance 0, R == 1).
//   AffordancePose: the source anchor touches the support surface: it lies in the
//                   support part's footprint and on or below its plane. The kernel
//                   decides this where the part's frame is exact
//                   (GroundedRelation::ap_*); see IsAffordanceContact.
//   Facing:         never engages.

// Source root is ON the referent (its footprint may host the root target).
inline bool IsOnReferent(const RelationPayload& p) {
  return p.kernel_type == RelationKernelType::kAvatarSpacing &&
         p.support.has_interior;
}

// Metric support distance of a payload, or +inf when unknown.
inline double SupportDistanceM(const RelationPayload& p) {
  const double d = p.grounded_relation.support_distance_m;
  if (std::isfinite(d) && d >= 0.0) return d;
  return std::numeric_limits<double>::infinity();
}

// Spacing contact: R == 1, the support-relative condition holds exactly (SB
// 5_system.tex:99-100). The first update in which this holds starts the
// interaction's latency timer (space/interaction_latency.h).
inline bool IsSpacingContact(const RelationPayload& p) {
  if (p.kernel_type != RelationKernelType::kAvatarSpacing) return false;
  if (SupportDistanceM(p) < kContactEpsilonM) return true;
  // No metric distance (a sender that predates it): R == 1 up to the same epsilon.
  return !std::isfinite(SupportDistanceM(p)) && std::isfinite(p.grounded_relation.proximity) &&
         p.grounded_relation.proximity > 1.0 - kContactEpsilonM;
}

// A Spacing contact that is an INTERACTION of the source, which starts a latency timer (L1) and is
// what the target's ARRIVED is read against (L2): the root is at R == 1 on a referent. The
// container the source is inside is its scope, not an interaction.
inline bool IsTimedSpacingContact(const RelationPayload& p) {
  return IsSpacingContact(p) && !p.support.container_self;
}

// The anchor touches the support surface of an AffordancePose relation: its
// in-plane offset from the support point is zero (the anchor is inside the
// part's footprint, so the nearest support point is the anchor's own projection)
// and its height above the plane is not positive (on the surface, or having
// crossed it). A pure function of the anchor and the surface; no tolerance.
inline bool IsAffordanceContact(const GroundedRelation& g) {
  if (std::isfinite(g.ap_offset_u_m) && std::isfinite(g.ap_offset_v_m) &&
      std::isfinite(g.ap_height_m))
    return std::hypot(g.ap_offset_u_m, g.ap_offset_v_m) < kContactEpsilonM && g.ap_height_m < kContactEpsilonM;
  // A sender that predates the pose terms reports only the metric distance from
  // the anchor to its support: touching is that distance being 0. With no metric
  // distance either, R = 1 / (1 + d) is 1 exactly at contact.
  if (std::isfinite(g.support_distance_m) && g.support_distance_m >= 0.0)
    return g.support_distance_m < kContactEpsilonM;
  return std::isfinite(g.proximity) && g.proximity > 1.0 - kContactEpsilonM;
}

inline bool IsEngaged(const RelationPayload& p) {
  switch (p.kernel_type) {
    case RelationKernelType::kAvatarSpacing:
      return IsOnReferent(p) || SupportDistanceM(p) < kContactEpsilonM;
    case RelationKernelType::kAvatarAffordancePose:
      return IsAffordanceContact(p.grounded_relation);
    case RelationKernelType::kAvatarFacing:
      return false;
  }
  return false;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_ENGAGEMENT_H_
