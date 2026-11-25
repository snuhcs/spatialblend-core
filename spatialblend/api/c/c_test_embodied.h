// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Test support for the C ABI tests: the test plays the application, which owns
// each receiver avatar's embodied poses and publishes them through the per-anchor
// sampling ports (core keeps no pose and proposes nothing until one exists).
#ifndef SPATIALBLEND_C_C_TEST_EMBODIED_H_
#define SPATIALBLEND_C_C_TEST_EMBODIED_H_

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"

namespace spatialblend_c_test {

// Publish the pose of the ROOT anchor (anchor id 0) with the next sequence of
// that actor's root port.
inline int32_t PublishStartPose(sb_space_t space, sb_actor_t actor, double x = 0.0,
                                double y = 0.0, double yaw = 0.0) {
  static std::map<std::pair<sb_space_t, sb_actor_t>, uint64_t> sequence;
  sb_anchor_pose_t pose{};
  pose.struct_size = sizeof(pose);
  pose.x = x;
  pose.y = y;
  pose.orientation.struct_size = sizeof(sb_orientation_t);
  pose.orientation.yaw = yaw;
  pose.epoch = 1;
  return sb_publish_anchor_pose(space, actor, 0, &pose, 0.0,
                                ++sequence[{space, actor}]);
}

// One proposal set read through the ABI 4 API: the set header and the root
// anchor's proposal flattened (the root is anchor id 0 and always first), plus
// every anchor.
struct ProposalView {
  int32_t rc = -1;          // sb_read_proposals' return
  sb_proposal_set_t set{};
  sb_anchor_proposal_t root{};
  std::vector<sb_anchor_proposal_t> anchors;  // every anchor, root first

  // The root anchor's fields, named as ABI 3's sb_proposal_t named them.
  int32_t status = SB_PROPOSAL_NONE;
  uint64_t revision = 0;
  int64_t source_sequence = 0;
  int32_t sample_valid = 0;
  uint64_t sample_sequence = 0;
  uint64_t sample_epoch = 0;
  double sample_age_s = 0.0;
  uint32_t cause = 0;
  double x = 0.0, y = 0.0, z = 0.0, yaw = 0.0;
  uint32_t flags = 0;
  int32_t orientation_type = 0;
  double pitch = 0.0, roll = 0.0;
  double err_spacing_target = -1.0, err_spacing_sample = -1.0;
  double err_facing_target = -1.0, err_facing_sample = -1.0;
  double err_scale_m = 0.0;

  bool position_unspecified() const {
    return (flags & SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED) != 0;
  }
  bool engaged() const { return (flags & SB_ANCHOR_PROPOSAL_FLAG_ENGAGED) != 0; }
  bool transfer() const { return (flags & SB_ANCHOR_PROPOSAL_FLAG_TRANSFER) != 0; }
  const sb_anchor_proposal_t* Anchor(uint32_t id) const {
    for (const auto& a : anchors)
      if (a.anchor_id == id) return &a;
    return nullptr;
  }
};

inline ProposalView ReadProposals(sb_space_t space, sb_actor_t actor,
                                  int32_t cap = 16) {
  ProposalView v;
  std::vector<sb_anchor_proposal_t> buf(static_cast<size_t>(cap));
  for (auto& a : buf) a.struct_size = sizeof(sb_anchor_proposal_t);
  v.set.struct_size = sizeof(sb_proposal_set_t);
  v.rc = sb_read_proposals(space, actor, &v.set, buf.data(), cap);
  if (v.rc != 0) return v;
  buf.resize(v.set.anchor_count);
  v.anchors = buf;
  v.revision = v.set.revision;
  v.source_sequence = v.set.source_sequence;
  v.sample_valid = v.set.sample_valid;
  v.sample_sequence = v.set.sample_sequence;
  v.sample_epoch = v.set.sample_epoch;
  v.sample_age_s = v.set.sample_age_s;
  if (!buf.empty()) {
    v.root = buf[0];
    v.status = v.root.status;
    v.cause = v.root.cause;
    v.x = v.root.x;
    v.y = v.root.y;
    v.z = v.root.z;
    v.yaw = v.root.orientation.yaw;
    v.flags = v.root.flags;
    v.orientation_type = v.root.orientation.type;
    v.pitch = v.root.orientation.pitch;
    v.roll = v.root.orientation.roll;
    v.err_spacing_target = v.root.err_spacing_target;
    v.err_spacing_sample = v.root.err_spacing_sample;
    v.err_facing_target = v.root.err_facing_target;
    v.err_facing_sample = v.root.err_facing_sample;
    v.err_scale_m = v.root.err_scale_m;
  }
  return v;
}

}  // namespace spatialblend_c_test

#endif  // SPATIALBLEND_C_C_TEST_EMBODIED_H_
