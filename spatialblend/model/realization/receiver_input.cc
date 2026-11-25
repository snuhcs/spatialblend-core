// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/realization/receiver_input.h"

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace spatialblend {
namespace {
bool RigidPose(const Eigen::Matrix4d& pose) {
  if (!pose.allFinite() ||
      (pose.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9)
    return false;
  const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
  return (r.transpose() * r - Eigen::Matrix3d::Identity()).norm() <= 1e-6 &&
         std::abs(r.determinant() - 1.0) <= 1e-6;
}
}  // namespace

int64_t ReceiverInputNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool ReceiverPoseYaw(const Eigen::Matrix4d& pose, double* yaw) {
  if (!yaw || pose.block<2, 1>(0, 0).squaredNorm() < 1e-12) return false;
  *yaw = std::atan2(pose(1, 0), pose(0, 0));
  return std::isfinite(*yaw);
}

bool ReceiverInputState::AdvanceRevision() {
  if (poisoned_ || stamp_.revision == std::numeric_limits<uint64_t>::max()) {
    poisoned_ = true;
    accepting_ = has_frame_ = false;
    return false;
  }
  ++stamp_.revision;
  return true;
}

bool ReceiverInputState::Begin(const std::string& root_anchor,
                               const std::vector<std::string>& layout,
                               uint64_t rig_revision,
                               uint64_t calibration_revision, uint64_t* epoch) {
  if (!epoch || root_anchor.empty() || layout.empty() ||
      layout.size() > kMaxAnchors || rig_revision == 0 ||
      calibration_revision == 0 || poisoned_)
    return false;
  std::unordered_map<std::string, ReceiverAnchorInput> next;
  for (const auto& name : layout) {
    if (name.empty() || name.size() > kMaxNameBytes ||
        name.find('\0') != std::string::npos ||
        !next.emplace(name, ReceiverAnchorInput{name}).second)
      return false;
  }
  if (!next.count(root_anchor)) return false;
  if (stamp_.epoch == std::numeric_limits<uint64_t>::max()) {
    poisoned_ = true;
    accepting_ = has_frame_ = false;
    return false;
  }
  if (!AdvanceRevision()) return false;
  ++stamp_.epoch;
  stamp_.sequence = 0;
  stamp_.rig_revision = rig_revision;
  stamp_.calibration_revision = calibration_revision;
  stamp_.observed_at_ns = stamp_.valid_until_ns = 0;
  root_anchor_ = root_anchor;
  anchors_ = std::move(next);
  accepting_ = true;
  has_frame_ = false;
  *epoch = stamp_.epoch;
  return true;
}

bool ReceiverInputState::Commit(uint64_t epoch, uint64_t sequence,
                                int64_t observed_at_ns, int64_t valid_until_ns,
                                const std::vector<ReceiverAnchorInput>& anchors,
                                int64_t now_ns) {
  if (!accepting_ || poisoned_ || epoch != stamp_.epoch || sequence == 0 ||
      sequence <= stamp_.sequence || anchors.size() != anchors_.size() ||
      observed_at_ns <= 0 || observed_at_ns > now_ns ||
      observed_at_ns < stamp_.observed_at_ns || valid_until_ns <= now_ns ||
      valid_until_ns <= observed_at_ns ||
      valid_until_ns - observed_at_ns > kMaxAgeNs)
    return false;
  std::unordered_map<std::string, ReceiverAnchorInput> next;
  for (const auto& anchor : anchors) {
    if (!anchors_.count(anchor.name) || !RigidPose(anchor.world_pose) ||
        !next.emplace(anchor.name, anchor).second)
      return false;
  }
  double root_yaw;
  const auto& root = next.at(root_anchor_);
  if (root.available && !ReceiverPoseYaw(root.world_pose, &root_yaw))
    return false;
  if (!AdvanceRevision()) return false;
  anchors_ = std::move(next);
  stamp_.sequence = sequence;
  stamp_.observed_at_ns = observed_at_ns;
  stamp_.valid_until_ns = valid_until_ns;
  has_frame_ = true;
  return true;
}

bool ReceiverInputState::Clear(uint64_t epoch) {
  if (!accepting_ || epoch != stamp_.epoch || poisoned_) return false;
  if (has_frame_ && !AdvanceRevision()) return false;
  has_frame_ = false;
  return true;
}

void ReceiverInputState::Invalidate() {
  if (accepting_ || has_frame_) AdvanceRevision();
  accepting_ = has_frame_ = false;
}

ReceiverInputStamp ReceiverInputState::Stamp(int64_t now_ns) const {
  auto stamp = stamp_;
  const auto root = anchors_.find(root_anchor_);
  stamp.known = accepting_ && has_frame_ && !poisoned_ &&
                root != anchors_.end() && root->second.available &&
                now_ns >= stamp.observed_at_ns && now_ns < stamp.valid_until_ns;
  return stamp;
}

bool ReceiverInputState::CopyPose(const std::string& anchor,
                                  Eigen::Matrix4d* out) const {
  const auto it = anchors_.find(anchor);
  if (!out || !accepting_ || !has_frame_ || poisoned_ || it == anchors_.end() ||
      !it->second.available)
    return false;
  *out = it->second.world_pose;
  return true;
}

}  // namespace spatialblend
