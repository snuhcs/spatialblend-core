// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_SPACE_RECEIVER_INPUT_H_
#define SPATIALBLEND_SPACE_RECEIVER_INPUT_H_

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace spatialblend {

// External geometric input, not Actor display state, source observation, a
// complete skeleton, or proof of measurement/calibration/contact. Poses are
// rigid receiver-world Z-up transforms. Core uses their planar position/yaw
// for locomotion/facing and retains XYZ for independent anchor positions.
struct ReceiverAnchorInput {
  std::string name;
  bool available = false;
  Eigen::Matrix4d world_pose = Eigen::Matrix4d::Identity();
};

struct ReceiverInputStamp {
  uint64_t epoch = 0;
  uint64_t revision = 0;
  uint64_t sequence = 0;
  uint64_t rig_revision = 0;
  uint64_t calibration_revision = 0;
  int64_t observed_at_ns = 0;
  int64_t valid_until_ns = 0;
  bool known = false;
};

// No internal mutex: the owning Space serializes begin/commit/read/solve.
// Explicit `now_ns` makes expiry independently testable with a virtual clock;
// production Space callers always supply the native steady clock.
class ReceiverInputState {
 public:
  static constexpr size_t kMaxAnchors = 64;
  static constexpr size_t kMaxNameBytes = 128;
  static constexpr int64_t kMaxAgeNs = 1000000000;

  bool Begin(const std::string& root_anchor,
             const std::vector<std::string>& layout, uint64_t rig_revision,
             uint64_t calibration_revision, uint64_t* epoch);
  bool Commit(uint64_t epoch, uint64_t sequence, int64_t observed_at_ns,
              int64_t valid_until_ns,
              const std::vector<ReceiverAnchorInput>& anchors, int64_t now_ns);
  // Clear keeps the sequence/time fence and permits recovery in this epoch.
  bool Clear(uint64_t epoch);
  // Structural changes require a new Begin, never old-epoch recovery.
  void Invalidate();
  ReceiverInputStamp Stamp(int64_t now_ns) const;
  bool CopyPose(const std::string& anchor, Eigen::Matrix4d* out) const;
  const std::string& root_anchor() const { return root_anchor_; }

 private:
  bool AdvanceRevision();
  ReceiverInputStamp stamp_;
  bool accepting_ = false;
  bool has_frame_ = false;
  bool poisoned_ = false;
  std::string root_anchor_;
  std::unordered_map<std::string, ReceiverAnchorInput> anchors_;
};

int64_t ReceiverInputNowNs();
bool ReceiverPoseYaw(const Eigen::Matrix4d& pose, double* yaw);

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_RECEIVER_INPUT_H_
