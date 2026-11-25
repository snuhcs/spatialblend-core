// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/embodiment/embodiment.h
//
// An embodiment registration (ABI 4.7): the skeleton an app or driver
// publishes, and the subject anchors it declares against that skeleton -- each anchor's bone, its offset in
// that bone's frame, the relation families it serves and where the offset came from.
//
// Pure data and arithmetic. No Space, no thread, no clock. The offset of an anchor lives here and nowhere
// else: the driver publishes BONES, and the anchor pose the core reads is `bone * offset` (ComposeAnchor).
// Nothing in this file is a body measurement; every number arrives in a registered spec, and a non-identity
// offset must carry provenance or the spec is refused.
#ifndef SPATIALBLEND_SPACE_EMBODIMENT_H_
#define SPATIALBLEND_SPACE_EMBODIMENT_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace spatialblend {

// Relation-family bits of an anchor's declared service (mirror SB_FAMILY_* in spatialblend_type.h; the bit of
// family f is 1 << sb_kernel_type_t).
constexpr uint32_t kFamilySpacing = 1u << 0;
constexpr uint32_t kFamilyFacing = 1u << 1;
constexpr uint32_t kFamilyAffordancePose = 1u << 2;

// A rigid transform, local -> parent (bone -> world for a bone pose, anchor frame -> bone frame for an offset).
struct RigidTransform {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();  // unit
};

// yaw / pitch / roll of a rotation in the core's Z x Y x X order (CONTRIBUTING.md): +pitch = the forward axis tips
// down, as the head pitch the port already carries.
struct EulerZyx {
  double yaw = 0.0, pitch = 0.0, roll = 0.0;
};

struct EmbodimentJoint {
  std::string name;
  int parent = -1;  // index of an EARLIER joint, or -1
};

struct EmbodimentAnchor {
  std::string name;
  int bone = -1;  // index into the skeleton; -1 = the actor root (stated as today, never composed)
  RigidTransform offset;  // anchor frame -> bone frame
  bool identity_offset = true;
  uint32_t families = 0;  // kFamily* bits
  // The provenance object as canonical JSON text ("" when the anchor has none). Stored verbatim; the core
  // does not interpret it.
  std::string provenance_json;
  // ABI 4.11 (affordance poses): the anchor's STANDING height above the actor's root, in metres,
  // measured on the embodiment's own body (for the pelvis: the standing sit-bone height). It instantiates the
  // `kActorRoot` affordance pose templates ("stand") for this embodiment. The normalized kinematic chain cannot give
  // it: the chain's origin is its first root joint, not the floor. Optional;
  // when present it needs provenance like an offset (`standing.provenance`). Never a core constant.
  bool has_standing = false;
  double standing_height_m = 0.0;
  std::string standing_provenance_json;
};

// The normalized kinematic chain. The hierarchy is the skeleton's; the
// chain adds each joint's REST pose in the core's canonical axes and optional joint limits. Defined and stored
// only: no solve reads it.
struct ChainJoint {
  RigidTransform rest;  // joint frame -> model space, canonical axes; the first root joint is at the origin
  bool has_lo[3] = {false, false, false}, has_hi[3] = {false, false, false};
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};  // radians, ZYX Euler (x = roll, y = pitch, z = yaw) of rest^-1 * pose
  bool HasLimits() const {
    for (int i = 0; i < 3; ++i)
      if (has_lo[i] || has_hi[i]) return true;
    return false;
  }
};

struct Chain {
  bool present = false;
  std::string rest_pose;          // the embodiment's own bind pose, labelled ("A", "T", ...); never converted
  std::string provenance_json;    // required: how the rest pose was obtained (procedure, and a 64-hex asset_sha256 or a source)
  std::string limits_provenance_json;  // required when any joint has a limit
  std::vector<ChainJoint> joints;  // skeleton order, same size as Embodiment::joints
};

// The body's floor footprint (the radius comes from the embodiment, registered per actor).
struct EmbodimentBody {
  bool present = false;
  double floor_footprint_radius_m = 0.0;  // the radius of the circle on the floor that bounds the body's footprint
  std::string provenance_json;            // required: measured or declared by the embodiment (procedure + hash/source)
};

class Embodiment {
 public:
  std::string id;
  std::string units = "m";
  std::string frame = "z-up, right-handed";
  std::string bone_frame;
  std::vector<EmbodimentJoint> joints;
  std::vector<EmbodimentAnchor> anchors;
  Chain chain;
  EmbodimentBody body;

  // Parse and validate a spec (schema spatialblend.embodiment 1.x). False with a reason in `error`.
  static bool FromJson(const std::string& text, Embodiment* out, std::string* error);
  // Structural validation (names, parents, bones, quaternions, families, provenance of a non-identity offset).
  bool Validate(std::string* error) const;
  // The canonical JSON of the registration (keys sorted, doubles round-trip).
  std::string ToJson() const;

  int BoneId(const std::string& name) const;
  const EmbodimentAnchor* FindAnchor(const std::string& name) const;
  int AnchorIndex(const std::string& name) const;
};

// anchor = bone * offset, and its inverse bone = anchor * offset^-1. Pure.
RigidTransform ComposeAnchor(const EmbodimentAnchor& anchor, const RigidTransform& bone);
RigidTransform BoneFromAnchor(const EmbodimentAnchor& anchor, const RigidTransform& anchor_pose);
EulerZyx EulerFromRotation(const Eigen::Quaterniond& rotation);

// The length of the segment that ends at joint `i` (parent rest position -> joint rest position); 0 for a root.
// Derived, never stored: the rest positions are the one source.
double SegmentLength(const Embodiment& e, int i);

// A change of basis from a skeleton's native axes to the canonical ones (z up, x forward, y left, right-handed).
// `rows` maps native coordinates to canonical: c = B * n, a signed permutation (exactly one +-1 per row and
// column). det(B) = -1 is a handedness change (Unity); a rotation then converts as B R B^T, still proper.
struct AxisBasis {
  Eigen::Matrix3d b = Eigen::Matrix3d::Identity();
  static bool FromRows(const double rows[9], AxisBasis* out, std::string* error);
};
RigidTransform ToCanonical(const AxisBasis& basis, const RigidTransform& native);
RigidTransform FromCanonical(const AxisBasis& basis, const RigidTransform& canonical);

// True when `name` is a registered anchor of `e` attached to a bone (so its pose is derived, never stated).
inline bool IsBoneBound(const std::shared_ptr<const Embodiment>& e, const std::string& name) {
  const EmbodimentAnchor* a = e ? e->FindAnchor(name) : nullptr;
  return a && a->bone >= 0;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_EMBODIMENT_H_
