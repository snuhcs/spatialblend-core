// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// ABI 4.7: embodiment registration through the C API.
// The spec is a FIXTURE (the numbers of the evaluation harness's (not public) SMPL-X sit-bone point, to exercise the schema).
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>

#include "spatialblend/api/c/spatialblend.h"

namespace {

const char* kSpec = R"({
 "format":"spatialblend.embodiment","schema_version":"1.0.0","id":"fixture",
 "skeleton":{"joints":[{"name":"pelvis","parent":-1},{"name":"head","parent":0}]},
 "anchors":[
  {"name":"root","bone":null,"families":["Spacing"]},
  {"name":"head","bone":"head","families":["Facing"]},
  {"name":"pelvis","bone":"pelvis","offset":{"position":[0.0634,0.0,-0.1860]},"families":["AffordancePose"],
   "provenance":{"procedure":"fixture","source":"test fixture"}}],
 "chain":{"rest_pose":"T","provenance":{"procedure":"fixture","source":"test fixture"},
  "limits_provenance":{"procedure":"fixture","source":"test fixture"},
  "joints":[{"name":"pelvis","rest":{"position":[0,0,0],"rotation":[0,0,0,1]}},
            {"name":"head","rest":{"position":[0.01,0,0.55],"rotation":[0,0,0,1]},"limits":{"y":[-0.6,null]}}]}
})";

sb_bone_pose_t Pose(double x, double y, double z, double yaw) {
  sb_bone_pose_t p;
  std::memset(&p, 0, sizeof p);
  p.struct_size = sizeof p;
  p.position[0] = x;
  p.position[1] = y;
  p.position[2] = z;
  p.rotation[2] = std::sin(yaw / 2);
  p.rotation[3] = std::cos(yaw / 2);
  p.epoch = 1;
  return p;
}

struct Rig {
  sb_session_t session = sb_create_session();
  sb_space_t space = sb_add_space(session);
  sb_actor_t actor = sb_create_actor(space);
  ~Rig() { sb_cleanup_all(); }
};

TEST(EmbodimentC, RegisterReadBackAndRefusals) {
  Rig r;
  char err[256] = {0};
  EXPECT_EQ(sb_register_embodiment_json(r.space, r.actor, "not json", 8, err, sizeof err), -2);
  EXPECT_NE(std::string(err).find("JSON"), std::string::npos) << err;
  EXPECT_EQ(sb_register_embodiment_json(r.space, r.actor, nullptr, 0, err, sizeof err), -1);
  EXPECT_EQ(sb_get_embodiment_json(r.space, r.actor, nullptr, 0), 0) << "nothing registered yet";
  EXPECT_EQ(sb_get_chain_joint_count(r.space, r.actor), 0);

  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), err,
                                        sizeof err),
            0)
      << err;
  EXPECT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), err,
                                        sizeof err),
            -3);

  const int32_t need = sb_get_embodiment_json(r.space, r.actor, nullptr, 0);
  ASSERT_GT(need, 0);
  std::string json(need + 1, '\0');
  ASSERT_EQ(sb_get_embodiment_json(r.space, r.actor, &json[0], need + 1), need);
  json.resize(need);
  EXPECT_NE(json.find("\"id\":\"fixture\""), std::string::npos);
  EXPECT_NE(json.find("\"chain\""), std::string::npos);
  char small[8];
  EXPECT_EQ(sb_get_embodiment_json(r.space, r.actor, small, sizeof small), need);  // truncated, full length returned
  EXPECT_EQ(std::strlen(small), 7u);

  EXPECT_EQ(sb_get_bone_id(r.space, r.actor, "head"), 1);
  EXPECT_EQ(sb_get_bone_id(r.space, r.actor, "tail"), -1);

  // The chain reads back typed.
  ASSERT_EQ(sb_get_chain_joint_count(r.space, r.actor), 2);
  sb_chain_joint_t j;
  std::memset(&j, 0, sizeof j);
  j.struct_size = sizeof j;
  char name[16];
  ASSERT_EQ(sb_get_chain_joint(r.space, r.actor, 1, &j, name, sizeof name), 0);
  EXPECT_STREQ(name, "head");
  EXPECT_EQ(j.parent, 0);
  EXPECT_DOUBLE_EQ(j.rest_position[0], 0.01);
  EXPECT_DOUBLE_EQ(j.rest_position[2], 0.55);
  EXPECT_DOUBLE_EQ(j.rest_rotation[3], 1.0);
  EXPECT_NEAR(j.segment_length_m, std::sqrt(0.01 * 0.01 + 0.55 * 0.55), 1e-15);
  EXPECT_EQ(j.limit_mask, 1u << 2);  // y has a min, no max
  EXPECT_DOUBLE_EQ(j.limit_min[1], -0.6);
  EXPECT_EQ(sb_get_chain_joint(r.space, r.actor, 2, &j, name, sizeof name), -1);
  j.struct_size = 1;
  EXPECT_EQ(sb_get_chain_joint(r.space, r.actor, 0, &j, name, sizeof name), -1);
}

TEST(EmbodimentC, BonesAreThePortAndAnchorsAreDerived) {
  Rig r;
  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), nullptr, 0), 0);
  uint32_t pelvis_anchor = 0;
  ASSERT_EQ(sb_get_anchor_id(r.space, r.actor, "pelvis", &pelvis_anchor), 0);

  sb_anchor_pose_t a;
  std::memset(&a, 0, sizeof a);
  a.struct_size = sizeof a;
  a.orientation.struct_size = sizeof a.orientation;
  a.orientation.type = SB_ORIENTATION_YAW;
  EXPECT_EQ(sb_publish_anchor_pose(r.space, r.actor, pelvis_anchor, &a, 1.0, 1), -3) << "a bone-bound anchor has no port";

  sb_bone_pose_t b = Pose(1.5, -2.0, 0.636, 0.7);
  EXPECT_EQ(sb_publish_bone_pose(r.space, r.actor, 0, &b, 1.0, 1), 0);
  EXPECT_EQ(sb_publish_bone_pose(r.space, r.actor, 0, &b, 1.0, 1), -2);
  EXPECT_EQ(sb_publish_bone_pose(r.space, r.actor, 9, &b, 1.0, 2), -1);
  b.struct_size = 4;
  EXPECT_EQ(sb_publish_bone_pose(r.space, r.actor, 0, &b, 1.0, 3), -1);

  // The registration's arithmetic, both ways.
  sb_bone_pose_t bone = Pose(1.5, -2.0, 0.636, 0.7), anchor, back;
  ASSERT_EQ(sb_embodiment_anchor_pose_from_bone(r.space, r.actor, "pelvis", &bone, &anchor), 0);
  EXPECT_NEAR(anchor.position[0], 1.5 + std::cos(0.7) * 0.0634, 1e-14);
  EXPECT_NEAR(anchor.position[2], 0.636 - 0.1860, 1e-14);
  ASSERT_EQ(sb_embodiment_bone_pose_from_anchor(r.space, r.actor, "pelvis", &anchor, &back), 0);
  for (int i = 0; i < 3; ++i) EXPECT_NEAR(back.position[i], bone.position[i], 1e-14);
  EXPECT_EQ(sb_embodiment_anchor_pose_from_bone(r.space, r.actor, "root", &bone, &anchor), -1) << "the root is not bone-bound";
  EXPECT_EQ(sb_embodiment_anchor_pose_from_bone(r.space, r.actor, "nope", &bone, &anchor), -1);
}

TEST(EmbodimentC, DeclaredFamiliesAreEnforcedAtBind) {
  Rig r;
  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), nullptr, 0), 0);
  EXPECT_EQ(sb_bind(r.space, r.actor, "pelvis", SB_KERNEL_SPACING), 0);
  EXPECT_EQ(sb_bind_affordance(r.space, r.actor, "pelvis", "Sittable"), 1);
  EXPECT_EQ(sb_bind(r.space, r.actor, "head", SB_KERNEL_FACING), 1);
  EXPECT_EQ(sb_bind(r.space, r.actor, "head", SB_KERNEL_SPACING), 0);
}

TEST(EmbodimentC, SourceBoneFrameIsComposedFromTheRegistration) {
  Rig r;
  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), nullptr, 0), 0);
  // Needs frames enabled like sb_apply_actor_observation_frame.
  sb_named_bone_pose_t bones[1];
  bones[0].bone_name = "pelvis";
  bones[0].pose = Pose(0.4, 0.2, 0.636, 1.1);
  sb_actor_observation_t root = {"root", 0.4, 0.2, 0.0, 1.1};
  EXPECT_EQ(sb_apply_actor_bone_frame(r.space, r.actor, 1, 0, 0.0, bones, 1, &root, 1), 0) << "frames not enabled";
  ASSERT_EQ(sb_enable_actor_observation_frames(r.session), 1);
  const uint64_t epoch = sb_begin_actor_observation_epoch(r.space, r.actor, "root");
  ASSERT_NE(epoch, 0u);
  EXPECT_EQ(sb_apply_actor_bone_frame(r.space, r.actor, epoch, 0, 0.0, bones, 1, &root, 1), 1);
  // A bone-bound anchor stated as an observation too is two sources for one fact.
  sb_actor_observation_t twice[] = {{"root", 0.4, 0.2, 0.0, 1.1}, {"pelvis", 0, 0, 0, 0}};
  EXPECT_EQ(sb_apply_actor_bone_frame(r.space, r.actor, epoch, 1, 0.1, bones, 1, twice, 2), 0);
  // The legacy observation frame refuses a bone-bound anchor as well.
  sb_actor_observation_t legacy[] = {{"root", 0.4, 0.2, 0.0, 1.1}, {"pelvis", 0.4, 0.2, 0.45, 1.1}};
  EXPECT_EQ(sb_apply_actor_observation_frame(r.space, r.actor, epoch, 1, 0.1, legacy, 2), 0);
  EXPECT_EQ(sb_apply_actor_observation_frame(r.space, r.actor, epoch, 1, 0.1, legacy, 1), 1) << "root alone is fine";
  bones[0].bone_name = "tail";
  EXPECT_EQ(sb_apply_actor_bone_frame(r.space, r.actor, epoch, 1, 0.1, bones, 1, &root, 1), 0);
}

TEST(EmbodimentC, ASourceOnALegacySessionFeedsItsBoneBoundAnchorsFromBonesWithoutObservationFrames) {
  Rig r;  // a plain session: observation frames are NOT enabled (the networked FC session)
  sb_named_bone_pose_t bones[1];
  bones[0].bone_name = "pelvis";
  bones[0].pose = Pose(0.4, 0.2, 0.636, 1.1);
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, bones, 1), -1) << "no embodiment registered";
  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), nullptr, 0), 0);
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, bones, 1), 1) << "the pelvis anchor, composed from its registration";
  sb_named_bone_pose_t both[2];
  both[0] = bones[0];
  both[1].bone_name = "head";
  both[1].pose = Pose(0.4, 0.2, 1.2, 1.1);
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, both, 2), 2);
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, nullptr, 0), 0);
  bones[0].bone_name = "tail";
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, bones, 1), -1) << "an unknown bone";
  both[1].bone_name = "pelvis";
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, both, 2), -1) << "a repeated bone";
  EXPECT_EQ(sb_set_bone_anchor_transforms(r.space, r.actor, nullptr, 1), -1);
  EXPECT_EQ(sb_set_bone_anchor_transforms(0, r.actor, both, 1), -1);
  // It works next to the legacy updates of the same session.
  (void)sb_update_actor_state(r.space, r.actor, 0.4, 0.2, 1.1);  // no crash, no interference
}

TEST(EmbodimentC, ARegistrationThatContradictsAnExistingBindIsRefusedWithTheReason) {
  Rig r;
  sb_add_anchor(r.space, r.actor, "pelvis", 0, 0, 0.9, nullptr);
  ASSERT_EQ(sb_bind(r.space, r.actor, "pelvis", SB_KERNEL_SPACING), 1);
  char err[256] = {0};
  EXPECT_EQ(sb_register_embodiment_json(r.space, r.actor, kSpec, static_cast<int32_t>(std::strlen(kSpec)), err, sizeof err), -3);
  EXPECT_NE(std::string(err).find("pelvis@Spacing"), std::string::npos) << err;
  EXPECT_EQ(sb_get_embodiment_json(r.space, r.actor, nullptr, 0), 0);
}

TEST(EmbodimentC, TheBodyRadiusIsDeclaredPerActorByTheEmbodiment) {
  Rig r;
  EXPECT_EQ(sb_set_body_radius(r.space, 0.3), 0);  // the space-wide radius of actors that register none
  std::string spec = kSpec;
  const std::string body =
      R"("body":{"floor_footprint_radius_m":0.33,"provenance":{"procedure":"fixture","source":"test fixture"}},)";
  spec.insert(spec.find("\"skeleton\""), body);
  char err[256] = {0};
  ASSERT_EQ(sb_register_embodiment_json(r.space, r.actor, spec.c_str(), static_cast<int32_t>(spec.size()), err, sizeof err), 0) << err;
  EXPECT_EQ(sb_set_body_radius(r.space, 0.5), 0) << "the space-wide radius is still settable: it is the other actors' fallback";
  const int32_t need = sb_get_embodiment_json(r.space, r.actor, nullptr, 0);
  std::string json(need + 1, '\0');
  sb_get_embodiment_json(r.space, r.actor, &json[0], need + 1);
  EXPECT_NE(json.find("\"floor_footprint_radius_m\":0.33"), std::string::npos);
}

TEST(EmbodimentC, ConvertPoseBasisAndStructSizes) {
  const double lh[9] = {0, 0, 1, -1, 0, 0, 0, 1, 0};
  sb_bone_pose_t native = Pose(0.45, 1.31, 0.12, 0.0), canonical, again;
  ASSERT_EQ(sb_convert_pose_basis(lh, 1, &native, &canonical), 0);
  EXPECT_DOUBLE_EQ(canonical.position[0], 0.12);
  EXPECT_DOUBLE_EQ(canonical.position[1], -0.45);
  EXPECT_DOUBLE_EQ(canonical.position[2], 1.31);
  ASSERT_EQ(sb_convert_pose_basis(lh, 0, &canonical, &again), 0);
  for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(again.position[i], native.position[i]);
  const double bad[9] = {1, 1, 0, 0, 0, 1, 0, 1, 0};
  EXPECT_EQ(sb_convert_pose_basis(bad, 1, &native, &canonical), -1);
  EXPECT_EQ(sb_sizeof(SB_STRUCT_BONE_POSE), sizeof(sb_bone_pose_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_NAMED_BONE_POSE), sizeof(sb_named_bone_pose_t));
  EXPECT_EQ(sb_sizeof(SB_STRUCT_CHAIN_JOINT), sizeof(sb_chain_joint_t));
  EXPECT_EQ(sb_abi_minor_version(), 14u);
}

}  // namespace
