// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "spatialblend/api/c/spatialblend.h"

namespace {
class COwnedCorrespondenceTest : public ::testing::Test {
 public:
  void SetUp() override {
    session = sb_create_session();
    source = sb_add_space(session);
    target = sb_add_space(session);
    sb_element_t a = 0, b = 0;
    for (int i = 0; i < 4; ++i) {
      a = sb_add_element(source, a);
      b = sb_add_element(target, b);
      pairs.push_back({source, a, target, b});
    }
  }
  void TearDown() override {
    if (session) sb_destroy_session(session);
  }
  int64_t Register(const std::vector<sb_support_correspondence_t>& p) {
    return sb_try_register_owned_correspondences(session, 1, p.data(),
                                                 p.size());
  }
  int Validate(int64_t lease) {
    return sb_try_validate_owned_correspondences(session, lease, 1);
  }
  sb_session_t session = 0;
  sb_space_t source = 0, target = 0;
  std::vector<sb_support_correspondence_t> pairs;
};

TEST_F(COwnedCorrespondenceTest,
       ExactWholePathOwnershipWithoutSemanticContact) {
  EXPECT_EQ(sb_get_owned_correspondence_feature_version(), 1u);
  const auto lease = Register(pairs);
  ASSERT_GT(lease, 0);
  EXPECT_EQ(Validate(lease), 1);
  // No actor, observation, seat label, geometry, or Contact is necessary.
  EXPECT_EQ(Register(pairs), 0);
  EXPECT_EQ(sb_remove_local_element(source, pairs[1].source_element), 0);
  EXPECT_EQ(sb_remove_local_element(target, pairs[1].target_element), 0);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, lease, 1), 1);
  EXPECT_EQ(Validate(lease), 0);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, lease, 1), 1);
  EXPECT_EQ(sb_remove_local_element(source, pairs[1].source_element), 1);
  EXPECT_EQ(sb_remove_local_element(target, pairs[1].target_element), 1);
}

TEST_F(COwnedCorrespondenceTest, RejectsInvalidAndCrossSessionArguments) {
  EXPECT_EQ(sb_try_register_owned_correspondences(session, 2, pairs.data(), 4),
            0);
  EXPECT_EQ(sb_try_register_owned_correspondences(session, 1, nullptr, 4), 0);
  EXPECT_EQ(sb_try_register_owned_correspondences(session, 1, pairs.data(), 0),
            0);
  EXPECT_EQ(
      sb_try_register_owned_correspondences(session, 1, pairs.data(), 129), 0);
  EXPECT_EQ(sb_try_validate_owned_correspondences(session, 0, 1), -1);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, 0, 1), -1);
  auto bad = pairs;
  bad.back().target_element = 999999;
  EXPECT_EQ(Register(bad), 0);
  bad = pairs;
  bad.back().source_element = -1;
  EXPECT_EQ(Register(bad), 0);
  const auto stranger = sb_create_session();
  const auto foreign = sb_add_space(stranger);
  bad = pairs;
  bad.back().target_space = foreign;
  bad.back().target_element = sb_add_element(foreign, 0);
  EXPECT_EQ(Register(bad), 0);
  const auto lease = Register(pairs);
  ASSERT_GT(lease, 0);
  EXPECT_EQ(sb_try_validate_owned_correspondences(stranger, lease, 1), 0);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(stranger, lease, 1), 0);
  EXPECT_EQ(sb_try_validate_owned_correspondences(session, lease, 2), -1);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, lease, 2), -1);
  EXPECT_EQ(Validate(lease), 1);
  sb_destroy_session(stranger);
}

TEST_F(COwnedCorrespondenceTest, MidBatchConflictDoesNotInstallEarlierPairs) {
  const auto borrowed =
      sb_register_support_correspondences(session, &pairs.back(), 1);
  ASSERT_GT(borrowed, 0);
  EXPECT_EQ(Register(pairs), 0);
  EXPECT_EQ(Validate(borrowed),
            0);  // A legacy lease cannot prove strict ownership.
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, borrowed, 1), 0);
  std::vector<sb_support_correspondence_t> prefix(pairs.begin(),
                                                  pairs.end() - 1);
  const auto lease = Register(prefix);
  ASSERT_GT(lease, 0);  // Failed whole-path install left no prefix behind.
  EXPECT_EQ(Validate(lease), 1);
  sb_dispose_support_correspondences(borrowed);
  EXPECT_EQ(Validate(lease), 1);
  const auto suffix = Register({pairs.back()});
  EXPECT_GT(suffix, 0);
}

TEST_F(COwnedCorrespondenceTest, DuplicateAndManyToOneRequestsFailAtomically) {
  auto bad = pairs;
  bad.push_back(pairs[0]);
  EXPECT_EQ(Register(bad), 0);
  bad = pairs;
  bad.back().target_element = pairs[0].target_element;
  EXPECT_EQ(Register(bad), 0);
  bad = pairs;
  bad.back().target_space = source;
  EXPECT_EQ(Register(bad), 0);
  const auto lease = Register(pairs);
  ASSERT_GT(lease, 0);
  const sb_support_correspondence_t ambiguous{
      source, sb_add_element(source, pairs[0].source_element), target,
      pairs[3].target_element};
  const auto legacy =
      sb_register_support_correspondences(session, &ambiguous, 1);
  ASSERT_GT(legacy, 0);
  EXPECT_EQ(Validate(lease), 0);
  sb_dispose_support_correspondences(legacy);
  EXPECT_EQ(Validate(lease), 1);
}

TEST_F(COwnedCorrespondenceTest, LegacyRetirementAndSessionShutdownAreInert) {
  auto lease = Register(pairs);
  ASSERT_GT(lease, 0);
  sb_dispose_support_correspondences(lease);
  EXPECT_EQ(Validate(lease), 0);
  const auto replacement = Register(pairs);
  ASSERT_GT(replacement, 0);
  EXPECT_NE(replacement, lease);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(session, lease, 1), 1);
  EXPECT_EQ(Validate(replacement), 1);
  const auto retired = session;
  sb_destroy_session(session);
  session = 0;
  EXPECT_EQ(sb_try_validate_owned_correspondences(retired, replacement, 1), 0);
  EXPECT_EQ(sb_try_dispose_owned_correspondences(retired, replacement, 1), 1);
  EXPECT_EQ(sb_try_register_owned_correspondences(retired, 1, pairs.data(), 4),
            0);
}
}  // namespace
