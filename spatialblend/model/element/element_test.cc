// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/model/element/element.h"

#include <gtest/gtest.h>

#include "spatialblend/model/element/navigable_spatial_definition.h"

namespace spatialblend {
namespace test {

TEST(ElementTest, CreateWithId) {
  Element elem(1);
  EXPECT_EQ(elem.GetId(), 1);
  EXPECT_EQ(elem.GetParentId(), kInvalidElementId);
  EXPECT_EQ(elem.GetSharedId(), kInvalidSharedElementId);
  EXPECT_EQ(elem.GetGeometryRole(), GeometryRole::kUnspecified);
  EXPECT_TRUE(elem.GetChildIds().empty());
  EXPECT_TRUE(elem.IsRoot());
  EXPECT_TRUE(elem.IsLeaf());
}

TEST(ElementTest, SharedIdentity) {
  Element sofa(1);
  sofa.SetSharedId(100);
  EXPECT_EQ(sofa.GetSharedId(), 100);

  Element couch(2);
  couch.SetSharedId(100);
  EXPECT_EQ(sofa.GetSharedId(), couch.GetSharedId());
}

TEST(ElementTest, ParentChild) {
  Element room(1);
  Element table(2);

  table.SetParentId(1);
  room.AddChildId(2);

  EXPECT_EQ(table.GetParentId(), 1);
  EXPECT_FALSE(table.IsRoot());
  EXPECT_FALSE(room.IsLeaf());
  ASSERT_EQ(room.GetChildIds().size(), 1);
  EXPECT_EQ(room.GetChildIds()[0], 2);
}

TEST(ElementTest, RemoveChild) {
  Element room(1);
  room.AddChildId(2);
  room.AddChildId(3);
  room.RemoveChildId(2);
  ASSERT_EQ(room.GetChildIds().size(), 1);
  EXPECT_EQ(room.GetChildIds()[0], 3);
}

TEST(ElementTest, SpatialDefinition) {
  Element room(1);
  EXPECT_EQ(room.GetSpatialDefinition(), nullptr);

  auto walkable = std::make_shared<spatialblend::Polygon2D>(
      std::vector<Eigen::Vector2d>{{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  auto nav = std::make_shared<NavigableSpatialDefinition>(walkable);
  room.SetSpatialDefinition(nav);

  ASSERT_NE(room.GetSpatialDefinition(), nullptr);
  EXPECT_EQ(room.GetGeometryRole(), GeometryRole::kNavigable);
  EXPECT_TRUE(room.HasGeometryRole(GeometryRole::kNavigable));
  EXPECT_TRUE(room.GetSpatialDefinition()->Contains(Eigen::Vector2d(5, 5)));
}

TEST(ElementTest, GroundingEntityId) {
  Element table(2);
  EXPECT_EQ(table.GetGroundingEntityId(), spatialblend::kInvalidEntityId);

  table.SetGroundingEntityId(42);
  EXPECT_EQ(table.GetGroundingEntityId(), 42);
}

TEST(ElementTest, LocalTransform) {
  Element elem(1);
  EXPECT_TRUE(elem.GetLocalTransform().isIdentity());

  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 5.0;
  t(1, 3) = 3.0;
  elem.SetLocalTransform(t);
  EXPECT_DOUBLE_EQ(elem.GetLocalTransform()(0, 3), 5.0);
  EXPECT_DOUBLE_EQ(elem.GetLocalTransform()(1, 3), 3.0);
}

TEST(ElementTest, TagDefaultEmpty) {
  Element elem(1);
  EXPECT_TRUE(elem.GetTag().empty());
}

TEST(ElementTest, SetAndGetTag) {
  Element elem(1);
  elem.SetTag("Writable");
  EXPECT_EQ(elem.GetTag(), "Writable");
}

TEST(ElementTest, AddAndGetAnchor) {
  Element e(1);
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  t(0, 3) = 1.0;
  t(1, 3) = 2.0;
  t(2, 3) = 3.0;
  e.AddAnchor("surface_center", t);

  const auto* anchor = e.GetAnchor("surface_center");
  ASSERT_NE(anchor, nullptr);
  EXPECT_NEAR((*anchor)(0, 3), 1.0, 1e-9);
  EXPECT_NEAR((*anchor)(1, 3), 2.0, 1e-9);
  EXPECT_NEAR((*anchor)(2, 3), 3.0, 1e-9);
}

TEST(ElementTest, GetAnchorNotFound) {
  Element e(1);
  EXPECT_EQ(e.GetAnchor("nonexistent"), nullptr);
}

TEST(ElementTest, MultipleAnchors) {
  Element e(1);
  Eigen::Matrix4d t1 = Eigen::Matrix4d::Identity();
  t1(2, 3) = 1.5;
  Eigen::Matrix4d t2 = Eigen::Matrix4d::Identity();
  t2(2, 3) = 0.45;
  e.AddAnchor("surface_center", t1);
  e.AddAnchor("seat", t2);

  EXPECT_EQ(e.GetAnchors().size(), 2u);
  EXPECT_NE(e.GetAnchor("surface_center"), nullptr);
  EXPECT_NE(e.GetAnchor("seat"), nullptr);
}

}  // namespace test
}  // namespace spatialblend
