// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/space/builder/space_builder.h"

#include <gtest/gtest.h>

#include "spatialblend/space/builder/kernel_shorthands.h"
#include "spatialblend/space/local_context.h"

namespace spatialblend {
namespace test {

TEST(BuilderTest, FluentElementHierarchy) {
  auto context = LocalContext::Create();
  auto* space = context->AddSpace();
  SpaceBuilder sb(space);

  auto room = sb.AddElement("Room")
                   .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                   .WithSharedId(100);
  auto board = room.AddChild("Board").At(5, 15).WithSharedId(200);
  auto writable = board.AddChild("Writable").Tagged("Writable");

  EXPECT_NE(room.id(), board.id());
  EXPECT_NE(board.id(), writable.id());
  EXPECT_EQ(space->GetElementTree()->GetElement(room.id())->GetSharedId(),
            100);
  EXPECT_EQ(space->GetElementTree()->GetElement(board.id())->GetSharedId(),
            200);
}

TEST(BuilderTest, ElementPosition) {
  auto context = LocalContext::Create();
  auto* space = context->AddSpace();
  SpaceBuilder sb(space);
  auto room =
      sb.AddElement("Room").WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  auto board = room.AddChild("Board").At(5, 15);
  auto pos = board.Position();
  EXPECT_NEAR(pos.x(), 5.0, 0.01);
  EXPECT_NEAR(pos.y(), 15.0, 0.01);
}

TEST(BuilderTest, FluentActorBinding) {
  auto context = LocalContext::Create();
  auto* space = context->AddSpace();
  SpaceBuilder sb(space);
  auto room = sb.AddElement("Room")
                   .WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}})
                   .WithSharedId(100);
  room.AddChild("Board").At(5, 15).WithSharedId(200);
  auto actor = sb.AddActor()
                   .WithAnchor("root", 0, 0, 0)
                   .WithAnchor("chest", 0, 0, 1.2)
                   .InScope(room);
  actor.Bind("root", Spacing());
  actor.Bind("chest", Facing());
  EXPECT_EQ(space->GetBindings(actor.id()).size(), 2u);
}

TEST(BuilderTest, QueryShorthands) {
  auto context = LocalContext::Create();
  auto* space = context->AddSpace();
  SpaceBuilder sb(space);
  auto room =
      sb.AddElement("Room").WithPolygon({{0, 0}, {20, 0}, {20, 20}, {0, 20}});
  room.AddChild("Board").At(5, 15).Tagged("Writable");
  // Children() and Descendants() removed — scope is kernel-defined now.
}

}  // namespace test
}  // namespace spatialblend
