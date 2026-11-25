// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "spatialblend/net/proto_util.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "spatialblend/geometry/polygon2d.h"
#include "spatialblend/model/relation/grounded_relation.h"
#include "spatialblend/model/relation/support.h"
#include "spatialblend/space/relation_event.h"
#include "spatialblend/model/realization/remote_update.h"

namespace spatialblend {
namespace {

TEST(ProtoUtilTest, RelationUpdateRoundtrip) {
  RemoteRelationUpdate update;
  update.source_space_id = 42;
  update.actor_shared_id = 7;
  update.step_duration = 0.05;
  update.sequence = 17;  // dedupe counter, must survive the round trip.
  update.source_clock = 3.14159;  // real source clock, must survive too.

  // First payload
  RelationPayload p1;
  p1.referent_shared_id = 100;
  p1.subject_anchor_name = "head";
  p1.grounded_relation.encoded_state = Eigen::VectorXd::Ones(3);
  p1.grounded_relation.proximity = 0.8;
  // The source scope diameter proximity/support_distance_m were
  // normalized against, so the decoder can reconstruct the exact metric
  // approach-branch standoff (see RelationSolver::DecodeTargetPoint).
  p1.grounded_relation.support_distance_m = 2.4;
  p1.grounded_relation.source_scope_diameter_m = 12.0;
  p1.kernel_type = RelationKernelType::kAvatarFacing;
  p1.estimated_time = 1.5;
  p1.update_count = 4.0;
  p1.facing_rotation_sign = -1;
  p1.facing_opening = true;
  p1.affordance_pose_tag = "seated";  // ABI 4.11: an inner affordance-pose relation

  // Second payload
  RelationPayload p2;
  p2.referent_shared_id = 200;
  p2.subject_anchor_name = "torso";
  p2.grounded_relation.encoded_state = Eigen::VectorXd::Zero(5);
  p2.grounded_relation.encoded_state[2] = 3.14;
  p2.grounded_relation.proximity = 0.3;
  p2.kernel_type = RelationKernelType::kAvatarSpacing;
  p2.estimated_time = 0.0;

  update.payloads = {p1, p2};

  auto proto = ToProto(update);
  auto result = FromProto(proto);

  EXPECT_EQ(result.source_space_id, 42);
  EXPECT_EQ(result.actor_shared_id, 7);
  EXPECT_DOUBLE_EQ(result.step_duration, 0.05);
  EXPECT_EQ(result.sequence, 17);
  EXPECT_DOUBLE_EQ(result.source_clock, 3.14159);

  ASSERT_EQ(result.payloads.size(), 2u);

  const auto& r1 = result.payloads[0];
  EXPECT_EQ(r1.referent_shared_id, 100);
  EXPECT_EQ(r1.subject_anchor_name, "head");
  EXPECT_EQ(r1.grounded_relation.encoded_state.size(), 3);
  EXPECT_DOUBLE_EQ(r1.grounded_relation.proximity, 0.8);
  EXPECT_EQ(r1.kernel_type, RelationKernelType::kAvatarFacing);
  EXPECT_DOUBLE_EQ(r1.estimated_time, 1.5);
  EXPECT_DOUBLE_EQ(r1.update_count, 4.0);
  EXPECT_EQ(r1.facing_rotation_sign, -1);
  EXPECT_TRUE(r1.facing_opening);
  EXPECT_EQ(r1.affordance_pose_tag, "seated");
  EXPECT_DOUBLE_EQ(r1.grounded_relation.support_distance_m, 2.4);
  EXPECT_DOUBLE_EQ(r1.grounded_relation.source_scope_diameter_m, 12.0);

  const auto& r2 = result.payloads[1];
  EXPECT_EQ(r2.referent_shared_id, 200);
  EXPECT_EQ(r2.subject_anchor_name, "torso");
  EXPECT_EQ(r2.grounded_relation.encoded_state.size(), 5);
  EXPECT_DOUBLE_EQ(r2.grounded_relation.encoded_state[2], 3.14);
  EXPECT_DOUBLE_EQ(r2.grounded_relation.proximity, 0.3);
  EXPECT_EQ(r2.kernel_type, RelationKernelType::kAvatarSpacing);
  EXPECT_DOUBLE_EQ(r2.estimated_time, 0.0);
  EXPECT_FALSE(std::isfinite(r2.update_count));
  EXPECT_TRUE(r2.affordance_pose_tag.empty());  // absent: not an inner relation
}

// The source binding (ordinal, revision) survives the wire; an unknown
// revision stays unknown (absent field), and a legacy sender that never wrote
// it decodes as unknown.
TEST(ProtoUtilTest, BindingRevisionRoundtripKeepsKnownness) {
  RemoteRelationUpdate update;
  RelationPayload known;
  known.referent_shared_id = 1;
  known.source_binding_ordinal = 3;
  known.binding_revision = 41;
  known.binding_revision_known = true;
  RelationPayload unknown;
  unknown.referent_shared_id = 2;
  unknown.source_binding_ordinal = 4;
  unknown.binding_revision = 99;  // not certified: must not travel
  unknown.binding_revision_known = false;
  update.payloads = {known, unknown};
  const auto result = FromProto(ToProto(update));
  ASSERT_EQ(result.payloads.size(), 2u);
  EXPECT_TRUE(result.payloads[0].binding_revision_known);
  EXPECT_EQ(result.payloads[0].binding_revision, 41u);
  EXPECT_EQ(result.payloads[0].source_binding_ordinal, 3u);
  EXPECT_FALSE(result.payloads[1].binding_revision_known);
  EXPECT_EQ(result.payloads[1].binding_revision, 0u);
  EXPECT_EQ(result.payloads[1].source_binding_ordinal, 4u);

  proto::RelationUpdateProto legacy = ToProto(update);
  legacy.mutable_payloads(0)->clear_binding_revision();
  legacy.mutable_payloads(0)->clear_source_binding_ordinal();
  const auto legacy_result = FromProto(legacy);
  EXPECT_FALSE(legacy_result.payloads[0].binding_revision_known);
  EXPECT_EQ(legacy_result.payloads[0].source_binding_ordinal, UINT64_MAX);
}

// The AffordancePose anchor-pose offset/height round-trips, and a
// pre-pose-fit sender (none of the three fields present) decodes to NaN/unknown,
// never 0.
TEST(ProtoUtilTest, AffordancePoseOffsetRoundtripAndOldSenderIsUnknown) {
  RemoteRelationUpdate update;
  RelationPayload known;
  known.referent_shared_id = 1;
  known.kernel_type = RelationKernelType::kAvatarAffordancePose;
  known.grounded_relation.ap_offset_u_m = 0.05;
  known.grounded_relation.ap_offset_v_m = -0.02;
  known.grounded_relation.ap_height_m = 0.1;
  RelationPayload unknown;  // never set: NaN by GroundedRelation's default
  unknown.referent_shared_id = 2;
  unknown.kernel_type = RelationKernelType::kAvatarAffordancePose;
  update.payloads = {known, unknown};
  const auto result = FromProto(ToProto(update));
  ASSERT_EQ(result.payloads.size(), 2u);
  EXPECT_DOUBLE_EQ(result.payloads[0].grounded_relation.ap_offset_u_m, 0.05);
  EXPECT_DOUBLE_EQ(result.payloads[0].grounded_relation.ap_offset_v_m, -0.02);
  EXPECT_DOUBLE_EQ(result.payloads[0].grounded_relation.ap_height_m, 0.1);
  EXPECT_TRUE(std::isnan(result.payloads[1].grounded_relation.ap_offset_u_m));
  EXPECT_TRUE(std::isnan(result.payloads[1].grounded_relation.ap_offset_v_m));
  EXPECT_TRUE(std::isnan(result.payloads[1].grounded_relation.ap_height_m));

  // Old-sender compatibility: a pre-pose-fit wire message never carries these
  // fields at all (not merely absent on this one payload).
  proto::RelationUpdateProto pre_a4 = ToProto(update);
  pre_a4.mutable_payloads(0)->clear_ap_offset_u_m();
  pre_a4.mutable_payloads(0)->clear_ap_offset_v_m();
  pre_a4.mutable_payloads(0)->clear_ap_height_m();
  const auto pre_a4_result = FromProto(pre_a4);
  EXPECT_TRUE(std::isnan(pre_a4_result.payloads[0].grounded_relation.ap_offset_u_m));
  EXPECT_TRUE(std::isnan(pre_a4_result.payloads[0].grounded_relation.ap_offset_v_m));
  EXPECT_TRUE(std::isnan(pre_a4_result.payloads[0].grounded_relation.ap_height_m));
}

// The timing mode, observation epoch and time, and the sender's quiet
// assertion survive the network. An older sender's update (none of the
// fields) decodes as legacy, not quiet.
TEST(ProtoUtilTest, TimingEpochTimeAndQuietRoundTrip) {
  RemoteRelationUpdate stamped;
  stamped.timing = RelationUpdateTiming::kTimestampedObservation;
  stamped.source_epoch = 7;
  stamped.source_time = 12.25;
  stamped.sequence = 40;
  auto decoded = FromProto(ToProto(stamped));
  EXPECT_EQ(decoded.timing, RelationUpdateTiming::kTimestampedObservation);
  EXPECT_EQ(decoded.source_epoch, 7u);
  EXPECT_DOUBLE_EQ(decoded.source_time, 12.25);
  EXPECT_EQ(decoded.sequence, 40);
  EXPECT_FALSE(decoded.quiet_since_last_send);

  RemoteRelationUpdate reset;
  reset.timing = RelationUpdateTiming::kObservationReset;
  reset.source_epoch = 8;
  reset.source_time = 0.0;
  decoded = FromProto(ToProto(reset));
  EXPECT_EQ(decoded.timing, RelationUpdateTiming::kObservationReset);
  EXPECT_EQ(decoded.source_epoch, 8u);

  RemoteRelationUpdate heartbeat;
  heartbeat.sequence = 41;
  heartbeat.source_clock = 3.0;
  heartbeat.quiet_since_last_send = true;
  const proto::RelationUpdateProto wire = ToProto(heartbeat);
  EXPECT_FALSE(wire.has_timing());  // legacy stays absent on the wire
  decoded = FromProto(wire);
  EXPECT_EQ(decoded.timing, RelationUpdateTiming::kLegacy);
  EXPECT_TRUE(decoded.quiet_since_last_send);

  proto::RelationUpdateProto older = wire;  // a sender without the fields
  older.clear_quiet_since_last_send();
  decoded = FromProto(older);
  EXPECT_EQ(decoded.timing, RelationUpdateTiming::kLegacy);
  EXPECT_FALSE(decoded.quiet_since_last_send);
  EXPECT_EQ(decoded.source_epoch, 0u);

  proto::RelationUpdateProto unknown = wire;  // a newer timing value
  unknown.set_timing(static_cast<proto::RelationUpdateTimingProto>(99));
  decoded = FromProto(unknown);
  EXPECT_EQ(decoded.timing, RelationUpdateTiming::kLegacy);
}

TEST(ProtoUtilTest, NativeDisabledPayloadIsOmittedByLegacyProjection) {
  RemoteRelationUpdate only_disabled;
  RelationPayload disabled;
  disabled.referent_shared_id = 9;
  disabled.enabled = false;
  only_disabled.payloads = {disabled};
  EXPECT_TRUE(FromProto(ToProto(only_disabled)).payloads.empty());

  RemoteRelationUpdate mixed;
  RelationPayload enabled;
  enabled.referent_shared_id = 10;
  enabled.subject_anchor_name = "head";
  mixed.payloads = {enabled, disabled};
  const auto decoded = FromProto(ToProto(mixed));
  ASSERT_EQ(decoded.payloads.size(), 1u);
  EXPECT_EQ(decoded.payloads[0].referent_shared_id, 10);
  EXPECT_TRUE(decoded.payloads[0].enabled);
}

TEST(ProtoUtilTest, ObservedPointCannotBeProjectedAsLegacyGeometry) {
  RelationPayload point;
  point.referent_shared_id = 17;
  point.subject_anchor_name = "head";
  point.observed_facing_point = true;
  point.observed_logical_actor = 88;
  point.observed_source_point = {1, 7};
  point.observed_source_point_lease = 3;
  point.observed_valid_until_ns = 9000;
  point.grounded_relation.encoded_state = Eigen::VectorXd::Zero(1);
  EXPECT_FALSE(TryToProtoLegacy(point).has_value());
  RelationPayload disabled;
  disabled.enabled = false;
  EXPECT_FALSE(TryToProtoLegacy(disabled).has_value());
  RelationPayload ordinary;
  ordinary.referent_shared_id = 18;
  RemoteRelationUpdate update;
  update.payloads = {point, ordinary, disabled};
  const auto decoded = FromProto(ToProto(update));
  ASSERT_EQ(decoded.payloads.size(), 1u);
  EXPECT_EQ(decoded.payloads[0].referent_shared_id, 18);
  EXPECT_FALSE(decoded.payloads[0].observed_facing_point);
  EXPECT_EQ(decoded.payloads[0].observed_source_point_lease, 0u);
}

TEST(ProtoUtilTest, SupportWithContactRoundtrip) {
  Support support;

  // Set a non-identity world_transform
  support.world_transform = Eigen::Matrix4d::Zero();
  support.world_transform(0, 0) = 0.5;
  support.world_transform(1, 1) = 0.5;
  support.world_transform(2, 2) = 1.0;
  support.world_transform(3, 3) = 1.0;
  support.world_transform(0, 3) = 10.0;  // x position
  support.world_transform(1, 3) = 20.0;  // y position
  support.world_transform(2, 3) = 5.0;   // z position

  // Boundary polygon
  std::vector<Eigen::Vector2d> pts = {
      {0.0, 0.0}, {4.0, 0.0}, {4.0, 3.0}, {0.0, 3.0}};
  support.boundary = std::make_shared<Polygon2D>(pts);

  support.has_contact = true;
  support.contact_normalized.strategy =
      spatialblend::NormalizedPolygonPosition::Strategy::kMVC;
  support.contact_normalized.angle = 1.234;
  // Fractional radial_ratio: the old wire format cast this to int32 and lost it.
  support.contact_normalized.radial_ratio = 0.37;
  support.contact_normalized.mvc_weights = {0.1, 0.2, 0.3, 0.4};
  support.contact_normalized.height_ratio = 0.6;

  // Canonical interior chart coordinate.
  support.has_interior = true;
  support.interior.kind = NormalizedInterior::Kind::kMVC;
  support.interior.uvw = Eigen::Vector3d(0.5, -0.4, 0.0);
  support.interior.weights = {0.25, 0.25, 0.5};
  support.sub_region_name = "right_patch";
  // The source referent's own local-frame contact point, needed by
  // RelationSolver::DecodeTargetPoint to reconstruct the SOURCE actor's
  // support-relative (not center-relative) approach offset.
  support.contact_point_local = Eigen::Vector2d(1.5, -2.5);
  // The start of the actor's final geodesic leg (v1's approach direction).
  support.has_approach_origin = true;
  support.approach_origin_local = Eigen::Vector2d(-0.75, 3.25);
  support.look_at_vertex = 3;
  support.look_at_boundary_crossed = true;

  auto proto = ToProto(support);
  auto result = SupportFromProto(proto);

  // world_transform
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      EXPECT_DOUBLE_EQ(result.world_transform(row, col),
                        support.world_transform(row, col))
          << "Mismatch at (" << row << "," << col << ")";
    }
  }

  // boundary
  ASSERT_NE(result.boundary, nullptr);
  const auto& result_pts = result.boundary->GetPoints();
  ASSERT_EQ(result_pts.size(), pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    EXPECT_DOUBLE_EQ(result_pts[i].x(), pts[i].x());
    EXPECT_DOUBLE_EQ(result_pts[i].y(), pts[i].y());
  }

  EXPECT_TRUE(result.has_contact);
  EXPECT_EQ(result.contact_normalized.strategy,
            spatialblend::NormalizedPolygonPosition::Strategy::kMVC);
  EXPECT_DOUBLE_EQ(result.contact_normalized.angle, 1.234);
  EXPECT_DOUBLE_EQ(result.contact_normalized.radial_ratio, 0.37);
  ASSERT_EQ(result.contact_normalized.mvc_weights.size(), 4u);
  EXPECT_DOUBLE_EQ(result.contact_normalized.mvc_weights[2], 0.3);
  EXPECT_DOUBLE_EQ(result.contact_normalized.height_ratio, 0.6);

  EXPECT_TRUE(result.has_interior);
  EXPECT_EQ(result.interior.kind, NormalizedInterior::Kind::kMVC);
  EXPECT_DOUBLE_EQ(result.interior.uvw.x(), 0.5);
  EXPECT_DOUBLE_EQ(result.interior.uvw.y(), -0.4);
  ASSERT_EQ(result.interior.weights.size(), 3u);
  EXPECT_DOUBLE_EQ(result.interior.weights[2], 0.5);
  EXPECT_EQ(result.sub_region_name, "right_patch");
  EXPECT_DOUBLE_EQ(result.contact_point_local.x(), 1.5);
  EXPECT_DOUBLE_EQ(result.contact_point_local.y(), -2.5);
  EXPECT_TRUE(result.has_approach_origin);
  EXPECT_DOUBLE_EQ(result.approach_origin_local.x(), -0.75);
  EXPECT_DOUBLE_EQ(result.approach_origin_local.y(), 3.25);
  EXPECT_EQ(result.look_at_vertex, 3);
  EXPECT_TRUE(result.look_at_boundary_crossed);

  // An old sender omits it: absent, not a zero origin.
  Support old;
  old.has_contact = true;
  EXPECT_FALSE(SupportFromProto(ToProto(old)).has_approach_origin);
}

// NormalizedPolygonPosition's scalar members (strategy, angle,
// radial_ratio, height_ratio) previously lacked default member initializers,
// so a default-constructed Support::contact_normalized (has_contact == false,
// i.e. never actually grounded) held garbage stack bytes. A round trip
// through the wire proto must not surface that garbage — every scalar must
// come back at its documented default (see
// spatialblend/geometry/polygon_coordinate_mapper.h).
TEST(ProtoUtilTest, DefaultSupportRoundtripHasNoGarbageContactNormalized) {
  Support support;  // has_contact stays false; contact_normalized untouched

  auto proto = ToProto(support);
  auto result = SupportFromProto(proto);

  EXPECT_FALSE(result.has_contact);
  EXPECT_EQ(result.contact_normalized.strategy,
            spatialblend::NormalizedPolygonPosition::Strategy::kRadial);
  EXPECT_DOUBLE_EQ(result.contact_normalized.angle, 0.0);
  EXPECT_DOUBLE_EQ(result.contact_normalized.radial_ratio, 0.0);
  EXPECT_DOUBLE_EQ(result.contact_normalized.height_ratio, 0.0);
  EXPECT_TRUE(result.contact_normalized.mvc_weights.empty());
}

TEST(ProtoUtilTest, EmptyPayloadRoundtrip) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 2;

  auto proto = ToProto(update);
  auto result = FromProto(proto);

  EXPECT_EQ(result.source_space_id, 1);
  EXPECT_EQ(result.actor_shared_id, 2);
  EXPECT_TRUE(result.payloads.empty());
}

TEST(ProtoUtilTest, EncodedStatePreservation) {
  RelationPayload payload;
  payload.referent_shared_id = 1;

  // Use values that stress double precision
  Eigen::VectorXd state(6);
  state << 1.0 / 3.0, M_PI, std::sqrt(2.0),
      1e-15, 1e15, -0.0;
  payload.grounded_relation.encoded_state = state;

  auto proto = TryToProtoLegacy(payload);
  ASSERT_TRUE(proto.has_value());
  auto result = PayloadFromProto(*proto);

  ASSERT_EQ(result.grounded_relation.encoded_state.size(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(result.grounded_relation.encoded_state[i], state[i])
        << "Mismatch at index " << i;
  }
}

// Regression: ElementEventProto must carry the source handle. Dropping it left
// the receiver unable to record pending_share_source_ / create the ElementLinker
// link on resolve-on-share, so the temp duplicate survived (same shared id) and a
// cross-space referent resolved via shared-id to that temp at the origin — the
// clone realized at the room centre instead of the linked furniture.
TEST(ProtoUtilTest, ElementEventSourceHandleSurvivesRoundTrip) {
  ElementEvent event;
  event.type = ElementEvent::kAdded;
  event.shared_id = 76056;
  event.parent_shared_id = 1;
  event.tag = "object";
  event.user_data = "payload";
  event.source_space_id = 1;
  event.source_element_id = 42;

  auto result = FromProto(ToProto(event));

  EXPECT_EQ(result.source_space_id, 1);
  EXPECT_EQ(result.source_element_id, 42);
  EXPECT_EQ(result.shared_id, 76056);
  EXPECT_EQ(result.parent_shared_id, 1);
  EXPECT_EQ(result.tag, "object");
  EXPECT_EQ(result.user_data, "payload");
  EXPECT_EQ(result.type, ElementEvent::kAdded);
}

TEST(ProtoUtilTest, RelationEventRoundtrip) {
  RelationEvent ev;
  ev.actor_shared_id = 7;
  ev.anchor_name = "hand_r";
  ev.referent_shared_id = 42;
  ev.kernel_type = RelationKernelType::kAvatarAffordancePose;
  ev.from = RelationPhase::kActive;
  ev.to = RelationPhase::kContact;
  ev.timestamp = 3.5;

  auto proto = ToProto(ev);
  auto result = FromProto(proto);

  EXPECT_EQ(result.actor_shared_id, 7);
  EXPECT_EQ(result.anchor_name, "hand_r");
  EXPECT_EQ(result.referent_shared_id, 42);
  EXPECT_EQ(result.kernel_type, RelationKernelType::kAvatarAffordancePose);
  EXPECT_EQ(result.from, RelationPhase::kActive);
  EXPECT_EQ(result.to, RelationPhase::kContact);
  EXPECT_DOUBLE_EQ(result.timestamp, 3.5);
}

// Containment transfer (D5, ABI 4): one optional wire field, the identity of the
// transfer element only. The pose, chart and blend weight of ABI 3 are gone: the
// leaf is the receiver's scope and the empty-scope mapping maps the root.
// ABI 4.14: the per-anchor chart observations ride the update (field 22) and survive the bytes.
TEST(ProtoUtilTest, AnchorChartObservationsRoundtrip) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 7;
  AnchorChartObservation root, head;
  root.anchor_name = "root";
  root.scope_shared_id = 300;
  root.scope_source_space_id = 1;
  root.scope_source_element_id = 2;
  root.x = 0.3;
  root.y = -0.1;
  root.yaw = 0.4;
  head = root;
  head.anchor_name = "head";
  head.z = 1.58;
  head.yaw = -1.2;
  update.anchor_charts = {root, head};
  std::string bytes;
  ASSERT_TRUE(ToProto(update).SerializeToString(&bytes));
  proto::RelationUpdateProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  const RemoteRelationUpdate again = FromProto(parsed);
  ASSERT_EQ(again.anchor_charts.size(), 2u);
  EXPECT_TRUE(again.anchor_charts[0] == root);
  EXPECT_TRUE(again.anchor_charts[1] == head);
  // None: nothing on the wire.
  RemoteRelationUpdate none;
  EXPECT_EQ(ToProto(none).anchor_charts_size(), 0);
}

TEST(ProtoUtilTest, ContainmentObservationRoundtrip) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 7;
  ContainmentObservation c;
  c.scope_shared_id = 300;
  c.scope_source_space_id = 1;
  c.scope_source_element_id = 2;
  update.containment = c;

  const auto proto = ToProto(update);
  ASSERT_TRUE(proto.has_containment());
  const RemoteRelationUpdate result = FromProto(proto);
  ASSERT_TRUE(result.containment.has_value());
  EXPECT_TRUE(*result.containment == c);

  // Through the serialized bytes, as a real peer would receive it.
  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  proto::RelationUpdateProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  const RemoteRelationUpdate again = FromProto(parsed);
  ASSERT_TRUE(again.containment.has_value());
  EXPECT_TRUE(*again.containment == c);
}

// ABI 4: a source-app exception rides the update with an opaque payload, and is
// absent on the wire when the source raised none.
TEST(ProtoUtilTest, SourceExceptionRoundtripsOpaquePayload) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 7;
  update.has_exception = true;
  update.exception_id = 42;
  update.exception_payload = {0, 1, 255, 0, 7};  // NULs survive: opaque bytes
  const auto proto = ToProto(update);
  ASSERT_TRUE(proto.has_exception_id());
  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  proto::RelationUpdateProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  const RemoteRelationUpdate again = FromProto(parsed);
  EXPECT_TRUE(again.has_exception);
  EXPECT_EQ(again.exception_id, 42u);
  EXPECT_EQ(again.exception_payload, update.exception_payload);

  RemoteRelationUpdate none;
  none.source_space_id = 1;
  none.actor_shared_id = 7;
  const auto quiet = ToProto(none);
  EXPECT_FALSE(quiet.has_exception_id());
  EXPECT_FALSE(FromProto(quiet).has_exception);
}

TEST(ProtoUtilTest, NoContainmentStaysAbsentAndOldPeerDecodesAbsent) {
  RemoteRelationUpdate update;
  update.source_space_id = 1;
  update.actor_shared_id = 7;
  const auto proto = ToProto(update);
  EXPECT_FALSE(proto.has_containment());
  EXPECT_FALSE(FromProto(proto).containment.has_value());

  // An old peer neither sends nor reads field 13: take a new sender's bytes,
  // drop the field the way an old decoder drops an unknown one, and decode.
  RemoteRelationUpdate with;
  with.source_space_id = 1;
  with.actor_shared_id = 7;
  with.containment = ContainmentObservation{};
  auto sent = ToProto(with);
  ASSERT_TRUE(sent.has_containment());
  sent.clear_containment();
  EXPECT_FALSE(FromProto(sent).containment.has_value());
  // And the rest of the update survives untouched.
  EXPECT_EQ(FromProto(sent).actor_shared_id, 7);

  // Bytes from a sender that predates the field parse into the new message
  // with no containment: serialize a message that never set it.
  proto::RelationUpdateProto legacy;
  legacy.set_source_space_id(1);
  legacy.set_actor_shared_id(7);
  std::string bytes;
  ASSERT_TRUE(legacy.SerializeToString(&bytes));
  proto::RelationUpdateProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  EXPECT_FALSE(parsed.has_containment());
  EXPECT_FALSE(FromProto(parsed).containment.has_value());
}

// ABI 4.9: an unlink names the sharer's element in its own fields. A peer that
// predates the type reads it as REMOVED; with no source element that removes
// nothing, instead of deleting a same-numbered element of the receiver's own.
TEST(ProtoUtilTest, UnlinkedEventRoundTripsAndIsInertToAnOlderPeer) {
  ElementEvent event;
  event.type = ElementEvent::kUnlinked;
  event.shared_id = kInvalidSharedElementId;
  event.source_space_id = 3;
  event.source_element_id = 41;
  event.peer_space_id = 9;
  const proto::ElementEventProto wire = ToProto(event);
  EXPECT_EQ(wire.type(), proto::ElementEventProto::UNLINKED);
  EXPECT_EQ(wire.source_element_id(), 0);
  EXPECT_EQ(wire.source_space_id(), 0);
  const ElementEvent back = FromProto(wire);
  EXPECT_EQ(back.type, ElementEvent::kUnlinked);
  EXPECT_EQ(back.source_space_id, 3);
  EXPECT_EQ(back.source_element_id, 41);
  EXPECT_EQ(back.peer_space_id, 9);
}

}  // namespace
}  // namespace spatialblend
