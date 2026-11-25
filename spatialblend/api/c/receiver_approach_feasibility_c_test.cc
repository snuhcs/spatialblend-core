// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "spatialblend/api/c/spatialblend.h"
#include "spatialblend/api/c/spatialblend_receiver_approach_context.h"
#include "spatialblend/api/c/spatialblend_receiver_approach_feasibility.h"
#include "spatialblend/api/c/spatialblend_receiver_input.h"

namespace {
const uint8_t* B(const char* s) { return reinterpret_cast<const uint8_t*>(s); }
void I(double* m, int n) { for (int r=0;r<n;++r) for(int c=0;c<n;++c) m[r*n+c]=r==c; }
sb_receiver_anchor_input_t A(const char* n,double x) { sb_receiver_anchor_input_t a{}; a.name_utf8=B(n); a.name_bytes=std::strlen(n); a.available=1; I(a.world_pose_row_major,4); a.world_pose_row_major[3]=x; return a; }

class CApproachFeasibility : public ::testing::Test {
 protected:
  void SetUp() override {
    s=sb_create_session(); sp=sb_add_space(s); e=sb_add_element(sp,0); double room[]={-5,-3,5,-3,5,3,-5,3}; sb_set_element_polygon(sp,e,SB_GEOMETRY_NAVIGABLE,room,4);
    a=sb_create_actor(sp); sb_add_anchor(sp,a,"root",0,0,0,nullptr); sb_add_anchor(sp,a,"pelvis",0,0,0,nullptr); sb_set_actor_element(sp,a,e);
    sb_receiver_input_feature_t f{}; sb_get_receiver_input_feature(&f,sizeof(f)); ASSERT_EQ(SB_RICH_RESULT_OK,sb_begin_receiver_input(sp,a,1,B("root"),4,3,5,&epoch));
    anchors={A("root",-3),A("pelvis",-3)}; frame={}; frame.struct_size=sizeof(frame); frame.feature_version=1; frame.anchor_count=2; frame.anchor_stride=sizeof(anchors[0]); frame.epoch=epoch; frame.sequence=1; frame.monotonic_clock_id=f.monotonic_clock_id; frame.monotonic_clock_epoch=f.monotonic_clock_epoch; frame.observed_at_ns=sb_read_rich_monotonic_ns(); frame.valid_until_ns=frame.observed_at_ns+f.max_age_ns; frame.anchors=anchors.data(); ASSERT_EQ(SB_RICH_RESULT_OK,sb_submit_receiver_input(sp,a,&frame,sizeof(frame)));
    navv={-5,-3,5,-3,5,3,-5,3}; obv={20,20,21,20,20,21}; supv={-.5,-.5,.5,-.5,.5,.5,-.5,.5}; nav=P(100,navv); obs=P(101,obv); sup=P(102,supv);
  }
  void TearDown() override { if(s) sb_destroy_session(s); }
  sb_receiver_approach_polygon_t P(uint64_t id,const std::vector<double>& v) { sb_receiver_approach_polygon_t p{}; p.struct_size=sizeof(p);p.vertex_count=v.size()/2;p.nav_element_id=id;p.region_id=30;p.coordinate_convention=1;p.affine_convention=1;I(p.local_to_space_row_major,3);p.vertices_xy=v.data();return p; }
  sb_receiver_approach_context_registration_t R() { sb_receiver_approach_context_registration_t r{};r.struct_size=sizeof(r);r.feature_version=1;r.expected_receiver_input_epoch=epoch;r.expected_receiver_input_sequence=1;r.geometry_revision=11;r.requested_valid_until_ns=frame.observed_at_ns+500000000;r.navigation_root_utf8=B("root");r.navigation_root_bytes=4;r.coordinate_convention=1;I(r.navigation_root_pose_row_major,4);r.navigation_root_pose_row_major[3]=-3;r.navigable_regions=&nav;r.navigable_region_count=1;r.obstacle_regions=&obs;r.obstacle_region_count=1;r.footprint_radius_m=.25;r.required_clearance_m=.15;r.destination.struct_size=sizeof(r.destination);r.destination.space=sp;r.destination.region_id=30;r.destination.object=e;r.destination.affordance_id=50;r.destination.support_revision=12;r.destination.local_support=sup;r.destination.coordinate_convention=1;I(r.destination.safe_approach_pose_row_major,4);r.destination.safe_approach_pose_row_major[3]=3;return r; }
  sb_session_t s=0;sb_space_t sp=0;sb_element_t e=0;sb_actor_t a=0;uint64_t epoch=0;sb_receiver_input_frame_t frame{};std::vector<sb_receiver_anchor_input_t> anchors;std::vector<double> navv,obv,supv;sb_receiver_approach_polygon_t nav{},obs{},sup{};
};

TEST(CApproachFeasibilityLayout, NegotiatesWithoutChangingExistingFeatures) { sb_receiver_approach_feasibility_feature_t f{}; ASSERT_EQ(sizeof(f),sb_get_receiver_approach_feasibility_feature(&f,sizeof(f))); EXPECT_EQ(1u,f.feature_version); EXPECT_EQ(sizeof(sb_receiver_approach_feasibility_header_t),f.header_size); EXPECT_EQ(16384u,f.max_route_points); EXPECT_EQ(27u,sb_get_receiver_approach_feasibility_field_count(2)); EXPECT_EQ(UINT32_MAX,sb_get_receiver_approach_feasibility_field_offset(2,27)); }

TEST_F(CApproachFeasibility, C3cIngressMutationAndTwoPassCopyRemainPinned) {
  auto r=R(); uint64_t context=0; ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK,sb_register_receiver_approach_context(s,sp,a,1,&r,sizeof(r),&context));
  navv[0]=999; r.navigation_root_utf8=nullptr;  // caller memory is now irrelevant.
  sb_receiver_approach_feasibility_header_t h{}; ASSERT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL,sb_copy_receiver_approach_feasibility(s,sp,a,1,context,0,nullptr,0,&h,sizeof(h))); ASSERT_GE(h.route_point_count,2u); EXPECT_EQ(-3,h.start_x); EXPECT_EQ(3,h.safe_approach_x); EXPECT_EQ(context,h.context_revision);
  std::vector<double> route(h.route_point_count*2,999); const auto calculation=h.calculation_revision; ASSERT_NE(0u,calculation); std::vector<double> short_route((h.route_point_count-1)*2,1234); ASSERT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL,sb_copy_receiver_approach_feasibility(s,sp,a,1,context,calculation,short_route.data(),h.route_point_count-1,&h,sizeof(h))); for(double v:short_route) EXPECT_EQ(1234,v); ASSERT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_OK,sb_copy_receiver_approach_feasibility(s,sp,a,1,context,calculation,route.data(),h.route_point_count,&h,sizeof(h))); EXPECT_EQ(-3,route.front()); EXPECT_EQ(3,route[route.size()-2]); EXPECT_EQ(11u,h.geometry_revision); EXPECT_EQ(12u,h.support_revision); EXPECT_EQ(epoch,h.receiver_input_epoch); EXPECT_EQ(1u,h.receiver_input_sequence);
}

TEST_F(CApproachFeasibility, VersionCapacityAndReplacementNeverPrefixSucceed) {
  auto r=R();uint64_t context=0;ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK,sb_register_receiver_approach_context(s,sp,a,1,&r,sizeof(r),&context)); sb_receiver_approach_feasibility_header_t h{}; double sentinel[2]={77,77}; EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_UNSUPPORTED_VERSION,sb_copy_receiver_approach_feasibility(s,sp,a,2,context,0,sentinel,1,&h,sizeof(h))); ASSERT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL,sb_copy_receiver_approach_feasibility(s,sp,a,1,context,0,sentinel,0,&h,sizeof(h))); EXPECT_EQ(77,sentinel[0]); r.expected_registration_generation=1;r.geometry_revision=13;uint64_t replacement=0;ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK,sb_register_receiver_approach_context(s,sp,a,1,&r,sizeof(r),&replacement)); EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_REPLACED,sb_copy_receiver_approach_feasibility(s,sp,a,1,context,h.calculation_revision,nullptr,0,&h,sizeof(h))); }
TEST_F(CApproachFeasibility, ReceiverInputSequenceStalesFrozenCalculation) { auto r=R();uint64_t c=0;ASSERT_EQ(SB_RECEIVER_APPROACH_CONTEXT_OK,sb_register_receiver_approach_context(s,sp,a,1,&r,sizeof(r),&c));sb_receiver_approach_feasibility_header_t h{};ASSERT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_BUFFER_TOO_SMALL,sb_copy_receiver_approach_feasibility(s,sp,a,1,c,0,nullptr,0,&h,sizeof(h)));frame.sequence=2;frame.observed_at_ns=sb_read_rich_monotonic_ns();frame.valid_until_ns=frame.observed_at_ns+1000000000;ASSERT_EQ(SB_RICH_RESULT_OK,sb_submit_receiver_input(sp,a,&frame,sizeof(frame)));EXPECT_EQ(SB_RECEIVER_APPROACH_FEASIBILITY_STALE,sb_copy_receiver_approach_feasibility(s,sp,a,1,c,h.calculation_revision,nullptr,0,&h,sizeof(h))); }
}  // namespace
