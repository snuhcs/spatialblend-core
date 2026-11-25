// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/space/testing/ideal_driver.h
//
// Test support: a zero-lag, unbounded DRIVER for tests that assert the
// relation-preserving target itself, not how a body gets there. It owns the
// embodied pose (its own copy of nothing in core): after every proposal it
// publishes the proposed target as the new embodied pose through the sampling
// port, which is exactly what core's old direct-apply path did, so a test that
// continues from "where the avatar is" keeps its meaning. Bounded motion is the
// harness's and Unity's business (spatialblend/space/testing/test_driver.h).
#ifndef SPATIALBLEND_SPACE_IDEAL_DRIVER_H_
#define SPATIALBLEND_SPACE_IDEAL_DRIVER_H_

#include <Eigen/Dense>
#include <cstdint>
#include <memory>

#include "spatialblend/space/embodied_port.h"
#include "spatialblend/space/space.h"

namespace spatialblend {

// The next sequence for `actor`'s port: one past the sample it holds, so any
// number of test-side drivers can write one port without colliding.
inline uint64_t NextPortSequence(Space* space, spatialblend::entity_id_t actor) {
  EmbodiedSample s;
  space->SampleEmbodiedPose(actor, &s);
  return s.valid ? s.sequence + 1 : 1;
}

// Publishes `start` as the embodied pose of `actor` and makes the actor adopt
// every subsequent proposal at once. Several actors of one space may be
// followed. The returned handle keeps the follower registered; destroy it (or
// the space) to stop.
class IdealDriver {
 public:
  IdealDriver(Space* space, spatialblend::entity_id_t actor,
              const Eigen::Vector2d& start = Eigen::Vector2d::Zero(),
              double yaw = 0.0)
      : space_(space), actor_(actor) {
    Publish(start.x(), start.y(), yaw);
    Space* const s = space;
    const spatialblend::entity_id_t a = actor;
    observer_ = space->AddProposalObserver(
        [s, a](spatialblend::entity_id_t id, const ReceiverProposal& p) {
          if (id != a) return;
          EmbodiedPoseInput in;
          in.epoch = 1;
          in.yaw = p.yaw;
          if (p.IsAbsolute()) {
            // First proposal or a source exception: the proposal is the absolute
            // solve; put the body on it. The continuation target is not a walk.
            in.x = p.x;
            in.y = p.y;
            in.flags = kEmbodiedTeleported;
          } else if (p.status == ProposalStatus::kExact ||
                     p.status == ProposalStatus::kInexact) {
            in.x = p.x;
            in.y = p.y;
          } else {
            return;
          }
          s->PublishEmbodiedPose(a, in, 0.0, NextPortSequence(s, a));
        });
  }
  ~IdealDriver() { space_->RemoveProposalObserver(observer_); }
  IdealDriver(const IdealDriver&) = delete;
  IdealDriver& operator=(const IdealDriver&) = delete;

  // Place the body (a driver-side teleport): bumps nothing, just publishes.
  void Publish(double x, double y, double yaw) {
    EmbodiedPoseInput in;
    in.x = x;
    in.y = y;
    in.yaw = yaw;
    in.epoch = 1;
    space_->PublishEmbodiedPose(actor_, in, 0.0,
                                NextPortSequence(space_, actor_));
  }

  Eigen::Vector2d position() const {
    EmbodiedSample s;
    space_->SampleEmbodiedPose(actor_, &s);
    return Eigen::Vector2d(s.x, s.y);
  }
  double yaw() const {
    EmbodiedSample s;
    space_->SampleEmbodiedPose(actor_, &s);
    return s.yaw;
  }
  ReceiverProposal proposal() const {
    ReceiverProposal p;
    space_->ReadProposal(actor_, &p);
    return p;
  }

 private:
  Space* space_;
  spatialblend::entity_id_t actor_;
  uint64_t observer_ = 0;
};

// Fire-and-forget form for fixtures that build many actors in helper
// functions: publishes `start` and registers a zero-lag follower that lives as
// long as the space (the observer only ever runs inside the space's own
// solve, so it cannot outlive it). Publishing for a SOURCE actor is harmless:
// nothing is proposed for it and its tree pose is its own feed.
inline void FollowProposals(Space* space, spatialblend::entity_id_t actor,
                            const Eigen::Vector2d& start = Eigen::Vector2d::Zero(),
                            double yaw = 0.0) {
  EmbodiedPoseInput in;
  in.x = start.x();
  in.y = start.y();
  in.yaw = yaw;
  in.epoch = 1;
  space->PublishEmbodiedPose(actor, in, 0.0, NextPortSequence(space, actor));
  space->AddProposalObserver(
      [space, actor](spatialblend::entity_id_t id, const ReceiverProposal& p) {
        if (id != actor) return;
        EmbodiedPoseInput next;
        next.epoch = 1;
        next.yaw = p.yaw;
        if (p.IsAbsolute()) {
          next.x = p.x;
          next.y = p.y;
          next.flags = kEmbodiedTeleported;
        } else if (p.status == ProposalStatus::kExact ||
                   p.status == ProposalStatus::kInexact) {
          next.x = p.x;
          next.y = p.y;
        } else {
          return;
        }
        space->PublishEmbodiedPose(actor, next, 0.0,
                                   NextPortSequence(space, actor));
      });
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_SPACE_IDEAL_DRIVER_H_
