// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// spatialblend/model/element/affordance_pose.h
//
// An AFFORDANCE POSE of an object (affordance poses are part of the object's scope definition, solved
// by the shared solver one level down inside the object's spatial definition). It is carried by an interaction-role
// child of the object (GeometryRole::kInteractionRegion): the child's TAG names the pose ("stand", "seated",
// "stand-at-board", "write"), and the template says which body anchor it is a pose of and where that anchor is, in the
// frame of the object's part. The template holds NO body measurement: each side instantiates it for its own
// registered embodiment (space/affordance_pose_sd.h), so a 0.45 m chair and a 0.62 m chair, and two bodies of different
// size, each get their own instances.
#ifndef SPATIALBLEND_ELEMENT_AFFORDANCE_POSE_H_
#define SPATIALBLEND_ELEMENT_AFFORDANCE_POSE_H_

#include <string>

namespace spatialblend {

struct AffordancePoseTemplate {
  // The subject anchor this pose places (e.g. "pelvis", "hand").
  std::string subject_anchor;
  // Where the anchor is in the pose.
  //   kPart       the anchor's registered contact lies ON the named part of the parent object's spatial definition
  //               (a SubRegion: the seat, the board face). The support is that part's surface, in the part frame.
  //   kActorRoot  the anchor at its registered STANDING offset over the actor's root (the embodiment's standing
  //               sit-bone height for the pelvis). The support is that one point.
  enum class Support { kPart = 0, kActorRoot = 1 };
  Support support = Support::kPart;
  // kPart: the SubRegion name of the parent's spatial definition. Ignored for kActorRoot.
  std::string part;
};

inline bool operator==(const AffordancePoseTemplate& a, const AffordancePoseTemplate& b) {
  return a.subject_anchor == b.subject_anchor && a.support == b.support && a.part == b.part;
}

}  // namespace spatialblend

#endif  // SPATIALBLEND_ELEMENT_AFFORDANCE_POSE_H_
