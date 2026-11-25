// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_TYPE_H_
#define SPATIALBLEND_C_SPATIALBLEND_TYPE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t sb_session_t;
typedef int64_t sb_space_t;
typedef int64_t sb_actor_t;
typedef int64_t sb_element_t;
typedef int64_t sb_handle_t;
// Opaque, process-local handle of one tunneled core link (ABI 4.5): the byte
// stream between this core and one remote peer, carried by the application.
typedef int64_t sb_tunnel_t;
// Opaque, process-local capability for a native support-correspondence batch.
// It is not a network or shared identity.
typedef int64_t sb_support_correspondence_registration_t;
// Opaque, process-local capability for an atomic observed-source setup.  It is not
// a network identity and is meaningful only with the Space that created it.
typedef int64_t sb_observed_source_setup_t;

// An explicit element correspondence. Both endpoints must be existing
// elements in the same sb_session_t when registered.
typedef struct {
  sb_space_t source_space;
  sb_element_t source_element;
  sb_space_t target_space;
  sb_element_t target_element;
} sb_support_correspondence_t;

typedef enum {
  SB_GEOMETRY_UNSPECIFIED = 0,
  SB_GEOMETRY_NAVIGABLE = 1,
  SB_GEOMETRY_CONTAINMENT = 2,
  SB_GEOMETRY_OBSTACLE = 3,
  SB_GEOMETRY_SURFACE = 4,
  SB_GEOMETRY_INTERACTION_REGION = 5
} sb_geometry_role_t;

typedef enum {
  SB_KERNEL_SPACING = 0,
  SB_KERNEL_FACING = 1,
  SB_KERNEL_AFFORDANCE_POSE = 2
} sb_kernel_type_t;

/**
 * Primitive collider shapes a client can hand the core to auto-derive an element's
 * Containment "envelope" (its enclosing footprint) when no explicit boundary is
 * given. The core converts the primitive to a footprint; the client only supplies
 * params (see sb_set_element_collider).
 */
typedef enum {
  SB_COLLIDER_BOX = 0,      /* params: cx,cy,cz, sx,sy,sz (center, FULL size) */
  SB_COLLIDER_SPHERE = 1,   /* params: cx,cy,cz, r */
  SB_COLLIDER_CAPSULE = 2   /* params: cx,cy,cz, r, height, axis(0=X,1=Y,2=Z) */
} sb_collider_type_t;

/* ---------------------------------------------------------------------------
 * Orientation of an actor beyond yaw (ABI 3.1, unchanged).
 *
 *   SB_ORIENTATION_YAW             yaw only
 *   SB_ORIENTATION_YAW_PITCH       yaw + pitch (head pitch)
 *   SB_ORIENTATION_YAW_PITCH_ROLL  full 3D rotation as yaw, pitch, roll
 * Pitch and roll are carried from the source to the receiver's proposal of the SAME
 * anchor AS A PASS-THROUGH (the root: the actor's sb_update_actor_pose orientation;
 * another anchor: its own, from its bone pose): core never
 * solves them and they take part in no relation. The
 * facing yaw stays the solved yaw. Radians, Z-up; a field that the type does not
 * carry is 0.
 * ------------------------------------------------------------------------- */
typedef enum {
  SB_ORIENTATION_YAW = 0,
  SB_ORIENTATION_YAW_PITCH = 1,
  SB_ORIENTATION_YAW_PITCH_ROLL = 2
} sb_orientation_type_t;

typedef struct {
  uint32_t struct_size; /* sizeof(sb_orientation_t) */
  int32_t type;         /* sb_orientation_type_t */
  double yaw;
  double pitch;
  double roll;
} sb_orientation_t;

/* ---------------------------------------------------------------------------
 * Per-anchor sampling ports and anchor pose proposals (ABI 4).
 *
 * Single ownership: the receiver's embodied poses belong to the DRIVER (Unity,
 * or the app). The driver is the only writer of each anchor's port
 * (sb_publish_anchor_pose); core only samples the latest snapshot when it solves.
 * Core keeps no pose of its own, never writes, integrates or extrapolates a
 * sample, and never falls back to a pose when none has been published.
 *
 * Core PROPOSES (sb_read_proposals): ONE pose proposal for every relation-bearing
 * anchor of the receiver actor (root, pelvis, head, ...), all from one solve. The
 * driver (avatar controller / IK) APPLIES them within its own kinetic bounds
 * (speed, turn rate, pacing) and publishes the resulting poses back through the
 * ports. Containment, seat and placement are not side channels: they are the
 * proposals of the anchors that carry them and cause bits. Core does no pacing,
 * slewing, fading or rate limiting.
 * ------------------------------------------------------------------------- */

/* sb_anchor_pose_t::flags */
#define SB_EMBODIED_TELEPORTED 0x1u /* the driver placed the body (not walked) */

/** The driver's pose of one anchor, receiver space, Z-up, radians. */
typedef struct {
  uint32_t struct_size; /* sizeof(sb_anchor_pose_t) */
  uint32_t flags;       /* SB_EMBODIED_* */
  double x;
  double y;
  double z; /* the anchor's own height; the planar root may leave it 0 */
  sb_orientation_t orientation;
  /* Bumps whenever the driver teleports, respawns or resets this anchor. A new
   * epoch restarts `sequence`; an older epoch is rejected as stale. */
  uint64_t epoch;
} sb_anchor_pose_t;

/* ---------------------------------------------------------------------------
 * Embodiment registration (ABI 4.7).
 * ------------------------------------------------------------------------- */

/* sb_embodiment anchor families (sb_kernel_type_t as bits). */
#define SB_FAMILY_SPACING (1u << SB_KERNEL_SPACING)
#define SB_FAMILY_FACING (1u << SB_KERNEL_FACING)
#define SB_FAMILY_AFFORDANCE_POSE (1u << SB_KERNEL_AFFORDANCE_POSE)

/** The pose of one bone: bone-local -> world, Z-up right-handed, rotation a quaternion (x, y, z, w). */
typedef struct {
  uint32_t struct_size; /* sizeof(sb_bone_pose_t) */
  uint32_t flags;       /* SB_EMBODIED_* */
  double position[3];
  double rotation[4];   /* x, y, z, w; normalised by the core (zero length is rejected) */
  uint64_t epoch;       /* as sb_anchor_pose_t::epoch */
} sb_bone_pose_t;

typedef struct {
  const char* bone_name;
  sb_bone_pose_t pose;
} sb_named_bone_pose_t;

/* Surface chart kinds of a part (sb_set_sub_region_chart; ABI 4.10). */
#define SB_SURFACE_CHART_RADIAL 0        /* the radial chart of every part before 4.10; seats, stools, sofas */
#define SB_SURFACE_CHART_FRAME_STRETCH 1 /* rectangle (u,v) preserved: per-axis scaling */
#define SB_SURFACE_CHART_FRAME_CONTAIN 2 /* rectangle (u,v) preserved on the largest centred same-aspect rectangle (letterbox) */

/** A point of a part in chart coordinates (sb_part_chart_encode / _decode), ABI 4.10. It travels between spaces
 *  (a stroke point, a pointer) and is read on the matched part of the receiver; the hand/pelvis contact of the
 *  relation path goes through the same mapping. */
typedef struct {
  uint32_t struct_size;   /* sizeof(sb_part_chart_coord_t) */
  int32_t chart;          /* SB_SURFACE_CHART_*: the SOURCE part's kind; the receiver's part must declare the same */
  double normal_m;        /* metric offset along the part normal (+z of the part frame); both kinds */
  /* frame charts: normalized position in the source rectangle and the source extent (the letterbox needs its aspect) */
  double u, v;
  double src_width_m, src_height_m;
  /* radial chart: angle (rad, from the part +y) and radial ratio in [0,1] about the part's chart origin */
  double angle, radial_ratio;
} sb_part_chart_coord_t;

/** One joint of the registered normalized kinematic chain, as read back (ABI 4.7). */
typedef struct {
  uint32_t struct_size;       /* sizeof(sb_chain_joint_t) */
  int32_t parent;             /* joint index, -1 for a root */
  double rest_position[3];    /* model space, canonical axes (z up, x forward, y left), metres */
  double rest_rotation[4];    /* joint frame -> model space, quaternion x, y, z, w */
  double segment_length_m;    /* |rest_position - parent rest_position|; 0 for a root (derived, not stored) */
  uint32_t limit_mask;        /* bit 2*i: limit_min[i] present; bit 2*i+1: limit_max[i] present (axis 0 = x roll, 1 = y pitch, 2 = z yaw) */
  double limit_min[3];        /* radians; 0 where absent */
  double limit_max[3];
} sb_chain_joint_t;

typedef enum {
  /* Nothing proposed yet, or the source released every relation. */
  SB_PROPOSAL_NONE = 0,
  /* The driver has not published a pose for this anchor (or it is invalid): core
   * could not solve and does NOT substitute a pose. Target fields are 0. */
  SB_PROPOSAL_NO_SAMPLE = 1,
  /* The target is an exact minimiser of the weighted relation objective (for an
   * AffordancePose anchor: the mapped contact). Whether the driver has ARRIVED is
   * SB_ANCHOR_PROPOSAL_FLAG_ARRIVED, not this status. */
  SB_PROPOSAL_EXACT = 2,
  /* Best minimiser found, residual above epsilon. */
  SB_PROPOSAL_INEXACT = 3,
  /* 4 (HOLD) is retired: there is no infeasible-stay
   * status. A referent the receiver cannot map drops; with none left the status is NONE. */
  /* The solve hit its loop guard. Not a minimiser; pose fields are diagnostic. */
  SB_PROPOSAL_NOT_CONVERGED = 5
} sb_proposal_status_t;

/* Scene mutator status (sb_set_element_transform / _polygon / _box return it;
 * sb_get_scene_status reports the most recent one). A change that is not OK is
 * NOT registered. The containment layout must be nested and non-overlapping:
 * sibling elements may touch but not overlap, and a Containment/Navigable child
 * lies inside its Containment/Navigable parent, or sticks out of it at any
 * distance, which extends the parent's scope to include it, bridged across any
 * gap (the extension is rebuilt in the background). An extending child
 * must not overlap any footprint outside its own line (OVERLAP). */
typedef enum {
  SB_SCENE_OK = 0,
  SB_SCENE_INVALID_ARGUMENT = 1,
  SB_SCENE_CONTAINMENT_OVERLAP = 2,        /* two sibling elements' footprints overlap (positive area beyond the data precision; touching edges are fine) */
  SB_SCENE_CONTAINMENT_ESCAPES_PARENT = 3  /* no longer produced: a child region at any distance extends its parent's scope (kept for ABI) */
} sb_scene_status_t;

/* sb_anchor_proposal_t::cause bits: why THIS proposal differs from the previous
 * one. The live target is the PURSUED target. The placed destination (the chart image of the source pose,
 * which may jump with no cause bit) is used only for the first proposal and a source exception. Pacing and
 * body clearance are the driver's. So PROGRESS, CHART_SNAP and NO_REQUEST are set; NO_REQUEST follows the
 * event model (see its define). */
#define SB_PROPOSAL_CAUSE_FIRST_PROPOSAL 0x001u
#define SB_PROPOSAL_CAUSE_WEIGHT_SWITCH 0x002u /* the driving (max-weight) relation changed, or the softmax weights moved S0
                                         beyond what the approach points' own motion did */
#define SB_PROPOSAL_CAUSE_REFERENT_SET_CHANGE 0x004u
#define SB_PROPOSAL_CAUSE_SOURCE_EXCEPTION 0x010u /* the source app raised an exception (a teleport, the
                                                     first join, ...): the proposals are the absolute solve */
#define SB_PROPOSAL_CAUSE_CONTAIN_ENTER 0x020u
#define SB_PROPOSAL_CAUSE_CONTAIN_EXIT 0x040u
#define SB_PROPOSAL_CAUSE_SEAT_ENGAGE 0x080u  /* an AffordancePose anchor touched its surface */
#define SB_PROPOSAL_CAUSE_SEAT_RELEASE 0x100u
#define SB_PROPOSAL_CAUSE_EMBODIED_RESET 0x200u /* driver epoch change / teleport */
#define SB_PROPOSAL_CAUSE_FACING_ROOT_SWITCH 0x400u /* the Facing +-theta sign root changed, or a Facing look-at contact swapped silhouette vertex (the least-angle rule's own discontinuity) */
#define SB_PROPOSAL_CAUSE_STATUS_CHANGE 0x800u /* nothing proposed <-> a target */
#define SB_PROPOSAL_CAUSE_SCENE_CHANGE 0x1000u /* the receiver's own scene/geometry changed
                                                  (spatial-definition, structure, tag, anchor and
                                                  non-actor element-transform edits; an ACTOR's own
                                                  pose, the avatar mirror, is excluded: it moves the
                                                  tree every frame) */
#define SB_PROPOSAL_CAUSE_CLAMP_BRANCH 0x2000u /* the free-space clamp of S0 (NearestFree) moved to another
                                                  constraint branch: another obstacle, wall or footprint
                                                  edge is now the nearest, across the medial axis of the
                                                  free space (continuous inside a branch), or the set of
                                                  exempt footprints changed (the source engaged or
                                                  released a referent). */
#define SB_PROPOSAL_CAUSE_DECODE_BRANCH 0x4000u /* a driving referent's approach-point decode changed rule
                                                   (a stand-off root blocked / unblocked by an obstacle or
                                                   the room wall); continuous inside one rule */
#define SB_PROPOSAL_CAUSE_PROGRESS 0x8000u /* a spacing term closed d(sample)/n of its own gap.
                                              The harness recomputes that quotient from the recorded
                                              count and the sample distance and checks the proposal step. */
#define SB_PROPOSAL_CAUSE_FACING_TIE 0x40000u /* the Facing bearings' circular resultant is within the error
                                                 interval of its own computation (a tie of opposing bearings
                                                 the arithmetic cannot resolve): the yaw is the middle nearest
                                                 the embodied yaw; or the path from the previous proposal passes
                                                 a look-at point within the data's precision (a pole of its
                                                 bearing): the yaw may differ from the previous update's by the
                                                 pass alone. */
#define SB_PROPOSAL_CAUSE_FOLD 0x80000u /* ABI 4.11: an anchor's inner solve (its object's affordance poses)
   answered on another branch of a weighted pose term's distance (that pose's nearest support feature changed)
   with the same driving pose: the continuation crossed a fold of the objective. */
#define SB_PROPOSAL_CAUSE_NO_REQUEST 0x20000u /* no relation issues a request (a static, receding or
                                                 non-closing source has an infinite update count).
                                                 Event model: with no request core
                                                 publishes no new set (the revision does not advance),
                                                 and the driver continues its pending target until
                                                 arrival; no request is neither a sampled-position hold
                                                 nor a release. A set that is published for another
                                                 reason (another anchor's request, a status or scene
                                                 change) and carries this bit holds the anchor's
                                                 RETAINED pending target, not its sample. */
#define SB_PROPOSAL_CAUSE_CHART_SNAP 0x10000u /* the source is inside its referent or at contact (n = 1)
                                                 and the engaged relations carry all the weight: the
                                                 proposal is their chart-mapped pose, the minimiser of J.
                                                 With weight on any other relation the engaged referent
                                                 is one term of J (PROGRESS), not a snap. */

/* sb_anchor_proposal_t::flags */
#define SB_ANCHOR_PROPOSAL_FLAG_YAW_UNSPECIFIED 0x1u /* nothing authors this anchor's yaw (no Facing
   referent): orientation.yaw echoes the sampled yaw and the driver faces its direction of motion, as
   v1 did (yaw followed the path tangent). */
#define SB_ANCHOR_PROPOSAL_FLAG_ENGAGED 0x2u /* the relation's geometric condition holds: the root is
   on / touches a referent, or an AffordancePose anchor touches its surface. No hysteresis. */
#define SB_ANCHOR_PROPOSAL_FLAG_SAMPLED 0x4u /* this anchor's own sample was published */
#define SB_ANCHOR_PROPOSAL_FLAG_ARRIVED 0x8u /* the SAMPLED anchor already satisfies the relation
   (root: the Spacing contact the source holds is satisfied at the sample, within the solver's declared
   tolerance in metres, and with none held every positive-weight root relation is; AffordancePose: it
   touches the surface; Facing: its SIGNED angular error is within v1's angle, 1e-6 degrees): the target
   app has arrived. */
#define SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED 0x10u /* ABI 4.1 (NOT SET by the product,
   which publishes a no-request anchor as a position with cause NO_REQUEST; a no-request update
   publishes no new set, and a set published for another reason carries the anchor's retained pending target): core
   publishes NO position target
   for this anchor (the receiver holds: the source is static, so no relation issues a request, n =
   infinity; v1 sent nothing). x / y / z echo the sampled position and are NOT a target: the driver
   keeps its own position and applies no goal-seeking toward them. Analogous to YAW_UNSPECIFIED. Status
   NONE (release) is a different signal and is unchanged; the driver reads this flag, never the cause. */
#define SB_ANCHOR_PROPOSAL_FLAG_INNER 0x40u /* ABI 4.11: this anchor's proposal is its INNER
   solve: the anchor's position among its object's affordance poses inside the object's spatial definition (the
   source anchor is in the SD and not in contact), by the same relation solve as the root's, one level down.
   Position only: YAW_UNSPECIFIED is set. Details: sb_read_anchor_solve. */
#define SB_ANCHOR_PROPOSAL_FLAG_FACING_YAW 0x80u /* ABI 4.11: the anchor is ENGAGED (seated) or INNER and the root
   has a Facing yaw: sb_anchor_solve_t.facing_yaw is the direction the head and upper body follow (core's Facing,
   not the user's raw head yaw). */
#define SB_ANCHOR_PROPOSAL_FLAG_TRANSFER 0x20u /* ABI 4.3, root anchor only: this update's root proposal
   IS the transfer mapping. Set exactly when the source root is inside a transfer leaf (the
   source-side point test), this space has that leaf's counterpart (same shared id, itself declared
   a transfer leaf), AND the two shapes are congruent under a rigid transform at this update (existing
   geometry epsilon). Recomputed on every update, stored nowhere: a counterpart that appears late or a
   geometry edit is picked up on the next update, and a missing, undeclared or non-congruent one
   leaves it clear. CONTAIN_ENTER / CONTAIN_EXIT stay as the
   cause bits of the jump update. It replaces the contained event: an app derives enter and leave
   from the flag's frame-to-frame change, and the representation switch reads it together with
   ARRIVED of the same anchor (the realised body, not the proposal). */

/**
 * The proposal of ONE anchor. Plain data; set struct_size = sizeof(sb_anchor_proposal_t)
 * on the array elements you pass to sb_read_proposals.
 *
 * x, y, z, orientation: the proposed pose, receiver space. orientation.yaw is the
 * solved yaw; pitch and roll are the source's pass-through of this anchor's own orientation.
 *
 * err_* are |R_tgt - R_src| (R units, max over the used relations of the family) at
 * the TARGET and at the SAMPLED anchor pose (the relation error the body has now);
 * -1 = no relation of that family on this anchor. err_scale_m converts the
 * spacing R to metres (the scope diameter D of the heaviest spacing term).
 */
typedef struct {
  uint32_t struct_size;
  uint32_t anchor_id; /* sb_get_anchor_id: 0 is the actor's root anchor */
  int32_t status;     /* sb_proposal_status_t */
  uint32_t flags;     /* SB_ANCHOR_PROPOSAL_FLAG_* */
  uint32_t cause;     /* SB_PROPOSAL_CAUSE_* bitmask */
  uint32_t reserved;  /* 0 */
  double x;
  double y;
  double z;
  sb_orientation_t orientation;
  double err_spacing_target;
  double err_spacing_sample;
  double err_facing_target;
  double err_facing_sample;
  double err_pose_target;
  double err_pose_sample;
  double err_scale_m;
  int64_t referent_shared_id; /* the driving referent of this anchor's solve (0 = none) */
} sb_anchor_proposal_t;

/**
 * ABI 4.11: the inner level of ONE anchor's proposal (sb_read_anchor_solve). Plain data; set struct_size.
 *
 * flags mirrors the anchor's SB_ANCHOR_PROPOSAL_FLAG_* (INNER, FACING_YAW). For an INNER proposal: the object whose
 * spatial definition holds the poses, the receiver SD's scale D (the diameter of its region volume, metres), J at the
 * answer (metres^2), the solve's iterations and the tag of the driving pose (the largest Progressive weight; NUL
 * terminated, truncated to 31 bytes). facing_yaw (radians) is valid with FACING_YAW.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t anchor_id;
  uint32_t flags;      /* SB_ANCHOR_PROPOSAL_FLAG_* of the anchor's proposal */
  int32_t iterations;
  int64_t object_shared_id;
  double sd_scale_m;
  double objective_j;
  double facing_yaw;
  char driving_pose_tag[32];
} sb_anchor_solve_t;

/**
 * ABI 4.12 (the proposal carries the turn, not only an absolute yaw): the yaw turn of ONE anchor's latest
 * proposal (sb_read_anchor_turn). Plain data; set struct_size.
 *
 * yaw_turn (radians) is the signed, unwrapped turn from the anchor's sampled yaw (the yaw the proposal was computed
 * from) that the proposal's yaw is: yaw = sampled yaw + yaw_turn, modulo 2 pi. A driver that turns by yaw_turn instead
 * of the shortest arc to yaw keeps the requested direction.
 *  - pursuing (a Facing steps by its ratio): the turn the relations request, the branch nearest the source's own
 *    remaining turn, which may exceed pi (a source turning away the long way round); a receiver a little past its
 *    look-at turns back the short way;
 *  - arriving (every Facing at n <= 1): the whole remaining turn to the yaw, the shortest arc;
 *  - the first proposal, a placement (a source exception) or an engaged pose: the shortest arc to the yaw;
 *  - a hold (no valid count), an unspecified yaw (YAW_UNSPECIFIED) or no referent: 0.
 * yaw repeats the proposal's orientation.yaw; flags mirrors the anchor's SB_ANCHOR_PROPOSAL_FLAG_*.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t anchor_id;
  uint32_t flags;     /* SB_ANCHOR_PROPOSAL_FLAG_* of the anchor's proposal */
  uint32_t reserved;  /* 0 */
  double yaw;         /* the proposal's yaw (radians) */
  double yaw_turn;    /* the signed, unwrapped turn from the sampled yaw (radians) */
} sb_anchor_turn_t;

/** sb_anchor_chart_t.flags */
#define SB_ANCHOR_CHART_FLAG_VERIFIED (1u << 0) /* the source's footprint of the leaf rode the update and is congruent with
                                                   this space's counterpart (the root transfer flag's test) */

/**
 * ABI 4.14: the replay through the chart of ONE anchor, a separate output beside the anchor's
 * proposal (sb_read_anchor_chart). Core computes each family's output and does not choose between them; the driver
 * composes the Spacing position, the Facing yaw/pitch and this transform. Present while the source anchor is inside a
 * containment-transfer leaf whose counterpart this space declares a transfer leaf: the source anchor's OWN sample in
 * the leaf's frame, mapped through the counterpart (receiver space, Z-up), with the source's pitch and roll as in the
 * proposal. Belongs to the same proposal set (revision) as the anchor's proposal.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t anchor_id;
  uint32_t flags;         /* SB_ANCHOR_CHART_FLAG_* */
  uint32_t reserved;      /* 0 */
  int64_t leaf_shared_id; /* the transfer leaf (shared id) */
  double x, y, z;         /* the mapped position (receiver space, Z-up, metres) */
  sb_orientation_t orientation; /* yaw mapped through the leaf; pitch / roll the source's pass-through */
  uint64_t revision;      /* the proposal set this belongs to (sb_proposal_set_t.revision) */
} sb_anchor_chart_t;

/**
 * The header of one proposal set (sb_read_proposals): what the solve answered and the
 * samples it was computed from. sample_age_s is the age of the ROOT sample when the
 * solve ran, measured on core's steady clock from the driver's Publish call; a large age
 * means the driver is not publishing. sample_sequence / sample_epoch identify the root
 * sample so the driver can tell which of its own poses the set answers.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t anchor_count;       /* proposals written (min(total_anchor_count, cap)) */
  uint32_t total_anchor_count; /* relation-bearing anchors in the set, the root first */
  int32_t sample_valid;        /* the root sample */
  uint64_t revision;           /* increments with every published set */
  int64_t source_sequence;     /* the source update this answers (0 = hand-built) */
  uint64_t sample_sequence;
  uint64_t sample_epoch;
  double sample_timestamp_s;   /* the driver's own timestamp, echoed */
  double sample_age_s;
} sb_proposal_set_t;

/* ---------------------------------------------------------------------------
 * Actor events (sb_poll_actor_event): a source-app exception, and core's request
 * to a target app that has not arrived to teleport.
 * ------------------------------------------------------------------------- */
#define SB_EXCEPTION_PAYLOAD_MAX 256u /* capacity of the opaque blob, not a tuning value */
#define SB_EXCEPTION_JOIN 1u /* the exception id the first update of a source actor carries */

typedef enum {
  /* The source app raised an exception: exception_id and the opaque payload. */
  SB_ACTOR_EVENT_SOURCE_EXCEPTION = 1,
  /* Core asks this app to teleport `anchor_id` to the pose: the per-interaction
   * maximum latency elapsed and the target has not arrived. */
  SB_ACTOR_EVENT_TELEPORT_REQUEST = 2,
  /* ABI 4.13, opt-in: new target for only this anchor's position or yaw
     channel. */
  SB_ACTOR_EVENT_POSITION_REQUEST = 3,
  SB_ACTOR_EVENT_YAW_REQUEST = 4,
  /* Explicit cancellation of the channels in sb_target_request_payload_t; not
     an absent update. */
  SB_ACTOR_EVENT_TARGET_RELEASE = 5
} sb_actor_event_kind_t;

typedef enum {
  SB_INTERACTION_SPACING_CONTACT = 0,
  SB_INTERACTION_AFFORDANCE_POSE_ENGAGEMENT = 1
} sb_interaction_kind_t;

typedef struct {
  uint32_t struct_size;
  int32_t kind;              /* sb_actor_event_kind_t */
  uint64_t revision;         /* the proposal-set revision this event was raised with */
  int64_t source_sequence;
  int64_t source_shared_id;  /* the raising / engaged source actor's shared id */
  uint32_t exception_id;     /* SOURCE_EXCEPTION: app-defined id (SB_EXCEPTION_JOIN for a join) */
  uint32_t payload_size;     /* bytes used: opaque SOURCE_EXCEPTION or typed
                                target-request payload */
  int32_t interaction_kind;  /* TELEPORT_REQUEST: sb_interaction_kind_t */
  uint32_t
      anchor_id; /* TELEPORT_REQUEST / typed requests / release: the anchor */
  double x;      /* TELEPORT_REQUEST / typed requests: immutable target pose */
  double y;
  double z;
  sb_orientation_t orientation;
  uint8_t payload[SB_EXCEPTION_PAYLOAD_MAX];
} sb_actor_event_t;

#define SB_TARGET_CHANNEL_POSITION 1u
#define SB_TARGET_CHANNEL_YAW 2u
#define SB_TARGET_REQUEST_PLACEMENT 1u

/** ABI 4.13: payload of POSITION_REQUEST, YAW_REQUEST and TARGET_RELEASE. Copy
 * from event.payload with memcpy (the byte array is not guaranteed aligned),
 * require payload_size == sizeof this type and check struct_size. Native C
 * layout/endian, like the enclosing event; 64 bytes on supported targets. No
 * existing event layout changes.
 *
 * The event's anchor_id/x/y/z/orientation are its immutable target in receiver
 * Z-up coordinates. An ordinary YAW_REQUEST carries the exact producing sample
 * and signed turn: target = sample_yaw + yaw_turn, congruent to orientation.yaw
 * modulo 2 pi. It is NOT relative to the delivery-time pose or previous target. A
 * wrapped-yaw producer/derived bone stream must retain the continuous ANCHOR
 * yaw correspondence of (sample_epoch,sample_sequence), including the
 * registered bone-to-anchor transform. Direct pose ports preserve continuous
 * yaw; bone quaternions and receiver-frame matrices do not preserve winding.
 * They support exact signed turns only with an independent continuous
 * realized-yaw measurement in the same world coordinate domain. Arbitrary
 * asynchronous wrapped samples cannot supply that measurement: do not guess
 * winding or silently shorten a turn. A missing correspondence is an error, not
 * a replay. An older event epoch is valid while its native-accepted
 * correspondence is retained; this differs from an older producer publication,
 * which the port rejects. Position does not require yaw history. TARGET_RELEASE
 * cancels only `channels`, even without a matching proposal snapshot.
 * Flags/causes/status retain the proposal's meanings. facing_yaw is valid only
 * with FACING_YAW. An absent channel event leaves its pending target intact.
 * request_flags PLACEMENT marks an actual absolute solve, not merely a
 * FIRST_PROPOSAL cause. Placement yaw is absolute orientation.yaw; it may have
 * no accepted anchor sample (sample_epoch/sequence are then 0). Do not decode
 * placement as sample_yaw + yaw_turn or require an ordinary correspondence.
 * An unsampled engaged pose remains readable in the snapshot, but emits no
 * ordinary YAW_REQUEST until that anchor has its own accepted sample. These
 * immutable placement requests are lifecycle data, never coalescible motion.
 *
 * A complete publication revision (source exception, releases, own immutable
 * requests, latency teleport requests) becomes visible atomically after its
 * snapshot is stored; no producer appends later to that same revision. Exactly
 * one consumer per actor must drain without a per-frame event budget, grouping
 * until revision changes or a TRUE empty queue (sb_poll_actor_event returns 0).
 * A conversion/unknown-kind/layout error is NOT empty and must be surfaced, not
 * used to commit an incomplete group. Apply each placement revision atomically
 * using only that group's poses, never a later proposal snapshot. Earlier
 * absolute placements may advance the driver's local epoch, but MUST NOT reset
 * its continuous world-yaw domain or invalidate retained accepted samples.
 * Decode ordinary goals as C_accepted + yaw_turn in that persistent domain.
 * Lift an absolute yaw to the representative nearest actual continuous yaw
 * (not the pending target); at exactly +/-pi choose +pi deterministically.
 * Placements cancel earlier goals, not later goals in the drain. Parent/root
 * carry must update each carried child's realized continuous yaw as well.
 * Coalesce feedback from multiple placements into the next normal publication.
 * The application chooses its source exception realization policy, as before.
 * Event/release draining must not depend on a successful snapshot read.
 *
 * Bounded correspondence protocol, per anchor and single consumer: keep
 * previous and current NATIVE-ACCEPTED sample stamps/yaws, accepting at most
 * one new publication between completed drains. A successful drain through
 * native return 0 fences old-sample solves (see sb_poll_actor_event), so only
 * then may the previous entry be retired. A sole current entry survives any
 * placement or local epoch change. If E1 is current after drain 1, a late E1
 * request remains valid after E2 feedback and until drain 2 completes. On
 * native/decode/layout error or incomplete drain retain both entries and accept
 * no third publication; continue observing actual realized yaw even while
 * feedback publication is paused. Preserve/surface a popped malformed event; a
 * later 0 cannot erase its error. Do not evict by age, teleport to resync, or
 * replay source updates. Ownership changes require explicit subscription
 * teardown and new admission; never reuse another consumer's sample
 * correspondence.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t channels;
  uint64_t sample_epoch;
  uint64_t sample_sequence;
  double sample_yaw;
  double yaw_turn;
  uint32_t proposal_flags;
  uint32_t proposal_cause;
  double facing_yaw;
  int32_t proposal_status;
  uint32_t
      request_flags; /* SB_TARGET_REQUEST_*; unknown bits must be rejected */
} sb_target_request_payload_t;

/**
 * @brief Current state of an actor (position and orientation).
 *
 * Fields:
 *   position_x  - X position in the horizontal plane (Z-UP).
 *   position_y  - Y position in the horizontal plane (Z-UP).
 *   yaw         - Yaw rotation around the Z-axis (radians).
 */
typedef struct {
  double position_x;
  double position_y;
  double yaw;
} sb_actor_state_t;

typedef struct {
  const char* anchor_name;
  double position_x;
  double position_y;
  double position_z;
  double yaw;
} sb_actor_observation_t;

/**
 * @brief A single realization target describing a desired pose for an actor.
 *
 * Realization targets are delivered via the sb_on_targets_t callback and
 * represent the system's current recommendation for where and how an actor
 * should position itself given the active bindings.
 *
 * Fields:
 *   arrival_time   - Zero for the root target since ABI 3 (pacing belongs to the
 *                    driver; ABI 4 deleted the arrival budget).
 *   anchor_name    - Name of the anchor this target is associated with.
 *                    Pointer is valid only during the callback invocation.
 *   position_x     - Target X position in the horizontal plane (Z-UP).
 *   position_y     - Target Y position in the horizontal plane (Z-UP).
 *   position_z     - Target Z (height, Z-UP). Meaningful only for body-anchor
 *                    targets (e.g. a hand on a board, a pelvis on a seat),
 *                    where has_z is non-zero. Root locomotion targets are
 *                    ground-projected and leave has_z = 0.
 *   has_z          - Non-zero when position_z carries a real height. Root
 *                    targets set this to 0; affordance anchor targets set 1.
 *   yaw            - Target yaw rotation around the Z-axis (radians).
 *   kernel_type    - The kernel that produced this target.
 *   referent_id    - Element ID of the referent (e.g. affordance or object)
 *                    this target was computed relative to. 0 if none.
 *   weight         - Normalized importance weight in [0, 1]. Higher values
 *                    indicate higher priority targets.
 */
// Field order is chosen so the layout is UNAMBIGUOUS across the C ABI and the
// hand-matched C# [StructLayout(Sequential)] marshal struct: all 8-byte members
// (doubles + pointer) first, then the two 4-byte ints packed together, then the
// 8-byte int64 — no isolated int-before-double padding to disagree on. The C#
// SbRealizationTarget must mirror this order exactly. DispatchRealizationTargets
// assigns by field name, so the order here is free to optimize for ABI safety.
typedef struct {
  double arrival_time;
  const char* anchor_name;
  double position_x;
  double position_y;
  double position_z;
  double yaw;
  double weight;
  int32_t has_z;
  sb_kernel_type_t kernel_type;
  int64_t referent_id;
} sb_realization_target_t;

/**
 * @brief Lifecycle phase of a relation.
 * Ordered chain: dormant -> active -> driving -> contact.
 */
typedef enum {
  SB_PHASE_DORMANT = 0,
  SB_PHASE_ACTIVE = 1,
  SB_PHASE_DRIVING = 2,
  SB_PHASE_CONTACT = 3
} sb_relation_phase_t;

/**
 * @brief A relation lifecycle phase transition.
 *
 * Fields:
 *   actor_shared_id    - Shared id of the actor (or local element id when the
 *                        actor's element is not shared).
 *   anchor_name        - Anchor the relation is bound on. Pointer is valid
 *                        only during the callback invocation.
 *   referent_shared_id - Shared id (or local id) of the referent element.
 *   kernel_type        - Kernel that produced the relation.
 *   from_phase/to_phase- Phase transition.
 *   timestamp          - Space-local time (seconds) of the transition.
 */
typedef struct {
  int64_t actor_shared_id;
  const char* anchor_name;
  int64_t referent_shared_id;
  sb_kernel_type_t kernel_type;
  sb_relation_phase_t from_phase;
  sb_relation_phase_t to_phase;
  double timestamp;
} sb_relation_event_t;

/**
 * @brief How OnRemoteUpdate resolves an incoming payload's referent to a
 * local element (mirrors Space::ReferentResolveMode / env
 * SB_V2_REFERENT_RESOLVE). LINKER (default) uses the ElementLinker cross-
 * space identity, falling back to shared-id lookup when unlinked; SHAREDID
 * forces the legacy shared-id-only lookup even when a linker mapping exists.
 * LINKER_ONLY accepts only explicit ElementLinker mappings and never lets an
 * unlinked, coincident shared ID select local geometry.
 */
typedef enum {
  SB_REFERENT_RESOLVE_LINKER = 0,
  SB_REFERENT_RESOLVE_SHAREDID = 1,
  SB_REFERENT_RESOLVE_LINKER_ONLY = 2
} sb_referent_resolve_mode_t;

/**
 * @brief Realization weighting policy family (mirrors
 * RealizationPolicyType / env SB_V2_REALIZATION_POLICY).
 * PROGRESSIVE (default) is the paper's annealed-softmax schedule; UNIFORM and
 * GREEDY are eval/paper baselines.
 */
typedef enum {
  SB_REALIZATION_POLICY_PROGRESSIVE = 0,
  SB_REALIZATION_POLICY_UNIFORM = 1,
  SB_REALIZATION_POLICY_GREEDY = 2
} sb_realization_policy_t;

/**
 * @brief Identifies a C-ABI struct for sb_sizeof(), so a client (e.g. the
 * C# marshal layer) can assert its own [StructLayout(Sequential)] mirror is
 * the same size as the core's struct at load time, catching ABI drift early.
 */
typedef enum {
  SB_STRUCT_ACTOR_STATE = 0,
  SB_STRUCT_REALIZATION_TARGET = 1,
  SB_STRUCT_RELATION_EVENT = 2,
  /* Observability surface (spatialblend_observability.h), additive: */
  SB_STRUCT_RELATION_STATE = 3,
  SB_STRUCT_RELATION_STATE_V2 = 4,
  SB_STRUCT_SCOPE_RECORD = 5,
  SB_STRUCT_SCOPE_CANDIDATE = 6,
  SB_STRUCT_REALIZATION_POLICY_PARAMS = 7,
  SB_STRUCT_REALIZATION_WEIGHT = 8,
  SB_STRUCT_REALIZATION_CHANNEL = 9,
  SB_STRUCT_SOLVER_DIAGNOSTICS = 10,
  SB_STRUCT_SOLVER_GOAL = 11,
  SB_STRUCT_RELATION_PAYLOAD = 12,
  SB_STRUCT_RELATION_QUERY = 13,
  SB_STRUCT_RELATION_EVALUATION = 14,
  SB_STRUCT_REALIZATION_TARGET_V2 = 15,
  SB_STRUCT_V2_ABI_LAYOUT = 16,
  /* Scope gating (sb_set_relation_phase_config): */
  SB_STRUCT_RELATION_PHASE_CONFIG = 17,
  /* Target-side evaluation (sb_evaluate_target_relation): */
  SB_STRUCT_TARGET_RELATION_EVALUATION = 18,
  /* ids 19 and 20 (ABI 3 embodied pose and proposal) are retired, not reused.
   */
  /* ABI 3.1: */
  SB_STRUCT_ORIENTATION = 21,
  /* ABI 4 (anchor ports, anchor proposals, events): */
  SB_STRUCT_ANCHOR_POSE = 22,
  SB_STRUCT_ANCHOR_PROPOSAL = 23,
  SB_STRUCT_PROPOSAL_SET = 24,
  SB_STRUCT_ACTOR_EVENT = 25,
  SB_STRUCT_BONE_POSE = 26,
  SB_STRUCT_NAMED_BONE_POSE = 27,
  SB_STRUCT_CHAIN_JOINT = 28,
  /* ABI 4.10: */
  SB_STRUCT_PART_CHART_COORD = 29,
  /* ABI 4.11: */
  SB_STRUCT_ANCHOR_SOLVE = 30,
  /* ABI 4.12: */
  SB_STRUCT_ANCHOR_TURN = 31,
  /* ABI 4.13: */
  SB_STRUCT_TARGET_REQUEST_PAYLOAD = 32,
  /* ABI 4.14: */
  SB_STRUCT_ANCHOR_CHART = 33
} sb_struct_id_t;

/**
 * Descriptor for the C ABI contract used by higher-level bindings (e.g. C#).
 *
 * Legacy ABI v1 is a hard-breaking change from this descriptor; clients are
 * expected to fail fast if sb_abi_version() does not return the current value.
 */
typedef struct {
  uint32_t descriptor_size;
  uint32_t abi_version;
  uint32_t realization_target_size;
  uint32_t pointer_size;
  uint32_t kernel_type_size;
  uint32_t arrival_time_offset;
  uint32_t anchor_name_offset;
  uint32_t position_x_offset;
  uint32_t position_y_offset;
  uint32_t position_z_offset;
  uint32_t yaw_offset;
  uint32_t weight_offset;
  uint32_t has_z_offset;
  uint32_t kernel_type_offset;
  uint32_t referent_id_offset;
  uint32_t abi_minor; /* ABI 3.1: sb_abi_minor_version() */
} sb_v2_abi_layout_t;

/**
 * @brief Which polygon-coordinate strategy encoded a contact's normalized
 * position (mirrors NormalizedPolygonPosition::Strategy). Radial applies to
 * convex referents; MVC (mean-value-coordinate) applies to general simple
 * polygons. The MVC per-sample weight vector is variable-length and is
 * intentionally NOT surfaced through the C ABI (doesn't fit a POD struct) —
 * use the Python binding if you need it.
 */
typedef enum {
  SB_NORMALIZED_RADIAL = 0,
  SB_NORMALIZED_MVC = 1
} sb_normalized_strategy_t;

/**
 * @brief A read-only snapshot of one (anchor, referent) relation's currently
 * grounded state (anchor liveness / relation-state query).
 * Populated by sb_get_relation_states directly
 * from the same propagated match data as the realization-target callback and
 * relation-event callback for the same sb_update_actor_state call — never a
 * separate recomputation.
 *
 * Fields:
 *   anchor_name           - Anchor this relation is bound on. Pointer is
 *                            valid only until the next call into the space
 *                            (sb_get_relation_states copies out fixed-size
 *                            fields eagerly, but anchor_name is returned by
 *                            reference into space-owned storage).
 *   kernel_type            - Kernel that produced this relation.
 *   referent_element_id    - Local element id of the referent.
 *   referent_shared_id     - Shared id of the referent (equals
 *                            referent_element_id when not shared).
 *   proximity              - Normalized engagement R in [0, 1]; 1 means
 *                            fully engaged/grounded.
 *   has_contact            - Non-zero when a support contact point is valid.
 *   contact_point_world_x/y/z - Contact point on the referent boundary,
 *                            WORLD frame. Meaningful only when has_contact.
 *   normalized_strategy    - Which encoding populated the fields below.
 *   normalized_angle       - Radial encoding: angle from centroid (radians).
 *   normalized_radial_ratio- Radial encoding: 0 (centroid) to 1 (boundary).
 *   normalized_height_ratio- Z-axis normalization (both encodings).
 *   active                 - Non-zero when RelationPhaseTracker currently
 *                            holds this relation (kActive/kDriving/kContact):
 *                            it is in the family's scope and realized.
 */
// Field order follows the same ABI-safe convention as sb_realization_target_t:
// all 8-byte members (doubles + pointer) first, then 4-byte enums/ints packed
// together — no isolated int-before-double padding to disagree on across a
// hand-matched C# [StructLayout(Sequential)] marshal struct.
typedef struct {
  const char* anchor_name;
  double proximity;
  double contact_point_world_x;
  double contact_point_world_y;
  double contact_point_world_z;
  double normalized_angle;
  double normalized_radial_ratio;
  double normalized_height_ratio;
  int64_t referent_element_id;
  int64_t referent_shared_id;
  sb_kernel_type_t kernel_type;
  sb_normalized_strategy_t normalized_strategy;
  int32_t has_contact;
  int32_t active;
} sb_relation_state_t;

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_TYPE_H_
