// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SPATIALBLEND_C_SPATIALBLEND_H_
#define SPATIALBLEND_C_SPATIALBLEND_H_

#include <stddef.h>
#include <stdint.h>

#include "spatialblend_type.h"

#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SB_EXPORT __declspec(dllexport)
#else
#define SB_EXPORT __declspec(dllimport)
#endif  // SPATIALBLEND_BUILD_DLL
#else
#define SB_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#include "spatialblend_rich.h"
#include "spatialblend_receiver_approach_context.h"
#include "spatialblend_receiver_approach_feasibility.h"
#include "spatialblend_observed_collaborator.h"
#include "spatialblend_observed_source_setup.h"
#include "spatialblend_observability.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ============================================================================
 * COORDINATE SYSTEM CONVENTIONS
 * ============================================================================
 *
 * SpatialBlend uses a Z-UP coordinate system for all spatial operations:
 *
 * 2D Coordinates (horizontal plane):
 *   - X: horizontal axis (right)
 *   - Y: horizontal axis (forward)
 *   - Points are specified as (x, y) tuples
 *
 * 3D Coordinates:
 *   - X: horizontal axis (right)
 *   - Y: horizontal axis (forward)
 *   - Z: vertical axis (up)
 *   - Points are specified as (x, y, z) tuples
 *
 * Rotations:
 *   - Yaw: rotation around Z-axis (vertical axis), in radians
 *   - Pitch: rotation around Y-axis (lateral tilt)
 *   - Roll: rotation around X-axis (longitudinal tilt)
 *   - Rotation order: Z x Y x X (yaw, pitch, roll)
 *
 * Transform Matrices:
 *   - 4x4 row-major transformation matrices (16 doubles)
 *   - Column 3 (translation): [tx, ty, tz, 1]
 *
 * Actor State:
 *   - Position: (x, y) in the horizontal plane
 *   - Yaw: rotation around Z-axis (radians)
 *
 * ============================================================================
 */

/* --------------------------------------------------------------------------
 * Session management
 * -------------------------------------------------------------------------- */

/**
 * @brief Create a new session.
 * @return Session handle, or 0 on failure.
 */
SB_EXPORT sb_session_t sb_create_session();

/**
 * @brief Destroy a session and release all associated resources.
 * @param session Session handle returned by sb_create_session().
 */
SB_EXPORT void sb_destroy_session(sb_session_t session);

/* --------------------------------------------------------------------------
 * Space management
 * -------------------------------------------------------------------------- */

/**
 * @brief Add a new space to a session.
 * @param session Session handle.
 * @return Space handle, or 0 on failure.
 */
SB_EXPORT sb_space_t sb_add_space(sb_session_t session);

/**
 * @brief Register an explicit, local support-correspondence batch.
 *
 * This is an in-process Session/SpaceHub operation only; the returned opaque
 * registration is not a wire identity and does not publish anything to peers.
 * Every endpoint must be a live element in a space owned by @p session. The
 * complete batch is validated before mutation. A source can map to one target
 * per target space; a different existing target in that slot rejects the whole
 * batch. Multiple sources may intentionally map to the same support element.
 * Duplicate exact entries within one batch are deduplicated. Exact
 * pre-existing pairs are preserved and are observed unowned by the returned
 * registration; disposing that registration will not remove them. Requesting
 * a pair already owned by another live registration is rejected.
 * @return Non-zero registration handle on success, 0 on invalid/conflicting
 *         input, a contended scene preflight, or relation-callback reentry.
 *         Retry after the update/callback, not from inside it. A successful
 *         registration owns only pairs it installed. Existence is a preflight,
 *         not an endpoint lifetime lease; validity must be queried separately.
 */
SB_EXPORT sb_support_correspondence_registration_t
sb_register_support_correspondences(
    sb_session_t session, const sb_support_correspondence_t* correspondences,
    int64_t count);

// Optional strict-owned correspondence v1; existing ABI2/legacy helper unchanged.
// The feature uses the existing four-int64 pair layout, max 128 pairs. All
// endpoints must be positive, live and in distinct spaces. Requested sources
// and targets are each unique; every requested source->target-space slot is new.
// Targets may not have outgoing links (these shadow inverse translation).
// No borrowed exact pairs or many-to-one reverse ambiguity is accepted.
// Registration is atomic and returns 0 on invalid/conflict/contended/reentry.
SB_EXPORT uint32_t sb_get_owned_correspondence_feature_version();
SB_EXPORT sb_support_correspondence_registration_t
sb_try_register_owned_correspondences(sb_session_t session, uint32_t version,
    const sb_support_correspondence_t* pairs, uint32_t count);
// Validation: 1 exact owned versions/live endpoints, 0 unavailable, -1 invalid,
// -4 registry/linker busy or callback reentry. It is an as-of-query check, not a
// lock across later use. Scene/checkpoint validation remains independently needed.
SB_EXPORT int32_t sb_try_validate_owned_correspondences(sb_session_t session,
    sb_support_correspondence_registration_t registration, uint32_t version);
// Disposal: 1 acknowledged/already gone, 0 wrong owner/type, -1 invalid,
// -4 registry/linker busy/reentry (retain handle and retry; no partial Busy).
// Removes still-exact owned versions only; any ABA replacements survive cleanup.
SB_EXPORT int32_t sb_try_dispose_owned_correspondences(sb_session_t session,
    sb_support_correspondence_registration_t registration, uint32_t version);

/**
 * @brief Dispose a support-correspondence registration.
 *
 * Removes only exact pairs installed by this registration and is safe to call
 * repeatedly or after its session has been destroyed. It cannot restore stale
 * mappings or remove an unrelated/pre-existing pair.
 */
SB_EXPORT void sb_dispose_support_correspondences(
    sb_support_correspondence_registration_t registration);

/**
 * @brief Host a space on a TCP port so remote clients can connect.
 * @param space   Space handle.
 * @param port    Preferred port number (0 for any available port).
 * @return Async handle. Use sb_try_wait() to confirm the operation completed.
 */
SB_EXPORT sb_handle_t sb_host_space(sb_space_t space, int32_t port);

/**
 * @brief Connect a space to a remote hosted space.
 * @param space  Space handle.
 * @param host   Hostname or IP address of the remote host (null-terminated).
 * @param port   Port number the remote host is listening on.
 * @return Async handle. Use sb_try_wait() to confirm the connection succeeded.
 */
SB_EXPORT sb_handle_t sb_connect_space(sb_space_t space, const char* host,
                                       int32_t port);

/* --------------------------------------------------------------------------
 * Tunneled core link (ABI 4.5): hosting and joining a room with no address.
 *
 * No peer ever dials another peer's address. The core link (gRPC over HTTP/2,
 * unchanged) runs over a byte stream that the APPLICATION carries between the
 * two peers on a channel they already share (Fusion reliable data in the Unity
 * app). One sb_tunnel_t is one such stream, to one remote peer:
 *
 *   sb_tunnel_write: bytes that arrived from the peer, into core.
 *   sb_tunnel_read:  bytes core has for the peer, to be sent on the carrier.
 *
 * Neither call blocks; pump both every frame. Bytes are an ordered stream with
 * no framing of their own: the carrier must deliver them in order and without
 * loss, and may split or join them at any boundary (sb_tunnel_read may return
 * any prefix; sb_tunnel_write accepts any chunking). sb_tunnel_write applies
 * backpressure: while earlier bytes still wait for core it takes nothing and
 * returns 2 (would block); keep those bytes and offer them again after the next
 * sb_tunnel_read. The link therefore holds at most the last chunk it accepted.
 *
 * Host: sb_host_space_tunneled() once, then sb_accept_tunnel() for each joining
 * peer, and sb_tunnel_close() when that peer leaves.
 * Joiner: sb_connect_space_tunneled(); the returned handle completes (see
 * sb_try_wait) only while the link is being pumped, and sb_try_wait is what
 * applies the host's bootstrap to the space, on the calling thread (call it from
 * the thread that ticks the space). There is no timer: sb_tunnel_close() on the
 * joiner's link fails a pending join, and after it the session may join again
 * (a fresh sb_connect_space_tunneled). sb_cleanup_all and sb_destroy_session close
 * every link of the sessions they end.
 *
 * Available on POSIX targets (Android, macOS, Linux). On Windows every call
 * here fails (returns 0 / -1).
 * sb_host_space / sb_connect_space (TCP) remain for tests and LAN tooling.
 * -------------------------------------------------------------------------- */

/**
 * @brief Host a space with no listening socket; peers attach with sb_accept_tunnel.
 * @return Async handle (complete on return; sb_try_wait confirms) or 0 on failure.
 *         One session hosts or joins once, as for sb_host_space.
 */
SB_EXPORT sb_handle_t sb_host_space_tunneled(sb_space_t space);

/**
 * @brief Host side: attach one joining peer.
 * @param out_link Receives the peer's link on success.
 * @return 1 on success, 0 on failure (not hosted tunneled, host stopping, no
 *         POSIX sockets), -1 on an invalid argument.
 */
SB_EXPORT int32_t sb_accept_tunnel(sb_space_t space, sb_tunnel_t* out_link);

/**
 * @brief Join the hosting space over a tunneled link.
 * @param out_link Receives the link immediately; pump it while the handle is pending.
 * @return Async handle. It completes when the host's bootstrap has been applied
 *         (sb_try_wait non-zero), or fails with sb_get_error text when the link
 *         ends first or the bootstrap does not complete. 0 if it could not start.
 */
SB_EXPORT sb_handle_t sb_connect_space_tunneled(sb_space_t space,
                                                sb_tunnel_t* out_link);

/**
 * @brief Bytes from the remote peer into core.
 * @return 1 accepted (all n bytes), 2 would block (NOTHING was taken: offer the
 *         same bytes again after the next sb_tunnel_read), 0 the link is closed,
 *         -1 invalid link/argument.
 */
SB_EXPORT int32_t sb_tunnel_write(sb_tunnel_t link, const uint8_t* bytes,
                                  uint32_t n);

/**
 * @brief Bytes from core for the remote peer (non-blocking).
 * @param buf Destination; cap its capacity; *n receives the byte count (may be 0).
 * @return 1 ok, 0 the link ended (core closed it) and everything it sent was
 *         read, -1 invalid link/argument.
 */
SB_EXPORT int32_t sb_tunnel_read(sb_tunnel_t link, uint8_t* buf, uint32_t cap,
                                 uint32_t* n);

/**
 * @brief Close a link and forget it (host: the peer left; joiner: leave the room).
 *        Safe to call twice and on an unknown link.
 */
SB_EXPORT void sb_tunnel_close(sb_tunnel_t link);

/* --------------------------------------------------------------------------
 * Async handle utilities
 * -------------------------------------------------------------------------- */

/**
 * @brief Block until the async operation completes or the timeout expires.
 * @param handle     Handle returned by an async operation.
 * @param timeout_ms Maximum time to wait in milliseconds. 0 = no wait.
 * @return Non-zero if the operation completed successfully, 0 otherwise.
 */
SB_EXPORT int32_t sb_try_wait(sb_handle_t handle, int32_t timeout_ms);

/**
 * @brief Retrieve the error string for a failed async operation.
 * @param handle Handle returned by an async operation.
 * @return Null-terminated error string, or NULL if there is no error.
 *         The pointer is valid until the next call on the same handle.
 */
SB_EXPORT const char* sb_get_error(sb_handle_t handle);

/* --------------------------------------------------------------------------
 * Element management
 * -------------------------------------------------------------------------- */

/**
 * @brief Add a type-neutral element to a space.
 * @param space  Space handle.
 * @param parent Parent element ID, or 0 to add as a root element.
 * @return Element handle, or 0 on failure.
 */
SB_EXPORT sb_element_t sb_add_element(sb_space_t space, sb_element_t parent);

/**
 * @brief Set an element's parent-local transform (world for a root element).
 * @param space     Space handle.
 * @param elem      Element handle.
 * @param matrix_4x4 Row-major 4x4 transformation matrix (16 doubles, Z-UP).
 *        Must be finite, affine, and invertible. Invalid input is a no-op.
 * @return SB_SCENE_OK, or why the change was not registered (see
 *         sb_scene_status_t; a containment violation names both elements in
 *         sb_get_scene_status's message).
 */
SB_EXPORT sb_scene_status_t sb_set_element_transform(sb_space_t space,
                                                     sb_element_t elem,
                                                     const double* matrix_4x4);

/**
 * @brief Status of the most recent scene mutator on this space (the value the
 *        mutator returned), and its message.
 * @param message  Optional buffer for a null-terminated message naming the
 *                 elements involved (truncated to capacity); may be NULL.
 * @return SB_SCENE_OK, or why the last mutator did not register its change.
 */
SB_EXPORT sb_scene_status_t sb_get_scene_status(sb_space_t space, char* message,
                                                int32_t capacity);

/**
 * @brief Associate a shared (cross-space) identifier with an element.
 *
 * Shared IDs allow elements in different spaces to be linked together.
 * @param space     Space handle.
 * @param elem      Element handle.
 * @param shared_id Shared identifier to assign.
 */
SB_EXPORT void sb_set_element_shared_id(sb_space_t space, sb_element_t elem,
                                        int64_t shared_id);

/**
 * @brief Set a string tag on an element for use with tagged binding.
 * @param space Space handle.
 * @param elem  Element handle.
 * @param tag   Null-terminated tag string.
 */
SB_EXPORT void sb_set_element_tag(sb_space_t space, sb_element_t elem,
                                  const char* tag);

/**
 * @brief ABI 4.11: declare the affordance pose an interaction-role child carries for its parent
 *        object. The child's tag names the pose ("stand", "seated", ...). The pose places `subject_anchor`:
 *        support_kind 0 (PART): its registered contact lies on the parent's named part `part` (a sub-region);
 *        support_kind 1 (ACTOR_ROOT): it stands at the embodiment's registered `standing` height over the actor's
 *        root (`part` ignored). Each side instantiates the poses for its own registered embodiment.
 * @param subject_anchor NULL clears the element's pose.
 * @return 0 set or cleared; -1 invalid handle, unknown element, unknown support_kind, or a PART pose without `part`.
 */
SB_EXPORT int32_t sb_set_element_affordance_pose(sb_space_t space, sb_element_t elem,
                                                 const char* subject_anchor, int32_t support_kind,
                                                 const char* part);

/* --------------------------------------------------------------------------
 * COORDINATE FRAMES of the sb_set_element_* geometry setters (ABI 3.1).
 *
 * One rule: geometry is given in the element's OWN (local) frame, Z-up. The
 * element's world pose is the composition of its sb_set_element_transform and
 * its ancestors'; core lifts through it at every query, so geometry follows later
 * transform changes of the element or its ancestors. A ROOT element's local frame
 * IS the world (a root room is authored in world coordinates).
 *
 *   sb_set_element_polygon  element-local for every role except NAVIGABLE.
 *                           EXCEPTION, world-baked, listed explicitly: a NAVIGABLE
 *                           polygon (walkable ring, nested walkable zone) is read
 *                           as WORLD coordinates, because free space and obstacle
 *                           sync are world-frame. For the root room this IS the
 *                           local frame (no exception in effect); a nested NAVIGABLE
 *                           zone must be authored in world coordinates and re-issued
 *                           when an ancestor moves (Unity: compose it into the
 *                           world; keep nested zones under untransformed ancestors).
 *   sb_set_element_box      element-local (axis-aligned in the element frame).
 *   sb_set_element_polytope element-local.
 *   sb_set_element_collider element-local (params in the element's frame); the
 *                           derived envelope footprint is element-local too.
 *   sb_add_sub_region       the patch polygon is in the patch's own frame;
 *                           transform16 places the patch in the OWNING
 *                           DEFINITION's frame (the element frame for polygon /
 *                           box / polytope definitions, the world for a surface).
 *   sb_set_element_surface  EXCEPTION, world-baked, intentional: a surface is
 *                           3D vertices from which core derives its own plane
 *                           frame (affordance grounding is world-in / world-out
 *                           on it), so the vertices are WORLD coordinates. Unity
 *                           must compose a surface's vertices into the world and
 *                           call again when the element or an ancestor moves.
 *   sb_set_element_local_polygon (containment transfer header): identical to
 *                           sb_set_element_polygon for roles other than NAVIGABLE
 *                           and SURFACE; kept as an alias.
 *
 * Before ABI 3.1 sb_set_element_polygon (and the collider envelope) read their
 * points as WORLD coordinates for the containment path; clients that baked world
 * points into a transformed element must now pass the element-local points.
 * -------------------------------------------------------------------------- */

/**
 * @brief Set a 2D polygon geometry and role on an element.
 * @param space      Space handle.
 * @param elem       Element handle.
 * @param role       Role of this geometry in relation evaluation.
 * @param points_xy  Flat array of 2D polygon points [x1, y1, x2, y2, ...] in the
 *                   ELEMENT's local frame (a root element's local frame is the
 *                   world); NAVIGABLE is the one world-baked role, see the frame
 *                   rules above.
 * @param num_points Number of points (not number of doubles).
 *        Scene polygon/surface/sub-region inputs are capped at 65536 vertices.
 *        Buffers must contain the declared extent and finite coordinates.
 */
SB_EXPORT sb_scene_status_t sb_set_element_polygon(sb_space_t space,
                                                   sb_element_t elem,
                                                   sb_geometry_role_t role,
                                                   const double* points_xy,
                                                   int64_t num_points);

/**
 * @brief Set an axis-aligned 3D box volume and role on an element.
 * @param space Space handle.
 * @param elem  Element handle.
 * @param role  Role of this volume in relation evaluation.
 * @param min_x Minimum X coordinate.
 * @param min_y Minimum Y coordinate.
 * @param min_z Minimum Z coordinate.
 * @param max_x Maximum X coordinate.
 * @param max_y Maximum Y coordinate.
 * @param max_z Maximum Z coordinate.
 */
SB_EXPORT sb_scene_status_t sb_set_element_box(sb_space_t space, sb_element_t elem,
                                               sb_geometry_role_t role,
                                               double min_x, double min_y, double min_z,
                                               double max_x, double max_y, double max_z);

/**
 * @brief Auto-derive an element's Containment "envelope" from a primitive collider.
 *
 * Builds the enclosing footprint (role = Containment) IN THE CORE from the given
 * primitive and assigns it as the element's spatial definition ONLY IF the element
 * has no spatial definition yet — so an explicit boundary (e.g. an affordance
 * polygon set later via sb_set_element_polygon) always overrides the envelope.
 *
 * Why: the cross-space spacing relation needs a boundary. Without one, proximity is
 * measured to the element CENTRE (never reaches 1) and the reconstructed clone keeps
 * a residual stand-off; with one, proximity is boundary-relative and the approach
 * point maps onto the piece at prox->1. Clients only supply primitive params (data);
 * the primitive->footprint conversion lives in the core.
 *
 * @param space   Space handle.
 * @param elem    Element handle.
 * @param type    Collider primitive type.
 * @param params  Params in the element's LOCAL frame, Z-up (the derived footprint
 *                is element-local too). Layout per type:
 *                BOX={cx,cy,cz,sx,sy,sz}, SPHERE={cx,cy,cz,r},
 *                CAPSULE={cx,cy,cz,r,height,axis}.
 * @param num_params Number of doubles in @p params.
 */
SB_EXPORT sb_scene_status_t sb_set_element_collider(sb_space_t space,
                                                    sb_element_t elem,
                                                    sb_collider_type_t type,
                                                    const double* params,
                                                    int64_t num_params);

/**
 * @brief Set a planar 3D surface on an element from N xyz vertices (Z-UP).
 *        WORLD-baked (the one exception to the element-local rule, see the frame
 *        rules above): re-issue it when the element or an ancestor moves.
 * @param space      Space handle.
 * @param elem       Element handle.
 * @param points_xyz Flat array of 3*num_points doubles (x,y,z per vertex), in
 *                   WORLD coordinates.
 * @param num_points Number of vertices (>= 3).
 * @param role       Role of this surface in relation evaluation.
 */
SB_EXPORT sb_scene_status_t sb_set_element_surface(sb_space_t space,
                                                   sb_element_t elem,
                                                   const double* points_xyz,
                                                   int32_t num_points,
                                                   sb_geometry_role_t role);

/**
 * @brief Set a convex polytope on an element from vertices and faces.
 * @param space        Space handle.
 * @param elem         Element handle.
 * @param vertices_xyz Flat array of 3*num_vertices doubles, in the ELEMENT's
 *        local frame.
 * @param num_vertices Number of vertices.
 * @param face_indices Concatenation of every face's vertex indices.
 *        ABI2 requires trusted buffers: the declared concatenated extent must
 *        exist. Pointer allocation size is not knowable here; do not pass
 *        unvalidated network buffers directly. At most 65536 vertices/faces and
 *        1048576 indices; each face has 3..num_vertices in-range indices.
 *        Malformed numeric/count input leaves geometry intact.
 * @param face_sizes   Per-face vertex counts (length num_faces).
 * @param num_faces    Number of faces.
 * @param role         Role of this volume in relation evaluation.
 */
SB_EXPORT void sb_set_element_polytope(sb_space_t space, sb_element_t elem,
                                       const double* vertices_xyz,
                                       int32_t num_vertices,
                                       const int32_t* face_indices,
                                       const int32_t* face_sizes,
                                       int32_t num_faces,
                                       sb_geometry_role_t role);

/**
 * @brief Add a named, tagged sub-region patch to an element's spatial
 *        definition.
 *
 * The element must already have a spatial definition (set a surface / polygon /
 * box / polytope first); otherwise this is a no-op.
 *
 * @param space       Space handle.
 * @param elem        Element handle.
 * @param name        Null-terminated sub-region name.
 * @param tag         Null-terminated sub-region tag (e.g. "Sittable").
 * @param points_xy   Flat array of 2*num_points doubles in the patch's local
 *                    frame.
 * @param num_points  Number of patch boundary vertices (>= 3).
 * @param transform16 Row-major 4x4 transform placing the patch in the owning
 *        definition's frame (element-local for polygon/box/polytope, world for a
 *        surface), or NULL for identity.
 */
SB_EXPORT void sb_add_sub_region(sb_space_t space, sb_element_t elem,
                                 const char* name, const char* tag,
                                 const double* points_xy, int32_t num_points,
                                 const double* transform16);

/**
 * @brief Declare how a point of a named part is read on the matched part of another space (ABI 4.10).
 *
 * Default SB_SURFACE_CHART_RADIAL. The frame charts need an axis-aligned rectangular part boundary and preserve
 * the rectangle's (u,v) across boards of any size and aspect ratio (STRETCH: whole part to whole part; CONTAIN:
 * uniform scale, letterboxed). Both spaces declare the same kind for matched parts; the receiver reports a mismatch
 * (no proposal), it never guesses. The kind travels with the relation payload (additive wire field).
 *
 * @return 0 ok, -1 unknown space/element/part or bad chart value, -2 a frame chart on a boundary that is not an
 *         axis-aligned rectangle (nothing changed).
 */
SB_EXPORT int32_t sb_set_sub_region_chart(sb_space_t space, sb_element_t elem, const char* part_name, int32_t chart);

/**
 * @brief Read a world point as chart coordinates of a named part (ABI 4.10).
 *
 * The point is expressed in the part frame; the coordinate is its position in the part's own chart plus the metric
 * normal offset. Use it for anything that must land on the same place as the contact of the relation path (strokes,
 * pointers). The relation path (AffordancePose hand, pelvis) uses the very same mapping inside the core.
 *
 * @return 0 ok, -1 invalid arguments / unknown part, -2 the part's chart cannot be read (not a rectangle, degenerate).
 */
SB_EXPORT int32_t sb_part_chart_encode(sb_space_t space, sb_element_t elem, const char* part_name,
                                       const double* world_xyz /* 3 */, sb_part_chart_coord_t* out);

/**
 * @brief Read chart coordinates on a named part of THIS space; returns the world point (ABI 4.10).
 *
 * @p in->chart must equal the kind this part declares (-2 otherwise: matched parts declare the same chart).
 * @return 0 ok, -1 invalid arguments / unknown part, -2 chart kind mismatch or the part cannot be read,
 *         -3 invalid coordinate (non-finite, non-positive source extent).
 */
SB_EXPORT int32_t sb_part_chart_decode(sb_space_t space, sb_element_t elem, const char* part_name,
                                       const sb_part_chart_coord_t* in, double* out_world_xyz /* 3 */);

/**
 * @brief Find an element by its shared cross-space identifier.
 * @param space     Space handle.
 * @param shared_id Shared identifier to look up.
 * @return Element handle, or 0 if no element carries the shared ID.
 */
SB_EXPORT sb_element_t sb_find_element_by_shared_id(sb_space_t space,
                                                    int64_t shared_id);

/**
 * @brief Add a named anchor to an element at a given local transform.
 *
 * Anchors are named attachment points on an element used by grounding rules
 * to compute relation-specific support points.
 *
 * @param space      Space handle.
 * @param elem       Element handle.
 * @param name       Null-terminated anchor name (unique per element).
 * @param matrix_4x4 Column-major 4x4 local transformation matrix (16 doubles,
 *                   Z-UP). Uses Eigen column-major layout.
 */
SB_EXPORT void sb_add_element_anchor(sb_space_t space, sb_element_t elem,
                                     const char* name,
                                     const double* matrix_4x4);

/**
 * @brief Share an element across spaces via the hub.
 *
 * Propagates the element and its subtree to other spaces in the session
 * (or remote spaces connected via network).
 *
 * @param space          Space handle.
 * @param elem           Element handle to share.
 * @param user_data      Opaque application payload (e.g. serialized entity)
 *                       propagated verbatim and delivered to the receiving
 *                       space via sb_register_on_element_added(). May be NULL.
 * @param user_data_size Size of user_data in bytes (0 if none).
 */
SB_EXPORT void sb_share_element(sb_space_t space, sb_element_t elem,
                                const void* user_data, int64_t user_data_size);

/**
 * @brief Unshare a previously shared element.
 *
 * Removes the element from cross-space propagation and notifies peer spaces.
 * A receiver that had linked a local element to it (LinkToExisting) loses the
 * link and is told through sb_on_link_removed_t (SB_LINK_REMOVED_PEER_UNSHARED);
 * its element survives. A late joiner is no longer handed the share.
 *
 * @param space Space handle.
 * @param elem  Element handle to unshare.
 */
SB_EXPORT void sb_unshare_element(sb_space_t space, sb_element_t elem);

/**
 * @brief Remove the cross-space link of an element on both sides (ABI 4.9).
 *
 * `elem` is the receiving side's counterpart of a shared element: the existing
 * element a share was linked to (LinkToExisting) or the replica kept for it
 * (AddAsNew). The link is removed from this space and the sharer is told
 * through the hub (its sb_on_link_removed_t callback fires with
 * SB_LINK_REMOVED_PEER_UNLINKED), so it can release its side. The element is
 * NOT deleted: a LinkToExisting target is left exactly as it was; an AddAsNew
 * replica stops being source-owned and becomes a plain local element (a later
 * unshare of the source no longer removes it; sb_remove_local_element can).
 * The source stays shared, other receivers' links are untouched, and a late
 * joiner is not handed the share through this element. Relations from the
 * unlinked source never resolve onto `elem` again (not even through a shared
 * id) until the sharer shares it again and the app resolves it anew.
 *
 * A sharer's own source element is not unlinked by this call: use
 * sb_unshare_element. The call does not fire this space's own callback.
 *
 * @return the number of links removed (0: unknown element, or not the
 *         counterpart of a remote source).
 */
SB_EXPORT int32_t sb_unlink(sb_space_t space, sb_element_t elem);

/**
 * @brief Move an element, with its subtree, under another parent (ABI 4.8).
 *
 * The element keeps its identity, shared id, tag, geometry and every cross-space link; nothing is sent to peers (the
 * tree structure is each space's own). `matrix_4x4` is the element's NEW parent-local transform (row-major, Z-up, as
 * sb_set_element_transform) and is applied atomically with the move, so the sibling-overlap and containment checks see the
 * placement the element will have; pass null to keep its current local transform. Roots, received replicas, a new
 * parent that is not an element (0, -1 or unknown) and a new parent inside the element's own subtree are refused.
 *
 * @return SB_SCENE_OK, or why the move was not registered (the element is left where it was; a containment violation
 *         names both elements in sb_get_scene_status's message).
 */
SB_EXPORT sb_scene_status_t sb_reparent_element(sb_space_t space, sb_element_t elem,
                                                sb_element_t new_parent,
                                                const double* matrix_4x4);

/**
 * @brief Delete a local-only non-root element subtree without unsharing.
 *
 * Refuses invalid/root elements and any subtree that is received, shared, or
 * linked across spaces. On success, actors rooted in the subtree are detached
 * and their binding/controller/realization/anchor-override state is cleared.
 * This API never sends an element lifecycle event to peers.
 *
 * @return 1 on removal; 0 when the element is absent or protected.
 */
SB_EXPORT int32_t sb_remove_local_element(sb_space_t space,
                                          sb_element_t elem);

/* --------------------------------------------------------------------------
 * Actor management
 * -------------------------------------------------------------------------- */

/**
 * @brief Create a new actor in a space.
 * @param space Space handle.
 * @return Actor handle, or 0 on failure.
 */
SB_EXPORT sb_actor_t sb_create_actor(sb_space_t space);

/** @return 1 when the actor was live and retired, otherwise 0. */
SB_EXPORT int32_t sb_destroy_actor(sb_space_t space, sb_actor_t actor);

/**
 * @brief Add a named anchor to an actor at a given 3D offset.
 *
 * Anchors are named attachment points on an actor used by the kernel system
 * to compute realization targets.
 *
 * @param space  Space handle.
 * @param actor  Actor handle.
 * @param name   Null-terminated anchor name (unique per actor).
 * @param x      X offset relative to the actor origin (Z-UP).
 * @param y      Y offset relative to the actor origin (Z-UP).
 * @param z      Z offset relative to the actor origin (Z-UP).
 * @param parent Null-terminated name of the parent anchor in the same actor's
 *               anchor set, or NULL to attach directly to the actor root
 *               transform. Forms the shallow anchor tree.
 */
SB_EXPORT void sb_add_anchor(sb_space_t space, sb_actor_t actor,
                             const char* name, double x, double y, double z,
                             const char* parent);

/**
 * @brief Add a checked custom anchor with either horizontal or 3D position DOF.
 *
 * This is additive: sb_add_anchor retains its legacy Position2D + yaw
 * behavior. position_dimension must be exactly 2 or 3. Rotation remains yaw
 * only; this API does not accept or imply quaternion, pitch, or roll support.
 *
 * @return 1 only when a live actor accepts a finite, nonempty, unique anchor
 *         name and a valid dimension; otherwise 0 with no mutation.
 *
 * @param parent Optional parent-anchor name. A direct self-parent is rejected.
 *               Parent anchors may be added out of order; callers remain
 *               responsible for validating the complete parent graph.
 */
SB_EXPORT int32_t sb_add_anchor_with_position_dimension(
    sb_space_t space, sb_actor_t actor, const char* name, double x, double y,
    double z, const char* parent, int32_t position_dimension);

/**
 * @brief Register the canonical body anchor set on an actor.
 *
 * Adds six anchors with conventional parent links and DOF:
 *   root (pos2D+yaw, no parent), hips (parent root), chest (parent hips),
 *   head (pos3D+rot3D, parent chest), hand_l / hand_r (pos3D+rot3D, parent
 *   chest). Existing anchors of the same name are appended, not replaced.
 *
 * @param space Space handle.
 * @param actor Actor handle.
 */
SB_EXPORT void sb_add_standard_anchors(sb_space_t space, sb_actor_t actor);

/**
 * @brief Set a per-anchor WORLD-space pose override (anchor liveness).
 *
 * SubjectAnchor is normally a fixed local offset from the actor root
 * transform (set once via sb_add_anchor / sb_add_standard_anchors). This
 * override lets a client drive one named anchor directly from live tracking
 * data (e.g. a real Quest hand) instead of the actor-root + fixed-offset
 * computation, for every relation/target computation involving that anchor.
 * The actor ROOT anchor's position (used for scope resolution and the
 * authoritative 2D pose from sb_get_actor_state) is NEVER affected by this
 * override — it always tracks sb_update_actor_state.
 *
 * @param space       Space handle.
 * @param actor       Actor handle.
 * @param anchor_name Null-terminated name of an anchor already added via
 *                    sb_add_anchor / sb_add_standard_anchors.
 * @param x           World X position (Z-UP).
 * @param y           World Y position (Z-UP).
 * @param z           World Z (height, Z-UP).
 * @param yaw         World yaw rotation around the Z-axis (radians).
 */
SB_EXPORT void sb_set_anchor_transform(sb_space_t space, sb_actor_t actor,
                                       const char* anchor_name, double x,
                                       double y, double z, double yaw);

/**
 * @brief Clear a previously-set anchor world-pose override.
 *
 * After this call, the named anchor reverts to the fixed local-offset
 * computation from the actor root transform.
 *
 * @param space       Space handle.
 * @param actor       Actor handle.
 * @param anchor_name Null-terminated anchor name.
 */
SB_EXPORT void sb_clear_anchor_transform(sb_space_t space, sb_actor_t actor,
                                         const char* anchor_name);

/**
 * @brief Mark a named anchor present (nonzero) or absent (zero).
 *
 * Inactive anchors are omitted from transmitted relation observations. All
 * anchors begin active, preserving legacy fixed-offset behavior.
 */
SB_EXPORT int32_t sb_set_anchor_active(sb_space_t space, sb_actor_t actor,
                                       const char* anchor_name, int32_t active);

/**
 * @brief Associate an actor with a specific element (e.g. place actor in room).
 * @param space Space handle.
 * @param actor Actor handle.
 * @param elem  Element handle.
 */
SB_EXPORT void sb_set_actor_element(sb_space_t space, sb_actor_t actor,
                                    sb_element_t elem);

/**
 * @brief Find the actor associated with an element shared ID.
 * @param space     Space handle.
 * @param shared_id Shared identifier of the actor's element.
 * @return Actor handle, or 0 if no actor is associated with the element.
 */
SB_EXPORT sb_actor_t sb_find_actor_by_shared_id(sb_space_t space,
                                                int64_t shared_id);

/**
 * @brief Update the authoritative 2D pose of an actor.
 *
 * This is the "ground truth" state used by the solver. Distinct from the
 * receiver's embodied (rendered) pose, which the driver publishes through
 * sb_publish_anchor_pose().
 *
 * @param space Space handle.
 * @param actor Actor handle.
 * @param x     X position in the horizontal plane (Z-UP).
 * @param y     Y position in the horizontal plane (Z-UP).
 * @param yaw   Yaw rotation around the Z-axis (radians).
 * @return Async handle.
 */
SB_EXPORT sb_handle_t sb_update_actor_state(sb_space_t space, sb_actor_t actor,
                                            double x, double y, double yaw);

/**
 * @brief sb_update_actor_state with a declared orientation type (ABI 3.1).
 *
 * `orientation->type` selects yaw only, yaw + pitch, or yaw + pitch + roll.
 * Pitch and roll ride the source update to the receiver and are reported in its
 * proposal (sb_proposal_t::source_pitch / source_roll) as a pass-through: core
 * never solves them. A pitch/roll change alone is a reason to send an update.
 * sb_update_actor_state is the SB_ORIENTATION_YAW case (pitch and roll 0).
 * A NULL or invalid orientation (bad struct_size or type) updates nothing and
 * returns the completed handle.
 */
SB_EXPORT sb_handle_t sb_update_actor_pose(sb_space_t space, sb_actor_t actor,
                                           double x, double y,
                                           const sb_orientation_t* orientation);

SB_EXPORT int32_t sb_enable_actor_observation_frames(sb_session_t session);
SB_EXPORT uint64_t sb_begin_actor_observation_epoch(
    sb_space_t space, sb_actor_t actor, const char* root_anchor_name);
SB_EXPORT int32_t sb_apply_actor_observation_frame(
    sb_space_t space, sb_actor_t actor, uint64_t epoch,
    int64_t frame_sequence, double time_seconds,
    const sb_actor_observation_t* observations, int32_t count);
SB_EXPORT int32_t sb_clear_actor_observation_anchor(
    sb_space_t space, sb_actor_t actor, uint64_t epoch,
    const char* anchor_name);
SB_EXPORT int32_t sb_get_actor_observation_clock(
    sb_space_t space, sb_actor_t actor, uint64_t* epoch,
    int64_t* frame_sequence, double* time_seconds, int32_t* has_sample);

/**
 * @brief Last sequence actually emitted by a source actor.
 *
 * The value advances only when sb_update_actor_state propagated a relation
 * update; deadband-held calls leave it unchanged. `space` and `actor` are C
 * handles in the same local session namespace. Returns 0 for invalid handles
 * or before the actor has emitted an update.
 */
SB_EXPORT int64_t sb_get_last_emitted_update_sequence(sb_space_t space,
                                                      sb_actor_t actor);

/**
 * @brief Last exact source-actor sequence accepted by a receiver.
 *
 * `receiver_space` is the receiving C-space handle. `source_space` and
 * `source_actor` name the source C-space/actor handle; the implementation
 * maps them to RemoteRelationUpdate's wire `{source_space_id,
 * actor_shared_id}` identity. Returns 0 for invalid/unidentified actors or
 * when no sequence has been accepted. This is diagnostic progress only and
 * does not poll, replay, or mutate realization targets.
 */
SB_EXPORT int64_t sb_get_last_applied_remote_sequence(
    sb_space_t receiver_space, sb_space_t source_space, sb_actor_t source_actor);

/**
 * @brief Read the 2D pose of an actor as the space shows it.
 *
 * A source actor: the position and yaw last set via sb_update_actor_state().
 * A receiver avatar: the driver's latest published sample
 * (the root anchor, sb_publish_anchor_pose), (0, 0, 0) before one exists. Core adds no pose
 * of its own.
 *
 * @param space Space handle.
 * @param actor Actor handle.
 * @return sb_actor_state_t with position_x, position_y, and yaw (radians).
 */
SB_EXPORT sb_actor_state_t sb_get_actor_state(sb_space_t space,
                                               sb_actor_t actor);

/**
 * @brief The space-wide floor-footprint radius (m) of the actors that register no embodiment. The radius belongs to
 *        the DRIVER: the core's free space is the walkable minus the obstacle footprints, with no
 *        body-radius erosion, and an unset radius is not an error. The radius serves the driver's course
 *        (sb_query_course keeps a body of the actor's radius clear of the footprints; 0 when none is registered).
 *        The NAVIGABLE polygon is used as given.
 * @return 0 on success; -1 for an invalid handle or radius.
 */
SB_EXPORT int32_t sb_set_body_radius(sb_space_t space, double radius_m);
/* ABI 4.7: the radius is PER ACTOR. An embodiment registered with `body.floor_footprint_radius_m` (and its provenance)
 * declares that actor's radius, for its driver; avatars of different sizes coexist in one space (no refusal, no
 * averaging). An actor that registers no radius uses the space-wide one of sb_set_body_radius. */

/* --------------------------------------------------------------------------
 * Binding
 * -------------------------------------------------------------------------- */

/**
 * @brief Bind an actor anchor to a relation family (Spacing or Facing; any
 *        kernel is accepted, AffordancePose untagged).
 *
 * Scope belongs to the relation family entirely: Spacing is the containment tree plus its border, Facing is every
 * visible referent, AffordancePose is the parent object's region. A bind takes
 * no scope root and no tag. ABI 4.2: the former scope argument is removed.
 *
 * @return 1 when the actor, the anchor and the kernel are valid and the
 *         binding was added; 0 otherwise (nothing is mutated).
 */
SB_EXPORT int32_t sb_bind(sb_space_t space, sb_actor_t actor,
                          const char* anchor_name, sb_kernel_type_t kernel_type);

/**
 * @brief Bind an actor anchor to the affordance kind named by `tag` (e.g. a
 *        pelvis anchor to "Sittable").
 *
 * A tag is ONLY the AffordancePose binding (D6; ABI 4.2 removed the tag from every
 * other bind). `tag` may be null or empty: the anchor then binds every
 * affordance of its parent object's region. ABI 4.2: replaces sb_bind_tagged
 * and sb_bind_scoped.
 *
 * @return 1 on success, 0 otherwise (nothing is mutated).
 */
SB_EXPORT int32_t sb_bind_affordance(sb_space_t space, sb_actor_t actor,
                                     const char* anchor_name, const char* tag);

/* --------------------------------------------------------------------------
 * Realization target callback
 * -------------------------------------------------------------------------- */

/**
 * @brief Callback invoked when new realization targets are available.
 *
 * @param space      Space handle.
 * @param actor      Actor handle.
 * @param targets    Array of realization targets. Valid only during the call.
 * @param num_targets Number of entries in the targets array.
 * @param user_data  Opaque pointer supplied at registration time.
 */
typedef void (*sb_on_targets_t)(sb_space_t space, sb_actor_t actor,
                                const sb_realization_target_t* targets,
                                int64_t num_targets, void* user_data);

/**
 * @brief Register a callback to receive realization targets for actors.
 *
 * Only one callback can be registered per space. Registering a new callback
 * replaces any previously registered callback.
 *
 * @param space     Space handle.
 * @param callback  Function to call when new targets are computed.
 * @param user_data Opaque pointer passed through to each callback invocation.
 */
SB_EXPORT void sb_register_on_targets(sb_space_t space,
                                      sb_on_targets_t callback,
                                      void* user_data);

/**
 * @brief Callback invoked when a peer-shared element is added to this space.
 *
 * Fired when another space shares an element (sb_share_element) and it is
 * received locally. The receiver can use this to reconstruct the shared object
 * (e.g. instantiate a GameObject and run application OnShare logic).
 *
 * @param space          Space handle that received the element.
 * @param local_element  Local element handle created for the shared element.
 * @param shared_id      Cross-space shared identifier of the element.
 * @param tag            Element tag (null-terminated; valid only during call).
 * @param user_data      Opaque payload sent by the sharer (e.g. serialized
 *                       entity). Valid only during the call. NULL if none.
 * @param user_data_size Size of user_data in bytes.
 * @param cb_user_data   Opaque pointer supplied at registration time.
 */
typedef void (*sb_on_element_added_t)(sb_space_t space,
                                      sb_element_t local_element,
                                      int64_t shared_id, const char* tag,
                                      const void* user_data,
                                      int64_t user_data_size,
                                      void* cb_user_data);

/**
 * @brief Register a callback to receive peer-shared elements.
 *
 * Only one callback can be registered per space; registering replaces any
 * previous one. Pass NULL to clear.
 *
 * @param space     Space handle.
 * @param callback  Function to call when a shared element is received.
 * @param user_data Opaque pointer passed through to each invocation.
 */
SB_EXPORT void sb_register_on_element_added(sb_space_t space,
                                            sb_on_element_added_t callback,
                                            void* user_data);

/**
 * @brief Callback invoked after a source-owned received element was removed.
 *
 * LinkToExisting preserves the app's local element and therefore does not
 * invoke this callback (sb_on_link_removed_t reports that link's end). The callback may re-enter the API.
 */
typedef void (*sb_on_element_removed_t)(sb_space_t space,
                                        sb_element_t local_element,
                                        void* user_data);

/** Register (or clear with NULL) the per-space removal callback. */
SB_EXPORT void sb_register_on_element_removed(sb_space_t space,
                                              sb_on_element_removed_t callback,
                                              void* user_data);

/** Why a link was removed by the peer (sb_on_link_removed_t, ABI 4.9). */
typedef enum sb_link_removed_reason_t {
  /** The sharer unshared the source this space had linked a local element to.
   *  The local element survives (LinkToExisting); release the counterpart. An
   *  AddAsNew replica is deleted and reported through sb_on_element_removed_t. */
  SB_LINK_REMOVED_PEER_UNSHARED = 1,
  /** A receiver called sb_unlink on its counterpart of an element this space
   *  shares. The element is still shared. */
  SB_LINK_REMOVED_PEER_UNLINKED = 2,
} sb_link_removed_reason_t;

/**
 * @brief Callback invoked when the peer removed a link whose end in this space
 *        is a surviving element (ABI 4.9).
 *
 * @param local_element The element in this space that was linked (still
 *                      present): the receiver's counterpart for
 *                      SB_LINK_REMOVED_PEER_UNSHARED, the sharer's source
 *                      element for SB_LINK_REMOVED_PEER_UNLINKED.
 * @param peer_space    The peer's space id: the sharer for PEER_UNSHARED, the
 *                      unlinking receiver for PEER_UNLINKED.
 * @param reason        sb_link_removed_reason_t.
 * The callback may re-enter the API; it is not fired for a local sb_unlink.
 */
typedef void (*sb_on_link_removed_t)(sb_space_t space,
                                     sb_element_t local_element,
                                     int64_t peer_space, int32_t reason,
                                     void* user_data);

/** Register (or clear with NULL) the per-space link-removed callback. */
SB_EXPORT void sb_register_on_link_removed(sb_space_t space,
                                           sb_on_link_removed_t callback,
                                           void* user_data);

/**
 * @brief Callback invoked on each relation lifecycle phase transition.
 *
 * @param event     The transition. The struct and its anchor_name pointer are
 *                  valid only for the duration of the callback.
 * @param user_data Opaque pointer supplied at registration time.
 */
typedef void (*sb_on_relation_event_t)(const sb_relation_event_t* event,
                                       void* user_data);

/**
 * @brief Register a callback to receive relation lifecycle events.
 *
 * Only one callback can be registered per space; registering replaces any
 * previous one. The handler must not call back into the same space.
 *
 * @param space     Space handle.
 * @param callback  Function to call on each phase transition.
 * @param user_data Opaque pointer passed through to each invocation.
 */
SB_EXPORT void sb_register_on_relation_event(sb_space_t space,
                                             sb_on_relation_event_t callback,
                                             void* user_data);

/**
 * @brief Dispatch the latest realization targets for an actor, if available.
 *
 * This invokes the callback registered with sb_register_on_targets(). It is
 * useful for consumers that poll from a main thread after remote network
 * updates have been received on a background thread.
 *
 * @param space Space handle.
 * @param actor Actor handle.
 * @return Non-zero if a known target snapshot was dispatched, including an
 * authoritative empty snapshot (callback receives targets=NULL, num_targets=0)
 * that releases prior targets; 0 if no snapshot or callback is available.
 */
SB_EXPORT int32_t sb_poll_realization_targets(sb_space_t space,
                                              sb_actor_t actor);

/**
 * @brief Query the actor's currently grounded relation states.
 *
 * Pulls directly from the same per-binding match data that was just
 * propagated by the most recent sb_update_actor_state() call for this actor
 * — i.e. these values always agree with whatever was last dispatched via
 * sb_register_on_targets() / sb_register_on_relation_event() in the same
 * frame. This is a synchronous, allocation-free (on the caller's side) query,
 * not an event subscription.
 *
 * anchor_name pointers in the output structs point into space-owned storage
 * and are valid only until the next call that mutates this actor's bindings
 * (e.g. the next sb_update_actor_state).
 *
 * @param space  Space handle.
 * @param actor  Actor handle.
 * @param out    Caller-owned array of at least `cap` sb_relation_state_t.
 * @param cap    Capacity of `out`, in elements.
 * @return Number of relation states written into `out` (0 if actor/space
 *         invalid, or the actor currently has no bindings). If the actor has
 *         more relation states than `cap`, only the first `cap` are written
 *         but the full count is still returned (caller should re-query with
 *         a larger buffer if the return value exceeds `cap`).
 */
SB_EXPORT int32_t sb_get_relation_states(sb_space_t space, sb_actor_t actor,
                                         sb_relation_state_t* out,
                                         int32_t cap);

/* --------------------------------------------------------------------------
 * Per-anchor ports, anchor pose proposals and events (ABI 4)
 *
 * Replaces core-side avatar pacing, the ABI 3 single root port/proposal
 * (sb_publish_embodied_pose / sb_read_proposal: removed) and the declared source
 * discontinuity (sb_declare_source_discontinuity: removed). See the block comment
 * in spatialblend_type.h for the ownership contract.
 *
 * Per driver frame:
 *   1. sb_publish_anchor_pose(...)   the driver's current pose of each anchor (the
 *                                    only writer; never waits for a solve).
 *   2. sb_read_proposals(...)        the latest set: one proposal per
 *                                    relation-bearing anchor (root, pelvis, head...).
 *   3. sb_poll_actor_event(...)      source exceptions and teleport requests.
 *   4. Move each body part toward its proposal within the driver's own kinetic
 *      bounds (speed, turn rate, IK).
 *
 * Solves stay SOURCE-triggered: core solves when a source update arrives, on
 * whichever thread delivers it (the caller's thread for an in-process source, the
 * network completion thread for a remote one), using the latest published samples
 * at that moment. A driver frame never causes a solve.
 *
 * Driver contract:
 *   - It heads toward the LATEST proposals within its own kinematic bounds and
 *     publishes the poses it actually has. It keeps no per-cause logic: a proposal
 *     jump is carried at the driver's pace, and the `cause` bits say why it happened
 *     (WEIGHT_SWITCH, REFERENT_SET_CHANGE, FACING_ROOT_SWITCH, CONTAIN_*, SEAT_*...).
 *   - The ONLY teleports are an app-specific reaction to SB_ACTOR_EVENT_SOURCE_EXCEPTION
 *     (the first join is one) and core's SB_ACTOR_EVENT_TELEPORT_REQUEST once the
 *     per-interaction maximum latency elapses for a target that has not arrived.
 *   - Core does NOT continue from the body: the root proposal is a function of the
 *     source's relations and the receiver scene (the softmax-weighted point of the
 *     referents' approach points, clamped to free space; the yaw the conditioned
 *     circular mean of the Facing referents' decoded directions). The embodied
 *     position decides only NO_SAMPLE and connectivity.
 * -------------------------------------------------------------------------- */

/**
 * @brief The id of an anchor of `actor`, interning it on first use. Anchor id 0 is
 *        the actor's root anchor. Ids are stable for the actor's lifetime.
 * @return 0 and *id set; -1 for an invalid handle/actor/NULL.
 */
SB_EXPORT int32_t sb_get_anchor_id(sb_space_t space, sb_actor_t actor,
                                   const char* anchor_name, uint32_t* id);

/**
 * @brief The name of an anchor id, copied NUL-terminated into `buf`.
 * @return The name's length (excluding the NUL; > cap - 1 means truncated); -1 for
 *         an invalid handle/actor/id.
 */
SB_EXPORT int32_t sb_get_anchor_name(sb_space_t space, sb_actor_t actor,
                                     uint32_t anchor_id, char* buf, int32_t cap);

/**
 * @brief Publish the pose of one anchor of the receiver (driver -> core).
 *
 * The driver is the ONLY writer of each anchor's port. The call never takes the
 * space lock and never waits for a solve running on another thread: a seqlock write
 * (the one lock inside is the port's writer mutex, contended only by other
 * writers). Core never writes, integrates or extrapolates the stored sample.
 *
 * An actor that is ALSO a relation source (sb_update_actor_state) may publish here:
 * the source feed is the relation's ground truth, the port carries the pose the
 * avatar is DISPLAYED at. Core mirrors the ROOT sample into the element tree only
 * for receiver actors, never over a source-fed pose. An actor whose pose is owned
 * by a receiver-input stream refuses it (-3).
 *
 * @param pose        x/y/z and orientation in receiver space (Z-up, radians), the
 *                    epoch and flags (SB_EMBODIED_TELEPORTED).
 * @param timestamp_s The driver's own clock for this pose (echoed back in
 *                    sb_proposal_set_t::sample_timestamp_s; not interpreted).
 * @param sequence    Driver sequence, strictly increasing within an epoch of that
 *                    anchor, starting at 1.
 * @return 0 accepted; -1 invalid handle/unknown actor or anchor/NULL/non-finite/
 *         sequence 0/wrong struct_size; -2 stale (sequence not newer in the epoch,
 *         or an older epoch): the stored sample is unchanged; -3 refused (the actor's
 *         pose is owned by a receiver-input stream, sb_begin_receiver_input).
 */
SB_EXPORT int32_t sb_publish_anchor_pose(sb_space_t space, sb_actor_t actor,
                                         uint32_t anchor_id,
                                         const sb_anchor_pose_t* pose,
                                         double timestamp_s, uint64_t sequence);

/* ---------------------------------------------------------------------------
 * Embodiment registration (ABI 4.7).
 *
 * An app or embodiment driver registers its OWN skeleton and declares each subject anchor against it: the
 * bone, the offset in that bone's frame, the relation families it serves, and where the offset came from. The
 * driver then publishes BONES; the core reads an anchor as `bone * offset`, so the offset exists in exactly one
 * place and the sampling port stays the single source of truth for embodied state. Nothing here is a body
 * constant: the spec is data (schema spatialblend.embodiment 1.0.0), a non-identity offset without provenance
 * (procedure, and a 64-hex asset_sha256 or an explicit source) is refused, and an actor without a registration behaves exactly as before.
 * ------------------------------------------------------------------------- */

/**
 * @brief Register the actor's embodiment from its JSON spec.
 * @return 0 registered; -1 invalid handle / NULL / not a JSON object; -2 the spec is invalid (the reason is
 *         copied NUL-terminated into `error` when it is non-NULL and error_cap > 0); -3 the actor already
 *         registered one, a bone-bound anchor already has a driver-published port, an existing binding
 *         violates the declared families (reason "anchor@family" in `error`), or the actor's pose is
 *         owned by a receiver-input stream.
 * Effects, only for this actor: a bone-bound registered anchor is sampled from its bone (sb_publish_anchor_pose
 * of such an anchor returns -3: publish its bone; sb_apply_actor_observation_frame returns 0 for a frame naming one;
 * sb_set_anchor_transform leaves it unchanged); sb_bind / sb_bind_affordance return 0 for a registered
 * anchor and a family it did not declare. Unregistered anchors and the actor root are untouched.
 */
SB_EXPORT int32_t sb_register_embodiment_json(sb_space_t space, sb_actor_t actor, const char* json,
                                              int32_t json_len, char* error, int32_t error_cap);

/**
 * @brief The registered embodiment as canonical JSON (keys sorted, doubles round-trip), NUL-terminated into buf.
 * @return the byte length (excluding NUL) of the full text, which may exceed cap - 1 (then buf holds the
 *         truncated prefix); 0 when the actor has no embodiment; -1 invalid handle.
 */
SB_EXPORT int32_t sb_get_embodiment_json(sb_space_t space, sb_actor_t actor, char* buf, int32_t cap);

/** @return the id of bone `name` of the actor's registered skeleton (its joint index), or -1. */
SB_EXPORT int32_t sb_get_bone_id(sb_space_t space, sb_actor_t actor, const char* name);

/**
 * @brief RECEIVER: publish the pose of one bone (driver -> core). The driver is the only writer of the bone's
 *        port; the contract and the return codes are those of sb_publish_anchor_pose (0 accepted, -1 invalid,
 *        -2 stale, -3 refused).
 */
SB_EXPORT int32_t sb_publish_bone_pose(sb_space_t space, sb_actor_t actor, int32_t bone_id,
                                       const sb_bone_pose_t* pose, double timestamp_s, uint64_t sequence);

/**
 * @brief SOURCE: sb_apply_actor_observation_frame for an actor with an embodiment. `bones` are the world poses
 *        of the bones the app tracked this frame (by name); every registered bone-bound anchor whose bone is
 *        present is composed from its registration and observed at that pose. `observations` carries the
 *        anchors that are not bone-bound (the planar root). An anchor of the registration whose bone is absent
 *        is not observed this frame, as an omitted observation. Needs sb_enable_actor_observation_frames.
 * @return 1 applied, 0 refused (as sb_apply_actor_observation_frame; also an unknown bone, a repeated bone, or a
 *         bone-bound anchor named in `observations`).
 */
SB_EXPORT int32_t sb_apply_actor_bone_frame(sb_space_t space, sb_actor_t actor, uint64_t epoch,
                                            int64_t frame_sequence, double time_seconds,
                                            const sb_named_bone_pose_t* bones, int32_t bone_count,
                                            const sb_actor_observation_t* observations,
                                            int32_t observation_count);

/**
 * @brief SOURCE (legacy sessions: a host or client session, with sb_update_actor_state): feed the registered, bone-bound anchors
 *        from the bones the app tracked, ABI 4.10. Every bone-bound anchor whose bone is present is composed from its registration
 *        (anchor = bone * offset, the core's own arithmetic) and set as the anchor's world pose, as sb_set_anchor_transform does for an
 *        unregistered anchor (which refuses a bone-bound one). `bones` are world poses by bone name, Z-up. Anchors whose bone is absent
 *        keep their last pose. Unlike sb_apply_actor_bone_frame it needs no observation-frame session, so it works with the networked
 *        FC session and with the legacy updates other actors make in the same session; the next sb_update_actor_state carries the
 *        relations. One entry point for every bone-bound anchor family (pelvis, hands): nothing here is seat- or board-specific.
 * @return the number of anchors set (0 when no registered anchor's bone is present); -1 for an invalid handle or argument, an
 *         actor without an embodiment, an unknown or repeated bone name, or a bad pose.
 */
SB_EXPORT int32_t sb_set_bone_anchor_transforms(sb_space_t space, sb_actor_t actor, const sb_named_bone_pose_t* bones,
                                                int32_t bone_count);

/**
 * @brief The registration's own arithmetic, so a driver or a tool never re-implements it: anchor = bone * offset
 *        and bone = anchor * offset^-1. Pure: reads the registration, writes only the output.
 * @return 0; -1 for an invalid handle, NULL, a wrong struct_size, or an anchor that is not registered and bone-bound.
 */
SB_EXPORT int32_t sb_embodiment_anchor_pose_from_bone(sb_space_t space, sb_actor_t actor, const char* anchor_name,
                                                      const sb_bone_pose_t* bone, sb_bone_pose_t* anchor_out);
SB_EXPORT int32_t sb_embodiment_bone_pose_from_anchor(sb_space_t space, sb_actor_t actor, const char* anchor_name,
                                                      const sb_bone_pose_t* anchor, sb_bone_pose_t* bone_out);

/* ---- the normalized kinematic chain (ABI 4.7; defined and stored only, no solve reads it) ----
 *
 * The spec's optional "chain" object registers, with the
 * skeleton, each joint's REST pose in the core's canonical axes (z up, x forward, y left, right-handed; metres,
 * radians; the first root joint at the origin; the embodiment's own bind pose, labelled, never converted between
 * A and T) and optional joint limits. It travels in the same sb_register_embodiment_json call and comes back in
 * sb_get_embodiment_json; the typed read-back below is the same data. Reachability stays the driver's judgement. */

/** @return the number of chain joints (the skeleton's), 0 when the actor registered no chain, -1 invalid handle. */
SB_EXPORT int32_t sb_get_chain_joint_count(sb_space_t space, sb_actor_t actor);

/** @brief Joint `index` of the chain (and its name, NUL-terminated, truncated to name_cap; name may be NULL).
 *  @return 0, or -1 for an invalid handle, no chain, an index out of range, or a wrong out->struct_size. */
SB_EXPORT int32_t sb_get_chain_joint(sb_space_t space, sb_actor_t actor, int32_t index, sb_chain_joint_t* out,
                                     char* name, int32_t name_cap);

/**
 * @brief Convert a pose between a skeleton's native axes and the canonical ones, so no driver re-implements it.
 *        `basis` is row-major 3x3, c = B n, a signed permutation (exactly one +-1 per row and column); det -1 is a
 *        handedness change (Unity): positions convert as B p and rotations as B R B^T (still proper).
 * @param to_canonical non-zero: native -> canonical; zero: canonical -> native.
 * @return 0; -1 for NULL, a wrong struct_size, or a basis that is not a signed permutation.
 */
SB_EXPORT int32_t sb_convert_pose_basis(const double* basis, int32_t to_canonical, const sb_bone_pose_t* in,
                                        sb_bone_pose_t* out);

/**
 * @brief Read the latest proposal set of a receiver actor: one proposal per
 *        relation-bearing anchor, the root first, all from one solve.
 *
 * One consistent snapshot per call (the lock behind it is held for a pointer copy,
 * never across a solve). `set->revision` increments with every set, so a driver
 * polling per frame can tell a new set from the one it already applied. The root
 * proposal's status says whether a pose could be proposed (SB_PROPOSAL_NO_SAMPLE:
 * publish the root pose first; the next source update then solves).
 *
 * @param set `set->struct_size` must be set to sizeof(sb_proposal_set_t).
 * @param out Caller array of `cap` sb_anchor_proposal_t, each with struct_size set
 *            to sizeof(sb_anchor_proposal_t). May be NULL when cap is 0.
 * @return 0 filled (set->anchor_count written; set->total_anchor_count may exceed
 *         cap); -1 invalid handle/unknown actor/NULL/wrong struct_size.
 */
SB_EXPORT int32_t sb_read_proposals(sb_space_t space, sb_actor_t actor,
                                    sb_proposal_set_t* set,
                                    sb_anchor_proposal_t* out, int32_t cap);

/**
 * @brief ABI 4.11: the inner level of one anchor's latest proposal (see sb_anchor_solve_t).
 * @param out `out->struct_size` must be sizeof(sb_anchor_solve_t).
 * @return 1 written; 0 the latest proposal set has no such anchor; -1 invalid handle/unknown actor/NULL/wrong size.
 */
SB_EXPORT int32_t sb_read_anchor_solve(sb_space_t space, sb_actor_t actor, uint32_t anchor_id,
                                       sb_anchor_solve_t* out);

/**
 * @brief ABI 4.12: the signed, unwrapped yaw turn of one anchor's latest proposal (see sb_anchor_turn_t).
 * @param out `out->struct_size` must be sizeof(sb_anchor_turn_t).
 * @return 1 written; 0 the latest proposal set has no such anchor; -1 invalid handle/unknown actor/NULL/wrong size.
 */
SB_EXPORT int32_t sb_read_anchor_turn(sb_space_t space, sb_actor_t actor, uint32_t anchor_id,
                                      sb_anchor_turn_t* out);

/**
 * @brief ABI 4.14: the replay through the chart of one anchor of the latest proposal set (see sb_anchor_chart_t).
 * @param out `out->struct_size` must be sizeof(sb_anchor_chart_t).
 * @return 1 written; 0 the latest set has no such anchor, or the anchor has no chart transform (its source anchor is
 *         not inside a transfer leaf this space maps); -1 invalid handle/unknown actor/NULL/wrong size.
 */
SB_EXPORT int32_t sb_read_anchor_chart(sb_space_t space, sb_actor_t actor, uint32_t anchor_id,
                                       sb_anchor_chart_t* out);

/**
 * @brief The source app raises an app-specific exception (a teleport, a button next
 *        to a whiteboard...) with an opaque payload.
 *
 * It rides the next update this actor propagates, which it forces out through the
 * movement gate and the heartbeat, and every receiving space gets an
 * SB_ACTOR_EVENT_SOURCE_EXCEPTION carrying the id and the payload. The proposals of
 * that solve are the absolute relation and carry SB_PROPOSAL_CAUSE_SOURCE_EXCEPTION.
 * Core never interprets the payload; the receiving app realizes it (teleport,
 * animation, ...). The first update of every source actor carries the implicit
 * exception SB_EXCEPTION_JOIN with an empty payload; a receiver whose first placed
 * proposal answers a later update (its body had no sample then, or its space joined
 * the session later) gets SB_EXCEPTION_JOIN with that placed set instead, so every
 * receiver's join placement is announced exactly once. This is the only teleport path
 * besides core's latency request.
 *
 * @param payload      May be NULL when payload_size is 0.
 * @param payload_size At most SB_EXCEPTION_PAYLOAD_MAX.
 * @return 0 queued; -1 invalid handle/unknown actor/oversize payload.
 */
SB_EXPORT int32_t sb_raise_source_exception(sb_space_t space,
                                            sb_actor_t source_actor,
                                            uint32_t exception_id,
                                            const void* payload,
                                            uint32_t payload_size);

/**
 * @brief ABI 4.13: opt into independent position/yaw target events through
 * sb_poll_actor_event. Disabled by default: legacy consumers see only source
 * exceptions and latency teleports. Validate ABI minor >= 13 and the typed
 * payload size/field offsets before enabling. enabled is exactly 0/1. Repeating
 * the current state is idempotent. A transition to enabled, including re-enable
 * after a driver reload, refuses queued legacy source exceptions that lack
 * their own immutable target batches; refusal leaves subscription and queue
 * unchanged. Admit before source activity, or first consume those legacy
 * exceptions under their legacy contract. A successful transition queues
 * per-anchor releases and makes the next source evaluation a fresh placed
 * answer; it never replays a consumed sample-relative turn. Disabling affects
 * future typed deliveries only: already queued events, releases and immutable
 * lifecycle groups remain drainable. The app clears its own pending targets
 * when it abandons a subscription. This is a subscriber-lifetime operation, NOT
 * an ordinary yaw-correspondence recovery/teleport mechanism. Ordinary pending
 * motion uses at most one slot per anchor/channel, replacing older ordinary
 * targets. Real lifecycle events/releases and their immutable placement
 * requests are lossless and ordered, so storage is proportional to the number
 * of unpolled lifecycle transitions, not bounded under an indefinitely stalled
 * consumer with repeated resets. A client must drain the stream; no lifecycle
 * event is silently discarded to enforce a motion queue capacity. No
 * full-snapshot target callback/flag meaning changes; an empty known callback
 * still releases the entire snapshot.
 * @return 0 success; -1 invalid space/actor/enabled or refused legacy backlog.
 */
SB_EXPORT int32_t sb_set_target_event_delivery(sb_space_t space,
                                               sb_actor_t actor,
                                               int32_t enabled);

/**
 * @brief Pop the oldest pending event of a receiver actor, after its solve
 * lock. This call waits for an in-flight solve/publication. With at most one
 * newer accepted publication per anchor since the previous completed drain,
 * draining to native return 0 fences every solve that could have sampled the
 * previous accepted pose; future solves sample the current pose. This is the
 * retirement boundary for previous/current sample correspondence, NOT local
 * epoch changes. See sb_target_request_payload_t for the full single-consumer
 * protocol. Native errors and consumer conversion errors do not complete a
 * drain. Do not call from a synchronous proposal observer/callback: it takes
 * the solve lock. Native C++ observers may use the separate callback-safe
 * PollActorEvent queue pop, which does not provide this solve fence. Port
 * feedback remains independently publishable while this call waits.
 * @param out `out->struct_size` must be set to sizeof(sb_actor_event_t).
 * @return 1 an event was written; 0 none pending; -1 invalid handle/unknown
 * actor/ NULL/wrong struct_size.
 */
SB_EXPORT int32_t sb_poll_actor_event(sb_space_t space, sb_actor_t actor,
                                      sb_actor_event_t* out);

/**
 * @brief Stateless course query: the traversable polyline between two points
 *        in the actor's obstacle-aware free space (the geometry service the
 *        driver paces along). Reads no pose and keeps no state.
 *
 * @param xy_out Receives up to `cap` waypoints as (x, y) pairs, starting at
 *               (from_x, from_y) and ending at (to_x, to_y), EXCEPT when the goal lies
 *               within the clearance ring of a body of the actor's registered radius
 *               around a footprint (body clearance is the driver's;
 *               0 radius when none is registered): the course then ends at the nearest
 *               clear point, and a driver treats reaching it as arrival.
 * @return the course's waypoint count (min(count, cap) are written), 0 when
 *         no traversable course exists, -1 for an invalid handle/NULL/cap < 2.
 */
SB_EXPORT int32_t sb_query_course(sb_space_t space, sb_actor_t actor,
                                  double from_x, double from_y, double to_x,
                                  double to_y, double* xy_out, int32_t cap);

/**
 * @brief DRIVER service (ABI 4.8, additive): where a driver that PLACES its body (a first placement, a teleport on a
 *        source exception) puts it so it is clear of the footprints: the nearest point of the free space of a body of
 *        the actor's registered radius (sb_query_course's clearance; 0 when none is registered). The core's own
 *        proposal never applies a body radius and is never moved by this.
 * @param xy_out Receives (x, y).
 * @return 1 on success; 0 when the free space has no such point; -1 for an invalid handle, a null `xy_out` or a
 *         non-finite point.
 */
SB_EXPORT int32_t sb_query_nearest_clear_point(sb_space_t space, sb_actor_t actor, double x, double y,
                                               double* xy_out);

/* --------------------------------------------------------------------------
 * Update interval
 * -------------------------------------------------------------------------- */

/**
 * @brief Set the interval at which the solver updates realization targets.
 * @param space   Space handle.
 * @param seconds Update interval in seconds. Must be positive.
 */
SB_EXPORT void sb_set_update_interval(sb_space_t space, double seconds);

/* --------------------------------------------------------------------------
 * Global cleanup
 * -------------------------------------------------------------------------- */

/**
 * @brief Release all resources across all sessions and spaces.
 *
 * Call this once during application shutdown. After this call, all previously
 * returned handles are invalid.
 */
SB_EXPORT void sb_cleanup_all();

/* --------------------------------------------------------------------------
 * Shared-element resolution
 * -------------------------------------------------------------------------- */

/**
 * @brief Resolve a received shared element.
 *
 * Called by the app after receiving an sb_on_element_added callback to
 * communicate the resolution decision back to native:
 *   - AddAsNew:      pass received == resolved to keep the temp element and
 *                    link source -> received in the hub's ElementLinker.
 *   - LinkToExisting: pass resolved = an existing local element id to remove
 *                    the temp duplicate and link source -> resolved.
 *
 * @param space      Space handle that received the element.
 * @param received   Local element handle returned in the on_element_added
 *                   callback (the temp element created by native).
 * @param resolved   The element id to use as the canonical local target.
 *                   Pass the same value as received for AddAsNew.
 */
SB_EXPORT void sb_resolve_shared_element(sb_space_t space,
                                         sb_element_t received,
                                         sb_element_t resolved);

/* --------------------------------------------------------------------------
 * Runtime configuration
 *
 * These setters give a client explicit, per-space control over knobs that
 * were previously only settable via process-wide getenv() flags read once at
 * first use (SB_V2_REFERENT_RESOLVE, SB_V2_WEIGHT_SMOOTHING, SB_V2_MIN_MOVE,
 * SB_V2_MIN_YAW, SB_V2_REALIZATION_POLICY). The env vars remain the default
 * source: if a setter is never called, the space still lazily resolves its
 * value from the environment on first use, exactly as before. Calling a
 * setter pre-empts (and permanently overrides) that lazy env resolution for
 * the affected knob, on that space, for its remaining lifetime.
 * -------------------------------------------------------------------------- */

/**
 * @brief Set how OnRemoteUpdate resolves an incoming payload's referent to a
 * local element (see sb_referent_resolve_mode_t). Mirrors env
 * SB_V2_REFERENT_RESOLVE=linker|sharedid|linker_only.
 * @param space Space handle.
 * @param mode  Resolve mode.
 */
SB_EXPORT void sb_set_referent_resolve_mode(sb_space_t space,
                                            sb_referent_resolve_mode_t mode);

/**
 * @brief Set the EMA smoothing factor applied to per-referent engagement
 * weights before they drive realization targets. Mirrors env
 * SB_V2_WEIGHT_SMOOTHING (a value in [0, 1); 0 disables smoothing).
 * @param space Space handle.
 * @param alpha Smoothing factor.
 */
SB_EXPORT void sb_set_weight_smoothing(sb_space_t space, double alpha);

/**
 * @brief Set the minimum linear movement (meters) an actor must move before
 * its update is propagated to peer spaces (movement-gate deadband). Mirrors
 * env SB_V2_MIN_MOVE. Values < 0 are clamped to 0.
 * @param space  Space handle.
 * @param meters Minimum move distance in meters.
 */
SB_EXPORT void sb_set_min_move(sb_space_t space, double meters);

/**
 * @brief Set the minimum yaw change (radians) an actor must rotate before its
 * update is propagated to peer spaces (movement-gate deadband). Mirrors env
 * SB_V2_MIN_YAW. Values < 0 are clamped to 0.
 * @param space   Space handle.
 * @param radians Minimum yaw change in radians.
 */
SB_EXPORT void sb_set_min_yaw(sb_space_t space, double radians);

/**
 * @brief Set the realization weighting policy family (see
 * sb_realization_policy_t). Mirrors env
 * SB_V2_REALIZATION_POLICY=progressive|uniform|greedy.
 * @param space  Space handle.
 * @param policy Realization policy.
 */
SB_EXPORT void sb_set_realization_policy(sb_space_t space,
                                         sb_realization_policy_t policy);

/**
 * @brief Lifecycle labels for one relation family: the thresholds of the
 * DRIVING and CONTACT phases reported in relation events.
 *
 * ABI 4: the activation thresholds, the near field, the tracking-gap dwell and
 * the two scope radii are DELETED. Scope
 * belongs to the relation family; a relation in scope is realized, and enters
 * ACTIVE the moment it is observed. Nothing here narrows scope or withholds a
 * relation.
 *
 * Versioned: set struct_size = sizeof(sb_relation_phase_config_t)
 * (sb_sizeof(SB_STRUCT_RELATION_PHASE_CONFIG)); a smaller size is rejected.
 * Weight thresholds are on the relation's in-group realization weight, the
 * contact pair on its normalized proximity R in [0,1].
 * Defaults: driving 0.50/0.40 and contact 0.95/0.85 for every family.
 */
typedef struct {
  uint32_t struct_size;
  uint32_t reserved;
  double driving_weight;
  double driving_release;
  double contact_proximity;
  double contact_release;
} sb_relation_phase_config_t;

/**
 * @brief Set and/or read a relation family's lifecycle thresholds.
 * @param space         Space handle.
 * @param kernel_type   Relation family.
 * @param config        New thresholds, or NULL to only read the current ones.
 *                      Rejected (nothing changes) unless every enter/release
 *                      pair is finite and ordered: driving_release <=
 *                      driving_weight and contact_release <=
 *                      contact_proximity.
 * @param out_effective Optional; receives the thresholds in force after the
 *                      call (also after a rejection). The caller MUST set
 *                      out_effective->struct_size to its buffer size; only
 *                      that many bytes are written (struct_size is updated
 *                      to the bytes written). A size below the 8-byte header
 *                      returns 0 and writes nothing.
 * @return 1 on success (or a successful read), 0 on an invalid handle, family
 *         or config.
 */
SB_EXPORT int32_t sb_set_relation_phase_config(
    sb_space_t space, sb_kernel_type_t kernel_type,
    const sb_relation_phase_config_t* config,
    sb_relation_phase_config_t* out_effective);

/**
 * @brief Set the horizontal reach radius (metres) used to clamp a named
 * anchor's (e.g. "hand", "pelvis") RealizationTarget around the actor's
 * already-solved root target, so an anchor target can never place a body
 * part farther from the avatar's own root than is physically plausible.
 * Values < 0 are clamped to 0. Anchors with no explicit radius set use a
 * built-in default (hand: 0.9m, pelvis: 0.35m).
 * @param space       Space handle.
 * @param anchor_name Null-terminated anchor name (e.g. "hand").
 * @param radius      Reach radius in metres.
 */
SB_EXPORT void sb_set_anchor_reach(sb_space_t space, const char* anchor_name,
                                   double radius);

/**
 * @brief Enable or disable structured JSON-lines tracing (format:
 * trace.format.md) for `space` specifically.
 *
 * This is genuinely PER-SPACE state -- calling it binds `space`'s
 * own space_id to its own dedicated sink, independent of every other space's
 * tracing. A space that never gets its own sb_set_trace call keeps falling
 * back to the process-wide default sink (mirrors env SB_V2_TRACE /
 * SB_V2_TRACE_FILE), unaffected by any other space's sb_set_trace calls.
 * Passing the SAME path to two different spaces reproduces the behavior
 * before per-space routing (both spaces' lines interleaved in one shared file); passing
 * different paths now genuinely separates them into different files --
 * previously `space` only validated the call (every space shared one
 * process-wide sink regardless of which handle was passed), so the only way
 * to tell two spaces' lines apart was the "space" field in each JSON line.
 * @param space Space handle whose tracing this call controls.
 * @param path  Null-terminated file path to write trace lines to, or NULL to
 *              disable tracing for this space. If the file cannot be
 *              opened, tracing falls back to stderr.
 */
SB_EXPORT void sb_set_trace(sb_space_t space, const char* path);

/**
 * @brief Return the ABI version of this build of the core.
 *
 * Bump the returned value whenever a breaking C-ABI change is made (struct
 * layout, enum values, function signatures). Clients can compare this
 * against the version they were built against and fail fast on mismatch.
 * ABI v1 is legacy and incompatible.
 * @return ABI version (currently 4: per-anchor sampling ports and one pose
 *         proposal per relation-bearing anchor replace the root proposal and its
 *         containment / seat / placement side channels; source-app exceptions
 *         replace the declared discontinuity).
 */
SB_EXPORT uint32_t sb_abi_version();

/** ABI minor version: 14 for ABI 4.14. 4.14 adds sb_read_anchor_chart and
 * sb_anchor_chart_t (SB_STRUCT_ANCHOR_CHART): the per-anchor replay through the
 * chart of a containment-transfer leaf, an output beside the proposal (additive;
 * a new optional wire field; no struct or existing signature changed). 4.13 added opt-in
 * sb_set_target_event_delivery, typed position/yaw request and release events
 * through sb_poll_actor_event, and sb_target_request_payload_t
 * (SB_STRUCT_TARGET_REQUEST_PAYLOAD). Existing event/proposal layouts and
 * full-snapshot callback release semantics are unchanged. 4.12 added
 * sb_read_anchor_turn and sb_anchor_turn_t (SB_STRUCT_ANCHOR_TURN): the signed,
 * unwrapped yaw turn of an anchor's proposal, which may exceed pi (additive;
 * sb_anchor_proposal_t keeps its size: its one spare field is a uint32_t, too
 * small for the turn; no struct or existing signature changed). 4.11 added the
 * affordance poses on interaction-role children and sb_read_anchor_solve /
 * sb_anchor_solve_t. 4.10 added sb_set_sub_region_chart, sb_part_chart_encode /
 * _decode and sb_part_chart_coord_t (additive: per-part surface charts, one
 * affordance-pose mapping for boards and seats; a new optional wire field; no
 * struct or existing signature changed). 4.9 added sb_unlink and
 * sb_register_on_link_removed / sb_on_link_removed_t / sb_link_removed_reason_t
 * (additive: unlink across spaces and the link-removed signal to the peer; no
 * struct or existing signature changed). 4.8 added sb_reparent_element and
 * sb_query_nearest_clear_point (additive). 4.7 added embodiment registration:
 * sb_register_embodiment_json, sb_get_embodiment_json, sb_get_bone_id,
 * sb_publish_bone_pose, sb_apply_actor_bone_frame,
 *  sb_embodiment_anchor_pose_from_bone / _bone_pose_from_anchor, the normalized
 * kinematic chain (sb_get_chain_joint_count, sb_get_chain_joint,
 * sb_convert_pose_basis, sb_chain_joint_t) and sb_bone_pose_t /
 * sb_named_bone_pose_t (additive; behaviour changes only for an actor that
 * registers). Minor 6 is reserved for part correspondence, which
 * lands after this and renumbers nothing: the minor is checked for equality, so
 * the gap is harmless. No struct or existing signature changed. 4.5 added the
 * tunneled core link: sb_host_space_tunneled, sb_accept_tunnel,
 * sb_connect_space_tunneled, sb_tunnel_write/read/close and sb_tunnel_t
 *  (hosting and joining with no address; the app carries the link bytes).
 * sb_try_wait and sb_get_error now also serve the async handle of
 * sb_connect_space_tunneled. sb_tunnel_write returns 2 (would block, nothing
 * taken) under backpressure. No struct or signature changed. 4.4 added
 * sb_get_source_transfer (the source-side transfer fact) and moved shape
 * congruence into the per-update transfer flag. 4.3 added
 * SB_ANCHOR_PROPOSAL_FLAG_TRANSFER (the per-update transfer flag of the root
 * proposal) and removed the contained event (sb_register_on_containment_event,
 * sb_on_containment_event_t). 4.1 added
 * SB_ANCHOR_PROPOSAL_FLAG_POSITION_UNSPECIFIED; 4.2 removed the scope and
 * tag arguments: sb_bind(space, actor, anchor, kernel) and
 *  sb_bind_affordance(space, actor, anchor, tag) replace sb_bind /
 * sb_bind_tagged / sb_bind_scoped. No struct changed. */
SB_EXPORT uint32_t sb_abi_minor_version();

/**
 * @brief Fill and return the canonical layout descriptor for the native C ABI.
 *
 * All-or-nothing contract: if @p layout is NULL or @p layout_size is too
 * small, this writes nothing and returns the required descriptor size.
 *
 * @param layout Output storage for the descriptor.
 * @param layout_size Number of bytes available in @p layout.
 * @return Required descriptor size in bytes.
 */
SB_EXPORT uint32_t sb_get_v2_abi_layout(sb_v2_abi_layout_t* layout,
                                        uint32_t layout_size);

/**
 * @brief Return sizeof() for a C-ABI struct identified by sb_struct_id_t.
 *
 * Lets a client (e.g. the C# marshal layer) assert its own
 * [StructLayout(Sequential)] mirror is the same size as the core's struct at
 * load time, catching ABI drift early rather than via silent corruption.
 * @param struct_id Struct to query.
 * @return sizeof() the identified struct, or 0 if struct_id is unrecognized.
 */
SB_EXPORT size_t sb_sizeof(sb_struct_id_t struct_id);

#ifdef __cplusplus
}
#endif

#endif  // SPATIALBLEND_C_SPATIALBLEND_H_
