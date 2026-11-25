# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Static visualization for spatialblend spaces."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch

KERNEL_COLORS = {
    "SPACING": "#2ecc71",       # green
    "FACING": "#3498db",        # blue
    "AFFORDANCE_POSE": "#e67e22",  # orange
}


def _draw_polygon(ax, points, **kwargs):
    """Draw a closed polygon from a list of [x,y] points."""
    pts = np.array(points)
    polygon = plt.Polygon(pts, **kwargs)
    ax.add_patch(polygon)
    return polygon


def _draw_actor(ax, position, yaw, label="", color="red", size=0.5):
    """Draw an actor as an arrow showing position + direction."""
    dx = size * np.cos(yaw)
    dy = size * np.sin(yaw)
    ax.annotate("", xy=(position[0] + dx, position[1] + dy),
                xytext=(position[0], position[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.plot(position[0], position[1], "o", color=color, markersize=8)
    if label:
        ax.annotate(label, (position[0], position[1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color=color)


def _draw_element(ax, element, tree, color="blue"):
    """Draw an element at its world position."""
    world_t = tree.get_world_transform(element.id)
    pos = [world_t[0, 3], world_t[1, 3]]
    spatial_def = element.get_spatial_definition()

    # Draw boundary if available
    if spatial_def is not None:
        boundary = spatial_def.get_boundary()
        if boundary is not None:
            pts = boundary.get_points()
            if len(pts) >= 3:
                _draw_polygon(ax, pts, fill=False,
                              edgecolor=color, linestyle="--", linewidth=1)

    # Label
    tag = element.tag
    role_name = element.geometry_role.name
    label = f"{tag if tag else role_name}\n(id={element.id})"
    ax.annotate(label, pos, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                          alpha=0.2))


def _draw_relation_line(ax, anchor_pos, contact_pos, proximity,
                        color="green", kernel_type=None):
    """Draw a relation line from anchor to contact with thickness = proximity."""
    if kernel_type is not None:
        kt_name = kernel_type if isinstance(kernel_type, str) else kernel_type.name
        color = KERNEL_COLORS.get(kt_name, color)
    lw = max(0.5, proximity * 4)
    ax.plot([anchor_pos[0], contact_pos[0]],
            [anchor_pos[1], contact_pos[1]],
            color=color, linewidth=lw, alpha=0.6)
    # Proximity label
    mid = [(anchor_pos[0] + contact_pos[0]) / 2,
           (anchor_pos[1] + contact_pos[1]) / 2]
    ax.annotate(f"p={proximity:.2f}", mid, fontsize=6, color=color)


def _collect_all_element_ids(tree, root_id):
    """Collect all element IDs by recursively walking the tree from root_id."""
    ids = [root_id]
    children = tree.get_direct_child_ids(root_id)
    for child_id in children:
        ids.extend(_collect_all_element_ids(tree, child_id))
    return ids


def space(space_obj, actor_id=None, title="Space", ax=None, root_id=1):
    """Render a single space with elements and optional actor.

    Args:
        space_obj: Space instance.
        actor_id: Optional actor ID to draw.
        title: Plot title.
        ax: Optional matplotlib Axes; created if None.
        root_id: Root element ID to start tree traversal from (default 1).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    tree = space_obj.element_tree
    colors = {
        "NAVIGABLE": "gray",
        "CONTAINMENT": "lightblue",
        "OBSTACLE": "slategray",
        "SURFACE": "steelblue",
        "INTERACTION_REGION": "orange",
        "UNSPECIFIED": "steelblue",
    }

    # Collect all element IDs by walking tree from root
    root_elem = tree.get_element(root_id)
    if root_elem is not None:
        all_ids = _collect_all_element_ids(tree, root_id)
        for elem_id in all_ids:
            elem = tree.get_element(elem_id)
            if elem is None:
                continue
            role_name = elem.geometry_role.name
            color = colors.get(role_name, "gray")

            # Draw navigable polygon as filled background
            if role_name == "NAVIGABLE":
                spatial_def = elem.get_spatial_definition()
                if spatial_def is not None:
                    boundary = spatial_def.get_boundary()
                    if boundary is not None:
                        pts = boundary.get_points()
                        if len(pts) >= 3:
                            _draw_polygon(ax, pts, fill=True,
                                          facecolor="whitesmoke",
                                          edgecolor="black", linewidth=2)
            else:
                _draw_element(ax, elem, tree, color=color)

    # Draw actor
    if actor_id is not None:
        actor = space_obj.get_actor(actor_id)
        if actor is not None:
            pos = actor.get_position()
            t = actor.get_transform()
            yaw = np.arctan2(t[1, 0], t[0, 0])
            _draw_actor(ax, pos, yaw, label="Actor", color="red")

            # Draw bindings
            bindings = space_obj.get_bindings(actor_id)
            for binding in bindings:
                for match in binding.latest_matches:
                    support = match.eval.support
                    if support.has_contact:
                        contact_world = support.world_transform @ np.array([
                            support.contact_point_local[0],
                            support.contact_point_local[1], 0, 1])
                        contact_2d = contact_world[:2]
                    else:
                        contact_2d = support.get_position_2d()

                    proximity = match.eval.grounded.proximity
                    _draw_relation_line(ax, pos, contact_2d, proximity)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.autoscale()
    return fig


def cross_space(space_a, space_b, actor_id, avatar_id,
                title="Cross-Space", root_id=1):
    """Render two spaces side by side with actor and avatar.

    Args:
        space_a: Source space.
        space_b: Target space.
        actor_id: Actor ID in space_a.
        avatar_id: Avatar ID in space_b.
        title: Overall figure title.
        root_id: Root element ID used in both spaces (default 1).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    space(space_a, actor_id, title="Space A (Source)", ax=ax1, root_id=root_id)
    space(space_b, avatar_id, title="Space B (Target)", ax=ax2, root_id=root_id)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def trajectory(space_obj, positions, actor_id=None,
               title="Trajectory", root_id=1):
    """Render a space with a trajectory path overlaid.

    Args:
        space_obj: Space instance.
        positions: List of [x, y] positions forming the path.
        actor_id: Optional actor ID to draw at current position.
        title: Plot title.
        root_id: Root element ID for tree traversal (default 1).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    space(space_obj, actor_id, title=title, ax=ax, root_id=root_id)

    pts = np.array(positions)
    ax.plot(pts[:, 0], pts[:, 1], "r--", linewidth=1, alpha=0.5)
    ax.plot(pts[:, 0], pts[:, 1], "ro", markersize=4, alpha=0.5)
    ax.plot(pts[0, 0], pts[0, 1], "gs", markersize=10, label="Start")
    ax.plot(pts[-1, 0], pts[-1, 1], "r*", markersize=15, label="End")
    ax.legend()
    return fig
