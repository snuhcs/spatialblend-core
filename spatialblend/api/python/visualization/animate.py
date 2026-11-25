# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Animation for spatialblend SimulationResult visualization."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

from . import plot

# Color map for interaction timeline segments.
_INTERACTION_COLORS = {
    "LookAt": "#9b59b6",      # purple
    "ApproachTo": "#1abc9c",   # teal
    "ReachTo": "#e67e22",      # orange
    "Wander": "#95a5a6",       # gray
    "MoveTo": "#3498db",       # blue
}


def _compute_bounds(positions_list, padding=0.5):
    """Compute fixed (xmin, xmax, ymin, ymax) from a list of position arrays."""
    all_x, all_y = [], []
    for positions in positions_list:
        for p in positions:
            all_x.append(p[0])
            all_y.append(p[1])
    if not all_x:
        return -1, 1, -1, 1
    return (min(all_x) - padding, max(all_x) + padding,
            min(all_y) - padding, max(all_y) + padding)


def _get_room_bounds(space, root_id=1, padding=1.0):
    """Get bounds from the room polygon's spatial definition."""
    tree = space.element_tree
    root = tree.get_element(root_id)
    if root is None:
        return None
    sd = root.get_spatial_definition()
    if sd is None:
        return None
    poly = sd.get_boundary()
    if poly is None:
        return None
    pts = poly.get_points()
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs) - padding, max(xs) + padding,
            min(ys) - padding, max(ys) + padding)


def render_simulation(
    result,
    space_a,
    space_b,
    actor_id,
    avatar_id,
    interaction_labels=None,
    fps=30,
    title="Cross-Space Simulation",
):
    """Render a SimulationResult as a side-by-side animation.

    Args:
        result: SimulationResult from Simulator.Run().
        space_a: Source Space containing the actor.
        space_b: Target Space containing the avatar.
        actor_id: Actor entity_id in space_a.
        avatar_id: Avatar entity_id in space_b.
        interaction_labels: Optional list of (start_time, end_time, label_text)
            tuples for the timeline bar.
        fps: Frames per second for the animation.
        title: Overall figure title.

    Returns:
        A matplotlib FuncAnimation instance.
    """
    has_timeline = interaction_labels is not None and len(interaction_labels) > 0

    # Determine number of rows: spaces + optional relation graph + optional timeline
    n_rows = 1  # space panels
    row_ratios = [6]
    n_rows += 1  # relation proximity graph
    row_ratios.append(2)
    if has_timeline:
        n_rows += 1
        row_ratios.append(1)

    fig = plt.figure(figsize=(16, sum(row_ratios) * 1.1))
    gs = fig.add_gridspec(n_rows, 2, height_ratios=row_ratios, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_rel = fig.add_subplot(gs[1, :])
    ax_timeline = fig.add_subplot(gs[2, :]) if has_timeline else None

    fig.suptitle(title, fontsize=14)

    total_time = result.num_frames * result.dt

    # Pre-compute fixed axis bounds from room geometry
    bounds_a = _get_room_bounds(space_a)
    bounds_b = _get_room_bounds(space_b)
    if bounds_a is None:
        bounds_a = _compute_bounds([result.actor_positions])
    if bounds_b is None:
        bounds_b = _compute_bounds([result.avatar_positions])

    # Pre-compute relation proximity data for the graph
    # We need to collect proximity per element tag over all frames
    proximity_data = {}  # tag -> list of (time, proximity, kernel_name)
    for frame_i in range(result.num_frames):
        t = frame_i * result.dt
        pos = result.actor_positions[frame_i]
        yaw = result.actor_yaws[frame_i]
        space_a.update_actor_state(actor_id, np.array(pos), yaw)
        bindings = space_a.get_bindings(actor_id)
        for binding in bindings:
            kt_name = binding.kernel_type.name if hasattr(
                binding.kernel_type, 'name') else str(binding.kernel_type)
            for match in binding.latest_matches:
                elem = space_a.element_tree.get_element(match.element_id)
                tag = elem.tag if elem and elem.tag else f"id:{match.element_id}"
                key = f"{tag} ({kt_name})"
                if key not in proximity_data:
                    proximity_data[key] = {"times": [], "values": [],
                                           "kernel": kt_name}
                proximity_data[key]["times"].append(t)
                proximity_data[key]["values"].append(
                    match.eval.grounded.proximity)

    # Pre-assign colors per relation key
    rel_colors = {}
    for key, data in proximity_data.items():
        rel_colors[key] = plot.KERNEL_COLORS.get(
            data["kernel"], "#95a5a6")

    def _draw_timeline(current_time):
        """Draw the interaction timeline bar with playhead."""
        ax_timeline.clear()
        ax_timeline.set_xlim(0, total_time)
        ax_timeline.set_ylim(0, 1)
        ax_timeline.set_yticks([])
        ax_timeline.set_xlabel("Time (s)")

        for (start_t, end_t, label_text) in interaction_labels:
            first_word = label_text.split()[0] if label_text else ""
            color = _INTERACTION_COLORS.get(first_word, "#bdc3c7")
            width = end_t - start_t
            rect = mpatches.FancyBboxPatch(
                (start_t, 0.1), width, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.7)
            ax_timeline.add_patch(rect)
            mid_t = (start_t + end_t) / 2
            ax_timeline.text(mid_t, 0.5, label_text, ha="center", va="center",
                             fontsize=7, clip_on=True)

        ax_timeline.axvline(current_time, color="red", linewidth=2)

    def _draw_trail(ax, positions, frame_idx, color):
        """Draw trajectory trail with fading alpha."""
        if frame_idx < 1:
            return
        pts = np.array([[p[0], p[1]] for p in positions[:frame_idx + 1]])
        n = len(pts)
        for i in range(1, n):
            alpha = 0.1 + 0.6 * (i / n)
            ax.plot(pts[i - 1:i + 1, 0], pts[i - 1:i + 1, 1],
                    color=color, alpha=alpha, linewidth=1)

    def _draw_relation_graph(frame):
        """Draw proximity time-series up to current frame."""
        ax_rel.clear()
        current_time = frame * result.dt

        for key, data in proximity_data.items():
            times = data["times"]
            values = data["values"]
            # Only plot up to current frame
            mask = [i for i, t in enumerate(times) if t <= current_time]
            if mask:
                ts = [times[i] for i in mask]
                vs = [values[i] for i in mask]
                ax_rel.plot(ts, vs, color=rel_colors[key],
                            linewidth=1.2, label=key, alpha=0.8)

        ax_rel.set_xlim(0, total_time)
        ax_rel.set_ylim(0, 1.05)
        ax_rel.set_ylabel("Proximity")
        ax_rel.set_xlabel("Time (s)")
        ax_rel.axvline(current_time, color="red", linewidth=1, alpha=0.5)
        ax_rel.legend(loc="upper right", fontsize=6, ncol=2)
        ax_rel.grid(True, alpha=0.2)

    def init():
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Current positions and yaws
        actor_pos = result.actor_positions[frame]
        actor_yaw = result.actor_yaws[frame]
        avatar_pos = result.avatar_positions[frame]
        avatar_yaw = result.avatar_yaws[frame]

        # Update space states so elements render correctly
        space_a.update_actor_state(actor_id, np.array(actor_pos), actor_yaw)
        space_b.update_actor_state(avatar_id, np.array(avatar_pos), avatar_yaw)

        # Draw spaces (no relation lines — those go in the graph below)
        plot.space(space_a, actor_id,
                   title=f"Space A (Source)", ax=ax1)
        plot.space(space_b, avatar_id,
                   title=f"Space B (Target)", ax=ax2)

        # Trajectory trails
        _draw_trail(ax1, result.actor_positions, frame, "red")
        _draw_trail(ax2, result.avatar_positions, frame, "blue")

        # Realization targets in Space B (subtle markers)
        if frame < len(result.targets):
            for target in result.targets[frame]:
                tp = target.position
                kt_name = target.kernel_type.name if hasattr(
                    target.kernel_type, 'name') else str(target.kernel_type)
                color = plot.KERNEL_COLORS.get(kt_name, "gray")
                ax2.plot(tp[0], tp[1], "x", color=color,
                         markersize=6, markeredgewidth=1.5, alpha=0.5)

        # Fixed axes — tight to room geometry, no origin
        ax1.set_xlim(bounds_a[0], bounds_a[1])
        ax1.set_ylim(bounds_a[2], bounds_a[3])
        ax2.set_xlim(bounds_b[0], bounds_b[1])
        ax2.set_ylim(bounds_b[2], bounds_b[3])
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")

        # Relation proximity graph
        _draw_relation_graph(frame)

        # Timeline
        if ax_timeline is not None:
            current_time = frame * result.dt
            _draw_timeline(current_time)

        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=result.num_frames,
        interval=1000 // fps, blit=False)

    return anim
