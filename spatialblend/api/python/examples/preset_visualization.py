#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""
Visualize spatialblend preset data using Simulator + InteractionScript.

For each of the 3 presets (173_410, 733_173, 733_245), this script:
  1. Creates a Simulator with two spaces (source + target)
  2. Sets up room geometry and furniture from preset JSON
  3. Creates actor/avatar with root, chest, hand anchors
  4. Binds root -> Spacing, chest -> Facing
  5. Builds a trajectory from an InteractionScript
  6. Runs cross-space simulation
  7. Renders animation with interaction labels
  8. Exports to MP4 (falls back to PNG if ffmpeg unavailable)
  9. Prints quality metrics

Usage:
    bazel run //spatialblend/api/python/examples:preset_visualization
    bazel test //spatialblend/api/python:test_preset_visualization --test_output=all
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from spatialblend.api.python import (
    Simulator, SimulatorRunConfig, KernelType,
    Polygon2D, NavigableSpatialDefinition,
)
from spatialblend.api.python.interaction import InteractionScript
from spatialblend.api.python.interaction.script import _CommandType
from spatialblend.api.python.visualization import render_simulation, export_mp4


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESET_CONFIGS = {
    "173_410": {
        "label": "Affine",
        "mp4_name": "173_410_affine.mp4",
        "build_script": lambda: (
            InteractionScript(seed=42)
            .wander(3.0)
            .look_at("tv", 5.0)
            .approach_to("sofa", 3.0)
            .reach_to("table", "hand", 2.0)
        ),
    },
    "733_173": {
        "label": "Nonlinear",
        "mp4_name": "733_173_nonlinear.mp4",
        "build_script": lambda: (
            InteractionScript(seed=42)
            .look_at("chair", 4.0)
            .reach_to("sofa", "hand", 3.0)
            .wander(4.0)
            .look_at("table", 5.0)
        ),
    },
    "733_245": {
        "label": "Topo-different",
        "mp4_name": "733_245_topo_different.mp4",
        "build_script": lambda: (
            InteractionScript(seed=42)
            .approach_to("chair", 3.0)
            .reach_to("sofa", "hand", 2.0)
            .look_at("dresser", 5.0)
            .wander(3.0)
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_preset(preset_file):
    """Load a preset JSON file."""
    with open(preset_file) as f:
        return json.load(f)


def polygon_from_ring(ring_points):
    """Convert a list of [x,y] points to a Polygon2D."""
    points = [np.array(p, dtype=float) for p in ring_points]
    return Polygon2D(points)


def compute_interaction_labels(script):
    """Compute (start_time, end_time, label) tuples from script commands.

    Uses approximate transit times based on walk speed for labeling purposes.
    """
    _COMMAND_LABELS = {
        _CommandType.LOOK_AT: "LookAt",
        _CommandType.APPROACH_TO: "ApproachTo",
        _CommandType.REACH_TO: "ReachTo",
        _CommandType.MOVE_TO: "MoveTo",
        _CommandType.WANDER: "Wander",
    }
    labels = []
    t = 0.0
    for cmd in script._commands:
        label = _COMMAND_LABELS.get(cmd.type, str(cmd.type))
        tag = cmd.element_tag or "area"
        transit = 1.0  # approximate transit between commands
        t += transit
        labels.append((t, t + cmd.duration, f"{label} {tag}"))
        t += cmd.duration
    return labels


def setup_space(sim, data):
    """Set up a space from preset data using Simulator's fluent API.

    Returns (space_builder, room_handle, actor_handle).
    """
    sb = sim.add_space()

    # Room boundary
    walkable_points = [np.array(p, dtype=float)
                       for p in data["walkable_rings_m"][0]]
    room = sb.add_element("Room") \
        .with_polygon(walkable_points) \
        .with_shared_id(100)

    # Furniture objects as children of room
    entities = data.get("entity_positions_m", [])
    anchors = data.get("anchors", [])

    for i, (pos, anchor_data) in enumerate(zip(entities, anchors)):
        name = anchor_data.get("name", f"object_{i}")
        child = room.add_child(name) \
            .at(pos[0], pos[1]) \
            .with_shared_id(200 + i)
        if "polygon_m" in anchor_data:
            boundary_pts = [np.array(p, dtype=float)
                            for p in anchor_data["polygon_m"]]
            child.with_boundary(boundary_pts)

    # Create actor with root, chest, hand anchors
    actor = sb.add_actor() \
        .with_anchor("root", 0.0, 0.0, 0.0) \
        .with_anchor("chest", 0.0, 0.0, 1.2) \
        .with_anchor("hand", 0.4, 0.0, 0.9) \
        .in_scope(room)

    # Bind root -> Spacing, chest -> Facing
    actor.bind("root", KernelType.SPACING)
    actor.bind("chest", KernelType.FACING)

    return sb, room, actor


def _is_test_env():
    """Detect if running inside a bazel test (TEST_TMPDIR is set)."""
    return "TEST_TMPDIR" in os.environ


def run_preset_example(preset_id, output_dir, preset_cfg):
    """Run a full preset example: simulate, render, export, print metrics."""
    os.makedirs(output_dir, exist_ok=True)
    test_mode = _is_test_env()

    # Load preset JSON
    data_dir = os.path.join(os.path.dirname(__file__),
                            "../../../../spatialblend/sim/data/presets")
    preset = load_preset(os.path.join(data_dir, f"{preset_id}.json"))

    label = preset_cfg["label"]
    print(f"\n=== Preset {preset_id} ({label}) ===")
    print(f"  Source scene: {preset['source_scene']}")
    print(f"  Target scene: {preset['target_scene']}")
    print(f"  Mapping class: {preset['mapping_class']}")

    # Create Simulator with two spaces
    sim = Simulator.create()
    sa, room_a, actor = setup_space(sim, preset["source"])
    sb, room_b, avatar = setup_space(sim, preset["target"])

    # Place actor at center of source walkable area
    src_pts = preset["source"]["walkable_rings_m"][0]
    cx = np.mean([p[0] for p in src_pts])
    cy = np.mean([p[1] for p in src_pts])
    actor.move_to(np.array([cx, cy]), 0.0)

    # Place avatar at center of target walkable area
    tgt_pts = preset["target"]["walkable_rings_m"][0]
    tcx = np.mean([p[0] for p in tgt_pts])
    tcy = np.mean([p[1] for p in tgt_pts])
    avatar.move_to(np.array([tcx, tcy]), 0.0)

    # Flush starting positions so the avatar doesn't start at origin
    sim.tick(0.033)

    # Build trajectory from InteractionScript
    script = preset_cfg["build_script"]()
    traj = script.build(sa, actor)

    # Compute interaction labels for the timeline
    interaction_labels = compute_interaction_labels(script)

    # Run simulation — use coarser dt in test mode for speed
    dt = 0.2 if test_mode else 0.033
    config = SimulatorRunConfig()
    config.max_linear_speed = 1.4
    config.max_angular_speed = 2.0

    result = sim.run(actor, avatar, traj, dt, config)

    # Print quality metrics
    print(f"  Frames: {result.num_frames}, dt: {result.dt:.3f}")
    print(f"  Total avatar travel: {result.total_avatar_travel():.1f} m")
    print(f"  Max speed: {result.max_speed():.1f} m/s")
    print(f"  Max acceleration: {result.max_acceleration():.1f} m/s^2")
    print(f"  Mean jerk: {result.mean_jerk():.1f} m/s^3")
    print(f"  Mean relation error: {result.mean_relation_error():.2f}")

    # In test mode, skip heavy animation rendering — just save a static plot
    if test_mode:
        print("  [test mode] Skipping animation render")
        plt.close("all")
        return result

    # Render animation
    space_a = sa.get_space()
    space_b = sb.get_space()
    actor_id = actor.id()
    avatar_id = avatar.id()

    anim = render_simulation(
        result, space_a, space_b, actor_id, avatar_id,
        interaction_labels=interaction_labels,
        title=f"Preset {preset_id} ({label})",
    )

    # Export to MP4 (fall back to PNG if ffmpeg unavailable)
    mp4_path = os.path.join(output_dir, preset_cfg["mp4_name"])
    try:
        export_mp4(anim, mp4_path, fps=30, dpi=150)
        print(f"  Exported: {mp4_path}")
    except Exception as e:
        print(f"  MP4 export failed ({e}), saving PNG frames instead")
        png_path = os.path.join(
            output_dir,
            preset_cfg["mp4_name"].replace(".mp4", "_frame0.png"))
        fig = anim._fig
        anim._func(0)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        last_path = os.path.join(
            output_dir,
            preset_cfg["mp4_name"].replace(".mp4", "_frame_last.png"))
        anim._func(result.num_frames - 1)
        fig.savefig(last_path, dpi=150, bbox_inches="tight")
        print(f"  Saved fallback PNGs: {png_path}, {last_path}")

    plt.close("all")
    return result


def run_all_presets(output_dir):
    """Run visualization for all 3 presets."""
    for preset_id, cfg in PRESET_CONFIGS.items():
        try:
            run_preset_example(preset_id, output_dir, cfg)
        except Exception as e:
            print(f"Error on {preset_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/vis"
    run_all_presets(output_dir)
