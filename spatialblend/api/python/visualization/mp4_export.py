# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""MP4 export for matplotlib animations."""
import matplotlib.animation


def export_mp4(anim, path, fps=30, dpi=150):
    """Export a FuncAnimation to MP4 file.

    Args:
        anim: matplotlib FuncAnimation instance.
        path: Output file path (should end in .mp4).
        fps: Frames per second.
        dpi: Resolution.
    """
    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
