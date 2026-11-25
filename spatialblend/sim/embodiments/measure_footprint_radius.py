#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Measure the standing floor-footprint radius of a body model (SMPL / SMPL-X), numpy only.

The radius that erodes free space comes from the embodiment's registration, MEASURED from the
body model; the core has no default.

Definition. The footprint radius is the largest horizontal distance, from the vertical axis through
the pelvis joint (joint 0, the actor root), of any vertex of the body that stands on or moves through
the floor level: the legs, the feet and the lower trunk. Procedure (deterministic):
  1. Shaped rest model: v = v_template + shapedirs . betas (betas = 0 is the neutral shape), the joints
     J = J_regressor . v. The rest pose is the model's canonical pose (global_orient = identity, every
     joint at rest), whose frame is +y up and the legs hang straight down (the arms are out in a
     T-pose and are NOT part of the floor footprint).
  2. Footprint vertices: those whose dominant LBS joint (argmax of the skinning weights) is the pelvis
     (0), a hip (1, 2), spine1 (3), a knee (4, 5), an ankle (7, 8) or a foot (10, 11); SMPL-X's toes and
     SMPL's feet are inside the foot joints' weights.
  3. Radius = max over those vertices of sqrt((x - x_J0)^2 + (z - z_J0)^2).
Reports the radius (m) for the given shapes and the sha256 of the asset. Rest-pose model only; no
dataset data is read.  Usage:
  measure_footprint_radius.py --npz SMPLX_NEUTRAL.npz [--betas0 -2 --betas0 2]   (a SMPL_NEUTRAL.pkl also loads;
  a pickled sparse J_regressor needs scipy)
"""
import argparse
import hashlib
import json

import numpy as np

FOOTPRINT_JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11)  # pelvis, hips, spine1, knees, ankles, feet


def _stub_chumpy():
    """The licensed SMPL .pkl files pickle chumpy arrays: unpickle them as plain ndarrays, without chumpy."""
    import sys
    import types
    if "chumpy" in sys.modules:
        return

    class Ch:
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.__dict__["x"], dtype=dtype)

    mod, sub = types.ModuleType("chumpy"), types.ModuleType("chumpy.ch")
    sub.Ch = mod.Ch = Ch
    mod.ch = sub
    sys.modules["chumpy"], sys.modules["chumpy.ch"] = mod, sub


def load(path):
    if path.endswith(".pkl"):
        import pickle
        _stub_chumpy()
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
    else:
        d = np.load(path, allow_pickle=True)
    jr = d["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else np.asarray(jr)
    return (np.asarray(d["v_template"], dtype=np.float64), np.asarray(d["shapedirs"], dtype=np.float64),
            np.asarray(jr, dtype=np.float64), np.asarray(d["weights"], dtype=np.float64))


def radius(npz_path, betas=None):
    v_t, sd, jr, w = load(npz_path)
    b = np.zeros(sd.shape[-1])
    if betas is not None:
        b[: len(betas)] = betas
    v = v_t + np.einsum("vcs,s->vc", sd, b)
    j0 = jr[0] @ v
    dominant = np.argmax(w, axis=1)
    sel = np.isin(dominant, FOOTPRINT_JOINTS)
    horiz = np.hypot(v[sel, 0] - j0[0], v[sel, 2] - j0[2])
    k = int(np.argmax(horiz))
    return float(horiz[k]), int(sel.sum()), v[sel][k] - j0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--betas0", type=float, action="append", default=[],
                    help="also report with betas[0] set to this value (sensitivity)")
    a = ap.parse_args()
    sha = hashlib.sha256(open(a.npz, "rb").read()).hexdigest()
    r, n, off = radius(a.npz)
    out = {"asset_sha256": sha, "shape": "neutral, betas = 0", "footprint_vertices": n,
           "floor_footprint_radius_m": round(r, 5), "farthest_vertex_from_pelvis_axis": [round(float(x), 5) for x in off],
           "sensitivity_betas0": {}}
    for b0 in a.betas0:
        out["sensitivity_betas0"][str(b0)] = round(radius(a.npz, [b0])[0], 5)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
