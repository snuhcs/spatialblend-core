#!/usr/bin/env python3
# Copyright (c) 2024-2026 Seoul National University
# SPDX-License-Identifier: MIT

"""SYNTHETIC schema-1.1 fixtures for the 3D viewer (NOT producer output, never real data). Writes dev-data/fixtures-affordance/.

  SYN-SIT-01        chair -> armchair. pose3d pelvis anchors (source.anchor.pelvis, receiver.anchor.pelvis.proposal) + engaged flags.
                    No joints: the viewer draws the display-IK figure and says so.
  SYN-WRITE-01      wall whiteboard -> standing easel. hand_r anchors, vertical `board` parts, and a motiongen-style joints SIDECAR
                    (producer labelled; left hand raised on purpose so joints-vs-IK is visible).
  SYN-LOOKAHEAD-01  a walk with a MotionGen-style sidecar that also carries a `.lookahead` joints channel (buffer_revision, horizon).
  SYN-FURNITURE     one room with every furniture type (and parts, boards on two walls) for the procedural builders.

Geometry/dimensions are authored for the fixtures. The sidecar joints are made with the viewer's own
solver (fixture_joints.mjs), so they are fixtures, not a MotionGen result. Needs node.
"""
import json, math, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures-affordance")
SHA = "synthetic"
SCHEMA = "1.1.0"
SMPL22 = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
PAR = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
N, DT = 120, 0.05


def r3(v): return round(v, 3)
def r9(v): return round(v, 9)   # rotation entries: enough digits to stay orthonormal for validate.py
def qz(yaw): return [0.0, 0.0, round(math.sin(yaw / 2), 6), round(math.cos(yaw / 2), 6)]
def rect(cx, cy, w, d, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[r3(cx + x * c - y * s), r3(cy + x * s + y * c)] for x, y in [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]]
def ngon(cx, cy, r, n=16): return [[r3(cx + r * math.cos(2 * math.pi * i / n)), r3(cy + r * math.sin(2 * math.pi * i / n))] for i in range(n)]
def lrect(w, d): return [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
def wt(cx, cy, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [r9(c), r9(-s), 0, r3(cx), r9(s), r9(c), 0, r3(cy), 0, 0, 1, z, 0, 0, 0, 1]
IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def hpart(name, tag, cx, cy, yaw, z, w, d, origin="scene-spec"):
    return {"name": name, "tag": tag, "world_transform": wt(cx, cy, yaw, z), "footprint_local": lrect(w, d), "origin": origin}


def vpart(name, tag, x0, y0, z0, theta, w, h, origin="scene-spec"):
    """vertical board part: local x along (cos t, sin t), local y = world up, local z = x cross y = (sin t, -cos t, 0); origin = lower-left corner"""
    c, s = math.cos(theta), math.sin(theta)
    m = [r9(c), 0, r9(s), r3(x0), r9(s), 0, r9(-c), r3(y0), 0, 1, 0, r3(z0), 0, 0, 0, 1]
    return {"name": name, "tag": tag, "world_transform": m, "footprint_local": [[0, 0], [w, 0], [w, h], [0, h]], "origin": origin}


def vpart_world(p, u, v):
    m = p["world_transform"]
    return [m[0] * u + m[1] * v + m[3], m[4] * u + m[5] * v + m[7], m[8] * u + m[9] * v + m[11]]


def elem(i, name, kind, ground, h=None, wtm=None, local=None, parts=None, parent=1, shared=None, flags=None):
    e = {"id": i, "shared_id": shared if shared is not None else i, "name": name, "kind": kind, "parent": parent, "flags": flags or [],
         "world_transform": wtm or IDENT, "footprint_local": local or ground, "ground": ground, "shape3d": {"extrude_height_m": h} if h else None}
    if parts: e["parts"] = parts
    return e


def space(role, W, D, elems):
    room = elem(1, "room", "room", [[0, 0], [W, 0], [W, D], [0, D]], parent=None)
    return {"id": role, "role": role, "navigable_root": 1, "free_space": {"outer": [[0, 0], [W, 0], [W, D], [0, D]], "holes": [], "origin": "synthetic"}, "elements": [room] + elems}


def pose_ch(cid, space_id, actor, pts, yaws, valid=None, z=None):
    n = len(pts)
    ch = {"id": cid, "type": "pose3d", "space": space_id, "time_base": "tick",
          "data": {"n": n, "p": [c for i, p in enumerate(pts) for c in (r3(p[0]), r3(p[1]), r3(p[2] if len(p) > 2 else 0.0))], "q": [c for y in yaws for c in qz(y)], "valid": valid if valid else [1] * n}}
    if actor: ch["actor"] = actor
    return ch


def anchor_ch(cid, space_id, pts, valid, origin):
    ch = pose_ch(cid, space_id, None, [p if p else [0, 0, 0] for p in pts], [0.0] * len(pts), valid)
    ch["origin"] = origin; ch["doc"] = "SYNTHETIC anchor target (authored by the fixture generator)"
    return ch


def flag_ch(cid, space_id, vals):
    """`<anchor>.engaged` is an enum (FREE / ENGAGED), as in SCHEMA.md 1.1"""
    return {"id": cid, "type": "enum", "space": space_id, "time_base": "tick", "data": {"n": len(vals), "codes": [1 if v >= 0.5 else 0 for v in vals], "labels": ["FREE", "ENGAGED"]}}


def smooth(a, b, f): return a + (b - a) * (f * f * (3 - 2 * f))
def lerp(a, b, f): return a + (b - a) * f
def clamp(x, a, b): return max(a, min(b, x))
def seg(k, k0, k1): return clamp((k - k0) / max(1, k1 - k0), 0, 1)


def base_case(cid, title, rule, spaces, actors, channels, n=N, dt=DT, extra=None):
    case = {"format": "spatialblend.scenario-trace", "schema_version": SCHEMA, "synthetic": True,
            "provenance": {"producer": {"name": "make_fixtures_affordance.py", "version": "1"}, "repo": "spatialblend-core", "git_sha": SHA, "command": "make_fixtures_affordance.py", "generated_at": "1970-01-01T00:00:00Z", "schema_version": SCHEMA},
            "case": {"id": cid, "suite": "affordance-fixtures", "title": title, "spec": {"rule": rule, "level": "unit"}},
            "time": {"base": "tick", "n_ticks": n, "dt_s": dt, "t0_s": 0}, "spaces": spaces,
            "correspondence": [{"source_space": "source", "receiver_space": "receiver", "source": 2, "receiver": 2, "shared_id": 2}] if len(spaces) > 1 else [],
            "actors": actors, "channels": channels, "expectations": [], "results": [], "metrics": {"row": {"frames": n}, "guards": []},
            "missing": ["SYNTHETIC viewer fixture: no solver, no producer output"]}
    if extra: case.update(extra)
    return case


def actor(aid, space_id, role, root_ch, H=1.7):
    return {"id": aid, "space": space_id, "element": None, "role": role, "root_channel": root_ch, "body": {"capsule_radius_m": 0.2, "capsule_height_m": H}, "skeleton": None, "joints_channel": None}


def walk_state(pts, yaws):
    ph, amp, cum, last = [], [], 0.0, None
    for k, p in enumerate(pts):
        step = math.hypot(p[0] - last[0], p[1] - last[1]) if last else 0.0
        cum += step; last = p
        ph.append(cum / 0.7); amp.append(0.25 * clamp(step / DT / 0.25, 0, 1))
    return ph, amp


def node_joints(frames):
    r = subprocess.run(["node", os.path.join(HERE, "fixture_joints.mjs")], input=json.dumps(frames), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def joints_channel(cid, space_id, actor_id, frames, extra):
    J = node_joints(frames)
    n = len(J)
    ch = {"id": cid, "type": "joints", "space": space_id, "actor": actor_id, "time_base": "tick", "doc": "SYNTHETIC joints (fixture)",
          "data": {"n": n, "joint_count": 22, "p": [c for fr in J for j in fr for c in j], "q": [c for _ in range(n) for _ in range(22) for c in (0, 0, 0, 1)]}}
    ch.update(extra)
    return ch


SKEL = {"id": "smpl22", "joints": [{"name": n, "parent": p} for n, p in zip(SMPL22, PAR)], "units": "m", "frame": "world z-up"}
PROD = {"name": "SYNTHETIC motiongen-style fixture", "version": "0.0-fixture", "git_sha": "synthetic", "command": "make_fixtures_affordance.py", "generated_at": "1970-01-01T00:00:00Z"}


def sidecar(cid, actors, channels):
    return {"format": "spatialblend.scenario-trace.sidecar", "schema_version": SCHEMA, "case_id": cid, "producer": PROD, "actors": [{"id": a, "skeleton": SKEL} for a in actors], "channels": channels}


# ---------------------------------------------------------------------------------------------------------- SYN-SIT-01
def sit_case():
    HIP, SEAT_S, SEAT_R = 0.9, 0.45, 0.42
    # source: chair at (5,3) facing -x (yaw pi); receiver: armchair at (4,2.5) facing +y (yaw pi/2)
    cs, ca = (5.0, 3.0), (4.0, 2.5)
    def track(seat_xy, seat_yaw, seat_z, approach_from):
        """root path: walk to the front of the seat, turn back to it, sit (tick 70-80), hold, stand (100-110), leave"""
        fx, fy = math.cos(seat_yaw), math.sin(seat_yaw)          # seat forward
        front = (seat_xy[0] + fx * 0.8, seat_xy[1] + fy * 0.8)
        pts, yaws, pel, eng, valid = [], [], [], [], []
        face = seat_yaw + math.pi                                 # approaching: face the seat; sitting: face along seat forward
        for k in range(N):
            if k < 45:
                f = seg(k, 0, 45); p = (lerp(approach_from[0], front[0], f), lerp(approach_from[1], front[1], f)); y = math.atan2(front[1] - approach_from[1], front[0] - approach_from[0])
            elif k < 62:
                f = seg(k, 45, 62); p = front; y0 = math.atan2(front[1] - approach_from[1], front[0] - approach_from[0]); d = math.atan2(math.sin(seat_yaw - y0), math.cos(seat_yaw - y0)); y = y0 + d * smooth(0, 1, f)
            elif k < 80:
                f = seg(k, 62, 80); p = (lerp(front[0], seat_xy[0], f), lerp(front[1], seat_xy[1], f)); y = seat_yaw
            elif k < 100:
                p = seat_xy; y = seat_yaw
            elif k < 110:
                f = seg(k, 100, 110); p = (lerp(seat_xy[0], front[0], f), lerp(seat_xy[1], front[1], f)); y = seat_yaw
            else:
                f = seg(k, 110, N - 1); p = (lerp(front[0], approach_from[0], f), lerp(front[1], approach_from[1], f)); y = seat_yaw
            pts.append((p[0], p[1], 0.0)); yaws.append(y)
            if 55 <= k <= 108:
                if k < 70: z = HIP
                elif k < 80: z = lerp(HIP, seat_z, smooth(0, 1, seg(k, 70, 80)))
                elif k < 100: z = seat_z
                else: z = lerp(seat_z, HIP, smooth(0, 1, seg(k, 100, 108)))
                pel.append([p[0] if k >= 80 and k < 100 else lerp(p[0], seat_xy[0], seg(k, 62, 80)), p[1] if k >= 80 and k < 100 else lerp(p[1], seat_xy[1], seg(k, 62, 80)), z]); valid.append(1)
                eng.append(1.0 if 80 <= k < 100 else 0.0)
            else:
                pel.append(None); valid.append(0); eng.append(0.0)
        return pts, yaws, pel, valid, eng
    sp, sy, spel, sv, se = track(cs, math.pi, SEAT_S, (1.5, 3.0))
    # the pelvis anchor is the seat contact: x,y = seat centre while seated
    rp, ry, rpel, rv, re = track(ca, math.pi / 2, SEAT_R, (1.5, 1.0))
    chair = elem(2, "chair", "obstacle", rect(*cs, 0.45, 0.45, math.pi), h=SEAT_S, wtm=wt(*cs, math.pi), local=lrect(0.45, 0.45),
                 parts=[hpart("seat", "Sittable", cs[0], cs[1], math.pi, SEAT_S, 0.4, 0.4)])
    arm = elem(2, "armchair", "obstacle", rect(*ca, 0.8, 0.75, math.pi / 2), wtm=wt(*ca, math.pi / 2), local=lrect(0.8, 0.75),
               parts=[hpart("seat", "Sittable", ca[0], ca[1], math.pi / 2, SEAT_R, 0.55, 0.5)])
    table_s = elem(3, "table", "obstacle", rect(7.0, 4.6, 1.2, 0.7), h=0.75)
    table_r = elem(3, "table", "obstacle", rect(6.2, 4.4, 1.2, 0.7), h=0.75)
    sp_src = space("source", 8, 6, [chair, table_s]); sp_rx = space("receiver", 7, 5.5, [arm, table_r])
    chs = [pose_ch("source.root", "source", "src", sp, sy), pose_ch("receiver.body", "receiver", "rx", rp, ry),
           anchor_ch("source.anchor.pelvis", "source", spel, sv, "synthetic:fixture"), flag_ch("source.anchor.pelvis.engaged", "source", se),
           anchor_ch("receiver.anchor.pelvis.proposal", "receiver", [[ca[0], ca[1], SEAT_R] if v else None for v in rv], rv, "synthetic:fixture"), flag_ch("receiver.anchor.pelvis.engaged", "receiver", re)]
    # the receiver proposal target is the seat contact at the armchair seat plane only where engaged
    return base_case("SYN-SIT-01", "Sit: chair to armchair (SYNTHETIC)", "Pelvis anchor on a Sittable part: the receiver anchor proposal lands on the armchair seat plane (z 0.42). Display IK figure, no joints.",
                     [sp_src, sp_rx], [actor("src", "source", "source_actor", "source.root"), actor("rx", "receiver", "receiver_body", "receiver.body")], chs)


# ---------------------------------------------------------------------------------------------------------- SYN-WRITE-01
def write_case():
    wall_y = 5.95
    wb = vpart("board", "Drawable", 3.0, wall_y + 0.0, 0.9, 0.0, 2.0, 1.0)           # x along +x, normal (sin0, -cos0) = -y: into the room
    wall_board = elem(2, "whiteboard", "obstacle", [[3.0, 5.95], [5.0, 5.95], [5.0, 6.0], [3.0, 6.0]], wtm=IDENT, local=[[3.0, 5.95], [5.0, 5.95], [5.0, 6.0], [3.0, 6.0]], parts=[wb])
    th = math.radians(30)
    ex, ey = 3.4, 2.6
    eb = vpart("board", "Drawable", ex - math.cos(th) * 0.35, ey - math.sin(th) * 0.35, 0.8, th, 0.7, 1.0)
    east = elem(2, "standing easel", "obstacle", rect(ex, ey, 0.7, 0.12, th), wtm=IDENT, local=rect(ex, ey, 0.7, 0.12, th), parts=[eb])
    sp_src = space("source", 8, 6, [wall_board]); sp_rx = space("receiver", 7, 5.5, [east])
    n = N

    def stroke(k):
        """(u, v) of the source hand on the board (metres), or None. 45-70 horizontal sweep, 70-95 vertical stroke"""
        if k < 45 or k > 95: return None
        if k <= 70: return (lerp(0.4, 1.6, seg(k, 45, 70)), 0.5)
        return (1.6, lerp(0.5, 0.15, seg(k, 70, 95)))
    # ---- source actor
    spts, syaws, ph, hv = [], [], [], []
    for k in range(n):
        st = stroke(k)
        u = st[0] if st else (0.4 if k < 45 else 1.6)
        wx, wy, _ = vpart_world(wb, u, 0.0)
        root = (wx, wy - 0.5)
        if k < 40: f = seg(k, 0, 40); root = (lerp(1.5, root[0], f), lerp(2.0, root[1], f))
        if k > 100: f = seg(k, 100, n - 1); root = (lerp(root[0], 6.5, f), lerp(root[1], 2.0, f))
        spts.append((root[0], root[1], 0.0)); syaws.append(math.pi / 2 if 30 <= k <= 100 else math.atan2(2 - root[1], 6.5 - root[0]) if k > 100 else math.atan2(root[1] - 2.0, root[0] - 1.5))
        if k >= 36 and k <= 99:
            if st: p = vpart_world(wb, st[0], st[1]); h = 1
            else: p = vpart_world(wb, 0.4 if k < 45 else 1.6, 0.5); p = [p[0], p[1] - 0.12 * (1 - seg(k, 36, 45)) if k < 45 else p[1], p[2]]; h = 1
            ph.append(p); hv.append(1)
        else: ph.append(None); hv.append(0)
    eng_s = [1.0 if stroke(k) else 0.0 for k in range(n)]
    # ---- receiver: normalised board coordinates (u/2.0, v/1.0) mapped onto the easel board (0.7 x 1.0)
    rpts, ryaws, rh, rv = [], [], [], []
    nx, ny = math.sin(th), -math.cos(th)                       # easel board normal
    for k in range(n):
        st = stroke(k)
        u = (st[0] / 2.0 * 0.7) if st else (0.4 / 2.0 * 0.7 if k < 45 else 1.6 / 2.0 * 0.7)
        wx, wy, _ = vpart_world(eb, u, 0.0)
        root = (wx + nx * 0.5, wy + ny * 0.5)
        if k < 40: f = seg(k, 0, 40); root = (lerp(1.2, root[0], f), lerp(4.5, root[1], f))
        if k > 100: f = seg(k, 100, n - 1); root = (lerp(root[0], 5.5, f), lerp(root[1], 4.8, f))
        rpts.append((root[0], root[1], 0.0)); ryaws.append(math.atan2(-ny, -nx) if 30 <= k <= 100 else math.atan2(ny, nx))
        if 36 <= k <= 99:
            if st: p = vpart_world(eb, st[0] / 2.0 * 0.7, st[1])
            else: p = vpart_world(eb, u, 0.5)
            rh.append(p); rv.append(1)
        else: rh.append(None); rv.append(0)
    eng_r = eng_s
    chs = [pose_ch("source.root", "source", "src", spts, syaws), pose_ch("receiver.body", "receiver", "rx", rpts, ryaws),
           anchor_ch("source.anchor.hand_r", "source", ph, hv, "synthetic:fixture"), flag_ch("source.anchor.hand_r.engaged", "source", eng_s),
           anchor_ch("receiver.anchor.hand_r.proposal", "receiver", rh, rv, "synthetic:fixture"), flag_ch("receiver.anchor.hand_r.engaged", "receiver", eng_r)]
    cid = "SYN-WRITE-01"
    case = base_case(cid, "Write: wall whiteboard to standing easel (SYNTHETIC)", "Hand anchor on a vertical Drawable part: the receiver hand proposal follows the mapped stroke on the easel board plane. MotionGen-style joints come from a sidecar.",
                     [sp_src, sp_rx], [actor("src", "source", "source_actor", "source.root"), actor("rx", "receiver", "receiver_body", "receiver.body")], chs)
    # sidecar: joints from the viewer's own solver with the LEFT hand raised (so a joints figure differs visibly from display IK)
    def frames(pts, yaws, hands):
        out = []
        for k in range(n):
            left = [pts[k][0] - math.sin(yaws[k]) * 0.0 + math.cos(yaws[k]) * 0.35 + (-math.sin(yaws[k])) * 0.0, pts[k][1], 1.75]
            left = [pts[k][0] + math.cos(yaws[k]) * 0.3 - math.sin(yaws[k]) * 0.25, pts[k][1] + math.sin(yaws[k]) * 0.3 + math.cos(yaws[k]) * 0.25, 1.55]
            out.append({"root": pts[k], "yaw": yaws[k], "H": 1.7, "pelvis": None, "hands": {"right": hands[k], "left": left}, "walk": None})
        return out
    sc = sidecar(cid, ["src", "rx"], [
        joints_channel("source.joints", "source", "src", frames(spts, syaws, ph), {"producer": PROD, "origin": "synthetic:viewer-fixture", "derived": "motiongen"}),
        joints_channel("receiver.joints", "receiver", "rx", frames(rpts, ryaws, rh), {"producer": PROD, "origin": "synthetic:viewer-fixture", "derived": "motiongen"})])
    return case, sc


# ---------------------------------------------------------------------------------------------------------- SYN-LOOKAHEAD-01
def look_case():
    cid = "SYN-LOOKAHEAD-01"
    pts, yaws = [], []
    for k in range(N):
        f = k / (N - 1); a = f * 2 * math.pi * 0.8
        p = (4 + 2.6 * math.cos(a - 1.2), 3 + 1.8 * math.sin(a - 1.2), 0.0)
        pts.append(p)
    for k in range(N):
        a, b = pts[max(0, k - 1)], pts[min(N - 1, k + 1)]; yaws.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    ph, amp = walk_state(pts, yaws)
    sp = space("source", 8, 6, [elem(2, "table", "obstacle", rect(4, 3, 1.2, 0.7), h=0.75)])
    case = base_case(cid, "Look-ahead buffer (SYNTHETIC)", "A MotionGen-style sidecar carries the played joints and a look-ahead joints channel (buffer_revision, horizon): the not-yet-played buffer is drawn as a translucent ghost.",
                     [sp], [actor("src", "source", "source_actor", "source.root")], [pose_ch("source.root", "source", "src", pts, yaws)])
    fr = [{"root": pts[k], "yaw": yaws[k], "H": 1.7, "pelvis": None, "hands": {}, "walk": {"phase": ph[k], "amp": amp[k]}} for k in range(N)]
    HZ = 20
    sc = sidecar(cid, ["src"], [
        joints_channel("src.joints", "source", "src", fr, {"producer": PROD, "origin": "synthetic:viewer-fixture", "derived": "motiongen"}),
        joints_channel("src.joints.lookahead", "source", "src", fr, {"producer": PROD, "origin": "synthetic:viewer-fixture", "derived": "motiongen"})])
    look = sc["channels"][1]
    look["doc"] = "SYNTHETIC look-ahead buffer: at tick k the viewer draws this channel at tick k + horizon"
    look["data"]["horizon"] = HZ
    look["data"]["buffer_revision"] = [k // 10 for k in range(N)]
    return case, sc


# ---------------------------------------------------------------------------------------------------------- SYN-FURNITURE
def furniture_case():
    W, D = 14.0, 10.0
    E, eid = [], [10]
    def add(*a, **k):
        eid[0] += 1; e = elem(eid[0], *a, **k); E.append(e); return e
    def seat(name, cx, cy, yaw, w, d, h, sw=None, sd=None, round_=False):
        poly = ngon(cx, cy, w / 2) if round_ else rect(cx, cy, w, d, yaw)
        add(name, "obstacle", poly, h=h, wtm=wt(cx, cy, yaw), local=lrect(w, d), parts=[hpart("seat", "Sittable", cx, cy, yaw, h, sw or w * 0.9, sd or d * 0.9)])
    seat("chair", 1.5, 1.5, 0.6, 0.45, 0.45, 0.45)
    seat("armchair", 3.5, 1.6, -0.5, 0.8, 0.75, 0.42, 0.55, 0.5)
    seat("stool", 5.5, 1.5, 0.0, 0.32, 0.32, 0.65, round_=True)
    seat("bench", 8.0, 1.5, 0.3, 0.4, 1.5, 0.45, 0.4, 1.5)
    seat("sofa", 11.2, 1.9, 0.2, 0.8, 2.0, 0.42, 0.7, 1.9)
    seat("office chair", 1.5, 4.2, 0.9, 0.48, 0.48, 0.5, round_=True)
    add("table", "obstacle", rect(4, 4.2, 1.4, 0.8), h=0.75)
    add("desk", "obstacle", rect(7, 4.2, 1.2, 0.6, 0.5), h=0.74, parts=[hpart("desk_surface", "Touchable", 7, 4.2, 0.5, 0.74, 1.2, 0.6)])
    add("round table", "obstacle", ngon(10.2, 4.3, 0.55, 20), h=0.74)
    add("counter", "obstacle", rect(3.5, 7.0, 2.4, 0.6), h=0.9)
    add("shelf", "obstacle", rect(7.0, 9.75, 1.0, 0.3), h=1.5)
    add("whiteboard", "obstacle", [[10.0, 9.95], [12.0, 9.95], [12.0, 10.0], [10.0, 10.0]], parts=[vpart("board", "Drawable", 10.0, 9.95, 0.9, 0.0, 2.0, 1.0)])
    add("blackboard", "obstacle", [[13.95, 5.5], [14.0, 5.5], [14.0, 8.7], [13.95, 8.7]], parts=[vpart("board", "Drawable", 13.95, 8.7, 0.9, -math.pi / 2, 3.2, 1.0)])
    th = math.radians(20)
    add("easel", "obstacle", rect(1.5, 7.2, 0.7, 0.12, th), parts=[vpart("board", "Drawable", 1.5 - math.cos(th) * 0.35, 7.2 - math.sin(th) * 0.35, 0.8, th, 0.7, 1.0)])
    add("flip chart", "obstacle", rect(5.5, 7.8, 0.7, 0.1, -0.3), parts=[vpart("board", "Drawable", 5.5 + math.cos(-0.3) * -0.35, 7.8 + math.sin(-0.3) * -0.35, 0.7, -0.3, 0.7, 1.0)])
    add("wall panel", "obstacle", [[13.95, 3.9], [14.0, 3.9], [14.0, 4.02], [13.95, 4.02]], parts=[vpart("button", "Touchable", 13.95, 4.02, 1.2, -math.pi / 2, 0.12, 0.12)])
    add("crate", "obstacle", rect(9.5, 7.2, 0.8, 0.6, 0.4), h=0.5)
    add("rug", "surface", rect(9.5, 6.8, 2.0, 1.4, 0.0), flags=[])
    sp = space("source", W, D, E)
    pts = [(1.0, 9.0, 0.0), (1.2, 9.0, 0.0)]
    case = base_case("SYN-FURNITURE", "Furniture gallery (SYNTHETIC)", "One element per procedural builder (chair, armchair, stool, bench, sofa, office chair, table, desk, counter, shelf, whiteboard, blackboard, easel, flip chart, panel, fallback).",
                     [sp], [actor("src", "source", "source_actor", "source.root")], [pose_ch("source.root", "source", "src", pts, [0.0, 0.0])], n=2)
    return case


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), separators=(",", ":"))
    return os.path.getsize(path)


def main():
    cases = []
    sit = sit_case(); wr, wsc = write_case(); lk, lsc = look_case(); fu = furniture_case()
    entries = []
    for case, side in [(sit, None), (wr, wsc), (lk, lsc), (fu, None)]:
        cid = case["case"]["id"]
        size = write(os.path.join(OUT, "cases", cid + ".json"), case)
        e = {"id": cid, "suite": "affordance-fixtures", "title": case["case"]["title"], "rule": case["case"]["spec"]["rule"], "file": f"cases/{cid}.json", "bytes": size, "git_sha": SHA, "has_trace": True, "pass": None, "families": ["AffordancePose"], "modes": [], "statuses": []}
        if side:
            write(os.path.join(OUT, "sidecars", cid + ".motiongen.json"), side)
            e["sidecars"] = [{"file": f"sidecars/{cid}.motiongen.json", "producer": "motiongen"}]
        entries.append(e)
    index = {"format": "spatialblend.scenario-trace.index", "schema_version": SCHEMA, "provenance": {"producer": {"name": "make_fixtures_affordance.py", "version": "1"}, "repo": "spatialblend-core", "git_sha": SHA, "command": "make_fixtures_affordance.py", "generated_at": "1970-01-01T00:00:00Z", "schema_version": SCHEMA},
             "suites": [{"id": "affordance-fixtures", "title": "Affordance fixtures (SYNTHETIC)", "description": "Schema 1.1 fixtures for the 3D viewer (parts, anchors, sidecar joints, look-ahead). Never real data.", "kind": "cases"}],
             "cases": entries, "tables": [], "excluded": [], "missing": []}
    write(os.path.join(OUT, "index.json"), index)
    print("wrote", OUT, [e["id"] for e in entries])


if __name__ == "__main__":
    main()
