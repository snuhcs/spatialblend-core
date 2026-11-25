# Assets and display constants of the scenario viewer

Everything the 3D view draws that is not case data is listed here with its source. No Asset Store, Mixamo or other
restricted content is used. The viewer ships no texture, model or font file: textures are generated at run time.

## Library

| Item | Version | License | Source | Integrity |
|---|---|---|---|---|
| three.js (ES module build, loaded lazily on first 3D use) | 0.160.0 exact | MIT | https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js | SRI `sha384-8/hf5tabFfIo0BPku31TKUvGIKpWGYaMwNtlSNqNyZzXgZCMd4lnOGWbJElML9dv` (enforced with `fetch(..., {integrity})` in `js/render/three-loader.js`); sha256 `3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6`, 670681 bytes (checked 2026-10-19) |

The module build prints no console message (the UMD build `three.min.js` prints a deprecation warning). The same bytes are
served by jsdelivr (`three@0.160.0/build/three.module.js` is the unminified file, sha256 `76dea815...`, not used).

## Textures and materials

All procedural canvas textures (`js/render/textures.js`, seeded, deterministic): parquet floor, plaster wall, wood, fabric weave,
whiteboard, blackboard, brushed metal, carpet. One tile = 1 m of surface unless noted. No external image, so no source URL,
license or sha256 applies. If an external texture or model is ever added it must be CC0, MIT or CC-BY and be listed here with its
source URL, license and sha256.

## Humanoid segment lengths (display constants)

Source: D. A. Winter, *Biomechanics and Motor Control of Human Movement*, Figure 4.1 "Body segment lengths expressed as a fraction of
body height H" (the figure credits Drillis and Contini, 1966). Page 83 in the copy checked (a course handout page,
https://courses.grainger.illinois.edu/me481/sp2021/Anthro-Winter.pdf, fetched 2026-10-19; the page image was read by eye; the handout
does not print the edition, the page layout is that of the 4th edition, 2009: EDITION NOT VERIFIED). Only the numbers are used; no
figure or text is copied into the repository.

Read off the figure (fractions of stature H):

| Quantity | Fraction of H | In the figure |
|---|---|---|
| eye height | 0.936 | printed |
| chin height | 0.870 | printed |
| shoulder height | 0.818 | printed |
| elbow height | 0.630 | printed (not used) |
| wrist height | 0.485 | printed (not used) |
| knuckle height | 0.377 | printed (not used) |
| hip (greater trochanter) height | 0.530 | printed |
| knee height | 0.285 | printed |
| ankle height | 0.039 | printed |
| head | 0.130 | printed |
| upper arm | 0.186 | printed |
| forearm | 0.146 | printed |
| hand | 0.108 | printed |
| shoulder width | 0.259 (half = 0.129 printed) | printed |
| hip width | 0.191 | printed |
| foot length | 0.152 | printed |
| foot breadth | 0.055 | printed |
| thigh | 0.245 | DERIVED: hip height - knee height (0.530 - 0.285) |
| shank | 0.246 | DERIVED: knee height - ankle height (0.285 - 0.039) |

The code is `js/figure/anthro.js` (`WINTER_FIG_4_1`); the numbers sit next to this citation there. The two derived values are
differences of printed heights, not printed values.

Display choices that are NOT in the figure, and carry no anthropometric claim (`DISPLAY_SHAPE` in the same file):
the trunk between hip and shoulder height is divided evenly for spine1..spine3; the collar joints sit half way between the neck and the
shoulder joint; the drawn thickness of limbs, trunk and head (radii) and the foot box are shape only; the foot is drawn from the ankle
forward by the foot length.

**Stature has no default.** The IK figure is drawn at, in this order: (1) the value typed in the stage bar ("stature m", a display parameter);
(2) a stature stated by the case (`actor.body.stature_m` or `actor.stature_m`, none of the current exporters write one);
(3) derived from the case: the standing pelvis height stated by the track (the `source.anchor.pelvis` / `receiver.anchor.pelvis.proposal` target on
valid ticks where `.engaged` is not set, highest value, minus the root z) divided by 0.530 (Winter Fig. 4.1 hip height). The label says
"derived: standing pelvis ... / 0.530"; an actor with no pelvis anchor of its own may take the other actor's derived value, labelled as such. If
none of these exist there is NO figure: a capsule marker is drawn (the exported `actor.body` capsule, else a display-only marker of r 0.2 m x
H 0.5 m that is not a body size) with the label "stature unknown - enter a value". A joints-driven figure is drawn from its joints; only its
thickness uses a stature, derived from the mean hip-knee-ankle length / 0.491 (Winter Fig. 4.1: thigh + shank). A MotionGen sidecar's stature
appears only in its free-text `origin`, which the viewer does not parse. The HUD and the inspector show the value and where it came from.

## Sit-bone offset (registered by the embodiment; interim display fallback file)

Anchors and their offsets are registered by the app / embodiment driver, never constants in the viewer. The viewer reads the pelvis
anchor offset from the case's `embodiment` registration (`anchors[{id: "pelvis", offset}]`, offset as `[forward, left, up]` or `{forward, left, up}`). When a case that has anchors carries none, `embodiments/smplx.json` is used, flagged INTERIM in the file and in the UI
label ("sit-bone offset measured on the SMPL-X neutral body, interim display fallback"). With neither, the seated pelvis joint sits at the anchor and the posture
says so. The fallback file holds the measurement below. The pelvis anchor is the sit-bone (ischial tuberosity) contact and the display pelvis joint is placed from it with
the offset measured on the SMPL-X neutral body: (forward, left, up) = (+0.0634, 0, -0.1860) m. Source: SMPL-X neutral body, measured offline
2026-10-19 by the evaluation harness (not public); the licensed model files are not distributed. Procedure: SMPL-X neutral betas = 0, model seated with
hips and knees flexed 90 degrees, candidate vertices = buttock and posterior thigh, rigid seat plane through each side's lowest vertex, 10 mm contact patch,
midline mean. Convention of the measurement: the offset is in the pelvis frame and
yaw-rotated, anchor = J0 + R(yaw) x (forward, left) with `up` added to z, so the viewer uses J0 = anchor - R(yaw) x (forward, left) - up (code: `figure.js`, offset passed in). Applied only to the IK
figure while the pelvis anchor is engaged (seated); the pelvis joint therefore steps by the offset when `.engaged` flips, and nothing is smoothed. Joints figures
(MotionGen, recorded) are drawn from their own joints. The neutral shape is used; the offset scales with body size (the measurement gives up = -0.169 / -0.203 m
for betas[0] = -2 / +2). The standing pelvis height used to derive a display stature (above) is the track's anchor value and is not corrected by this offset.

## Other display values (authored, labelled "display-derived" in the UI)

`js/render/furniture.js` `DISPLAY`: default heights of furniture types used only when a case exports neither `shape3d` nor a part
plane (chair 0.45, armchair 0.42, stool 0.65, bench 0.45, sofa 0.42, office chair 0.50, table 0.75, desk 0.74, counter 0.90, shelf 1.8,
board bottom 0.9 / height 1.0 m: values authored for the fixtures, not data), back height 0.45 m, slab 0.04 m,
leg 0.04 m, room wall height 2.6 m. Leg and back styles of the procedural furniture are display choices.
The viewer states in the HUD: "furniture: display-derived".

## Display values (presentation only)

These shape how the 3D view looks or how the display IK searches. None is case data, none is a claim about people. Where a value has a source it
is cited, otherwise it is authored for display.

| Value | Where | Source |
|---|---|---|
| camera field of view 42 degrees; near = distance/200, far = 40 x distance + 200 | `perspective3d.js` | authored for display |
| orbit elevation limits 0.06..1.55 rad, zoom 0.5..2000 m, default azimuth -0.95, elevation 0.78, distance 1.25 x the 2D focus extent | `perspective3d.js` | authored for display |
| follow camera distance max(2.5 m, 0.3 x orbit distance), target height 0.9 m (orbit target height 0.5 m) | `perspective3d.js` | authored for display |
| hemisphere light 1.6 (sky 0xe6eefb, ground 0x8a7a66); sun 2.6 (0xfff1de) at (0.35, -0.6, 1.1) x the space size | `perspective3d.js` | authored for display (three.js 0.160 physical light units) |
| ACES filmic tone mapping, exposure 1.0 | `perspective3d.js` | three.js `ACESFilmicToneMapping`; exposure authored |
| shadow map 2048 x 2048 PCF-soft, frustum +-0.85 x space size, near 0.5, far 4 x space size, bias -0.0004, normalBias 0.03, radius 2 | `perspective3d.js` | authored for display (tuned against shadow acne in the headless screenshots) |
| room wall height 2.6 m, curb height 0.35 m (walls between the camera and the view target are lowered) | `furniture.js` DISPLAY | authored for display |
| default furniture heights, back height 0.45 m, slab 0.04 m, leg 0.04 m, leg and back styles | `furniture.js` DISPLAY | authored for display (listed above) |
| limb / trunk / head thickness, foot box, spine split, collar position | `anthro.js` DISPLAY_SHAPE, `figure3d.js` | authored for display (see above) |
| IK torso-lean search: step 2.5 degrees, at most 90 degrees (a horizontal torso), stop when the total hand miss is under 2 mm | `figure.js` | authored for display |
| a hand miss is reported as a residue only above 1 mm; a foot reach shortfall only above 0.1 mm | `figure.js` | authored for display (numeric noise floor) |
| two-bone IK pole vectors: knee bends toward the body forward; reaching elbow toward down and 0.3 x backward; hanging arm toward forward | `figure.js` | authored for display (bend convention) |
| two-bone IK reach clamp: |a - b| + 1e-6 .. a + b - 1e-6 | `figure.js` | numerical guard |
| display walk swing: leg and arm swing from the root path with stride 0.7 m and swing 0.25 m (the 2D glyph's constants, `glyph.js` G) | `figure.js`, `scene.js` | authored for display (2D glyph constants) |
| the seated feet are placed one thigh length ahead of the hips, on the floor | `figure.js` | authored for display |
| marker capsule r 0.2 m, H 0.5 m (when the case exports no body capsule): also the size of the proposal ghost then | `pose.js`, `perspective3d.js` | authored for display; not a body size |
| anchor marker radius 0.035 m, engaged ring 0.075 m; part overlay opacity 0.32; scope / hover halo opacities 0.18 .. 0.5 | `perspective3d.js` | authored for display |
| a stated stature (`stature_m`) is used only if it is a finite number in (0, 3] m; otherwise it is 'not usable' and the derived / unknown path applies | `scene.js` STATURE_GUARD_M | display-safety guard, not an anthropometric claim |
| viewer safety limits for sidecars: 64 MiB, 512 joints, 20 million samples, 2 million ticks | `loader.js` LIMITS | authored (resource guard) |
