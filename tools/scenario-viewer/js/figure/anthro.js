// Copyright (c) 2024-2026 Seoul National University
// SPDX-License-Identifier: MIT

// Display anthropometry. Segment lengths as fractions of stature H.
//
// Source: D. A. Winter, "Biomechanics and Motor Control of Human Movement", Figure 4.1 "Body segment lengths expressed
// as a fraction of body height H" (page 83 in the copy checked; the figure itself credits Drillis and Contini, 1966).
// The numbers below were READ OFF that figure (page image fetched 2026-10-19, see ASSETS.md for the URL). Two
// segments are not printed in the figure; they are differences of printed heights and are marked DERIVED.
// Nothing here is data of a case: it is the display constants of the figure the viewer draws, and the UI says so.
export const WINTER_FIG_4_1 = {
  citation: 'Winter, Biomechanics and Motor Control of Human Movement, Fig. 4.1 (after Drillis and Contini 1966)',
  heightFrac: {            // heights above the floor, fractions of H (printed in the figure)
    eye: 0.936, chin: 0.870, shoulder: 0.818, elbow: 0.630, wrist: 0.485, knuckle: 0.377,
    hip: 0.530, knee: 0.285, ankle: 0.039,
  },
  lengthFrac: {            // printed in the figure
    head: 0.130, upperArm: 0.186, forearm: 0.146, hand: 0.108,
    shoulderWidth: 0.259, hipWidth: 0.191, footLength: 0.152, footBreadth: 0.055,
  },
};
const F = WINTER_FIG_4_1;
export const SEG = {
  hip: F.heightFrac.hip, knee: F.heightFrac.knee, ankle: F.heightFrac.ankle, shoulder: F.heightFrac.shoulder,
  chin: F.heightFrac.chin, eye: F.heightFrac.eye,
  thigh: F.heightFrac.hip - F.heightFrac.knee,        // DERIVED 0.245: hip height - knee height
  shank: F.heightFrac.knee - F.heightFrac.ankle,      // DERIVED 0.246: knee height - ankle height
  upperArm: F.lengthFrac.upperArm, forearm: F.lengthFrac.forearm, hand: F.lengthFrac.hand,
  head: F.lengthFrac.head, shoulderHalf: F.lengthFrac.shoulderWidth / 2, hipHalf: F.lengthFrac.hipWidth / 2,
  footLength: F.lengthFrac.footLength, footBreadth: F.lengthFrac.footBreadth,
};

// DISPLAY choices that are NOT in the figure (shape only, no anthropometric claim):
//  * the trunk between hip height and shoulder height is divided evenly for spine1..spine3,
//  * collar joints sit half way between the neck and the shoulder joint,
//  * limb and trunk radii (drawn thickness) are a fraction of H chosen to look like a person.
export const DISPLAY_SHAPE = { collarFrac: 0.5, limbRadius: 0.028, thighRadius: 0.045, trunkHalfWidth: 0.115, trunkHalfDepth: 0.075, headRx: 0.075 };

// There is NO default stature. Stature is stated by the case, derived from its standing pelvis height (hip height 0.530 H, above),
// derived from joints (thigh + shank = 0.491 H), set by the user, or unknown (the viewer then draws a capsule marker).

// The sit-bone offset is NOT a constant of the viewer: it is read from the case's `embodiment` registration, or, when the case has none,
// from embodiments/smplx.json (labelled interim; its source and procedure are cited there and in ASSETS.md).
