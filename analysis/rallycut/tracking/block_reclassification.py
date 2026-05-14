"""A3 — BLOCK type reclassification heuristic.

USED BY PROBE ONLY, NO PRODUCTION HOOK.

A3 was NO-SHIP'd 2026-05-14 by a clean-GT fleet measurement (1/49 = 2.0%
precision, far below the 92% spec gate). The action_classifier.py wiring
was removed in the same commit that updated this notice. This module is
retained as evidence + reusable infrastructure for any future revisit
with cleaner GT and/or a more selective gate. The probe + fleet-rate
measurement script (``scripts/probe_a3_block_reclassification.py`` and
``scripts/measure_a3_block_reclass_rate.py``) continue to import
``should_reclassify_to_block`` and the per-condition checks below; no
other call sites exist.

Spec: ``docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md``,
section A3.

This module implements a pure helper that, given a rally's actions list +
team assignments + court calibration (and a wrist-y supplied by the caller
since pose inference lives upstream), evaluates whether an ``ATTACK``
action should be reclassified as ``BLOCK`` under all four heuristic
conditions:

  (a) at-net          — player BBOX-TOP (head) is within a small image-y
                        band around the net line (v2). v1 used feet
                        projected to court_y near 8m, which falsely
                        rejected mid-jump blockers whose feet are NOT
                        on the ground.
  (b) wrist-above-net — wrist keypoint Y in image coords above the net line.
                        **Production gate (ship-1, 2026-05-14): firm only**
                        — `True` accepts, `Unknown` (no wrist detected)
                        rejects. The v2 probe scored 0/3 on moderate
                        (Unknown) candidates vs 7/7 on strong (firm),
                        so the production gate requires firm pose.
  (c) dir-change ≤ 90 — ball direction-change at contact (deflected, not reversed)
  (d) opposing-team   — prev action is opposing team. Spec strict variant:
                        prev ∈ {attack, set}. **Production uses LOOSE**:
                        prev cross-team AND prev NOT a serve (captures F5
                        canonical case + the strict superset).

The helper is intentionally side-effect-free: it does **not** mutate the
input actions and does **not** run pose inference.

v1 → v2 refinements (2026-05-14, post-probe-v1):
- (a)′ image-coords HEAD-near-BALL combined with BALL-in-net-region
  replaces v1's feet-court-projection check (3 of v1's "rejected"
  cases were actual blocks; mid-jump feet project to court_y far from
  net). The "net line" derived from court calibration is the GROUND
  projection of the midline (where the net touches the ground) — the
  visible net top sits well above that. So we anchor (a)′ on the ball
  (which is by definition at the net's top at a block contact) and
  verify the player's head sits at roughly the ball's image-y.
- (d) loose: prev cross-team AND prev NOT a serve.

ship-1 refinements (2026-05-14):
- Production gate ``should_reclassify_to_block`` requires (b) FIRM TRUE
  by default. Pose-Unknown is REJECTED (the v2 probe scored 0/3 on
  moderate-tier candidates vs 7/7 on strong-tier).
- ``selected_strict`` / ``selected_loose`` updated to match the firm
  gate (legacy probe-facing fields).

Two variants of (d) are exposed:

- ``check_d_strict``  — prev cross-team AND prev ∈ {attack, set}
- ``check_d_loose``   — prev cross-team only (any prev action), excluding
                        prev=serve (a serve is never an attacking play in
                        possession terms; receiving a serve is not the
                        block-reclassification scenario).

The strict variant matches the spec wording verbatim. The loose variant
captures the F5 canonical pattern in keke rally ``99091ec6`` where the
prev action is ``receive(B)`` (not set/attack) but the suspect attack(A)
at frame 184 is the structural canonical block-mis-typed-as-attack case
from the design doc. The probe layer surfaces both — see
``probe_a3_block_reclassification.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Geometry constants -------------------------------------------------------

# Beach volleyball net is at court_y = 8 (half of COURT_LENGTH=16). See
# rallycut.court.calibration.
NET_COURT_Y = 8.0

# Near-net band (meters from net plane). Beach blockers position roughly
# 1-2 m off the net at takeoff; they can land 2-3m back. Use 3.0m to cover
# both takeoff and landing positions. Volleyball's "attack line" is at 3m
# from the net — anything inside is the front-row blocker/attacker zone.
NET_BAND_METERS = 3.0

# Normalized image-coord fallback band when court calibration is missing.
# Net usually runs through ~y_image=0.45-0.55 (camera centered on court);
# 0.15 spans ±15% which roughly covers the player-at-net region.
NET_BAND_IMAGE_NORMALIZED_FALLBACK = 0.15

# v2 (a)′ HEAD-near-net band. Default ±5% of normalized image height: a
# jumping blocker's head sits within a couple of head-heights of the net
# line; standing at the net puts the head ~0.05-0.10 above/below depending
# on player height and camera angle. 0.05 catches blockers (head at net or
# slightly above) while rejecting receivers/setters mid-court.
HEAD_NET_BAND_IMAGE_NORM = 0.05

# Ball direction-change cap (degrees). Spec: ≤ 90° means deflected but not
# fully reversed.
MAX_DIRECTION_CHANGE_DEG = 90.0

# Strict (d) — prev action must be one of these to satisfy the spec.
PREV_ATTACK_OR_SET = frozenset({"attack", "set"})

# Loose (d) exclusion: prev=serve is never a valid block-reclassification
# trigger (the responder to a serve is a receiver, not a blocker).
LOOSE_D_EXCLUDED_PREV = frozenset({"serve"})


# --- Inputs -------------------------------------------------------------------


@dataclass
class CandidateInputs:
    """Hand-off shape from the probe runner to the helper.

    All fields except ``wrist_y_image`` are read from the rally's
    actions_json / contacts_json. ``wrist_y_image`` is computed by the
    probe layer (pose inference) and passed in here.

    ``net_y_image`` is the net Y in normalized image coordinates,
    derived from the rally's court calibration corners (avg of the two
    mid-edge points). Pass ``None`` if calibration is unavailable; the
    helper will then SKIP condition (b) and treat it as unknown.
    """

    action_index: int                       # index in rally.actions
    action: dict[str, Any]                  # the ATTACK action dict
    prev_action: dict[str, Any] | None      # action[i-1] in the rally
    team_assignments: dict[str, str]        # {tid_str: "A"|"B"}
    player_court_xy: tuple[float, float] | None  # projected (x_m, y_m), None if no calibration
    net_y_image: float | None               # normalized image-y of net line, or None
    wrist_y_image: float | None             # normalized image-y of player's higher wrist, or None
    direction_change_deg: float             # from Contact at this frame
    ball_y_image: float                     # normalized image-y of ball at contact
    # v2 additions:
    player_bbox_top_y_image: float | None = None   # normalized image-y of player's bbox top (head)
    player_bbox_bottom_y_image: float | None = None  # normalized image-y of player's bbox bottom (feet) — for diagnostics


# --- Per-condition checks -----------------------------------------------------


def check_a_at_net(
    player_court_xy: tuple[float, float] | None,
    ball_y_image: float,
    *,
    net_band_meters: float = NET_BAND_METERS,
    net_y_image_norm: float = 0.5,
    net_band_image_norm: float = NET_BAND_IMAGE_NORMALIZED_FALLBACK,
) -> tuple[bool, str]:
    """(v1, kept for reference) Return (passes, source) where source is
    'court' or 'image-fallback'.

    Prefers court projection when available; falls back to ball-y proximity
    to the net line in normalized image coords.

    NOTE: v1 over-rejects mid-jump blockers — their feet project to a
    court_y far from 8m because they're airborne. Use ``check_a_at_net_v2``
    instead (head-near-net in image coords).
    """
    if player_court_xy is not None:
        _, court_y = player_court_xy
        return (abs(court_y - NET_COURT_Y) <= net_band_meters), "court"
    # Fallback: use ball-y vs nominal net-y in image coords.
    return (abs(ball_y_image - net_y_image_norm) <= net_band_image_norm), "image-fallback"


def check_a_at_net_v2(
    player_bbox_top_y_norm: float | None,
    net_y_image_norm: float | None,
    *,
    band_norm: float = HEAD_NET_BAND_IMAGE_NORM,
    ball_y_image: float | None = None,
    ball_y_fallback_band: float = NET_BAND_IMAGE_NORMALIZED_FALLBACK,
) -> tuple[bool, str]:
    """v2 (a)′: HEAD-near-net in image coords.

    Calibration note: ``net_y_image_norm`` is the IMAGE-Y of the
    GROUND projection of the court midline (where the net touches the
    ground). The visible NET TOP is ABOVE that line by ~2.5m, which
    in normalized image-y typically means somewhere between
    ``net_y_image - 0.25`` and ``net_y_image - 0.05`` depending on
    camera tilt / distance.

    A blocker's head at contact sits at roughly the BALL's image-y
    (which is by definition near the net's top at a block). So we
    treat (a)′ as:

      ``ball_y_image`` is between ``net_y_image_norm - 0.30`` and
      ``net_y_image_norm + 0.05`` (the ball is at-or-above the ground
      net line — i.e., in the net region or above)
    AND
      ``|head_y - ball_y| <= band_norm`` widened to 0.15
      (the player's head is at roughly the ball's height — i.e.,
      they're "up at the net" — including mid-jump blockers).

    Falls back gracefully when inputs missing.

    Returns (passes, source) where source is one of
    'head-near-ball-at-net', 'ball-y-fallback', 'ball-y-nominal',
    'no-data'.
    """
    # Wider band for head-vs-ball alignment (3x the original tight 0.05).
    HEAD_BALL_BAND = 0.15  # noqa: N806 — local geometry constant
    # How far below the ground-net line the ball can be (the ball can sag
    # ~5% below the ground projection in some camera angles).
    BALL_BELOW_NET_TOLERANCE = 0.05  # noqa: N806
    # Maximum distance ABOVE ground-net line for ball-y (ball can be up to
    # ~30% above ground-net in normalized image-y for tilted camera views;
    # this is the "ball is in the net region or above").
    BALL_ABOVE_NET_TOLERANCE = 0.35  # noqa: N806

    if (
        player_bbox_top_y_norm is not None
        and ball_y_image is not None
        and net_y_image_norm is not None
    ):
        # Ball should be in the net region (at-or-above ground-net line).
        ball_dy = ball_y_image - net_y_image_norm
        ball_in_net_region = (
            -BALL_ABOVE_NET_TOLERANCE <= ball_dy <= BALL_BELOW_NET_TOLERANCE
        )
        # Player's head should be at roughly ball's height.
        head_close_to_ball = abs(player_bbox_top_y_norm - ball_y_image) <= HEAD_BALL_BAND
        return (
            ball_in_net_region and head_close_to_ball,
            "head-near-ball-at-net",
        )

    # No head detected but we still have ball + net — fall back to ball-y
    # vs ground-net-y within a wide band.
    if ball_y_image is not None and net_y_image_norm is not None:
        ball_dy = ball_y_image - net_y_image_norm
        return (
            -BALL_ABOVE_NET_TOLERANCE <= ball_dy <= BALL_BELOW_NET_TOLERANCE,
            "ball-y-fallback",
        )

    # No calibration — use a nominal centered net assumption.
    if ball_y_image is not None:
        return (
            abs(ball_y_image - 0.5) <= ball_y_fallback_band,
            "ball-y-nominal",
        )
    return False, "no-data"


def check_b_wrist_above_net(
    wrist_y_image: float | None,
    net_y_image: float | None,
) -> tuple[bool | None, str]:
    """Return (passes, reason).

    Image-y is INVERTED — smaller y is HIGHER in the frame. So
    "wrist above net" ⇔ wrist_y_image < net_y_image.

    Returns (None, 'unknown') when either input is missing — the caller
    should treat unknown as a SOFT condition: we can't reject the
    candidate, but we can't confirm either. The probe groups by
    (a) ∧ (c) ∧ (d) first and surfaces (b) on the visual page.
    """
    if wrist_y_image is None:
        return None, "no-wrist-detected"
    if net_y_image is None:
        return None, "no-court-calibration"
    return (wrist_y_image < net_y_image), "ok"


def check_c_direction_change(
    direction_change_deg: float,
    *,
    max_deg: float = MAX_DIRECTION_CHANGE_DEG,
) -> bool:
    """≤ 90° means the ball deflected but did not fully reverse."""
    return direction_change_deg <= max_deg


def check_d_strict(
    action: dict[str, Any],
    prev_action: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Strict (d) per spec: prev is opposing-team AND prev ∈ {attack, set}.

    Returns (passes, reason).
    """
    if prev_action is None:
        return False, "no-prev-action"
    curr_team = action.get("team")
    prev_team = prev_action.get("team")
    if curr_team in (None, "unknown") or prev_team in (None, "unknown"):
        return False, "team-unknown"
    if curr_team == prev_team:
        return False, "same-team"
    prev_type = str(prev_action.get("action", ""))
    if prev_type not in PREV_ATTACK_OR_SET:
        return False, f"prev-type-{prev_type}"
    return True, "ok"


def check_d_loose(
    action: dict[str, Any],
    prev_action: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Loose (d): prev is opposing-team and prev NOT a serve.

    Per refinement (d) v2: ``prev.team != curr.team AND prev.action NOT
    IN {serve}``. Serves are excluded because the response to a serve is
    a receive, not a block — block-reclassification doesn't apply.

    This is the variant that captures the F5 canonical case in keke
    rally ``99091ec6`` (prev=receive cross-team). Used by the probe as a
    superset to surface F5 alongside strict-(d) candidates.
    """
    if prev_action is None:
        return False, "no-prev-action"
    curr_team = action.get("team")
    prev_team = prev_action.get("team")
    if curr_team in (None, "unknown") or prev_team in (None, "unknown"):
        return False, "team-unknown"
    if curr_team == prev_team:
        return False, "same-team"
    prev_type = str(prev_action.get("action", ""))
    if prev_type in LOOSE_D_EXCLUDED_PREV:
        return False, f"prev-type-{prev_type}"
    return True, "ok"


# --- Composite ---------------------------------------------------------------


@dataclass
class CandidateVerdict:
    """Per-condition outcome for one ATTACK action (v2)."""

    a_passes: bool
    a_source: str                  # "head-image" | "ball-y-fallback" | "ball-y-nominal" | "no-data" | (v1) "court" / "image-fallback"
    b_passes: bool | None          # None when wrist or net unknown
    b_reason: str
    c_passes: bool
    d_strict_passes: bool
    d_strict_reason: str
    d_loose_passes: bool
    d_loose_reason: str
    all_pass_strict: bool          # (a ∧ (b is True) ∧ c ∧ d_strict) — pre-v2 semantics
    all_pass_loose: bool           # (a ∧ (b is True OR None) ∧ c ∧ d_loose) — pre-v2 semantics
    # v2 picker semantics — (a) ∧ (c) ∧ (d_*) firm + (b) True or Unknown:
    selected_strict: bool          # a=True ∧ c=True ∧ d_strict=True ∧ b=True (firm)
    selected_loose: bool           # a=True ∧ c=True ∧ d_loose=True ∧ b=True (firm)
    confidence: str                # "strong" (all 4 True) | "moderate" (b is None, rest True) | "weak" (selected but b=False) | "none"


def evaluate_candidate(inp: CandidateInputs) -> CandidateVerdict:
    """Apply all four conditions to a single ATTACK action.

    Production picker semantics (ship-1):
    - ``selected_strict`` / ``selected_loose`` accept when (a), (c), and
      (d_*) are all True AND (b) is firm True. Pose-Unknown is REJECTED
      (the v2 probe scored 0/3 on moderate-tier and 7/7 on strong-tier).
    - ``confidence`` is ``strong`` when all four conditions confirm True,
      ``moderate`` when (b) is Unknown but the other three confirm,
      ``weak`` when (b) is False but the others are True, and ``none``
      otherwise. The diagnostic field is preserved for telemetry; the
      picker now requires firm.

    Legacy v1 fields ``all_pass_strict`` / ``all_pass_loose`` are kept
    for backward compatibility.
    """
    a_pass, a_src = check_a_at_net_v2(
        player_bbox_top_y_norm=inp.player_bbox_top_y_image,
        net_y_image_norm=inp.net_y_image,
        ball_y_image=inp.ball_y_image,
    )
    b_pass, b_reason = check_b_wrist_above_net(inp.wrist_y_image, inp.net_y_image)
    c_pass = check_c_direction_change(inp.direction_change_deg)
    d_strict, d_strict_reason = check_d_strict(inp.action, inp.prev_action)
    d_loose, d_loose_reason = check_d_loose(inp.action, inp.prev_action)

    # Legacy fields (pre-v2 semantics) kept for callers/tests.
    all_strict = bool(a_pass and (b_pass is True) and c_pass and d_strict)
    all_loose = bool(
        a_pass and (b_pass is True or b_pass is None) and c_pass and d_loose
    )

    # Production picker (ship-1, 2026-05-14): require (b) firm True.
    # Pose-Unknown is REJECTED. Justification: v2 probe scored 0/3 on
    # moderate (Unknown) candidates vs 7/7 on strong (firm), so the
    # production gate refuses Unknown.
    b_firm_ok = b_pass is True
    selected_strict = bool(a_pass and c_pass and d_strict and b_firm_ok)
    selected_loose = bool(a_pass and c_pass and d_loose and b_firm_ok)

    if a_pass and c_pass and (d_strict or d_loose) and b_pass is True:
        confidence = "strong"
    elif a_pass and c_pass and (d_strict or d_loose) and b_pass is None:
        confidence = "moderate"
    elif a_pass and c_pass and (d_strict or d_loose) and b_pass is False:
        confidence = "weak"
    else:
        confidence = "none"

    return CandidateVerdict(
        a_passes=a_pass,
        a_source=a_src,
        b_passes=b_pass,
        b_reason=b_reason,
        c_passes=c_pass,
        d_strict_passes=d_strict,
        d_strict_reason=d_strict_reason,
        d_loose_passes=d_loose,
        d_loose_reason=d_loose_reason,
        all_pass_strict=all_strict,
        all_pass_loose=all_loose,
        selected_strict=selected_strict,
        selected_loose=selected_loose,
        confidence=confidence,
    )


# --- Producer-side gate ------------------------------------------------------


def should_reclassify_to_block(
    action: dict[str, Any],
    prev_action: dict[str, Any] | None,
    direction_change_deg: float,
    ball_y_image: float,
    player_bbox_top_y_image: float | None,
    net_y_image: float | None,
    wrist_y_image: float | None,
    *,
    require_firm_wrist: bool = True,
    use_strict_d: bool = False,
) -> bool:
    """Production-grade gate for A3 BLOCK reclassification.

    Returns True iff ALL of:
      (a)′ at-net (HEAD-near-BALL-at-net in image coords) — see
           ``check_a_at_net_v2``.
      (b)′ wrist firmly above net — ``wrist_y_image < net_y_image`` AND
           wrist is detected (``wrist_y_image is not None``). When
           ``require_firm_wrist=True`` (production default, ship-1), pose-
           Unknown is REJECTED. The v2 probe scored 0/3 on moderate
           (Unknown) candidates and 7/7 on strong (firm), so the
           production gate refuses Unknown.
      (c)  ball direction-change ≤ 90° (deflected, not fully reversed).
      (d)  prev cross-team. ``use_strict_d=False`` (production default)
           uses LOOSE (prev cross-team AND prev NOT a serve) which
           captures the F5 canonical case (prev=receive) and the strict
           superset (prev ∈ {attack, set}).

    Side-effect-free: does NOT mutate ``action`` or ``prev_action``.

    Args:
        action: The ATTACK action dict. Must have ``team`` and (for
            diagnostics) ``frame``.
        prev_action: The immediately-preceding action dict in the rally
            (by frame order) or None if this is the first action.
        direction_change_deg: From the Contact at this action's frame.
        ball_y_image: Normalized image-y of the ball at contact (the
            action's ``ballY``).
        player_bbox_top_y_image: Normalized image-y of the suspect player's
            bbox top (head). None when no player position at that frame.
        net_y_image: Normalized image-y of the court midline ground
            projection. None when court calibration is missing.
        wrist_y_image: Normalized image-y of the higher of the suspect
            player's wrists. None when pose inference failed or wrist
            confidence was below the floor.
        require_firm_wrist: When True (default), reject if wrist is None.
            When False, accept wrist=None as soft-yes (matches the v2
            probe's ``selected_loose`` semantics — NOT for production).
        use_strict_d: When True, require prev ∈ {attack, set}. When False
            (default), use LOOSE — prev cross-team AND prev NOT a serve.
    """
    # (a)′ at-net
    a_pass, _a_src = check_a_at_net_v2(
        player_bbox_top_y_norm=player_bbox_top_y_image,
        net_y_image_norm=net_y_image,
        ball_y_image=ball_y_image,
    )
    if not a_pass:
        return False

    # (b)′ firm wrist-above-net
    b_pass, _b_reason = check_b_wrist_above_net(wrist_y_image, net_y_image)
    if require_firm_wrist:
        if b_pass is not True:
            return False
    else:
        # Soft mode: Unknown counts as soft-yes; False rejects.
        if b_pass is False:
            return False

    # (c) direction change
    if not check_c_direction_change(direction_change_deg):
        return False

    # (d) prev cross-team (loose by default — captures F5 + strict superset)
    if use_strict_d:
        d_pass, _ = check_d_strict(action, prev_action)
    else:
        d_pass, _ = check_d_loose(action, prev_action)
    return d_pass


# --- Court projection helper --------------------------------------------------

def project_image_to_court(
    point_norm: tuple[float, float],
    calibration_corners: list[dict[str, float]] | None,
) -> tuple[float, float] | None:
    """Project a normalized image point to court coordinates (meters).

    ``calibration_corners`` is the format stored in
    ``videos.court_calibration_json``: a list of 4 ``{"x", "y"}`` dicts
    (normalized to frame dims, may be outside [0,1] for wide-angle).

    Returns ``None`` if calibration is missing or invalid.

    Note: this duplicates a small bit of CourtCalibrator.image_to_court
    so the probe doesn't need to instantiate the full calibrator object.
    The math is identical.
    """
    if not calibration_corners or len(calibration_corners) != 4:
        return None
    try:
        import cv2  # local import — already a project dep

        src = np.array(
            [(c["x"], c["y"]) for c in calibration_corners],
            dtype=np.float64,
        )
        # Default beach VB court corners (matches CourtCalibrator default).
        # Order: near-left, near-right, far-right, far-left.
        # See rallycut/court/calibration.py:114-122.
        dst = np.array([
            (0.0, 0.0),
            (8.0, 0.0),
            (8.0, 16.0),
            (0.0, 16.0),
        ], dtype=np.float64)
        h_matrix, _ = cv2.findHomography(src, dst)
        if h_matrix is None:
            return None
        x, y = point_norm
        pt = np.array([x, y, 1.0], dtype=np.float64)
        result = h_matrix @ pt
        result = result / result[2]
        return float(result[0]), float(result[1])
    except Exception:  # noqa: BLE001
        return None


def estimate_net_y_image(
    calibration_corners: list[dict[str, float]] | None,
) -> float | None:
    """Estimate the net line's image-y from court calibration.

    The net runs through the midline between the near and far sidelines
    in court coords (y_court = 8 m). We project two points on the
    midline (left and right of the court) back to image coords and take
    the average of their image-y values as an approximation of where
    the net line crosses the frame center.

    Returns ``None`` if calibration is missing or projection fails.
    """
    if not calibration_corners or len(calibration_corners) != 4:
        return None
    try:
        import cv2

        src = np.array(
            [(c["x"], c["y"]) for c in calibration_corners],
            dtype=np.float64,
        )
        dst = np.array([
            (0.0, 0.0),
            (8.0, 0.0),
            (8.0, 16.0),
            (0.0, 16.0),
        ], dtype=np.float64)
        h_matrix, _ = cv2.findHomography(src, dst)
        if h_matrix is None:
            return None
        h_inv = np.linalg.inv(h_matrix)
        ys: list[float] = []
        for x_court in (0.0, 4.0, 8.0):
            pt = np.array([x_court, NET_COURT_Y, 1.0], dtype=np.float64)
            res = h_inv @ pt
            res = res / res[2]
            ys.append(float(res[1]))
        return float(np.mean(ys))
    except Exception:  # noqa: BLE001
        return None
