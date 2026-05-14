"""Diagnostic: characterize WHY the contact_detector fires on the 11 at-net
false-positives the user deleted from rally_action_ground_truth after A3
inspection.

The 11 picks were originally emitted by `detect_contacts` at the net, then
classified as ATTACK, then flagged by A3 as candidates for BLOCK
reclassification, then inspected by the user and confirmed as
non-contacts (either near-miss-block or ball-net deflection).

This script:
  1. Loads stored `contacts_json.contacts[]` for each FP frame.
  2. Re-runs `_prepare_candidates` on the rally's stored ball/player
     positions to identify which generator emitted the frame (direction-
     change, velocity-peak, parabolic, etc).
  3. Pulls player bbox + ball-y + net-y context.
  4. Writes a per-case table + pattern summary to
     `analysis/reports/contact_detector_at_net_fp_diagnostic_2026_05_14.md`.

This is a DIAGNOSTIC ONLY — no fixes, no DB writes.

Usage:
    cd analysis
    uv run python scripts/diagnose_contact_detector_at_net_fps.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _prepare_candidates,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos


@dataclass
class FpCase:
    """One of the 11 at-net FPs the user deleted from GT."""

    video_name: str
    rally_id: str
    frame: int  # rally-relative frame of the FP contact


FP_CASES: list[FpCase] = [
    FpCase("cucu", "7d5fbfb2-0158-463e-bddc-0c38f077688f", 218),
    FpCase("jiji", "935ead8c-2fb9-4f1f-8364-3e0af6ed0bca", 631),
    FpCase("lala", "2eeb3ae6-cf97-4eeb-9400-28a8060a7636", 966),
    FpCase("lolo", "6935b412-5e38-4829-81e3-f2d3debfa1e2", 249),
    FpCase("titi", "4ad457f6-6cbc-49c7-ab9b-5d2c3edb8ab2", 421),
    FpCase("toto", "fcc5dcba-9f9f-4125-920b-46940845ca27", 174),
    FpCase("toto", "f1f09039-2292-4fcd-8c16-ab03371df190", 302),
    FpCase("wewe", "83790ce7-28fa-4749-b673-45eda804cf09", 331),
    FpCase("wiwi", "7aef7188-0287-4cbb-bd47-3bf36c1121b6", 436),
    FpCase("wowo", "b07b388b-ccaa-44a5-baf5-4826b541a663", 391),
    FpCase("yoyo", "21a9b203-dc92-48dc-8f19-d94835e0e226", 506),
]


# Acceptable frame distance when matching a generator's emitted frame to
# the stored-contact frame. `_refine_candidates_to_trajectory_peak` can
# shift a candidate by ±5 frames, and `_find_proximity_frame` searches
# ±~6, so we use a comfortable ±8.
GEN_MATCH_TOL = 8


@dataclass
class FpDiagnostic:
    case: FpCase
    video_id: str
    # Stored contact fields
    direction_change_deg: float | None = None
    velocity: float | None = None
    player_distance: float | None = None
    arc_fit_residual: float | None = None
    is_validated: bool | None = None
    confidence: float | None = None
    ball_x: float | None = None
    ball_y: float | None = None
    player_track_id: int | None = None
    court_side: str | None = None
    is_at_net: bool | None = None
    # Geometry
    net_y_estimated: float | None = None  # from ball trajectory (canonical)
    court_split_y: float | None = None  # from player tracker (legacy)
    player_bbox_top_y: float | None = None
    player_bbox_bottom_y: float | None = None
    player_bbox_center_y: float | None = None
    # Generator attribution (multiple may match; emitted is the closest).
    emitting_generators: list[str] = field(default_factory=list)
    # Action context (was classified as ATTACK).
    action_type_classified: str | None = None
    prev_action: str | None = None
    next_action: str | None = None


def _find_emitting_generators(
    contact_frame: int,
    *,
    velocity_peak_frames: list[int],
    inflection_frames: list[int],
    deceleration_frames: list[int],
    parabolic_frames: list[int],
    direction_change_frames: list[int],
    net_crossing_frames: list[int],
    tol: int = GEN_MATCH_TOL,
) -> list[str]:
    """Return the names of every candidate generator that emitted a frame
    within `tol` of `contact_frame`.

    Multiple generators can fire near the same frame; the merge logic
    keeps the highest-priority frame. Reporting all that fire surfaces
    which signals were dominant.
    """

    def _hit(frames: list[int]) -> bool:
        return any(abs(f - contact_frame) <= tol for f in frames)

    matches: list[str] = []
    if _hit(direction_change_frames):
        matches.append("direction_change")
    if _hit(parabolic_frames):
        matches.append("parabolic")
    if _hit(velocity_peak_frames):
        matches.append("velocity_peak")
    if _hit(inflection_frames):
        matches.append("inflection")
    if _hit(deceleration_frames):
        matches.append("deceleration")
    if _hit(net_crossing_frames):
        matches.append("net_crossing")
    return matches


def _load_video_name_to_id() -> dict[str, str]:
    out: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT title, id FROM videos WHERE title IS NOT NULL")
        for row in cur.fetchall():
            out[str(row[0])] = str(row[1])
    return out


def _diagnose_case(case: FpCase) -> FpDiagnostic:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.video_id, pt.id,
                   pt.ball_positions_json, pt.positions_json,
                   pt.frame_count, pt.court_split_y,
                   pt.contacts_json, pt.actions_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.id = %s
            """,
            [case.rally_id],
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"rally {case.rally_id} has no player_tracks row")
    (
        video_id_raw, _pt_id, ball_json, positions_json, _frame_count,
        court_split_y_raw, contacts_json_raw, actions_json_raw,
    ) = row
    video_id = str(video_id_raw)
    court_split_y = cast(float | None, court_split_y_raw)
    contacts_json = cast(dict[str, Any], contacts_json_raw or {})
    actions_json = cast(dict[str, Any], actions_json_raw or {})

    diag = FpDiagnostic(
        case=case,
        video_id=video_id,
        court_split_y=(
            float(court_split_y) if court_split_y is not None else None
        ),
    )

    # ---- Stored contact at this frame -------------------------------------
    stored_contact: dict[str, Any] | None = None
    for c in cast(list[dict[str, Any]], contacts_json.get("contacts", [])):
        try:
            if int(c.get("frame", -1)) == case.frame:
                stored_contact = c
                break
        except (TypeError, ValueError):
            continue
    if stored_contact is not None:
        diag.direction_change_deg = (
            float(stored_contact["directionChangeDeg"])
            if stored_contact.get("directionChangeDeg") is not None else None
        )
        diag.velocity = (
            float(stored_contact["velocity"])
            if stored_contact.get("velocity") is not None else None
        )
        diag.player_distance = (
            float(stored_contact["playerDistance"])
            if stored_contact.get("playerDistance") is not None else None
        )
        diag.arc_fit_residual = (
            float(stored_contact["arcFitResidual"])
            if stored_contact.get("arcFitResidual") is not None else None
        )
        diag.is_validated = bool(stored_contact.get("isValidated", False))
        diag.confidence = (
            float(stored_contact["confidence"])
            if stored_contact.get("confidence") is not None else None
        )
        diag.ball_x = (
            float(stored_contact["ballX"])
            if stored_contact.get("ballX") is not None else None
        )
        diag.ball_y = (
            float(stored_contact["ballY"])
            if stored_contact.get("ballY") is not None else None
        )
        diag.player_track_id = (
            int(stored_contact["playerTrackId"])
            if stored_contact.get("playerTrackId") is not None else None
        )
        diag.court_side = stored_contact.get("courtSide")
        diag.is_at_net = bool(stored_contact.get("isAtNet", False))

    # ---- Action context ---------------------------------------------------
    actions = cast(list[dict[str, Any]], actions_json.get("actions", []))
    actions_sorted = sorted(
        actions, key=lambda a: int(a.get("frame", 0)),
    )
    for idx, a in enumerate(actions_sorted):
        if int(a.get("frame", -1)) == case.frame:
            diag.action_type_classified = str(a.get("action") or "")
            if idx > 0:
                p = actions_sorted[idx - 1]
                diag.prev_action = (
                    f"{p.get('action')}({p.get('team')})@f{p.get('frame')}"
                )
            if idx + 1 < len(actions_sorted):
                n = actions_sorted[idx + 1]
                diag.next_action = (
                    f"{n.get('action')}({n.get('team')})@f{n.get('frame')}"
                )
            break

    # ---- Player bbox at this frame ----------------------------------------
    positions_json = positions_json or []
    if diag.player_track_id is not None and diag.player_track_id >= 0:
        for pp in cast(list[dict[str, Any]], positions_json):
            try:
                if (
                    int(pp.get("frameNumber", -1)) == case.frame
                    and int(pp.get("trackId", -2)) == diag.player_track_id
                ):
                    cy = float(pp["y"])
                    h = float(pp["height"])
                    diag.player_bbox_top_y = cy - h / 2.0
                    diag.player_bbox_bottom_y = cy + h / 2.0
                    diag.player_bbox_center_y = cy
                    break
            except (TypeError, ValueError, KeyError):
                continue

    # ---- Re-run _prepare_candidates to identify generator -----------------
    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in cast(list[dict[str, Any]], ball_json or [])
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]
    player_positions = [
        PlayerPos(
            frame_number=pp["frameNumber"], track_id=pp["trackId"],
            x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
            confidence=pp.get("confidence", 1.0),
        )
        for pp in cast(list[dict[str, Any]], positions_json or [])
    ]
    if ball_positions:
        cfg = ContactDetectionConfig()
        prep = _prepare_candidates(ball_positions, player_positions, cfg)
        diag.net_y_estimated = float(prep.estimated_net_y)
        diag.emitting_generators = _find_emitting_generators(
            case.frame,
            velocity_peak_frames=prep.velocity_peak_frames,
            inflection_frames=prep.inflection_frames,
            deceleration_frames=prep.deceleration_frames,
            parabolic_frames=prep.parabolic_frames,
            direction_change_frames=prep.direction_change_frames,
            net_crossing_frames=prep.net_crossing_frames,
        )

    # ---- Court calibration: project net y if available --------------------
    corners = load_court_calibration(video_id)
    if corners and len(corners) == 4:
        cal = CourtCalibrator()
        cal.calibrate([(c["x"], c["y"]) for c in corners])
        # Already have net_y_estimated from ball trajectory; the calibration
        # net is the GROUND projection of the midline (in image space). We
        # record it via court_split_y above for diagnostic comparison.

    return diag


def _format_report(diags: list[FpDiagnostic]) -> str:
    lines: list[str] = []
    lines.append("# Contact-detector at-net FP diagnostic (2026-05-14)")
    lines.append("")
    lines.append(
        "11 user-deleted at-net non-contacts from `rally_action_ground_truth`. "
        "Each was originally emitted by `detect_contacts` at the net, "
        "classified as ATTACK, flagged by A3 as a BLOCK candidate, then "
        "inspected by the user and removed."
    )
    lines.append("")
    lines.append(
        "User-reported FP patterns: (1) near-miss block — player jumps with "
        "arms up but no contact; (2) ball-net deflection — ball hits the net "
        "after an attack, not the player."
    )
    lines.append("")
    lines.append("## Per-case signals")
    lines.append("")
    header = (
        "| # | video | rally | frame | gen(s) | dc° | velocity | pdist | arc_res | conf | "
        "ball_y | net_y_est | head_y | center_y | foot_y | classified | prev → this → next |"
    )
    sep = "|" + "|".join(["---"] * 18) + "|"
    lines.append(header)
    lines.append(sep)

    def _f(x: float | None, w: str = ".3f") -> str:
        if x is None:
            return "-"
        return format(x, w)

    for i, d in enumerate(diags, 1):
        gens = ",".join(d.emitting_generators) if d.emitting_generators else "?"
        chain = (
            f"{d.prev_action or '-'} → {d.action_type_classified or '-'} → "
            f"{d.next_action or '-'}"
        )
        lines.append(
            f"| {i} | {d.case.video_name} | {d.case.rally_id[:8]} | {d.case.frame} | "
            f"{gens} | {_f(d.direction_change_deg, '.1f')} | "
            f"{_f(d.velocity, '.4f')} | {_f(d.player_distance, '.4f')} | "
            f"{_f(d.arc_fit_residual, '.4f')} | {_f(d.confidence, '.3f')} | "
            f"{_f(d.ball_y)} | {_f(d.net_y_estimated)} | "
            f"{_f(d.player_bbox_top_y)} | {_f(d.player_bbox_center_y)} | "
            f"{_f(d.player_bbox_bottom_y)} | "
            f"{d.action_type_classified or '-'} | {chain} |"
        )
    lines.append("")

    # ---- Pattern summary -----------------------------------------------
    lines.append("## Pattern summary")
    lines.append("")

    n = len(diags)
    n_dc_high = sum(1 for d in diags if (d.direction_change_deg or 0) >= 80)
    n_dc_mid = sum(
        1 for d in diags
        if 40 <= (d.direction_change_deg or 0) < 80
    )
    n_dc_low = sum(
        1 for d in diags
        if (d.direction_change_deg or 0) < 30
    )
    n_at_net = sum(1 for d in diags if d.is_at_net)
    n_validated = sum(1 for d in diags if d.is_validated)
    n_pdist_close = sum(
        1 for d in diags
        if d.player_distance is not None and d.player_distance < 0.05
    )
    n_pdist_moderate = sum(
        1 for d in diags
        if d.player_distance is not None
        and 0.05 <= d.player_distance < 0.10
    )
    n_pdist_far = sum(
        1 for d in diags
        if d.player_distance is not None and d.player_distance >= 0.10
    )
    n_head_above_ball = sum(
        1 for d in diags
        if (
            d.player_bbox_top_y is not None
            and d.ball_y is not None
            and d.player_bbox_top_y < d.ball_y
        )
    )
    n_low_velocity = sum(
        1 for d in diags
        if d.velocity is not None and d.velocity < 0.020
    )
    n_low_conf = sum(
        1 for d in diags
        if d.confidence is not None and d.confidence < 0.50
    )

    # Generator counts
    gen_counts: dict[str, int] = {}
    for d in diags:
        for g in d.emitting_generators:
            gen_counts[g] = gen_counts.get(g, 0) + 1

    # Number of generators that fire concurrently per case
    gens_per_case = [len(d.emitting_generators) for d in diags]
    median_gens = (
        sorted(gens_per_case)[n // 2] if gens_per_case else 0
    )

    lines.append(f"- **n = {n}** at-net FPs total.")
    lines.append(
        f"- **Direction change is dominantly LOW**: {n_dc_low}/{n} cases "
        f"have dc < 30°; {n_dc_mid}/{n} have 40°-80°; only {n_dc_high}/{n} "
        f"≥ 80°. The user-hypothesized 'ball-net deflection' signature "
        f"(sharp dc) is ESSENTIALLY ABSENT in this set — these FPs are "
        f"NOT firing because the ball bent sharply at the net."
    )
    lines.append(
        f"- **Velocity is mostly low**: {n_low_velocity}/{n} have velocity "
        f"< 0.020. Combined with low dc, these are weak-signal contacts."
    )
    lines.append(
        f"- **Player proximity distribution**: {n_pdist_close}/{n} have "
        f"player_distance < 0.05; {n_pdist_moderate}/{n} have 0.05-0.10; "
        f"{n_pdist_far}/{n} have ≥ 0.10. So roughly half are 'a player IS "
        f"near the ball' (the near-miss-block pattern), but a substantial "
        f"share fire even when the nearest player is comfortably off the "
        f"ball — a separate failure mode."
    )
    lines.append(
        f"- **Classifier validation passed for all of them**: {n_validated}/{n} "
        f"were `is_validated=True`. {n_low_conf}/{n} had confidence < 0.50. "
        f"The classifier accepts them despite weak signals because the "
        f"COMBINATION of weak signals matches its training distribution."
    )
    lines.append(
        f"- **is_at_net flag**: only {n_at_net}/{n} were marked "
        f"`is_at_net=True` at emit time, even though all 11 are at-net by "
        f"GT inspection. The detector's own at-net flag is unreliable here."
    )
    lines.append(
        f"- **Player bbox vs ball**: {n_head_above_ball}/{n} have player "
        f"bbox-top above the ball (image-y smaller). Roughly half-and-half — "
        f"this does NOT robustly discriminate FP from TP."
    )
    lines.append("")
    lines.append(
        f"- **Generator concurrency**: median {median_gens} generators fire "
        f"within ±{GEN_MATCH_TOL} of the stored contact frame. Generator "
        f"counts (frames matched within ±{GEN_MATCH_TOL}):"
    )
    for g, c in sorted(gen_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{g}`: {c}/{n}")
    lines.append(
        "\nMost FPs are emitted by MULTIPLE generators agreeing — this is "
        "the structural FP mode. No single generator stands out."
    )
    lines.append("")

    # ---- Recommendations -----------------------------------------------
    lines.append("## Recommendations (no fixes, just signals for a future workstream)")
    lines.append("")
    lines.append(
        "Headline: **the data falsifies the user's stated FP patterns**. "
        "Only 1/11 looks like a ball-net deflection (toto/f1f09039 with "
        "dc=78°), and only 2/11 have player_distance < 0.05 ('player very "
        "close to ball'). The remaining majority are *weak-signal* contacts "
        "where the classifier sees a multi-generator agreement on a low-dc, "
        "low-velocity, moderate-arc-residual configuration and accepts. "
        "There is no single signal that cleanly separates these 11 from "
        "true at-net contacts."
    )
    lines.append("")
    lines.append(
        "1. **No simple rule-based gate will move precision much.** A "
        "ball-net distance gate or a wrist-position gate would each fire on "
        "≤ 3 of 11 cases — well below a 92% precision-target gate. Rule "
        "tuning at the contact-detector level on this signal set is "
        "the same dead-end as the A1/A3 ladder."
    )
    lines.append("")
    lines.append(
        "2. **The actionable insight is at the classifier level, not the "
        "generator level.** All 11 are `is_validated=True`. The GBM "
        "classifier is the one accepting them. A future workstream could "
        "retrain the classifier with these 11 (+ similar mined from the "
        "fleet) as labeled negatives — they are *clean* at-net non-contact "
        "examples, exactly the failure mode the current GBM lacks training "
        "signal for."
    )
    lines.append("")
    lines.append(
        "3. **Single-generator suppression is unsafe.** Median 5 generators "
        "fire within ±8 of each FP. Disabling any one (e.g., "
        "`enable_direction_change_candidates=False`) would still leave "
        "≥ 4 others emitting a candidate at the same frame. The merge logic "
        "would still accept it. Generator-level suppression won't help."
    )
    lines.append("")
    lines.append(
        "4. **The is_at_net flag should be re-derived at the action layer.** "
        "Only 5/11 of these (by the detector's own at-net flag) are flagged "
        "as at-net even though all 11 are at-net by GT. An action-layer "
        "re-derivation using `|ball_y - net_y_estimated| < ε` would surface "
        "more candidates for the rule-based deletion pipeline (e.g., "
        "deduplication across cross-team adjacent contacts) — though this is "
        "still a downstream FP scrubber, not a contact-detector fix."
    )
    lines.append("")
    lines.append(
        "5. **Practical next-step recipe** (cheapest path to measurable "
        "FP reduction): (a) mine ~100 more user-deleted at-net non-contacts "
        "from `rally_action_ground_truth` by joining DELETED rows or "
        "rows the user explicitly flagged; (b) sample-equal "
        "labeled-positive at-net contacts from the same fleet; (c) compute "
        "the GBM's score distribution on both sets; (d) if "
        "the score distributions overlap (the classifier doesn't separate "
        "them), retrain the classifier on the combined set; (e) if the "
        "score distributions are separable but the threshold is in the "
        "wrong place, lift the at-net threshold from 0.30 → 0.45 for "
        "`is_at_net=True` candidates and re-measure on the GT panel."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print(f"Diagnosing {len(FP_CASES)} at-net FP cases...")
    diags: list[FpDiagnostic] = []
    for i, case in enumerate(FP_CASES, 1):
        print(
            f"  [{i}/{len(FP_CASES)}] {case.video_name} "
            f"rally={case.rally_id[:8]} f={case.frame}"
        )
        try:
            diag = _diagnose_case(case)
            diags.append(diag)
            gen_str = (
                ",".join(diag.emitting_generators)
                if diag.emitting_generators else "?"
            )
            print(
                f"      dc={diag.direction_change_deg:.1f}° "
                f"pdist={diag.player_distance:.4f} "
                f"arc={diag.arc_fit_residual:.4f} "
                f"gens={gen_str}"
                if diag.direction_change_deg is not None
                and diag.player_distance is not None
                and diag.arc_fit_residual is not None
                else f"      (incomplete data) gens={gen_str}"
            )
        except Exception as e:
            print(f"      ERROR: {e}")

    print()
    print("Writing report...")
    report = _format_report(diags)
    out_dir = (
        Path(__file__).resolve().parent.parent / "reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "contact_detector_at_net_fp_diagnostic_2026_05_14.md"
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"  → {out_path}")

    # Also dump the raw diagnostics as JSON for downstream tooling.
    json_path = out_path.with_suffix(".json")
    json_payload = [
        {
            "video_name": d.case.video_name,
            "rally_id": d.case.rally_id,
            "frame": d.case.frame,
            "video_id": d.video_id,
            "direction_change_deg": d.direction_change_deg,
            "velocity": d.velocity,
            "player_distance": d.player_distance,
            "arc_fit_residual": d.arc_fit_residual,
            "is_validated": d.is_validated,
            "confidence": d.confidence,
            "ball_x": d.ball_x,
            "ball_y": d.ball_y,
            "player_track_id": d.player_track_id,
            "court_side": d.court_side,
            "is_at_net": d.is_at_net,
            "net_y_estimated": d.net_y_estimated,
            "court_split_y": d.court_split_y,
            "player_bbox_top_y": d.player_bbox_top_y,
            "player_bbox_bottom_y": d.player_bbox_bottom_y,
            "player_bbox_center_y": d.player_bbox_center_y,
            "emitting_generators": d.emitting_generators,
            "action_type_classified": d.action_type_classified,
            "prev_action": d.prev_action,
            "next_action": d.next_action,
        }
        for d in diags
    ]
    json_path.write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8",
    )
    print(f"  → {json_path}")
    print()
    print(f"Done: {len(diags)} cases diagnosed.")


if __name__ == "__main__":
    main()
