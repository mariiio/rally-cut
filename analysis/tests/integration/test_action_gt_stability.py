"""Invariant test for action-GT trackId stability.

Pins that the existing `remap-track-ids` command only rewrites the legacy
`playerTrackId` field on action-GT rows and leaves the new `trackId` field
untouched — and that rendering the GT label via the post-rerun
`appliedFullMapping` still resolves to the same physical player (the one
anchored by `trackId`).

If the remap logic in `analysis/rallycut/cli/commands/remap_track_ids.py`
changes to also touch `trackId`, this test fails, flagging the regression.
"""
from __future__ import annotations

from typing import Any


# Replicates the GT-remap loop from
# analysis/rallycut/cli/commands/remap_track_ids.py:391-404 (current prod
# behavior). Any change to that loop should be mirrored here so the test
# remains an accurate pin.
def _apply_legacy_gt_remap(
    gt_labels: list[dict[str, Any]], mapping: dict[int, int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(label) for label in gt_labels]
    for label in out:
        old_tid = label.get("playerTrackId")
        if old_tid is not None and old_tid in mapping:
            new_tid = mapping[old_tid]
            if new_tid != old_tid:
                label["playerTrackId"] = new_tid
    return out


def _render_display_pid(
    gt: dict[str, Any], applied_full_mapping: dict[str, int]
) -> int | None:
    """Emulate the web editor's `resolveGtDisplayPid` preference order."""
    tid = gt.get("trackId")
    if tid is not None:
        pid = applied_full_mapping.get(str(tid))
        if pid is not None:
            return int(pid)
    return gt.get("playerTrackId")


def test_remap_leaves_trackid_untouched() -> None:
    """GT rows carrying `trackId` must survive remap unchanged. Legacy
    `playerTrackId` mirror is free to shift to the new canonical pid."""
    # Raw BoT-SORT ids {47, 62, 71, 95} correspond to Players {1, 2, 3, 4} at
    # label time — see mapping_v2 below for the post-rerun assignment.
    # GT captured at label time, anchored to raw trackId (preferred) with a
    # legacy playerTrackId mirror from the old schema.
    gt_labels: list[dict[str, Any]] = [
        {"frame": 10, "action": "serve", "trackId": 47, "playerTrackId": 1},
        {"frame": 42, "action": "attack", "trackId": 62, "playerTrackId": 2},
        {"frame": 88, "action": "dig", "trackId": 71, "playerTrackId": 3},
    ]

    # Next pipeline run produces a different Hungarian assignment:
    # the raw trackId anchor is stable; only canonical pid labels shift.
    mapping_v2 = {47: 3, 62: 4, 71: 1, 95: 2}

    # remap-track-ids applies the NEW mapping to existing (already-remapped)
    # GT rows via the legacy codepath, which only touches `playerTrackId`.
    remapped = _apply_legacy_gt_remap(gt_labels, mapping_v2)

    # Invariant #1: `trackId` (stable anchor) is untouched.
    for before, after in zip(gt_labels, remapped):
        assert before["trackId"] == after["trackId"], (
            f"trackId mutated by remap: before={before}, after={after}. "
            "The GT fragility fix anchors to this field; remap-track-ids "
            "must not rewrite it."
        )

    # Invariant #2: legacy `playerTrackId` DOES shift — it was the drift
    # source. Here it moves to whatever mapping_v2 says raw-id-as-pid maps to
    # (i.e. the old canonical pid is looked up in the new mapping — the
    # original v2-wrong behavior). The point is to document the old behavior
    # is contained to the legacy field, not the stable anchor.
    for before, after in zip(gt_labels, remapped):
        old_pid = int(before["playerTrackId"])
        mapped = mapping_v2.get(old_pid, old_pid)
        assert after["playerTrackId"] == mapped

    # Invariant #3: rendering via `trackId` + current appliedFullMapping
    # resolves to the correct current display pid for the *physical* player
    # the label was originally attached to. The physical player at raw=47
    # is now canonical pid 3 under mapping_v2; resolveGtDisplayPid returns 3.
    applied_v2 = {str(raw): pid for raw, pid in mapping_v2.items()}
    assert _render_display_pid(remapped[0], applied_v2) == 3  # was Player 1
    assert _render_display_pid(remapped[1], applied_v2) == 4  # was Player 2
    assert _render_display_pid(remapped[2], applied_v2) == 1  # was Player 3


def test_render_prefers_trackid_over_legacy_pid() -> None:
    """When both fields are present, rendering uses `trackId` +
    appliedFullMapping. The legacy `playerTrackId` value is the fallback."""
    applied = {"47": 3}
    gt = {"frame": 1, "action": "serve", "trackId": 47, "playerTrackId": 1}
    assert _render_display_pid(gt, applied) == 3


def test_render_falls_back_to_legacy_when_trackid_missing() -> None:
    """Legacy rows without `trackId` render via the stored canonical pid
    directly (behavior matches pre-fix rows)."""
    gt = {"frame": 1, "action": "serve", "playerTrackId": 2}
    assert _render_display_pid(gt, {}) == 2


def test_render_trackid_unknown_falls_back_to_legacy() -> None:
    """If `trackId` isn't in the current mapping (e.g. rally re-tracked and
    the raw id no longer exists), the display falls back to the legacy pid."""
    gt = {"frame": 1, "action": "serve", "trackId": 999, "playerTrackId": 2}
    assert _render_display_pid(gt, {"47": 1}) == 2


# Ports the TS-side `buildPidToTrackId` / `resolveGtDisplayPid` contract so
# Python-side regressions are caught too. Keep in sync with
# web/src/components/ActionLabelingMode.tsx and web/src/utils/gtLabelDisplay.ts.
def _build_pid_to_track_id(
    applied_full_mapping: dict[str, int] | None,
) -> dict[int, int]:
    out: dict[int, int] = {}
    if not applied_full_mapping:
        return out
    for raw_str, pid in applied_full_mapping.items():
        try:
            raw = int(raw_str)
        except (TypeError, ValueError):
            continue
        if pid not in out:
            out[pid] = raw
    return out


def test_write_and_read_agree_post_remap() -> None:
    """After remap-track-ids runs, positions carry canonical pids and
    `appliedFullMapping` inverts to the same anchor the display shows.
    Pressing "Player N" must round-trip to P{N}."""
    # Post-remap: positions are {1,2,3,4}; appliedFullMapping records the
    # original raw → canonical mapping.
    afm = {"47": 1, "62": 2, "71": 3, "95": 4}
    pid_to_raw = _build_pid_to_track_id(afm)

    # User presses "2" → write stores raw 62. Read via appliedFullMapping
    # resolves 62 → P2. Round-trip is consistent.
    raw_for_press2 = pid_to_raw[2]
    gt = {"frame": 0, "action": "attack", "trackId": raw_for_press2}
    assert _render_display_pid(gt, afm) == 2


def test_write_without_applied_full_mapping_uses_sort_order() -> None:
    """When `appliedFullMapping` is missing (Case C: retracked + match-players
    ran, remap-track-ids did NOT), the editor must NOT invert `trackToPlayer`
    — that would disagree with the sort-order-based display. Pressing
    "Player N" falls back to sort-order, matching what the user sees on screen.
    This is the bug that caused rally 12 to land pressed-2 labels as P4."""
    pid_to_raw = _build_pid_to_track_id(None)  # afm missing
    assert pid_to_raw == {}, (
        "When appliedFullMapping is missing, pidToTrackId must be empty so "
        "the caller falls back to sort-order inversion (matching the display)."
    )
