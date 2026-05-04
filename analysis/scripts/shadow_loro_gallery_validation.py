"""
Layer 1 — Leave-One-Rally-Out gallery validation.

Validates the core primitive of the proposed identity-first matcher:
"For each rally, aggregate per-PID appearance profiles from other rallies
in the same video, then assign this rally's primary tracks to PIDs by
appearance similarity via Hungarian. Does this recover the existing
PIDs on healthy 4-track rallies?"

If yes (>=95% agreement on GT fixtures), the gallery primitive is
sound and the proposed design can proceed.
If no, the design fails fast — gallery-based identity is insufficient
and we need a richer signal.

Read-only. No DB writes. Uses persisted features from
match_analysis_json.rallyScratchpad.rallies[].track_stats — the same
features the production matcher uses.

Run:
    uv run python scripts/shadow_loro_gallery_validation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.evaluation.tracking.db import get_connection


GT_FIXTURES = [
    ("5c756c41", "5c756c41-1cc1-4486-a95c-97398912cfbe"),
    ("b5fb0594", "b5fb0594-d64f-4a0d-bad9-de8fc36414d0"),
    ("854bb250", "854bb250-3e91-47d2-944d-f62413e3cf45"),
    ("7d77980f", "7d77980f-3006-40e0-adc0-db491a5bb659"),
]

DD042609 = ("dd042609", "dd042609-e22e-4f60-83ed-038897c88c32")


@dataclass
class TrackFeature:
    """Per-track appearance signature, normalized for distance."""
    upper_hist: np.ndarray   # 128-dim L1-normalized
    lower_hist: np.ndarray   # 128-dim L1-normalized
    skin_tone: np.ndarray    # 3-dim HSV
    dominant_color: np.ndarray  # 3-dim HSV
    reid: np.ndarray | None  # 128-dim L2-normalized

    @classmethod
    def from_track_stats(cls, ts: dict) -> "TrackFeature | None":
        try:
            upper = np.asarray(ts.get("avg_upper_hist") or [], dtype=np.float32)
            lower = np.asarray(ts.get("avg_lower_hist") or [], dtype=np.float32)
            skin_raw = ts.get("avg_skin_tone_hsv") or []
            dom_raw = ts.get("avg_dominant_color_hsv") or []
            # Normalize HSV-like fields to fixed 3-dim — some tracks have empty
            skin = np.asarray(skin_raw, dtype=np.float32) if len(skin_raw) == 3 else np.zeros(3, dtype=np.float32)
            dom = np.asarray(dom_raw, dtype=np.float32) if len(dom_raw) == 3 else np.zeros(3, dtype=np.float32)
            reid_list = ts.get("reid_embedding")
            reid = np.asarray(reid_list, dtype=np.float32) if reid_list else None
        except Exception:
            return None
        if upper.size == 0 or lower.size == 0:
            return None
        # Histogram lengths must match across tracks for aggregation
        if upper.size != 128 or lower.size != 128:
            return None
        if reid is not None and reid.size != 128:
            reid = None
        return cls(upper, lower, skin, dom, reid)


def _aggregate(features: list[TrackFeature]) -> TrackFeature | None:
    """Per-PID gallery profile = mean of contributing track features."""
    if not features:
        return None
    upper = np.mean([f.upper_hist for f in features], axis=0)
    lower = np.mean([f.lower_hist for f in features], axis=0)
    skin = np.mean([f.skin_tone for f in features], axis=0)
    dom = np.mean([f.dominant_color for f in features], axis=0)
    reids = [f.reid for f in features if f.reid is not None]
    if reids:
        reid_avg = np.mean(reids, axis=0)
        norm = np.linalg.norm(reid_avg)
        if norm > 0:
            reid_avg = reid_avg / norm
    else:
        reid_avg = None
    return TrackFeature(upper, lower, skin, dom, reid_avg)


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between L1-normalized histograms in [0, 2]."""
    return float(np.sum(np.abs(a - b)))


def _hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hue circular + S/V Euclidean. a, b are 3-dim (H[0..180], S[0..255], V[0..255])."""
    if a.size != 3 or b.size != 3:
        return 1.0
    dh = abs(a[0] - b[0])
    dh = min(dh, 180.0 - dh) / 90.0  # normalize to [0, 1]
    ds = abs(a[1] - b[1]) / 255.0
    dv = abs(a[2] - b[2]) / 255.0
    return float(dh * 0.5 + ds * 0.25 + dv * 0.25)


def _reid_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine distance for L2-normalized embeddings, in [0, 2]."""
    if a is None or b is None:
        return None
    return float(1.0 - np.dot(a, b))


def cost(track: TrackFeature, gallery: TrackFeature) -> float:
    """Combined appearance cost. Lower = more similar.

    Same weighting style the production matcher uses (HSV histograms +
    ReID + dominant color + skin tone). Picked to be defensible, not
    tuned — if LORO works at this weighting, the primitive is sound.
    """
    upper_d = _hist_distance(track.upper_hist, gallery.upper_hist)
    lower_d = _hist_distance(track.lower_hist, gallery.lower_hist)
    dom_d = _hsv_distance(track.dominant_color, gallery.dominant_color)
    skin_d = _hsv_distance(track.skin_tone, gallery.skin_tone)
    reid_d = _reid_distance(track.reid, gallery.reid)
    if reid_d is not None:
        # Equal weight: ReID + 2 histograms + 2 HSV signals
        return 0.4 * reid_d + 0.2 * upper_d + 0.2 * lower_d + 0.1 * dom_d + 0.1 * skin_d
    return 0.4 * upper_d + 0.4 * lower_d + 0.1 * dom_d + 0.1 * skin_d


def _is_anchor_rally(rally_meta: dict, scratchpad_rally: dict) -> bool:
    """An anchor rally has 4 top_tracks, all valid track_stats, all 4 PIDs assigned."""
    top = scratchpad_rally.get("top_tracks", [])
    if len(top) != 4:
        return False
    ts = scratchpad_rally.get("track_stats", {})
    if not all(str(t) in ts for t in top):
        return False
    afm = rally_meta.get("appliedFullMapping", {})
    assigned_pids = {int(v) for k, v in afm.items() if int(v) > 0 and int(k) in top}
    return assigned_pids == {1, 2, 3, 4}


def _track_to_pid_for_top(rally_meta: dict, top: list[int]) -> dict[int, int]:
    """Extract the {track_id: pid} mapping for top tracks."""
    afm = rally_meta.get("appliedFullMapping", {})
    out = {}
    for k, v in afm.items():
        try:
            tid, pid = int(k), int(v)
        except Exception:
            continue
        if pid > 0 and tid in top:
            out[tid] = pid
    return out


def loro_validate_video(video_id: str, label: str) -> dict:
    """LORO validation for a single video."""
    with get_connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s", (video_id,)
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return {"label": label, "error": "no match_analysis_json"}
    maj = row[0]
    rallies = maj.get("rallies", [])
    sp_rallies = maj.get("rallyScratchpad", {}).get("rallies", [])
    if len(rallies) != len(sp_rallies):
        return {"label": label, "error": f"length mismatch {len(rallies)} vs {len(sp_rallies)}"}

    # Build per-rally feature inventory
    rally_inventory = []
    for i, (r, sp) in enumerate(zip(rallies, sp_rallies)):
        if not _is_anchor_rally(r, sp):
            rally_inventory.append(None)
            continue
        top = sp["top_tracks"]
        track_to_pid = _track_to_pid_for_top(r, top)
        if len(track_to_pid) != 4:
            rally_inventory.append(None)
            continue
        # Build features for the 4 top tracks
        features_by_pid = {}
        ok = True
        for tid in top:
            ts = sp["track_stats"].get(str(tid))
            if not ts:
                ok = False
                break
            f = TrackFeature.from_track_stats(ts)
            if f is None:
                ok = False
                break
            pid = track_to_pid[tid]
            features_by_pid[pid] = f
        if not ok or len(features_by_pid) != 4:
            rally_inventory.append(None)
            continue
        rally_inventory.append({
            "rally_idx": i,
            "rally_id": r.get("rallyId"),
            "side_switch": r.get("sideSwitchDetected", False),
            "top_tracks": top,
            "track_to_pid": track_to_pid,
            "features_by_pid": features_by_pid,
        })

    anchor_indices = [i for i, x in enumerate(rally_inventory) if x is not None]
    print(f"\n=== {label} ({video_id[:8]}) ===")
    print(f"  rallies: {len(rallies)} total, {len(anchor_indices)} anchor")
    if len(anchor_indices) < 2:
        return {"label": label, "n_total": len(rallies), "n_anchor": len(anchor_indices),
                "n_correct": 0, "n_loro": 0, "error": "insufficient anchors"}

    # LORO loop
    n_correct = 0
    n_loro = 0
    per_rally = []
    for held_idx in anchor_indices:
        held = rally_inventory[held_idx]
        # Build gallery from all OTHER anchor rallies
        per_pid_features: dict[int, list[TrackFeature]] = defaultdict(list)
        for other_idx in anchor_indices:
            if other_idx == held_idx:
                continue
            other = rally_inventory[other_idx]
            for pid, feat in other["features_by_pid"].items():
                per_pid_features[pid].append(feat)
        gallery = {pid: _aggregate(feats) for pid, feats in per_pid_features.items()}
        if any(g is None for g in gallery.values()) or set(gallery.keys()) != {1, 2, 3, 4}:
            continue

        # Cost matrix [4 tracks × 4 PIDs]
        top_list = held["top_tracks"]
        cost_mat = np.zeros((4, 4), dtype=np.float32)
        for ti, tid in enumerate(top_list):
            track_feat = held["features_by_pid"][held["track_to_pid"][tid]]
            for pj, pid in enumerate([1, 2, 3, 4]):
                cost_mat[ti, pj] = cost(track_feat, gallery[pid])

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        predicted = {top_list[ti]: int([1, 2, 3, 4][col_ind[ti]]) for ti in range(4)}

        actual = held["track_to_pid"]
        match = predicted == actual
        n_loro += 1
        if match:
            n_correct += 1

        per_rally.append({
            "rally_idx": held_idx,
            "rally_id": held["rally_id"][:8] if held["rally_id"] else "?",
            "side_switch": held["side_switch"],
            "actual": actual,
            "predicted": predicted,
            "match": match,
            "cost_diag": [float(cost_mat[i, list([1,2,3,4]).index(actual[top_list[i]])]) for i in range(4)],
        })

    pct = (n_correct / n_loro * 100) if n_loro else 0.0
    print(f"  LORO: {n_correct}/{n_loro} = {pct:.1f}%")
    for pr in per_rally:
        flag = "✓" if pr["match"] else "✗"
        sw = " [SW]" if pr["side_switch"] else ""
        print(f"    r{pr['rally_idx']+1:2d} ({pr['rally_id']}){sw} {flag} "
              f"actual={pr['actual']} predicted={pr['predicted']}")

    return {
        "label": label,
        "video_id": video_id,
        "n_total": len(rallies),
        "n_anchor": len(anchor_indices),
        "n_loro": n_loro,
        "n_correct": n_correct,
        "pct": pct,
        "per_rally": per_rally,
    }


def main() -> None:
    print("=== Layer 1: LORO gallery validation ===")
    print("Tests whether per-PID appearance gallery (built from other rallies in the same video)")
    print("can recover the existing PIDs of a held-out 4-track rally via Hungarian assignment.")
    print()

    results = []
    for label, vid in GT_FIXTURES + [DD042609]:
        results.append(loro_validate_video(vid, label))

    print("\n\n=== SUMMARY ===")
    print(f"{'video':<10} {'rallies':<10} {'anchors':<10} {'loro':<10} {'correct':<10} {'pct':<10}")
    for r in results:
        if "error" in r:
            print(f"{r['label']:<10} ERROR: {r['error']}")
            continue
        print(f"{r['label']:<10} {r['n_total']:<10} {r['n_anchor']:<10} "
              f"{r['n_loro']:<10} {r['n_correct']:<10} {r['pct']:<10.1f}")

    gt_results = [r for r in results if r["label"] in {x[0] for x in GT_FIXTURES} and "error" not in r]
    if gt_results:
        total_loro = sum(r["n_loro"] for r in gt_results)
        total_correct = sum(r["n_correct"] for r in gt_results)
        gt_pct = (total_correct / total_loro * 100) if total_loro else 0.0
        print(f"\nGT FIXTURES AGGREGATE: {total_correct}/{total_loro} = {gt_pct:.1f}%")
        if gt_pct >= 95.0:
            print("✓ PASS — gallery primitive is sound. Proceed to Layer 2.")
        elif gt_pct >= 85.0:
            print("⚠ MARGINAL — gallery primitive mostly works but some failures.")
            print("  Inspect failed rallies; may need within-team tie-breakers or side-aware gallery.")
        else:
            print("✗ FAIL — gallery primitive insufficient. Design needs richer signal.")


if __name__ == "__main__":
    main()
