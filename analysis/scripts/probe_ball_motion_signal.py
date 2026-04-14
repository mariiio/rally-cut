"""Phase B probe P1 (revised): ball image-space motion signal.

Replaces the Otsu-based blur probe which was measuring sky/tree ellipses
when WASB produced false positives.  Instead, measures the upper bound
on available blur information directly from ball GT:

    frame-to-frame image displacement in pixels

This is the *maximum* possible blur streak length — actual blur is
this value multiplied by the exposure-time fraction (usually 0.1–0.3
for consumer outdoor footage). If ball motion is already small, blur
cannot provide useful velocity cues regardless of how sophisticated
the detector is. If ball motion is substantial (≥ ~5 px/frame), then
BlurBall-style signals *could* be extracted by a learned detector.

Why this is a better probe than contour fitting
------------------------------------------------
The contour approach required reliable ball-region segmentation from
a 96×96 crop, which fails on random-background outdoor footage where
the largest bright blob is a tree, sky patch, or player uniform rather
than the ball. The motion-based probe uses ground-truth positions
directly, so it never lies about where the ball is and it measures the
information-theoretic upper bound of what any future blur-aware detector
could extract from the footage.

Gate
----
≥30% of consecutive ball GT frame-pairs show motion ≥ 5 px/frame at
the native resolution. This corresponds to at least ~1 px blur at a
typical 1/200s exposure (which is the upper bound for outdoor bright
sunlight auto-exposure on consumer cameras).

Also reports 3x, 10x, and 20x blur thresholds for sensitivity analysis.

Usage
-----
    cd analysis
    uv run python scripts/probe_ball_motion_signal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np

from rallycut.evaluation.db import get_connection

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

# Thresholds on per-frame image-space ball motion (pixels at native resolution).
MOTION_THRESHOLDS = {
    "low_5px": 5.0,     # ≥5 px/frame → at least 1 px blur at 1/200s exposure
    "medium_10px": 10.0,
    "high_20px": 20.0,
}
GATE_PASS_RATE = 0.30

# Window around action contacts to exclude (frames).
CONTACT_EXCLUSION = 5


def load_audit_data() -> list[dict[str, Any]]:
    if not AUDIT_FILE.exists():
        print(f"ERROR: audit file not found at {AUDIT_FILE}")
        sys.exit(1)
    audit_info = json.loads(AUDIT_FILE.read_text())
    rally_ids = [r["rally_id"] for r in audit_info["audit_rallies"]]
    meta_by_id = {r["rally_id"]: r for r in audit_info["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, pt.ground_truth_json, pt.action_ground_truth_json,
                   v.width, v.height, pt.fps
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = ANY(%s)
              AND pt.ground_truth_json IS NOT NULL
        """, (rally_ids,))
        rows = cur.fetchall()

    bundles: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row[0])
        meta = meta_by_id[rid]
        gt = row[1]
        action_gt = row[2] or []
        width = int(row[3] or 1920)
        height = int(row[4] or 1080)
        fps = float(row[5] or 30.0)
        ball_gt = [
            (int(p["frameNumber"]), float(p["x"]), float(p["y"]))
            for p in (gt.get("positions", []) if isinstance(gt, dict) else gt or [])
            if isinstance(p, dict) and (p.get("label") or "").lower() == "ball"
        ]
        ball_gt.sort(key=lambda r: r[0])
        if not ball_gt:
            continue
        bundles.append({
            "rally_id": rid,
            "tier": meta["tier"],
            "camera_height_m": meta.get("camera_height_m", 0.0),
            "ball_gt": ball_gt,
            "action_gt": action_gt,
            "width": width,
            "height": height,
            "fps": fps,
        })
    return bundles


def main() -> None:
    bundles = load_audit_data()
    print(f"Loaded {len(bundles)} rallies with dense ball GT")

    all_motions_px: list[float] = []
    per_rally: list[dict[str, Any]] = []

    for bundle in bundles:
        contact_frames = {
            int(lab["frame"]) for lab in bundle["action_gt"] if "frame" in lab
        }

        def in_contact(f: int) -> bool:
            return any(abs(f - c) <= CONTACT_EXCLUSION for c in contact_frames)

        # Frame-to-frame displacement on MID-FLIGHT frames only.
        motions_px: list[float] = []
        ball = bundle["ball_gt"]
        w = bundle["width"]
        h = bundle["height"]
        for i in range(1, len(ball)):
            f0, x0, y0 = ball[i - 1]
            f1, x1, y1 = ball[i]
            if f1 - f0 != 1:
                continue  # skip gaps
            if in_contact(f0) or in_contact(f1):
                continue
            dx = (x1 - x0) * w
            dy = (y1 - y0) * h
            motions_px.append(float(np.hypot(dx, dy)))

        n = len(motions_px)
        if n == 0:
            continue
        rates = {
            k: sum(1 for m in motions_px if m >= v) / n
            for k, v in MOTION_THRESHOLDS.items()
        }
        per_rally.append({
            "rally_id": bundle["rally_id"],
            "tier": bundle["tier"],
            "camera_height_m": bundle["camera_height_m"],
            "fps": bundle["fps"],
            "n_frame_pairs": n,
            "median_motion_px": float(np.median(motions_px)),
            "p70_motion_px": float(np.percentile(motions_px, 70)),
            "p90_motion_px": float(np.percentile(motions_px, 90)),
            "rate_low": rates["low_5px"],
            "rate_medium": rates["medium_10px"],
            "rate_high": rates["high_20px"],
        })
        all_motions_px.extend(motions_px)
        print(
            f"  [{bundle['tier']:4s}] {bundle['rally_id'][:10]}  "
            f"cam={bundle['camera_height_m']:.2f}m fps={bundle['fps']:.0f}  "
            f"n={n}  med={np.median(motions_px):.1f}px  "
            f"p70={np.percentile(motions_px, 70):.1f}px  "
            f"p90={np.percentile(motions_px, 90):.1f}px  "
            f"≥5={rates['low_5px'] * 100:.0f}% "
            f"≥10={rates['medium_10px'] * 100:.0f}% "
            f"≥20={rates['high_20px'] * 100:.0f}%"
        )

    if not all_motions_px:
        print("No data — no rallies had usable ball GT frame pairs.")
        return

    arr = np.array(all_motions_px)
    overall_rates = {
        k: float((arr >= v).mean())
        for k, v in MOTION_THRESHOLDS.items()
    }

    print()
    print("=" * 60)
    print(f"TOTAL frame pairs: {len(arr)}")
    print(f"  median motion  : {float(np.median(arr)):.1f} px")
    print(f"  p70            : {float(np.percentile(arr, 70)):.1f} px")
    print(f"  p90            : {float(np.percentile(arr, 90)):.1f} px")
    print(f"  max            : {float(arr.max()):.1f} px")
    print()
    for k, v in MOTION_THRESHOLDS.items():
        r = overall_rates[k]
        print(f"  ≥{v:.0f} px/frame: {r * 100:.1f}%")
    print()
    print(
        f"Gate P1 (low ≥5 px/frame): {overall_rates['low_5px'] * 100:.1f}%  "
        f"gate ≥{GATE_PASS_RATE:.0%}  "
        f"{'PASS' if overall_rates['low_5px'] >= GATE_PASS_RATE else 'FAIL'}"
    )
    print()
    print("What this means for BlurBall feasibility:")
    print("  ≥5 px/frame  → at least ~1 px blur streak at 1/200s exposure")
    print("  ≥10 px/frame → at least ~2 px blur streak at 1/200s exposure")
    print("  ≥20 px/frame → at least ~4 px blur streak at 1/200s exposure")
    print("  (blur length = motion × exposure/frame_time ≈ 0.1–0.3 × motion)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Histogram.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(arr, bins=60, range=(0, 60), color="#1f77b4", alpha=0.8)
    for k, v in MOTION_THRESHOLDS.items():
        colour = {"low_5px": "red", "medium_10px": "orange", "high_20px": "purple"}[k]
        ax.axvline(v, color=colour, linewidth=1.5, label=f"{k} ({v:.0f} px)")
    ax.set_xlabel("ball image-space motion (px / frame)")
    ax.set_ylabel("frame pairs")
    ax.set_title(f"Ball image-space motion on ball-GT rallies (n={len(arr)})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    hist_path = OUTPUT_DIR / "probe_blur.png"
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {hist_path}")

    out_path = OUTPUT_DIR / "probe_blur.json"
    out_path.write_text(json.dumps({
        "gate_pass_rate": GATE_PASS_RATE,
        "motion_thresholds_px": MOTION_THRESHOLDS,
        "n_frame_pairs": len(arr),
        "median_motion_px": float(np.median(arr)),
        "p70_motion_px": float(np.percentile(arr, 70)),
        "p90_motion_px": float(np.percentile(arr, 90)),
        "max_motion_px": float(arr.max()),
        "overall_rates": overall_rates,
        "pass": overall_rates["low_5px"] >= GATE_PASS_RATE,
        "per_rally": per_rally,
    }, indent=2, default=float))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
