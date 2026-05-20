# ruff: noqa: N806
"""Probe X-O: A/B v9 keypoint reader vs v8 NLE midpoint vs M4 LOO.

Same protocol as probe N1 (`probe_X_n_net_line_midpoint_ab.py`) but
adds a fourth row for the v9 8-keypoint direct reader. The 77 user-GT
videos are the eval set the model was trained on (subject to train/val
split — videos in the val split give unbiased v9 numbers; train-split
videos are still useful as a sanity floor).

Decision gate (from `~/.claude/plans/net_top_v9_sota_8keypoint.md`):

  Pass: v9 midpoint median |Δ| ≤ 0.007 (vs v8's 0.009 / M4's 0.008)
        AND v9 worst |Δ| ≤ 0.050 (vs v8's 0.047)
        AND tilt direction agreement ≥ 90% on the visibly-tilted subset
            (|GT tilt| > 0.010, n=10 in the labeled corpus).

  Fail: report deltas, do NOT bump CONTACT_PIPELINE_VERSION, do NOT
        wire production callers. Investigate (label noise? imbalanced
        train split? need more frames per video?).

Outputs:
  analysis/reports/net_top_v9_validation_2026_05_20/o1_v9_ab.csv
  analysis/reports/net_top_v9_validation_2026_05_20/o1_summary.md

Read-only: no DB writes, no pipeline edits.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from rallycut.court.net_line_estimator import estimate_net_line  # noqa: E402
from rallycut.court.net_top_keypoint_reader import read_net_top  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _net_top_probe_fetch import build_minio_index, fetch_video  # noqa: E402

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
OUT_DIR = ANALYSIS_ROOT / "reports" / "net_top_v9_validation_2026_05_20"
VIDEO_CACHE_DIR = Path("/tmp/net_top_validation_videos")

# Train/val split from the dataset export (read from metadata.json so we
# don't hardcode video IDs that could drift).
DATASET_META = ANALYSIS_ROOT / "datasets" / "court_keypoints_v6" / "metadata.json"


@dataclass
class Sample:
    vid: str
    name: str
    fps: float
    width: int
    height: int
    gt_left: float
    gt_right: float
    gt_mid: float
    gt_tilt: float
    corners: list[dict]
    proxy_s3_key: str | None
    original_s3_key: str | None


# ---------------------------------------------------------------------------
# M4 LOO (reused from probe N1)
# ---------------------------------------------------------------------------

def _line_intersect(p1, p2, p3, p4):
    denom = (p1["x"] - p2["x"]) * (p3["y"] - p4["y"]) - \
            (p1["y"] - p2["y"]) * (p3["x"] - p4["x"])
    if abs(denom) < 1e-9:
        return None
    t = ((p1["x"] - p3["x"]) * (p3["y"] - p4["y"]) -
         (p1["y"] - p3["y"]) * (p3["x"] - p4["x"])) / denom
    return (p1["x"] + t * (p2["x"] - p1["x"]),
            p1["y"] + t * (p2["y"] - p1["y"]))


def compute_features(corners: list[dict]) -> dict[str, float]:
    near_y = (corners[0]["y"] + corners[1]["y"]) / 2
    far_y = (corners[2]["y"] + corners[3]["y"]) / 2
    near_width_x = abs(corners[1]["x"] - corners[0]["x"])
    far_width_x = abs(corners[2]["x"] - corners[3]["x"])
    court_depth_y = max(1e-6, near_y - far_y)
    diag = _line_intersect(corners[0], corners[2], corners[1], corners[3])
    midline_y = diag[1] if diag else (near_y + far_y) / 2
    midline_width_x = (near_width_x + far_width_x) / 2
    if diag:
        baseline_vp = _line_intersect(corners[0], corners[1], corners[3], corners[2])
        if baseline_vp:
            diag_pt = {"x": diag[0], "y": diag[1]}
            vp_pt = {"x": baseline_vp[0], "y": baseline_vp[1]}
            left = _line_intersect(diag_pt, vp_pt, corners[3], corners[0])
            right = _line_intersect(diag_pt, vp_pt, corners[2], corners[1])
            if left and right:
                midline_width_x = abs(right[0] - left[0])
    return {
        "midline_y": midline_y,
        "near_y": near_y,
        "far_y": far_y,
        "court_depth_y": court_depth_y,
        "midline_width_x": midline_width_x,
        "near_width_x": near_width_x,
        "far_width_x": far_width_x,
        "trapezoid_aspect": ((near_width_x + far_width_x) / 2) / court_depth_y,
    }


M4_FEATURE_NAMES = [
    "midline_y", "near_y", "far_y", "court_depth_y",
    "midline_width_x", "near_width_x", "far_width_x", "trapezoid_aspect",
]


def m4_loo_predictions(samples: list[Sample]) -> dict[str, float]:
    n = len(samples)
    feats = [compute_features(s.corners) for s in samples]
    X = np.array([[f[k] for k in M4_FEATURE_NAMES] for f in feats])
    y = np.array([s.gt_mid for s in samples])
    preds: dict[str, float] = {}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X[mask]
        y_tr = y[mask]
        X_tr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        d = X_tr_b.shape[1]
        A = X_tr_b.T @ X_tr_b + 0.01 * np.eye(d)
        A[0, 0] -= 0.01
        b = X_tr_b.T @ y_tr
        coefs = np.linalg.solve(A, b)
        X_te_b = np.hstack([[1.0], X[i]])
        p = float(X_te_b @ coefs)
        preds[samples[i].vid] = max(0.0, min(1.0, p))
    return preds


# ---------------------------------------------------------------------------
# DB load + reporting
# ---------------------------------------------------------------------------

def load_samples() -> list[Sample]:
    out: list[Sample] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id::text, v.name, COALESCE(v.fps, 30),
                   COALESCE(v.width, 1920), COALESCE(v.height, 1080),
                   v.court_calibration_net_top_left_y,
                   v.court_calibration_net_top_right_y,
                   v.court_calibration_json,
                   v.proxy_s3_key, v.original_s3_key
            FROM videos v
            WHERE v.court_calibration_net_top_left_y IS NOT NULL
              AND v.court_calibration_net_top_right_y IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
              AND v.deleted_at IS NULL
            ORDER BY v.name, v.fps
            """,
        )
        for vid, vname, fps, w, h, gt_l, gt_r, corners_json, p_key, o_key in cur:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            if len(corners) != 4:
                continue
            out.append(Sample(
                vid=vid, name=vname, fps=float(fps),
                width=int(w), height=int(h),
                gt_left=float(gt_l), gt_right=float(gt_r),
                gt_mid=(float(gt_l) + float(gt_r)) / 2.0,
                gt_tilt=float(gt_r) - float(gt_l),
                corners=corners,
                proxy_s3_key=p_key, original_s3_key=o_key,
            ))
    return out


def _summarize(name: str, errors: list[float]) -> str:
    if not errors:
        return f"  {name:<24} n=0  (no successful runs)"
    abs_e = [abs(e) for e in errors]
    return (
        f"  {name:<24} n={len(errors):>3}  "
        f"med |Δ|={statistics.median(abs_e):.4f}  "
        f"mean |Δ|={statistics.mean(abs_e):.4f}  "
        f"worst |Δ|={max(abs_e):.4f}  "
        f">0.025: {sum(1 for a in abs_e if a > 0.025):<3}  "
        f">0.05: {sum(1 for a in abs_e if a > 0.05):<3}  "
        f">0.10: {sum(1 for a in abs_e if a > 0.10)}"
    )


def _tilt_dir_match(gt_tilt: float, pred_tilt: float, eps: float = 0.003) -> bool:
    """Same-sign tilt within tolerance. Treat |x| < eps as 'flat' (matches anything)."""
    if abs(gt_tilt) < eps:
        return True
    if abs(pred_tilt) < eps:
        return abs(gt_tilt) < eps
    return (gt_tilt > 0) == (pred_tilt > 0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    n = len(samples)
    print(f"Loaded {n} samples (videos with v9 net-top GT)", flush=True)

    val_ids: set[str] = set()
    if DATASET_META.exists():
        meta = json.loads(DATASET_META.read_text())
        val_ids = set(meta.get("val_video_ids", []))
        print(f"  val split: {len(val_ids)} videos (rest are train)", flush=True)
    else:
        print("  no metadata.json — eval treats all samples as a single set", flush=True)

    print("Building MinIO index...", flush=True)
    minio_index = build_minio_index()
    print(f"  indexed {len(minio_index)} unique vids in MinIO", flush=True)

    print("Computing M4 LOO predictions...", flush=True)
    m4_preds = m4_loo_predictions(samples)

    rows: list[dict] = []
    m4_errors: list[float] = []
    nle_errors: list[float] = []
    v9_errors: list[float] = []
    v9_tilt_errors: list[float] = []  # for visibly-tilted subset

    val_m4: list[float] = []
    val_nle: list[float] = []
    val_v9: list[float] = []
    val_v9_tilt_matches: list[bool] = []

    for i, s in enumerate(samples, start=1):
        video_path = fetch_video(
            s.vid, s.name, minio_index, VIDEO_CACHE_DIR,
            db_proxy_key=s.proxy_s3_key,
            db_original_key=s.original_s3_key,
        )
        if not video_path:
            print(f"[{i:>2}/{n}] {s.name:<8} FETCH_FAIL", flush=True)
            rows.append({
                "vid": s.vid, "name": s.name, "fps": s.fps,
                "in_val_split": s.vid in val_ids,
                "gt_left": s.gt_left, "gt_right": s.gt_right,
                "gt_mid": s.gt_mid, "gt_tilt": s.gt_tilt,
                "m4_pred": m4_preds[s.vid],
                "m4_delta": m4_preds[s.vid] - s.gt_mid,
                "status": "fetch_failed",
            })
            continue

        m4_pred = m4_preds[s.vid]
        m4_delta = m4_pred - s.gt_mid
        m4_errors.append(m4_delta)

        # v8 NLE midpoint
        nl = estimate_net_line(
            video_path, image_width=s.width, image_height=s.height,
            video_key=s.vid, use_cache=True,
        )
        nle_mid: float | None = None
        nle_delta: float | None = None
        if nl is not None and "sanity_failed" not in nl.warnings:
            nle_mid = (nl.top_left_xy[1] + nl.top_right_xy[1]) / 2.0
            nle_delta = nle_mid - s.gt_mid
            nle_errors.append(nle_delta)

        # v9 keypoint reader
        ntl = read_net_top(
            video_path, video_key=s.vid, use_cache=True,
        )
        v9_left_y: float | None = None
        v9_right_y: float | None = None
        v9_mid: float | None = None
        v9_delta: float | None = None
        v9_tilt: float | None = None
        v9_tilt_match: bool | None = None
        if ntl is not None and "sanity_failed" not in ntl.warnings:
            v9_left_y = ntl.top_left_xy[1]
            v9_right_y = ntl.top_right_xy[1]
            v9_mid = (v9_left_y + v9_right_y) / 2.0
            v9_delta = v9_mid - s.gt_mid
            v9_tilt = v9_right_y - v9_left_y
            v9_errors.append(v9_delta)
            if abs(s.gt_tilt) > 0.010:
                v9_tilt_errors.append(v9_delta)
                v9_tilt_match = _tilt_dir_match(s.gt_tilt, v9_tilt)

        if s.vid in val_ids:
            val_m4.append(m4_delta)
            if nle_delta is not None:
                val_nle.append(nle_delta)
            if v9_delta is not None:
                val_v9.append(v9_delta)
            if v9_tilt_match is not None:
                val_v9_tilt_matches.append(v9_tilt_match)

        # Per-video log
        nle_str = f"NLE={nle_mid:.3f} Δ={nle_delta:+.4f}" if nle_delta is not None else "NLE=FAIL"
        v9_str = (
            f"v9={v9_mid:.3f} Δ={v9_delta:+.4f} tilt={v9_tilt:+.4f}"
            if v9_delta is not None else "v9=FAIL"
        )
        split_tag = "VAL " if s.vid in val_ids else "trn "
        print(
            f"[{i:>2}/{n}] {split_tag}{s.name:<8} fps={s.fps:>5.1f}  "
            f"gt_mid={s.gt_mid:.3f} tilt={s.gt_tilt:+.4f}  "
            f"M4={m4_pred:.3f} Δ={m4_delta:+.4f}  {nle_str}  {v9_str}",
            flush=True,
        )

        rows.append({
            "vid": s.vid, "name": s.name, "fps": s.fps,
            "in_val_split": s.vid in val_ids,
            "gt_left": s.gt_left, "gt_right": s.gt_right,
            "gt_mid": s.gt_mid, "gt_tilt": s.gt_tilt,
            "m4_pred": m4_pred, "m4_delta": m4_delta,
            "nle_mid": nle_mid, "nle_delta": nle_delta,
            "v9_left": v9_left_y, "v9_right": v9_right_y,
            "v9_mid": v9_mid, "v9_delta": v9_delta, "v9_tilt": v9_tilt,
            "v9_tilt_dir_match": v9_tilt_match,
            "status": "ok",
        })

    csv_path = OUT_DIR / "o1_v9_ab.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-video table → {csv_path}", flush=True)

    # Aggregate (all 77)
    print("\n=== Aggregate (all videos) ===", flush=True)
    print(_summarize("M4 (LOO ridge)", m4_errors), flush=True)
    print(_summarize("NLE midpoint (v8)", nle_errors), flush=True)
    print(_summarize("v9 8-kpt midpoint", v9_errors), flush=True)

    # Val-only aggregate (unbiased)
    if val_ids:
        print("\n=== Val-only (unbiased — model didn't see these) ===", flush=True)
        print(_summarize("M4 LOO (val)", val_m4), flush=True)
        print(_summarize("NLE midpoint (val)", val_nle), flush=True)
        print(_summarize("v9 8-kpt (val)", val_v9), flush=True)

    # Tilt direction agreement (visibly-tilted subset)
    tilted_rows = [r for r in rows if r.get("v9_tilt_dir_match") is not None]
    if tilted_rows:
        matched = sum(1 for r in tilted_rows if r["v9_tilt_dir_match"])
        pct = 100.0 * matched / len(tilted_rows)
        print(
            f"\nTilt direction agreement on visibly-tilted subset "
            f"(|gt_tilt|>0.010, n={len(tilted_rows)}): {matched}/{len(tilted_rows)} "
            f"= {pct:.1f}%",
            flush=True,
        )
    else:
        print("\nNo visibly-tilted videos with v9 output — tilt gate not assessable", flush=True)

    # Decision gate
    print("\n=== Decision gate ===", flush=True)
    gate_pass = True
    reasons: list[str] = []
    if v9_errors:
        med_v9 = statistics.median([abs(e) for e in v9_errors])
        worst_v9 = max(abs(e) for e in v9_errors)
        if med_v9 > 0.007:
            gate_pass = False
            reasons.append(f"med v9 |Δ|={med_v9:.4f} > 0.007")
        if worst_v9 > 0.050:
            gate_pass = False
            reasons.append(f"worst v9 |Δ|={worst_v9:.4f} > 0.050")
    else:
        gate_pass = False
        reasons.append("v9 produced no usable outputs")
    if tilted_rows:
        pct = 100.0 * sum(1 for r in tilted_rows if r["v9_tilt_dir_match"]) / len(tilted_rows)
        if pct < 90.0:
            gate_pass = False
            reasons.append(f"tilt direction agreement {pct:.1f}% < 90%")
    verdict = "PASS — ship v9" if gate_pass else "FAIL — investigate"
    print(f"  Verdict: {verdict}", flush=True)
    for r in reasons:
        print(f"    {r}", flush=True)

    # Markdown summary
    md_path = OUT_DIR / "o1_summary.md"
    abs_m4 = [abs(e) for e in m4_errors]
    abs_nle = [abs(e) for e in nle_errors]
    abs_v9 = [abs(e) for e in v9_errors]
    with md_path.open("w") as f:
        f.write("# Probe X-O — v9 keypoint reader vs v8 NLE vs M4\n\n")
        f.write(f"Samples: {n} videos with v9 L/R net-top GT.\n\n")
        f.write("## Aggregate (all 77)\n\n")
        f.write("| Estimator | n | med \\|Δ\\| | mean \\|Δ\\| | worst | >0.025 | >0.05 |\n")
        f.write("|-----------|---|-----------|------------|-------|--------|-------|\n")
        for label, errs in (("M4 LOO", abs_m4), ("NLE v8 midpoint", abs_nle),
                            ("v9 8-kpt midpoint", abs_v9)):
            if not errs:
                f.write(f"| {label} | 0 | n/a | n/a | n/a | n/a | n/a |\n")
                continue
            f.write(
                f"| {label} | {len(errs)} | {statistics.median(errs):.4f} | "
                f"{statistics.mean(errs):.4f} | {max(errs):.4f} | "
                f"{sum(1 for a in errs if a > 0.025)} | "
                f"{sum(1 for a in errs if a > 0.05)} |\n"
            )
        if tilted_rows:
            matched = sum(1 for r in tilted_rows if r["v9_tilt_dir_match"])
            f.write(
                f"\n## Tilt direction agreement (visibly-tilted, n={len(tilted_rows)})\n\n"
                f"{matched}/{len(tilted_rows)} = "
                f"{100.0*matched/len(tilted_rows):.1f}%\n"
            )
        f.write(f"\n## Gate verdict: **{verdict}**\n")
        for r in reasons:
            f.write(f"- {r}\n")
    print(f"Wrote markdown summary → {md_path}", flush=True)

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
