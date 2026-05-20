"""Probe X-N: A/B `estimate_net_line` midpoint vs M4 (LOO) on the 77 user-GT videos.

Question: does the L/R midpoint of `estimate_net_line()`'s solvePnP
output match or beat the M4 ridge regression on the same per-video
scalar GT that was used to fit M4?

For each labeled video:
  * Compute M4 leave-one-out prediction (refit ridge on the other 76
    using the same feature pipeline as probe X-L). This is the
    unbiased M4 error — comparable to NLE which sees neither GT nor
    each other video.
  * Run `estimate_net_line(video, video_key=id, use_cache=True)`.
  * Extract `top_left_xy.y`, `top_right_xy.y`, midpoint, signed tilt.
  * Record M4 |Δ|, NLE |Δ|, and the winner.

Aggregates: median / mean / worst |Δ| for each estimator, threshold
counts (>0.01, >0.025, >0.05, >0.10), NLE failure breakdown (None,
sanity_failed, mirrored sides, low confidence).

Outputs:
  analysis/reports/net_top_tilt_validation_2026_05_20/n1_midpoint_ab.csv
  analysis/reports/net_top_tilt_validation_2026_05_20/n1_summary.md

Decision rule the user will read from the summary:
  - NLE_mid med |Δ| within +0.002 of M4_LOO med |Δ| → C2/C3 viable.
  - NLE_mid med |Δ| > M4_LOO med |Δ| + 0.005 → C2/C3 dead on midpoint.
  - >10 NLE failures → C2 needs robust M4 fallback.

The probe is read-only: no DB writes, no shipped-code edits.
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

# Repo-relative paths so the probe works from `cd analysis && uv run python ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from rallycut.court.net_line_estimator import estimate_net_line  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _net_top_probe_fetch import build_minio_index, fetch_video  # noqa: E402

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
OUT_DIR = ANALYSIS_ROOT / "reports" / "net_top_tilt_validation_2026_05_20"
VIDEO_CACHE_DIR = Path("/tmp/net_top_validation_videos")


# ---------------------------------------------------------------------------
# M4 feature extraction (mirrors probe_X_l + rallycut.court.net_top_regressor)
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    vid: str
    name: str
    fps: float
    width: int
    height: int
    gt: float
    corners: list[dict]
    features: dict[str, float]
    proxy_s3_key: str | None
    original_s3_key: str | None


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
    """For each sample, refit ridge on the other 76 and predict it."""
    n = len(samples)
    X = np.array([[s.features[k] for k in M4_FEATURE_NAMES] for s in samples])
    y = np.array([s.gt for s in samples])
    preds: dict[str, float] = {}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X[mask]
        y_tr = y[mask]
        X_tr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        d = X_tr_b.shape[1]
        A = X_tr_b.T @ X_tr_b + 0.01 * np.eye(d)
        A[0, 0] -= 0.01  # don't regularise bias
        b = X_tr_b.T @ y_tr
        coefs = np.linalg.solve(A, b)
        X_te_b = np.hstack([[1.0], X[i]])
        p = float(X_te_b @ coefs)
        preds[samples[i].name] = max(0.0, min(1.0, p))
    return preds


# ---------------------------------------------------------------------------
# DB + video access
# ---------------------------------------------------------------------------


def load_samples() -> list[Sample]:
    out: list[Sample] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id::text, v.name, COALESCE(v.fps, 30) AS fps,
                   COALESCE(v.width, 1920) AS width,
                   COALESCE(v.height, 1080) AS height,
                   v.court_calibration_net_top_y AS gt,
                   v.court_calibration_json,
                   v.proxy_s3_key,
                   v.original_s3_key
            FROM videos v
            WHERE v.court_calibration_net_top_y IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
            ORDER BY v.name, v.fps
            """,
        )
        for vid, vname, fps, width, height, gt, corners_json, proxy_key, original_key in cur:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            if len(corners) != 4:
                continue
            out.append(Sample(
                vid=vid,
                name=vname,
                fps=float(fps),
                width=int(width),
                height=int(height),
                gt=float(gt),
                corners=corners,
                features=compute_features(corners),
                proxy_s3_key=proxy_key,
                original_s3_key=original_key,
            ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summarize(name: str, errors: list[float]) -> str:
    abs_e = [abs(e) for e in errors]
    return (
        f"  {name:<28} n={len(errors):>3}  "
        f"med |Δ|={statistics.median(abs_e):.3f}  "
        f"mean |Δ|={statistics.mean(abs_e):.3f}  "
        f"worst |Δ|={max(abs_e):.3f}  "
        f"mean signed Δ={statistics.mean(errors):+.3f}  "
        f">0.025: {sum(1 for a in abs_e if a > 0.025):<3}  "
        f">0.05: {sum(1 for a in abs_e if a > 0.05):<3}  "
        f">0.10: {sum(1 for a in abs_e if a > 0.10)}"
    )


def _winner(m4_delta: float, nle_delta: float | None, tie_eps: float = 0.005) -> str:
    if nle_delta is None:
        return "NLE_FAIL"
    if abs(m4_delta - nle_delta) < tie_eps:
        return "tie"
    return "NLE" if nle_delta < m4_delta else "M4"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    n = len(samples)
    print(f"Loaded {n} samples (videos with corners + scalar net_top GT)", flush=True)

    print("Building MinIO index (one aws s3 ls --recursive)...", flush=True)
    minio_index = build_minio_index()
    print(f"  indexed {len(minio_index)} unique vids in MinIO", flush=True)

    print("Computing M4 LOO predictions (refit ridge per held-out video)...", flush=True)
    m4_preds = m4_loo_predictions(samples)

    csv_path = OUT_DIR / "n1_midpoint_ab.csv"
    rows: list[dict] = []

    m4_errors: list[float] = []
    nle_errors: list[float] = []  # only for videos where NLE succeeded
    failure_reasons: dict[str, int] = {
        "fetch_failed": 0,
        "nle_none": 0,
        "sanity_failed": 0,
        "low_conf": 0,
        "left_mirrored": 0,
        "right_mirrored": 0,
        "few_frames_aggregated": 0,
    }

    for i, s in enumerate(samples, start=1):
        m4_pred = m4_preds[s.name]
        m4_delta = m4_pred - s.gt
        m4_errors.append(m4_delta)

        video_path = fetch_video(
            s.vid, s.name, minio_index, VIDEO_CACHE_DIR,
            db_proxy_key=s.proxy_s3_key,
            db_original_key=s.original_s3_key,
        )
        if not video_path:
            failure_reasons["fetch_failed"] += 1
            print(f"[{i:>2}/{n}] {s.name:<8} M4 Δ={m4_delta:+.4f}  NLE=FETCH_FAIL", flush=True)
            rows.append({
                "name": s.name, "vid": s.vid, "fps": s.fps,
                "gt": s.gt, "m4_pred": m4_pred, "m4_delta": m4_delta,
                "nle_status": "fetch_failed",
                "nle_top_l_y": None, "nle_top_r_y": None,
                "nle_mid": None, "nle_tilt": None, "nle_delta": None,
                "nle_conf": None, "nle_warnings": None,
                "nle_left_src": None, "nle_right_src": None,
                "winner": "NLE_FAIL",
            })
            continue

        nl = estimate_net_line(
            video_path,
            image_width=s.width,
            image_height=s.height,
            video_key=s.vid,
            use_cache=True,
        )
        if nl is None:
            failure_reasons["nle_none"] += 1
            print(f"[{i:>2}/{n}] {s.name:<8} M4 Δ={m4_delta:+.4f}  NLE=NONE", flush=True)
            rows.append({
                "name": s.name, "vid": s.vid, "fps": s.fps,
                "gt": s.gt, "m4_pred": m4_pred, "m4_delta": m4_delta,
                "nle_status": "none",
                "nle_top_l_y": None, "nle_top_r_y": None,
                "nle_mid": None, "nle_tilt": None, "nle_delta": None,
                "nle_conf": None, "nle_warnings": None,
                "nle_left_src": None, "nle_right_src": None,
                "winner": "NLE_FAIL",
            })
            continue

        nle_top_l_y = nl.top_left_xy[1]
        nle_top_r_y = nl.top_right_xy[1]
        nle_mid = (nle_top_l_y + nle_top_r_y) / 2.0
        nle_tilt = nle_top_r_y - nle_top_l_y  # signed: positive = right side lower
        nle_delta = nle_mid - s.gt
        nle_errors.append(nle_delta)

        warns = ",".join(nl.warnings) if nl.warnings else ""
        for w in nl.warnings:
            failure_reasons[w] = failure_reasons.get(w, 0) + 1
        if nl.left_source == "mirrored":
            failure_reasons["left_mirrored"] += 1
        if nl.right_source == "mirrored":
            failure_reasons["right_mirrored"] += 1
        if nl.confidence < 0.5:
            failure_reasons["low_conf"] += 1

        winner = _winner(abs(m4_delta), abs(nle_delta))
        print(
            f"[{i:>2}/{n}] {s.name:<8} fps={s.fps:>5.1f}  "
            f"gt={s.gt:.3f}  "
            f"M4={m4_pred:.3f} Δ={m4_delta:+.4f}  "
            f"NLE_mid={nle_mid:.3f} Δ={nle_delta:+.4f}  "
            f"tilt={nle_tilt:+.4f}  conf={nl.confidence:.2f}  "
            f"{winner}{('  W:' + warns) if warns else ''}",
            flush=True,
        )

        rows.append({
            "name": s.name, "vid": s.vid, "fps": s.fps,
            "gt": s.gt, "m4_pred": m4_pred, "m4_delta": m4_delta,
            "nle_status": "ok",
            "nle_top_l_y": nle_top_l_y, "nle_top_r_y": nle_top_r_y,
            "nle_mid": nle_mid, "nle_tilt": nle_tilt, "nle_delta": nle_delta,
            "nle_conf": nl.confidence, "nle_warnings": warns,
            "nle_left_src": nl.left_source, "nle_right_src": nl.right_source,
            "winner": winner,
        })

    # CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-video table → {csv_path}", flush=True)

    # Aggregate report
    print("\n=== Aggregate comparison ===", flush=True)
    print(_summarize("M4 (LOO ridge)", m4_errors), flush=True)
    if nle_errors:
        print(_summarize("NLE midpoint", nle_errors), flush=True)
    print(
        f"\nNLE failures: total={len(samples) - len(nle_errors)} of {len(samples)}",
        flush=True,
    )
    for k, v in failure_reasons.items():
        if v > 0:
            print(f"  {k}: {v}", flush=True)

    # Winner breakdown (only over videos where both succeeded)
    winner_counts: dict[str, int] = {}
    for r in rows:
        winner_counts[r["winner"]] = winner_counts.get(r["winner"], 0) + 1
    print("\nWinner counts (paired):", flush=True)
    for k in sorted(winner_counts):
        print(f"  {k}: {winner_counts[k]}", flush=True)

    # Markdown summary
    md_path = OUT_DIR / "n1_summary.md"
    abs_m4 = [abs(e) for e in m4_errors]
    abs_nle = [abs(e) for e in nle_errors] if nle_errors else []
    with md_path.open("w") as f:
        f.write("# Probe N1 — `estimate_net_line` midpoint vs M4 (LOO)\n\n")
        f.write(f"Samples: {n} videos with corners + scalar net_top GT.\n\n")
        f.write("## Aggregate\n\n")
        f.write("| Estimator    | n   | med \\|Δ\\| | mean \\|Δ\\| | worst \\|Δ\\| | >0.025 | >0.05 | >0.10 |\n")
        f.write("|--------------|-----|-----------|------------|-------------|--------|-------|-------|\n")
        f.write(
            f"| M4 (LOO ridge) | {len(abs_m4)} | {statistics.median(abs_m4):.4f} | "
            f"{statistics.mean(abs_m4):.4f} | {max(abs_m4):.4f} | "
            f"{sum(1 for a in abs_m4 if a > 0.025)} | "
            f"{sum(1 for a in abs_m4 if a > 0.05)} | "
            f"{sum(1 for a in abs_m4 if a > 0.10)} |\n",
        )
        if abs_nle:
            f.write(
                f"| NLE midpoint | {len(abs_nle)} | {statistics.median(abs_nle):.4f} | "
                f"{statistics.mean(abs_nle):.4f} | {max(abs_nle):.4f} | "
                f"{sum(1 for a in abs_nle if a > 0.025)} | "
                f"{sum(1 for a in abs_nle if a > 0.05)} | "
                f"{sum(1 for a in abs_nle if a > 0.10)} |\n",
            )
        f.write("\n## NLE failures\n\n")
        if any(v > 0 for v in failure_reasons.values()):
            f.write("| reason | count |\n|--------|-------|\n")
            for k, v in failure_reasons.items():
                if v > 0:
                    f.write(f"| {k} | {v} |\n")
        else:
            f.write("None.\n")
        f.write("\n## Winner counts (per-video, ties within ±0.005)\n\n")
        f.write("| winner | count |\n|--------|-------|\n")
        for k in sorted(winner_counts):
            f.write(f"| {k} | {winner_counts[k]} |\n")
        f.write("\n## Decision rule\n\n")
        f.write("* If NLE midpoint med \\|Δ\\| is within +0.002 of M4 LOO med \\|Δ\\| → C2/C3 viable on midpoint quality.\n")
        f.write("* If NLE midpoint med \\|Δ\\| > M4 LOO med \\|Δ\\| + 0.005 → C2/C3 die on midpoint; C1 is the path.\n")
        f.write("* If >10 NLE failures (fetch/None/sanity) → C2 needs robust M4 fallback; C3 stays viable.\n")
    print(f"Wrote markdown summary → {md_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
