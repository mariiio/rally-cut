"""Probe X-N2: visual catalog of tilt prevalence + solvePnP tilt direction.

For each of the 77 user-GT videos, render one representative frame
with three lines overlaid:

  GREEN  horizontal   = user-set GT scalar (`courtCalibrationNetTopY`)
  BLUE   horizontal   = M4 LOO prediction (ridge regression scalar)
  ORANGE tilted line  = `estimate_net_line()` solvePnP tilt
                        (top_left_xy → top_right_xy)

Each video is bucketed by |solvePnP tilt|:

  flat        <0.005  (negligible visible tilt)
  mild     0.005–0.015
  notable  0.015–0.030
  pronounced  >0.030

Output:

  analysis/reports/net_top_tilt_validation_2026_05_20/
    frames/<bucket>/<name>.jpg
    n2_catalog.md      (markdown index grouped by bucket, with file links)

This probe is read-only. It does NOT recompute keypoints; it relies on
the `estimate_net_line` per-video cache (populated by probe N1 or by
any prior run). Cache misses fall through to a fresh computation,
which is fine — just slower.

Decision rule the user reads from the catalog:

  * If "notable" + "pronounced" together contain ≤5 videos →
    tilt is rare; close as low-EV (no implementation).
  * If the orange (solvePnP) line clearly hugs the visible net better
    than green/blue on the tilted cases → C2 / C3 viable.
  * If the orange line is wrong on tilted cases (inverted, wandering)
    → solvePnP teacher signal unreliable; C1 (human GT) is the path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from rallycut.court.net_line_estimator import estimate_net_line  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _net_top_probe_fetch import build_minio_index, fetch_video  # noqa: E402

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
OUT_DIR = ANALYSIS_ROOT / "reports" / "net_top_tilt_validation_2026_05_20"
FRAMES_DIR = OUT_DIR / "frames"
VIDEO_CACHE_DIR = Path("/tmp/net_top_validation_videos")


BUCKET_THRESHOLDS = [
    ("flat",       0.000, 0.005),
    ("mild",       0.005, 0.015),
    ("notable",    0.015, 0.030),
    ("pronounced", 0.030, 1.000),
]


# ---------------------------------------------------------------------------
# M4 feature extraction (same as probe X-L / X-M / X-N)
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    vid: str
    name: str
    fps: float
    width: int
    height: int
    duration_ms: int
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
        A[0, 0] -= 0.01
        b = X_tr_b.T @ y_tr
        coefs = np.linalg.solve(A, b)
        X_te_b = np.hstack([[1.0], X[i]])
        p = float(X_te_b @ coefs)
        preds[samples[i].name] = max(0.0, min(1.0, p))
    return preds


# ---------------------------------------------------------------------------
# DB + video access (mirrors probe X-M / X-N)
# ---------------------------------------------------------------------------


def load_samples() -> list[Sample]:
    out: list[Sample] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id::text, v.name, COALESCE(v.fps, 30) AS fps,
                   COALESCE(v.width, 1920) AS width,
                   COALESCE(v.height, 1080) AS height,
                   COALESCE(v.duration_ms, 60000) AS duration_ms,
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
        for vid, vname, fps, width, height, duration_ms, gt, corners_json, proxy_key, original_key in cur:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            if len(corners) != 4:
                continue
            out.append(Sample(
                vid=vid,
                name=vname,
                fps=float(fps),
                width=int(width),
                height=int(height),
                duration_ms=int(duration_ms),
                gt=float(gt),
                corners=corners,
                features=compute_features(corners),
                proxy_s3_key=proxy_key,
                original_s3_key=original_key,
            ))
    return out


def bucket_of(tilt: float) -> str:
    a = abs(tilt)
    for name, lo, hi in BUCKET_THRESHOLDS:
        if lo <= a < hi:
            return name
    return "pronounced"


# ---------------------------------------------------------------------------
# Frame rendering — PIL pattern from probe X-M, extended with tilted line
# ---------------------------------------------------------------------------


def annotate(
    frame_in: Path,
    frame_out: Path,
    vname: str,
    bucket: str,
    gt_y: float | None,
    m4_y: float | None,
    top_l_xy: tuple[float, float] | None,
    top_r_xy: tuple[float, float] | None,
    tilt: float | None,
    conf: float | None,
    warnings: list[str],
) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(frame_in).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
        big = font

    # GREEN horizontal = user GT scalar
    if gt_y is not None:
        y_px = int(gt_y * h)
        draw.rectangle([0, y_px - 2, w, y_px + 2], fill=(0, 0, 0, 140))
        draw.rectangle([0, y_px - 1, w, y_px + 1], fill=(80, 255, 80, 240))
        draw.text(
            (10, y_px - 22), f"GT  {gt_y:.3f}",
            fill=(80, 255, 80, 255), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0, 220),
        )

    # BLUE horizontal = M4 LOO
    if m4_y is not None:
        y_px = int(m4_y * h)
        draw.rectangle([0, y_px - 2, w, y_px + 2], fill=(0, 0, 0, 140))
        draw.rectangle([0, y_px - 1, w, y_px + 1], fill=(110, 170, 255, 240))
        delta_label = f"  Δ={m4_y - gt_y:+.3f}" if gt_y is not None else ""
        draw.text(
            (10, y_px + 6), f"M4  {m4_y:.3f}{delta_label}",
            fill=(110, 170, 255, 255), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0, 220),
        )

    # ORANGE tilted line = solvePnP top_left → top_right
    if top_l_xy is not None and top_r_xy is not None:
        # Extrapolate to image edges so the line spans the full frame
        # (the endpoints are at the sideline corners' x, not the
        # image edges). Use linear extrapolation from (xL, yL)→(xR, yR).
        xL, yL = top_l_xy
        xR, yR = top_r_xy
        if abs(xR - xL) > 1e-6:
            slope = (yR - yL) / (xR - xL)
            y_at_0 = yL + slope * (0.0 - xL)
            y_at_1 = yL + slope * (1.0 - xL)
        else:
            y_at_0 = (yL + yR) / 2
            y_at_1 = (yL + yR) / 2
        x0_px, y0_px = 0, int(y_at_0 * h)
        x1_px, y1_px = w - 1, int(y_at_1 * h)
        draw.line([(x0_px, y0_px - 1), (x1_px, y1_px - 1)], fill=(0, 0, 0, 200), width=4)
        draw.line([(x0_px, y0_px), (x1_px, y1_px)], fill=(255, 165, 0, 250), width=2)
        # Endpoint markers
        for x, y in ((int(xL * w), int(yL * h)), (int(xR * w), int(yR * h))):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 165, 0, 255), width=2)
        label = f"NLE tilt={tilt:+.4f}  conf={conf:.2f}"
        if warnings:
            label += f"  [{','.join(warnings)}]"
        draw.text(
            (w - 12 * len(label) - 12, int((y_at_0 + y_at_1) / 2 * h) + 8),
            label,
            fill=(255, 165, 0, 255), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0, 220),
        )

    # Title bar
    title = f"{vname}  ({bucket})"
    draw.rectangle([5, 5, 5 + 14 * len(title), 36], fill=(0, 0, 0, 200))
    draw.text((10, 8), title, fill=(255, 255, 255, 255), font=big)

    img.save(frame_out, "JPEG", quality=88)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    for bucket_name, _, _ in BUCKET_THRESHOLDS:
        (FRAMES_DIR / bucket_name).mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    n = len(samples)
    print(f"Loaded {n} samples with GT", flush=True)

    print("Building MinIO index (one aws s3 ls --recursive)...", flush=True)
    minio_index = build_minio_index()
    print(f"  indexed {len(minio_index)} unique vids in MinIO", flush=True)

    print("Computing M4 LOO predictions...", flush=True)
    m4_preds = m4_loo_predictions(samples)

    # bucket_name → list of (sample, m4_y, tilt, |Δ|_mid, |Δ|_M4, conf, warnings, jpeg_path)
    by_bucket: dict[str, list[dict]] = {n: [] for n, _, _ in BUCKET_THRESHOLDS}
    failures: list[str] = []

    for i, s in enumerate(samples, start=1):
        m4_y = m4_preds[s.name]
        video_path = fetch_video(
            s.vid, s.name, minio_index, VIDEO_CACHE_DIR,
            db_proxy_key=s.proxy_s3_key,
            db_original_key=s.original_s3_key,
        )
        if not video_path:
            failures.append(f"{s.name}: video fetch failed")
            print(f"[{i:>2}/{n}] {s.name:<8} FETCH_FAIL", flush=True)
            continue

        nl = estimate_net_line(
            video_path,
            image_width=s.width,
            image_height=s.height,
            video_key=s.vid,
            use_cache=True,
        )

        top_l_xy = nl.top_left_xy if nl else None
        top_r_xy = nl.top_right_xy if nl else None
        tilt = (top_r_xy[1] - top_l_xy[1]) if (nl and top_l_xy and top_r_xy) else None
        conf = nl.confidence if nl else None
        warns = list(nl.warnings) if nl else []
        bucket = bucket_of(tilt) if tilt is not None else "flat"

        # Pick a representative frame near video midpoint
        time_sec = max(2.0, (s.duration_ms or 60_000) / 2_000.0)
        raw = FRAMES_DIR / bucket / f"{s.name}_raw.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{time_sec:.3f}", "-i", str(video_path),
                 "-frames:v", "1", str(raw)],
                check=True,
            )
        except subprocess.CalledProcessError:
            failures.append(f"{s.name}: frame extract failed")
            print(f"[{i:>2}/{n}] {s.name:<8} FRAME_FAIL", flush=True)
            continue

        out_jpg = FRAMES_DIR / bucket / f"{s.name}.jpg"
        try:
            annotate(
                raw, out_jpg, s.name, bucket,
                gt_y=s.gt, m4_y=m4_y,
                top_l_xy=top_l_xy, top_r_xy=top_r_xy,
                tilt=tilt, conf=conf, warnings=warns,
            )
        except Exception as e:
            failures.append(f"{s.name}: annotate failed: {e}")
            print(f"[{i:>2}/{n}] {s.name:<8} ANNOTATE_FAIL {e}", flush=True)
            continue
        raw.unlink(missing_ok=True)

        m4_delta = abs(m4_y - s.gt)
        nle_mid_delta = (
            abs((top_l_xy[1] + top_r_xy[1]) / 2 - s.gt)
            if (top_l_xy and top_r_xy) else None
        )

        by_bucket[bucket].append({
            "name": s.name,
            "fps": s.fps,
            "gt": s.gt,
            "m4_y": m4_y,
            "tilt": tilt,
            "m4_delta": m4_delta,
            "nle_mid_delta": nle_mid_delta,
            "conf": conf,
            "warnings": warns,
            "rel_jpg": Path("frames") / bucket / out_jpg.name,
        })

        tilt_str = f"{tilt:+.4f}" if tilt is not None else "n/a"
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        print(
            f"[{i:>2}/{n}] {s.name:<8} fps={s.fps:>5.1f}  "
            f"gt={s.gt:.3f}  M4={m4_y:.3f}  tilt={tilt_str}  "
            f"conf={conf_str}  bucket={bucket}",
            flush=True,
        )

    # Bucket counts
    print("\n=== Bucket counts ===", flush=True)
    for bucket_name, lo, hi in BUCKET_THRESHOLDS:
        items = by_bucket[bucket_name]
        print(f"  {bucket_name:<11} ({lo:.3f}–{hi:.3f}): {len(items)} videos", flush=True)
    if failures:
        print(f"\nFailures ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)

    # Markdown index
    md_path = OUT_DIR / "n2_catalog.md"
    with md_path.open("w") as f:
        f.write("# Probe N2 — Tilt prevalence + solvePnP visual catalog\n\n")
        f.write(f"Samples: {n} videos. Bucket thresholds by |solvePnP tilt|.\n\n")
        f.write("## Bucket counts\n\n")
        f.write("| bucket | |tilt| range | count |\n|--------|--------------|-------|\n")
        for bucket_name, lo, hi in BUCKET_THRESHOLDS:
            f.write(f"| {bucket_name} | {lo:.3f}–{hi:.3f} | {len(by_bucket[bucket_name])} |\n")
        f.write("\n## Decision rule\n\n")
        f.write("* `notable + pronounced` ≤ 5 → tilt is rare; close as low-EV.\n")
        f.write("* Orange line hugs the visible net on tilted cases → C2 / C3 viable.\n")
        f.write("* Orange line is wrong on tilted cases → solvePnP teacher unreliable; C1 only.\n\n")
        for bucket_name, _, _ in BUCKET_THRESHOLDS:
            items = by_bucket[bucket_name]
            if not items:
                continue
            f.write(f"\n## {bucket_name} ({len(items)})\n\n")
            f.write("| name | fps | gt | M4 | |Δ|M4 | NLE mid |Δ| | tilt | conf | warnings | frame |\n")
            f.write("|------|-----|----|----|------|---------------|------|------|----------|-------|\n")
            for it in sorted(items, key=lambda x: abs(x["tilt"] or 0.0), reverse=True):
                wstr = ",".join(it["warnings"]) if it["warnings"] else ""
                nle_str = f"{it['nle_mid_delta']:.4f}" if it["nle_mid_delta"] is not None else "n/a"
                tilt_str = f"{it['tilt']:+.4f}" if it["tilt"] is not None else "n/a"
                conf_str = f"{it['conf']:.2f}" if it["conf"] is not None else "n/a"
                f.write(
                    f"| {it['name']} | {it['fps']:.1f} | {it['gt']:.3f} | "
                    f"{it['m4_y']:.3f} | {it['m4_delta']:.4f} | {nle_str} | "
                    f"{tilt_str} | {conf_str} | {wstr} | "
                    f"![]({it['rel_jpg'].as_posix()}) |\n",
                )
    print(f"\nWrote catalog → {md_path}", flush=True)
    print(f"Frames under  → {FRAMES_DIR}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
