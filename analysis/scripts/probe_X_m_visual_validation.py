# ruff: noqa: UP031
"""Probe X-M: visual validation of M4 predictions vs user GT.

For a representative sample of labeled videos, render a frame with:
  - WHITE line  = user-set GT net top
  - ORANGE line = M4 prediction
  - GREEN line  = A0 baseline (median per-rally netY from contact_detector)

Saves to /tmp/net_top_validation_m4/ for visual inspection.

Sample is curated to cover:
  - top 5 A0-worst cases (where M4 rescues hardest):
      caca, hehe, michu, macho, kiki
  - 4 typical cases (A0 already close):
      titi, lulu, juju, gigi
  - 3 edge cases (60fps, indoor look, small/short rally count):
      yoyo (60fps long), mama, mech

The transitive argument is: GT is visually correct (user dragged the
handle to the visible net). LOO-CV says M4 ≈ GT within 0.008. So M4
should visually sit on the net top tape. This probe verifies that
chain.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
S3_BASE = "s3://rallycut-dev/videos/00000000-0000-0000-0000-000000000001"
OUT_DIR = Path("/tmp/net_top_validation_m4")

# M4 coefficients (ridge α=0.01, fit on all 77 samples, from probe X-L).
M4_INTERCEPT = -0.0736
M4_COEFS = {
    "midline_y":        +0.2983,
    "near_y":           +0.3083,
    "far_y":            +0.3966,
    "court_depth_y":    -0.0882,
    "midline_width_x":  -0.1879,
    "near_width_x":     -0.0708,
    "far_width_x":      -0.1226,
    "trapezoid_aspect": +0.0038,
}

SAMPLE_VIDEOS = [
    # A0-worst rescues (largest M4 wins)
    "caca", "hehe", "michu", "macho", "kiki",
    # Typical median cases
    "titi", "lulu", "juju", "gigi",
    # Edge cases
    "yoyo", "mama", "mech",
]


def _line_intersect(p1, p2, p3, p4):
    denom = (p1["x"] - p2["x"]) * (p3["y"] - p4["y"]) - \
            (p1["y"] - p2["y"]) * (p3["x"] - p4["x"])
    if abs(denom) < 1e-9:
        return None
    t = ((p1["x"] - p3["x"]) * (p3["y"] - p4["y"]) -
         (p1["y"] - p3["y"]) * (p3["x"] - p4["x"])) / denom
    return (p1["x"] + t * (p2["x"] - p1["x"]),
            p1["y"] + t * (p2["y"] - p1["y"]))


def compute_features(corners):
    near_y = (corners[0]["y"] + corners[1]["y"]) / 2
    far_y = (corners[2]["y"] + corners[3]["y"]) / 2
    near_width_x = abs(corners[1]["x"] - corners[0]["x"])
    far_width_x = abs(corners[2]["x"] - corners[3]["x"])
    court_depth_y = max(1e-6, near_y - far_y)
    diag = _line_intersect(corners[0], corners[2], corners[1], corners[3])
    midline_y = diag[1] if diag else (near_y + far_y) / 2
    midline_width_x = (near_width_x + far_width_x) / 2  # fallback proxy
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


def predict_m4(corners):
    f = compute_features(corners)
    y = M4_INTERCEPT + sum(M4_COEFS[k] * f[k] for k in M4_COEFS)
    return max(0.0, min(1.0, y))


def fetch_video(vid, vname):
    out = Path(f"/tmp/{vname}_for_verify.mp4")
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    for fname in (f"{vname}_proxy.mp4", f"{vname}.mp4"):
        src = f"{S3_BASE}/{vid}/{fname}"
        try:
            subprocess.run(
                ["aws", "s3", "cp", src, str(out),
                 "--endpoint-url", "http://localhost:9000"],
                env={"PATH": "/opt/homebrew/bin:/usr/bin:/bin",
                     "AWS_ACCESS_KEY_ID": "minioadmin",
                     "AWS_SECRET_ACCESS_KEY": "minioadmin"},
                check=True, capture_output=True,
            )
            return out
        except subprocess.CalledProcessError:
            continue
    return None


def annotate(frame_in, frame_out, gt_y, m4_y, a0_y, vname, n_rallies):
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

    def hline(y_norm, color, width, label):
        y_px = int(y_norm * h)
        # Backing stroke for contrast
        draw.rectangle([0, y_px - width - 1, w, y_px + width + 1], fill=(0, 0, 0, 140))
        draw.rectangle([0, y_px - width, w, y_px + width], fill=color)
        return y_px

    # GT first (background), then M4 (foreground)
    if gt_y is not None:
        gt_px = hline(gt_y, (255, 255, 255, 230), 1, f"GT {gt_y:.3f}")
        draw.text((10, gt_px - 22), f"GT  {gt_y:.3f}",
                  fill=(255, 255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0, 220))
    if a0_y is not None:
        a0_px = hline(a0_y, (100, 255, 100, 230), 1, f"A0 {a0_y:.3f}")
        delta = a0_y - (gt_y or 0)
        draw.text((10, a0_px + 6), f"A0  {a0_y:.3f}  Δ={delta:+.3f}",
                  fill=(100, 255, 100, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0, 220))
    if m4_y is not None:
        m4_px = hline(m4_y, (255, 165, 0, 240), 2, f"M4 {m4_y:.3f}")
        delta = m4_y - (gt_y or 0)
        draw.text((w - 280, m4_px - 22), f"M4  {m4_y:.3f}  Δ={delta:+.3f}",
                  fill=(255, 165, 0, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0, 220))

    # Title
    title = f"{vname}   ({n_rallies} rallies)"
    draw.rectangle([5, 5, 5 + 12 * len(title), 36], fill=(0, 0, 0, 200))
    draw.text((10, 8), title, fill=(255, 255, 255, 255), font=big)

    img.save(frame_out, "JPEG", quality=90)


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    placeholders = ",".join(["%s"] * len(SAMPLE_VIDEOS))
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            f"""
            SELECT v.id::text, v.name, COALESCE(v.fps, 30) AS fps,
                   v.court_calibration_net_top_y AS gt,
                   v.court_calibration_json
            FROM videos v
            WHERE v.name IN ({placeholders})
              AND v.court_calibration_net_top_y IS NOT NULL
            """,
            SAMPLE_VIDEOS,
        )
        videos = cur.fetchall()

        # Per-video, compute A0 (median per-rally netY) and pick a frame
        for vid, vname, fps, gt, corners_json in videos:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            m4_y = predict_m4(corners)

            # Pull rally with most ball positions for the cleanest frame
            cur2 = conn.execute(
                """
                SELECT r.start_ms, pt.contacts_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s AND pt.contacts_json IS NOT NULL
                ORDER BY r.start_ms
                LIMIT 1
                """,
                (vid,),
            )
            row = cur2.fetchone()
            if not row:
                print(f"  {vname}: no rally data, skipping", flush=True)
                continue
            start_ms, _ = row

            # A0 = median across rallies
            cur3 = conn.execute(
                """
                SELECT pt.contacts_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s AND pt.contacts_json IS NOT NULL
                """,
                (vid,),
            )
            net_ys = []
            for (cj,) in cur3:
                cj = cj if isinstance(cj, dict) else json.loads(cj or '{}')
                ny = cj.get("netY")
                if ny is not None and ny > 0:
                    net_ys.append(float(ny))
            a0_y = (sorted(net_ys)[len(net_ys) // 2]) if net_ys else None
            n_rallies = len(net_ys)

            video = fetch_video(vid, vname)
            if not video:
                print(f"  {vname}: video fetch failed", flush=True)
                continue
            time_sec = float(start_ms) / 1000.0 + 1.5
            raw = OUT_DIR / f"{vname}_raw.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-ss", f"{time_sec:.3f}", "-i", str(video),
                     "-frames:v", "1", str(raw)],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print(f"  {vname}: frame extract failed", flush=True)
                continue

            out_path = OUT_DIR / f"{vname}_validation.jpg"
            try:
                annotate(raw, out_path, gt, m4_y, a0_y, vname, n_rallies)
            except Exception as e:
                print(f"  {vname}: annotate failed: {e}", flush=True)
                continue

            print(
                f"  {vname:<8} fps={fps:>5.1f} gt={gt:.3f}  "
                f"M4={m4_y:.3f} (Δ={m4_y - gt:+.3f})  "
                f"A0={'%.3f' % a0_y if a0_y else 'n/a'}"
                f"{' (Δ=%+.3f)' % (a0_y - gt) if a0_y else ''}",
                flush=True,
            )

    print(f"\nAll validation frames written to {OUT_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
