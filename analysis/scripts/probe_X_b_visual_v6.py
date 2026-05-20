"""Probe X-B visual v6: render v5 vs v6 at-net zones on the 6 NOT_AT_NET frames.

For each of the 6 NOT_AT_NET GT block contacts, overlay on the actual
frame:

  - RED line   = pipeline-estimated net_y (the geometric net top)
  - RED band   = v5 at-net zone (|Δ| < 0.08, symmetric)
  - GREEN band = v6 at-net zone (-0.15 ≤ Δ ≤ 0.08, asymmetric above)
  - YELLOW dot = actual ball position at the contact frame
  - Verdict label: v5_isAtNet / v6_isAtNet

The user can visually confirm that the green (v6) band covers the
above-net block contacts (gigi, moma, pupu, titi) while leaving the
two truly-misestimated cases (kiki, popo) still outside — those
need a separate net_y estimation fix.

Output: /tmp/net_verify_v6/<video>_<rally>_v6.jpg
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

# All 6 NOT_AT_NET block cases (same set X-E uses).
CASES = [
    ("gigi", "b8d333ae", 234, "b097dd2a-6953-4e0e-a603-5be3552f462e"),
    ("kiki", "0bc00f94", 435, "647d5b3c-0ded-478c-a295-113d634dbd82"),
    ("moma", "e1929103", 202, "1a5da176-8755-4e0d-8afd-ed1cab746fe3"),
    ("popo", "c1052008", 195, "b03b461b-b1c1-4f53-8cce-79c0afe8a049"),
    ("pupu", "e11fa028", 386, "6d2f646c-1551-4917-a26c-7707a48ba0e9"),
    ("titi", "21029e9f", 194, "2e984c43-cef6-4215-8d8e-50d892b510b9"),
]

S3_BASE = "s3://rallycut-dev/videos/00000000-0000-0000-0000-000000000001"
OUT_DIR = Path("/tmp/net_verify_v6")

# Zone widths (image-normalised, image-y down → negative delta = ABOVE net).
V5_NET_ZONE = 0.08
V6_ABOVE = 0.15
V6_BELOW = 0.08


def _v5_at_net(ball_y: float, net_y: float) -> bool:
    return abs(ball_y - net_y) < V5_NET_ZONE


def _v6_at_net(ball_y: float, net_y: float) -> bool:
    delta = ball_y - net_y
    return -V6_ABOVE <= delta <= V6_BELOW


def fetch_video(vid: str, vname: str) -> Path | None:
    out = Path(f"/tmp/{vname}_for_verify.mp4")
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    for fname in (f"{vname}_proxy.mp4", f"{vname}.mp4"):
        src = f"{S3_BASE}/{vid}/{fname}"
        try:
            subprocess.run(
                ["aws", "s3", "cp", src, str(out), "--endpoint-url", "http://localhost:9000"],
                env={"PATH": "/opt/homebrew/bin:/usr/bin:/bin",
                     "AWS_ACCESS_KEY_ID": "minioadmin",
                     "AWS_SECRET_ACCESS_KEY": "minioadmin"},
                check=True, capture_output=True,
            )
            return out
        except subprocess.CalledProcessError:
            continue
    return None


def annotate(
    frame_in: Path, frame_out: Path, pipe_net_y: float,
    ball_x: float, ball_y: float, vname: str, rally_prefix: str,
    gt_frame: int, v5: bool, v6: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(frame_in).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        big_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
        big_font = font

    net_y_px = int(pipe_net_y * h)
    v5_top = int((pipe_net_y - V5_NET_ZONE) * h)
    v5_bot = int((pipe_net_y + V5_NET_ZONE) * h)
    v6_top = int((pipe_net_y - V6_ABOVE) * h)
    v6_bot = int((pipe_net_y + V6_BELOW) * h)

    # GREEN v6 band (drawn first, behind v5).
    draw.rectangle([0, v6_top, w, v6_bot], fill=(0, 200, 0, 60))
    # RED v5 band.
    draw.rectangle([0, v5_top, w, v5_bot], fill=(255, 0, 0, 70))
    # Solid net line (white) on top.
    draw.rectangle([0, net_y_px - 2, w, net_y_px + 2], fill=(255, 255, 255, 230))
    draw.text((w - 220, net_y_px - 25), f"NET y={pipe_net_y:.3f}",
              fill=(255, 255, 255, 255), font=font)

    # YELLOW ball
    bx_px = int(ball_x * w)
    by_px = int(ball_y * h)
    r = 16
    draw.ellipse(
        [bx_px - r, by_px - r, bx_px + r, by_px + r],
        outline=(255, 255, 0, 255), width=3,
    )
    draw.text((bx_px + r + 5, by_px - 10), "BALL", fill=(255, 255, 0, 255), font=font)

    # Title
    title = f"{vname}/{rally_prefix}  GT block frame {gt_frame}"
    draw.rectangle([5, 5, 5 + 14 * len(title), 38], fill=(0, 0, 0, 200))
    draw.text((10, 8), title, fill=(255, 255, 255, 255), font=big_font)

    # Verdict block
    delta = ball_y - pipe_net_y
    lines = [
        f"ball.y - net_y = {delta:+.3f}",
        f"v5 (sym 0.08):       isAtNet={v5}",
        f"v6 (asym 0.15/0.08): isAtNet={v6}",
    ]
    box_w = 360
    box_h = 24 * len(lines) + 16
    draw.rectangle([5, h - box_h - 5, 5 + box_w, h - 5], fill=(0, 0, 0, 210))
    for i, line in enumerate(lines):
        colour = (255, 255, 255, 255)
        if "v6" in line:
            colour = (130, 255, 130, 255) if v6 else (255, 130, 130, 255)
        elif "v5" in line:
            colour = (255, 130, 130, 255)
        draw.text((12, h - box_h + 4 + i * 24), line, fill=colour, font=font)

    # Legend (right side)
    leg_x = w - 280
    leg_y = 50
    legend = [
        ((255, 0, 0, 200), "v5 band  (|Δ| < 0.08)"),
        ((0, 200, 0, 200), "v6 band  (-0.15 .. +0.08)"),
        ((255, 255, 255, 230), "estimated net top"),
        ((255, 255, 0, 255), "ball position at contact"),
    ]
    draw.rectangle([leg_x - 5, leg_y - 5, leg_x + 290, leg_y + 24 * len(legend) + 5],
                   fill=(0, 0, 0, 200))
    for i, (col, lbl) in enumerate(legend):
        draw.rectangle([leg_x, leg_y + i * 24 + 6, leg_x + 18, leg_y + i * 24 + 18],
                       fill=col)
        draw.text((leg_x + 24, leg_y + i * 24 + 4), lbl, fill=(255, 255, 255, 255), font=font)

    img.save(frame_out, "JPEG", quality=92)


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"v5 net_zone:    symmetric ±{V5_NET_ZONE}", flush=True)
    print(f"v6 net_zone:    asymmetric (-{V6_ABOVE} above, +{V6_BELOW} below)", flush=True)
    print()

    flipped = 0
    unchanged = 0
    with psycopg.connect(DB_DSN) as conn:
        for vname, rally_prefix, gt_frame, vid in CASES:
            cur = conn.execute(
                """
                SELECT pt.contacts_json, r.start_ms, COALESCE(v.fps, 30) AS fps
                FROM rallies r
                JOIN videos v ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.id::text LIKE %s || '%%'
                """,
                (vname, rally_prefix),
            )
            row = cur.fetchone()
            if not row:
                print(f"  {vname} {rally_prefix}: rally not found", flush=True)
                continue
            cj, start_ms, fps = row
            cj = cj if isinstance(cj, dict) else json.loads(cj or '{}')
            pipe_net_y = cj.get("netY", 0.5)
            # Find contact closest to GT frame
            contact = None
            best = 1000
            for c in cj.get("contacts", []):
                d = abs(c.get("frame", -10000) - gt_frame)
                if d < best:
                    best = d
                    contact = c
            if not contact:
                continue
            ball_x = contact["ballX"]
            ball_y = contact["ballY"]
            delta = ball_y - pipe_net_y
            v5 = _v5_at_net(ball_y, pipe_net_y)
            v6 = _v6_at_net(ball_y, pipe_net_y)
            status = (
                "✓ FIXED  (False → True)" if (not v5 and v6)
                else "still False" if (not v5 and not v6)
                else "unchanged True" if (v5 and v6)
                else "REGRESSION (True → False)"
            )
            if v6 and not v5:
                flipped += 1
            elif v5 == v6:
                unchanged += 1
            print(
                f"  {vname:<6} {rally_prefix:<10} gt_frame={gt_frame:>4} "
                f"ball=({ball_x:.3f},{ball_y:.3f}) net_y={pipe_net_y:.3f} "
                f"delta={delta:+.3f} | v5={str(v5):<5} v6={str(v6):<5} | {status}",
                flush=True,
            )

            video = fetch_video(vid, vname)
            if not video:
                print("    video fetch failed (annotation skipped)", flush=True)
                continue

            time_sec = float(start_ms) / 1000.0 + gt_frame / fps
            raw_frame = OUT_DIR / f"{vname}_raw.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-ss", f"{time_sec:.3f}", "-i", str(video),
                     "-frames:v", "1", str(raw_frame)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"    extract failed: {e}", flush=True)
                continue
            ann_frame = OUT_DIR / f"{vname}_{rally_prefix}_v6.jpg"
            try:
                annotate(raw_frame, ann_frame, pipe_net_y, ball_x, ball_y,
                         vname, rally_prefix, gt_frame, v5, v6)
                print(f"    → {ann_frame}", flush=True)
            except Exception as e:
                print(f"    annotate failed: {e}", flush=True)

    print()
    print(f"Summary: {flipped} flipped to True under v6, {unchanged} unchanged.", flush=True)
    print(f"All v6-annotated frames in {OUT_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
