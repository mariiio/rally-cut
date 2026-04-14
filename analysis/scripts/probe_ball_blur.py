"""Phase B probe P1: ball motion-blur feasibility.

For each audit rally, fetches the cached video frame at every WASB ball
detection in the mid-flight segments (excluding ±5 frames around action
GT contacts), crops a 96×96 patch centred on the ball centroid, and
measures whether the ball appears as an elongated motion streak.

Method: Otsu threshold the crop, find the largest connected bright
blob, fit its minimum-area bounding rectangle, and report major/minor
axis lengths. A "measurable blur" is a blob whose major axis exceeds
2× the nominal ball diameter AND whose eccentricity is ≥0.4.

Why this matters
----------------
BlurBall (arXiv v3, Mar 2026) is the plan's strongest "new information"
candidate — it adds a per-frame velocity-direction signal that's
partially independent of the 2D ball centroid path. But its entire
advantage depends on the ball actually producing visible motion blur
on consumer footage. If most clips are high-shutter, the signal
vanishes and the channel is not viable for our data regardless of
how well the model is engineered.

Gate
----
≥30% of mid-flight frames show measurable blur:
    principal axis length ≥ 2 × nominal ball radius
    AND eccentricity ≥ 0.4

Output
------
    outputs/ball_3d_rig/probe_blur.json — summary + per-rally breakdown
    outputs/ball_3d_rig/probe_blur_gallery.png — sample crops with measurements

Usage
-----
    cd analysis
    uv run python scripts/probe_ball_blur.py
    uv run python scripts/probe_ball_blur.py --limit 5  # test mode
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.tracking.ball_tracker import BallPosition

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

# Crop size around ball centroid in pixels (at proxy resolution).
CROP_HALF = 48  # 96×96 crop

# Nominal ball radius in pixels (empirical: WASB balls span ~18-22 px at typical
# depths on 1920px frames, so ~9-11 px radius).
BALL_RADIUS_PX = 10

# Measurable blur thresholds.
BLUR_MAJOR_AXIS_THRESHOLD = 2.0 * BALL_RADIUS_PX  # ≥20 px major axis
BLUR_ECCENTRICITY_THRESHOLD = 0.4

# Window around action contacts to exclude (frames) — the ball is static/slow
# at contact moments and the probe should measure mid-flight blur only.
CONTACT_EXCLUSION = 5

# Gate.
P1_PASS_RATE = 0.30

# Maximum frames to sample per rally (uniform stride).
MAX_FRAMES_PER_RALLY = 80


@dataclass
class RallyBundle:
    rally_id: str
    video_id: str
    tier: str
    camera_height_m_meta: float
    ball_positions: list[BallPosition]
    action_gt: list[dict]
    s3_key: str | None
    content_hash: str | None


def _parse_ball_gt(gt_json: Any) -> list[BallPosition]:
    """Extract ball positions from Label Studio ground_truth_json.

    These are human-verified positions — far more reliable than WASB
    detections for the blur probe, where false positives in WASB produce
    sky/tree crops with no ball.
    """
    if not gt_json:
        return []
    positions = gt_json.get("positions", []) if isinstance(gt_json, dict) else gt_json
    out: list[BallPosition] = []
    for p in positions:
        if not isinstance(p, dict):
            continue
        if (p.get("label") or "").lower() != "ball":
            continue
        out.append(BallPosition(
            frame_number=int(p.get("frameNumber", 0)),
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            confidence=1.0,
        ))
    return sorted(out, key=lambda b: b.frame_number)


def load_bundles(audit_info: dict) -> list[RallyBundle]:
    rally_ids = [r["rally_id"] for r in audit_info["audit_rallies"]]
    meta_by_id = {r["rally_id"]: r for r in audit_info["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id,
                   pt.ground_truth_json,
                   pt.action_ground_truth_json,
                   v.s3_key, v.content_hash
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = ANY(%s)
              AND pt.ground_truth_json IS NOT NULL
        """, (rally_ids,))
        rows = cur.fetchall()

    bundles: list[RallyBundle] = []
    for row in rows:
        rid = str(row[0])
        meta = meta_by_id[rid]
        ball_gt = _parse_ball_gt(row[2])
        if not ball_gt:
            continue
        bundles.append(RallyBundle(
            rally_id=rid,
            video_id=str(row[1]),
            tier=meta["tier"],
            camera_height_m_meta=meta.get("camera_height_m", 0.0),
            ball_positions=ball_gt,
            action_gt=row[3] or [],
            s3_key=row[4],
            content_hash=row[5],
        ))
    return bundles


# ---------------------------------------------------------------------------
# Video + frame fetch
# ---------------------------------------------------------------------------


class VideoReader:
    def __init__(self) -> None:
        self.resolver = VideoResolver()
        self._cap: cv2.VideoCapture | None = None
        self._cap_hash: str | None = None

    def get_frame(
        self, content_hash: str, rally_start_ms: int, frame_within_rally: int,
    ) -> tuple[np.ndarray, int, int] | None:
        """Return (frame_bgr, width, height) at the absolute video frame.

        The rally's per-rally frame index is relative to the rally segment,
        but the cached videos are per-rally segments (not the full match),
        so we can seek directly to frame_within_rally.
        """
        if self._cap_hash != content_hash:
            if self._cap is not None:
                self._cap.release()
            cached = self.resolver.get_cached_path(content_hash)
            if cached is None:
                return None
            self._cap = cv2.VideoCapture(str(cached))
            self._cap_hash = content_hash
        if self._cap is None or not self._cap.isOpened():
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_within_rally)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        return frame, w, h

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Blur measurement
# ---------------------------------------------------------------------------


@dataclass
class BlurMeasurement:
    rally_id: str
    frame: int
    ball_x: float
    ball_y: float
    major_axis: float
    minor_axis: float
    eccentricity: float
    area_px: float
    has_blur: bool
    crop: np.ndarray | None = None


def measure_blur(
    frame: np.ndarray, ball_x: float, ball_y: float,
) -> BlurMeasurement | None:
    """Measure blur attributes of the ball in the frame.

    ball_x, ball_y are normalised (0-1) image coordinates.
    """
    h, w = frame.shape[:2]
    cx = int(ball_x * w)
    cy = int(ball_y * h)

    # Clip crop region to image bounds.
    x0 = max(cx - CROP_HALF, 0)
    y0 = max(cy - CROP_HALF, 0)
    x1 = min(cx + CROP_HALF, w)
    y1 = min(cy + CROP_HALF, h)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    crop_bgr = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Volleyball balls are typically bright (white/yellow) on a darker
    # sand background. Otsu threshold separates them.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Only keep the blob closest to the crop centre (the ball we care about).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    ch, cw = gray.shape
    center = (cw / 2, ch / 2)
    best = None
    best_d = float("inf")
    for c in contours:
        if cv2.contourArea(c) < 10:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        d = math.hypot(ccx - center[0], ccy - center[1])
        if d < best_d:
            best_d = d
            best = c
    if best is None:
        return None

    # Fit minimum area rectangle to get major/minor axis + orientation.
    rect = cv2.minAreaRect(best)
    (_, _), (rw, rh), _ = rect
    major = float(max(rw, rh))
    minor = float(min(rw, rh))
    if major < 1e-6:
        return None
    ecc = math.sqrt(max(0.0, 1.0 - (minor / major) ** 2))
    area = float(cv2.contourArea(best))

    has_blur = major >= BLUR_MAJOR_AXIS_THRESHOLD and ecc >= BLUR_ECCENTRICITY_THRESHOLD
    return BlurMeasurement(
        rally_id="",  # filled in by caller
        frame=0,  # filled in by caller
        ball_x=ball_x,
        ball_y=ball_y,
        major_axis=major,
        minor_axis=minor,
        eccentricity=ecc,
        area_px=area,
        has_blur=has_blur,
        crop=crop_bgr,
    )


# ---------------------------------------------------------------------------
# Per-rally probe
# ---------------------------------------------------------------------------


def probe_rally(
    bundle: RallyBundle, reader: VideoReader, max_frames: int,
) -> list[BlurMeasurement]:
    if not bundle.content_hash:
        return []

    # Exclude contact windows.
    contact_frames = {int(l["frame"]) for l in bundle.action_gt if "frame" in l}
    def in_contact_window(f: int) -> bool:
        return any(abs(f - c) <= CONTACT_EXCLUSION for c in contact_frames)

    mid_flight = [bp for bp in bundle.ball_positions if not in_contact_window(bp.frame_number)]
    if not mid_flight:
        return []

    # Uniform-stride sample.
    if len(mid_flight) > max_frames:
        idxs = np.linspace(0, len(mid_flight) - 1, max_frames, dtype=int)
        mid_flight = [mid_flight[i] for i in idxs]

    measurements: list[BlurMeasurement] = []
    for bp in mid_flight:
        result = reader.get_frame(bundle.content_hash, 0, bp.frame_number)
        if result is None:
            continue
        frame, _w, _h = result
        meas = measure_blur(frame, bp.x, bp.y)
        if meas is None:
            continue
        meas.rally_id = bundle.rally_id
        meas.frame = bp.frame_number
        measurements.append(meas)

    return measurements


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit rallies")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_RALLY)
    args = parser.parse_args()

    if not AUDIT_FILE.exists():
        print(f"ERROR: audit file not found at {AUDIT_FILE}")
        sys.exit(1)
    audit_info = json.loads(AUDIT_FILE.read_text())
    bundles = load_bundles(audit_info)
    if args.limit:
        bundles = bundles[: args.limit]
    print(f"Loaded {len(bundles)} audit rallies")

    reader = VideoReader()
    per_rally: list[dict[str, Any]] = []
    all_measurements: list[BlurMeasurement] = []

    for i, bundle in enumerate(bundles, 1):
        if not bundle.content_hash or not reader.resolver.is_cached(bundle.content_hash):
            print(f"  [{i}/{len(bundles)}] {bundle.rally_id[:10]}  SKIP — video not cached")
            continue

        measurements = probe_rally(bundle, reader, args.max_frames)
        n_frames = len(measurements)
        n_blur = sum(1 for m in measurements if m.has_blur)
        rate = n_blur / n_frames if n_frames else 0.0
        per_rally.append({
            "rally_id": bundle.rally_id,
            "tier": bundle.tier,
            "camera_height_m": bundle.camera_height_m_meta,
            "n_frames": n_frames,
            "n_with_blur": n_blur,
            "blur_rate": rate,
            "major_axes": [m.major_axis for m in measurements],
            "eccentricities": [m.eccentricity for m in measurements],
        })
        all_measurements.extend(measurements)
        print(
            f"  [{i}/{len(bundles)}] [{bundle.tier:4s}] {bundle.rally_id[:10]}  "
            f"frames={n_frames}  blur={n_blur}/{n_frames} ({rate * 100:.1f}%)"
        )

    reader.close()

    total_frames = len(all_measurements)
    total_blur = sum(1 for m in all_measurements if m.has_blur)
    overall_rate = total_blur / total_frames if total_frames else 0.0

    print()
    print("=" * 60)
    print(f"TOTAL mid-flight frames analysed: {total_frames}")
    print(f"  with measurable blur: {total_blur}/{total_frames} ({overall_rate * 100:.1f}%)")
    if all_measurements:
        majors = np.array([m.major_axis for m in all_measurements])
        eccs = np.array([m.eccentricity for m in all_measurements])
        print(f"  major axis: median={float(np.median(majors)):.1f} px  "
              f"p90={float(np.percentile(majors, 90)):.1f} px")
        print(f"  eccentricity: median={float(np.median(eccs)):.2f}  "
              f"p90={float(np.percentile(eccs, 90)):.2f}")
    print()
    print(
        f"Gate P1: ≥{P1_PASS_RATE:.0%} mid-flight frames show blur  "
        f"→ {overall_rate * 100:.1f}%  "
        f"{'PASS' if overall_rate >= P1_PASS_RATE else 'FAIL'}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary JSON.
    summary_path = OUTPUT_DIR / "probe_blur.json"
    summary_path.write_text(json.dumps({
        "gate_pass_rate": P1_PASS_RATE,
        "threshold_major_axis_px": BLUR_MAJOR_AXIS_THRESHOLD,
        "threshold_eccentricity": BLUR_ECCENTRICITY_THRESHOLD,
        "nominal_ball_radius_px": BALL_RADIUS_PX,
        "n_frames": total_frames,
        "n_with_blur": total_blur,
        "blur_rate": overall_rate,
        "pass": overall_rate >= P1_PASS_RATE,
        "per_rally": per_rally,
    }, indent=2, default=float))
    print(f"\nWrote {summary_path}")

    # Build a gallery of sample crops.
    if all_measurements:
        # Pick ~24 varied examples: some with blur, some without.
        with_blur = [m for m in all_measurements if m.has_blur]
        no_blur = [m for m in all_measurements if not m.has_blur]
        samples = (with_blur[::max(len(with_blur) // 12, 1)][:12]
                   + no_blur[::max(len(no_blur) // 12, 1)][:12])
        if samples:
            n = len(samples)
            cols = 6
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
            axes_flat = axes.flat if rows > 1 else [axes] if cols == 1 else axes
            for ax, m in zip(axes_flat, samples):
                if m.crop is not None:
                    rgb = cv2.cvtColor(m.crop, cv2.COLOR_BGR2RGB)
                    ax.imshow(rgb)
                ax.set_title(
                    f"{'BLUR' if m.has_blur else 'NO'}\n"
                    f"maj={m.major_axis:.0f}px ecc={m.eccentricity:.2f}",
                    fontsize=7,
                    color="red" if m.has_blur else "gray",
                )
                ax.axis("off")
            for ax in list(axes_flat)[len(samples):]:
                ax.axis("off")
            plt.tight_layout()
            gallery_path = OUTPUT_DIR / "probe_blur_gallery.png"
            plt.savefig(gallery_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {gallery_path}")


if __name__ == "__main__":
    main()
