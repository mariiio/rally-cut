"""Phase B probe P1 (reality check): measure actual ball blur streak length.

De-risks the Tier 2 commit by answering the open question from
probe_ball_motion_signal.py: motion between GT frames is plentiful
(84.7% at ≥5 px/frame), but does the actual *blur streak in individual
frames* match that upper bound, or does short exposure time collapse
the signal?

Approach
--------
For each dense-ball-GT position in every audit rally:

    1. Crop tightly (48×48 or 64×64) around the GT ball centre — the ball
       is GUARANTEED to be near the centre because GT is human-verified.
    2. Apply light background suppression: subtract the local median
       intensity so the ball (usually brighter than sand) stands out.
    3. Threshold the crop and find the connected component *containing*
       (or closest to) the centre pixel. This is the ball specifically,
       not the largest bright feature elsewhere in the crop.
    4. Fit a minimum-area rectangle to that blob and compute:
         major / minor axes (px), eccentricity, area
    5. A frame has "measurable blur" if:
         major ≥ 2 × nominal ball radius (20 px)  (i.e. blob is ≥2 ball-diams long)
         AND eccentricity ≥ 0.4  (i.e. not circular)

Gate
----
≥30% of mid-flight frames show measurable blur. Same threshold as the
plan's original P1 gate for BlurBall feasibility.

This is the probe v1 (probe_ball_blur.py) tried to run — but v1 used
WASB ball positions (which include false positives into trees/sky)
and picked "largest bright blob" which systematically matched background
rather than the ball. v2 fixes both: GT-anchored position + blob
selection constrained to the crop centre.

Usage
-----
    cd analysis
    uv run python scripts/probe_ball_blur_v2.py
    uv run python scripts/probe_ball_blur_v2.py --limit 5 --save-gallery
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
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

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

CROP_HALF = 32  # 64×64 crop around ball centre
BALL_RADIUS_PX = 10  # nominal at native resolution
BLUR_MAJOR_AXIS_THRESHOLD = 2.0 * BALL_RADIUS_PX  # ≥20 px
BLUR_ECCENTRICITY_THRESHOLD = 0.4

# Window around action contacts to exclude.
CONTACT_EXCLUSION = 5

# Gate: ≥30% of mid-flight frames with measurable blur.
P1_PASS_RATE = 0.30

# Max frames per rally for runtime budget.
MAX_FRAMES_PER_RALLY = 100


@dataclass
class BlurFrame:
    rally_id: str
    frame: int
    major: float
    minor: float
    ecc: float
    area: float
    has_blur: bool
    crop: np.ndarray | None = None
    mask: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Blob finding — centre-anchored
# ---------------------------------------------------------------------------


def find_ball_blob(gray: np.ndarray) -> np.ndarray | None:
    """Find the ball via flood fill from the GT-anchored crop centre.

    Approach (robust to patterned balls and noisy backgrounds):
      1. Estimate background level from the crop annulus (radius > ball diameter).
      2. Measure the intensity at the centre pixel; the ball is either much
         brighter or much darker than the background, so decide polarity.
      3. Starting from the centre pixel, flood-fill all pixels whose intensity
         is within a tolerance band of the centre-pixel intensity. This
         follows the ball's own texture rather than the brightest spots.
      4. Fill holes (ball panels can create internal dark patches).
      5. Return the resulting mask.
    """
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Centre intensity (median of the 5×5 centre patch to suppress pixel noise).
    centre_patch = gray[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
    centre_val = int(np.median(centre_patch))

    # Background level: mean intensity in the crop corners (away from centre).
    corner_size = 8
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    bg_val = int(np.mean([c.mean() for c in corners]))

    # Decide polarity.
    polarity = 1 if centre_val > bg_val else -1

    # If centre and background are within 10 intensity units, the ball is
    # not clearly distinguishable — skip.
    if abs(centre_val - bg_val) < 10:
        return None

    # Adaptive tolerance: half the contrast between centre and background.
    half_contrast = max(abs(centre_val - bg_val) // 2, 15)

    # Flood fill from the centre.
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = gray.copy()
    lo = (half_contrast, 0, 0, 0)
    hi = (half_contrast, 0, 0, 0)
    cv2.floodFill(
        flood, mask, (cx, cy), 255,
        loDiff=lo, upDiff=hi,
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE,
    )
    blob_mask = mask[1:-1, 1:-1]

    # Fill holes — ball panels/shadows create internal dark patches.
    kernel = np.ones((3, 3), np.uint8)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sanity checks on size. Max area ≈ 3× nominal ball area (π × 10² ≈ 314)
    # to exclude flood-fill leakage into large background regions. Real
    # motion-blurred balls stretch a ball-diameter ball into ~2-3 diameters
    # long, so max area ≈ 3 × 314 = 942. Use 1000 as a round cap.
    area = int((blob_mask > 0).sum())
    if area < 50 or area > 1000:
        return None

    # Verify the blob actually reached near the centre.
    ys, xs = np.where(blob_mask > 0)
    if len(xs) == 0:
        return None
    cx_blob = xs.mean()
    cy_blob = ys.mean()
    if math.hypot(cx_blob - cx, cy_blob - cy) > 12:
        return None

    return blob_mask


def measure(frame_bgr: np.ndarray, ball_x: float, ball_y: float) -> BlurFrame | None:
    h, w = frame_bgr.shape[:2]
    cx_px = int(ball_x * w)
    cy_px = int(ball_y * h)
    x0 = max(cx_px - CROP_HALF, 0)
    y0 = max(cy_px - CROP_HALF, 0)
    x1 = min(cx_px + CROP_HALF, w)
    y1 = min(cy_px + CROP_HALF, h)
    if x1 - x0 < CROP_HALF or y1 - y0 < CROP_HALF:
        return None  # near-edge crops

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    mask = find_ball_blob(gray)
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = contours[0]
    rect = cv2.minAreaRect(blob)
    (_, _), (rw, rh), _ = rect
    major = float(max(rw, rh))
    minor = float(min(rw, rh))
    if major < 1e-6:
        return None
    ecc = math.sqrt(max(0.0, 1.0 - (minor / major) ** 2))
    area = float(cv2.contourArea(blob))
    # Reject absurdly-long "streaks" that are almost certainly background
    # leakage rather than real ball blur (real blur at 10m depth, 20m/s,
    # 1/120s exposure is at most ~6 ball diameters = 120 px, but that's
    # rare; 60 px is a sensible practical cap).
    if major > 60:
        return None
    has_blur = major >= BLUR_MAJOR_AXIS_THRESHOLD and ecc >= BLUR_ECCENTRICITY_THRESHOLD
    return BlurFrame(
        rally_id="",  # filled by caller
        frame=0,
        major=major,
        minor=minor,
        ecc=ecc,
        area=area,
        has_blur=has_blur,
        crop=crop_bgr,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bundles() -> list[dict[str, Any]]:
    if not AUDIT_FILE.exists():
        print(f"ERROR: audit file not found at {AUDIT_FILE}")
        sys.exit(1)
    audit = json.loads(AUDIT_FILE.read_text())
    rally_ids = [r["rally_id"] for r in audit["audit_rallies"]]
    meta = {r["rally_id"]: r for r in audit["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, pt.ground_truth_json, pt.action_ground_truth_json,
                   v.content_hash, pt.fps
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
        m = meta[rid]
        gt = row[1]
        positions = gt.get("positions", []) if isinstance(gt, dict) else gt or []
        ball_gt = [
            (int(p["frameNumber"]), float(p["x"]), float(p["y"]))
            for p in positions
            if isinstance(p, dict) and (p.get("label") or "").lower() == "ball"
        ]
        ball_gt.sort(key=lambda r: r[0])
        if not ball_gt:
            continue
        bundles.append({
            "rally_id": rid,
            "tier": m["tier"],
            "camera_height_m": m.get("camera_height_m", 0.0),
            "ball_gt": ball_gt,
            "action_gt": row[2] or [],
            "content_hash": row[3],
            "fps": float(row[4] or 30.0),
        })
    return bundles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class VideoReader:
    def __init__(self) -> None:
        self.resolver = VideoResolver()
        self._cap: cv2.VideoCapture | None = None
        self._hash: str | None = None

    def get_frame(self, content_hash: str, frame_idx: int) -> np.ndarray | None:
        if self._hash != content_hash:
            if self._cap is not None:
                self._cap.release()
            path = self.resolver.get_cached_path(content_hash)
            if path is None:
                return None
            self._cap = cv2.VideoCapture(str(path))
            self._hash = content_hash
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_RALLY)
    parser.add_argument("--save-gallery", action="store_true")
    args = parser.parse_args()

    bundles = load_bundles()
    if args.limit:
        bundles = bundles[: args.limit]
    print(f"Loaded {len(bundles)} rallies with dense ball GT")

    reader = VideoReader()
    all_frames: list[BlurFrame] = []
    per_rally: list[dict[str, Any]] = []

    for i, bundle in enumerate(bundles, 1):
        if not bundle["content_hash"] or not reader.resolver.is_cached(bundle["content_hash"]):
            print(f"  [{i}/{len(bundles)}] {bundle['rally_id'][:10]}  SKIP — no cached video")
            continue

        contact_frames = {int(l["frame"]) for l in bundle["action_gt"] if "frame" in l}

        def in_contact(f: int) -> bool:
            return any(abs(f - c) <= CONTACT_EXCLUSION for c in contact_frames)

        mid_flight = [(f, x, y) for f, x, y in bundle["ball_gt"] if not in_contact(f)]
        if not mid_flight:
            continue
        if len(mid_flight) > args.max_frames:
            idxs = np.linspace(0, len(mid_flight) - 1, args.max_frames, dtype=int)
            mid_flight = [mid_flight[i] for i in idxs]

        rally_frames: list[BlurFrame] = []
        for f, x, y in mid_flight:
            frame = reader.get_frame(bundle["content_hash"], f)
            if frame is None:
                continue
            meas = measure(frame, x, y)
            if meas is None:
                continue
            meas.rally_id = bundle["rally_id"]
            meas.frame = f
            rally_frames.append(meas)

        all_frames.extend(rally_frames)
        n = len(rally_frames)
        n_blur = sum(1 for m in rally_frames if m.has_blur)
        rate = n_blur / n if n else 0.0
        if rally_frames:
            majors = [m.major for m in rally_frames]
            eccs = [m.ecc for m in rally_frames]
            print(
                f"  [{i}/{len(bundles)}] [{bundle['tier']:4s}] {bundle['rally_id'][:10]}  "
                f"fps={bundle['fps']:.0f}  n={n}  "
                f"blur={n_blur}/{n} ({rate * 100:.0f}%)  "
                f"major_med={np.median(majors):.1f}px  "
                f"ecc_med={np.median(eccs):.2f}"
            )
        per_rally.append({
            "rally_id": bundle["rally_id"],
            "tier": bundle["tier"],
            "camera_height_m": bundle["camera_height_m"],
            "fps": bundle["fps"],
            "n_frames": n,
            "n_with_blur": n_blur,
            "blur_rate": rate,
            "major_median": float(np.median([m.major for m in rally_frames])) if rally_frames else None,
            "major_p90": float(np.percentile([m.major for m in rally_frames], 90)) if rally_frames else None,
            "ecc_median": float(np.median([m.ecc for m in rally_frames])) if rally_frames else None,
            "ecc_p90": float(np.percentile([m.ecc for m in rally_frames], 90)) if rally_frames else None,
        })

    reader.close()

    total = len(all_frames)
    total_blur = sum(1 for m in all_frames if m.has_blur)
    overall_rate = total_blur / total if total else 0.0

    print()
    print("=" * 70)
    print(f"TOTAL mid-flight frames analysed: {total}")
    if total:
        majors = np.array([m.major for m in all_frames])
        eccs = np.array([m.ecc for m in all_frames])
        print(f"  with measurable blur: {total_blur}/{total} ({overall_rate * 100:.1f}%)")
        print(f"  major axis median: {float(np.median(majors)):.1f} px")
        print(f"  major axis p90   : {float(np.percentile(majors, 90)):.1f} px")
        print(f"  eccentricity median: {float(np.median(eccs)):.2f}")
        print(f"  eccentricity p90   : {float(np.percentile(eccs, 90)):.2f}")
    print()
    print(
        f"Gate P1 (reality check): ≥{P1_PASS_RATE:.0%} frames with "
        f"major≥{BLUR_MAJOR_AXIS_THRESHOLD:.0f}px AND ecc≥{BLUR_ECCENTRICITY_THRESHOLD} "
        f"→ {overall_rate * 100:.1f}%  "
        f"{'PASS' if overall_rate >= P1_PASS_RATE else 'FAIL'}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Histograms.
    if all_frames:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        majors = np.array([m.major for m in all_frames])
        eccs = np.array([m.ecc for m in all_frames])
        axes[0].hist(majors, bins=50, range=(0, 60), color="#1f77b4", alpha=0.8)
        axes[0].axvline(BLUR_MAJOR_AXIS_THRESHOLD, color="red", linewidth=1.5,
                        label=f"{BLUR_MAJOR_AXIS_THRESHOLD:.0f}px blur gate")
        axes[0].set_xlabel("major axis (px)")
        axes[0].set_ylabel("frames")
        axes[0].set_title(f"Ball blob major axis length (n={total})")
        axes[0].legend(fontsize=9)

        axes[1].hist(eccs, bins=40, range=(0, 1), color="#d62728", alpha=0.8)
        axes[1].axvline(BLUR_ECCENTRICITY_THRESHOLD, color="green", linewidth=1.5,
                        label=f"ecc ≥ {BLUR_ECCENTRICITY_THRESHOLD} gate")
        axes[1].set_xlabel("eccentricity")
        axes[1].set_ylabel("frames")
        axes[1].set_title("Ball blob eccentricity")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        hist_path = OUTPUT_DIR / "probe_blur_v2.png"
        plt.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {hist_path}")

    if args.save_gallery and all_frames:
        with_blur = [m for m in all_frames if m.has_blur]
        no_blur = [m for m in all_frames if not m.has_blur]
        samples = (with_blur[::max(len(with_blur) // 12, 1)][:12]
                   + no_blur[::max(len(no_blur) // 12, 1)][:12])
        if samples:
            cols = 6
            rows_n = (len(samples) + cols - 1) // cols
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2, rows_n * 2.3))
            axes_flat = [a for a in axes.flat]
            for ax, m in zip(axes_flat, samples):
                if m.crop is not None:
                    rgb = cv2.cvtColor(m.crop, cv2.COLOR_BGR2RGB)
                    ax.imshow(rgb)
                    # Overlay mask in red.
                    if m.mask is not None:
                        overlay = rgb.copy()
                        overlay[m.mask > 0] = [255, 0, 0]
                        ax.imshow(
                            np.where(m.mask[..., None] > 0, overlay, rgb),
                            alpha=0.7,
                        )
                ax.set_title(
                    f"{'BLUR' if m.has_blur else 'NO'}\n"
                    f"maj={m.major:.0f} ecc={m.ecc:.2f}",
                    fontsize=7,
                    color="red" if m.has_blur else "gray",
                )
                ax.axis("off")
            for ax in axes_flat[len(samples):]:
                ax.axis("off")
            plt.tight_layout()
            gpath = OUTPUT_DIR / "probe_blur_v2_gallery.png"
            plt.savefig(gpath, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {gpath}")

    out = OUTPUT_DIR / "probe_blur_v2.json"
    out.write_text(json.dumps({
        "gate_pass_rate": P1_PASS_RATE,
        "threshold_major_axis_px": BLUR_MAJOR_AXIS_THRESHOLD,
        "threshold_eccentricity": BLUR_ECCENTRICITY_THRESHOLD,
        "nominal_ball_radius_px": BALL_RADIUS_PX,
        "n_frames": total,
        "n_with_blur": total_blur,
        "blur_rate": overall_rate,
        "pass": overall_rate >= P1_PASS_RATE,
        "major_median_px": float(np.median([m.major for m in all_frames])) if all_frames else None,
        "major_p90_px": float(np.percentile([m.major for m in all_frames], 90)) if all_frames else None,
        "ecc_median": float(np.median([m.ecc for m in all_frames])) if all_frames else None,
        "ecc_p90": float(np.percentile([m.ecc for m in all_frames], 90)) if all_frames else None,
        "per_rally": per_rally,
    }, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
