#!/usr/bin/env python3
"""Annotate court corners on images for court keypoint training.

Click 4 corners on each image: near-left, near-right, far-right, far-left.
The canvas extends below the image (30% padding) so you can place near corners
off-screen — just like the real dataset handles off-screen corners.

Controls:
  - Left click: place next corner
  - Right click / Backspace: undo last corner
  - S: skip image
  - Q: quit

Usage:
    uv run python scripts/annotate_court_corners.py images/
    uv run python scripts/annotate_court_corners.py images/ --output-dir datasets/court_keypoints_external
    uv run python scripts/annotate_court_corners.py images/ --pad-ratio 0.3
    uv run python scripts/annotate_court_corners.py images/ --resume  # skip already-labeled images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]
CORNER_COLORS = [
    (0, 0, 255),    # near-left: red
    (0, 165, 255),  # near-right: orange
    (0, 255, 0),    # far-right: green
    (255, 0, 0),    # far-left: blue
]
FLIP_IDX = [1, 0, 3, 2]


def corners_to_yolo_pose(
    corners: list[tuple[float, float]],
    orig_h: int,
    pad_ratio: float,
) -> str:
    """Convert 4 corner pixel positions to YOLO-pose format.

    corners are in padded-image pixel coordinates.
    """
    padded_h = int(orig_h * (1 + pad_ratio))
    scale_y = 1.0 / (1.0 + pad_ratio)

    # Assume we don't know width from corners alone — we need it passed in
    # Actually we get it from the image, so let's adjust signature
    raise NotImplementedError("Use corners_to_yolo_pose_norm instead")


def corners_to_yolo_pose_norm(
    corners_norm: list[tuple[float, float]],
    pad_ratio: float,
) -> str | None:
    """Convert 4 corners in normalized padded-image coords to YOLO-pose format.

    Args:
        corners_norm: [(x, y), ...] in [0,1] range of the padded image.
        pad_ratio: Bottom padding ratio.

    Returns:
        YOLO-pose format string.
    """
    scale_y = 1.0 / (1.0 + pad_ratio)

    keypoints = []
    for x, y in corners_norm:
        if 0 <= x <= 1 and 0 <= y <= scale_y:
            vis = 2  # visible in original image
        elif 0 <= x <= 1 and y <= 1.0:
            vis = 1  # in padding zone
        else:
            vis = 0  # off-canvas

        x_c = max(0.0, min(1.0, x))
        y_c = max(0.0, min(1.0, y))
        keypoints.append((x_c, y_c, vis))

    # Bounding box
    xs = [c[0] for c in corners_norm]
    ys = [c[1] for c in corners_norm]
    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = x_max - x_min
    bh = y_max - y_min

    if bw < 0.05 or bh < 0.05:
        return None

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y, v in keypoints:
        parts.append(f"{x:.6f} {y:.6f} {v}")

    return " ".join(parts)


class CornerAnnotator:
    """Interactive corner annotation with off-screen support."""

    # Horizontal padding as fraction of width (15% each side)
    H_PAD_RATIO = 0.15

    def __init__(self, img: np.ndarray, pad_ratio: float, title: str = ""):
        self.orig_img = img
        self.orig_h, self.orig_w = img.shape[:2]
        self.pad_ratio = pad_ratio
        self.title = title

        # Horizontal padding (left + right)
        self.pad_left = int(self.orig_w * self.H_PAD_RATIO)
        self.pad_right = self.pad_left
        # Vertical padding (bottom only)
        pad_bottom = int(self.orig_h * pad_ratio)

        self.canvas_w = self.pad_left + self.orig_w + self.pad_right
        self.canvas_h = self.orig_h + pad_bottom

        # Create canvas with gray padding everywhere
        self.canvas_base = np.full((self.canvas_h, self.canvas_w, 3), 40, dtype=np.uint8)
        # Place original image in center-top
        self.canvas_base[: self.orig_h, self.pad_left : self.pad_left + self.orig_w] = img

        # Draw boundary rectangle around original image area
        cv2.rectangle(
            self.canvas_base,
            (self.pad_left, 0),
            (self.pad_left + self.orig_w - 1, self.orig_h - 1),
            (0, 255, 255),
            1,
        )

        self.corners: list[tuple[int, int]] = []  # pixel coords in padded image
        self.done = False
        self.skipped = False
        self.quit = False

    def _draw(self) -> np.ndarray:
        canvas = self.canvas_base.copy()

        # Draw placed corners
        for i, (px, py) in enumerate(self.corners):
            color = CORNER_COLORS[i]
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                CORNER_NAMES[i],
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Draw lines between corners
        if len(self.corners) >= 2:
            for i in range(len(self.corners)):
                if i + 1 < len(self.corners):
                    cv2.line(canvas, self.corners[i], self.corners[i + 1], (200, 200, 200), 1)
            if len(self.corners) == 4:
                cv2.line(canvas, self.corners[3], self.corners[0], (200, 200, 200), 1)

        # Instructions
        if len(self.corners) < 4:
            next_corner = CORNER_NAMES[len(self.corners)]
            next_color = CORNER_COLORS[len(self.corners)]
            msg = f"Click: {next_corner} ({len(self.corners)+1}/4)"
            cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, next_color, 2)
        else:
            cv2.putText(
                canvas,
                "ENTER=save  BACKSPACE=undo  S=skip",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Title
        if self.title:
            cv2.putText(
                canvas,
                self.title,
                (10, self.canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                if len(self.corners) == 4:
                    cv2.imshow("Annotate", self._draw())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.corners:
                self.corners.pop()

    def run(self) -> list[tuple[float, float]] | None:
        """Run annotation UI. Returns normalized corners or None if skipped."""
        window = "Annotate"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # Scale window to reasonable size
        scale = min(1200 / self.canvas_w, 900 / self.canvas_h, 1.0)
        cv2.resizeWindow(window, int(self.canvas_w * scale), int(self.canvas_h * scale))

        cv2.setMouseCallback(window, self._on_mouse)

        while True:
            cv2.imshow(window, self._draw())
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                self.quit = True
                cv2.destroyAllWindows()
                return None
            elif key == ord("s"):
                self.skipped = True
                cv2.destroyAllWindows()
                return None
            elif key == 8 or key == 127:  # Backspace / Delete
                if self.corners:
                    self.corners.pop()
            elif key == 13 or key == 10:  # Enter
                if len(self.corners) == 4:
                    cv2.destroyAllWindows()
                    # Convert canvas pixels to padded-image normalized coords.
                    # x: subtract left padding, normalize by orig width
                    #    (can be <0 or >1 for off-screen)
                    # y: normalize by canvas height (orig + bottom pad)
                    #    (canvas has no top padding, so y=0 is top of image)
                    padded_h = self.orig_h + int(self.orig_h * self.pad_ratio)
                    return [
                        (
                            (px - self.pad_left) / self.orig_w,
                            py / padded_h,
                        )
                        for px, py in self.corners
                    ]

        return None


def pad_frame_bottom(frame: np.ndarray, pad_ratio: float) -> np.ndarray:
    """Add black padding to the bottom of a frame (for saving).

    Only bottom padding — matches the export pipeline format where the saved
    image has the same width as the original but extra height below.
    """
    h, w = frame.shape[:2]
    pad_h = int(h * pad_ratio)
    padding = np.zeros((pad_h, w, 3), dtype=frame.dtype)
    return np.vstack([frame, padding])


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate court corners on images")
    parser.add_argument("input_dir", type=Path, help="Directory with images to annotate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/court_keypoints_external"),
        help="Output directory for annotated dataset",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.3,
        help="Bottom padding ratio (0.3 = 30%% extra height for off-screen corners)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have labels",
    )
    args = parser.parse_args()

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in args.input_dir.rglob("*") if p.suffix.lower() in extensions
    )

    if not images:
        print(f"No images found in {args.input_dir}")
        return

    # Create output dirs
    img_dir = args.output_dir / "images" / "train"
    lbl_dir = args.output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Track what's already labeled
    existing = set()
    if args.resume:
        existing = {p.stem for p in lbl_dir.glob("*.txt")}
        if existing:
            print(f"Resuming: {len(existing)} images already labeled")

    # Annotations metadata
    meta_path = args.output_dir / "annotations.json"
    annotations: dict[str, list[dict[str, float]]] = {}
    if meta_path.exists():
        annotations = json.loads(meta_path.read_text())

    labeled = 0
    skipped = 0

    print(f"Found {len(images)} images in {args.input_dir}")
    print("Controls: click 4 corners, ENTER=save, BACKSPACE=undo, S=skip, Q=quit")
    print(f"Corner order: {', '.join(CORNER_NAMES)}")
    print(f"Yellow line = original image boundary (below = off-screen padding)\n")

    for i, img_path in enumerate(images):
        stem = img_path.stem

        if stem in existing:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i+1}/{len(images)}] {img_path.name}  SKIP (cannot read)")
            continue

        title = f"[{i+1}/{len(images)}] {img_path.name} | labeled={labeled} skipped={skipped}"
        annotator = CornerAnnotator(img, args.pad_ratio, title)
        corners_norm = annotator.run()

        if annotator.quit:
            print(f"\nQuit. Labeled {labeled} images.")
            break

        if corners_norm is None:
            skipped += 1
            print(f"  [{i+1}/{len(images)}] {img_path.name}  SKIPPED")
            continue

        # Convert to YOLO-pose label
        label = corners_to_yolo_pose_norm(corners_norm, args.pad_ratio)
        if label is None:
            print(f"  [{i+1}/{len(images)}] {img_path.name}  SKIP (degenerate bbox)")
            skipped += 1
            continue

        # Save padded image + label
        fname = f"ext_{stem}"
        padded = pad_frame_bottom(img, args.pad_ratio)
        cv2.imwrite(str(img_dir / f"{fname}.jpg"), padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_dir / f"{fname}.txt").write_text(label + "\n")

        # Save human-readable annotation
        annotations[img_path.name] = [
            {"x": x, "y": y} for x, y in corners_norm
        ]
        meta_path.write_text(json.dumps(annotations, indent=2) + "\n")

        labeled += 1
        vis_str = " ".join(
            f"{'VIS' if y <= 1/(1+args.pad_ratio) else 'PAD'}"
            for _, y in corners_norm
        )
        print(f"  [{i+1}/{len(images)}] {img_path.name}  SAVED  [{vis_str}]")

    cv2.destroyAllWindows()

    print(f"\nDone. Labeled: {labeled}, Skipped: {skipped}")
    print(f"Output: {args.output_dir}")

    if labeled > 0:
        print(f"\nTo merge into main dataset and retrain:")
        print(f"  1. Copy images+labels into datasets/court_keypoints/images/train/ and labels/train/")
        print(f"  2. Or re-export with: uv run python scripts/export_court_keypoint_dataset.py")
        print(f"  3. Retrain: uv run python scripts/train_court_keypoint_model.py")


if __name__ == "__main__":
    main()
