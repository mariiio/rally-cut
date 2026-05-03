"""Probe whether OSNet (current ReID model) actually discriminates the
4 visually distinct players in 5c756c41.

For each rally, pull all 4 PIDs' representative crops and compute the
4×4 cosine-similarity matrix between them. ALSO compute the same matrix
using:
  (a) The fine-tuned ReID head (current production: 128-dim).
  (b) The raw OSNet backbone (512-dim, no projection head).
  (c) DINOv2 ViT-L/14 (1024-dim, frozen pretrained).

If (a) shows high similarity between obviously-different players, the
fine-tuned head is the bug. If (a) and (b) both fail but (c) succeeds,
swapping to DINOv2 will help. If all three fail, the crops themselves
are the problem (background dominance / occlusion).

Usage:
    uv run python scripts/probe_osnet_discrimination.py 5c756c41-1cc1-4486-a95c-97398912cfbe
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as functional  # noqa: N812

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)


def _crop_for_pid(
    cap: cv2.VideoCapture,
    rally_start_ms: int,
    fps: float,
    fw: int,
    fh: int,
    positions_for_pid: list[dict],
) -> np.ndarray | None:
    if not positions_for_pid:
        return None
    sorted_pos = sorted(positions_for_pid, key=lambda p: p.get("frameNumber", 0))
    target_idx = len(sorted_pos) // 2
    # Search for a confident, full-body crop around mid-rally.
    for offset in range(0, 30):
        for sign in (1, -1):
            i = target_idx + offset * sign
            if not (0 <= i < len(sorted_pos)):
                continue
            p = sorted_pos[i]
            cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
            w, h = float(p.get("width", 0)), float(p.get("height", 0))
            if w <= 0.02 or h <= 0.04:
                continue
            if (h / max(w, 1e-6)) < 1.4:
                continue
            if float(p.get("confidence", 0)) < 0.5:
                continue
            ms = rally_start_ms + (int(p.get("frameNumber", 0)) * 1000 / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            x1 = max(0, int((cx - w / 2) * fw))
            y1 = max(0, int((cy - h / 2) * fh))
            x2 = min(fw, int((cx + w / 2) * fw))
            y2 = min(fh, int((cy + h / 2) * fh))
            if x2 <= x1 or y2 <= y1:
                continue
            return frame[y1:y2, x1:x2]
    return None


def _cos_matrix(emb: np.ndarray) -> np.ndarray:
    """4x4 cosine similarity matrix. emb is (4, D) L2-normalized."""
    return emb @ emb.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--rally", type=int, default=2,
                   help="Rally index to probe (default: 2 = seed-rally-ish)")
    args = p.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    if args.rally >= len(rallies):
        sys.exit(f"rally index {args.rally} out of range ({len(rallies)} rallies)")
    rally = rallies[args.rally]
    rally_id = rally.rally_id

    video_path = get_video_path(args.video_id)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load positions
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT positions_json FROM player_tracks WHERE rally_id = %s",
                [rally_id],
            )
            row = cur.fetchone()
            pos = row[0] if row and row[0] else []
            if isinstance(pos, str):
                pos = json.loads(pos)

    by_pid: dict[int, list[dict]] = {}
    for q in pos:
        tid = q.get("trackId")
        if tid in (1, 2, 3, 4):
            by_pid.setdefault(tid, []).append(q)

    crops: dict[int, np.ndarray] = {}
    for pid in (1, 2, 3, 4):
        c = _crop_for_pid(cap, rally.start_ms, fps, fw, fh, by_pid.get(pid, []))
        if c is not None:
            crops[pid] = c
            print(f"  PID{pid}: crop shape = {c.shape}")
        else:
            print(f"  PID{pid}: NO CROP (skipping)")
    cap.release()

    if len(crops) < 2:
        sys.exit("need >= 2 PID crops for a similarity comparison")

    pids = sorted(crops.keys())
    crop_list = [crops[pid] for pid in pids]

    # ===== (a) Fine-tuned head (current production, 128-dim) =====
    from rallycut.tracking.reid_general import WEIGHTS_PATH, GeneralReIDModel
    if WEIGHTS_PATH.exists():
        print("\n=== (a) Fine-tuned OSNet + projection head (128-dim) ===")
        m = GeneralReIDModel(weights_path=WEIGHTS_PATH)
        emb = m.extract_embeddings(crop_list)
        cm = _cos_matrix(emb)
        print(f"    PIDs: {pids}")
        for i, pid_a in enumerate(pids):
            print(f"    PID{pid_a}: " + "  ".join(f"{cm[i, j]:+.3f}" for j in range(len(pids))))
        # Off-diagonal stats
        offdiag = [cm[i, j] for i in range(len(pids)) for j in range(len(pids)) if i != j]
        print(f"    OFF-DIAG (different-player similarity): "
              f"min={min(offdiag):.3f} mean={sum(offdiag)/len(offdiag):.3f} max={max(offdiag):.3f}")
        print("    Different players should have low similarity (< 0.5).")

    # ===== (b) Raw OSNet backbone, NO projection head (512-dim) =====
    print("\n=== (b) Raw OSNet backbone, no projection head (512-dim) ===")
    from rallycut.tracking.reid_general import _load_osnet_backbone

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone, embed_dim = _load_osnet_backbone(device)
    backbone.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensors = []
    for c in crop_list:
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 256))
        tensors.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)
    x = torch.stack(tensors).to(device)
    x = (x - mean) / std
    with torch.inference_mode():
        feats = backbone(x)
    feats = functional.normalize(feats, p=2, dim=1)
    emb_b = feats.detach().cpu().numpy().astype(np.float32)
    cm_b = _cos_matrix(emb_b)
    print(f"    PIDs: {pids}")
    for i, pid_a in enumerate(pids):
        print(f"    PID{pid_a}: " + "  ".join(f"{cm_b[i, j]:+.3f}" for j in range(len(pids))))
    offdiag_b = [cm_b[i, j] for i in range(len(pids)) for j in range(len(pids)) if i != j]
    print(f"    OFF-DIAG: min={min(offdiag_b):.3f} mean={sum(offdiag_b)/len(offdiag_b):.3f} max={max(offdiag_b):.3f}")

    # (c) DINOv2 alternative removed 2026-05-03; per
    # `dormant_flag_audit_2026_05_03.md` it benchmarked WORSE than
    # fine-tuned OSNet on this task (mean off-diagonal cosine 0.70
    # vs 0.05). The probe's two-backbone comparison is sufficient to
    # judge OSNet's projection-head behavior.
    print("\n=== INTERPRETATION ===")
    print("  Lower OFF-DIAG = better discrimination between different players.")
    print("  If (a) much higher than (b): the projection head is destroying signal.")
    print("  If both high: bbox/background is dominating the embeddings.")


if __name__ == "__main__":
    main()
