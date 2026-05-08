"""Direct unit test of InstrumentedBotSort with synthetic detections.

Runs the wrapper through 5 frames with hand-crafted detections to surface
any exceptions hidden by player_tracker.py's try/except. If this prints
"OK 5/5 frames", the wrapper is correct and the integration bug is
elsewhere.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

import torch  # noqa: E402

from rallycut.tracking.botsort_instrumented import InstrumentedBotSort  # noqa: E402
from rallycut.tracking.player_tracker import _IdentityCMC  # noqa: E402


def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="forensic_unit_")
    os.environ["BOTSORT_FORENSIC_LOG_DIR"] = tmpdir
    os.environ["BOTSORT_FORENSIC_RALLY_TAG"] = "test/r01"

    print(f"Sidecar dir: {tmpdir}", flush=True)
    tracker = InstrumentedBotSort(
        reid_weights=Path("unused"),
        device=torch.device("cpu"),
        half=False,
        track_high_thresh=0.25,
        track_low_thresh=0.08,
        new_track_thresh=0.35,
        track_buffer=45,
        match_thresh=0.90,
        proximity_thresh=0.5,
        appearance_thresh=0.30,
        cmc_method="sof",
        frame_rate=30,
        fuse_first_associate=False,
        with_reid=False,
        min_hits=1,
    )
    tracker.with_reid = True
    tracker.cmc = _IdentityCMC()

    # Synthetic 4 detections per frame, slowly moving rightward.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    base_boxes = np.array([
        [100, 200, 200, 400, 0.9, 0],
        [300, 200, 400, 400, 0.9, 0],
        [600, 200, 700, 400, 0.9, 0],
        [900, 200, 1000, 400, 0.9, 0],
    ], dtype=np.float32)
    embs = np.random.RandomState(0).randn(4, 128).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    n_ok = 0
    for f in range(5):
        dets = base_boxes.copy()
        dets[:, 0:4] += f * 5  # shift right by 5px each frame
        try:
            out = tracker.update(dets, frame, embs)
            print(
                f"  frame {f}: ok, output shape={out.shape}",
                flush=True,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  frame {f}: FAILED {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            break
    tracker.close()

    sidecar = Path(tmpdir) / "test" / "r01.jsonl"
    if sidecar.exists():
        nlines = len(sidecar.read_text().splitlines())
        print(f"\nSidecar: {sidecar} → {nlines} lines (1 meta + {nlines - 1} frames)", flush=True)
    print(f"OK {n_ok}/5 frames", flush=True)


if __name__ == "__main__":
    main()
