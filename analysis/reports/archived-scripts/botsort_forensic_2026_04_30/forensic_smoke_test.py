"""Quick smoke test: capture forensic sidecar for a single panel rally,
then read the first 3 frame records and report shape integrity.

Run from analysis/:
  uv run python scripts/forensic_smoke_test.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)
# Show debug from the rallycut tracker chain only, not the world.
logging.getLogger("rallycut.tracking").setLevel(logging.DEBUG)

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from scripts.forensic_capture_panel import (  # noqa: E402
    OUT_DIR,
    resolve_panel_to_rally_ids,
    run_one_rally,
)


def main() -> None:
    rallies = resolve_panel_to_rally_ids()
    if not rallies:
        print("FAIL: no rallies resolved", flush=True)
        sys.exit(1)

    target = next((r for r in rallies if not r.is_error), rallies[0])
    # Truncate to ~5 seconds for fast smoke test.
    smoke_end_ms = min(target.end_ms, target.start_ms + 5000)
    target.end_ms = smoke_end_ms
    print(
        f"Smoke test target: {target.rally_tag} "
        f"(rally_id={target.rally_id}, desc={target.desc}, "
        f"limited to start_ms={target.start_ms} end_ms={target.end_ms})",
        flush=True,
    )

    ok, elapsed, msg = run_one_rally(target)
    print(f"run_one_rally → ok={ok} elapsed={elapsed:.1f}s msg={msg}", flush=True)
    if not ok:
        sys.exit(1)

    sidecar_path = OUT_DIR / f"{target.rally_tag}.jsonl"
    print(f"Sidecar path: {sidecar_path}", flush=True)
    print(f"Sidecar size: {sidecar_path.stat().st_size} bytes", flush=True)

    lines: list[str] = sidecar_path.read_text().splitlines()
    print(f"Total lines: {len(lines)} (1 meta + {len(lines) - 1} frames)", flush=True)
    if len(lines) < 4:
        print(f"FAIL: too few frame records ({len(lines) - 1})", flush=True)
        sys.exit(1)

    meta = json.loads(lines[0])
    print(f"Meta: type={meta.get('type')} rally_tag={meta.get('rally_tag')}", flush=True)
    sample_indices = [1, 2, len(lines) // 2, len(lines) - 1]
    for idx in sample_indices:
        f = json.loads(lines[idx])
        print(
            f"\nFrame line {idx}: f={f.get('f')} "
            f"n_dets_first={f.get('n_dets_first')} "
            f"n_strack_pool={f.get('n_strack_pool')} "
            f"active_pre={len(f.get('active_pre', []))} "
            f"active_post={len(f.get('active_post', []))}",
            flush=True,
        )
    print("\nSMOKE TEST OK", flush=True)


if __name__ == "__main__":
    main()
