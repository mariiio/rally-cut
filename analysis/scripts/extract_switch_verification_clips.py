"""Extract short clips around proposed sideSwitch fixes for visual verification.

For each proposed switch change, extracts clips for rallies at:
  switch-1, switch, switch+1
so the user can see if teams physically swap sides between consecutive rallies.

Usage:
    cd analysis
    uv run python scripts/extract_switch_verification_clips.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from scripts.extract_serve_debug_clips import (  # noqa: E402
    extract_clip,
    load_rally_data,
)

OUTPUT_DIR = Path("outputs/switch_verification_clips")

# Proposed fixes from audit_side_switches.py
# Format: (video_id_prefix, video_name, description, current_switches, proposed_switches)
PROPOSED_FIXES = [
    ("ae81fff5", "lele", "move 27→26", [6, 13, 20, 27], [6, 13, 20, 26]),
    ("bbd880f2", "papa", "move 3→4", [3], [4]),
    ("16458e78", "pepe", "move 5→6", [5], [6]),
    ("2d105b7b", "pipi", "add @6", [3], [3, 6]),
    ("7d77980f", "tata", "move 19→20", [6, 13, 19], [6, 13, 20]),
    ("d5a6932f", "tete1", "move 14→13", [7, 14], [7, 13]),
    ("2e984c43", "titi", "move 14→13", [7, 14, 21, 28], [7, 13, 21, 28]),
    ("ff175026", "lolo", "add @27", [7, 14, 21], [7, 14, 21, 27]),
    # dd042609 tete2 has multiple possible shifts, skip for now
]


def main() -> int:
    # Always start fresh — remove old clips to avoid confusion
    if OUTPUT_DIR.exists():
        for old in OUTPUT_DIR.glob("*.mp4"):
            old.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.eval_score_tracking import load_score_gt

    video_rallies = load_score_gt()

    # Load video names
    video_names: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        vids = list(video_rallies.keys())
        ph = ", ".join(["%s"] * len(vids))
        cur.execute(f"SELECT id, s3_key FROM videos WHERE id IN ({ph})", vids)
        for vid, s3_key in cur.fetchall():
            video_names[vid] = Path(s3_key).stem if s3_key else vid[:8]

    for vid_prefix, vname, desc, current_sw, proposed_sw in PROPOSED_FIXES:
        # Find full video ID
        vid = None
        for v in video_rallies:
            if v.startswith(vid_prefix):
                vid = v
                break
        if vid is None:
            print(f"Video {vid_prefix} ({vname}) not found, skipping")
            continue

        rallies = video_rallies[vid]
        print(f"\n=== {vname} ({desc}) — {len(rallies)} rallies ===")

        # Determine which rally indices to extract clips for
        # We need rallies around each CHANGED switch point
        changed_indices: set[int] = set()
        removed = set(current_sw) - set(proposed_sw)
        added = set(proposed_sw) - set(current_sw)
        for idx in removed | added:
            for offset in [-1, 0, 1]:
                ri = idx + offset
                if 0 <= ri < len(rallies):
                    changed_indices.add(ri)

        # Also add rallies around current switch points that moved
        for old_sw in current_sw:
            for new_sw in proposed_sw:
                if old_sw != new_sw and abs(old_sw - new_sw) <= 2:
                    for idx in [old_sw, new_sw]:
                        for offset in [-1, 0, 1]:
                            ri = idx + offset
                            if 0 <= ri < len(rallies):
                                changed_indices.add(ri)

        target_indices = sorted(changed_indices)
        print(f"  Extracting rallies at indices: {target_indices}")

        # Get rally IDs for these indices
        target_rally_ids = [rallies[i].rally_id for i in target_indices]
        clip_data = load_rally_data(target_rally_ids)
        data_by_id = {r["rally_id"]: r for r in clip_data}

        # Extract clips with index prefix for easy ordering
        for idx in target_indices:
            r = rallies[idx]
            if r.rally_id not in data_by_id:
                print(f"  Skipping rally {idx} ({r.rally_id[:8]}) — no clip data")
                continue

            rally = data_by_id[r.rally_id]

            # Mark if this index is a current or proposed switch
            markers = []
            if idx in current_sw:
                markers.append("CUR_SW")
            if idx in proposed_sw and idx not in current_sw:
                markers.append("NEW_SW")
            if idx in current_sw and idx not in proposed_sw:
                markers.append("DEL_SW")
            marker_str = f"_{'_'.join(markers)}" if markers else ""

            filename = (
                f"{vname}_{idx:02d}_gt{r.gt_serving_team}"
                f"{marker_str}_{r.rally_id[:8]}.mp4"
            )
            output_path = OUTPUT_DIR / filename

            # Determine GT physical side for overlay
            # (We pass None here since we're checking switches, not formation)
            print(f"  [{idx:2d}] {filename} (gt={r.gt_serving_team}, "
                  f"flip={'Y' if r.side_flipped else 'N'})...")
            ok = extract_clip(rally, output_path, duration_s=3.0)
            if ok:
                print(f"    -> {output_path}")
            else:
                print(f"    FAILED")

    print(f"\n=== How to review ===")
    print(f"For each video, watch the clips in index order.")
    print(f"At a switch point, teams should physically swap court sides.")
    print(f"")
    print(f"  CUR_SW = current switch in GT (may be wrong)")
    print(f"  NEW_SW = proposed new switch location")
    print(f"  DEL_SW = switch proposed for removal")
    print(f"")
    print(f"Watch clip N-1 and clip N. If teams swap → switch at N is correct.")
    print(f"If they DON'T swap → switch is wrong (off-by-one or missing).")
    print(f"\nClips saved to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
