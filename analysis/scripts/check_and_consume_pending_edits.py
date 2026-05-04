"""
Pre-flight + post-flight handling of pendingAnalysisEditsJson for the
diagnostic refresh path (`refresh_videos.sh`).

The TS service `runMatchAnalysis` honors a contract:
  edit -> queued in pendingAnalysisEditsJson -> next runMatchAnalysis
  consumes them via consumePendingEdits + plans rerun via planStages.

The CLI/diagnostic path bypasses that contract today, leaving edits
queued indefinitely while match-players runs against possibly-stale
state. This causes editor↔DB drift (e.g., dd042609 r18 in the
2026-05-04 validation session).

This script enforces a safe subset of the contract from the CLI side:

Modes:
  --check <video_id>    -- exit non-zero if any pending edit needs retrack
                           (extend / create / merge / refCrop). For these
                           kinds the diagnostic flow cannot faithfully
                           re-process — abort with instructions to use
                           the editor's Re-analyze flow instead.
  --consume <video_id>  -- after match-players+remap completed cleanly,
                           clear pendingAnalysisEditsJson IF AND ONLY IF
                           all entries are safe-for-CLI kinds (scalar,
                           shorten, delete, split). Otherwise no-op +
                           warn (we don't claim to have handled what we
                           didn't).
"""

from __future__ import annotations

import sys

from rallycut.evaluation.tracking.db import get_connection

# Edits the CLI path cannot faithfully apply (need retrack and/or full
# canonical-map rebuild).
NEEDS_FULL_PIPELINE = {"extend", "create", "merge", "refCrop"}

# Edits whose effects are already in the DB tables (rally bounds, rally
# row) and which only need match-players + remap re-run -- safe to
# consume from the CLI path.
SAFE_FOR_CLI = {"scalar", "shorten", "delete", "split"}


def _read_pending(video_id: str) -> list[dict]:
    with get_connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT pending_analysis_edits_json FROM videos WHERE id = %s",
            (video_id,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return []
    return row[0].get("entries", []) or []


def _null_pending(video_id: str) -> None:
    with get_connection() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE videos SET pending_analysis_edits_json = NULL WHERE id = %s",
            (video_id,),
        )
        c.commit()


def cmd_check(video_id: str) -> int:
    entries = _read_pending(video_id)
    if not entries:
        print(f"[pending-edits] {video_id[:8]}: queue empty - OK")
        return 0

    blocking = [e for e in entries if e.get("editKind") in NEEDS_FULL_PIPELINE]
    safe = [e for e in entries if e.get("editKind") in SAFE_FOR_CLI]
    other = [e for e in entries
             if e.get("editKind") not in NEEDS_FULL_PIPELINE and e.get("editKind") not in SAFE_FOR_CLI]

    if blocking:
        print(f"[pending-edits] {video_id[:8]}: {len(blocking)} edit(s) require the full")
        print("  match-analysis pipeline (retrack + repair-identities + reattribute-actions)")
        print("  which the CLI script does not run. Refusing to proceed with stale state.")
        for e in blocking:
            print(f"    - rally {e.get('rallyId', '?')[:8]:>8}  {e.get('editKind')}  at {e.get('at')}")
        print()
        print("  Resolution options (pick one):")
        print("  1. In the web editor, click \"Re-analyze\" on this video.")
        print(f"  2. Trigger the API: curl -X POST http://localhost:3001/v1/videos/{video_id}/trigger-match-analysis")
        print("     (requires X-Visitor-Id header)")
        print("  3. If you intend to discard the pending edits and run a fresh diagnostic")
        print("     refresh, clear them first: psql -c \"UPDATE videos SET")
        print(f"     pending_analysis_edits_json = NULL WHERE id = '{video_id}';\"")
        return 1

    if safe or other:
        print(f"[pending-edits] {video_id[:8]}: {len(safe)} CLI-safe edit(s) queued; "
              f"will be cleared post-refresh")
        for e in safe:
            print(f"    - {e.get('editKind')} on rally {e.get('rallyId', '?')[:8]}")
        if other:
            print(f"  WARNING: {len(other)} edit(s) of unknown kind, will NOT be cleared:")
            for e in other:
                print(f"    - {e.get('editKind')} on rally {e.get('rallyId', '?')[:8]}")
    return 0


def cmd_consume(video_id: str) -> int:
    """Clear the queue if and only if all entries are CLI-safe."""
    entries = _read_pending(video_id)
    if not entries:
        return 0

    if any(e.get("editKind") in NEEDS_FULL_PIPELINE for e in entries):
        print(f"[pending-edits] {video_id[:8]}: refusing to consume "
              f"(blocking edits still present)")
        return 1

    unknown = [e for e in entries if e.get("editKind") not in SAFE_FOR_CLI]
    if unknown:
        print(f"[pending-edits] {video_id[:8]}: {len(unknown)} edit(s) of unknown kind "
              f"left in queue - not clearing")
        return 0

    _null_pending(video_id)
    print(f"[pending-edits] {video_id[:8]}: cleared {len(entries)} CLI-safe edit(s)")
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("--check", "--consume"):
        print("Usage: check_and_consume_pending_edits.py --check|--consume <video_id>",
              file=sys.stderr)
        return 2
    mode, video_id = sys.argv[1], sys.argv[2]
    if mode == "--check":
        return cmd_check(video_id)
    return cmd_consume(video_id)


if __name__ == "__main__":
    sys.exit(main())
