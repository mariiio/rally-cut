"""Run measure_pid_accuracy on all 9 Phase-0 fixtures and report PERMUTED.

Wrapper around measure_pid_accuracy.py that loops over the canonical
attribution-rebuild fixtures and aggregates PERMUTED + DIRECT + ID-stability.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURE_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_rebuild" / "fixture_video_ids_2026_04_24.json"
)


def _parse_metric_from_log(log: str) -> dict[str, float]:
    """Parse direct%, permuted%, id-stability average from measure_pid_accuracy."""
    out: dict[str, float] = {}
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("OVERALL:"):
            # OVERALL: 27/53 = 50.9%
            pct = s.split("=")[-1].strip().rstrip("%")
            try:
                pct_val = float(pct)
            except ValueError:
                continue
            if "direct" not in out:
                out["direct"] = pct_val
            else:
                out["permuted"] = pct_val
        elif s.startswith("AVERAGE distinct PIDs per GT player:"):
            # AVERAGE distinct PIDs per GT player: 1.00 (1.00 = perfectly stable...)
            try:
                out["id_stability"] = float(s.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    return out


def main() -> int:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]
    print(f"{'fixture':<8} {'direct':>8} {'permuted':>10} {'id_stab':>8}")
    print("-" * 40)
    rows: list[tuple[str, float, float, float]] = []
    for fixture_name, finfo in fixture_map.items():
        video_id = finfo["video_id"]
        import os
        result = subprocess.run(
            ["python", "-u", "scripts/measure_pid_accuracy.py", video_id],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            print(f"{fixture_name:<8} ERROR: {result.stderr.splitlines()[-1] if result.stderr else 'no stderr'}")
            continue
        metrics = _parse_metric_from_log(result.stdout)
        direct = metrics.get("direct", 0.0)
        permuted = metrics.get("permuted", 0.0)
        id_stab = metrics.get("id_stability", 0.0)
        rows.append((fixture_name, direct, permuted, id_stab))
        print(f"{fixture_name:<8} {direct:>7.1f}% {permuted:>9.1f}% {id_stab:>8.2f}")

    print("-" * 40)
    if rows:
        avg_direct = sum(r[1] for r in rows) / len(rows)
        avg_permuted = sum(r[2] for r in rows) / len(rows)
        avg_stab = sum(r[3] for r in rows) / len(rows)
        print(f"{'AVG':<8} {avg_direct:>7.1f}% {avg_permuted:>9.1f}% {avg_stab:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
