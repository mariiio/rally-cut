"""User visual verdicts for the 13-rally cross-rally identity panel as of
2026-05-01, after the 7307c1d revert (commit ce7b08c).

Source of truth: `panel_visual_verdict_2026_05_01.md` in the user's memory.
The 13 rallies are the same fixtures as the original 04-29 panel (mirrored
in `forensic_panel_ground_truth.py`), but the verdicts have shifted: 5
panel errors cleared to GOOD, 4 errors took on different shapes, and 1
control regressed.

Schema:

    PANEL_2026_05_01: list[PanelVerdict]

Each entry encodes:
  - rally identifier: (video_id, rally_idx) — same ordering as
    `forensic_panel_ground_truth._resolve_rally_id` (start_ms ascending,
    filtered to player_tracks with positions_json)
  - is_control: True for the 4 control rallies (background expectation)
  - expected_verdict: "GOOD" if the user marked the rally clean, "BAD"
    otherwise. Bool-coerced to drive AGREES/DISAGREES totals.
  - expected_shape: descriptive label for the anomaly the user observed
    (verdict tool reports its own measured signal; the human reads both)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelVerdict:
    video_id: str
    rally_idx: int  # 0-based; matches forensic_panel_ground_truth ordering
    is_control: bool
    expected_verdict: str  # "GOOD" or "BAD"
    expected_shape: str  # informational; describes the user-observed anomaly

    @property
    def short_id(self) -> str:
        return self.video_id[:8]

    @property
    def rally_tag(self) -> str:
        return f"{self.short_id}/r{self.rally_idx + 1:02d}"


PANEL_2026_05_01: list[PanelVerdict] = [
    # --- 7d77980f fixture ---
    PanelVerdict(
        video_id="7d77980f-3006-40e0-adc0-db491a5bb659",
        rally_idx=0, is_control=True,
        expected_verdict="GOOD", expected_shape="clean (control held)",
    ),
    PanelVerdict(
        video_id="7d77980f-3006-40e0-adc0-db491a5bb659",
        rally_idx=1, is_control=False,
        expected_verdict="BAD",
        expected_shape="mixed: p1 starts as p2 (fixes after occlusion) + NEW p2<->p4 cross-team swap",
    ),
    PanelVerdict(
        video_id="7d77980f-3006-40e0-adc0-db491a5bb659",
        rally_idx=12, is_control=False,
        expected_verdict="GOOD", expected_shape="cleared (was p2<->p4 swap)",
    ),
    PanelVerdict(
        video_id="7d77980f-3006-40e0-adc0-db491a5bb659",
        rally_idx=18, is_control=False,
        expected_verdict="GOOD", expected_shape="cleared (was p2<->p4 swap)",
    ),

    # --- b5fb0594 fixture ---
    PanelVerdict(
        video_id="b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
        rally_idx=0, is_control=True,
        expected_verdict="BAD",
        expected_shape="control REGRESSED: tracks 3 should be 4, 2 should be 3 (only 3 distinct PIDs)",
    ),
    PanelVerdict(
        video_id="b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
        rally_idx=3, is_control=False,
        expected_verdict="GOOD", expected_shape="cleared (was within-team p3<->p4)",
    ),
    PanelVerdict(
        video_id="b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
        rally_idx=5, is_control=False,
        expected_verdict="GOOD", expected_shape="cleared (was within-team p3<->p4)",
    ),
    PanelVerdict(
        video_id="b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
        rally_idx=9, is_control=False,
        expected_verdict="BAD",
        expected_shape="starts GOOD then NEW p1<->p2 within-team swap after occlusion",
    ),

    # --- 5c756c41 fixture ---
    PanelVerdict(
        video_id="5c756c41-1cc1-4486-a95c-97398912cfbe",
        rally_idx=0, is_control=True,
        expected_verdict="GOOD", expected_shape="clean (control held)",
    ),
    PanelVerdict(
        video_id="5c756c41-1cc1-4486-a95c-97398912cfbe",
        rally_idx=2, is_control=False,
        expected_verdict="BAD", expected_shape="NEW p2<->p3 cross-team start swap",
    ),
    PanelVerdict(
        video_id="5c756c41-1cc1-4486-a95c-97398912cfbe",
        rally_idx=6, is_control=False,
        expected_verdict="BAD",
        expected_shape="NEW BOTH p1<->p2 AND p3<->p4 within-team swaps",
    ),

    # --- 854bb250 fixture ---
    PanelVerdict(
        video_id="854bb250-3e91-47d2-944d-f62413e3cf45",
        rally_idx=0, is_control=False,
        expected_verdict="GOOD", expected_shape="cleared (was within-team p2<->p4)",
    ),
    PanelVerdict(
        video_id="854bb250-3e91-47d2-944d-f62413e3cf45",
        rally_idx=1, is_control=True,
        expected_verdict="GOOD", expected_shape="clean (control held)",
    ),
]


def panel_summary() -> dict[str, int]:
    """Quick summary used by the verdict tool to sanity-check totals."""
    n_total = len(PANEL_2026_05_01)
    n_good = sum(1 for v in PANEL_2026_05_01 if v.expected_verdict == "GOOD")
    n_bad = sum(1 for v in PANEL_2026_05_01 if v.expected_verdict == "BAD")
    n_ctrl = sum(1 for v in PANEL_2026_05_01 if v.is_control)
    return {"total": n_total, "good": n_good, "bad": n_bad, "controls": n_ctrl}


if __name__ == "__main__":
    s = panel_summary()
    print(f"Panel 2026-05-01: {s['total']} rallies "
          f"({s['good']} GOOD, {s['bad']} BAD, {s['controls']} controls)")
    for v in PANEL_2026_05_01:
        kind = "CTRL" if v.is_control else "PANEL"
        print(f"  {v.rally_tag:<14} {kind:<5} {v.expected_verdict:<4} {v.expected_shape}")
