"""SubTrackCandidate: a time-bounded slice of a parent track for Hungarian assignment.

Used by `match_tracker._segment_tracks_by_appearance` to break a single BoT-SORT
track into two halves when within-track appearance evidence shows the physical
player carrying that track_id changed mid-rally (silent swap, early-rally
co-tracking).

The synthetic_track_id is negative so it cannot collide with real positive
track_ids in `track_to_player` mappings or downstream consumers. Sub-tracks
keep a pointer to the parent's `track_id` so post-Hungarian can rewrite the
final mapping back into the parent's track_id space, but at frame-level
granularity (each frame in [f_start, f_end] inherits the sub-track's pid).
"""
from __future__ import annotations

from dataclasses import dataclass

from rallycut.tracking.player_features import TrackAppearanceStats


@dataclass
class SubTrackCandidate:
    parent_track_id: int
    segment_index: int  # 0 = pre-flip, 1 = post-flip; allows future >2 segments.
    f_start: int  # Inclusive (rally-relative frame number).
    f_end: int  # Inclusive.
    appearance_stats: TrackAppearanceStats
    aggregated_argmax_pid: int | None = None
    aggregated_margin: float | None = None

    @property
    def synthetic_track_id(self) -> int:
        # Negative, deterministic, parent+segment unique. Magnitude well above
        # any plausible real track_id (BoT-SORT track ids stay in the hundreds
        # at most for a single rally).
        return -1000 * (self.segment_index + 1) - self.parent_track_id - 2

    def overlaps(self, other: SubTrackCandidate) -> bool:
        if self.parent_track_id == other.parent_track_id and self.segment_index == other.segment_index:
            return True
        return not (self.f_end < other.f_start or other.f_end < self.f_start)
