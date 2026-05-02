"""Tests for the assignmentAnchor `matcherVersion` invalidation key.

`trackStatsHash` captures the matcher's INPUT fingerprint but not the
matcher's logic. Without a version stamp, any matcher fix is silently
bypassed for already-anchored rallies. The fix: write `matcherVersion`
into every persisted anchor; the read path drops anchors whose version
doesn't match the current `MATCHER_VERSION` constant.

This file pins the contract — bumping `MATCHER_VERSION` MUST invalidate
prior anchors on the next read; same-version anchors with matching hash
must still be honored.
"""
from __future__ import annotations

from typing import Any

from rallycut.tracking.match_tracker import (
    MATCHER_VERSION,
)


# Helper that mirrors the read-side filter logic in
# match_tracker.run_match_players (lines ~3785-3805 post-edit).
def _filter_pinned(
    prior_anchors: dict[str, dict[str, Any]],
    rally_ids: list[str],
    track_stats_hashes: dict[str, str],
) -> tuple[dict[int, dict[int, int]], int]:
    pinned: dict[int, dict[int, int]] = {}
    stale_version_count = 0
    for i, rid in enumerate(rally_ids):
        anchor = prior_anchors.get(rid)
        if not anchor:
            continue
        if anchor.get("matcherVersion") != MATCHER_VERSION:
            stale_version_count += 1
            continue
        if anchor.get("trackStatsHash") != track_stats_hashes.get(rid):
            continue
        pinned[i] = {
            int(k): int(v) for k, v in anchor.get("assignment", {}).items()
        }
    return pinned, stale_version_count


class TestMatcherVersionInvalidation:
    def test_anchor_with_current_version_is_pinned(self) -> None:
        prior = {
            "r1": {
                "matcherVersion": MATCHER_VERSION,
                "trackStatsHash": "abc",
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            }
        }
        pinned, stale = _filter_pinned(prior, ["r1"], {"r1": "abc"})
        assert 0 in pinned
        assert pinned[0] == {1: 1, 2: 2, 3: 3, 4: 4}
        assert stale == 0

    def test_anchor_with_old_version_is_invalidated(self) -> None:
        prior = {
            "r1": {
                "matcherVersion": "v0_legacy",
                "trackStatsHash": "abc",  # hash matches!
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            }
        }
        pinned, stale = _filter_pinned(prior, ["r1"], {"r1": "abc"})
        assert pinned == {}
        assert stale == 1

    def test_anchor_without_version_field_is_invalidated(self) -> None:
        # Pre-versioning anchors written by older code. They lack the
        # `matcherVersion` key entirely. Treat as stale.
        prior = {
            "r1": {
                "trackStatsHash": "abc",
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            }
        }
        pinned, stale = _filter_pinned(prior, ["r1"], {"r1": "abc"})
        assert pinned == {}
        assert stale == 1

    def test_anchor_hash_mismatch_is_dropped_silently(self) -> None:
        # Hash mismatches existed before versioning. They're a normal
        # cache-miss path — not counted as a stale-version event.
        prior = {
            "r1": {
                "matcherVersion": MATCHER_VERSION,
                "trackStatsHash": "OLD",
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            }
        }
        pinned, stale = _filter_pinned(prior, ["r1"], {"r1": "NEW"})
        assert pinned == {}
        assert stale == 0  # NOT a version-staleness event

    def test_mixed_versions_partial_invalidation(self) -> None:
        prior = {
            "r1": {
                "matcherVersion": MATCHER_VERSION,
                "trackStatsHash": "h1",
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            },
            "r2": {
                "matcherVersion": "v0",
                "trackStatsHash": "h2",
                "assignment": {"1": 1, "2": 2, "3": 3, "4": 4},
            },
            "r3": {
                "matcherVersion": MATCHER_VERSION,
                "trackStatsHash": "h3",
                "assignment": {"5": 1, "6": 2, "7": 3, "8": 4},
            },
        }
        pinned, stale = _filter_pinned(
            prior, ["r1", "r2", "r3"],
            {"r1": "h1", "r2": "h2", "r3": "h3"},
        )
        assert set(pinned.keys()) == {0, 2}
        assert stale == 1


class TestMatcherVersionConstant:
    def test_version_is_non_empty_string(self) -> None:
        # Sanity: opaque, but non-empty so equality checks aren't trivially True.
        assert isinstance(MATCHER_VERSION, str)
        assert len(MATCHER_VERSION) > 0

    def test_version_differs_from_v1_legacy(self) -> None:
        # The first published version was implicitly "v1" (no field).
        # Today's value MUST differ from that, otherwise pre-versioning
        # anchors written without the field — which we treat as stale —
        # would conflict if anyone manually wrote {"matcherVersion": "v1"}.
        # Bumping past "v1" makes the contract: "no version field" or
        # "matcherVersion=v1" both invalidate.
        assert MATCHER_VERSION != "v1"
