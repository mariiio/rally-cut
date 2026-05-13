"""Stale-version skip semantics in pid_invariants.run_all.

I-3 + I-7 depend on actions_json -> skip if actions_pipeline_version is stale.
I-4 depends on contacts_json -> skip if contacts_pipeline_version is stale.
I-1, I-2, I-5, I-6, I-8 don't depend on that content -> unaffected.
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
from rallycut.tracking.pid_invariants import StaleVersionReport


def test_stale_version_report_dataclass_shape() -> None:
    report = StaleVersionReport(
        total_rallies=10,
        skipped_stale_actions=frozenset({"rally-A", "rally-B"}),
        skipped_stale_contacts=frozenset({"rally-A"}),
        current_actions_version=ACTION_PIPELINE_VERSION,
        current_contacts_version=CONTACT_PIPELINE_VERSION,
        observed_actions_versions={"v0": 2, "v1": 8},
        observed_contacts_versions={"v0": 1, "v1": 9},
    )
    assert report.total_rallies == 10
    assert "rally-A" in report.skipped_stale_actions
    assert report.has_stale is True


def test_run_all_returns_tuple_signature() -> None:
    """Smoke-test the signature: run_all returns (list, StaleVersionReport)."""
    # Use a known video id; if the DB is unreachable we get an exception
    # at the connection level, not a signature mismatch. Just verify the
    # function exists with the new return type by reading its source.
    import inspect

    from rallycut.tracking import pid_invariants

    sig = inspect.signature(pid_invariants.run_all)
    assert (
        "tuple[list[Violation], StaleVersionReport]" in str(sig.return_annotation)
        or "Tuple" in str(sig.return_annotation)
        or "tuple" in str(sig.return_annotation)
    ), f"run_all return type should be tuple; got {sig.return_annotation}"


def test_has_stale_false_when_no_skips() -> None:
    report = StaleVersionReport(
        total_rallies=5,
        skipped_stale_actions=frozenset(),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={"v1": 5},
        observed_contacts_versions={"v1": 5},
    )
    assert report.has_stale is False


def test_mixed_vintage_row_skips_only_the_stale_axis() -> None:
    """A row with actions=current + contacts=stale skips I-4 but runs I-3+I-7.

    The two columns are independent — this is the load-bearing invariant of
    the design (`reattribute_actions` updates actions without touching
    contacts, producing legitimately mixed-vintage rows in production).
    Run_all's per-invariant gates must respect that independence.
    """
    from unittest.mock import patch

    from rallycut.tracking import pid_invariants

    # Rally row tuple shape (matches the SELECT in run_all):
    # (rally_id, primary_track_ids, positions_json, actions_json,
    #  contacts_json, actions_pipeline_version, contacts_pipeline_version)
    # Actions current (no I-3/I-7 skip); contacts stale (I-4 skip).
    fake_rows = [
        (
            "rally-mixed",
            [1, 2, 3, 4],
            [{"trackId": 1, "frameNumber": 0}],
            # actions content: a non-primary playerTrackId (99) → would fire I-3 if checked
            {"actions": [{"playerTrackId": 99, "action": "set", "frame": 5}],
             "teamAssignments": {"1": "A", "2": "A", "3": "B", "4": "B"}},
            # contacts content: a non-primary playerTrackId (88) → would fire I-4 if checked
            [{"playerTrackId": 88, "frame": 10}],
            ACTION_PIPELINE_VERSION,  # actions current
            "v0",                     # contacts stale
        ),
    ]

    with patch("rallycut.tracking.pid_invariants.get_connection") as mock_conn:
        cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        # Two SELECTs run in run_all: rally rows, then video match_analysis_json
        cur.fetchall.return_value = fake_rows
        cur.fetchone.return_value = (None,)  # no match_analysis_json

        violations, stale = pid_invariants.run_all(video_id="vid-mixed")

    seen = {v.invariant for v in violations}

    # I-3 should fire (actions are current — content is checked, finds offender 99)
    assert "I-3" in seen, "actions are current; I-3 should have been checked"
    # I-4 should NOT fire (contacts are stale — skipped; offender 88 not surfaced)
    assert "I-4" not in seen, "contacts are stale; I-4 must be skipped"

    # Stale report reflects per-axis skip
    assert "rally-mixed" not in stale.skipped_stale_actions
    assert "rally-mixed" in stale.skipped_stale_contacts


def test_mixed_vintage_inverse_row_skips_the_other_axis() -> None:
    """Inverse case: actions stale, contacts current. I-3+I-7 skip; I-4 runs."""
    from unittest.mock import patch

    from rallycut.tracking import pid_invariants

    fake_rows = [
        (
            "rally-mixed-inv",
            [1, 2, 3, 4],
            [{"trackId": 1, "frameNumber": 0}],
            {"actions": [{"playerTrackId": 99, "action": "set", "frame": 5}],
             "teamAssignments": {"1": "A", "2": "A", "3": "B", "4": "B"}},
            [{"playerTrackId": 88, "frame": 10}],
            "v0",                          # actions stale
            CONTACT_PIPELINE_VERSION,      # contacts current
        ),
    ]

    with patch("rallycut.tracking.pid_invariants.get_connection") as mock_conn:
        cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = fake_rows
        cur.fetchone.return_value = (None,)

        violations, stale = pid_invariants.run_all(video_id="vid-mixed-inv")

    seen = {v.invariant for v in violations}

    assert "I-3" not in seen, "actions stale → I-3 skipped"
    assert "I-4" in seen, "contacts current → I-4 must check; offender 88 should surface"
    assert "rally-mixed-inv" in stale.skipped_stale_actions
    assert "rally-mixed-inv" not in stale.skipped_stale_contacts


def test_null_version_with_content_is_flagged_as_stale() -> None:
    """A row with content but NULL version columns is stale (CG4 fix).

    This case arises legitimately from mixed-vintage rally merges in
    rallySlicing.ts (concatPlayerTracks returns null when halves disagree)
    OR illegitimately from a producer that forgot to stamp. Either way
    the audit must surface it.
    """
    from unittest.mock import patch

    from rallycut.tracking import pid_invariants

    fake_rows = [
        (
            "rally-unstamped",
            [1, 2, 3, 4],
            [{"trackId": 1, "frameNumber": 0}],
            {"actions": [{"playerTrackId": 99, "action": "set", "frame": 5}],
             "teamAssignments": {"1": "A", "2": "A", "3": "B", "4": "B"}},
            [{"playerTrackId": 88, "frame": 10}],
            None,  # actions_pipeline_version NULL
            None,  # contacts_pipeline_version NULL
        ),
    ]

    with patch("rallycut.tracking.pid_invariants.get_connection") as mock_conn:
        cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = fake_rows
        cur.fetchone.return_value = (None,)

        violations, stale = pid_invariants.run_all(video_id="vid-unstamped")

    seen = {v.invariant for v in violations}

    # Both content-dependent invariants must skip — content exists but no version stamp.
    assert "I-3" not in seen, "null actions version with content → I-3 must skip"
    assert "I-4" not in seen, "null contacts version with content → I-4 must skip"
    # And surface in the report.
    assert "rally-unstamped" in stale.skipped_stale_actions
    assert "rally-unstamped" in stale.skipped_stale_contacts
