"""Unit tests for rallycut.evaluation.corpus_freshness.

Does NOT touch the DB or any eval harness — freshness logic is pure data.
Reproduction is mocked via an in-test stub reproduce_canary_fn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rallycut.evaluation.corpus_freshness import (
    StaleCorpusError,
    build_meta_header,
    compute_canary_fingerprint,
    iter_errors,
    read_meta_header,
    verify_corpus_fresh,
)

SAMPLE_ERRORS = [
    {
        "rally_id": "r-alpha",
        "video_id": "v-canary",
        "gt_frame": 100,
        "gt_action": "serve",
        "pred_frame": None,
        "pred_action": None,
        "error_class": "FN_contact",
        "fn_subcategory": "no_candidate",
    },
    {
        "rally_id": "r-beta",
        "video_id": "v-canary",
        "gt_frame": 200,
        "gt_action": "dig",
        "pred_frame": 201,
        "pred_action": "set",
        "error_class": "wrong_action",
        "fn_subcategory": None,
    },
    {
        "rally_id": "r-gamma",
        "video_id": "v-other",
        "gt_frame": 50,
        "gt_action": "attack",
        "pred_frame": None,
        "pred_action": None,
        "error_class": "FN_contact",
        "fn_subcategory": "rejected_by_classifier",
    },
]


def test_fingerprint_deterministic():
    a = compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
    b = compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
    assert a == b, "Fingerprint must be deterministic across calls"


def test_fingerprint_order_independent():
    shuffled = [SAMPLE_ERRORS[1], SAMPLE_ERRORS[0], SAMPLE_ERRORS[2]]
    assert (
        compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
        == compute_canary_fingerprint(shuffled, "v-canary")
    ), "Fingerprint must be invariant to input order"


def test_fingerprint_ignores_non_canary_video():
    reduced = [SAMPLE_ERRORS[0], SAMPLE_ERRORS[1]]  # drop v-other
    assert (
        compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
        == compute_canary_fingerprint(reduced, "v-canary")
    ), "Errors from non-canary videos must NOT affect the fingerprint"


def test_fingerprint_changes_on_structural_diff():
    modified = [dict(e) for e in SAMPLE_ERRORS]
    modified[0]["error_class"] = "wrong_player"  # structural change
    assert (
        compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
        != compute_canary_fingerprint(modified, "v-canary")
    ), "Fingerprint must react to changes in hashed fields"


def test_fingerprint_stable_under_float_jitter():
    modified = [dict(e) for e in SAMPLE_ERRORS]
    # A float field that's NOT part of _CANARY_HASH_FIELDS should have no effect.
    modified[0]["classifier_conf"] = 0.12345678
    modified[1]["classifier_conf"] = 0.87654321
    assert (
        compute_canary_fingerprint(SAMPLE_ERRORS, "v-canary")
        == compute_canary_fingerprint(modified, "v-canary")
    ), "Fields outside the hash projection must not affect the fingerprint"


def test_build_meta_header_fields():
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=3, n_gt=12,
        tp=10, fn=1, fp=1, wrong_action=1,
        f1=0.8, action_acc=0.9,
    )
    assert meta["_meta"] is True
    assert meta["n_rallies"] == 3
    assert meta["canary_video_id"] == "v-canary"  # first sorted
    assert meta["canary_fingerprint"].startswith("sha256:")


def _write_corpus(tmp_path: Path, meta: dict | None, errors: list) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w") as f:
        if meta is not None:
            f.write(json.dumps(meta) + "\n")
        for e in errors:
            f.write(json.dumps(e) + "\n")
    return path


def test_read_meta_header_present(tmp_path: Path):
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=1, n_gt=3,
        tp=2, fn=1, fp=0, wrong_action=1, f1=0.5, action_acc=0.5,
    )
    path = _write_corpus(tmp_path, meta, SAMPLE_ERRORS)
    round_tripped = read_meta_header(path)
    assert round_tripped is not None
    assert round_tripped["canary_video_id"] == "v-canary"


def test_read_meta_header_missing(tmp_path: Path):
    path = _write_corpus(tmp_path, meta=None, errors=SAMPLE_ERRORS)
    assert read_meta_header(path) is None


def test_iter_errors_skips_meta(tmp_path: Path):
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=1, n_gt=3,
        tp=2, fn=1, fp=0, wrong_action=1, f1=0.5, action_acc=0.5,
    )
    path = _write_corpus(tmp_path, meta, SAMPLE_ERRORS)
    got = list(iter_errors(path))
    assert len(got) == len(SAMPLE_ERRORS)
    assert got[0]["rally_id"] == SAMPLE_ERRORS[0]["rally_id"]


def test_verify_corpus_fresh_match(tmp_path: Path):
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=1, n_gt=3,
        tp=2, fn=1, fp=0, wrong_action=1, f1=0.5, action_acc=0.5,
    )
    path = _write_corpus(tmp_path, meta, SAMPLE_ERRORS)

    def repro(vid: str) -> list[dict]:
        return [e for e in SAMPLE_ERRORS if e["video_id"] == vid]

    assert verify_corpus_fresh(path, reproduce_canary_fn=repro) is True


def test_verify_corpus_fresh_stale_raises(tmp_path: Path):
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=1, n_gt=3,
        tp=2, fn=1, fp=0, wrong_action=1, f1=0.5, action_acc=0.5,
    )
    path = _write_corpus(tmp_path, meta, SAMPLE_ERRORS)
    stale_sample = [dict(e) for e in SAMPLE_ERRORS]
    stale_sample[0]["error_class"] = "wrong_player"  # structural diff

    def repro(vid: str) -> list[dict]:
        return [e for e in stale_sample if e["video_id"] == vid]

    with pytest.raises(StaleCorpusError):
        verify_corpus_fresh(path, reproduce_canary_fn=repro, abort_on_stale=True)


def test_verify_corpus_fresh_stale_soft_return(tmp_path: Path):
    meta = build_meta_header(
        errors=SAMPLE_ERRORS, n_rallies=1, n_gt=3,
        tp=2, fn=1, fp=0, wrong_action=1, f1=0.5, action_acc=0.5,
    )
    path = _write_corpus(tmp_path, meta, SAMPLE_ERRORS)
    stale_sample = [dict(e) for e in SAMPLE_ERRORS]
    stale_sample[0]["error_class"] = "wrong_player"

    def repro(vid: str) -> list[dict]:
        return [e for e in stale_sample if e["video_id"] == vid]

    assert verify_corpus_fresh(path, reproduce_canary_fn=repro, abort_on_stale=False) is False


def test_verify_corpus_fresh_legacy_no_meta_soft(tmp_path: Path):
    path = _write_corpus(tmp_path, meta=None, errors=SAMPLE_ERRORS)

    def repro(vid: str) -> list[dict]:
        return SAMPLE_ERRORS

    assert verify_corpus_fresh(path, reproduce_canary_fn=repro,
                                abort_on_stale=False, abort_on_legacy=False) is False


def test_verify_corpus_fresh_legacy_no_meta_hard(tmp_path: Path):
    path = _write_corpus(tmp_path, meta=None, errors=SAMPLE_ERRORS)

    def repro(vid: str) -> list[dict]:
        return SAMPLE_ERRORS

    with pytest.raises(StaleCorpusError):
        verify_corpus_fresh(path, reproduce_canary_fn=repro, abort_on_legacy=True)
