"""Corpus freshness verification — git-independent.

Rationale. Stored eval artifacts (corpus_eval_reconciled.jsonl,
fn_stage_attribution.jsonl) can go stale when code/data changes between the
time the artifact was built and the time analysis reads it. On 2026-04-21
this silently invalidated ~12% of FN records — Phase 4-6 analysis of the
contact-detection review was built on errors that no longer reproduced
under current code, and the drift was only caught by accident.

Approach. Instead of relying on git_sha (which misses uncommitted changes
and over-fires on no-op commits), the corpus writer records a
**canary-fold fingerprint**: a canonical hash of the error records for a
single deterministic video fold. The freshness pre-flight re-runs that one
fold under current code and compares hashes. Adds ~1 minute to any
analysis script that consumes the corpus.

The canary is the FIRST video (sorted by video_id) — this guarantees the
same video is always used and keeps the verify cheap. Deterministic.

Public API:

    compute_canary_fingerprint(errors: list[dict], canary_video_id: str) -> str
    write_meta_header(fh, meta: dict) -> None
    read_meta_header(corpus_path: Path) -> dict | None
    verify_corpus_fresh(corpus_path: Path) -> None      # raises on stale

Do NOT consult git. This module is intentionally git-independent.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

# Fields included in the canary hash. Structural fields only — we do NOT
# hash floats like classifier_conf (tiny FP jitter is not a staleness
# signal). Add fields here if their change means the pipeline changed.
_CANARY_HASH_FIELDS = (
    "rally_id",
    "gt_frame",
    "gt_action",
    "pred_frame",
    "pred_action",
    "error_class",
    "fn_subcategory",
)


class StaleCorpusError(RuntimeError):
    """Raised when a corpus no longer reproduces under current code."""


def _canonicalize_error(record: dict) -> dict:
    """Project an error record to the fields we hash."""
    return {k: record.get(k) for k in _CANARY_HASH_FIELDS}


def compute_canary_fingerprint(
    errors: Iterable[dict],
    canary_video_id: str,
) -> str:
    """SHA256 of the canonically-sorted canary-fold error records."""
    canary = [e for e in errors if e.get("video_id") == canary_video_id]
    canonical = sorted(
        (_canonicalize_error(e) for e in canary),
        key=lambda e: (
            str(e.get("rally_id", "")),
            int(e.get("gt_frame", 0) or 0),
            str(e.get("error_class", "")),
        ),
    )
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_meta_header(
    *,
    errors: Iterable[dict],
    n_rallies: int,
    n_gt: int,
    tp: int,
    fn: int,
    fp: int,
    wrong_action: int,
    f1: float,
    action_acc: float,
) -> dict:
    """Produce the _meta dict a corpus builder should write as line 1."""
    errors_list = list(errors)
    video_id_set: set[str] = {
        str(e["video_id"]) for e in errors_list
        if e.get("video_id")
    }
    video_ids: list[str] = sorted(video_id_set)
    # Canary = first video with errors. If zero errors in the first video of
    # the corpus (unusual), we still pick it by sort order — the fingerprint
    # of an empty fold is stable and still valid.
    canary_video_id: str = video_ids[0] if video_ids else ""
    fingerprint = compute_canary_fingerprint(errors_list, canary_video_id)
    return {
        "_meta": True,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_rallies": n_rallies,
        "n_gt": n_gt,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "wrong_action": wrong_action,
        "f1": round(f1, 6),
        "action_acc": round(action_acc, 6),
        "canary_video_id": canary_video_id,
        "canary_fingerprint": fingerprint,
        "canary_hash_fields": list(_CANARY_HASH_FIELDS),
    }


def read_meta_header(corpus_path: Path) -> dict | None:
    """Read the first line of a corpus jsonl and return it if it's a _meta line.

    Returns None if the corpus has no meta header (legacy corpus without
    embedded fingerprint — callers must handle this gracefully, e.g., warn
    instead of hard-abort).
    """
    with corpus_path.open() as fh:
        first = fh.readline()
    if not first.strip():
        return None
    try:
        record = json.loads(first)
    except json.JSONDecodeError:
        return None
    if isinstance(record, dict) and record.get("_meta") is True:
        return record
    return None


def iter_errors(corpus_path: Path) -> Iterable[dict]:
    """Yield error records from a corpus, skipping any _meta header line."""
    with corpus_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("_meta") is True:
                continue
            yield record


def verify_corpus_fresh(
    corpus_path: Path,
    *,
    reproduce_canary_fn: Callable[[str], Iterable[dict]],
    abort_on_stale: bool = True,
    abort_on_legacy: bool = False,
) -> bool:
    """Re-run the canary fold and compare its fingerprint to the corpus header.

    Args:
        corpus_path: jsonl corpus file to verify.
        reproduce_canary_fn: callable (canary_video_id: str) -> list[dict] that
            re-runs the eval for ONLY that video and returns its error records.
            The caller wires this to whichever eval harness produced the
            corpus (e.g. build_eval_reconciled_corpus._process_fold).
        abort_on_stale: raise StaleCorpusError on mismatch (default) vs. return
            False and print a warning.
        abort_on_legacy: raise if the corpus has no _meta header. Default False
            (warn only) so legacy corpora don't break first-time callers.

    Returns:
        True if fresh (fingerprint matches). False if legacy-no-meta and
        abort_on_legacy=False.

    Raises:
        StaleCorpusError if reproduction fingerprint differs and
            abort_on_stale=True.
    """
    meta = read_meta_header(corpus_path)
    if meta is None:
        msg = (
            f"Corpus {corpus_path.name} has no _meta header (legacy format). "
            f"Cannot verify freshness — rebuild with the current builder to "
            f"embed a canary fingerprint."
        )
        if abort_on_legacy:
            raise StaleCorpusError(msg)
        print(f"[corpus_freshness] WARNING: {msg}")
        return False

    canary_video_id = meta["canary_video_id"]
    expected = meta["canary_fingerprint"]
    print(f"[corpus_freshness] Verifying canary fold {canary_video_id[:8]}... "
          f"(expected {expected[:20]})")
    fresh_errors = reproduce_canary_fn(canary_video_id)
    actual = compute_canary_fingerprint(fresh_errors, canary_video_id)

    if actual == expected:
        print("[corpus_freshness] OK — corpus is fresh.")
        return True

    msg = (
        f"Corpus {corpus_path.name} is STALE.\n"
        f"  Canary fold:    {canary_video_id}\n"
        f"  Expected hash:  {expected}\n"
        f"  Actual hash:    {actual}\n"
        f"  Built at:       {meta.get('built_at')}\n"
        f"  Fix: rebuild the corpus with its builder (e.g.\n"
        f"       uv run python scripts/build_eval_reconciled_corpus.py)."
    )
    if abort_on_stale:
        raise StaleCorpusError(msg)
    print(f"[corpus_freshness] WARNING: {msg}")
    return False
