"""Extract stride-parameterized VideoMAE features for all GT videos on Modal.

Phase 1 of the VideoMAE contact validation plan.

Design:
- Idempotent: skips videos whose features are already in the local
  ``FeatureCache`` (``~/.cache/rallycut/features/``). Re-runs resume cleanly.
- Recoverable: each video is an independent Modal call. A crash in the local
  driver preserves all prior extractions; the next run only processes
  whatever's still missing.
- Apples-to-apples: writes to the same cache format and key the local LOO
  training scripts read from. No divergence between extraction + consumption.
- Efficient: Modal T4 GPU ($0.59/hr) at batch_size=16; optional parallel
  dispatch via ``modal.Function.starmap``. Atomic writes (tmp → rename) so
  an interrupted write never corrupts the cache.
- Safe by default: pre-flight validates DB, S3 credentials, Modal auth, and
  cache dir writability BEFORE spending any GPU time. Early-aborts after
  ``--early-abort-failures`` consecutive failures so bugs don't burn the
  whole budget silently.

Before running:
    # Deploy the Modal app once (idempotent, fast)
    modal deploy analysis/rallycut/service/platforms/modal_features.py

Usage (cd analysis):
    # Smoke-test on 3 videos first — validates the full path end-to-end
    uv run python scripts/extract_contact_features.py --limit 3

    # Full run on all 68 GT videos
    uv run python scripts/extract_contact_features.py

    # Parallel dispatch (faster wall-clock, same cost)
    uv run python scripts/extract_contact_features.py --parallel 4

    # Preview without running
    uv run python scripts/extract_contact_features.py --dry-run

    # Specific videos
    uv run python scripts/extract_contact_features.py --videos <id1>,<id2>

    # Force re-extraction (default: skip cached)
    uv run python scripts/extract_contact_features.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.temporal.features import FeatureCache, FeatureMetadata

console = Console()

DEFAULT_STRIDE = 4
DEFAULT_BACKBONE = "videomae-v1"
EST_MIN_PER_VIDEO = 10  # ballpark for budget display only
T4_HOURLY_USD = 0.59


def _cache_key(content_hash: str, stride: int, backbone: str) -> str:
    """Match ``rallycut.temporal.features._compute_cache_key`` verbatim."""
    key_data = f"{content_hash}:stride={stride}:backbone={backbone}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _list_volume_videos() -> set[str]:
    """Return the set of ``{content_hash}{ext}`` file names currently on the
    ``rallycut-features-data`` volume under ``/videos/``. Empty set if the
    volume is empty or missing.
    """
    import modal

    vol = modal.Volume.from_name("rallycut-features-data", create_if_missing=True)
    present: set[str] = set()
    try:
        for entry in vol.iterdir("/videos"):
            name = Path(entry.path).name
            if name:
                present.add(name)
    except Exception:
        # Volume may be empty (iterdir raises on missing dir). Treat as empty.
        pass
    return present


def _preflight(videos: list[EvaluationVideo]) -> None:
    """Validate environment before any GPU time is spent.

    Checks (fails loudly if any fail):
      1. Local FeatureCache dir is writable.
      2. All candidate videos have ``content_hash`` + ``s3_key``.
      3. Modal auth works (``ping`` function responds).
      4. Every target video has been uploaded to the Modal data volume.
    """
    cache = FeatureCache()
    if not os.access(cache.cache_dir, os.W_OK):
        raise RuntimeError(f"Feature cache dir not writable: {cache.cache_dir}")

    missing_meta = [v for v in videos if not v.s3_key or not v.content_hash]
    if missing_meta:
        sample = ", ".join(v.id[:8] for v in missing_meta[:5])
        raise RuntimeError(
            f"{len(missing_meta)} videos missing s3_key or content_hash (first few: {sample})"
        )

    try:
        import modal

        ping_fn = modal.Function.from_name("rallycut-features", "ping")
        result = ping_fn.remote()
    except Exception as e:
        raise RuntimeError(
            f"Modal ping failed: {e}. Has the app been deployed? Run: "
            "modal deploy rallycut/service/platforms/modal_features.py"
        ) from e

    if result.get("status") != "ok":
        raise RuntimeError(f"Modal ping returned non-ok status: {result}")
    console.print("[dim]Modal auth: OK[/dim]")

    # Verify every candidate video has been uploaded to the data volume.
    present = _list_volume_videos()
    missing_on_vol = []
    for v in videos:
        ext = Path(v.s3_key).suffix or ".mp4"
        if f"{v.content_hash}{ext}" not in present:
            # Also allow any ext match (upload may have preserved a different suffix)
            if not any(name.startswith(v.content_hash + ".") for name in present):
                missing_on_vol.append(v)

    if missing_on_vol:
        sample = ", ".join(
            f"{v.id[:8]}/{v.filename[:20]}" for v in missing_on_vol[:5]
        )
        raise RuntimeError(
            f"{len(missing_on_vol)}/{len(videos)} videos not on Modal volume "
            f"'rallycut-features-data'. First few: {sample}. Run: "
            "uv run python scripts/upload_videos_to_modal.py"
        )
    console.print(f"[dim]Modal volume: {len(videos)} videos staged[/dim]")


def _save_features_atomic(
    cache: FeatureCache,
    content_hash: str,
    stride: int,
    backbone: str,
    features: np.ndarray,
    metadata_dict: dict,
) -> Path:
    """Write ``.npy`` + ``.json`` atomically into the local FeatureCache.

    Uses tmp-then-rename so an interrupted write cannot corrupt the cache.
    Returns the final features path.
    """
    cache_key = _cache_key(content_hash, stride, backbone)
    feat_path = cache.cache_dir / f"{cache_key}.npy"
    meta_path = cache.cache_dir / f"{cache_key}.json"
    tmp_feat = feat_path.with_name(feat_path.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")

    # Materialize FeatureMetadata then re-dump — catches schema drift early.
    metadata = FeatureMetadata(**metadata_dict)
    # Use an open file handle; bare np.save() appends ".npy" to path strings
    # that don't end in ".npy", which breaks our tmp→final rename scheme.
    with open(tmp_feat, "wb") as f:
        np.save(f, features)
    tmp_meta.write_text(json.dumps(asdict(metadata), indent=2))
    tmp_feat.rename(feat_path)
    tmp_meta.rename(meta_path)
    return feat_path


def _filter_needed(
    videos: list[EvaluationVideo],
    cache: FeatureCache,
    stride: int,
    backbone: str,
    force: bool,
) -> tuple[list[EvaluationVideo], list[EvaluationVideo]]:
    """Split videos into (already-cached-locally, to-extract)."""
    already: list[EvaluationVideo] = []
    todo: list[EvaluationVideo] = []
    for v in videos:
        if not force and cache.has(v.content_hash, stride=stride, backbone=backbone):
            already.append(v)
        else:
            todo.append(v)
    return already, todo


def _print_plan(
    total: int,
    already: int,
    todo: int,
    stride: int,
    backbone: str,
    parallel: int,
) -> None:
    table = Table(title="Extraction Plan", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total GT videos", str(total))
    table.add_row("Already cached locally", str(already))
    table.add_row("To extract on Modal", str(todo))
    table.add_row("Stride", str(stride))
    table.add_row("Backbone", backbone)
    table.add_row("Parallel workers", str(parallel))
    est_min = todo * EST_MIN_PER_VIDEO / max(1, parallel)
    est_cost = todo * EST_MIN_PER_VIDEO / 60 * T4_HOURLY_USD
    table.add_row("Est. wall-clock", f"~{est_min:.0f} min")
    table.add_row("Est. Modal cost", f"~${est_cost:.2f}")
    console.print(table)


def _iter_extract_one(
    videos: list[EvaluationVideo],
    stride: int,
    backbone: str,
    parallel: int,
) -> Iterator[tuple[EvaluationVideo, dict]]:
    """Dispatch Modal extractions and yield ``(video, result)`` pairs.

    In sequential mode (parallel=1) results arrive in video order. In parallel
    mode, results arrive in **completion order** (``order_outputs=False``) so
    one slow video (e.g. IMG_2313.MOV's 3.3 GB original) doesn't block the
    display or cache-writing of videos that finished ahead of it. We match
    each result back to its source video by the ``content_hash`` field the
    Modal function echoes back.
    """
    import modal

    extract_fn = modal.Function.from_name("rallycut-features", "extract_features_remote")

    def _ext(v: EvaluationVideo) -> str:
        return Path(v.s3_key).suffix or ".mp4"

    if parallel <= 1:
        for v in videos:
            result = extract_fn.remote(
                content_hash=v.content_hash,
                ext=_ext(v),
                stride=stride,
                backbone=backbone,
            )
            yield v, result
        return

    # Completion-order dispatch: avoid head-of-line blocking on long videos.
    # - order_outputs=False → results yield in completion order.
    # - return_exceptions=True → one failing input does NOT abort the map; we
    #   get the exception back as a value and record it as a per-video failure.
    by_hash = {v.content_hash: v for v in videos}
    remaining = list(videos)
    args_iter = [(v.content_hash, _ext(v), stride, backbone) for v in videos]
    for result in extract_fn.starmap(
        args_iter, order_outputs=False, return_exceptions=True,
    ):
        # Normalize Exception objects into the same dict shape our caller
        # expects, so the progress reporter + loop don't have to special-case.
        if isinstance(result, BaseException):
            result = {
                "ok": False,
                "content_hash": None,
                "error": f"{type(result).__name__}: {result}",
            }

        ch = result.get("content_hash") if isinstance(result, dict) else None
        v = by_hash.get(ch) if ch else None
        if v is None:
            # starmap yields in completion order but doesn't tell us which
            # input produced the exception when return_exceptions=True.
            # Consume one unclaimed video so progress still advances; the
            # user can diagnose via the error string.
            v = remaining[0]
        by_hash.pop(v.content_hash, None)
        try:
            remaining.remove(v)
        except ValueError:
            pass
        yield v, result


def _report(
    idx: int,
    total: int,
    v: EvaluationVideo,
    result: dict,
) -> bool:
    """Print per-video progress. Returns True if extraction succeeded."""
    ok = result.get("ok", False)
    name = f"{v.id[:8]} {v.filename[:44]:<44}"
    if not ok:
        err = result.get("error", "unknown")
        console.print(f"  [{idx}/{total}] [red]FAILED[/red] {name} — {err}")
        return False

    nw = result.get("num_windows", "?")
    extract_s = result.get("extract_s", 0.0)
    locate_s = result.get("locate_s", 0.0)
    console.print(
        f"  [{idx}/{total}] [green]OK[/green]     {name} "
        f"→ {nw} windows (locate={locate_s:.1f}s ext={extract_s:.0f}s)"
    )
    return True


def _verify_random(
    videos: list[EvaluationVideo],
    cache: FeatureCache,
    stride: int,
    backbone: str,
    n: int = 3,
) -> int:
    """Sanity-check N random cached features load correctly. Returns #failures."""
    import random

    sample = random.sample(videos, min(n, len(videos)))
    fails = 0
    for v in sample:
        cached = cache.get(v.content_hash, stride=stride, backbone=backbone)
        if cached is None:
            console.print(f"  [red]FAIL[/red] {v.id[:8]}: not in local cache")
            fails += 1
            continue
        feats, meta = cached
        shape_ok = feats.ndim == 2 and feats.shape[1] == 768
        meta_ok = meta.num_windows == feats.shape[0]
        if shape_ok and meta_ok:
            console.print(
                f"  [green]OK[/green]   {v.id[:8]}: shape={feats.shape}, "
                f"fps={meta.fps:.1f}, backbone={meta.backbone}"
            )
        else:
            console.print(
                f"  [red]FAIL[/red] {v.id[:8]}: shape={feats.shape}, "
                f"meta.num_windows={meta.num_windows}"
            )
            fails += 1
    return fails


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N videos (smoke-test)")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video IDs to extract")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent Modal calls (1 = sequential)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the plan without invoking Modal")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached locally")
    parser.add_argument("--early-abort-failures", type=int, default=2,
                        help="Abort after N consecutive failures (default 2)")
    parser.add_argument("--all-videos", action="store_true",
                        help="Extract all GT videos (default: only videos with "
                             "action_ground_truth_json — 68 of 69)")
    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be >= 1")
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    console.print("[bold]Loading GT videos from DB...[/bold]")
    all_videos = load_evaluation_videos(require_ground_truth=True)
    console.print(f"  {len(all_videos)} videos with rally GT")

    # Phase 1 cares about videos with action ground truth (the 68-video LOO
    # corpus). Default to that set so we don't waste extractions on videos the
    # VideoMAE contact/action head will never train or eval on.
    if not args.all_videos:
        from scripts.eval_action_detection import load_rallies_with_action_gt

        action_video_ids = {r.video_id for r in load_rallies_with_action_gt()}
        all_videos = [v for v in all_videos if v.id in action_video_ids]
        console.print(
            f"  {len(all_videos)} videos have action ground truth "
            "(pass --all-videos to include the rest)"
        )

    videos = list(all_videos)
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",") if v.strip()}
        videos = [v for v in videos if v.id in wanted]
        missing = wanted - {v.id for v in videos}
        if missing:
            console.print(
                f"[yellow]Warning: {len(missing)} requested IDs not found: "
                f"{sorted(missing)[:5]}[/yellow]"
            )

    if args.limit:
        videos = videos[: args.limit]

    if not videos:
        console.print("[red]No videos to process[/red]")
        sys.exit(1)

    cache = FeatureCache()
    already, todo = _filter_needed(
        videos, cache, args.stride, args.backbone, args.force
    )

    _print_plan(len(videos), len(already), len(todo), args.stride, args.backbone, args.parallel)

    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN — no extraction[/bold yellow]")
        for v in todo:
            ck = _cache_key(v.content_hash, args.stride, args.backbone)
            console.print(f"  would extract: {v.id[:8]} {v.filename[:40]:<40} → {ck}")
        return

    if not todo:
        console.print("[green]All videos already cached — nothing to do.[/green]")
        return

    # Pre-flight AFTER planning so --dry-run works without Modal auth.
    console.print("\n[bold]Pre-flight checks...[/bold]")
    try:
        _preflight(todo)
    except Exception as e:
        console.print(f"[red]Pre-flight failed: {e}[/red]")
        sys.exit(2)

    # Extract loop
    console.print(
        f"\n[bold]Extracting {len(todo)} videos "
        f"(parallel={args.parallel}, stride={args.stride})...[/bold]"
    )
    t_start = time.time()
    n_ok = n_fail = 0
    consecutive_failures = 0
    failures: list[tuple[EvaluationVideo, dict]] = []

    try:
        for idx, (v, result) in enumerate(
            _iter_extract_one(todo, args.stride, args.backbone, args.parallel),
            start=1,
        ):
            ok = _report(idx, len(todo), v, result)
            if ok:
                # Write to local cache atomically — this is the source of truth
                # for consumers. Volume coordination is avoided by design.
                try:
                    _save_features_atomic(
                        cache,
                        v.content_hash,
                        args.stride,
                        args.backbone,
                        result["features"],
                        result["metadata"],
                    )
                except Exception as e:
                    console.print(
                        f"  [red]local-save FAILED[/red] {v.id[:8]} — {e}"
                    )
                    failures.append((v, {"error": f"local-save: {e}"}))
                    n_fail += 1
                    consecutive_failures += 1
                else:
                    n_ok += 1
                    consecutive_failures = 0
            else:
                n_fail += 1
                consecutive_failures += 1
                failures.append((v, result))

            if (
                args.parallel == 1
                and consecutive_failures >= args.early_abort_failures
            ):
                console.print(
                    f"\n[red]Aborting after {consecutive_failures} consecutive "
                    f"failures (--early-abort-failures={args.early_abort_failures}).[/red]"
                )
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Restart to resume.[/yellow]")

    elapsed = time.time() - t_start

    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Wall-clock:  {elapsed/60:.1f} min")
    console.print(f"  OK:          {n_ok}")
    console.print(f"  Failed:      {n_fail}")
    if n_fail:
        console.print("\n[red]Failures:[/red]")
        for v, r in failures:
            console.print(f"  {v.id[:8]} {v.filename[:40]:<40} — {r.get('error', 'unknown')}")

    # Verify
    if n_ok > 0:
        console.print("\n[bold]Verifying random samples...[/bold]")
        completed = [v for v in todo if cache.has(v.content_hash, stride=args.stride, backbone=args.backbone)]
        vfails = _verify_random(completed, cache, args.stride, args.backbone, n=3)
        if vfails:
            console.print(f"[red]{vfails} verification failure(s)[/red]")
            sys.exit(4)

    if n_fail:
        sys.exit(3)


if __name__ == "__main__":
    main()
