"""Upload GT videos from local MinIO to the Modal features-data volume.

Videos in the development environment live in MinIO (localhost:9000), which is
not reachable from Modal's GPUs. Before Phase 1 VideoMAE feature extraction we
stage all 68 GT-action videos to the ``rallycut-features-data`` Modal volume
at ``/videos/{content_hash}{ext}``, where the extraction function reads them.

Design:
- Idempotent: skips content-hashes already present on the volume.
- Resumable: a crash mid-run preserves uploads; next invocation continues.
- Efficient: ``modal.Volume.batch_upload()`` parallelises the upload.
- Local caching: downloaded source files land in VideoResolver's cache
  (``~/.cache/rallycut/evaluation/``) so they can be reused across runs and
  by downstream debugging. Pass ``--purge-local`` to delete the local copy
  after a successful upload (saves ~20 GB disk if tight).
- Dry-run: preview what would upload.

Usage (cd analysis):
    # Preview plan
    uv run python scripts/upload_videos_to_modal.py --dry-run

    # Smoke-test 3 videos
    uv run python scripts/upload_videos_to_modal.py --limit 3

    # Full upload (68 videos)
    uv run python scripts/upload_videos_to_modal.py

Prerequisites:
    - MinIO running locally (make dev)
    - S3_BUCKET_NAME + creds in api/.env (dev config default)
    - Modal auth: ``modal token new`` has been run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.video_resolver import VideoResolver

console = Console()

VOLUME_NAME = "rallycut-features-data"
REMOTE_VIDEO_DIR = "/videos"


def _ext_for(v: EvaluationVideo) -> str:
    return Path(v.s3_key).suffix or ".mp4"


def _remote_name(v: EvaluationVideo) -> str:
    return f"{v.content_hash}{_ext_for(v)}"


def _list_present(vol) -> set[str]:
    """Return basename set of videos already on the volume."""
    present: set[str] = set()
    try:
        for entry in vol.iterdir(REMOTE_VIDEO_DIR):
            name = Path(entry.path).name
            if name:
                present.add(name)
    except Exception:
        pass
    return present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N videos (smoke-test)")
    parser.add_argument("--videos", type=str, default=None,
                        help="Comma-separated video IDs")
    parser.add_argument("--all-videos", action="store_true",
                        help="Upload all rally-GT videos (default: only action-GT videos)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--purge-local", action="store_true",
                        help="Delete local cache copy after successful upload")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload even if already on volume")
    args = parser.parse_args()

    console.print("[bold]Loading GT videos from DB...[/bold]")
    all_videos = load_evaluation_videos(require_ground_truth=True)

    if not args.all_videos:
        from scripts.eval_action_detection import load_rallies_with_action_gt

        action_ids = {r.video_id for r in load_rallies_with_action_gt()}
        all_videos = [v for v in all_videos if v.id in action_ids]

    videos = list(all_videos)
    if args.videos:
        wanted = {x.strip() for x in args.videos.split(",") if x.strip()}
        videos = [v for v in videos if v.id in wanted]
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        console.print("[red]No videos to upload[/red]")
        sys.exit(1)
    console.print(f"  Target: {len(videos)} videos")

    # Inspect volume state
    import modal

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    present = _list_present(vol)
    console.print(f"  Volume '{VOLUME_NAME}' currently has {len(present)} file(s)")

    todo: list[EvaluationVideo] = []
    skipped: list[EvaluationVideo] = []
    for v in videos:
        if not args.force and _remote_name(v) in present:
            skipped.append(v)
        else:
            todo.append(v)

    # Probe MinIO for sizes to show the plan
    resolver = VideoResolver()
    sizes_mb: dict[str, float] = {}
    probe_fail: list[EvaluationVideo] = []
    for v in todo:
        try:
            resp = resolver.s3.head_object(Bucket=resolver.bucket_name, Key=v.s3_key)
            sizes_mb[v.id] = float(resp["ContentLength"]) / 1e6
        except Exception as e:
            probe_fail.append(v)
            console.print(
                f"  [yellow]MinIO head_object failed for {v.id[:8]} {v.filename}: {e}[/yellow]"
            )

    total_mb = sum(sizes_mb.values())
    table = Table(title="Upload Plan", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Target videos", str(len(videos)))
    table.add_row("Already on volume", str(len(skipped)))
    table.add_row("To upload", str(len(todo)))
    table.add_row("MinIO bucket", resolver.bucket_name)
    table.add_row("Modal volume", VOLUME_NAME)
    table.add_row("Total size to upload", f"{total_mb:,.0f} MB ({total_mb/1024:.1f} GB)")
    if probe_fail:
        table.add_row("MinIO probe failures", f"[red]{len(probe_fail)}[/red]")
    console.print(table)

    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN — no upload[/bold yellow]")
        for v in todo:
            size = sizes_mb.get(v.id, 0.0)
            console.print(
                f"  {v.id[:8]} {v.filename[:30]:<30} {size:>7.1f} MB "
                f"→ {_remote_name(v)}"
            )
        return

    if not todo:
        console.print("[green]All videos already staged on volume.[/green]")
        return

    if probe_fail:
        console.print(
            f"[red]Cannot proceed: {len(probe_fail)} videos failed MinIO probe.[/red]"
        )
        sys.exit(2)

    # Download + upload. VideoResolver caches locally by content_hash; we use
    # that cache as the intermediate staging area so resumes are free.
    console.print(
        f"\n[bold]Downloading {len(todo)} videos from MinIO to local cache, "
        "then uploading to Modal...[/bold]"
    )
    t_start = time.time()

    # Phase A: ensure all locals are downloaded
    local_paths: list[tuple[EvaluationVideo, Path]] = []
    for i, v in enumerate(todo, start=1):
        t0 = time.time()
        try:
            local = resolver.resolve(v.s3_key, v.content_hash)
        except Exception as e:
            console.print(
                f"  [{i}/{len(todo)}] [red]DL FAIL[/red] {v.id[:8]} {v.filename[:30]} — {e}"
            )
            continue
        local_paths.append((v, local))
        size_mb = local.stat().st_size / 1e6
        console.print(
            f"  [{i}/{len(todo)}] [green]DL[/green] {v.id[:8]} {v.filename[:30]:<30} "
            f"{size_mb:>7.1f} MB ({time.time()-t0:.1f}s)"
        )

    if not local_paths:
        console.print("[red]No local files prepared for upload.[/red]")
        sys.exit(3)

    # Phase B: upload batch via Modal Volume.batch_upload
    console.print(
        f"\n[bold]Uploading {len(local_paths)} videos to volume '{VOLUME_NAME}'...[/bold]"
    )
    t_up = time.time()
    uploaded = 0
    with vol.batch_upload() as batch:
        for v, local in local_paths:
            remote = f"{REMOTE_VIDEO_DIR}/{_remote_name(v)}"
            batch.put_file(str(local), remote)
            uploaded += 1
    # batch_upload commits on context exit
    upload_s = time.time() - t_up
    console.print(
        f"  Uploaded {uploaded} files in {upload_s/60:.1f} min "
        f"(avg {sum(p.stat().st_size for _v, p in local_paths)/1e6/upload_s:.1f} MB/s)"
    )

    # Post-verify
    present_after = _list_present(vol)
    missing_after = [v for v, _ in local_paths if _remote_name(v) not in present_after]
    if missing_after:
        console.print(f"[red]{len(missing_after)} videos not visible on volume after commit[/red]")
        for v in missing_after:
            console.print(f"  {v.id[:8]} {v.filename}")
        sys.exit(4)

    console.print(f"[green]All {len(local_paths)} videos staged on '{VOLUME_NAME}'.[/green]")

    # Optional local cleanup
    if args.purge_local:
        total_freed = 0
        for _v, p in local_paths:
            try:
                sz = p.stat().st_size
                p.unlink()
                total_freed += sz
            except Exception:
                pass
        console.print(f"[dim]Purged local cache: {total_freed/1e9:.1f} GB freed[/dim]")

    console.print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
