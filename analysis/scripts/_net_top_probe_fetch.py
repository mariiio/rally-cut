"""Robust MinIO video fetcher shared by `probe_X_n*` net-top probes.

Solves two issues discovered in the first N1 run:

1. Some `Video` rows have NULL `original_s3_key` and `proxy_s3_key`
   even though the file exists in MinIO under a non-default
   `user_id` prefix. Resolving by `<vid>` alone (across any user
   prefix) recovers these orphans.

2. Local cache keyed on `<name>.mp4` collides on duplicate-name DB
   rows (e.g. two `yaya` Video rows with different fps and different
   GT but the same shared corners). Keying on `<vid>` disambiguates.

Public API:
  build_minio_index() -> dict[vid, list[s3 path]]
  fetch_video(vid, name, minio_index, local_dir) -> Path | None
"""
from __future__ import annotations

import subprocess
from pathlib import Path

MINIO_ENDPOINT = "http://localhost:9000"
BUCKET = "rallycut-dev"
AWS_ENV = {
    "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
}

# Priority order for which copy of a video to download per `vid`. Lower
# index = preferred. Proxy is smallest (~20-30 MB) and fastest to
# process; original is largest. Posters and JSON sidecars are excluded.
# `.MOV` covers raw iPhone footage that never got processed into a proxy
# (rare, but a handful of orphan records have only this).
_PRIORITY_SUFFIXES = (
    "_proxy.mp4", "_optimized.mp4", ".mp4",
    ".mov", ".MOV",
)


def build_minio_index() -> dict[str, list[str]]:
    """Run `aws s3 ls --recursive` once and bucket paths by `vid` (UUID).

    Returns {vid: [s3_path, ...]} for every `videos/<user>/<vid>/<file>`
    entry. Each list is sorted in `_PRIORITY_SUFFIXES` order so the
    fetch helper can take the first match without further filtering.
    """
    proc = subprocess.run(
        ["aws", "s3", "ls", f"s3://{BUCKET}/", "--recursive",
         "--endpoint-url", MINIO_ENDPOINT],
        env=AWS_ENV,
        capture_output=True, text=True, check=True,
    )
    index: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines():
        # Format: "<date> <time> <size> <path>"
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        path = parts[3]
        if not path.startswith("videos/"):
            continue
        segs = path.split("/", 3)
        if len(segs) < 4:
            continue
        # segs = ["videos", "<user_id>", "<vid>", "<filename>"]
        vid = segs[2]
        # Skip anything that's not a real video (poster .jpg, json, etc.)
        if not any(path.endswith(s) for s in _PRIORITY_SUFFIXES):
            continue
        index.setdefault(vid, []).append(path)
    # Sort each vid's paths by suffix priority
    def _rank(p: str) -> int:
        for i, sfx in enumerate(_PRIORITY_SUFFIXES):
            if p.endswith(sfx):
                return i
        return len(_PRIORITY_SUFFIXES)
    for vid in index:
        index[vid].sort(key=_rank)
    return index


def fetch_video(
    vid: str,
    name: str,
    minio_index: dict[str, list[str]],
    local_dir: Path,
    *,
    db_proxy_key: str | None = None,
    db_original_key: str | None = None,
) -> Path | None:
    """Resolve the best MinIO file for `vid` and download to a
    vid-keyed local path. Returns the local path on success.

    Resolution order:
      1. `db_proxy_key` (if non-empty)
      2. `db_original_key` (if non-empty)
      3. MinIO index lookup by `vid`, taking the highest-priority
         path (proxy > optimized > original).

    Local cache filename is `<vid>__<name>.mp4` — `vid` disambiguates
    duplicate-name rows, `name` is purely for human inspection.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    out = local_dir / f"{vid}__{name}.mp4"
    if out.exists() and out.stat().st_size > 1_000_000:
        return out

    candidates: list[str] = []
    if db_proxy_key:
        candidates.append(db_proxy_key)
    if db_original_key:
        candidates.append(db_original_key)
    candidates.extend(minio_index.get(vid, []))

    for s3_path in candidates:
        src = f"s3://{BUCKET}/{s3_path}"
        try:
            subprocess.run(
                ["aws", "s3", "cp", src, str(out),
                 "--endpoint-url", MINIO_ENDPOINT],
                env=AWS_ENV,
                check=True, capture_output=True,
            )
            if out.exists() and out.stat().st_size > 1_000_000:
                return out
        except subprocess.CalledProcessError:
            continue
    return None
