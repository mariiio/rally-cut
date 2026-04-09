"""Upload WASB training data to Modal with progress reporting.

Uploads CSVs + manifest first (fast), then image directories one by one
with per-directory progress.

Usage:
    uv run python scripts/upload_wasb_data.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_modal_put(src: str, dst: str) -> bool:
    """Run modal volume put, return True on success."""
    result = subprocess.run(
        ["uv", "run", "python3", "-m", "modal", "volume", "put", "-f",
         "rallycut-training", src, dst],
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0


def main() -> None:
    data_dir = Path("experiments/wasb_pseudo_labels")
    if not data_dir.exists():
        print("No data at experiments/wasb_pseudo_labels/")
        sys.exit(1)

    # Step 1: Upload manifest
    print("Uploading manifest...", flush=True)
    if not run_modal_put(str(data_dir / "manifest.json"), "wasb_data/manifest.json"):
        print("Failed to upload manifest")
        sys.exit(1)
    print("  Done", flush=True)

    # Step 2: Upload CSVs (batch)
    csvs = sorted(data_dir.glob("*.csv"))
    print(f"Uploading {len(csvs)} CSVs...", flush=True)
    # Upload all CSVs in one directory put
    csv_tmp = Path("/tmp/wasb_csv_upload")
    csv_tmp.mkdir(exist_ok=True)
    for c in csvs:
        (csv_tmp / c.name).unlink(missing_ok=True)
        (csv_tmp / c.name).symlink_to(c.resolve())
    if not run_modal_put(str(csv_tmp) + "/", "wasb_data/"):
        print("Failed to upload CSVs")
        sys.exit(1)
    print(f"  Done ({len(csvs)} CSVs)", flush=True)

    # Step 3: Upload image directories one by one with progress
    img_dir = data_dir / "images"
    dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    total = len(dirs)
    print(f"\nUploading {total} image directories...", flush=True)

    t0 = time.time()
    for i, d in enumerate(dirs):
        n_files = len(list(d.iterdir()))
        src = str(d) + "/"
        dst = f"wasb_data/images/{d.name}/"

        ok = run_modal_put(src, dst)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0

        status = "OK" if ok else "FAILED"
        print(
            f"  [{i + 1}/{total}] {d.name[:10]}... "
            f"({n_files} files) {status} "
            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}m remaining]",
            flush=True,
        )

        if not ok:
            print(f"  WARNING: Failed to upload {d.name}, continuing...")

    total_time = time.time() - t0
    print(f"\nDone! {total} directories in {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
