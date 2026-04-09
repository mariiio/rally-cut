"""Fast WASB data upload to Modal using batch_upload API.

Usage:
    uv run python scripts/upload_wasb_fast.py
"""

import time
from pathlib import Path

import modal


def main() -> None:
    data_dir = Path("experiments/wasb_pseudo_labels")
    vol = modal.Volume.from_name("rallycut-training")

    img_dir = data_dir / "images"
    img_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    print(f"Uploading: 208 CSVs + manifest + {len(img_dirs)} image dirs", flush=True)

    t0 = time.time()
    with vol.batch_upload(force=True) as batch:
        # Manifest
        batch.put_file(str(data_dir / "manifest.json"), "wasb_data/manifest.json")

        # CSVs
        for csv in sorted(data_dir.glob("*.csv")):
            batch.put_file(str(csv), f"wasb_data/{csv.name}")

        print(f"  CSVs queued ({time.time() - t0:.0f}s)", flush=True)

        # Image directories
        for i, d in enumerate(img_dirs):
            batch.put_directory(str(d), f"wasb_data/images/{d.name}")
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(img_dirs)}] dirs queued ({elapsed:.0f}s)", flush=True)

        print(f"  All {len(img_dirs)} dirs queued, uploading...", flush=True)

    total = time.time() - t0
    print(f"\nDone! Upload completed in {total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
