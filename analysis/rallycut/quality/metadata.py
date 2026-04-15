"""Metadata-only quality checks (duration, resolution, framerate).

These are cheap and run at upload time (no decoding required beyond ffprobe).
These are hard invariants, not calibrated thresholds: a 5-second video is too
short to find rallies regardless of how well every other check scores.

Brightness, darkness, and overexposure checks were dropped on 2026-04-15:
calibration showed zero predictive lift against the 63-video GT and the
validation fixtures (5 negatives, 2 positives) never triggered them. The
brightness metric is still computed and persisted server-side
(`api/src/services/processingService.ts::computeBrightness`) — this module
owns only the metadata invariants.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

MIN_DURATION_S = 10.0
MIN_WIDTH = 1280  # 720p
MIN_FPS = 24.0


@dataclass(frozen=True)
class VideoMetadata:
    duration_s: float
    width: int
    height: int
    fps: float

    @classmethod
    def from_ffprobe(cls, video_path: str) -> VideoMetadata:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        num, den = kv.get("r_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0.0
        return cls(
            duration_s=float(kv.get("duration", 0) or 0),
            width=int(kv.get("width", 0) or 0),
            height=int(kv.get("height", 0) or 0),
            fps=fps,
        )


def check_metadata(meta: VideoMetadata) -> CheckResult:
    issues: list[Issue] = []

    if meta.duration_s < MIN_DURATION_S:
        issues.append(Issue(
            id="video_too_short",
            tier=Tier.BLOCK,
            severity=1.0,
            message=f"Video is only {meta.duration_s:.1f}s long — we need at least {MIN_DURATION_S:.0f}s to find rallies.",
            source="upload",
            data={"durationS": meta.duration_s},
        ))

    if meta.width < MIN_WIDTH and meta.width > 0:
        issues.append(Issue(
            id="resolution_too_low",
            tier=Tier.GATE,
            severity=min(1.0, (MIN_WIDTH - meta.width) / MIN_WIDTH),
            message=f"Resolution is {meta.width}×{meta.height}. Recording in 720p or higher gives noticeably better tracking.",
            source="upload",
            data={"width": float(meta.width), "height": float(meta.height)},
        ))

    if 0 < meta.fps < MIN_FPS:
        issues.append(Issue(
            id="fps_too_low",
            tier=Tier.GATE,
            severity=min(1.0, (MIN_FPS - meta.fps) / MIN_FPS),
            message=f"Frame rate is {meta.fps:.1f} fps — {MIN_FPS:.0f} fps or higher gives better ball tracking.",
            source="upload",
            data={"fps": meta.fps},
        ))

    metrics = {
        "durationS": meta.duration_s,
        "width": float(meta.width),
        "height": float(meta.height),
        "fps": meta.fps,
    }
    return CheckResult(issues=issues, metrics=metrics)
