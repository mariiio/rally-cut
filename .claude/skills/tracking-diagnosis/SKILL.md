---
name: tracking-diagnosis
description: Debug ball/player tracking issues - filter pipeline reference, diagnostic scripts, known problem rallies, stage-by-stage comparison methodology.
allowed-tools: Bash, Read, Grep, Glob, Edit
---

# Tracking Diagnosis

## First Step: Load Context

Read `analysis/baselines.json` for current per-rally baselines, tuned filter config parameters, known problem rallies, and video-to-rally ID mapping.

## Ball Filter Pipeline (in order)

1. Motion energy filter (zeroes stationary FPs)
2. Stationarity filter (consecutive frames within tight spread, off by default)
3. Exit ghost detection (on raw data)
4. Segment pruning (split at large jumps, discard short fragments, anchor recovery)
5. Exit ghost removal (applied to post-pruned data)
6. Oscillation pruning (cluster-based, disabled in ensemble tuned config)
7. Outlier removal
8. Blip removal (two-pass, compact cluster check)
9. Re-prune (clean up post-outlier shrunk segments)
10. Interpolation (fill small gaps)

Threshold values are in `analysis/baselines.json` under `filter_config_tuned`.

## Diagnostic Script

```bash
uv run python scripts/diagnose_ball_tracking.py --rally <rally-id>
uv run python scripts/diagnose_ball_tracking.py --all
```

## Debug Methodology

1. Load GT: `load_labeled_rallies(rally_id=..., ball_gt_only=True)`
2. Get raw: `tracker.track_video(path, enable_filtering=False, preserve_raw=True)`
3. Enable filters one at a time, measure match% after each stage
4. Find the stage that drops match% or lets FPs through
5. Check spatial patterns: edge (ghost)? player position (motion/blip)? oscillating?
6. Frame numbers are rally-relative (0-indexed)

## Source-Aware (ensemble)

WASB positions: `motion_energy >= 1.0` (sentinel). Protected from outlier/blip/oscillation removal. Halved min_segment_frames, wider recovery proximity (75%).

## Key Files

- `tracking/ball_filter.py` — main filter pipeline
- `tracking/ball_ensemble.py` — ensemble tracker
- `scripts/diagnose_ball_tracking.py` — stage-by-stage diagnostic
- `evaluation/tracking/ball_metrics.py` — metrics
