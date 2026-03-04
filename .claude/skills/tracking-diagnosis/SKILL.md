---
name: tracking-diagnosis
description: Debug ball and player tracking issues - filter pipeline reference, diagnostic scripts, known problem rallies, stage-by-stage comparison methodology.
allowed-tools: Bash, Read, Grep, Glob, Edit
---

# Tracking Diagnosis

## Process Discipline

Follow these rules for ALL diagnostic/eval runs:

1. **Before running**: Validate inputs first (rally exists, video downloadable, model loads). Tell the user what you're running and expected duration. If the script lacks per-item progress, ADD IT before running.
2. **While running**: Use `run_in_background: true` for anything >30s. Check output periodically. STOP EARLY on errors.
3. **Output**: Per-item results as they complete. Write a script file instead of long inline Python.
4. **After running**: Review output for anomalies before reporting. Check per-rally regressions.

## First Step: Load Context

Read `analysis/baselines.json` for current per-rally baselines, tuned filter config, known problem rallies, and video-to-rally ID mapping.

## Ball Filter Pipeline (in order)

1. Motion energy filter (zeroes stationary FPs, threshold=0.02)
2. Stationarity filter (consecutive frames within tight spread, off by default)
3. Exit ghost detection (on raw data)
4. Segment pruning (split at large jumps, discard short fragments, anchor recovery)
5. Exit ghost removal (applied to post-pruned data)
6. Oscillation pruning (cluster-based, disabled in tuned config)
7. Outlier removal
8. Blip removal (two-pass, compact cluster check)
9. Re-prune (clean up post-outlier shrunk segments)
10. Interpolation (fill small gaps)

## Ball Tracking Diagnostics

```bash
cd analysis
uv run python scripts/diagnose_ball_tracking.py --rally <rally-id>  # Single rally (~30s)
uv run python scripts/diagnose_ball_tracking.py --all               # All 16 rallies (~5min)
uv run python scripts/eval_wasb.py                                  # WASB eval (~2-5min)
```

## Player Tracking Diagnostics

```bash
cd analysis
uv run python scripts/retrack_labeled_rallies.py --stride 2             # All labeled rallies (~10-20min)
uv run python scripts/retrack_labeled_rallies.py --rally <id> --stride 2  # Single rally (~1-2min)
uv run python scripts/diagnose_net_interactions.py --rally <id>         # ID switch analysis at net
uv run python scripts/diagnose_yolo_detections.py <video> --start <ms> --end <ms>  # Detection check
```

## Player Tracking Pipeline

1. YOLO11s@1280 detection with ROI masking (calibration or default)
2. BoT-SORT tracking
3. Stationary background filter (spread < 0.015, presence > 80%)
4. Spatial consistency (jump splits at >0.25 displacement)
5. Color-based track repair (HSV histogram, Bhattacharyya distance)
6. Team classification
7. Global identity resolution (appearance + spatial + bbox costs)
8. Court identity resolution (side-aware, dynamic net_y)
9. Tracklet linking (team-constrained)
10. Stabilize track IDs + PlayerFilter

## Debug Methodology

### Ball
1. Load GT: `load_labeled_rallies(rally_id=..., ball_gt_only=True)`
2. Get raw positions (filtering disabled)
3. Enable filters one at a time, measure match% after each stage
4. Find the stage that drops match% or lets FPs through

### Players
1. Re-track with `--stride 2` and compare IDsw against stored baseline
2. Use `diagnose_net_interactions.py` to check if IDsw are net-related (94% are)
3. Check `--skip-global-identity` / `--skip-court-identity` flags for isolation
4. For detection issues: `diagnose_yolo_detections.py` at multiple thresholds

## Key Files

- `tracking/ball_filter.py` — ball filter pipeline
- `tracking/player_tracker.py` — player detection + tracking
- `tracking/spatial_consistency.py` — jump splitting
- `tracking/color_repair.py` — HSV-based track splitting
- `tracking/global_identity.py` — cross-team identity resolution
- `tracking/court_identity.py` — side-aware identity resolution
- `scripts/diagnose_ball_tracking.py` — ball stage-by-stage diagnostic
- `scripts/diagnose_net_interactions.py` — player ID switch analysis
- `evaluation/tracking/ball_metrics.py` — ball metrics
