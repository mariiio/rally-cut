---
name: ml-experiment
description: Run ML experiments - ball tracking evaluation, filter tuning, rally detection, player tracking. Provides eval scripts, GT loading, and baseline metrics.
allowed-tools: Bash, Read, Grep, Glob
---

# ML Experiment Runner

## Process Discipline

Follow these rules for ALL eval/training/tuning runs:

1. **Before running**: Validate inputs will work first (files exist, data loads, model loads). Tell the user what you're running, how many items, and expected duration. If the script loops over items without per-item progress, ADD IT before running.
2. **While running**: Use `run_in_background: true` for anything >30s. Check output periodically. If first few items show errors or unexpected results, STOP EARLY and investigate.
3. **Output**: Every loop must print per-item results as they complete (e.g., `[3/16] rally_id: HOTA=89.0%`). Include running aggregates. Print a final summary table. Write a script file instead of long inline Python.
4. **After running**: Review output for per-item regressions and anomalies before reporting. Don't just relay the final summary.

## First Step: Load Baselines

Read `analysis/baselines.json` for current baselines, per-rally metrics, video-to-rally ID mapping, and tuned filter config. Compare all results against these.

## Ball Tracking (WASB)

WASB is the sole ball tracking model (threshold=0.3).

```bash
cd analysis

# Evaluation (16 labeled rallies, ~2-5min)
uv run python scripts/eval_wasb.py

# Ball filter tuning (grid search, ~10-30min depending on grid)
uv run rallycut evaluate-tracking tune-ball-filter --all

# Diagnostics
uv run python scripts/diagnose_ball_tracking.py --all      # Stage-by-stage pipeline metrics
uv run python scripts/audit_ball_filter.py                 # Ablation + sensitivity analysis
```

## Rally Detection

```bash
cd analysis

# Train-on-all eval (29 videos, ~2min)
uv run rallycut evaluate --temporal-maxer

# Leave-one-out CV (29-fold, ~30min — run in background)
uv run python scripts/loo_cv_temporal_maxer.py

# Boundary analysis
uv run python scripts/analyze_rally_detection.py
```

## Player Tracking

```bash
cd analysis

# Evaluate all 16 labeled rallies (~5min)
uv run rallycut evaluate-tracking --all

# Ball metrics only
uv run rallycut evaluate-tracking --all --ball-only

# Re-track and compare (~10-20min)
uv run python scripts/retrack_labeled_rallies.py --stride 2
```

## Ground Truth

```python
from rallycut.evaluation.tracking.db import load_labeled_rallies
rallies = load_labeled_rallies(rally_id="1bfcbc4f", ball_gt_only=True)
```

16 labeled rallies with validated ball GT. 29 videos with rally GT labels.

## Cached Positions

- Ball grid search raw: `~/.cache/rallycut/ball_grid_search/`

Report results as tables. Compare against baselines from `analysis/baselines.json`. Flag per-rally regressions.
