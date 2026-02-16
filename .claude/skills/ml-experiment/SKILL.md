---
name: ml-experiment
description: Run ML experiments - ball tracking evaluation, filter tuning, model comparison. Provides eval scripts, GT loading, cached position paths, and baseline metrics.
allowed-tools: Bash, Read, Grep, Glob
---

# ML Experiment Runner

## First Step: Load Baselines

Read `analysis/baselines.json` for current baselines, per-rally metrics, video-to-rally ID mapping, and tuned filter config. Compare all results against these.

## Evaluation Commands

```bash
# Ball tracking (ensemble is default)
uv run python scripts/eval_ensemble.py            # 9-rally ensemble eval
uv run python scripts/eval_wasb.py                # WASB-only eval
uv run python scripts/eval_tracknet.py            # TrackNet eval

# Diagnostics
uv run python scripts/diagnose_ball_tracking.py   # Stage-by-stage pipeline metrics
uv run python scripts/audit_ball_filter.py        # Ablation + sensitivity analysis

# Filter tuning
uv run python scripts/tune_ensemble_filter.py     # Grid search ensemble filter params

# Rally detection
uv run rallycut evaluate                           # Auto-selects best pipeline
uv run rallycut evaluate --temporal-maxer          # TemporalMaxer (75% LOO F1)

# Player tracking
uv run rallycut evaluate-tracking --all            # All labeled rallies
uv run rallycut evaluate-tracking --all --ball-only  # Ball metrics only
uv run rallycut evaluate-tracking tune-ball-filter --all --grid ensemble  # Grid search
```

## Ground Truth

```python
from rallycut.evaluation.tracking.db import load_labeled_rallies
rallies = load_labeled_rallies(rally_id="1bfcbc4f", ball_gt_only=True)
```

9 labeled rallies with validated ball GT (Feb 2026).

## Cached Positions

- Ball grid search raw: `~/.cache/rallycut/ball_grid_search/`
- Ensemble grid search: `~/.cache/rallycut/ensemble_grid_search/`

Report results as tables. Compare against baselines from `analysis/baselines.json`. Note per-rally regressions. After significant improvements, suggest updating baselines.json.
