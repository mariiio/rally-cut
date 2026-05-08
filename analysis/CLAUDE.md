# RallyCut Analysis

Volleyball video analysis CLI. Uses ML (VideoMAE) to detect game states and remove dead time.

## Stack

- Python 3.11+, uv package manager
- PyTorch + Transformers (VideoMAE)
- Typer CLI, Pydantic config, FFmpeg, OpenCV

## Commands

> All shell commands in this file assume you are in the `analysis/` directory (`cd analysis` from repo root). Paths like `scripts/...`, `weights/...`, and `reports/...` are relative to `analysis/`.

```bash
# Core commands (TemporalMaxer is default when model+features exist)
uv run rallycut cut <video.mp4>                 # Auto-selects best pipeline
uv run rallycut cut <video.mp4> --temporal-maxer # Force TemporalMaxer (95% LOO F1)
uv run rallycut cut <video.mp4> --heuristics    # Force heuristics (57% F1)
uv run rallycut cut <video.mp4> --model beach   # Use beach volleyball model
uv run rallycut profile <video.mp4>             # Performance profiling

# Quality checks (A1 ship)
uv run rallycut preflight <video.mp4>           # Full preflight: metadata invariants + camera geometry. Emits JSON QualityReport. (Brightness, camera-distance, crowd, shakiness, tilt, dark/overexposed dropped on 2026-04-15 after validation — see reports/quality_calibration_2026-04-14.json.)
uv run rallycut preview-check <frames-dir>      # Lightweight pre-upload check on a directory of JPEG frames. Used by the web client-side upload gate. Runs court-keypoint detection + camera_geometry + CLIP beach-VB classifier (open-clip ViT-B/32 via [preflight] extra).
uv run rallycut tilt-detect <frames-dir>        # Emit {tiltDeg, courtConfidence, framesScored} from a directory of JPEGs. Used server-side at POST /v1/videos/:id/confirm to decide auto-rotate.

# Pre-extract features (optional — TemporalMaxer auto-extracts on first run)
uv run rallycut train extract-features --stride 12

# Useful options for cut
uv run rallycut cut video.mp4 --debug           # Timeline visualization + diagnostics
uv run rallycut cut video.mp4 --profile         # Performance breakdown
uv run rallycut cut video.mp4 --json            # Export segments as JSON
uv run rallycut cut video.mp4 --segments s.json # Load pre-computed segments
uv run rallycut cut video.mp4 --limit 60        # Analyze first 60s only
uv run rallycut cut video.mp4 --no-ball-boundary  # Disable ball-based boundary refinement (on by default)

# Training (beach model fine-tuning) - FULL WORKFLOW
rm -rf training_data/                                 # Clean old local data
uv run rallycut train export-dataset --name beach_v1  # Export labeled data from DB
uv run rallycut train prepare                         # Generate samples + extract frames (~6.6GB)
modal volume rm -r rallycut-training training_data/   # Clean old Modal data (IMPORTANT!)
modal volume put -f rallycut-training training_data/ training_data/  # Upload fresh data
uv run rallycut train modal --epochs 30 --fresh       # Train on T4 GPU (~4hrs, ~$0.59/hr)
uv run rallycut train modal --download                # Download trained model
uv run rallycut train modal --cleanup                 # Delete from Modal (~$0.75/GB/mo)

# Training optimizations (applied by default):
# - 2-class training (SERVICE merged into PLAY) - reduces class confusion
# - Video-level train/val split prevents data leakage (honest validation)
# - 9/12 encoder layers frozen to prevent catastrophic forgetting
# - Lower learning rate (5e-6) for stable fine-tuning
# - Pre-extracted frames for 15x faster training (~1.4s/step vs 20s/step)
# - Use --freeze-layers 0 and --lr 1e-5 for full fine-tuning (not recommended)

# Adding new labeled videos (RECOMMENDED WORKFLOW)
# Step 1: Export and backup
uv run rallycut train export-dataset --name beach_v3  # Export all labeled data from DB
uv run rallycut train push --name beach_v3            # Back up to S3 (deduplicates videos)

# Step 2a: Retrain TemporalMaxer (fast, local, recommended first)
uv run rallycut train extract-features --stride 12    # Extract features for new videos
uv run rallycut train temporal-maxer --epochs 50      # Retrain TemporalMaxer

# Step 2b: Fine-tune VideoMAE (slow, GPU required, only if needed)
rm -rf training_data/
uv run rallycut train prepare                         # Generate samples from all videos
modal volume rm -r rallycut-training training_data/   # Clean old Modal data
modal volume put -f rallycut-training training_data/ training_data/  # Upload fresh
uv run rallycut train modal --upload-model            # Upload existing model weights
uv run rallycut train modal --resume-from-model --lr 1e-6 --epochs 10 --fresh
uv run rallycut train modal --download --cleanup      # Download and clean up

# S3 backup/restore (survives DB resets and MinIO clears)
# Dataset export includes these per-video/per-rally artifacts when present:
#   ground_truth.json                   (always) — rally bounds
#   tracking_ground_truth.json          (opt)    — player/ball positions
#   action_ground_truth.json            (opt)    — serve/receive/set/attack/dig/block
#   player_matching_ground_truth.json   (opt)    — cross-rally player IDs + side switches
#   score_ground_truth.json             (opt)    — gt_serving_team + gt_side_switch
# Optional GT files match rallies by (content_hash, start_ms, end_ms) for
# resilience against rally UUID churn on restore. Score GT restore is
# per-field NULL-only so re-running never clobbers newer DB labels.
uv run rallycut train push --name beach_v2            # Upload dataset to S3
uv run rallycut train pull --name beach_v2            # Download dataset from S3
uv run rallycut train restore --name beach_v2         # Re-import ground truth into DB
uv run rallycut train restore --name beach_v2 --upload-to-app-s3  # + upload to MinIO
uv run rallycut train restore --name beach_v2 --dry-run  # Preview without changes
uv run rallycut train list-remote                     # List datasets backed up in S3

# Model weight backup (survives local machine loss and Modal volume clears)
uv run rallycut train push-weights                      # Push trained weights (latest)
uv run rallycut train push-weights --name pre-retrain   # Named snapshot + update latest
uv run rallycut train push-weights --include-external   # Also backup external weights
uv run rallycut train push-weights --dry-run            # Preview what would upload
uv run rallycut train pull-weights                      # Pull latest weights
uv run rallycut train pull-weights --name pre-retrain   # Pull specific snapshot
uv run rallycut train list-weights                      # List snapshots in S3

# Training is preemption-resilient:
# - Auto-retries up to 2 times on GPU preemption
# - Checkpoints saved every 100 steps (~5 min max loss)
# - Resumes from latest checkpoint automatically

# WASB HRNet fine-tuning (improve ball detection on beach volleyball)
# IMPORTANT: Always re-export before retraining. The export writes a manifest.json
# that lists GT rally IDs — training uses it to hold GT rallies out of the training
# split. Stale exports will cause the upload step to error (missing manifest).
uv run python -m experiments.pseudo_label_export \
    --output-dir experiments/wasb_pseudo_labels \
    --all-tracked --extract-frames                    # Export pseudo-labels + manifest
uv run rallycut train wasb-modal --upload           # Upload data + pretrained weights to Modal
uv run rallycut train wasb-modal --epochs 30 --fresh  # Train on A10G GPU (~$1.10/hr)
uv run rallycut train wasb-modal --download         # Download fine-tuned model (auto-deletes ONNX)
uv run python scripts/eval_wasb.py                  # Evaluate WASB
uv run rallycut train wasb-modal --cleanup          # Delete from Modal volume

# TemporalMaxer training (95% LOO F1 at IoU=0.4)
uv run rallycut train export-dataset --name beach_v3  # Export labeled data
uv run rallycut train extract-features --stride 12    # Extract VideoMAE features
uv run rallycut train temporal-maxer --epochs 50      # Train TemporalMaxer TAS model
# Model saved to weights/temporal_maxer/best_temporal_maxer.pt

# Evaluation
uv run rallycut evaluate                              # Evaluate (auto-selects TemporalMaxer)
uv run rallycut evaluate --temporal-maxer             # Force TemporalMaxer (95% LOO F1)
uv run rallycut evaluate --heuristics                 # Force heuristics evaluation (57% F1)
uv run rallycut evaluate --model beach --iou 0.5      # Evaluate beach model

# Stratified evaluation (by video characteristics)
uv run python scripts/eval_stratified.py             # Group metrics by brightness/camera/complexity
uv run python scripts/eval_stratified.py --ball-only # Ball metrics only

# Standalone subcommands (used by api/services)
uv run rallycut detect-court <video>                     # Court-keypoint detection standalone
uv run rallycut compute-match-stats <video-id>           # Aggregate match stats from DB
uv run rallycut reinterpolate-primary <video-id>         # Retro-fix stale primary-track interpolation
uv run rallycut analyze actions <video-id>               # Action-classifier subcommand group
uv run rallycut analyze highlights <video-id>            # Highlight detection subcommand group

# Cross-rally player matching & post-processing
uv run rallycut match-players <video-id>                 # Assign consistent player IDs 1-4
uv run rallycut match-players <video-id> -o result.json  # Export assignments to JSON
uv run rallycut match-players <video-id> --num-samples 20  # More frames per track (default: 12)
uv run rallycut match-players <video-id> -q              # Quiet mode
uv run rallycut repair-identities <video-id>             # Fix within-rally ID switches using match profiles
uv run rallycut repair-identities <video-id> --dry-run   # Preview repairs without updating DB
uv run rallycut remap-track-ids <video-id>               # Remap stored track IDs to player IDs
uv run rallycut remap-track-ids <video-id> --dry-run     # Preview changes without updating DB
uv run rallycut reattribute-actions <video-id>           # Re-attribute actions using match-level teams (default --min-confidence 0.70, aligned with eval)
uv run rallycut reattribute-actions <video-id> --min-confidence 0.90  # Only re-attribute on the highest-confidence rallies

# Development
uv run pytest tests                    # Run tests (excludes slow ML tests)
uv run pytest tests --run-slow         # Include slow ML inference tests
uv run mypy rallycut/                  # Type check
uv run ruff check rallycut/            # Lint
```

## Structure

Tree below is rooted at `analysis/`:

```
analysis/
├── rallycut/
│   ├── cli/commands/    # Typer commands (cut, profile, train, evaluate)
│   ├── core/            # Config, models, Video wrapper, caching, profiler
│   ├── analysis/        # GameStateAnalyzer (VideoMAE ML classifier)
│   ├── processing/      # VideoCutter, FFmpegExporter
│   ├── temporal/        # TemporalMaxer rally detection pipeline
│   │   ├── temporal_maxer/       # TemporalMaxer TAS model
│   │   │   ├── model.py          # Multi-scale temporal pyramid architecture
│   │   │   ├── training.py       # Sequence-level training loop
│   │   │   └── inference.py      # Inference and segment extraction
│   │   ├── features.py           # VideoMAE feature extraction and caching
│   │   └── inference.py          # Segment extraction and anti-overmerge
│   ├── tracking/        # Ball and player tracking
│   │   ├── ball_tracker.py       # Ball tracker factory and data types
│   │   ├── ball_filter.py        # Temporal filter for ball tracking post-processing
│   │   ├── ball_features.py      # Ball features, server ID, reactivity scoring
│   │   ├── player_tracker.py     # YOLO + BoT-SORT player tracking
│   │   ├── player_filter.py      # Multi-stage player filtering with court/ball scoring
│   │   ├── player_features.py    # Appearance extraction (skin tone, jersey, proportions)
│   │   └── match_tracker.py      # Cross-rally player ID consistency
│   ├── court/           # Court calibration
│   │   └── calibration.py        # Homography for image→court projection
│   ├── statistics/      # Match statistics aggregation
│   │   └── aggregator.py         # Rally grouping, action counts
│   ├── labeling/        # Ground truth labeling with Label Studio
│   │   ├── ground_truth.py      # GroundTruthPosition/Result data structures
│   │   └── studio_client.py     # Label Studio API client
│   ├── evaluation/      # Ground truth loading, metrics, parameter tuning
│   └── service/         # Cloud services (Modal deployment)
│       ├── platforms/modal_app.py       # Modal GPU detection function
│       ├── platforms/modal_tracking.py  # Modal GPU batch tracking function
│       └── player_tracking_runner.py    # Local player tracking subprocess
├── training/
│   └── within_team_reid/  # SupCon-trained OSNet over player crops (Session 3, 2026-04-16)
├── lib/volleyball_ml/    # ML model wrappers (VideoMAE)
└── tests/
    ├── unit/             # Fast tests with mocked ML
    └── integration/      # Full pipeline tests
```

## Configuration

Nested Pydantic config with YAML/env var support. Key sections:

| Section | Key Settings |
|---------|-------------|
| `game_state` | `stride=12` (frames between samples), `window_size=16`, `batch_size=8` |
| `segment` | `min_play_duration=1.0`, `min_gap=5.0`, `rally_continuation=2.0` |
| `proxy` | 480p@30fps normalized for faster ML |

**Loading priority:**
1. `./rallycut.yaml` (current dir)
2. `~/.config/rallycut/rallycut.yaml`
3. Environment: `RALLYCUT_GAME_STATE__STRIDE=32` (nested with `__`)
4. Defaults in `core/config.py`

## Caching

Analysis results cached in `~/.cache/rallycut/analysis/`:
- **Cache key**: SHA256(file_signature + stride + proxy_mode)
- **File signature**: filename + size + mtime + first 1MB hash
- **Invalidation**: Automatic when video changes

Proxy videos cached in `~/.cache/rallycut/proxies/`.

## Model Variants

Two model variants available via `--model` flag:

| Model | Weights | Use Case |
|-------|---------|----------|
| `beach` (default) | `weights/videomae/game_state_classifier/` | Beach volleyball |
| `indoor` | Same as beach (tuned heuristics) | Indoor volleyball courts |

**Note:** Both variants use the same model weights with different post-processing heuristics. Beach has stricter thresholds to prevent over-merging rallies.

Each model has optimized post-processing heuristics defined in `MODEL_PRESETS` (see `core/config.py`).

## Detection Pipelines

Two detection pipelines are available:

| Pipeline | F1 (IoU=0.5) | F1 (IoU=0.4) | Overmerge | Command |
|----------|--------------|--------------|-----------|---------|
| TemporalMaxer (default) | - | 94.4% (LOO) | 0% | `rallycut cut video.mp4 --temporal-maxer` |
| Heuristics (fallback) | 57% | - | ~10% | `rallycut cut video.mp4 --heuristics` |

**Note:** IoU=0.4 better reflects detection accuracy when labeling marks serve toss start (model detects ~2s later when play begins). TemporalMaxer's F1 is honest leave-one-video-out CV (stride=12, 43 videos).

**Pipeline auto-selection:**
1. If `--temporal-maxer` flag: use TemporalMaxer (auto-extracts features if not cached)
2. If `--heuristics` flag: use heuristics
3. Auto: TemporalMaxer if model+features exist, else heuristics

**Cloud (DetectionService/Modal):** Automatically uses TemporalMaxer when weights are in the image. Features are extracted inline on first run and cached for subsequent runs.

**Enabling best pipeline (local):**
```bash
# Train TemporalMaxer (features auto-extracted on first run, or pre-extract for speed)
uv run rallycut train extract-features --stride 12  # Optional: pre-extract for faster first run
uv run rallycut train temporal-maxer --epochs 50
```

## Processing Heuristics

Fallback pipeline (57% F1) applies confidence extension, rally continuation, and density filtering. Model-specific thresholds in `MODEL_PRESETS` (`core/config.py`). Beach uses stricter thresholds to prevent over-merging. See `processing/cutter.py`.

## Cloud Services (Modal)

Production ML runs on Modal GPUs. Two separate Modal apps for independent deploys:

**Detection** (`rallycut-detection`): TemporalMaxer rally detection. Auto-enabled when
`weights/temporal_maxer/best_temporal_maxer.pt` is in the image.

**Batch Tracking** (`rallycut-tracking`): GPU-accelerated player + ball tracking.
Downloads video once, tracks all rallies on T4 GPU (~80 FPS vs 6 FPS local CPU).
Sends per-rally webhooks for progressive results. Set `MODAL_TRACKING_URL` in API env
to enable; falls back to local CPU when unset.

```bash
# Deploy
modal deploy rallycut/service/platforms/modal_app.py       # Detection
modal deploy rallycut/service/platforms/modal_tracking.py  # Batch tracking

# Local testing
uv run python -m rallycut.service.local_runner
```

API triggers Modal via webhook. Results posted back on completion.

## Player Tracking

Uses YOLO for person detection with BoT-SORT for temporal tracking. BoT-SORT with ReID (enabled by default) reduces ID switches when players cross paths. GMC (camera motion compensation) is disabled for fixed tripod cameras.

**YOLO Model Options (imgsz=1280):**
| Model | FPS | F1 | HOTA | Far Recall | Use Case |
|-------|-----|-----|------|------------|----------|
| **yolo11s** | 6.1 | 92.5% | 91.3% | 96.3% | Default - best accuracy/speed tradeoff |
| yolov8n | 7.7 | 79.4% | 80.3% | 63.5% | Faster, lower far-court recall |
| yolo11m | 2.4 | 77.0% | 82.4% | 89.4% | Mid-recall regression, slow |
| yolov8n@640 | 15.3 | 74.1% | 79.2% | 55.3% | Fastest, lowest accuracy |

**Tracker Options:**
| Tracker | ReID | Description |
|---------|------|-------------|
| botsort | YOLO backbone | Ultralytics built-in BoT-SORT. Fast, but ReID uses detection features (not person-ReID trained) |
| **boxmot-botsort** (default) | Fine-tuned OSNet | BoxMOT BoT-SORT with our SupCon-trained OSNet-x1.0 (128-dim). +2.8pp F1, -59% FP. Requires `weights/reid/general_reid.pt` |
| bytetrack | None | Motion-only tracking (legacy) |

```bash
uv run rallycut track-players video.mp4                          # Default (yolo11s + BoxMOT + OSNet ReID)
uv run rallycut track-players video.mp4 --tracker botsort        # Legacy ultralytics BoT-SORT
uv run rallycut track-players video.mp4 --yolo-model yolov8n     # Faster, lower accuracy
uv run rallycut track-players video.mp4 --calibration '[...]'    # Court calibration for ROI + teams
```

## Player Tracking Filtering

Multi-stage filtering to identify 4 active players and exclude non-players. Key insight: positively identify players by volleyball behaviors (ball engagement, court coverage, movement) rather than detecting "non-players". Includes stationary background removal, primary track identification with hard/soft filters and safety net, per-frame filtering, and track merging. See `tracking/player_filter.py`.

## Ball Tracking

WASB HRNet is the sole ball tracker (96.9% keyframes-only match on 43 GT rallies, 33 FPS ONNX+CoreML). Multi-stage temporal filter handles false positives (static positions, player lock-on, oscillation, exit ghosts, non-ball objects, trajectory blips) and interpolates gaps. All parameters and filter stages are in `BallFilterConfig` in `tracking/ball_filter.py`. See `reports/ball_gap_diagnosis_2026_04_15.md` and `scripts/diagnose_ball_gaps.py` for the diagnostic tooling.

## Cross-Rally Player Consistency

Maintains consistent player IDs (1-4) across a match via appearance-based Hungarian assignment (2 passes). Features: upper/lower body color histograms, dominant clothing color, skin tone. Includes side switch detection and global within-team voting. See `tracking/match_tracker.py` and `tracking/player_features.py`.

### Assignment-anchor cache (Phase 2, 2026-05-01)

Per-rally `assignmentAnchor` on `match_analysis_json.rallies[]` decouples each rally's MatchSolver decision from cross-rally input drift. After a blind-path solve, each rally persists `{trackStatsHash, assignment, confidence}`. On the next run, when the rally's structural fingerprint (top_tracks, court_sides, positions) still matches its anchor, MatchSolver pins its prior assignment and skips re-decision — preventing the cascade where re-tracking ONE rally would shift every other rally's Hungarian via `_build_appearance_cost`.

| Flag / constant | Default | Effect |
|---|---|---|
| `ENABLE_ASSIGNMENT_ANCHORS=0` | (default ON) | Disable the cache; every rally re-solves every run |
| `match-players --reset-anchors` | off | Strip all prior anchors before solving (use after upstream pipeline retunes that don't change track IDs) |
| `ANCHOR_MIN_CONFIDENCE=0.50` | code constant | Below this assignment_confidence, the rally's anchor is NOT written — low-confidence rallies re-solve every run instead of locking in an uncertain decision |

Anchors are written on every blind-path run regardless of the read flag. The cache naturally invalidates when track IDs change (re-tracking with BoT-SORT). Tests: `tests/unit/test_match_solver_pinned.py`.

### Within-rally appearance-based ID-switch repair (Phase 1+2, 2026-05-03)

`ENABLE_WITHIN_RALLY_REPAIR=1` activates an appearance-consistency-based detector for within-rally identity drift. Per-track 3-window split, relative-gate trigger (`max intra-window cost > k × median inter-track cost`), changepoint localization, conservative re-Hungarian (more confident half re-assigns to its best other-track; other half keeps parent's matcher PID), then Phase 2 cross-track overlap clipping (sub-tracks yield to other tracks with the same PID). Module: `tracking/_within_rally_id_switch.py`. Tests: `tests/unit/test_within_rally_id_switch.py` + `tests/integration/test_within_rally_repair_e2e.py`.

Default OFF — addresses the BoT-SORT track-jump pattern but provides no measurable PERMUTED gain on the current 4-fixture panel. Cross-track merge (Phase 3) is the proper full fix for cases like 7d77980f / 09553ef1 where two BoT-SORT tracks represent the same physical player. See `feedback_validation_clean_state.md` for the validation protocol that revealed Phase 1+2 is no-regression but no-improvement.

### Cross-rally matcher validation protocol (MANDATORY)

The matcher's behavior depends on persistent caches (`assignmentAnchor`, `canonical_pid_map_json`). Stale state from prior runs contaminates measurements. **For ANY A/B comparison between matcher configurations, reset state before each measurement:**

```bash
# Reset state for a single video
uv run python scripts/reset_matcher_state.py <video_id>

# Reset all GT-labeled videos
uv run python scripts/reset_matcher_state.py --all-with-gt

# Wrapper script that resets + measures (preferred)
scripts/eval_cross_fixture.sh                # baseline (no flags)
scripts/eval_cross_fixture.sh --flag-on      # ENABLE_WITHIN_RALLY_REPAIR=1
scripts/eval_cross_fixture.sh <vid> <vid>    # custom panel
```

Without this, sequential runs are non-deterministic relative to a clean baseline — see `feedback_validation_clean_state.md` for the original incident (a session of failed iterations chasing a "regression" that turned out to be stale state).

#### Panel-tracking refresh protocol (MANDATORY before any panel-baseline change)

The panel eval (`scripts/eval_cross_fixture.sh`) is **matcher-only** — it reads stored tracking from DB and never re-runs `player_tracker`. So the PERMUTED numbers reflect whatever tracking is currently in DB; they do NOT measure fresh `player_tracker` output.

After ANY change to `analysis/rallycut/tracking/player_tracker.py` or its dependencies (chimera-stitching, BoT-SORT config, ReID model, court calibration), the panel videos must be retracked with the new code BEFORE re-baselining the panel. Otherwise the eval will measure the new matcher on stale tracking and produce misleading numbers.

```bash
# Refresh all panel tracking with current player_tracker code (~50 min)
uv run python scripts/retrack_panel_stale.py        # 28 originally-fragmented panel rallies
uv run python scripts/retrack_panel_remainder.py    # 12 panel rallies on older code
# Or for one video at a time:
uv run python scripts/retrack_b5fb0594_only.py      # all 11 b5fb0594 rallies

# Then reset matcher cache + re-measure
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh
```

The committed snapshot (`analysis/tests/fixtures/panel_player_tracks/{video}_summary.csv` + `analysis/reports/cross_fixture_baseline_2026_05_08.log`) is the regression-floor reference for the post-cleanup baseline (90.1 panel avg). Diff `primary_track_ids` per rally against the snapshot to detect drift; significant changes mean the panel needs a refresh + new baseline log + memory update. See `panel_baseline_regression_2026_05_07.md` for the original incident (a 21pp drop diagnosed as mixed-vintage tracking, not code regression).

`scripts/measure_pid_accuracy.py` reports THREE metrics:
1. **DIRECT** (matcher_pid == gt_pid as-is) — diagnostic only.
2. **PERMUTED** (Hungarian-permuted matcher → gt) — load-bearing quality metric. User has no canonical convention (`feedback_no_canonical_pid_convention.md`).
3. **ID-stability** (distinct matcher PIDs per GT player across the video) — surfaces user-perceived "PID flicker" that PERMUTED hides. 1.0 = perfectly stable; ≥1.5 = significant flicker.

### Diagnostic env flags

- `MATCH_PLAYERS_PROBE=1` — write per-(iteration, rally) sidecar JSON to `analysis/reports/profile_drift_probe/` capturing MatchSolver iteration state + post-solve `_update_profiles` snapshots. Default OFF; used to falsify hypotheses about cross-rally cascade.

## Ground Truth Labeling

Label Studio integration for tracking accuracy evaluation. See `labeling/studio_client.py`.

```bash
# Setup (one-time)
pip3 install label-studio
python3 -m label_studio start  # Opens at http://localhost:8080
export LABEL_STUDIO_API_KEY=your_token_here  # From Profile → Account & Settings

# Workflow
uv run rallycut track-players video.mp4 -o tracking.json
uv run rallycut label open video.mp4 -p tracking.json  # Opens browser, shows task ID
# Correct annotations in Label Studio, click Submit
uv run rallycut label save 123 -o ground_truth.json
uv run rallycut compare-tracking tracking.json ground_truth.json

# Evaluate from database (after ground truth is synced)
uv run rallycut evaluate-tracking --all              # All labeled rallies (player metrics)
uv run rallycut evaluate-tracking --all --ball-only  # Ball tracking metrics only
uv run rallycut evaluate-tracking --all --retrack    # Re-run tracking pipeline (~2 min/rally)
uv run rallycut evaluate-tracking --all --retrack --cached  # Cache raw detections, replay post-processing only (~5s/rally)
uv run rallycut evaluate-tracking -r <rally-id>      # Specific rally

# Grid search for filter parameters (use --help for grid options)
uv run rallycut evaluate-tracking tune-filter --all --grid quick
uv run rallycut evaluate-tracking tune-ball-filter --all --grid segment-pruning
```


## Scripts directory

Live scripts are in `analysis/scripts/`. One-off diagnostic / probe scripts from closed investigations have been moved to `analysis/reports/archived-scripts/<category>/` — see the [manifest](reports/archived-scripts/MANIFEST.md) for what's archived and links to the corresponding NO-GO / SHIPPED memory memos. Restoring an archived script is a `git mv` away.

Canonical entry points for the most common workflows live in the relevant skills (`ml-experiment`, `tracking-diagnosis`, `court-detection`, `contact-detection`, `video-analysis`); start there before grep-spelunking.


## Code Style

- Ruff: E, F, I, N, W, UP rules (line length 100, E501 ignored)
- MyPy: strict mode (`disallow_untyped_defs = true`)
- Type hints required on all functions

## Key Patterns

- **Lazy loading**: ML models loaded on first use, not import
- **Proxy videos**: 480p@30fps cached copies for faster ML (10-100x speedup)
- **Training proxies**: Same 480p proxies used for training (45% smaller than originals)
- **Temporal smoothing**: Median filter on ML results fixes isolated errors
- **Sequential reading**: Use `video.iter_frames()` not seeking
- **FPS normalization**: High-FPS videos (60fps+) subsampled to 30fps for VideoMAE

## Caveats

- **First run downloads ~500MB** of models from HuggingFace
- **Stride scales with FPS**: `--auto-stride` adjusts stride proportionally (stride 32 @ 30fps = stride 64 @ 60fps)
- **ONNX disabled on MPS**: CoreML provider has compatibility issues
- **Avoid seeking**: VideoMAE needs sequential frame reading for proper temporal dynamics
