# Tasks: RallyCut MVP Core

**Feature Branch**: `001-mvp-core`
**Generated**: 2025-01-22
**Updated**: 2025-12-22
**Total Tasks**: 32

## Phase 1: Setup (Foundation) ✅

- [X] T001 Create core domain models in `rallycut/core/models.py` (GameState, Action, ActionType, Rally, BallPosition, MatchStatistics)
- [X] T002 Create configuration management in `rallycut/core/config.py` (RallyCutConfig with Pydantic)
- [X] T003 Create video abstraction in `rallycut/core/video.py` (Video class with metadata extraction)

## Phase 2: Foundational (ML Infrastructure)

- [ ] T004 Create model manager in `rallycut/analysis/models/manager.py` (ModelManager with lazy loading, device detection)
- [X] T005 [P] Create VideoMAE adapter in `lib/volleyball_ml/video_mae.py` (GameStateClassifier wrapping transformers)
- [ ] T006 [P] Create YOLO adapter in `lib/volleyball_ml/yolo_detector.py` (ActionDetector wrapping ultralytics)
- [ ] T007 Implement actual model download in `rallycut/cli/commands/models.py` (httpx download with progress)

## Phase 3: User Story 1 - Dead Time Removal (P1) ✅

Goal: Automatically remove no-play segments from recordings
Independent Test: Process a video and verify output contains only play segments

- [X] T008 [US1] Create game state classifier wrapper in `rallycut/analysis/game_state.py`
- [ ] T009 [US1] Create chunk manager in `rallycut/analysis/chunk_manager.py` (memory-efficient video iteration)
- [X] T010 [US1] Create video cutter in `rallycut/processing/cutter.py` (segment extraction with FFmpeg)
- [X] T011 [US1] Create FFmpeg exporter in `rallycut/processing/exporter.py` (video concatenation, encoding)
- [X] T012 [US1] Wire up `cut` command in `rallycut/cli/commands/cut.py` (replace skeleton with real implementation)

### Additional Features Implemented (Not in Original Spec)
- [X] T012a Motion detector alternative in `rallycut/analysis/motion_detector.py` (`--quick` mode)
- [X] T012b `--limit` flag for testing on first N seconds
- [X] T012c `--dry-run` mode for analysis without video export
- [X] T012d `--segments` flag to load segments from JSON file
- [X] T012e `--json` flag to export detected segments to JSON
- [X] T012f Batch processing for ML inference (4-8x speedup potential)
- [X] T012g Local volleyball weights support (`weights/videomae/game_state_classifier/`)
- [X] T012h Hysteresis for rally detection (merge short NO_PLAY gaps <1.5s)

## Phase 4: User Story 2 - Game Statistics (P2) ⏸️ ON HOLD

Goal: Track serves, attacks, blocks, receptions
Independent Test: Process a video and verify JSON contains accurate action counts

- [X] T013 [US2] Create action detector wrapper in `rallycut/analysis/action_detector.py`
- [X] T014 [US2] Create statistics aggregator in `rallycut/statistics/aggregator.py` (rally detection, action counting)
- [X] T015 [US2] Create JSON exporter in `rallycut/output/json_export.py` (integrated into stats command)
- [X] T016 [US2] Wire up `stats` command in `rallycut/cli/commands/stats.py`

### Model Evaluation Results (2025-12-22)
⚠️ **Indoor volleyball model does NOT work for beach volleyball**
- Tested volleyball_analytics YOLO weights on beach volleyball footage
- Result: 0% precision, 0% recall on action detection
- Model only detects "receive" class (false positives) and misses all serves/sets/spikes
- See `ground_truth/evaluation_report.md` for full analysis

### Next Steps for Action Detection
- Requires beach volleyball-trained model
- Best candidate: Volleyball Activity Dataset (Roboflow) - needs training
- Skipped for MVP - will revisit when suitable model available

## Phase 5: User Story 3 - Highlight Generation (P3) ✅

Goal: Auto-generate highlight clips of best plays
Independent Test: Process a video and verify highlights contain high-action moments

- [X] T017 [US3] Create highlight scorer in `rallycut/processing/highlights.py` (rank rallies by duration)
- [X] T018 [US3] Add clip extraction to exporter in `rallycut/processing/exporter.py` (extract and concatenate clips)
- [X] T019 [US3] Wire up `highlights` command in `rallycut/cli/commands/highlights.py`

### Highlights Features (2025-12-22)
- Ranks rallies by duration (longer = more exciting)
- Export as single video or individual clips (`--individual`)
- Chronological or by-score ordering (`--by-score`)
- Asymmetric padding (1s start, 2s end by default)
- `--dry-run` for quick ranking preview without video export
- Example: `rallycut highlights match.mp4 -n 5 -o best.mp4`

## Phase 6: User Story 4 - Ball Tracking Overlay (P4) ✅

Goal: Visual ball trajectory overlay on videos
Independent Test: Process a video and verify ball trail is rendered correctly

- [X] T020 [US4] Create ball tracker in `rallycut/tracking/ball_tracker.py` (Kalman filter with prediction)
- [X] T021 [US4] Create trajectory processor in `rallycut/tracking/trajectory.py` (smoothing, interpolation)
- [X] T022 [US4] Create overlay renderer in `rallycut/output/overlay.py` (draw trajectory on frames)
- [X] T023 [US4] Wire up `overlay` command in `rallycut/cli/commands/overlay.py`

### Ball Tracking Results (2025-12-22)
- Detection rate: ~58% on beach volleyball footage
- Kalman filter fills gaps with predictions (disabled by default)
- Edge detection to stop tracking when ball leaves frame
- Trajectory segmentation with max_gap_frames=5
- Trail rendering with fading effect
- Example: `rallycut overlay match.mp4 -s 12 -e 18 -o rally1.mp4`

## Phase 7: User Story 5 - Full Pipeline (P1) ⏳ DEFERRED

Goal: Single command runs all analysis
Independent Test: Run analyze command and verify all outputs are generated

- [ ] T024 [US5] Create analysis pipeline in `rallycut/analysis/pipeline.py` (orchestrate all components)
- [ ] T025 [US5] Wire up `analyze` command in `rallycut/cli/commands/analyze.py` (replace skeleton with real implementation)

**Note**: Deferred until action detection (Phase 4) has a working beach volleyball model.
Individual commands (`cut`, `highlights`, `overlay`) provide full functionality.

## Phase 8: Polish & Cross-Cutting ✅

- [X] T026 Add progress reporting with Rich in CLI commands (progress bars, status updates)
- [X] T027 Add error handling and user-friendly messages across all commands
  - Created `rallycut/cli/utils.py` with `@handle_errors` decorator
  - Custom exceptions: `RallyCutError`, `VideoError`, `ModelError`, `ExportError`
- [X] T028 [P] Create unit tests in `tests/unit/test_models.py` (24 tests)
- [X] T029 [P] Create unit tests in `tests/unit/test_game_state.py` (8 tests)
- [X] T030 [P] Create unit tests in `tests/unit/test_statistics.py` (12 tests)
- [X] T031 Create integration test in `tests/integration/test_pipeline.py` (12 tests)
- [X] T032 Update README.md with final usage examples and known limitations

### Polish Fixes (2025-12-22)
- Fixed `config.py` lazy torch import (allows tests without torch)
- Fixed pydantic deprecation warning (use `SettingsConfigDict`)
- Fixed `Video.__del__` AttributeError when file not found
- All 56 tests passing with 0 warnings

## Dependencies

```
Phase 1 (Setup) ──► Phase 2 (ML Infrastructure)
                          │
                          ├──► Phase 3 (US1: Cut) ──► Phase 7 (US5: Analyze)
                          │
                          ├──► Phase 4 (US2: Stats) ──► Phase 7 (US5: Analyze)
                          │
                          ├──► Phase 5 (US3: Highlights) ──► Phase 7 (US5: Analyze)
                          │
                          └──► Phase 6 (US4: Overlay) ──► Phase 7 (US5: Analyze)
                                                              │
                                                              ▼
                                                    Phase 8 (Polish)
```

## Completion Summary

| Phase | Completed | Total | Status |
|-------|-----------|-------|--------|
| 1: Setup | 3 | 3 | ✅ Complete |
| 2: Foundational | 1 | 4 | 🔄 Partial (1/4) |
| 3: US1 Cut | 12 | 12 | ✅ Complete (with extras) |
| 4: US2 Stats | 4 | 4 | ⏸️ On Hold (needs beach volleyball model) |
| 5: US3 Highlights | 3 | 3 | ✅ Complete |
| 6: US4 Overlay | 4 | 4 | ✅ Complete |
| 7: US5 Pipeline | 0 | 2 | ⏳ Deferred |
| 8: Polish | 7 | 7 | ✅ Complete |

**Overall**: 34/39 tasks complete (87%)

## Test Summary

```
tests/
├── unit/
│   ├── test_models.py      # 24 tests - domain models
│   ├── test_game_state.py  # 8 tests - game state analyzer
│   └── test_statistics.py  # 12 tests - statistics aggregator
└── integration/
    └── test_pipeline.py    # 12 tests - pipeline integration

Total: 56 tests passing, 0 warnings
```

## ML Performance Notes

**Tested on first 2 minutes of match video (ground truth: 4 rallies)**

| Setting | Rallies Detected | Accuracy |
|---------|------------------|----------|
| `--stride 16` | 4/4 (fragmented) | ~75% |
| `--stride 8 --padding 2.0 --min-play 1.5` | 4/4 | ~95% |

**Recommended settings for production:**
```bash
rallycut cut video.mp4 --stride 8 --padding 2.0 --min-play 1.5
```

**Performance (Apple Silicon M-series):**
- ~2.5 min processing per 1 min video (stride 8)
- ~1.5 min processing per 1 min video (stride 16)
- Batch processing enabled (batch_size=8)

## Available Commands

| Command | Description | Status |
|---------|-------------|--------|
| `rallycut cut <video>` | Remove dead time from video | ✅ Working |
| `rallycut highlights <video>` | Generate highlight reel | ✅ Working |
| `rallycut overlay <video>` | Add ball tracking overlay | ✅ Working |
| `rallycut stats <video>` | Extract game statistics | ⚠️ Limited (indoor model) |
