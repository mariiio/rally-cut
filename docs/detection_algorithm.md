# Rally Detection Algorithm

This document describes how RallyCut detects rallies in beach volleyball videos.

## Overview

The rally detection pipeline works in 3 main phases:

```
Video → ML Classification → Post-Processing Heuristics → Final Segments
```

## Phase 1: ML Classification (VideoMAE)

**File:** `rallycut/analysis/game_state.py`

The ML model classifies video windows into 3 states: `SERVICE`, `PLAY`, `NO_PLAY`

### Parameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `window_size` | 16 | `GameStateConfig` | Frames per classification window |
| `stride` | 48 | `GameStateConfig` | Frames between windows (at 30fps = 1.6s between samples) |
| `batch_size` | 8 | `GameStateConfig` | Windows processed per batch |
| `analysis_size` | (224, 224) | `GameStateConfig` | Input resolution for model |

### How it works

```
Video (e.g., 120s at 30fps = 3600 frames)
    ↓
Proxy generation (480p, 30fps normalized)
    ↓
Sliding window extraction:
  - Window 1: frames 0-15   → classify → SERVICE (conf=1.0)
  - Window 2: frames 48-63  → classify → PLAY (conf=1.0)
  - Window 3: frames 96-111 → classify → NO_PLAY (conf=1.0)
  - ...
    ↓
List of GameStateResult (state, confidence, start_frame, end_frame, probabilities)
```

### FPS Normalization

- High-FPS videos (50/60fps) are subsampled to ~30fps
- This ensures the 16-frame window covers ~0.5s of content (optimal for model)
- See `docs/video_normalization_analysis.md` for details

## Phase 2: Post-Processing Heuristics

**File:** `rallycut/processing/cutter.py`

Raw ML predictions are noisy. Multiple heuristics clean them up:

### 2.1 Confidence-Based Boundary Extension

**Purpose:** Extend PLAY segments when adjacent NO_PLAY has high play probability

```python
BOUNDARY_CONFIDENCE_THRESHOLD = 0.35
```

**Logic:**
```
Before: PLAY → NO_PLAY (play_prob=0.4) → PLAY
After:  PLAY → PLAY    (extended)      → PLAY
```

- If NO_PLAY is between two PLAY segments AND `play_prob + service_prob > 0.25` → extend
- If NO_PLAY is adjacent to PLAY AND `play_prob + service_prob > 0.35` → extend

### 2.2 Rally Continuation Heuristic

**Purpose:** Keep rallies active through brief NO_PLAY gaps

**Parameter:** `rally_continuation_seconds` (default: 2.0) in `SegmentConfig`

**Logic:**
```
Frame-by-frame tracking:
- Once PLAY/SERVICE detected → in_rally = True
- Count consecutive NO_PLAY frames
- Only end rally when NO_PLAY count >= (2.0 * fps) frames
- Convert intermediate NO_PLAY → PLAY with reduced confidence
```

**Example:**
```
Before: PLAY PLAY NO_PLAY NO_PLAY PLAY PLAY (1.5s gap)
After:  PLAY PLAY PLAY    PLAY    PLAY PLAY (gap bridged)

Before: PLAY PLAY NO_PLAY NO_PLAY NO_PLAY NO_PLAY PLAY (3s gap)
After:  PLAY PLAY NO_PLAY NO_PLAY NO_PLAY NO_PLAY PLAY (gap NOT bridged, >2s)
```

### 2.3 Segment Merging

**Purpose:** Merge adjacent same-state results into continuous segments

```
Results: [PLAY, PLAY, PLAY, NO_PLAY, NO_PLAY, PLAY, PLAY]
Segments: [PLAY 0-2], [NO_PLAY 3-4], [PLAY 5-6]
```

### 2.4 Gap Bridging

**Purpose:** Merge PLAY segments separated by short NO_PLAY gaps

**Parameter:** `min_gap_seconds` (default: 5.0) in `SegmentConfig`

**Logic:**
```
If NO_PLAY gap < 5.0 seconds AND substantial PLAY after:
  → Merge through the gap

Before: [PLAY 0-30s] [NO_PLAY 30-33s] [PLAY 33-50s]
After:  [PLAY 0-50s] (3s gap bridged)

Before: [PLAY 0-30s] [NO_PLAY 30-38s] [PLAY 38-50s]
After:  [PLAY 0-30s] [PLAY 38-50s] (8s gap NOT bridged)
```

## Phase 3: Segment Filtering

### 3.1 Minimum Active Windows

**Purpose:** Filter isolated false positives

```python
MIN_ACTIVE_WINDOWS = 1  # At least 1 PLAY/SERVICE window required
```

### 3.2 Minimum Play Duration

**Purpose:** Filter very short segments

**Parameter:** `min_play_duration` (default: 1.0) in `SegmentConfig`

Only applies to multi-window segments. Single confident windows are kept.

### 3.3 Active Density Filter

**Purpose:** Filter sparse detections spanning large time ranges

```python
MIN_ACTIVE_DENSITY = 0.25  # 25% of stride intervals must be active
```

**Example:**
```
Segment spans 10 seconds at stride=48 (30fps) = ~6 stride intervals
If only 1 active window → density = 1/6 = 16% < 25% → filtered
```

## Phase 4: Padding & Final Output

### Padding

**Parameters in `SegmentConfig`:**
- `padding_seconds` (default: 2.0) - Added before segment start
- End padding is automatically `padding_seconds + 1.5` (default: 3.5s)

```
Raw segment:  [10.0s - 20.0s]
With padding: [8.0s - 23.5s]
```

### Overlapping Segment Merge

Adjacent/overlapping padded segments are merged:
```
Segment 1: [8.0s - 23.5s]
Segment 2: [22.0s - 35.0s]
Merged:    [8.0s - 35.0s]
```

## Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO                              │
│                    (e.g., 2 min volleyball match)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROXY GENERATION                              │
│              480p, 30fps normalized                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ML CLASSIFICATION (VideoMAE)                    │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ NP  │ │ SRV │ │PLAY │ │ NP  │ │PLAY │ │PLAY │ │ NP  │ ...   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│  window_size=16, stride=48                                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONFIDENCE-BASED EXTENSION                          │
│  Extend PLAY into adjacent low-confidence NO_PLAY                │
│  threshold=0.35                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              RALLY CONTINUATION HEURISTIC                        │
│  Keep rally active until 2s consecutive NO_PLAY                  │
│  rally_continuation_seconds=2.0                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SEGMENT MERGING                                │
│  Merge adjacent same-state predictions                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GAP BRIDGING                                  │
│  Merge PLAY segments with <5s NO_PLAY gaps                       │
│  min_gap_seconds=5.0                                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FILTERING                                    │
│  - MIN_ACTIVE_WINDOWS >= 1                                       │
│  - MIN_ACTIVE_DENSITY >= 0.25                                    │
│  - min_play_duration >= 1.0s (multi-window only)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PADDING                                     │
│  +2.0s start, +3.5s end                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OVERLAP MERGE                                   │
│  Combine overlapping padded segments                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                  │
│  List of TimeSegment(start_time, end_time, state)                │
│  e.g., [Rally 1: 8.0-23.5s], [Rally 2: 30.0-52.0s], ...         │
└─────────────────────────────────────────────────────────────────┘
```

## Parameter Reference

### config.py - GameStateConfig

| Parameter | Default | CLI Flag | Effect |
|-----------|---------|----------|--------|
| `stride` | 48 | `--stride` | Lower = more predictions, slower |
| `window_size` | 16 | - | Frames per ML window |
| `enable_temporal_smoothing` | False | - | Median filter on predictions |
| `temporal_smoothing_window` | 3 | - | Smoothing window size |

### config.py - SegmentConfig

| Parameter | Default | CLI Flag | Effect |
|-----------|---------|----------|--------|
| `min_play_duration` | 1.0s | `--min-play` | Minimum segment length |
| `padding_seconds` | 2.0s | `--padding` | Start padding |
| `min_gap_seconds` | 5.0s | `--min-gap` | Max bridgeable NO_PLAY gap |
| `rally_continuation_seconds` | 2.0s | `--rally-continuation` | NO_PLAY duration to end rally |

### cutter.py - Constants

| Parameter | Value | Effect |
|-----------|-------|--------|
| `BOUNDARY_CONFIDENCE_THRESHOLD` | 0.35 | Confidence for boundary extension |
| `MIN_ACTIVE_WINDOWS` | 1 | Min PLAY windows per segment |
| `MIN_ACTIVE_DENSITY` | 0.25 | Min active window density |

## Tuning Guide

### To detect more rallies (higher recall):
- Decrease `stride` (more predictions, slower)
- Increase `rally_continuation_seconds` (bridge longer gaps)
- Increase `min_gap_seconds` (merge more segments)
- Decrease `min_play_duration` (keep shorter segments)

### To reduce false positives (higher precision):
- Increase `stride` (fewer, more confident predictions)
- Decrease `rally_continuation_seconds`
- Decrease `min_gap_seconds`
- Increase `min_play_duration`
- Increase `MIN_ACTIVE_DENSITY` in code

### Performance vs Accuracy:
- `stride=48` (default): ~1.6s between samples, good balance
- `stride=32`: ~1.0s between samples, 50% slower, slightly better coverage
- `stride=64`: ~2.1s between samples, 33% faster, may miss short rallies
