# Feature Specification: RallyCut MVP Core

**Feature Branch**: `001-mvp-core`
**Created**: 2025-01-22
**Updated**: 2025-12-22
**Status**: MVP Complete (87% tasks done)
**Input**: Beach volleyball video analysis CLI with auto-cut, statistics, highlights, and ball tracking

## User Scenarios & Testing

### User Story 1 - Dead Time Removal (Priority: P1) ✅ COMPLETE

As a content creator, I want to automatically remove no-play segments from my beach volleyball recordings so my match videos become watchable without hours of manual editing.

**Why this priority**: This is the core value proposition. A 2-hour match recording typically has 60-70% dead time. Removing it automatically provides immediate, tangible value.

**Independent Test**: Upload a 10-minute test video containing both play and no-play segments. Verify the output video only contains play segments with appropriate transitions.

**Acceptance Scenarios**:

1. **Given** a video with SERVICE, PLAY, and NO_PLAY segments, **When** I run `rallycut cut video.mp4`, **Then** the output video contains only SERVICE and PLAY segments ✅
2. **Given** a video file, **When** I run `rallycut cut video.mp4 --padding 2.0`, **Then** 2 seconds of context is preserved before and after each play segment ✅
3. **Given** an invalid video file, **When** I run `rallycut cut invalid.mp4`, **Then** I receive a clear error message explaining the issue ✅
4. **Given** a video file, **When** I run `rallycut cut video.mp4 --dry-run`, **Then** I see detected segments without generating output video ✅
5. **Given** a video file, **When** I run `rallycut cut video.mp4 --quick`, **Then** fast motion detection is used instead of ML model ✅
6. **Given** a video file, **When** I run `rallycut cut video.mp4 --json segments.json`, **Then** detected segments are exported to JSON ✅
7. **Given** a JSON segments file, **When** I run `rallycut cut video.mp4 --segments segments.json`, **Then** the specified segments are used (skip analysis) ✅

---

### User Story 2 - Game Statistics (Priority: P2) ⏸️ ON HOLD

As a coach, I want to see statistics on serves, attacks, blocks, and receptions so I can analyze team and player performance without manually counting actions.

**Why this priority**: Statistics provide analytical value that complements the edited video. Essential for serious players and coaches.

**Independent Test**: Process a video and verify the JSON output contains accurate counts for each action type compared to manual counting.

**Status**: ⚠️ Indoor volleyball YOLO model doesn't work for beach volleyball (0% accuracy). Requires beach volleyball-trained model.

**Acceptance Scenarios**:

1. **Given** a beach volleyball video, **When** I run `rallycut stats video.mp4`, **Then** I receive JSON output with counts for serves, attacks, blocks, receptions ⚠️ (works but inaccurate)
2. **Given** a video, **When** I run `rallycut stats video.mp4 --format csv`, **Then** I receive statistics in CSV format ❌ (not implemented)
3. **Given** a video, **When** I run `rallycut stats video.mp4 --output stats.json`, **Then** statistics are saved to the specified file ❌ (not implemented)

---

### User Story 3 - Highlight Generation (Priority: P3) ✅ COMPLETE

As a player, I want automatically generated highlight clips of the best plays so I can easily share exciting moments on social media without editing skills.

**Why this priority**: Adds shareability and social value on top of the core analysis. Uses rally detection from game state classifier.

**Independent Test**: Process a video and verify the output contains clips of high-action moments (long rallies) concatenated into a highlight reel.

**Acceptance Scenarios**:

1. **Given** a beach volleyball video, **When** I run `rallycut highlights video.mp4`, **Then** I receive a video containing the top 5 longest rallies ✅
2. **Given** a video, **When** I run `rallycut highlights video.mp4 -n 10`, **Then** I receive a highlight reel with 10 clips ✅
3. **Given** a video with few rallies, **When** I run `rallycut highlights video.mp4`, **Then** I receive all available highlights ✅
4. **Given** a video, **When** I run `rallycut highlights video.mp4 --dry-run`, **Then** I see highlight rankings without generating video ✅
5. **Given** a video, **When** I run `rallycut highlights video.mp4 --individual`, **Then** individual clip files are exported ✅

---

### User Story 4 - Ball Tracking Overlay (Priority: P4) ✅ COMPLETE

As an analyst, I want to see the ball trajectory overlaid on the video so I can study ball movement patterns and player positioning.

**Why this priority**: Advanced visualization feature that enhances understanding but isn't essential for basic use.

**Independent Test**: Process a short video clip and verify the output shows a colored trail following the ball's trajectory.

**Acceptance Scenarios**:

1. **Given** a beach volleyball video, **When** I run `rallycut overlay video.mp4`, **Then** the output video shows ball positions with trailing trajectory visualization ✅
2. **Given** a video, **When** I run `rallycut overlay video.mp4 --trail 20`, **Then** the ball trail shows the last 20 frames of movement ✅
3. **Given** a video where the ball goes out of frame, **When** processing, **Then** the trajectory stops at frame edge (no predictions by default) ✅
4. **Given** a video, **When** I run `rallycut overlay video.mp4 -s 12 -e 25`, **Then** only the specified time range is processed ✅

---

### User Story 5 - Full Analysis Pipeline (Priority: P1) ⏳ DEFERRED

As a user, I want to run a single command that performs all analysis and generates all outputs so I don't have to run multiple commands.

**Why this priority**: Essential for user experience - most users want everything at once.

**Status**: Deferred until action detection (US2) has a working beach volleyball model. Individual commands provide full functionality.

**Acceptance Scenarios**:

1. **Given** a beach volleyball video, **When** I run `rallycut analyze video.mp4`, **Then** I receive cut video, statistics JSON, and can optionally add highlights/overlay ❌
2. **Given** a video, **When** I run `rallycut analyze video.mp4 --output-dir ./results`, **Then** all outputs are saved to the specified directory ❌
3. **Given** processing in progress, **When** I check the terminal, **Then** I see a progress bar with estimated time remaining ✅ (implemented in individual commands)

---

### Edge Cases

- What happens when video resolution is very low (480p)?
  → System should warn but continue processing with reduced accuracy expectations
- How does system handle videos without any volleyball content?
  → System reports "no play segments detected" and exits gracefully ✅
- What happens when video is corrupted mid-file?
  → System should process available portion and report partial results
- How does system handle vertical (9:16) phone videos?
  → System should auto-detect and handle rotation, warning if court is partially visible
- What happens when the ball is not visible for extended periods?
  → Ball tracking stops when ball leaves frame, trajectory segments split at gaps ✅

## Requirements

### Functional Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| FR-001 | System MUST process MP4, MOV, and AVI video formats | ✅ |
| FR-002 | System MUST classify frames as SERVICE, PLAY, or NO_PLAY | ✅ |
| FR-003 | System MUST detect actions: serve, reception, set, attack, block | ⚠️ (indoor model only) |
| FR-004 | System MUST track ball position when visible | ✅ |
| FR-005 | System MUST export cut video with configurable padding | ✅ |
| FR-006 | System MUST generate JSON statistics with action counts and timing | ⚠️ (inaccurate for beach) |
| FR-007 | System MUST show progress during long operations | ✅ |
| FR-008 | System MUST work without GPU (CPU-only mode) | ✅ |
| FR-009 | System MUST download ML models on first use | ❌ (manual setup) |
| FR-010 | System MUST cache analysis results for processed videos | ❌ (not implemented) |

### Key Entities

- **Video**: Input video file with metadata (resolution, fps, duration, codec) ✅
- **GameState**: Classification result (SERVICE, PLAY, NO_PLAY) with timestamp range and confidence ✅
- **Action**: Detected action (type, timestamp, bounding box, confidence) ✅
- **Rally**: Sequence of actions from service to point end ✅
- **BallPosition**: Frame-indexed ball coordinates with confidence and prediction flag ✅
- **Trajectory**: Smoothed sequence of ball positions for visualization ✅
- **MatchStatistics**: Aggregated counts and metrics for the full video ✅

## Success Criteria

### Measurable Outcomes

| ID | Criterion | Status |
|----|-----------|--------|
| SC-001 | Dead time removal produces video with >95% play content (vs original ~30-40%) | ✅ |
| SC-002 | Game state classification achieves >85% accuracy on test videos | ✅ ~95% with tuned settings |
| SC-003 | Action detection achieves >75% recall for major actions (serve, attack) | ❌ 0% (wrong model) |
| SC-004 | Processing completes within 1x video duration on Apple Silicon (1hr video → <1hr processing) | ⚠️ ~2.5x with stride 8 |
| SC-005 | CPU-only processing completes within 2x video duration | ❓ Not tested |
| SC-006 | Memory usage stays below 8GB for 1080p video processing | ✅ |
| SC-007 | Users can process a video successfully on first attempt without reading documentation | ✅ |

## Summary

**MVP Status**: 87% complete (34/39 tasks)

**Working Features**:
- `rallycut cut` - Dead time removal with VideoMAE classifier
- `rallycut highlights` - Highlight generation from top rallies
- `rallycut overlay` - Ball tracking with trajectory visualization

**On Hold**:
- `rallycut stats` - Action detection requires beach volleyball-trained model
- `rallycut analyze` - Full pipeline deferred until stats working

**Test Coverage**: 56 tests passing, 0 warnings
