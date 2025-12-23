# Model Evaluation Report

**Date**: 2025-12-22
**Test Video**: match.mp4 (first 46 seconds)
**Ground Truth**: 15 annotated actions across 2 rallies

## Summary

The volleyball_analytics YOLO model (trained on indoor volleyball) does **not work** for beach volleyball action detection.

## Test Results

### Overall Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.00% |
| Recall | 0.00% |
| F1 Score | 0.00% |
| Ground Truth Actions | 15 |
| Model Predictions | 50 |

### Per-Class Breakdown

| Class | Ground Truth | Detected | Notes |
|-------|--------------|----------|-------|
| serve | 2 | 1 (misaligned) | Model detected serve at 43.9s, GT has serves at 12s and 32s |
| receive | 6 | 49 | Massive false positives - detects any player stance as "receive" |
| set | 4 | 0 | Not detected at all |
| spike | 3 | 0 | Not detected at all |
| block | 0 | 0 | N/A |

### Direct Frame Testing

Testing model on exact ground truth frames with low confidence threshold (0.1):

| Frame | Expected | Detected |
|-------|----------|----------|
| 360 (12.0s) | serve | NONE |
| 420 (14.0s) | receive | ball only |
| 480 (16.0s) | set | NONE |
| 540 (18.0s) | spike | ball only |
| 960 (32.0s) | serve | receive (wrong!), ball |

## Root Cause

The model was trained on **indoor volleyball** footage which has:
- Different court (hardwood vs sand)
- Different lighting (indoor vs outdoor/varied)
- Different player count (6v6 vs 2v2)
- Different camera angles
- Different player attire and postures

Beach volleyball visual patterns are fundamentally different, so the model fails to recognize actions.

## Recommendations

### Option 1: Alternative Models (Limited)
- Roboflow beach volleyball models exist but are for **ball/person detection only**, not action classification
- No publicly available beach volleyball action detection models found

### Option 2: Video-Based Action Recognition
- Use temporal models like VideoMAE, TimeSformer, or X3D
- Pros: Can learn beach volleyball patterns with smaller dataset
- Cons: Requires labeled video clips for training

### Option 3: Heuristic Approach
- Use ball trajectory analysis (ball detector works)
- Detect key moments: high ball (set/serve), fast downward (spike), low ball (receive)
- Combine with game state detection (already working)

### Option 4: Focus on Working Features
- The `cut` command works well with VideoMAE game state detection
- Ball tracking appears functional (ball class detected correctly)
- Skip action classification for MVP, add later when suitable model available

## Next Steps

1. **Validate ball detector** - Test if ball detection is accurate for trajectory analysis
2. **Consider hybrid approach** - Use ball trajectory + simple heuristics for action hints
3. **Update documentation** - Note that stats command requires beach volleyball-trained model
4. **Continue with working features** - Cut command, ball tracking for overlay

## Files

- `ground_truth/annotations.json` - Manual annotations (15 actions)
- `ground_truth/model_results.json` - Model detection output
- `rallycut/evaluation/metrics.py` - Evaluation metrics code
