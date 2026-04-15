# Project C — Sport-Sanity + Auto-Rotate Design

## Context

Project A1 validation (2026-04-15) confirmed the original e2e-pipeline-redesign spec was right to split "richer non-VB rejection" and "auto-rotate tilted clips" into their own project:

- **Indoor/non-VB slip-through:** `~/Desktop/rallies/Negative/indoor 2.mp4` (courtConfidence 0.78) and `not related.mp4` (0.83) bypass the A1 `wrong_angle_or_not_volleyball` block because the beach-trained court-keypoint model confidently hallucinates a "court" in both. A1 blocks the three obvious negatives (`bad angle`, `very bad angle`, `indoor 1`) via low court confidence, but the model can't distinguish "not beach VB" from "beach VB at an odd angle."
- **Auto-rotate:** A1 shipped a `video_rotated` advisory with no fix. Calibration never saw tilt in the 67-video GT and the A1 validation fixtures were all straight, so the check was dropped on 2026-04-15 rather than ship an advisory that does nothing. Project C re-introduces tilt detection *paired with* a fix, so the advisory is meaningful or absent.

Project C ships two independent capabilities that each plug into an existing pipeline stage. No new orchestration, no schema migration beyond one additive `autoFixes` field.

## Scope

**In**
1. **A — Beach-VB classifier.** Zero-shot CLIP scene-content check wired into the existing client-side pre-upload gate (`rallycut preview-check` CLI). Blocks indoor volleyball and arbitrary non-VB uploads before bandwidth is spent.
2. **B — Silent auto-rotate during optimization.** Tilt detection at video-confirm time; FFmpeg rotation filter appended to the existing optimize pass when both `tiltDeg > 5°` and `courtConfidence > 0.8`; info-tier "auto-straightened" badge surfaced to the user.

**Out (future)**
- Other sport-type disambiguation (grass, indoor beach, etc.) beyond the binary beach-VB / not-beach-VB cut.
- Non-court tilt sources (phone held sideways while camera itself isn't tilted — rare and already partly covered by ffprobe `rotate` metadata at upload).
- Client-side preview rotation (keep rotate server-side only; browser ffmpeg.wasm is too slow on long videos).

## Guiding Principle: favor false-accept over false-reject

A false-reject (good match blocked, or straight video rotated unnecessarily) is user-visible, damaging, and hard to recover from. A false-accept (bad match slips through, slight tilt left alone) produces at worst A1's current behavior — poor tracking, recoverable.

Both components apply this asymmetrically:
- **A** ships only if a ≥ 0.15 gap exists between the highest-scoring negative and the lowest-scoring positive on the calibration set. No gap, no ship.
- **B** requires **both** `tiltDeg > 5°` **and** `courtConfidence > 0.8`. Either check alone admits edge cases; AND-composed, they require both the symptom and model confidence before touching the video.

## Final Check Set Delta vs A1

A1 post-validation (merged at `ee7ab1f`):

| Check | Tier | Source |
|---|---|---|
| `video_too_short` | block | ffprobe |
| `resolution_too_low` | gate | ffprobe |
| `fps_too_low` | gate | ffprobe |
| `wrong_angle_or_not_volleyball` | block | court-keypoint confidence |

Project C adds:

| Check / capability | Tier / kind | Source | Where it runs |
|---|---|---|---|
| `not_beach_volleyball` | block | open-clip ViT-B/32 zero-shot on 5 frames | preview-check CLI (pre-upload) |
| auto-straighten | `autoFixes` entry (new field) | FFmpeg `rotate` filter | processingService optimize pass (post-upload, pre-preflight) |

## Architecture

### Component A — Beach-VB classifier

**Model + dependency**
- `open-clip-torch` (package) with `ViT-B-32` / `laion2b_s34b_b79k` weights. ~150 MB on first download, cached in `~/.cache/clip/`.
- Added to `analysis/pyproject.toml` under a `[project.optional-dependencies] preflight` extra so CI or machines that don't run preflight can skip the heavy install.
- Project already depends on PyTorch for VideoMAE; no new framework.

**Prompts — binary**

The existing stub at `analysis/rallycut/quality/beach_vb_classifier.py` has three prompts (`beach_vb`, `indoor_vb`, `other`). Switch to binary:

```python
PROMPTS = (
    "a beach volleyball match played on sand",
    "a video that is not beach volleyball",
)
```

Per-frame score = softmax prob of the first prompt. Video score = mean across the 5 frames the preview gate extracts. Simpler calibration, cleaner decision boundary.

**Issue emitted**

```python
Issue(
    id="not_beach_volleyball",   # distinct id from wrong_angle_or_not_volleyball
    tier=Tier.BLOCK,
    severity=1.0 - avg_beach_vb_prob,
    message="This doesn't look like a beach volleyball match. RallyCut is tuned for beach volleyball only.",
    source="preview",
    data={"avgBeachVbProb": avg_beach_vb_prob},
)
```

Separate `id` from `wrong_angle_or_not_volleyball` even though user-facing messages read similarly. Keeps post-ship debugging tractable: we can ask "was this block from the geometry check or the scene check?" without parsing message text.

**Integration point**

- `rallycut preview-check` CLI (already runs court-keypoint on 5 frames) gains a second check. Both results merge via the existing `QualityReport.from_checks`.
- API side: `runPreviewChecks` in `api/src/services/qualityService.ts` is untouched. The shape of the returned `{pass, issues}` stays the same; only the content of `issues` grows.
- Web side: `UploadFlow` already aborts on any `block`-tier issue. No change needed.

**Failure mode**

open-clip import or weights download fails → the new check logs a warning and returns zero issues. The existing court-confidence block still runs on the same 5 frames, so the system degrades cleanly to A1 behavior (blocks 3 of 5 validation negatives). Upload is never silently aborted by a tooling failure.

### Component B — Silent auto-rotate

**Detection — when it runs**

Tilt requires court corners, which require court detection. To make the detection value available *before* the optimize pass, run a lightweight one-shot court detection during the existing `POST /v1/videos/:id/confirm` flow — same place `computeBrightness` runs today in `api/src/services/processingService.ts`.

New CLI: `rallycut tilt-detect <video>` — extracts 3–5 evenly-sampled frames, runs `CourtKeypointDetector.detect_from_frame` on each, returns the median tilt from corners with confidence ≥ 0.5, along with the median confidence. ~2–3 s inline per video on CPU.

Emitted JSON:
```json
{"tiltDeg": 7.3, "courtConfidence": 0.82, "framesScored": 5}
```

Written to `Video.qualityReportJson` as raw metrics (`tiltDeg`, `courtConfidence`) alongside brightness.

**Trigger — AND-composed**

```ts
const shouldRotate =
  qualityReport.tiltDeg > 5 &&
  qualityReport.courtConfidence > 0.8 &&
  !qualityReport.autoRotated;  // idempotency
```

**Fix — FFmpeg filter during optimize**

`processingService.ts::optimizeVideo` today runs `ffmpeg` to produce the optimized mp4. Append the `rotate` filter to that invocation when `shouldRotate`:

```
-vf "rotate=<radians>:ow=iw:oh=ih:c=black"
```

- `ow=iw:oh=ih` keeps output dimensions identical to input.
- `c=black` pads rotated corners with black rather than cropping or scaling. No content loss; black triangles sit outside the court polygon where gameplay never happens.
- The rotation angle is the *negative* of the measured tilt (we're correcting, not replicating).

After successful optimize:
```ts
qualityReportJson.autoRotated = true;
qualityReportJson.tiltDeg = 0;  // the video really is 0° now
qualityReportJson.autoFixes = [{
  id: "auto_straightened",
  message: `Auto-straightened by ${Math.round(originalTilt)}°`,
  appliedAt: new Date().toISOString(),
  data: { originalTiltDeg: originalTilt },
}];
// Clear saved calibration — it was computed on the un-rotated frames
await prisma.video.update({ where: {id}, data: { courtCalibrationJson: null } });
```

**Idempotency**

The `autoRotated: true` flag in `qualityReportJson` prevents compounding rotations on retry. If `optimizeVideo` crashes mid-encode, the flag never gets written, so the next retry re-applies safely. If it succeeds, the flag blocks any subsequent rotation attempt.

**UX — autoFixes, not issues**

The auto-straighten message doesn't belong in `issues` — `issues` is "things the user may want to act on," which is the wrong shape for "here's what we already did." Schema addition:

```ts
type AutoFix = {
  id: "auto_straightened" | string;
  message: string;
  appliedAt: string;          // ISO timestamp
  data?: Record<string, number>;
};

interface QualityReport {
  version: 2;
  issues: Issue[];
  autoFixes?: AutoFix[];      // NEW
  preflight?: { ... };
  brightness?: number | null;
  resolution?: { ... };
  autoRotated?: boolean;      // NEW internal flag
  tiltDeg?: number | null;    // NEW metric
  courtConfidence?: number | null;  // NEW metric
}
```

Frontend: `QualityReportBanner` stays focused on issues. A lighter-weight sibling component renders `autoFixes` as a single-line checkmark note ("✓ Auto-straightened by 7°") or a one-shot toast on first open. Not banner real estate.

No new `Tier` enum value. `pickTopIssues` logic untouched. `mergeQualityReports` concatenates `autoFixes` arrays, deduping by `id` and keeping the first occurrence (mirrors the existing "first non-null wins" semantics for scalar fields).

**Merge caveat for state transitions.** `mergeQualityReports` uses "first non-null wins" for scalar fields, which is correct for combining reports from different sources (confirm + preflight) but wrong for *updating* state in place — writing `{autoRotated: true, tiltDeg: 0}` via merge against a prior `{autoRotated: false, tiltDeg: 7}` would leave the old non-null values intact. Post-rotate writes therefore use read-mutate-write, not merge: load `Video.qualityReportJson`, patch `autoRotated`, `tiltDeg`, append to `autoFixes`, write the whole object back in one transaction. The initial confirm-step write (which populates `tiltDeg` for the first time) *does* use merge — prior values are null there, so both strategies produce the same result.

**Court-calibration coupling**

Saved `Video.courtCalibrationJson` — if any — was computed on un-rotated frames and is invalidated by rotation. Clear it in the same transaction that sets `autoRotated: true`. This is edge-case: `courtCalibrationJson` is only set after detection runs in the editor, so it's almost never populated at confirm time.

**Failure mode — rotate ffmpeg crashes**

Log, fall through to optimize *without* the rotate filter, write `autoRotated: false`, and let preflight surface `video_rotated` back as an advisory ("We couldn't auto-straighten this — you can re-upload a straight version"). Never block the pipeline on a fix failure.

## Data Flow

```
[Upload]
file selected
  └─ browser extracts 5 frames (ffmpeg.wasm, existing)
     └─ POST /v1/videos/preflight-preview {frames[], metadata}
        ├─ rallycut preview-check:
        │   ├─ court-keypoint → wrong_angle_or_not_volleyball (A1 block)
        │   └─ open-clip beach-VB scoring → not_beach_volleyball (NEW A)
        ├─ pass → presigned upload URL issued
        └─ block → upload cancelled, UX explains reason

upload completes → /v1/videos/:id/confirm
  ├─ poster + brightness (existing)
  ├─ rallycut tilt-detect → writes {tiltDeg, courtConfidence, autoRotated: false} (NEW B)
  └─ optimize dispatch
     └─ if tiltDeg > 5° AND courtConfidence > 0.8 AND !autoRotated:
        ├─ ffmpeg -vf "rotate=...:c=black" (NEW B)
        ├─ clear Video.courtCalibrationJson
        └─ write qualityReportJson: {autoRotated: true, tiltDeg: 0,
                                     autoFixes: [{id: "auto_straightened", ...}]}

[Analyze — unchanged]
user clicks "Analyze Match"
  └─ existing preflight runs on the now-rotated optimized video
     └─ court-keypoint sees tilt ≈ 0 (already corrected)
     └─ standard A1 checks run; component B already did its work
```

## Calibration (ship gate for A)

A ships only if calibration produces a clean threshold.

1. Score the 5 A1 negatives (`bad angle`, `very bad angle`, `indoor 1`, `indoor 2`, `not related`) + 2 A1 positives (`match.mp4`, Newport Beach) + ~10 videos sampled from the 63-video GT — prefer borderline cases (low brightness, tournament setups, crowded scenes).
2. The gap between the highest-scoring negative and the lowest-scoring positive must be ≥ 0.15. Pick the threshold at `lowest_positive_score - 0.1`, rounded down to the nearest 0.05. Biased toward false-accept per the guiding principle.
3. If gap < 0.15, the classifier is dropped or prompts retuned. A1's experience shows "conservative default threshold with no calibration" ships six useless checks; we won't repeat that.

Deliverable: `analysis/reports/beach_vb_calibration_YYYY-MM-DD.json` with per-video scores, prompt text, threshold, and the lift number (P(not beach-vb | block fires) / P(not beach-vb | block doesn't fire)) for the record.

## Manufactured-Tilt Fixtures (ship gate for B)

B doesn't need statistical calibration — its thresholds are invariants — but it does need behavioral validation.

Generate from existing clean positives:
```bash
# inputs: match.mp4, Newport-Beach.mp4
# outputs: tilt_3deg.mp4, tilt_6deg.mp4, tilt_10deg.mp4, tilt_15deg.mp4
for deg in 3 6 10 15; do
  ffmpeg -i match.mp4 \
    -vf "rotate=${deg}*PI/180:ow=iw:oh=ih:c=black" \
    -c:a copy tilt_${deg}deg.mp4
done
```

Stored at `~/Desktop/rallies/Negative/tilt_*.mp4` alongside existing fixtures.

Expected behavior:
| Input tilt | tilt-detect output | shouldRotate | Final tilt |
|---|---|---|---|
| 3° | `tiltDeg ≈ 3` | false (below 5° floor) | ~3° (unchanged, barely perceptible) |
| 6° | `tiltDeg ≈ 6`, conf > 0.8 | true | ≤ 1° |
| 10° | `tiltDeg ≈ 10`, conf > 0.8 | true | ≤ 1° |
| 15° | `tiltDeg ≈ 15`, conf > 0.8 | true | ≤ 1° |

## Testing

### Unit

**A:**
- `test_quality_beach_vb.py` — update to the binary-prompt flow. Mock `embed_and_score_frames` to return known per-frame probs, verify: above-threshold passes, below-threshold blocks, empty list is no-op, `data.avgBeachVbProb` is the mean of inputs.

**B:**
- `test_quality_tilt_detect.py` (new) — stub `CourtKeypointDetector.detect_from_frame`, verify CLI emits the expected JSON and computes median correctly when frame confidences vary.
- `test_processing_service_rotate.ts` (new) — mock the FFmpeg child process, verify: rotate flag is added when both thresholds cross, omitted when either fails, omitted when `autoRotated: true`, output-dim flags (`ow=iw:oh=ih:c=black`) are present.
- `test_quality_report_merge.ts` — update for `autoFixes` concat-and-dedupe-by-id behavior.

### Integration

**A:** End-to-end preview check against fixture frames extracted from the 5 negatives + 2 positives. Smoke-run via an extended version of `analysis/scripts/validate_quality_checks.py` (which already handles the preflight path).

**B:** Upload a manufactured 10°-tilted fixture, run confirm twice (simulating retry). Assert the optimized output is rotated exactly once, final tilt ≈ 0°, `autoRotated: true`, `autoFixes` has one entry.

### Regression

- Existing A1 harness (`analysis/scripts/validate_quality_checks.py`) re-run on the 7 A1 fixtures: the 3 obvious negatives still blocked by `wrong_angle_or_not_volleyball` (court-confidence path unchanged); 2 additional negatives now blocked by `not_beach_volleyball`; 2 positives still pass.
- `analysis` pytest suite green.
- `api` `tsc --noEmit` green.

## Ship Criteria

1. All 5 A1 negative fixtures blocked (either by existing court-conf OR new CLIP).
2. 2 A1 positive fixtures + calibration positives pass both checks.
3. 10° tilt fixture: optimized output straight within ± 1°; downstream detection + tracking run on it with no regression vs the untilted original.
4. 3° tilt fixture (below threshold): optimize leaves it alone.
5. Idempotency test passes (double-confirm applies rotation once).
6. Analysis pytest green, API tsc green, pre-commit hook clean.
7. A1 validation harness report on the 7 A1 fixtures unchanged for positives and strengthened for negatives.

## Files to Change

| File | Change |
|---|---|
| `analysis/pyproject.toml` | Add `open-clip-torch` under `[project.optional-dependencies] preflight`. |
| `analysis/rallycut/quality/beach_vb_classifier.py` | Switch to binary prompts, wire into preview-check. |
| `analysis/rallycut/cli/commands/preview_check.py` | Call CLIP classifier after court-geometry, merge results. |
| `analysis/rallycut/cli/commands/tilt_detect.py` (new) | One-shot tilt-detect CLI that emits `{tiltDeg, courtConfidence}`. |
| `analysis/rallycut/quality/camera_geometry.py` | Re-add `_baseline_tilt_deg` helper (it was deleted when `video_rotated` was dropped). |
| `analysis/tests/unit/test_quality_beach_vb.py` | Update for binary prompts. |
| `analysis/tests/unit/test_quality_tilt_detect.py` (new) | CLI JSON-shape tests. |
| `analysis/scripts/calibrate_beach_vb.py` (new) | Scoring sweep; emits calibration JSON. |
| `api/src/services/processingService.ts` | Call tilt-detect during confirm; add rotate filter to optimize; set `autoRotated` + `autoFixes`; clear `courtCalibrationJson` on rotate. |
| `api/src/services/qualityReport.ts` | Add `autoFixes`, `autoRotated`, `tiltDeg`, `courtConfidence` fields to the `QualityReport` type; concat `autoFixes` in `mergeQualityReports`. |
| `api/tests/qualityService.test.ts` | Coverage for `autoFixes` merge + rotate filter trigger conditions. |
| `web/src/types/rally.ts` | Add `autoFixes` to `QualityReport` TypeScript type. |
| `web/src/components/QualityReportBanner.tsx` OR sibling | Render auto-fixes as lightweight checkmark note / toast. |

## Open Questions (resolved during brainstorm, recorded for reference)

- **Runtime placement for A**: client-side gate only, not server-side too. Simpler architecture; failure mode degrades to A1. Post-ship: if a case slips through that would have been caught by a preflight-stage run, add the preflight fallback then.
- **Prompt strategy for A**: binary, not 3-way. Cleaner signal, easier calibration.
- **UX for B**: silent server-side rotate, not user-confirmed. The AND-composed thresholds gate the safety-critical case.
- **auto-rotate as issue vs. auto-fix**: new `autoFixes` field in `qualityReportJson`. Doesn't pollute the top-3 issue feed; no new `Tier` enum value.
- **Auto-rotate surface**: server-side only. ffmpeg.wasm client-side is too slow on 30-minute matches.
- **Rotation padding**: black pad (no content loss) over crop (loses corners) or scale (distorts).
- **Scope**: A + B bundled in one branch + one PR. They don't interact; merging separately adds ceremony without benefit.

## Out of Scope / Follow-ups

- Other sport-type disambiguation beyond beach / not-beach.
- Non-tilt rotation correction (e.g., ffprobe `rotate` metadata at upload — already handled today by some players).
- Undo-auto-rotate UX control. Add only if users complain that false rotations happen in production.
- S3 backup of the negative + manufactured-tilt fixtures. Deferred; fixtures live locally. Revisit if a second machine or team member needs them.
- Preflight-stage CLIP fallback for cases that slip the 5-frame preview gate. Add if observed in production.
