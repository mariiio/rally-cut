# RallyCut API

REST API backend for the RallyCut video editing platform.

## Stack

- Express.js with TypeScript
- Prisma ORM with PostgreSQL
- Zod for validation
- AWS S3 + CloudFront for video storage/streaming

## Commands

```bash
npm run dev              # Development server (port 3001)
npm run build            # Compile TypeScript
npx tsc --noEmit         # Type check
npx prisma studio        # Database GUI
npx prisma migrate dev   # Run migrations
```

## Structure

```
src/
├── index.ts           # Express app setup
├── routes/            # REST endpoints (sessions, videos, rallies, highlights, exports, webhooks, shares)
├── services/          # Business logic (sync, detection, export, processing, tier, share)
├── schemas/           # Zod validation schemas
├── middleware/        # Error handling, validation, user resolution
├── lib/               # S3, CloudFront, Prisma clients
└── config/env.ts      # Zod-validated environment config
prisma/
├── schema.prisma      # Database schema
└── migrations/        # Migration history
```

## User Identification

All endpoints require `X-Visitor-Id` header (UUID):
- Anonymous users auto-created on first request
- Maps to User via AnonymousIdentity table
- Returns 400 if header missing (except health/webhooks)

## Security

- All `/v1/*` routes must use `requireUser` middleware (except webhooks)
- Service functions must verify `resource.userId === userId` before mutations
- Use `crypto.timingSafeEqual()` for secret comparison (prevents timing attacks)
- Use `env.CORS_ORIGIN` for CORS headers, never `*`
- Wrap check-then-act patterns in `prisma.$transaction()` to prevent race conditions

## Tier System

| Feature | FREE | PRO ($9.99) | ELITE ($24.99) |
|---------|------|-------------|----------------|
| Detections/month | 2 | 15 | 40 |
| Monthly uploads | 5 | 20 | 50 |
| Max video duration | 30 min | 60 min | 90 min |
| Max file size | 500 MB | 3 GB | 5 GB |
| Storage cap | 2 GB | 20 GB | 75 GB |
| Export quality | 720p + watermark | Original | Original |
| Server export | No (browser only) | Lambda | Lambda |
| Server sync | No (localStorage only) | Yes | Yes |
| Original quality retention | 7 days | 14 days | 60 days |
| Inactivity deletion | 60 days | Unlimited | Unlimited |

**Configuration**: All limits defined in `src/config/tiers.ts` (single source of truth).
**Enforcement**: `tierService.ts` checks limits, `getUserTier()` resolves tier from user.
**Expiration**: Paid tier expires → auto-downgrade to FREE.

### Retention Policy

Videos follow a two-phase cleanup based on tier:

1. **Original quality downgrade**: After `originalQualityDays` (FREE: 7, PRO: 14, ELITE: 60):
   - Original/optimized quality deleted, video accessible at 720p proxy only

2. **Inactivity deletion**: After `inactivityDeleteDays` inactive (FREE: 60, PRO/ELITE: null = never while subscribed):
   - All content hard deleted (videos, sessions, S3 files)

**Storage quota**: Enforced per-user, calculated from all video `fileSizeBytes`. Upload blocked when over cap.

Activity tracking: User's `lastActiveAt` updated (1hr debounce) when accessing sessions.
Cleanup job: `cleanupExpiredContent()` handles both phases per tier.

## Key Flows

### Video Upload
1. `POST /v1/videos/upload-url` → presigned S3 URL (validates quota, duration, size)
2. Frontend uploads directly to S3
3. `POST /v1/videos/:id/confirm` → triggers processing + detection
4. **Multipart** (files >100MB): Use `/multipart/init`, upload parts, `/multipart/complete`

### Video Processing (after upload)
1. **Poster**: Synchronous 1280px JPEG extraction (~2s)
2. **Brightness**: Computed during poster generation (grayscale frame sampling), stored in `Video.qualityReportJson.brightness`
3. **Optimization**: Async H.264 + faststart if needed (high bitrate or moov not at start)
4. **Proxy**: 720p version for faster editing

Outputs: `{base}_poster.jpg`, `{base}_optimized.mp4`, `{base}_proxy.mp4`

### Detection
- `POST /v1/videos/:id/detect-rallies` → triggers Modal ML
  - Body: `{ model?: "indoor" | "beach" }` - model variant (default: beach)
- Progress via `POST /v1/webhooks/detection-progress`
- Completion via `POST /v1/webhooks/detection-complete`
- **Model variants**: `beach` (default) or `indoor` (original)
- **Deduplication**: Same `contentHash` reuses existing detection results

### State Sync
- `POST /v1/sessions/:id/sync-state` receives full rally/highlight state
- Reconciles: creates new, updates existing, deletes removed
- **Permissions**: Role-based — Viewers are read-only, Editors/Admins can modify rallies and highlights

### Export
- `POST /v1/export-jobs` → triggers Lambda or local FFmpeg
- Supports camera edits (keyframes per rally)
- FREE = 720p + watermark (browser export), PRO/ELITE = original quality (server export)
- Poll `GET /v1/export-jobs/:id` for status

### Session Sharing
- `POST /v1/sessions/:id/share` → creates 3 share links (one per role: VIEWER, EDITOR, ADMIN)
- `GET /v1/sessions/:id/share` → returns all share links + members
- `POST /v1/share/:token/accept` → join as member with the role from the link used
- **Roles**: VIEWER (read-only), EDITOR (edit rallies/highlights), ADMIN (manage members/share)
- **Multiple links**: Each link grants exactly one role. Share the appropriate link based on desired access level.
- `PATCH /v1/sessions/:id/share/members/:userId/role` → change member role after joining

### Video Sharing
- `POST /v1/videos/:id/share` → creates 3 share links (one per role: VIEWER, EDITOR, ADMIN)
- `GET /v1/videos/:id/share` → returns all share links + members
- `DELETE /v1/videos/:id/share` → revoke all access (owner only)
- `DELETE /v1/videos/:id/share/members/:userId` → remove specific member
- `PATCH /v1/videos/:id/share/members/:userId/role` → change member role
- **Unified accept**: `GET /v1/share/:token` and `POST /v1/share/:token/accept` detect share type (session vs video)
- **Permission checks**: `canAccessVideoRallies()` checks both direct video membership and session membership

### Analysis Pipeline
- `POST /v1/videos/preflight-preview` → client-side pre-upload gate
  - Multipart: `frames[]` (JPEG thumbnails, up to 10 × 2 MB each) + `width`, `height`, `durationS` form fields
  - Runs `rallycut preview-check` on the first frame: court-keypoint detection → `camera_geometry` check
  - Returns `{ pass: boolean, issues: Issue[] }`; web aborts the upload if any `block`-tier issue fires
  - **No Video row required** — this endpoint runs before upload commits
- `POST /v1/videos/:id/assess-quality` → full preflight quality check (runs on "Analyze Match")
  - Downloads video from S3, runs `rallycut preflight` → metadata invariants + camera geometry (court-keypoint confidence + behind-baseline heuristic)
  - Merges result into `Video.qualityReportJson`; sets `Video.status = REJECTED` if any block-tier issue fires, reverts REJECTED → UPLOADED if a re-run passes
- `GET /v1/videos/:id/analysis-pipeline-status` → unified pipeline status (quality, detection, tracking, stats)
- **Service**: `qualityService.ts` (entry points + CLI spawning) + `qualityReport.ts` (pure merge/pick-top-3 helpers, test-safe)
- **Checks shipped** (post-validation 2026-04-15): hard-invariant metadata (`video_too_short`, `resolution_too_low`, `fps_too_low`) + camera-geometry block (`wrong_angle_or_not_volleyball`, lift 2.13 at courtConfidence<0.6). The original A1 set also shipped `camera_too_far`, `crowded_scene`, `shaky_camera`, `too_dark`, `overexposed`, and `video_rotated`; all six were dropped after calibration (`analysis/reports/quality_calibration_2026-04-14.json`) showed zero or negative lift and a validation sweep against 5 negative + 2 positive fixtures reproduced `camera_too_far` as a false positive on normal footage.
- **Project C additions** (branch `feat/c-sport-sanity`):
  - Component A: `not_beach_volleyball` block at the pre-upload gate via open-clip ViT-B/32 zero-shot (5-way positive-anchored prompts, threshold 0.886). Closes the indoor/non-VB slip-through A1 could not (beach-trained court model returned 0.78–0.83 confidence on non-VB content).
  - Component B: silent auto-rotate during the local FFmpeg optimize pass when `tiltDeg > 5° AND courtConfidence > 0.6` (wired in `processingService.triggerLocalProcessing`; Lambda path not yet wired). Adds `qualityReportJson.autoFixes[]` — a separate field from `issues[]` — logging "auto-straightened by N°". Tilt detection runs during confirm via `runTiltDetect` spawning `rallycut tilt-detect`.
  - Known gaps: `very bad angle.mp4` slips pre-upload gate (single-frame court check clears 0.6) but is still blocked by the 60s preflight CLI server-side; court-keypoint model's tilt measurement collapses at >~10° rotation — fine-tune on rotated data to close.

### Court Calibration
- `PUT /v1/videos/:id/court-calibration` → save 4 corner points (persisted per video)
  - Body: `{ corners: [{x,y}]x4 }` - normalized coordinates (0-1)
- `DELETE /v1/videos/:id/court-calibration` → clear calibration
- Calibration returned in `GET /v1/videos/:id/editor` response as `courtCalibrationJson`

### Player Tracking
- `POST /v1/rallies/:id/track-players` → triggers local YOLO + ByteTrack tracking
  - Body: `{ calibrationCorners?: [{x,y}]x4 }` - optional court calibration
  - Calibration: uses frontend-provided corners, falls back to DB (courtCalibrationJson), or CLI auto-detection (confidence > 0.4)
  - Returns player positions, ball trajectory, contacts, actions, and quality report
- `GET /v1/rallies/:id/player-track` → retrieves existing tracking data
- `POST /v1/rallies/:id/player-track/swap` → swap two track IDs from a frame onward
  - Body: `{ trackA: number, trackB: number, fromFrame: number }`
  - Fixes YOLO+BoT-SORT ID switches when players overlap/cross paths
  - Only modifies `positionsJson` (filtered positions), not `rawPositionsJson`
- Post-tracking, no direct writes to `Video.qualityReportJson` occur (tracking metrics are rally-scoped in `PlayerTrack.qualityReportJson`). Earlier cameraDistance/sceneComplexity writes were removed in A1 — nothing in the web banner reads them.
- **Batch tracking**: `POST /v1/videos/:id/track-all-rallies` tracks all confirmed rallies
  - If `MODAL_TRACKING_URL` is set: sends to Modal T4 GPU (~80 FPS, ~$0.02/batch)
  - If not set: processes locally on CPU (~6 FPS, blocks API server)
  - Modal path sends per-rally webhooks (`/v1/webhooks/tracking-rally-complete`) for progressive DB updates
  - Batch completion webhook (`/v1/webhooks/tracking-batch-complete`) updates batch status; match analysis is triggered client-side after a 5-second idle window (Project A2a)
  - `POST /v1/videos/:id/track-untracked` (A2a): wraps `trackAllRallies` with `{skipTracked:true}` — filters `playerTrack: null` so only rallies without existing tracking are queued. Returns `{jobId: string | null, totalRallies: number}`. jobId is null when nothing needs tracking. Used by the client-side debounce to catch up on rallies created mid-batch before firing match-analysis.
  - `POST /v1/videos/:id/trigger-match-analysis` (A2a): fire-and-forget endpoint that wraps `runMatchAnalysis` with an in-memory running-set guard. Returns 202 on success, 409 with `details.reason='MATCH_ANALYSIS_IN_PROGRESS'` if already running (client treats 409 as no-op). Replaces the auto-trigger that used to live inside the Modal `tracking-batch-complete` webhook.
  - **Match analysis pipeline**: validate-rallies → match-players → repair-identities → remap-track-ids → reattribute-actions → compute-match-stats (all best-effort, non-fatal)
  - **Rally validation**: Demotes ball-pass FPs to SUGGESTED (rejectionReason=BALL_PASS) using post-tracking signals (contact count, serve detection, duration). Skips user-modified rallies and rallies with low ball detection rate.

### Resilience (A2b)
- **Webhook idempotency**: every Modal webhook handler is wrapped in a `tryRecordDelivery` gate (see `services/webhookIdempotency.ts`). The key is `body.deliveryId` if provided by Modal, else a SHA-256 canonical fingerprint over `(webhookPath, body)` with keys sorted so key-insertion order doesn't affect the result. Duplicates return 200 `{deduplicated: true}` with zero side effects. Records are pruned by the 5-min sweeper after 7 days.
- **EXTEND-rally retrack**: when a rally's bounds are extended (start moved earlier OR end moved later), `markRetrackIfExtended` flips `PlayerTrack.needsRetrack=true` inside the same transaction as the rally update. The catch-up pipeline (`trackAllRallies({skipTracked:true})`) picks up such rallies alongside rallies without any `PlayerTrack`. Cleared on the next successful `saveTrackingResult`.
- **Stale-job sweeper**: a 5-minute `setInterval` (see `jobs/staleJobSweeper.ts`) sweeps `BatchTrackingJob` + `RallyDetectionJob` globally, expiring jobs with no progress in 10 minutes. Also prunes old `WebhookDelivery` rows. Skipped under `NODE_ENV=test`. `unref()`'d so it doesn't block shutdown.

### Rally-Edit Propagation (Project B)
- **Split**: `POST /v1/rallies/:id/split` with `{firstEndMs, secondStartMs}`. Cut-with-gap: middle frames `[firstEndFrame, secondStartFrame)` are discarded. Slices the existing `PlayerTrack` JSON arrays in place (`slicePlayerTrack` in `services/rallySlicing.ts`) — back-half frame indices shift down by `secondStartFrame`. Inheritance: first child scores default from the previous rally's end-state (or 0/0), second child inherits parent's scores/`servingTeam`; both children duplicate `notes` and `confidence`. Both children receive the parent's `trackToPlayer` mapping (raw track IDs are preserved by slicing, so Hungarian output stays valid); children are marked `canonicalLocked: false`.
- **Merge**: `POST /v1/rallies/merge` with `{rallyIds: [string, string]}`. Two sub-cases. **No gap** (`a.endMs === b.startMs`): concatenates both `PlayerTrack`s with B's frame indices shifted up by `a.frameCount` (`concatPlayerTracks` in `services/rallySlicing.ts`). **With gap**: drops both parent `PlayerTrack`s via cascade, new merged rally is created with `PlayerTrack=null` so the A2a catch-up pipeline retracks the full merged span. Both sub-cases record `editKind: 'merge'` → full rerun on next match-analysis trigger.
- **Unlock**: `POST /v1/rallies/:id/unlock` clears `Video.matchAnalysisJson.rallies[i].canonicalLocked = false`. Does NOT wipe `PlayerTrack.groundTruthJson` / `actionGroundTruthJson` — GT data is preserved for possible re-lock. Idempotent.
- **Delete + confirm**: `DELETE /v1/rallies/:id` accepts optional `{confirmUnlock: true}` body. Canonical-locked rallies return 409 `LOCKED_RALLY_REQUIRES_CONFIRM` with payload `{gtFrameCount}` unless the flag is set; when set, an audit log line `rally.locked.deleted` is emitted. Delete also prunes the rally's entry from `matchAnalysisJson.rallies[]` so stats don't reference a dead rally.
- **Canonical-lock guard**: `services/canonicalLockGuard.ts` exports `isRallyLocked` + `assertNotLocked(tx, rallyId, op)`. Called inside every structural edit path (`updateRally` extend branch, `splitRally`, `mergeRallies`) to throw `LockedRallyError` (409) when the rally is locked.
- **Edit-type-gated match analysis**: rally CRUD writes `{rallyId, editKind, at}` markers to `Video.pendingAnalysisEditsJson` as the last mutation inside its transaction (`appendEdit` / `appendEditsBatch` in `services/pendingAnalysisEdits.ts`). `runMatchAnalysis` starts by calling `consumePendingEdits(videoId)` (single-transaction read-then-null), passes the entries to `planStages` (`services/matchAnalysisPlanning.ts`). `extend | create | merge` → `fullRerun: true` (run all 6 stages). `scalar | delete | shorten | split`-only → skip stages 2 (match-players) and 3 (repair-identities); stages 4/5 run only for `changedRallyIds` using the new `--rally-ids <csv>` flag on the Python CLIs. Stage 6 (compute-match-stats) always runs.
- **Stage-timing telemetry**: every `runMatchAnalysis` run emits a single `match_analysis.stage_timings` structured JSON log line with `{videoId, plan, timingsMs, totalMs}`. Use to quantify whether warm-start Hungarian is worth future work.
- **State matrix** for structural edits (split / merge / extend): forbidden when `PlayerTrack.status` is `PROCESSING` (409 `RALLY_TRACKING_IN_PROGRESS`) or `FAILED` (409 `RALLY_TRACKING_FAILED`). Allowed when `null`, `COMPLETED`, or `COMPLETED + needsRetrack=true` (children inherit the flag on split; retrack fires after).
- `Video.matchAnalysisRanAt` is written on every successful run — useful for staleness checks and cross-reference against the stage-timings log.

### Label Studio Integration (Ground Truth)
- `GET /v1/rallies/:id/label-studio` → status (hasTrackingData, hasGroundTruth, taskId)
- `POST /v1/rallies/:id/label-studio/export` → exports tracking to Label Studio
  - Body: `{ videoUrl, apiKey?, apiUrl?, forceRegenerate? }` - video URL (must use API server URL for CORS)
  - Returns: `{ taskUrl }` - opens in Label Studio with pre-filled predictions
  - **Labels**: player_1 (green), player_2 (blue), player_3 (orange), player_4 (purple), ball (red)
  - **Task reuse**: Returns existing task if already exported, unless `forceRegenerate: true`
  - **Rally bounds**: Labels only appear during rally duration (enabled: false after end)
  - **Frame timing**: Frames calculated at 30fps (Label Studio's default) regardless of video fps
- `POST /v1/rallies/:id/label-studio/import` → imports corrected annotations
  - Body: `{ taskId, apiKey?, apiUrl? }` - Label Studio task ID
  - Saves to `PlayerTrack.groundTruthJson`
- **Config**: `LABEL_STUDIO_URL` (default: localhost:8082), `LABEL_STUDIO_API_KEY`
- **Project**: Auto-created "RallyCut Ground Truth" on first export

## Database Schema

```
User → Session[] → Video[] → Rally[]
     → Highlight[] → HighlightRally[] → Rally

Rally → RallyCameraEdit → CameraKeyframe[]  # Instagram-style zoom/pan

Video.status: PENDING → UPLOADED → DETECTING → DETECTED → ERROR
              (plus REJECTED, set by preflight on block-tier issues; reverts to UPLOADED if re-run passes)
Video.processingStatus: PENDING → PROCESSING → COMPLETED/FAILED (separate from status)
Video.qualityReportJson: { version: 2, issues: Issue[], autoFixes?: AutoFix[], preflight?, brightness?, resolution?, autoRotated?, tiltDeg?, courtConfidence? }
```

Key tables: User, Session, Video, Rally, Highlight, ExportJob, RallyDetectionJob, SessionShare, VideoShare

## Camera Edits

Per-rally camera effects for Instagram-style exports:
- `RallyCameraEdit`: aspectRatio (ORIGINAL/VERTICAL), enabled flag
- `CameraKeyframe`: timeOffset (0-1), positionX/Y, zoom (1-3), easing

Stored in database, exported via Lambda with FFmpeg crop filters.

## Environment

Required:
```
DATABASE_URL=postgresql://...
AWS_REGION, S3_BUCKET_NAME
MODAL_WEBHOOK_SECRET
CORS_ORIGIN
```

Optional (local dev uses fallbacks):
```
S3_ENDPOINT=http://localhost:9000          # MinIO
MODAL_TRACKING_URL                         # Omit for local CPU tracking
EXPORT_LAMBDA_FUNCTION_NAME                # Omit for local FFmpeg
PROCESSING_LAMBDA_FUNCTION_NAME            # Omit for local FFmpeg
CLOUDFRONT_DOMAIN, CLOUDFRONT_KEY_PAIR_ID  # For signed URLs
```

## Caveats

- **Two status fields**: `Video.status` (upload/detection lifecycle) vs `Video.processingStatus` (optimization). Both can be in different states.
- **Soft vs hard delete**: Sessions use soft delete (`deletedAt`), videos have both soft delete AND hard `DELETE` endpoint
- **Fire-and-forget Lambda**: API returns 202 immediately, relies on webhooks for completion
- **Content hash**: SHA-256 of first+last 10MB + metadata for large file deduplication
- **Rally confirmation**: Separate feature that creates trimmed videos (PRO/ELITE only), stores timestamp mappings
