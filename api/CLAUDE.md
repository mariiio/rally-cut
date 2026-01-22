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
| Detections/month | 2 | 15 | 50 |
| Monthly uploads | 3 | 20 | 50 |
| Max video duration | 15 min | 45 min | 90 min |
| Max file size | 500 MB | 2 GB | 5 GB |
| Storage cap | 1 GB | 20 GB | 75 GB |
| Export quality | 720p + watermark | Original | Original |
| Server export | No (browser only) | Lambda | Lambda |
| Server sync | No (localStorage only) | Yes | Yes |
| Original quality retention | 3 days | 14 days | 60 days |
| Inactivity deletion | 30 days | 6 months | 1 year |

**Configuration**: All limits defined in `src/config/tiers.ts` (single source of truth).
**Enforcement**: `tierService.ts` checks limits, `getUserTier()` resolves tier from user.
**Expiration**: Paid tier expires → auto-downgrade to FREE.

### Retention Policy

Videos follow a two-phase cleanup based on tier:

1. **Original quality downgrade**: After `originalQualityDays` (FREE: 3, PRO: 14, ELITE: 60):
   - Original/optimized quality deleted, video accessible at 720p proxy only

2. **Inactivity deletion**: After `inactivityDeleteDays` inactive (FREE: 30, PRO: 180, ELITE: 365):
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
2. **Optimization**: Async H.264 + faststart if needed (high bitrate or moov not at start)
3. **Proxy**: 720p version for faster editing

Outputs: `{base}_poster.jpg`, `{base}_optimized.mp4`, `{base}_proxy.mp4`

### Detection
- `POST /v1/videos/:id/detect-rallies` → triggers Modal ML
  - Body: `{ model?: "indoor" | "beach" }` - model variant (default: indoor)
- Progress via `POST /v1/webhooks/detection-progress`
- Completion via `POST /v1/webhooks/detection-complete`
- **Model variants**: `indoor` (original) or `beach` (fine-tuned with optimized heuristics)
- **Deduplication**: Same `contentHash` reuses existing detection results

### State Sync
- `POST /v1/sessions/:id/sync-state` receives full rally/highlight state
- Reconciles: creates new, updates existing, deletes removed
- **Permissions**: Members can view rallies but not edit them

### Export
- `POST /v1/export-jobs` → triggers Lambda or local FFmpeg
- Supports camera edits (keyframes per rally)
- FREE = 720p + watermark (browser export), PRO/ELITE = original quality (server export)
- Poll `GET /v1/export-jobs/:id` for status

### Session Sharing
- `POST /v1/sessions/:id/share` → creates share token
- `POST /v1/share/:token/accept` → join as member
- Members can add their own highlights, view rallies (read-only)

## Database Schema

```
User → Session[] → Video[] → Rally[]
     → Highlight[] → HighlightRally[] → Rally

Rally → RallyCameraEdit → CameraKeyframe[]  # Instagram-style zoom/pan

Video.status: PENDING → UPLOADED → DETECTING → DETECTED → ERROR
Video.processingStatus: PENDING → PROCESSING → COMPLETED/FAILED (separate from status)
```

Key tables: User, Session, Video, Rally, Highlight, ExportJob, RallyDetectionJob, SessionShare

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
