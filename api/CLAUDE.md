# RallyCut API

REST API backend for the RallyCut video editing platform.

## Stack

- Express.js with TypeScript
- Prisma ORM with PostgreSQL
- Zod for validation
- AWS S3 + CloudFront for video storage/streaming

## Commands

```bash
npm run dev              # Development server (port 4000)
npm run build            # Compile TypeScript
npx tsc --noEmit         # Type check
npx prisma studio        # Database GUI
npx prisma migrate dev   # Run migrations
```

## Structure

```
src/
├── index.ts           # Express app setup
├── lambda.ts          # AWS Lambda entry
├── routes/            # REST endpoints
│   ├── sessions.ts    # Session CRUD + sync-state
│   ├── videos.ts      # Upload URLs, status
│   ├── rallies.ts     # Rally CRUD
│   ├── highlights.ts  # Highlight CRUD
│   ├── detection.ts   # Start ML jobs
│   ├── exports.ts     # Video export jobs
│   └── webhooks.ts    # Modal/Lambda callbacks
├── services/          # Business logic
│   ├── syncService.ts # Full state sync (rallies + highlights)
│   ├── detectionService.ts # ML job management
│   └── exportService.ts # Video export (Lambda or local FFmpeg)
├── schemas/           # Zod validation schemas
├── middleware/        # Error handling, validation
└── lib/               # S3, CloudFront, Prisma clients
```

## Key Patterns

### State Sync
- `POST /v1/sessions/:id/sync-state` receives full rally/highlight state
- Backend reconciles: creates new, updates existing, deletes removed
- Frontend IDs (`videoId_rally_n`) mapped to backend UUIDs

### Video Upload Flow
1. Frontend requests presigned URL via `POST /v1/videos/upload-url`
2. Frontend uploads directly to S3
3. Frontend confirms via `POST /v1/videos/:id/confirm-upload`
4. Backend triggers ML detection on Modal

### Detection Webhooks
- Modal sends progress updates to `POST /v1/webhooks/detection/progress`
- On completion, sends results to `POST /v1/webhooks/detection/complete`
- Results stored as rallies with `order` field for sequencing

### Video Processing (Optimization)
After upload confirmation, videos are processed for faster web playback:

1. **Poster generation** - 1280px JPEG thumbnail for instant display
2. **Video optimization** - H.264 + faststart if needed (high bitrate or moov atom not at start)
3. **Proxy generation** - 720p version for faster editing

**Outputs stored in S3:**
- `{baseKey}_poster.jpg` - Poster image (~50KB)
- `{baseKey}_optimized.mp4` - Optimized full-quality video
- `{baseKey}_proxy.mp4` - 720p proxy for editing

**Local development**: Set `PROCESSING_LAMBDA_FUNCTION_NAME` to empty to use local FFmpeg.

### Video Export
- `POST /v1/export-jobs` creates export job and triggers Lambda (or local FFmpeg)
- `GET /v1/export-jobs/:id` returns job status/progress (polling)
- `GET /v1/export-jobs/:id/download` returns presigned download URL
- Lambda sends completion webhook to `POST /v1/webhooks/export-complete`
- **Tiers**: FREE (720p + watermark), PREMIUM (original quality)
- **Local fallback**: If `EXPORT_LAMBDA_FUNCTION_NAME` not set, uses local FFmpeg

## Database Schema

```prisma
Session -> Video[] -> Rally[]
        -> Highlight[] -> HighlightRally[] -> Rally
```

- `Rally.order` determines sequence within video
- `HighlightRally.order` determines sequence within highlight
- Soft cascade: deleting session removes all related data

## Code Style

- TypeScript strict mode
- Zod schemas in `schemas/` folder
- Services handle business logic, routes handle HTTP
- All routes wrapped in try/catch with error middleware

## Environment

Required in `.env`:
- `DATABASE_URL` - PostgreSQL connection
- `AWS_*` - S3 credentials and bucket
- `CLOUDFRONT_*` - Optional, for signed cookies
- `MODAL_WEBHOOK_SECRET` - Webhook authentication

Optional (for server-side export):
- `EXPORT_LAMBDA_FUNCTION_NAME` - Lambda function name (omit for local FFmpeg)
- `API_BASE_URL` - Base URL for Lambda callbacks
