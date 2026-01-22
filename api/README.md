# RallyCut API

REST API backend for the RallyCut video editing platform.

## Stack

- **Framework**: Express.js with TypeScript
- **Database**: PostgreSQL with Prisma ORM
- **Storage**: AWS S3 for videos, CloudFront for streaming
- **Validation**: Zod schemas
- **Deployment**: Serverless Framework (AWS Lambda) or Docker

## Getting Started

```bash
# Start PostgreSQL
docker-compose up -d

# Install dependencies
npm install

# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev

# Start development server
npm run dev
```

Server runs at [http://localhost:3001](http://localhost:3001).

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/rallycut"

# AWS S3
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
S3_BUCKET_NAME="rallycut-videos"

# CloudFront (optional, for signed cookies)
CLOUDFRONT_DOMAIN="xxx.cloudfront.net"
CLOUDFRONT_KEY_PAIR_ID="..."
CLOUDFRONT_PRIVATE_KEY="..."

# Modal ML Service
MODAL_WEBHOOK_SECRET="..."
```

## Project Structure

```
src/
├── index.ts              # Express app entry point
├── lambda.ts             # AWS Lambda handler
├── config/
│   └── env.ts            # Environment validation
├── lib/
│   ├── prisma.ts         # Prisma client
│   ├── s3.ts             # S3 operations
│   ├── cloudfront.ts     # Signed cookies
│   └── modal.ts          # Modal API client
├── middleware/
│   ├── errorHandler.ts   # Error handling
│   ├── requestLogger.ts  # Request logging
│   └── validateRequest.ts # Zod validation
├── routes/
│   ├── sessions.ts       # Session CRUD + sync
│   ├── videos.ts         # Video upload + status
│   ├── rallies.ts        # Rally CRUD
│   ├── highlights.ts     # Highlight CRUD
│   ├── detection.ts      # Trigger ML detection
│   └── webhooks.ts       # Modal callbacks
├── schemas/
│   ├── common.ts         # Shared schemas
│   ├── session.ts        # Session validation
│   ├── video.ts          # Video validation
│   ├── rally.ts          # Rally validation
│   ├── highlight.ts      # Highlight validation
│   ├── batch.ts          # Batch operations
│   └── sync.ts           # State sync
└── services/
    ├── sessionService.ts # Session logic
    ├── videoService.ts   # Video + S3 logic
    ├── rallyService.ts   # Rally logic
    ├── highlightService.ts # Highlight logic
    ├── detectionService.ts # ML job management
    ├── batchService.ts   # Batch updates
    └── syncService.ts    # Full state sync
```

## API Endpoints

### Sessions
- `GET /v1/sessions` - List sessions (paginated)
- `POST /v1/sessions` - Create session
- `GET /v1/sessions/:id` - Get session with videos, rallies, highlights
- `PATCH /v1/sessions/:id` - Update session
- `DELETE /v1/sessions/:id` - Delete session and all data

### Videos
- `GET /v1/videos` - List user's videos
- `POST /v1/videos/upload-url` - Get presigned upload URL
- `POST /v1/videos/:id/confirm` - Confirm upload complete
- `GET /v1/videos/:id/editor` - Get video for editor with rallies
- `DELETE /v1/videos/:id` - Delete video

### Detection
- `POST /v1/videos/:id/detect-rallies` - Start ML detection job
- `GET /v1/videos/:id/detection-status` - Get detection status

### Webhooks
- `POST /v1/webhooks/detection-progress` - ML progress updates
- `POST /v1/webhooks/detection-complete` - ML job complete

### Sync
- `POST /v1/sessions/:id/sync-state` - Sync full rally/highlight state

## Data Model

```prisma
model Session {
  id         String      @id @default(uuid())
  name       String
  videos     Video[]
  highlights Highlight[]
  createdAt  DateTime    @default(now())
  updatedAt  DateTime    @updatedAt
}

model Video {
  id              String   @id @default(uuid())
  sessionId       String
  originalName    String
  s3Key           String
  proxyS3Key      String?
  duration        Float?
  fps             Float?
  width           Int?
  height          Int?
  status          VideoStatus @default(UPLOADED)
  rallies         Rally[]
  session         Session  @relation(...)
}

model Rally {
  id       String @id @default(uuid())
  videoId  String
  startMs  Int
  endMs    Int
  order    Int
  video    Video  @relation(...)
}

model Highlight {
  id              String           @id @default(uuid())
  sessionId       String
  name            String
  color           String
  highlightRallies HighlightRally[]
  session         Session          @relation(...)
}
```

## Deployment

### Docker
```bash
docker build -t rallycut-api .
docker run -p 4000:4000 --env-file .env rallycut-api
```

### AWS Lambda
```bash
npx serverless deploy
```

## Development

```bash
# Type check
npx tsc --noEmit

# Lint
npm run lint

# Prisma Studio (database GUI)
npx prisma studio
```
