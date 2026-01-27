# RallyCut

Volleyball video analysis platform. Monorepo with ML pipeline, web editor, and REST API.

## Projects

| Directory | Purpose | Stack |
|-----------|---------|-------|
| `analysis/` | ML video analysis CLI | Python, PyTorch, VideoMAE, YOLO |
| `web/` | Rally editor frontend | Next.js 15, React 19, MUI, Zustand |
| `api/` | REST backend | Express, Prisma, PostgreSQL |
| `lambda/` | Serverless video export | AWS Lambda, FFmpeg |

## Which Directory?

- **ML/detection logic** → `analysis/`
- **UI components, state** → `web/`
- **API routes, database** → `api/`
- **Video export processing** → `lambda/`

## Development

```bash
make setup    # First-time: install deps, start services, migrate
make dev      # Start local dev (db + minio + api + web)
make stop     # Stop all services
```

## Local Development with MinIO

RallyCut supports fully local development without AWS dependencies using MinIO for S3-compatible storage.

### Quick Start

```bash
make setup    # First time: install deps, create .env, start services, migrate
make dev      # Start all services (auto-creates .env if missing)
```

### Development Modes

| Mode | Command | Storage | Database | Video Processing |
|------|---------|---------|----------|------------------|
| **Local** | `make dev` | MinIO | Local PostgreSQL | Local FFmpeg |
| **Production-local** | `make dev-prod` | AWS S3 | Production DB | AWS Lambda |

For production-local mode, create `api/.env.prod` from `api/.env.prod-local.example` with your credentials.

### Local Services

| Service | URL | Credentials |
|---------|-----|-------------|
| PostgreSQL | localhost:5436 | postgres/postgres |
| MinIO S3 API | localhost:9000 | minioadmin/minioadmin |
| MinIO Console | localhost:9001 | minioadmin/minioadmin |
| API | localhost:3001 | - |
| Web | localhost:3000 | - |

### MinIO Management

```bash
make minio-console  # Open MinIO web UI
make reset-storage  # Clear all stored files
```

### Local Processing

In local development mode:
- **Upload**: Presigned URLs work with MinIO
- **Optimization**: Local FFmpeg runs in API process
- **Streaming**: API proxies video requests from MinIO
- **Export**: Local FFmpeg handles rally extraction
- **ML Detection**: Still uses Modal (cloud)

## Tier System

Three subscription tiers with increasing limits:

| Feature | Basic (Free) | Pro ($9.99/mo) | Elite ($24.99/mo) |
|---------|--------------|----------------|-------------------|
| AI Detections/month | 2 | 15 | 40 |
| Monthly uploads | 5 | 20 | 50 |
| Max video duration | 30 min | 60 min | 90 min |
| Max file size | 500 MB | 3 GB | 5 GB |
| Storage cap | 2 GB | 20 GB | 75 GB |
| Export quality | 720p + watermark | Original | Original |
| Watermark | Yes | No | No |
| Server export | No | Yes | Yes |
| Original quality retention | 7 days | 14 days | 60 days |
| Video retention | 60 days inactive | Unlimited | Unlimited |
| Cloud sync | No | Yes | Yes |

Pay-per-match credits: $0.99 per match. No subscription needed.

Tier configuration defined in `api/src/config/tiers.ts`. See `api/CLAUDE.md` for enforcement details.

## Cross-Project Flow

1. **Upload**: web → api (presigned S3 URL) → S3
2. **Detection**: api → Modal (GPU ML) → webhook → api → PostgreSQL
3. **Edit**: web ↔ api (rally/highlight CRUD)
4. **Export**: api → Lambda (FFmpeg) → S3 → web (download)

## See Also

- [analysis/CLAUDE.md](analysis/CLAUDE.md) - ML pipeline details
- [api/CLAUDE.md](api/CLAUDE.md) - REST API patterns
- [web/CLAUDE.md](web/CLAUDE.md) - Frontend architecture
- [lambda/CLAUDE.md](lambda/CLAUDE.md) - Serverless functions
