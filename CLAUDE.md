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

See `api/CLAUDE.md` for tier limits and enforcement. Config in `api/src/config/tiers.ts`.

## Cross-Project Flow

1. **Upload**: web → api (presigned S3 URL) → S3
2. **Detection**: api → Modal (GPU ML) → webhook → api → PostgreSQL
3. **Tracking**: api → Modal (GPU YOLO+WASB) → per-rally webhooks → api → PostgreSQL (or local CPU fallback)
4. **Edit**: web ↔ api (rally/highlight CRUD)
5. **Export**: api → Lambda (FFmpeg) → S3 → web (download)

## Running Diagnostics & Long Processes

**Before running:**
- Validate inputs/config will work BEFORE starting the long operation. Run a quick dry-run or sanity check (e.g., verify files exist, data loads, model loads, one item processes successfully) rather than discovering errors 10 minutes in.
- Tell the user what you're about to run, how many items it will process, and roughly how long to expect.
- If the script doesn't already have per-item progress output, ADD IT before running. Every script that loops over rallies/videos/items must print progress per item (e.g., `[3/16] rally_id: HOTA=89.0%, IDsw=3 (12.4s)`).

**While running:**
- Run in background (`run_in_background: true`) for anything that takes >30 seconds.
- Check output periodically with non-blocking reads — don't wait silently for completion.
- If the first few items show errors or unexpected results, STOP EARLY and investigate rather than letting the full run complete.

**Output requirements:**
- Per-item results as they complete (not just a summary at the end).
- Running totals or aggregates so partial output is already useful if the process is interrupted.
- Print a final summary table at the end.
- Avoid long inline Python in Bash — write a script file instead.

**After running:**
- Review the output for anomalies before reporting results. Don't just relay the final summary — check for per-item regressions, errors, or unexpected patterns.

## See Also

- [analysis/CLAUDE.md](analysis/CLAUDE.md) - ML pipeline details
- [api/CLAUDE.md](api/CLAUDE.md) - REST API patterns
- [web/CLAUDE.md](web/CLAUDE.md) - Frontend architecture
- [lambda/CLAUDE.md](lambda/CLAUDE.md) - Serverless functions
