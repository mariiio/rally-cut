# RallyCut

Beach volleyball video analysis platform. Monorepo with ML pipeline, web editor, and REST API.

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
make setup    # First-time: install deps, start db, migrate
make dev      # Start db + api + web
make stop     # Stop all services
```

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
