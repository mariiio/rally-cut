---
name: dev-environment
description: Start, stop, and manage the RallyCut development environment. Use when setting up the project, starting services, or troubleshooting the dev stack. (project)
allowed-tools: Bash, Read
---

# RallyCut Dev Environment

## Start Development

```bash
make setup        # First-time setup (install deps + db + minio + migrate)
make dev          # Start everything (db + minio + api + web)
make dev-prod     # Production-local mode (AWS S3 + production DB)
```

## Individual Services

```bash
make dev-db       # PostgreSQL only (port 5436)
make dev-api      # API server only (port 3001)
make dev-web      # Web frontend only (port 3000)
```

## Stop & Clean

```bash
make stop         # Stop all services
make clean        # Stop + remove volumes
make reset-storage  # Clear all MinIO stored files
```

## Local Services

| Service | URL | Credentials |
|---------|-----|-------------|
| PostgreSQL | localhost:5436 | postgres/postgres |
| MinIO S3 API | localhost:9000 | minioadmin/minioadmin |
| MinIO Console | localhost:9001 | minioadmin/minioadmin |
| API | localhost:3001 | - |
| Web | localhost:3000 | - |

## Health Checks

- Web: http://localhost:3000
- API: http://localhost:3001/health
- MinIO Console: http://localhost:9001
- DB: `docker ps` shows postgres running

## Troubleshooting

1. **Port in use**: `lsof -i :3001` then `kill <PID>`
2. **DB connection failed**: `make dev-db` then wait 5s
3. **Missing deps**: `make install`
4. **MinIO issues**: `make minio-console` to inspect, `make reset-storage` to clear
