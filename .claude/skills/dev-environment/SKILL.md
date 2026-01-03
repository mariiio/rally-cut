---
name: dev-environment
description: Start, stop, and manage the RallyCut development environment. Use when setting up the project, starting services, or troubleshooting the dev stack. (project)
allowed-tools: Bash, Read
---

# RallyCut Dev Environment

## Start Development

```bash
make dev          # Start everything (db + api + web)
make setup        # First-time setup (install + db + migrate)
```

## Individual Services

```bash
make dev-db       # PostgreSQL only (port 5432)
make dev-api      # API server only (port 4000)
make dev-web      # Web frontend only (port 3000)
make dev-runner   # Local ML runner (no Modal)
```

## Stop & Clean

```bash
make stop         # Stop all services
make clean        # Stop + remove volumes
```

## Health Checks

- Web: http://localhost:3000
- API: http://localhost:4000/health
- DB: `docker ps` shows postgres running

## Troubleshooting

1. **Port in use**: `lsof -i :4000` then `kill <PID>`
2. **DB connection failed**: `make dev-db` then wait 5s
3. **Missing deps**: `make install`
