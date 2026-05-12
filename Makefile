# RallyCut Development Makefile
# Usage: make dev (starts everything with local services)

.PHONY: dev dev-db dev-minio dev-api dev-web dev-runner dev-prod setup install migrate help
.PHONY: stop logs clean minio-console reset-storage backup-db check-action-gt-cleanup

# Default target
help:
	@echo "RallyCut Development Commands:"
	@echo ""
	@echo "  make setup        - First-time setup (install deps, start services, migrate)"
	@echo "  make dev          - Start local dev (db, minio, api, web)"
	@echo "  make dev-prod     - Start with production services (api, web only)"
	@echo ""
	@echo "  make dev-db       - Start PostgreSQL only"
	@echo "  make dev-minio    - Start MinIO only"
	@echo "  make dev-api      - Start API server only"
	@echo "  make dev-web      - Start web frontend only"
	@echo "  make dev-runner   - Start local ML runner"
	@echo ""
	@echo "  make minio-console  - Open MinIO web console"
	@echo "  make reset-storage  - Clear all MinIO data"
	@echo ""
	@echo "  make stop         - Stop all services"
	@echo "  make logs         - Show service logs"
	@echo "  make clean        - Stop services and remove volumes"
	@echo ""

# First-time setup
setup: install
	@if [ ! -f api/.env ]; then \
		cp api/.env.local.example api/.env; \
		echo "Created api/.env from .env.local.example"; \
	fi
	@make dev-db dev-minio migrate
	@echo ""
	@echo "Setup complete! Run 'make dev' to start development."
	@echo ""

# Install all dependencies
install:
	@echo "📦 Installing dependencies..."
	cd api && npm install
	cd web && npm install
	cd analysis && uv sync
	@echo "✅ Dependencies installed"

# Database
dev-db:
	@echo "Starting PostgreSQL..."
	@cd api && docker compose up -d postgres
	@echo "Waiting for database to be ready..."
	@sleep 2
	@echo "PostgreSQL running on localhost:5436"

# MinIO (S3-compatible storage)
dev-minio:
	@echo "Starting MinIO..."
	@cd api && docker compose up -d minio minio-init
	@echo "Waiting for MinIO to be ready..."
	@sleep 3
	@echo "MinIO running:"
	@echo "  - S3 API: http://localhost:9000"
	@echo "  - Console: http://localhost:9001 (minioadmin/minioadmin)"

# Run migrations
migrate:
	@echo "🔄 Running database migrations..."
	cd api && npx prisma migrate dev
	@echo "✅ Migrations complete"

# Start API server
dev-api:
	@echo "🚀 Starting API server..."
	cd api && npm run dev

# Start web frontend
dev-web:
	@echo "🌐 Starting web frontend..."
	cd web && npm run dev

# Start local ML runner (for testing detection without Modal)
dev-runner:
	@echo "🤖 Starting local ML runner..."
	cd analysis && uv run python -m rallycut.service.local_runner

# Start everything (local development with MinIO)
dev:
	@if [ ! -f api/.env ]; then \
		cp api/.env.local.example api/.env; \
		echo "Created api/.env from .env.local.example"; \
	fi
	@make dev-db dev-minio
	@echo "Running database migrations..."
	@cd api && npx prisma migrate deploy
	@echo ""
	@echo "Starting RallyCut (local mode)..."
	@echo ""
	@echo "Services:"
	@echo "  - PostgreSQL: localhost:5436"
	@echo "  - MinIO S3:   localhost:9000"
	@echo "  - MinIO UI:   localhost:9001"
	@echo "  - API:        localhost:3001"
	@echo "  - Web:        localhost:3000"
	@echo ""
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@trap 'kill 0' EXIT; \
		(cd api && npm run dev) & \
		(cd web && npm run dev) & \
		wait

# Start with production services (no local MinIO/PostgreSQL)
dev-prod:
	@if [ ! -f api/.env.prod ]; then \
		echo "ERROR: api/.env.prod not found"; \
		echo "Create it from api/.env.prod-local.example with your production credentials"; \
		exit 1; \
	fi
	@cp api/.env.prod api/.env
	@echo ""
	@echo "Starting RallyCut (production-local mode)..."
	@echo "WARNING: Connected to production services!"
	@echo ""
	@echo "Services:"
	@echo "  - API: localhost:3001"
	@echo "  - Web: localhost:3000"
	@echo ""
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@trap 'kill 0' EXIT; \
		(cd api && npm run dev) & \
		(cd web && npm run dev) & \
		wait

# Stop all services
stop:
	@echo "Stopping services..."
	@cd api && docker compose down
	@echo "Services stopped"

# Show service logs
logs:
	cd api && docker compose logs -f

# Clean up everything (removes data volumes)
clean:
	@echo "Cleaning up..."
	@cd api && docker compose down -v
	@echo "Cleanup complete"

# MinIO management
minio-console:
	@echo "Opening MinIO console at http://localhost:9001"
	@open http://localhost:9001 2>/dev/null || xdg-open http://localhost:9001 2>/dev/null || echo "Open http://localhost:9001 in your browser"

reset-storage:
	@echo "Resetting MinIO storage..."
	@cd api && docker compose stop minio
	@docker volume rm api_minio_data 2>/dev/null || true
	@make dev-minio
	@echo "Storage reset complete"

# Pre-cutover guard for action-GT decoupling (2026-05-12).
# Verifies that no application code (outside the documented migration/export
# paths) references the soon-to-be-dropped player_tracks.action_ground_truth_json
# column. Exits non-zero if stale references remain — run before the column
# drop migration ships to production.
#
# Allowed paths (each contains a comment explaining why the reference stays):
#   analysis/rallycut/cli/commands/migrate_action_gt.py — backfill reads old col
#   analysis/rallycut/training/restore.py               — restore reads file-key
#   analysis/rallycut/cli/commands/train.py             — export writes file-key
#   analysis/reports/archived-scripts/                  — frozen historical scripts
check-action-gt-cleanup:
	@echo "Scanning for stale references to action_ground_truth_json..."
	@hits=$$(grep -rn "actionGroundTruthJson\|action_ground_truth_json" \
		api/src web/src analysis/rallycut analysis/scripts analysis/tests \
		--include="*.ts" --include="*.tsx" --include="*.py" 2>/dev/null \
		| grep -v "archived-scripts" \
		| grep -v "migrate_action_gt.py" \
		| grep -v "rallycut/training/restore.py" \
		| grep -v "rallycut/cli/commands/train.py"); \
	if [ -n "$$hits" ]; then \
		echo "ERROR: stale references to action_ground_truth_json remain:"; \
		echo "$$hits"; \
		echo ""; \
		echo "Either migrate the file to rally_action_ground_truth or add it to the allow-list in this Makefile target."; \
		exit 1; \
	fi
	@echo "✓ No stale references to action_ground_truth_json"

# Database backup
backup-db:
	@mkdir -p backups
	@echo "Dumping database to backups/rallycut_$$(date +%Y%m%d).dump..."
	@docker exec rallycut-postgres pg_dump -U postgres -Fc rallycut > backups/rallycut_$$(date +%Y%m%d).dump
	@echo "Database backup complete: backups/rallycut_$$(date +%Y%m%d).dump"
