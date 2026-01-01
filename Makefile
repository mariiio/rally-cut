# RallyCut Development Makefile
# Usage: make dev (starts everything)

.PHONY: dev dev-db dev-api dev-web dev-runner setup install migrate help

# Default target
help:
	@echo "RallyCut Development Commands:"
	@echo ""
	@echo "  make setup      - First-time setup (install deps, start db, run migrations)"
	@echo "  make dev        - Start all services (db, api, web)"
	@echo "  make dev-db     - Start PostgreSQL only"
	@echo "  make dev-api    - Start API server only"
	@echo "  make dev-web    - Start web frontend only"
	@echo "  make dev-runner - Start local ML runner"
	@echo "  make stop       - Stop all services"
	@echo "  make logs       - Show database logs"
	@echo "  make clean      - Stop services and remove volumes"
	@echo ""

# First-time setup
setup: install dev-db migrate
	@echo "✅ Setup complete! Run 'make dev' to start development."

# Install all dependencies
install:
	@echo "📦 Installing dependencies..."
	cd api && npm install
	cd web && npm install
	cd analysis && uv sync
	@echo "✅ Dependencies installed"

# Database
dev-db:
	@echo "🐘 Starting PostgreSQL..."
	@cd api && docker-compose up -d
	@echo "⏳ Waiting for database to be ready..."
	@sleep 2
	@echo "✅ PostgreSQL running on localhost:5432"

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

# Start everything (uses terminal multiplexing or background processes)
dev:
	@echo "🚀 Starting RallyCut development environment..."
	@make dev-db
	@echo ""
	@echo "Starting API and Web servers..."
	@echo "Press Ctrl+C to stop all services"
	@echo ""
	@trap 'kill 0' EXIT; \
		(cd api && npm run dev) & \
		(cd web && npm run dev) & \
		wait

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	@cd api && docker-compose down
	@echo "✅ Services stopped"

# Show database logs
logs:
	cd api && docker-compose logs -f

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	@cd api && docker-compose down -v
	@echo "✅ Cleanup complete"
