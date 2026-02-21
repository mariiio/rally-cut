# RallyCut

RallyCut helps volleyball players and coaches get more out of their match videos — without spending hours reviewing footage.

Upload a match recording, and RallyCut does the rest. It automatically finds every rally, identifies who's on court, tracks the ball, and breaks down what happened — serves, sets, spikes, digs — point by point.

- **Skip the boring parts** — dead time between rallies is cut automatically
- **Pull out highlights** — export specific rallies or standout plays as clips to share
- **See match stats** — get a score breakdown and action counts without manual tallying
- **Review plays visually** — watch with player and ball tracking overlays to study positioning

It turns a raw camera recording into organized, actionable match data — something that normally takes a dedicated analyst or hours of manual work.

## Overview

RallyCut is a monorepo containing four projects:

| Project | Description | Stack |
|---------|-------------|-------|
| **[analysis/](analysis/)** | ML-powered video analysis CLI | Python, PyTorch, VideoMAE, YOLO |
| **[web/](web/)** | Rally editor web application | Next.js 15, React 19, MUI, Zustand |
| **[api/](api/)** | Backend REST API | Express, Prisma, PostgreSQL, S3 |
| **[lambda/](lambda/)** | Serverless video processing | AWS Lambda, FFmpeg, ARM64 |

## Features

- **Rally Detection**: ML classifier identifies active play vs dead time
- **Ball Tracking**: Frame-by-frame ball position tracking with trained neural networks
- **Player Tracking**: Identifies and follows players throughout each rally
- **Contact Detection**: Recognizes serves, receives, sets, spikes, and digs
- **Match Statistics**: Automated scoring, action breakdowns, and per-rally analysis
- **Timeline Editor**: Drag-and-drop interface for adjusting rally boundaries
- **Highlights**: Create custom highlight reels from selected rallies
- **Video Export**: Export edited videos with fade transitions

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- Docker (for PostgreSQL)
- FFmpeg

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rallycut.git
cd rallycut

# Install dependencies, start database, run migrations
make setup

# Copy environment files
cp api/.env.example api/.env
```

### Development

```bash
# Start everything (database, API, web)
make dev

# Or start services individually:
make dev-db     # PostgreSQL only
make dev-api    # API server only (in separate terminal)
make dev-web    # Web frontend only (in separate terminal)
make dev-runner # Local ML runner (optional)

# Stop all services
make stop
```

Open [http://localhost:3000](http://localhost:3000) for the web app, [http://localhost:4000](http://localhost:4000) for the API.

### ML Analysis CLI

```bash
cd analysis && uv sync
uv run rallycut cut <video.mp4>      # Remove dead time
uv run rallycut overlay <video.mp4>  # Ball tracking overlay
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web App   │────▶│   REST API  │────▶│  PostgreSQL │
│  (Next.js)  │     │  (Express)  │     │   (Prisma)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       │            │  S3 + CDN   │
       │            │ (CloudFront)│
       │            └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│ Modal (GPU) │     │   Lambda    │
│   Webhook   │     │  ML Service │     │ Video Export│
└─────────────┘     └─────────────┘     └─────────────┘
```

## Project Structure

```
rallycut/
├── analysis/           # Python ML CLI
│   ├── rallycut/       # Main package
│   │   ├── cli/        # Typer commands
│   │   ├── core/       # Config, models, video wrapper
│   │   ├── analysis/   # GameStateAnalyzer (VideoMAE)
│   │   ├── processing/ # VideoCutter, FFmpegExporter
│   │   ├── tracking/   # BallTracker (YOLO)
│   │   └── service/    # Cloud detection service
│   └── tests/          # Unit and integration tests
├── web/                # Next.js frontend
│   └── src/
│       ├── app/        # App router pages
│       ├── components/ # React components
│       ├── stores/     # Zustand state management
│       └── services/   # API client, sync service
├── api/                # Express backend
│   └── src/
│       ├── routes/     # REST endpoints
│       ├── services/   # Business logic
│       ├── schemas/    # Zod validation
│       └── lib/        # S3, CloudFront, Prisma
├── lambda/             # Serverless functions
│   └── video-export/   # FFmpeg video processing (ARM64)
└── docs/               # Setup guides
```

## Documentation

- [Analysis CLI README](analysis/README.md) - ML pipeline and CLI usage
- [Detection Algorithm](analysis/docs/detection_algorithm.md) - How rally detection works
- [Video Export Lambda](lambda/video-export/README.md) - Server-side video processing
- [AWS Setup](docs/aws-setup.md) - S3 and CloudFront configuration
- [Modal Setup](docs/modal-setup.md) - GPU cloud deployment

## Development

### Code Quality

```bash
# Python (analysis/)
uv run mypy rallycut/          # Type checking
uv run ruff check rallycut/    # Linting
uv run pytest tests/           # Unit tests

# TypeScript (web/, api/)
npm run lint                   # ESLint
npx tsc --noEmit              # Type checking
```

### Testing

```bash
# Fast unit tests (no ML)
cd analysis && uv run pytest tests/unit

# Full tests including ML inference
uv run pytest tests --run-slow
```

## License

MIT
