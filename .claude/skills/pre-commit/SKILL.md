---
name: pre-commit
description: Run all code quality checks before committing - type checking, linting, and tests for Python and TypeScript. Use before creating commits. (project)
allowed-tools: Bash, Read
---

# RallyCut Pre-Commit Checks

Run these checks before committing to ensure code quality.

## Quick Check (All Projects)

```bash
# Python (analysis/)
cd analysis && uv run mypy rallycut/ && uv run ruff check rallycut/ && uv run pytest tests

# API TypeScript
cd api && npx tsc --noEmit

# Web TypeScript
cd web && npx tsc --noEmit && npm run lint
```

## Detailed Checks

### Python (analysis/)
```bash
uv run mypy rallycut/           # Type checking (strict mode)
uv run ruff check rallycut/     # Linting
uv run ruff check rallycut/ --fix  # Auto-fix lint issues
uv run pytest tests             # Unit tests (fast)
```

### API (api/)
```bash
npx tsc --noEmit                # TypeScript type check
npm run test                    # Vitest unit tests
```

### Web (web/)
```bash
npx tsc --noEmit                # TypeScript type check
npm run lint                    # ESLint
npm run build                   # Verify build works
```

## Fix Common Issues

- **Import order (Python)**: `uv run ruff check --fix`
- **Missing types (Python)**: Add type annotations to function
- **Unused import**: Remove it or add `# noqa: F401` if intentional
