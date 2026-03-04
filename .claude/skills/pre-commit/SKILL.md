---
name: pre-commit
description: Run RallyCut code quality checks and tests - type checking (mypy, tsc), linting (ruff, eslint), and tests (pytest, vitest). Use before committing, when checking code quality, or when running tests. (project)
allowed-tools: Bash, Read, Edit, Grep, Glob
---

# RallyCut Pre-Commit Checks

## Quick Check (All Projects)

```bash
# Python (analysis/)
cd analysis && uv run mypy rallycut/ && uv run ruff check rallycut/ && uv run pytest tests

# API TypeScript
cd api && npx tsc --noEmit

# Web TypeScript
cd web && npx tsc --noEmit && npm run lint
```

Only run checks for projects with changes. If you only touched `analysis/`, skip api/web checks.

## Python (analysis/)

```bash
cd analysis
uv run mypy rallycut/              # Type check (strict mode — all functions need type hints)
uv run ruff check rallycut/        # Lint (E, F, I, N, W, UP rules, line length 100)
uv run ruff check rallycut/ --fix  # Auto-fix lint issues
uv run pytest tests                # Unit tests (fast, no ML)
uv run pytest tests --run-slow     # Include ML inference tests
uv run pytest tests -k "test_name" # By name pattern
uv run pytest tests/unit/test_foo.py::test_bar -v  # Single test
```

## API (api/)

```bash
cd api
npx tsc --noEmit         # TypeScript type check
npm run test             # Vitest unit tests
npm run test -- --watch  # Watch mode
```

## Web (web/)

```bash
cd web
npx tsc --noEmit   # TypeScript type check
npm run lint        # ESLint
npm run build       # Verify build works
```

## Debugging Test Failures

```bash
cd analysis
uv run pytest -v --tb=long    # Verbose with full traceback
uv run pytest --pdb            # Drop into debugger on failure
uv run pytest tests --cov=rallycut --cov-report=html  # Coverage report
```

## Common Fixes

| Issue | Fix |
|-------|-----|
| Import order (Python) | `uv run ruff check --fix` |
| Missing return type | Add `-> ReturnType` annotation |
| Unused variable | Remove or prefix with `_` |
| Unused import | Remove or add `# noqa: F401` if intentional |
