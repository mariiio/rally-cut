---
name: code-quality
description: Run RallyCut code quality checks - mypy type checking and ruff linting. Use when checking code quality, fixing type errors, or before committing changes.
allowed-tools: Bash, Read, Edit, Grep
---

# RallyCut Code Quality

## Commands (run in order)
1. Type check: `uv run mypy rallycut/`
2. Lint: `uv run ruff check rallycut/`
3. Auto-fix: `uv run ruff check rallycut/ --fix`

## Standards
- **Ruff rules:** E, F, I, N, W, UP (line length 100, E501 ignored)
- **MyPy:** Strict mode - all functions require type hints

## Common Issues
- Missing return type → Add `-> ReturnType` to function signature
- Untyped parameter → Add type annotation `param: Type`
- Import order → Run `uv run ruff check --fix` to auto-sort

## Pre-commit Checklist
1. `uv run mypy rallycut/` - No errors
2. `uv run ruff check rallycut/` - No errors
3. `uv run pytest tests` - All pass
