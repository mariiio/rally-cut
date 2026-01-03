---
name: code-quality
description: Run RallyCut code quality checks - type checking and linting for Python (mypy, ruff) and TypeScript (tsc, eslint). Use when checking code quality or before committing. (project)
allowed-tools: Bash, Read, Edit, Grep
---

# RallyCut Code Quality

## Python (analysis/)

```bash
uv run mypy rallycut/              # Type check (strict)
uv run ruff check rallycut/        # Lint
uv run ruff check rallycut/ --fix  # Auto-fix
```

### Standards
- **Ruff:** E, F, I, N, W, UP rules (line length 100)
- **MyPy:** Strict mode - all functions need type hints

## TypeScript - API (api/)

```bash
cd api && npx tsc --noEmit
```

## TypeScript - Web (web/)

```bash
cd web && npx tsc --noEmit
cd web && npm run lint
```

## Common Fixes

| Issue | Fix |
|-------|-----|
| Missing return type | Add `-> ReturnType` |
| Import order | `uv run ruff check --fix` |
| Unused variable | Remove or prefix with `_` |
| Type error | Add proper annotation |
