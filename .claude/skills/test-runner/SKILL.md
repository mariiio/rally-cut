---
name: test-runner
description: Run RallyCut test suites - Python pytest (unit/integration/ML), API Vitest. Use when running tests, debugging failures, or checking coverage. (project)
allowed-tools: Bash, Read, Grep, Glob
---

# RallyCut Test Runner

## Python Tests (analysis/)

```bash
uv run pytest tests                              # Fast tests (no ML)
uv run pytest tests --run-slow                   # Include ML inference
uv run pytest tests/unit/test_game_state.py -v  # Single file
uv run pytest tests -k "test_name"              # By name pattern
```

### Test Organization
- `tests/unit/` - Fast, mocked ML models
- `tests/integration/` - Full pipeline with real ML
- `@pytest.mark.slow` - ML tests (skipped by default)

## API Tests (api/)

```bash
cd api && npm run test           # Run all tests
cd api && npm run test -- --watch  # Watch mode
```

## Debugging

```bash
uv run pytest -v --tb=long                    # Verbose with traceback
uv run pytest --pdb                           # Drop into debugger on failure
uv run pytest tests/path/test_file.py::test_name -v  # Single test
```

## Coverage

```bash
uv run pytest tests --cov=rallycut --cov-report=html
```
