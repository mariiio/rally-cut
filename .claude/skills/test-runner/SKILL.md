---
name: test-runner
description: Run RallyCut test suites - unit tests, integration tests, or slow ML inference tests. Use when running tests, debugging test failures, or checking test coverage.
allowed-tools: Bash, Read, Grep, Glob
---

# RallyCut Test Runner

## Test Commands
- Fast tests: `uv run pytest tests` (excludes slow ML tests)
- Full tests: `uv run pytest tests --run-slow` (includes ML inference)
- Specific test: `uv run pytest tests/unit/test_game_state.py -v`
- Detection quality: `uv run pytest tests/integration/test_detection_quality.py --run-slow`

## Test Organization
- `tests/unit/` - Fast, mocked ML models
- `tests/integration/` - Full pipeline with real ML
- `@pytest.mark.slow` - ML inference tests (skipped by default)

## Debugging Failed Tests
1. Run with verbose output: `uv run pytest -v --tb=long`
2. Run single test: `uv run pytest tests/path/test_file.py::test_name -v`
3. Check fixtures in `tests/conftest.py`
