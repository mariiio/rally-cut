#!/bin/bash
# Pre-commit hook: runs linters/type checks before git commit
# Intercepts Bash tool calls containing "git commit"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only trigger on git commit commands
if ! echo "$COMMAND" | grep -qE '^\s*git\s+commit'; then
  exit 0
fi

PROJECT_DIR="/Users/mario/Personal/Projects/RallyCut"
ERRORS=""

# List staged files (Added/Copied/Modified/Renamed)
STAGED=$(cd "$PROJECT_DIR" && git diff --cached --name-only --diff-filter=ACMR 2>/dev/null)

if [ -z "$STAGED" ]; then
  exit 0
fi

# Pipeline version-bump enforcement.
# When action_classifier.py or contact_detector.py is staged, require either:
#  - the corresponding *_PIPELINE_VERSION constant is added/changed in the
#    same commit's diff, OR
#  - the commit -m message contains the marker [no-version-bump].
for ENTRY in \
  "analysis/rallycut/tracking/action_classifier.py:ACTION_PIPELINE_VERSION" \
  "analysis/rallycut/tracking/contact_detector.py:CONTACT_PIPELINE_VERSION"; do
  FILE="${ENTRY%:*}"
  CONST="${ENTRY##*:}"
  if echo "$STAGED" | grep -qFx "$FILE"; then
    DIFF=$(cd "$PROJECT_DIR" && git diff --cached -- "$FILE")
    if ! echo "$DIFF" | grep -qE "^\+${CONST}[[:space:]]*=[[:space:]]*\"v[0-9]+\""; then
      if ! echo "$COMMAND" | grep -qF '[no-version-bump]'; then
        ERRORS="${ERRORS}${FILE} modified without bumping ${CONST}. Add '[no-version-bump]' to the commit message if behavior is unchanged.\n"
      fi
    fi
  fi
done

# Python: ruff + mypy only on staged rallycut/ files. Other analysis/
# subdirs (scripts/, tests/, reports/archived-scripts/, experiments/) are
# intentionally out of strict-mode scope.
RC_STAGED_REL=$(echo "$STAGED" | grep -E '^analysis/rallycut/.*\.py$' | sed 's|^analysis/||')
if [ -n "$RC_STAGED_REL" ]; then
  echo "Running ruff + mypy on staged rallycut/ files..." >&2
  cd "$PROJECT_DIR/analysis"
  if ! uv run ruff check $RC_STAGED_REL 2>&1; then
    ERRORS="${ERRORS}ruff check failed\n"
  fi
  if ! uv run mypy $RC_STAGED_REL 2>&1; then
    ERRORS="${ERRORS}mypy check failed\n"
  fi
fi

# API TypeScript checks (project-wide; tsc --noEmit can't be per-file)
if echo "$STAGED" | grep -qE '^api/.*\.(ts|tsx|prisma)$'; then
  echo "Running API type check..." >&2
  cd "$PROJECT_DIR/api"
  if ! npx tsc --noEmit 2>&1; then
    ERRORS="${ERRORS}API tsc failed\n"
  fi
fi

# Web TypeScript checks (project-wide; tsc --noEmit can't be per-file)
if echo "$STAGED" | grep -qE '^web/.*\.(ts|tsx)$'; then
  echo "Running Web type check..." >&2
  cd "$PROJECT_DIR/web"
  if ! npx tsc --noEmit 2>&1; then
    ERRORS="${ERRORS}Web tsc failed\n"
  fi
fi

if [ -n "$ERRORS" ]; then
  echo "" >&2
  printf "%b" "$ERRORS" >&2
  echo "Pre-commit checks FAILED. Fix the issues above before committing." >&2
  # Exit 2 = block the tool call
  exit 2
fi

echo "Pre-commit checks passed." >&2
exit 0
