#!/usr/bin/env bash

# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
pip install -r requirements_dev.txt -q

PYTHON_FILES=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./.*" -not -path "./docs_and_repos/*")
SHELL_FILES=$(find . -name "*.sh" -not -path "./.venv/*" -not -path "./.*" -not -path "./docs_and_repos/*")
[[ -z "$PYTHON_FILES" ]] && { echo "No Python files found"; exit 0; }

FAILED=()

# Auto-format with black
# shellcheck disable=SC2086
# black $PYTHON_FILES

# Run linters
echo "Running ruff..."
# shellcheck disable=SC2086
ruff check $PYTHON_FILES || FAILED+=("ruff")

echo "Running basedpyright..."
# shellcheck disable=SC2086
basedpyright $PYTHON_FILES || FAILED+=("basedpyright")

echo "Running mccabe..."
# shellcheck disable=SC2086
python -m mccabe --min 11 $PYTHON_FILES || FAILED+=("mccabe")

echo "Running pylint..."
# shellcheck disable=SC2086
pylint $PYTHON_FILES --score=y || FAILED+=("pylint")

if [[ -n "$SHELL_FILES" ]]; then
    echo "Running shellcheck..."
    # shellcheck disable=SC2086
    shellcheck $SHELL_FILES || FAILED+=("shellcheck")
fi

# Summary
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "✅ All linters passed!"
    exit 0
else
    echo "❌ Failed: ${FAILED[*]}"
    exit 1
fi
