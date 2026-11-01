#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

step() {
    printf '\n==> %s\n' "$1"
}

step "Required files"
for file in \
    README.md \
    LICENSE \
    CONTRIBUTING.md \
    SECURITY.md \
    CODE_OF_CONDUCT.md \
    CHANGELOG.md \
    AGENTS.md \
    CLAUDE.md \
    docs/MASTER_DOCUMENT.md
do
    if [ ! -f "$file" ]; then
        echo "Missing required file: $file" >&2
        exit 1
    fi
done

step "Docs/config validation"
uv run python scripts/check_repo.py

step "Ruff lint"
uv run ruff check .

step "Pytest suite"
uv run pytest -q

step "Whitespace"
if ! base=$(git merge-base HEAD origin/develop 2>/dev/null); then
    base=HEAD
fi
git diff --check "$base"

step "Done"
echo "Local CI passed."
