#!/bin/bash
# Lint, format and type-check with the pre-commit hooks CI runs
# (.pre-commit-config.yaml: ruff and ruff-format at their pinned versions,
# whitespace and file checks, the examples and device checks, the
# backend-isolation check, codespell and Pyrefly). A clean ./lint.sh means a
# green Lint job.
#
#   ./lint.sh [fix] [PATH...]   run the hooks, applying their fixes, then run
#                               them again so the result is checked (default)
#   ./lint.sh check [PATH...]   one pass, as CI does it: fails when a hook
#                               reports a problem or changes a file, and shows
#                               the changes (which stay in the tree)
#   ./lint.sh unsafe [PATH...]  fix, with ruff's unsafe fixes as well
#   ./lint.sh install           install pre-commit and the pinned Pyrefly and
#                               fetch the hook environments
#
# PATHs (files or directories, tracked or not) narrow the hooks to those
# files; Pyrefly still checks the whole project, and the backend-isolation
# check runs when a helion/ file is among them.

# The first argument is the action when it names one; otherwise every
# argument is a path and the action is fix.
case "${1:-}" in
  fix|check|unsafe|install) ACTION="$1"; shift ;;
  *) ACTION="fix" ;;
esac

# Pyrefly runs as a system hook, so pre-commit does not pin it; this is the
# pin. Keep the literal `pyrefly==X` form: devenv's workspace setup reads it.
PYREFLY_REQUIREMENT="pyrefly==1.1.1"
PYREFLY_PIN="${PYREFLY_REQUIREMENT#pyrefly==}"

if [ "$ACTION" = "install" ]; then
  set -ex
  pip install pre-commit "$PYREFLY_REQUIREMENT"
  pre-commit install-hooks
  exit 0
fi

for tool in pre-commit pyrefly; do
  if ! command -v "$tool" >/dev/null; then
    echo "$tool is not installed. Run ./lint.sh install" 1>&2
    exit 1
  fi
done
if [ "$(pyrefly --version 2>/dev/null)" != "pyrefly $PYREFLY_PIN" ]; then
  echo "warning: $(pyrefly --version 2>/dev/null) on PATH; CI uses pyrefly $PYREFLY_PIN" 1>&2
fi

# pre-commit's --files takes files, not directories.
SCOPE=(--all-files)
if [ $# -gt 0 ]; then
  FILES=()
  while IFS= read -r file; do
    FILES+=("$file")
  done < <(git ls-files --cached --others --exclude-standard -- "$@")
  if [ ${#FILES[@]} -eq 0 ]; then
    echo "no files under: $* (usage: ./lint.sh [fix|check|unsafe|install] [PATH...])" 1>&2
    exit 1
  fi
  SCOPE=(--files "${FILES[@]}")
fi

# The tracked diff plus the untracked files' contents: whether a pass fixed
# anything.
function tree_state {
  {
    git diff --no-ext-diff --binary
    git ls-files --others --exclude-standard -z | xargs -0 -r sha1sum
  } | sha1sum
}

# A hook that fixes files fails that pass, so run again to check the result;
# when the first pass changed nothing, its failures need you, not a rerun.
function fix_then_check {
  local before
  before=$(tree_state)
  pre-commit run "${SCOPE[@]}" && return 0
  if [ "$(tree_state)" = "$before" ]; then
    return 1
  fi
  pre-commit run "${SCOPE[@]}"
}

case "$ACTION" in
  fix)
    fix_then_check
    ;;
  unsafe)
    pre-commit run --hook-stage manual ruff-unsafe "${SCOPE[@]}"
    fix_then_check
    ;;
  check)
    pre-commit run --show-diff-on-failure "${SCOPE[@]}"
    ;;
esac
