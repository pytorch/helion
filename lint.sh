#!/bin/bash
if [ "$1" = "" ]; then
  ACTION="fix"
else
  ACTION="$1"
fi

if [ "$ACTION" = "install" ]; then
  set -ex
  pip install ruff==0.15.0 pyrefly==0.51.1
  exit 0
fi

if ! (which ruff >/dev/null && which pyrefly >/dev/null); then
  echo "ruff/pyrefly not installed. Run ./lint.sh install"
  exit 1
fi

VALID_ACTION="false"
ERRORS=""

function run {
  echo "+" $@ 1>&2
  $@
  if [ $? -ne 0 ]; then
    ERRORS="$ERRORS"$'\n'"ERROR running: $@"
  fi
  VALID_ACTION="true"
}

# On macOS, pyrefly cannot find triton, cutlass, etc.
# Restrict to ruff only for mac.
RUN_PYREFLY="true"
if [ "$(uname -s)" = "Darwin" ]; then
  RUN_PYREFLY="false"
fi

if [ "$ACTION" = "fix" ]; then
  run ruff format
  run ruff check --fix
  if [ "$RUN_PYREFLY" = "true" ]; then run pyrefly check; fi
fi

if [ "$ACTION" = "unsafe" ]; then
  run ruff format
  run ruff check --fix --unsafe-fixes
  if [ "$RUN_PYREFLY" = "true" ]; then run pyrefly check; fi
fi

if [ "$ACTION" = "check" ]; then
  run ruff format --check --diff
  run ruff check --no-fix
  if [ "$RUN_PYREFLY" = "true" ]; then run pyrefly check; fi
fi

if [ "$ERRORS" != "" ]; then
  echo "$ERRORS" 1>&2
  exit 1
fi

if [ "$VALID_ACTION" = "false" ]; then
  echo "Invalid argument: $ACTION" 1>&2
  echo "Usage: ./lint.sh [fix|check|install|unsafe]" 1>&2
  exit 1
fi

exit 0
