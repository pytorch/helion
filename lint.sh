#!/bin/bash
if [ "$1" = "" ];
then
  ACTION="fix"
else
  ACTION="$1"
fi

if [ "$ACTION" = "install" ];
then
    set -ex
    pip install ruff==0.14.2 pyright==1.1.407
    exit 0
fi

if ! (which ruff > /dev/null && which pyright > /dev/null);
then
    echo "ruff/pyright not installed. Run ./lint.sh install"
    exit 1
fi

VALID_ACTION="false"
ERRORS=""

function run
{
    echo "+" $@ 1>&2
    $@
    if [ $? -ne 0 ];
    then
        ERRORS="$ERRORS"$'\n'"ERROR running: $@"
    fi
    VALID_ACTION="true"
}

if [ "$ACTION" = "fix" ];
then
    run ruff format
    run ruff check --fix
    run pyright
fi

if [ "$ACTION" = "unsafe" ];
then
    run ruff format
    run ruff check --fix --unsafe-fixes
    run pyright
fi

if [ "$ACTION" = "check" ];
then
    run ruff format --check --diff
    run ruff check --no-fix
    run pyright
fi

if [ "$ERRORS" != "" ];
then
    echo "$ERRORS" 1>&2
    exit 1
fi

if [ "$VALID_ACTION" = "false" ];
then
    echo "Invalid argument: $ACTION" 1>&2
    echo "Usage: ./lint.sh [fix|check|install|unsafe]" 1>&2
    exit 1
fi

exit 0
