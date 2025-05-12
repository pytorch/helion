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
    # pyre requires click<8.2
    pip install ruff==0.11.9 pyre-check==0.9.23 click==8.1.0
    exit 0
fi

if ! (which ruff > /dev/null && which pyre > /dev/null);
then
    echo "ruff/pyre not installed. Run ./lint.sh install"
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
    run pyre check
fi

if [ "$ACTION" = "unsafe" ];
then
    run ruff format
    run ruff check --fix --unsafe-fixes
    run pyre check
fi

if [ "$ACTION" = "check" ];
then
    run ruff format --check --diff
    run ruff check --no-fix
    run pyre check
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
