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
    # NOTE: Unfortunately the pyre-check binary from pip expects GLIBC_2.29 but our CI machine's linux image only has GLIBC_2.28.
    # So we have to build the pyre-check binary from source. (See https://github.com/facebook/pyre-check/issues/985)
    # Q: If we are building from source anyway, why do we still need to do `pip install pyre-check==0.9.23`?
    # A: I tried that and the from-source Python client actually generates many more type errors (likely related to `typeshed` config).
    #    So in the interest of time, I decided to just use the pip-installed version for the Python client, but use the from-source version for the server binary.
    pip install ruff==0.9.8 pyre-check==0.9.23
    (
        mkdir -p /tmp/$USER
        pushd /tmp/$USER
        rm -rf pyre-check-for-helion/ || true
        git clone https://github.com/facebook/pyre-check.git -b v0.9.23 pyre-check-for-helion/
        (
            pushd pyre-check-for-helion/

            # Install toolchain
            conda config --set channel_priority strict
            conda install -y -c conda-forge conda=25.3.1 conda-libmamba-solver
            conda config --set solver libmamba
            conda install -y -c conda-forge "rust>=1.77"
            conda install -y -c conda-forge bubblewrap opam

            # Build pyre-check
            ./scripts/setup.sh --local --no-tests
            install -m755 ./source/_build/default/main.exe "$CONDA_PREFIX/bin/pyre.bin"
            rm -rf ./source/_build/

            ldd --version  # shows the host's GLIBC version
            conda run pyre.bin --version  # check there is no GLIBC version mismatch

            popd
        )
        rm -rf pyre-check-for-helion/
        popd
    )
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
