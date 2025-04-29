#!/bin/bash
if [ "$1" = "" ];
then
  ACTION="full"
else
  ACTION="$1"
fi

# Check if the action is valid
if [ "$ACTION" != "full" ] && [ "$ACTION" != "python-stub-only" ]; then
    echo "Error: Invalid action '$ACTION'. Must be either 'full' or 'python-stub-only'." >&2
    exit 1
fi

if [ "$ACTION" = "full" ];
then
    set -ex
    (
        mkdir -p /tmp/$USER
        pushd /tmp/$USER
        pip uninstall -y triton pytorch-triton || true
        rm -rf triton/ || true
        git clone https://github.com/triton-lang/triton.git  # install triton latest main
        (
            pushd triton/
            conda install -y conda=25.3.1 conda-libmamba-solver -c conda-forge
            conda config --set solver libmamba
            conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gcc=13 gxx=13
            pip install -r python/requirements.txt
            pip install .  # install to conda site-packages/ folder
            popd
        )
        rm -rf triton/
        popd
    )
    exit 0
fi

if [ "$ACTION" = "python-stub-only" ];
then
    set -ex
    (
        mkdir -p /tmp/$USER
        pushd /tmp/$USER
        pip uninstall -y triton pytorch-triton || true
        rm -rf triton/ || true
        git clone https://github.com/triton-lang/triton.git  # install triton latest main
        # NOTE: Unfortunately building triton from source will crash the CPU CI machines (c5.2xlarge or c5.4xlarge).
        # But since for lint jobs we actually don't need the .so files, we can just install and use the python stubs.
        export SITE_PKGS_PATH=`python -c "import sysconfig, json; print(sysconfig.get_paths()['purelib'])"`
        mkdir ${SITE_PKGS_PATH}/triton
        cp -r triton/python/triton/* ${SITE_PKGS_PATH}/triton
        rm -rf triton/
    )
    exit 0
fi
