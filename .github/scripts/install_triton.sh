#!/bin/bash
set -ex
(
    mkdir -p /tmp/$USER
    pushd /tmp/$USER
    rm -rf triton/ || true
    pip uninstall -y triton pytorch-triton || true
    git clone https://github.com/triton-lang/triton.git  # install triton latest main
    (
        pushd triton/
        dnf install -y gcc-toolset-12
        source /opt/rh/gcc-toolset-12/enable
        export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH
        pip install -r python/requirements.txt
        pip install .  # install to conda site-packages/ folder
        popd
    )
    popd
)
exit 0
