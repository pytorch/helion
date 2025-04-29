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
        conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 gcc=12 gxx=12
        pip install -r python/requirements.txt
        pip install .  # install to conda site-packages/ folder
        popd
    )
    popd
)
exit 0
