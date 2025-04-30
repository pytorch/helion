#!/bin/bash
set -ex
(
    mkdir -p /tmp/$USER
    pushd /tmp/$USER
    pip uninstall -y triton pytorch-triton || true
    rm -rf triton/ || true
    git clone https://github.com/triton-lang/triton.git  # install triton latest main
    (
        pushd triton/
        conda config --set channel_priority strict
        conda install -y -c conda-forge conda=25.3.1 conda-libmamba-solver
        conda config --set solver libmamba
        conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gcc=13 gxx=13
        pip install -r python/requirements.txt
        # Use TRITON_PARALLEL_LINK_JOBS=2 to avoid OOM on CPU CI machines
        MAX_JOBS=$(nproc) TRITON_PARALLEL_LINK_JOBS=2 pip install .  # install to conda site-packages/ folder
        popd
    )
    rm -rf triton/
    popd
)
exit 0
