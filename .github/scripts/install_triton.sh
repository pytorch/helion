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
        # conda install -y -n venv conda=25.3.1 conda-libmamba-solver -c conda-forge
        # conda config --set solver libmamba
        conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gcc=13 gxx=13
        pip install -r python/requirements.txt
        pip install .  # install to conda site-packages/ folder
        popd
    )
    popd
)
exit 0
