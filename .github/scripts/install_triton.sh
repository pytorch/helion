#!/bin/bash
set -ex

# Parse command line arguments
USE_CPU_BACKEND=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_CPU_BACKEND=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

(
    mkdir -p /tmp/$USER
    pushd /tmp/$USER
    pip uninstall -y triton pytorch-triton || true
    rm -rf triton/ || true

    # Clone the appropriate repository based on backend
    if [ "$USE_CPU_BACKEND" = true ]; then
        # Install triton-cpu from triton-cpu repository
        git clone --recursive https://github.com/triton-lang/triton-cpu.git triton
    else
        # Install triton from main repository for GPU backend
        git clone https://github.com/triton-lang/triton.git triton
    fi

    # Shared build process for both backends
    (
        pushd triton/
        conda config --set channel_priority strict
        conda install -y -c conda-forge conda=25.3.1 conda-libmamba-solver
        conda config --set solver libmamba
        conda install -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gcc=13 gxx=13
        pip install -r python/requirements.txt
        # Use TRITON_PARALLEL_LINK_JOBS=2 to avoid OOM on CPU CI machines
        if [ "$USE_CPU_BACKEND" = true ]; then
            MAX_JOBS=$(nproc) TRITON_PARALLEL_LINK_JOBS=2 pip install -e python  # install to conda site-packages/ folder
        else
            MAX_JOBS=$(nproc) TRITON_PARALLEL_LINK_JOBS=2 pip install .  # install to conda site-packages/ folder
        fi
        popd
    )
    #rm -rf triton/
    popd
)
exit 0
