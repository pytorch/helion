#!/bin/bash
# Build the HazyResearch Megakernels low-latency Llama-3.2-1B megakernel with L layers
# (the kernel bakes LLAMA_1B_NUM_LAYERS in) into $OUT/L$L/mk_llama*.so.
# Same flags as demos/low-latency-llama/Makefile (GPU=B200), CUDA 12.8 nvcc, pybind11 headers
# from TK_PY's torch.
#   TK_DIR=<Megakernels checkout> build_tk.sh <L> <OUT>
# Env: TK_DIR (required), TK_PY (python, default python), TK_CUDA_HOME (default /usr/local/cuda-12.8).
set -euo pipefail
L=$1
OUT=$2
: "${TK_DIR:?set TK_DIR to the Megakernels checkout}"
TK_CUDA_HOME=${TK_CUDA_HOME:-/usr/local/cuda-12.8}
NVCC=$TK_CUDA_HOME/bin/nvcc
PY=${TK_PY:-python}
[ -x "$NVCC" ] || { echo "build_tk.sh: no nvcc at $NVCC (set TK_CUDA_HOME to CUDA 12.8)" >&2; exit 1; }
SRC=$TK_DIR/demos/low-latency-llama
D=$OUT/L$L
mkdir -p "$D/src"
cp "$SRC"/*.cu "$SRC"/*.cuh "$D/src/"
sed -i "s/^#define LLAMA_1B_NUM_LAYERS .*/#define LLAMA_1B_NUM_LAYERS $L/" "$D/src/llama.cuh"
grep -q "^#define LLAMA_1B_NUM_LAYERS $L\$" "$D/src/llama.cuh"
TORCH_INC=$($PY -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cd "$D/src"
"$NVCC" llama.cu -DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi \
  -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xptxas=--warn-on-spills \
  -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo \
  -I"$TK_DIR/ThunderKittens/include" -I"$TK_DIR/include" -I"$TORCH_INC" -I"$PYINC" \
  -shared -fPIC -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a \
  -o "$D/mk_llama$EXT" > "$D/build.log" 2>&1
echo "built $D/mk_llama$EXT"
