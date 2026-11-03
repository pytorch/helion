#!/bin/bash
# ThunderKittens Megakernels Llama-3.2-1B per-layer cold-L2 decode cost: build the L = 1, 2, 4, 8, 16
# megakernels (if missing), run $NPROC rounds of one process per L on one GPU, then merge into
# $RESULTS/tk_llama32_1b_layer.json (../mirage/summarize.py).
#   GPU=6 RESULTS=/path/to/results bash run_all.sh
# Env: GPU, RESULTS (required); RUNS (per-process JSONs, default $RESULTS/tk_runs); NPROC (2);
#   LAYERS (1,2,4,8,16); from ../env.sh: TK_DIR (the Megakernels checkout with its ThunderKittens
#   submodule) and VLLM_PY; TK_PY (Python with torch 2.14.0+cu130 and transformers, default
#   $VLLM_PY); TK_CUDA_HOME (CUDA 12.8: nvcc for the build, libcublas.so.12 at run time, default
#   /usr/local/cuda-12.8); TK_BUILD (build directory, default $RESULTS/tk_build, so every results
#   directory builds its own megakernels with its TK_PY).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../env.sh
source "$HERE/../env.sh"
: "${GPU:?set GPU to the index of an idle B200}"
: "${RESULTS:?set RESULTS to the results directory}"
NPROC=${NPROC:-2}
RUNS=${RUNS:-$RESULTS/tk_runs}
LAYERS=${LAYERS:-1,2,4,8,16}
: "${TK_PY:=$VLLM_PY}"
: "${TK_CUDA_HOME:=/usr/local/cuda-12.8}"
: "${TK_BUILD:=$RESULTS/tk_build}"
die() { echo "thunderkittens/run_all.sh: $*" >&2; exit 1; }
[ -x "$TK_PY" ] || die "no Python at TK_PY=$TK_PY (torch 2.14.0+cu130 and transformers; see README.md)"
[ -d "$TK_DIR/demos/low-latency-llama" ] || die "TK_DIR=$TK_DIR is not a Megakernels checkout (see README.md)"
[ -d "$TK_DIR/ThunderKittens/include" ] || die "$TK_DIR lacks its ThunderKittens submodule (git submodule update --init)"
[ -x "$TK_CUDA_HOME/bin/nvcc" ] || die "no nvcc at TK_CUDA_HOME=$TK_CUDA_HOME (CUDA 12.8)"
[ -e "$TK_CUDA_HOME/lib64/libcublas.so.12" ] || die "no libcublas.so.12 in $TK_CUDA_HOME/lib64"
export TK_DIR TK_PY TK_CUDA_HOME
export PATH=$(dirname "$TK_PY"):$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TK_CUDA_HOME/lib64   # libcublas.so.12 of the build
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RUNS"
for L in ${LAYERS//,/ }; do
  ls "$TK_BUILD/L$L"/mk_llama*.so > /dev/null 2>&1 || "$HERE/build_tk.sh" "$L" "$TK_BUILD" || exit 1
done
uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
wait_free() {  # no other process on the GPU for 30 s (then env.sh's timed waits, polls the GPU every
  # 2 s while the process runs and repeats it if another process showed up; the driver itself polls
  # every 0.5 s during its timing session and exits 3 if another process showed up)
  local quiet=0 last_log=-1000
  while [ $quiet -lt 30 ]; do
    if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$uuid"; then
      quiet=0
      if [ $((SECONDS - last_log)) -ge 300 ]; then
        echo "$(date +%T) GPU $GPU busy: $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" | tr '\n' ' ')"
        last_log=$SECONDS
      fi
    else
      quiet=$((quiet + 5))
    fi
    sleep 5
  done
}
one() {  # one driver process (from /tmp: no stray modules on sys.path)
  cd /tmp && timeout 14400 "$TK_PY" "$HERE/tk_llama_layers.py" --build-dir "$TK_BUILD" "$@"
}
# One process per L (a process can load only one build); round p visits the layer counts in
# alternating order.
for p in $(seq 1 "$NPROC"); do
  order=${LAYERS//,/ }
  [ $((p % 2)) = 0 ] && order=$(echo "$order" | tr ' ' '\n' | tac | tr '\n' ' ')
  for L in $order; do
    for attempt in 1 2 3 4 5; do
      wait_free
      echo "$(date +%T) start tk round $p L=$L"
      f=$RUNS/tk_llama32_1b_r${p}_L$L
      timed "$GPU" one --layers "$L" --round "$p" --out "$f.json" > "$f.log" 2>&1
      rc=$?
      if [ $rc = 0 ]; then tail -1 "$f.log"; break; fi
      mv "$f.log" "$f.attempt$attempt.log"
      echo "$(date +%T) round $p L=$L attempt $attempt failed (exit $rc), see $f.attempt$attempt.log"
    done
  done
done
"$TK_PY" "$HERE/../mirage/summarize.py" --runs "$RUNS" --results "$RESULTS"
