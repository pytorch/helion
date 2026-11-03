# Environments and checkouts of the B200 megakernel benchmark.  Source this file;
# set any variable first to point at your own copies.  README.md lists how each
# environment is built.
#
#   Helion  torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1 (this repo)
#   vLLM    main@6a95c89c37, torch 2.14.0+cu130, triton 3.8.0, flashinfer 0.7.0.post1
#   SGLang  main@02b983603, torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1
#   Mirage  mpk@2edbcf40 (the default branch; `main` is stale)
#   ThunderKittens Megakernels main@7309cec

MEGAKERNEL_SM100=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HELION_REPO=$(cd "$MEGAKERNEL_SM100/../.." && pwd)

# Defaults are the machine the numbers were taken on.
: "${HELION_PY:=$HOME/.conda/envs/muse-dev/bin/python}"
: "${VLLM_PY:=$HOME/local/hvm_envs/vllm_main/bin/python}"
: "${SGLANG_PY:=$HOME/local/hvm_envs/sglang/bin/python}"
: "${MIRAGE_DIR:=$HOME/local/mirage-mpk}"
: "${TK_DIR:=$HOME/local/Megakernels}"
: "${CUDA_HOME:=/usr/local/cuda-13.0}"
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export NVSHMEM_DISABLE_CUDA_VMM=1
# vLLM compiles its layers afresh in every run instead of loading artifacts a
# test or an earlier checkout left in its shared cache.
export VLLM_DISABLE_COMPILE_CACHE=1

# Host fixes for that machine, each applied only where its path exists: a newer
# libstdc++ for FlashInfer's prebuilt libraries (HOST_LIBS), a local FlashInfer
# cubin cache (FLASHINFER_CUBIN_DIR, which VLLM_HAS_FLASHINFER_CUBIN tells vLLM
# about: otherwise vLLM probes NVIDIA's download server and, finding it
# unreachable, turns TRT-LLM attention off), and pyfix, which falls back to
# FlashInfer's JIT build when one of its AOT libraries cannot load.
: "${HOST_LIBS:=$HOME/.conda/envs/muse-dev/lib}"
: "${PYFIX:=$MEGAKERNEL_SM100/pyfix}"
: "${FLASHINFER_CUBIN_DIR:=$HOME/local/hvm_envs/fi_cubins}"
if [ -d "$HOST_LIBS" ]; then
  export LD_LIBRARY_PATH=$HOST_LIBS:${LD_LIBRARY_PATH:-}
fi
if [ -d "$FLASHINFER_CUBIN_DIR" ]; then
  export FLASHINFER_CUBIN_DIR
  export VLLM_HAS_FLASHINFER_CUBIN=1
else
  unset FLASHINFER_CUBIN_DIR
fi

# Run an environment's Python with its own tools (ninja for JIT builds) first on PATH.
in_env() {
  local python=$1
  shift
  if [ ! -x "$python" ]; then
    echo "env.sh: no Python at $python; set HELION_PY, VLLM_PY or SGLANG_PY" >&2
    return 1
  fi
  PATH=$(dirname "$python"):$PATH PYTHONPATH=$HELION_REPO:$PYFIX "$python" "$@"
}
helion_python() { in_env "$HELION_PY" "$@"; }
vllm_python() { in_env "$VLLM_PY" "$@"; }
sglang_python() { in_env "$SGLANG_PY" "$@"; }

# Print which of the given GPUs (comma-separated indices) run a process.
busy_gpus() {
  local busy="" uuid
  for gpu in ${1//,/ }; do
    uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then
      busy="$busy $gpu"
    fi
  done
  echo "$busy"
}

# descends PID ROOT: PID is ROOT or one of its descendants (walks parent PIDs).
descends() {
  local p=$1
  while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
    [ "$p" = "$2" ] && return 0
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  return 1
}

# gpu_pids GPUS: print the PIDs of the processes on the given GPUs.
gpu_pids() {
  local uuids=" " uuid pid
  for gpu in ${1//,/ }; do
    uuids="$uuids$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader) "
  done
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
    while IFS=', ' read -r uuid pid; do
      if [[ "$uuids" == *" $uuid "* ]]; then
        echo "$pid"
      fi
    done
}

# Wait until none of the given GPUs runs another process: timings on a shared
# GPU are not comparable.
wait_idle() {
  local deadline=$((SECONDS + ${MAX_WAIT:-3600})) busy
  while true; do
    busy=$(busy_gpus "$1")
    [ -z "$busy" ] && return 0
    if [ $SECONDS -ge $deadline ]; then
      echo "GPUs$busy stayed busy; not timing on a shared GPU" >&2
      return 1
    fi
    echo "waiting for GPUs$busy to go idle" >&2
    sleep 30
  done
}

# timed GPUS COMMAND...: run a timing command once its GPUs are idle, checking
# every 2 s while it runs for another process on them; run it again if one
# appeared (up to 3 attempts).
timed() {
  local gpus=$1 pid shared mine p
  shift
  for attempt in 1 2 3; do
    wait_idle "$gpus" || return 1
    "$@" &
    pid=$!
    shared=""
    # The step's own processes seen on its GPUs: NVML keeps listing a rank for
    # a while after it exits, when its parent can no longer be looked up.
    mine=" "
    while kill -0 "$pid" 2>/dev/null; do
      for p in $(gpu_pids "$gpus"); do
        if [[ "$mine" == *" $p "* ]]; then
          continue
        elif descends "$p" "$pid"; then
          mine="$mine$p "
        else
          echo "foreign process on GPUs $gpus: $p $(ps -o ppid=,comm= -p "$p" 2>/dev/null)" >&2
          shared=1
        fi
      done
      sleep 2
    done
    wait "$pid" || return 1
    [ -z "$shared" ] && return 0
    echo "another process was on GPUs $gpus during attempt $attempt; timing again" >&2
  done
  echo "GPUs $gpus stayed shared; giving up" >&2
  return 1
}
