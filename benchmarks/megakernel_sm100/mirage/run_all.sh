#!/bin/bash
# Mirage MPK per-layer cold-L2 decode cost: every workload in $NPROC separate processes (process
# p > 1 uses --placement-seed p-1, so its builds land at other addresses), Qwen3 with split-KV
# attention in $NPROC processes too (it can be the Qwen3 headline), Llama with demo/llama3's graph in
# one process, all on one GPU; then summarize.py writes $RESULTS/mirage_*.json.
#   GPU=6 RESULTS=/path/to/results bash run_all.sh [workload ...]
# Env: GPU, RESULTS (required); RUNS (per-process JSONs, default $RESULTS/mirage_runs); NPROC (2);
#   from ../env.sh: MIRAGE_DIR (the mpk checkout with mirage_mpk_bench.patch, built; README.md),
#   CUDA_HOME (CUDA 13.0); MIRAGE_PY (Python with Mirage's requirements, default
#   $HOME/local/mirage-bench/.venv/bin/python); MIRAGE_BUILD (where the megakernels are compiled,
#   default $RESULTS/mirage_build, so concurrent runs do not share builds).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../env.sh
source "$HERE/../env.sh"
: "${GPU:?set GPU to the index of an idle B200}"
: "${RESULTS:?set RESULTS to the results directory}"
NPROC=${NPROC:-2}
RUNS=${RUNS:-$RESULTS/mirage_runs}
: "${MIRAGE_PY:=$HOME/local/mirage-bench/.venv/bin/python}"
: "${MIRAGE_BUILD:=$RESULTS/mirage_build}"
die() { echo "mirage/run_all.sh: $*" >&2; exit 1; }
[ -x "$MIRAGE_PY" ] || die "no Python at MIRAGE_PY=$MIRAGE_PY (a venv on the Helion env with Mirage's requirements.txt; see README.md)"
cuh=$MIRAGE_DIR/include/mirage/persistent_kernel/persistent_kernel.cuh
[ -f "$cuh" ] || die "MIRAGE_DIR=$MIRAGE_DIR is not a Mirage checkout (see README.md)"
grep -q mpk_bench_set_steps "$cuh" || die "$MIRAGE_DIR lacks mirage_mpk_bench.patch (see README.md)"
ls "$MIRAGE_DIR"/python/mirage/core*.so > /dev/null 2>&1 || die "$MIRAGE_DIR is not built (no python/mirage/core*.so; see README.md)"
command -v nvcc > /dev/null || die "no nvcc in CUDA_HOME=$CUDA_HOME (Mirage compiles each megakernel with CUDA 13.0)"
export MIRAGE_HOME=$MIRAGE_DIR   # mirage_layers.py imports and compiles against this checkout
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RUNS"
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
  cd /tmp && timeout 14400 "$MIRAGE_PY" "$HERE/mirage_layers.py" --build-dir "$MIRAGE_BUILD" "$@"
}

run() {  # run <name> <driver args...>: retried (exit 3 = GPU shared while timing; timeout = hang)
  local name=$1 rc
  shift
  for attempt in 1 2 3 4 5; do
    wait_free
    echo "$(date +%T) start $name (attempt $attempt)"
    timed "$GPU" one "$@" --out "$RUNS/$name.json" > "$RUNS/$name.log" 2>&1
    rc=$?
    if [ $rc = 0 ]; then
      tail -1 "$RUNS/$name.log"
      return 0
    fi
    mv "$RUNS/$name.log" "$RUNS/$name.attempt$attempt.log"
    echo "$(date +%T) $name attempt $attempt failed (exit $rc), see $RUNS/$name.attempt$attempt.log"
  done
  return 1
}

# (Llama + split-KV is not possible on mpk: its split-KV attention always applies q/k norm.)
WORKLOADS=${*:-"qwen3_8b qwen3_8b_splitkv llama32_1b qwen3_30b_moe dsv3_mla llama32_1b_demo"}
for w in $WORKLOADS; do
  n=$NPROC
  [ "$w" = dsv3_mla ] && n=$((2 * NPROC))  # MLA: per-layer cost ~ MPK's termination noise, so more builds
  for p in $(seq 1 "$n"); do
    seed=$((p - 1))
    case $w in
      qwen3_8b)      run qwen3_8b_builder_p$p --workload qwen3_8b --variant builder --placement-seed $seed ;;
      llama32_1b)    run llama32_1b_builder_p$p --workload llama32_1b --variant builder --placement-seed $seed ;;
      qwen3_30b_moe) run qwen3_30b_moe_demo_p$p --workload qwen3_30b_moe --placement-seed $seed ;;
      dsv3_mla)      run dsv3_mla_tp8_p$p --workload dsv3_mla --layers 1,2,4,8,16 --instances 4 --placement-seed $seed ;;
      # alternative Mirage configurations, to check the default is Mirage's best (summarize.py: the
      # headline is the fastest config with >= 2 processes)
      qwen3_8b_splitkv)   run qwen3_8b_builder-splitkv_p$p --workload qwen3_8b --variant builder-splitkv --placement-seed $seed ;;
      llama32_1b_demo)    [ "$p" = 1 ] && run llama32_1b_demo_p1 --workload llama32_1b --variant demo ;;
      *) die "unknown workload $w" ;;
    esac
  done
done
"$MIRAGE_PY" "$HERE/summarize.py" --runs "$RUNS" --results "$RESULTS"
