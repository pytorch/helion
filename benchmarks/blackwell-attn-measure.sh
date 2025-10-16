#!/bin/bash
set -euxo pipefail

RUNID=$(echo result_* | xargs -n1 | wc -l)
RUNDIR=$PWD/result_$RUNID
mkdir $RUNDIR
nvidia-smi > $RUNDIR/nvidia-smi.log
lscpu > $RUNDIR/lscpu.log
hostname > $RUNDIR/hostname.log
uv pip list > $RUNDIR/pip-list.log
find . -type d -name ".git" | while read gitdir; do
    repo_dir=$(dirname "$gitdir")
    commit_hash=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null)
    if [ -n "$commit_hash" ]; then
        echo "$repo_dir: $commit_hash" >> $RUNDIR/git-list.log
    fi
done

cd helion
HIDDEN_DIM=2048
TOTAL_TOKENS=16384
export WITH_GLUON=1
export HELION_BENCHMARK_DISABLE_LOGGING=1
for DHEAD in 64 128; do
  NHEADS=$(($HIDDEN_DIM / $DHEAD))
  for SEQLEN in 2048 4096 8192; do
    BATCH=$(($TOTAL_TOKENS / $SEQLEN))
    for only in triton_tutorial_flash_dp_persistent_blackwell gluon_blackwell_tutorial_persistent_fwd cudnn_sdpa helion_blackwell_attention_tritonbench; do
      python benchmarks/run.py --kernel blackwell_attentions --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only.log
    done

    pushd benchmarks/tritonbench
      for only in aten sdpa triton_tutorial_flash_v2 triton_tutorial_flash_v2_tma flex_attention; do
        python run.py --op flash_attention --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only.log
      done
    popd
  done
done
