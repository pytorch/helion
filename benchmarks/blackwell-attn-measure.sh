#!/bin/bash
set -uxo pipefail

RUNID=$(echo result_* | xargs -n1 | grep -v '\*' | wc -l)
RUNDIR=$PWD/result_$RUNID
mkdir $RUNDIR
set -e
nvidia-smi > $RUNDIR/nvidia-smi.log
lscpu > $RUNDIR/lscpu.log
hostname > $RUNDIR/hostname.log
. ./venv-fb-triton/bin/activate
uv pip list > $RUNDIR/fb-pip-list.log
deactivate
. venv-stock-triton/bin/activate
uv pip list > $RUNDIR/stock-pip-list.log
deactivate
find . -type d -name ".git" | while read gitdir; do
    repo_dir=$(dirname "$gitdir")
    commit_hash=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null)
    if [ -n "$commit_hash" ]; then
        echo "$repo_dir: $commit_hash" >> $RUNDIR/git-list.log
    fi
done

root=$PWD
cd helion
HIDDEN_DIM=2048
TOTAL_TOKENS=16384
export WITH_GLUON=1
export HELION_BENCHMARK_DISABLE_LOGGING=1
for DHEAD in 64 128; do
  NHEADS=$(($HIDDEN_DIM / $DHEAD))
  for SEQLEN in 2048 4096 8192; do
    BATCH=$(($TOTAL_TOKENS / $SEQLEN))
    for only in cudnn_sdpa; do
      $root/venv-stock-triton/bin/python benchmarks/run.py --kernel blackwell_attentions --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only-venv_stock-triton.log
    done

    for venv in stock-triton fb-triton; do
      for only in helion_blackwell_attention_tritonbench; do
        $root/venv-$venv/bin/python benchmarks/run.py --kernel blackwell_attentions --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only-venv_$venv.log
      done
    done

    for only in helion_blackwell_attention_tritonbench; do
      WITH_ACC=1 $root/venv-fb-triton/bin/python benchmarks/run.py --kernel blackwell_attentions --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only-venv_fb-triton-acc.log
    done

      pushd benchmarks/tritonbench
      for only in sdpa triton_tutorial_flash_v2 triton_tutorial_flash_v2_tma flex_attention; do
        $root/venv-stock-triton/bin/python run.py --op flash_attention --d-head $DHEAD --seq-len $SEQLEN --batch $BATCH --n-heads $NHEADS --metrics tflops --simple-output --rep 3000 --sleep 1.0 --num-inputs 1 --only $only --force --input-id 0 |& tee $RUNDIR/dhead_$DHEAD-seqlen_$SEQLEN-only_$only-venv_stock-triton.log
      done
    popd
  done
done
