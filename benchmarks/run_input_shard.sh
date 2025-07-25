CUDA_VISIBLE_DEVICES=${RANK} python benchmarks/run.py --input-shard $((RANK+1))/${WORLD_SIZE} >benchmarks_autotune_$(date +%s)_input_shard_$((RANK+1))_of_${WORLD_SIZE}.txt 2>&1
