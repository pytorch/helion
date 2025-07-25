[[ -z "$RANK_OFFSET" ]] && { echo "Error: RANK_OFFSET is not set"; exit 1; }
[[ -z "$SHARD" ]] && { echo "Error: SHARD is not set"; exit 1; }
[[ -z "$WORLD_SIZE" ]] && { echo "Error: WORLD_SIZE is not set"; exit 1; }
CUDA_VISIBLE_DEVICES=$((RANK_OFFSET+SHARD)) python benchmarks/run.py --input-shard $((SHARD+1))/${WORLD_SIZE} --metrics tflops,gbps,speedup >benchmarks_autotune_$(date +%s)_input_shard_$((SHARD+1))_of_${WORLD_SIZE}.txt 2>&1
