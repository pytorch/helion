[[ -z "$RANK_OFFSET" ]] && { echo "Error: RANK_OFFSET is not set"; exit 1; }
[[ -z "$SHARD" ]] && { echo "Error: SHARD is not set"; exit 1; }
[[ -z "$WORLD_SIZE" ]] && { echo "Error: WORLD_SIZE is not set"; exit 1; }

# Capture timestamp once for consistent filename
TIMESTAMP=$(date +%s)
OUTPUT_FILE="benchmarks_autotune_${TIMESTAMP}_input_shard_${SHARD}_of_${WORLD_SIZE}.txt"
CSV_OUTPUT_DIR="benchmarks_autotune_${TIMESTAMP}_input_shard_${SHARD}_of_${WORLD_SIZE}_csv"

# Retry until success
attempt=0
while true; do
# while (( attempt < 10 )); do
    attempt=$((attempt + 1))
    echo "Attempt $attempt: Running benchmark for shard ${SHARD}/${WORLD_SIZE}..."

    # TIMESTAMP=$(date +%s)
    # OUTPUT_FILE="benchmarks_autotune_${TIMESTAMP}_input_shard_${SHARD}_of_${WORLD_SIZE}.txt"

    mkdir -p ${CSV_OUTPUT_DIR} || true
    CUDA_VISIBLE_DEVICES=$((RANK_OFFSET+SHARD-1)) python benchmarks/run.py --input-shard ${SHARD}/${WORLD_SIZE} --metrics accuracy,tflops,gbps,speedup --csv --output-dir ${CSV_OUTPUT_DIR} >"$OUTPUT_FILE" 2>&1

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Success! Benchmark completed for shard ${SHARD}/${WORLD_SIZE}"
        break
    else
        echo "Failed with exit code $exit_code. Retrying..."
        sleep 10  # wait a few seconds before retrying
    fi
done

# Runs the 1st shard of input on GPU-0:
# SHARD=1 RANK_OFFSET=4 WORLD_SIZE=4 bash benchmarks/run_input_shard.sh
