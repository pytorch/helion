[[ -z "$RANK_OFFSET" ]] && { echo "Error: RANK_OFFSET is not set"; exit 1; }
[[ -z "$SHARD" ]] && { echo "Error: SHARD is not set"; exit 1; }
[[ -z "$WORLD_SIZE" ]] && { echo "Error: WORLD_SIZE is not set"; exit 1; }

GPU_ID=$((RANK_OFFSET+SHARD-1))

# NOTE: use `nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,clocks.video --format=csv` to check value after set

# Keep the driver loaded and clocks resident
sudo nvidia-smi -pm 1 -i ${GPU_ID}

# Lock SM clocks
# No need to lock HBM clocks since SM clocks are low and won't trigger thermal throttling
sudo nvidia-smi -lgc 1620,1620 -i ${GPU_ID}   # H100 and B200

# Set power limit
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -n1 | awk '{print $2}')
if [[ "$GPU_MODEL" == "H100" ]]; then
    DESIRED_POWER=500
elif [[ "$GPU_MODEL" == "GB200" ]]; then
    DESIRED_POWER=1200
elif [[ "$GPU_MODEL" == "B200" ]]; then
    DESIRED_POWER=750
else
    DESIRED_POWER=500
fi
sudo nvidia-smi --power-limit=${DESIRED_POWER} -i ${GPU_ID}

# Capture timestamp once for consistent filename
TIMESTAMP=$(date +%s)
OUTPUT_DIR="benchmarks_results/benchmarks_autotune_${TIMESTAMP}_input_shard_${SHARD}_of_${WORLD_SIZE}"

KERNEL_NAME_LIST=(
    "rms_norm"
    "layer_norm"
    "softmax"
    "cross_entropy"
    "sum"
    "jagged_mean"
    "vector_add"
    "embedding"
    "vector_exp"
)

# Retry until success
attempt=0
for KERNEL_NAME in "${KERNEL_NAME_LIST[@]}"; do
    mkdir -p ${OUTPUT_DIR} || true
    while true; do
    # while (( attempt < 10 )); do
        attempt=$((attempt + 1))
        echo "Attempt $attempt: Running benchmark (Helion autotuning run) for shard ${SHARD}/${WORLD_SIZE}..."

        # OUTPUT_FILE="benchmarks_autotune_${TIMESTAMP}_input_shard_${SHARD}_of_${WORLD_SIZE}.txt"

        OUTPUT_FILE="${OUTPUT_DIR}/${KERNEL_NAME}_autotune.log"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python benchmarks/run.py --input-shard ${SHARD}/${WORLD_SIZE} --kernel ${KERNEL_NAME} --metrics latency,speedup,accuracy,tflops,gbps --latency-measure-mode profiler --csv --output-dir ${OUTPUT_DIR} >"${OUTPUT_FILE}" 2>&1

        exit_code=$?
        # Check for success: exit code 0 AND no exception message in output
        if [ $exit_code -eq 0 ] && ! grep -q "Caught exception, terminating early with partial results" "${OUTPUT_FILE}"; then
            mv ${OUTPUT_DIR}/${KERNEL_NAME}.csv ${OUTPUT_DIR}/${KERNEL_NAME}_autotune.csv
            echo "Success! Benchmark completed for shard ${SHARD}/${WORLD_SIZE}"
            break
        else
            echo "Failed with exit code $exit_code. Retrying..."
            sleep 10  # wait a few seconds before retrying
        fi
    done

    echo "Sleeping for 300 seconds to avoid GPU thermal throttling before running cached benchmark for ${KERNEL_NAME}..."
    sleep 300  # wait a period of time to avoid GPU thermal throttling

    echo "Running benchmark (Helion cached run) for shard ${SHARD}/${WORLD_SIZE}..."
    OUTPUT_FILE="${OUTPUT_DIR}/${KERNEL_NAME}_cached.log"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python benchmarks/run.py --input-shard ${SHARD}/${WORLD_SIZE} --kernel ${KERNEL_NAME} --metrics latency,speedup,accuracy,tflops,gbps --latency-measure-mode profiler --csv --output-dir ${OUTPUT_DIR} >"${OUTPUT_FILE}" 2>&1
    mv ${OUTPUT_DIR}/${KERNEL_NAME}.csv ${OUTPUT_DIR}/${KERNEL_NAME}_cached.csv
done

# Unlock SM clocks
sudo nvidia-smi -rgc -i ${GPU_ID}

# Reset power limit
if [[ "$GPU_MODEL" == "H100" ]]; then
    POWER_CAP=650
elif [[ "$GPU_MODEL" == "GB200" ]]; then
    POWER_CAP=1200
elif [[ "$GPU_MODEL" == "B200" ]]; then
    POWER_CAP=750
else
    POWER_CAP=500
fi
sudo nvidia-smi --power-limit=${POWER_CAP} -i ${GPU_ID}

# Example: Runs the 1st shard of input on GPU-0:
# SHARD=1 RANK_OFFSET=4 WORLD_SIZE=4 bash benchmarks/run_input_shard.sh
