# GPU Benchmark Isolation and Contention Analysis

This directory contains scripts for running isolated GPU benchmarks and analyzing resource contention in Helion's auto-tuning system.

## Quick Start

### 1. Run a Simple Isolated Benchmark

```bash
# Run on specific GPUs with full isolation
python monitoring_scripts/isolated_benchmark_runner.py \
    --benchmark flash_attention_v2 \
    --mode concurrent \
    --gpus 0,1,2,3 \
    --duration 300
```

### 2. Run Complete Contention Experiment

```bash
# This runs baseline, concurrent naive, and concurrent isolated benchmarks
# then analyzes the results
python monitoring_scripts/run_contention_experiment.py \
    --benchmark flash_attention_v2 \
    --duration 300
```

## Why Process Isolation Matters

When running `python benchmarks/run.py` on multiple GPUs simultaneously using only `CUDA_VISIBLE_DEVICES`, several resources are shared:

1. **CPU Resources**: All processes compete for CPU cores
2. **Memory Bandwidth**: Shared system memory access
3. **PCIe Bus**: Shared bandwidth for GPU communication
4. **CUDA Runtime**: Shared driver state and synchronization

This causes:
- **Noisy auto-tuning results**: Timing measurements vary due to contention
- **Different optimal configurations**: Auto-tuner may select suboptimal configs
- **Performance degradation**: Overall slower execution

## Isolation Techniques

### Level 1: Basic GPU Isolation (Not Recommended)
```bash
# Naive approach - high contention
CUDA_VISIBLE_DEVICES=0 python benchmarks/run.py --benchmark flash_attention_v2 &
CUDA_VISIBLE_DEVICES=1 python benchmarks/run.py --benchmark flash_attention_v2 &
```

### Level 2: CPU Pinning
```bash
# Better - reduces CPU contention
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python benchmarks/run.py --benchmark flash_attention_v2 &
CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 python benchmarks/run.py --benchmark flash_attention_v2 &
```

### Level 3: NUMA Binding
```bash
# Even better - adds memory locality
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 python benchmarks/run.py --benchmark flash_attention_v2 &
CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 python benchmarks/run.py --benchmark flash_attention_v2 &
```

### Level 4: Full Isolation (Recommended)
```bash
# Best - combines all techniques
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 numactl --cpunodebind=0 --membind=0 \
    python benchmarks/run.py --benchmark flash_attention_v2 &
```

## Scripts Overview

### `isolated_benchmark_runner.py`
Main script for running isolated benchmarks with automatic CPU/NUMA affinity configuration.

**Features:**
- Automatic topology detection
- CPU core allocation per GPU
- NUMA-aware memory binding
- Process isolation
- Result aggregation

**Usage:**
```bash
# Sequential runs (with cooldown)
python isolated_benchmark_runner.py --benchmark gemm_benchmark --mode sequential

# Concurrent isolated runs
python isolated_benchmark_runner.py --benchmark flash_attention_v2 --mode concurrent

# Specific GPUs only
python isolated_benchmark_runner.py --benchmark softmax --gpus 0,2 --mode concurrent
```

### `run_contention_experiment.py`
Orchestrates a complete experiment to measure resource contention effects.

**Phases:**
1. **Baseline**: Sequential isolated runs on each GPU
2. **Concurrent Naive**: Simultaneous runs with only CUDA_VISIBLE_DEVICES
3. **Concurrent Isolated**: Simultaneous runs with full isolation
4. **Analysis**: Compares results to quantify contention

**Usage:**
```bash
# Full experiment
python run_contention_experiment.py

# Custom settings
python run_contention_experiment.py \
    --benchmark gemm_benchmark \
    --gpus 0,1 \
    --duration 600 \
    --output-dir my_experiment
```

### `gpu_monitor.py`
Monitors GPU metrics during benchmark execution.

**Metrics:**
- GPU/Memory utilization
- Power consumption
- Temperature
- PCIe throughput
- Process count

**Usage:**
```bash
# Monitor for 5 minutes
python gpu_monitor.py output.json --duration 300

# Higher frequency sampling
python gpu_monitor.py output.json --interval 0.05
```

### `system_monitor.py`
Monitors system-wide metrics.

**Metrics:**
- CPU utilization (per-core)
- Memory usage
- NUMA statistics
- Process information
- I/O statistics

**Usage:**
```bash
python system_monitor.py output.json --duration 300
```

### `analyze_variance.py`
Analyzes auto-tuning variance between different runs.

**Analysis includes:**
- Best time comparison
- Configuration differences
- Timing variance statistics
- Cross-GPU consistency

**Usage:**
```bash
# Compare baseline vs concurrent
python analyze_variance.py \
    baseline_gpu0.json,baseline_gpu1.json \
    concurrent_gpu0.json,concurrent_gpu1.json
```

### `simple_isolated_example.py`
Educational script showing progression from naive to fully isolated execution.

**Examples:**
1. Naive concurrent execution
2. Basic process isolation
3. CPU pinning with taskset
4. NUMA-aware execution
5. Full isolation (recommended)
6. Using the helper script

## Expected Results

### Without Isolation (Naive)
- **Timing variance**: High (CV > 0.1)
- **Config mismatches**: 20-40% of runs
- **Performance degradation**: 5-20% slower
- **Auto-tuning quality**: Suboptimal

### With Full Isolation
- **Timing variance**: Low (CV < 0.05)
- **Config mismatches**: < 5%
- **Performance**: Consistent
- **Auto-tuning quality**: Optimal

## System Requirements

- Linux system with NVIDIA GPUs
- Python packages: `psutil`, `nvidia-ml-py`
- System tools: `taskset`, `numactl`, `nvidia-smi`
- Multi-core CPU (preferably multi-socket for NUMA)

## Best Practices

1. **Always use isolation** for production auto-tuning
2. **Match GPU-NUMA affinity** when possible
3. **Allow cooldown periods** between sequential runs
4. **Monitor resource usage** to verify isolation
5. **Save auto-tuning results** for reuse

## Troubleshooting

### "Command not found: taskset/numactl"
```bash
# Install on Ubuntu/Debian
sudo apt-get install util-linux numactl

# Install on RHEL/CentOS
sudo yum install util-linux numactl
```

### "Permission denied" errors
- Run with appropriate permissions for process affinity
- Check if cgroups are restricting CPU access

### Inconsistent results
- Verify no other GPU workloads are running
- Check for thermal throttling
- Ensure consistent GPU boost clocks

## Example Workflow

```bash
# 1. First, check your system topology
nvidia-smi topo -m
lscpu | grep NUMA

# 2. Run a quick test
python monitoring_scripts/isolated_benchmark_runner.py \
    --benchmark flash_attention_v2 \
    --mode concurrent \
    --gpus 0,1 \
    --duration 60

# 3. Run full experiment if results look good
python monitoring_scripts/run_contention_experiment.py \
    --benchmark flash_attention_v2 \
    --duration 300

# 4. Analyze results
cd contention_experiment_results/experiment_*/
python ../../monitoring_scripts/analyze_variance.py \
    baseline/*.json concurrent/*.json
```

## Conclusion

Proper process isolation is crucial for accurate auto-tuning results in multi-GPU systems. Use the provided scripts to ensure your benchmarks run with minimal resource contention, leading to consistent and optimal auto-tuning configurations.