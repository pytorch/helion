# Multi-GPU Resource Contention Verification Strategy

## Experiment Design

### 1. Baseline Measurement (Single GPU)
Run each benchmark in isolation to establish baseline auto-tuning results:

```bash
# Run benchmarks one at a time
for gpu in 0 1 2 3; do
    for bench in flash_attention_v2 gemm_benchmark softmax; do
        CUDA_VISIBLE_DEVICES=$gpu python benchmarks/run.py --benchmark $bench \
            --output baseline_gpu${gpu}_${bench}.json
    done
done
```

### 2. Concurrent Execution Test
Run benchmarks simultaneously on different GPUs:

```bash
# Run all GPUs concurrently
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python benchmarks/run.py --benchmark flash_attention_v2 \
        --output concurrent_gpu${gpu}.json &
done
wait
```

### 3. Process Isolation Test
Use separate processes with proper isolation:

```bash
# Create isolation script
cat > isolated_benchmark.sh << 'EOF'
#!/bin/bash
GPU=$1
BENCH=$2
OUTPUT=$3

# Set CPU affinity based on GPU
if [ $GPU -eq 0 ]; then
    taskset -c 0-15
elif [ $GPU -eq 1 ]; then
    taskset -c 16-31
elif [ $GPU -eq 2 ]; then
    taskset -c 32-47
else
    taskset -c 48-63
fi

# Set NUMA node
numactl --cpunodebind=$GPU --membind=$GPU \
    env CUDA_VISIBLE_DEVICES=$GPU \
    python benchmarks/run.py --benchmark $BENCH --output $OUTPUT
EOF

chmod +x isolated_benchmark.sh

# Run with isolation
for gpu in 0 1 2 3; do
    ./isolated_benchmark.sh $gpu flash_attention_v2 isolated_gpu${gpu}.json &
done
wait
```

## Monitoring Scripts

### 1. GPU Resource Monitor
```python
# gpu_monitor.py
import nvidia_ml_py as nvml
import time
import json
from datetime import datetime
import sys

def monitor_gpus(output_file, duration=300):
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    
    results = {
        'start_time': datetime.now().isoformat(),
        'samples': []
    }
    
    start_time = time.time()
    while time.time() - start_time < duration:
        sample = {
            'timestamp': time.time() - start_time,
            'gpus': []
        }
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get power
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to W
            
            # Get temperature
            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            
            # Get PCIe throughput
            tx_bytes = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
            rx_bytes = nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
            
            gpu_data = {
                'gpu_id': i,
                'gpu_util': util.gpu,
                'mem_util': util.memory,
                'mem_used_gb': mem_info.used / (1024**3),
                'power_w': power,
                'temp_c': temp,
                'pcie_tx_mb': tx_bytes / (1024**2) if tx_bytes else 0,
                'pcie_rx_mb': rx_bytes / (1024**2) if rx_bytes else 0
            }
            sample['gpus'].append(gpu_data)
        
        results['samples'].append(sample)
        time.sleep(0.1)  # 100ms sampling
    
    nvml.nvmlShutdown()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    monitor_gpus(sys.argv[1] if len(sys.argv) > 1 else 'gpu_monitor.json')
```

### 2. System Resource Monitor
```python
# system_monitor.py
import psutil
import time
import json
from datetime import datetime
import subprocess

def get_numa_stats():
    try:
        result = subprocess.run(['numastat', '-c'], capture_output=True, text=True)
        return result.stdout
    except:
        return None

def monitor_system(output_file, duration=300):
    results = {
        'start_time': datetime.now().isoformat(),
        'samples': []
    }
    
    start_time = time.time()
    while time.time() - start_time < duration:
        sample = {
            'timestamp': time.time() - start_time,
            'cpu_percent': psutil.cpu_percent(interval=0.1, percpu=True),
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'used_gb': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'numa_stats': get_numa_stats()
        }
        
        # Get per-process stats for python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if 'python' in proc.info['name']:
                python_processes.append({
                    'pid': proc.info['pid'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent']
                })
        
        sample['python_processes'] = python_processes
        results['samples'].append(sample)
        time.sleep(0.1)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    import sys
    monitor_system(sys.argv[1] if len(sys.argv) > 1 else 'system_monitor.json')
```

### 3. PCIe Bandwidth Monitor
```bash
#!/bin/bash
# pcie_monitor.sh

OUTPUT_FILE=${1:-pcie_monitor.log}
DURATION=${2:-300}

# Monitor PCIe bandwidth using nvidia-smi
end=$((SECONDS+DURATION))

echo "timestamp,gpu,pcie_rx_mb,pcie_tx_mb" > $OUTPUT_FILE

while [ $SECONDS -lt $end ]; do
    timestamp=$(date +%s.%N)
    
    # Get PCIe throughput for each GPU
    for gpu in 0 1 2 3; do
        stats=$(nvidia-smi -i $gpu --query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.rx.throughput,pcie.tx.throughput --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ]; then
            rx=$(echo $stats | cut -d',' -f3)
            tx=$(echo $stats | cut -d',' -f4)
            echo "$timestamp,$gpu,$rx,$tx" >> $OUTPUT_FILE
        fi
    done
    
    sleep 0.1
done
```

## Analysis Scripts

### 1. Variance Analysis
```python
# analyze_variance.py
import json
import numpy as np
import sys

def analyze_tuning_variance(baseline_files, concurrent_files):
    """Compare auto-tuning results between baseline and concurrent runs"""
    
    baseline_results = {}
    concurrent_results = {}
    
    # Load baseline results
    for f in baseline_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            kernel_name = data.get('kernel_name', f)
            baseline_results[kernel_name] = {
                'best_config': data.get('best_config'),
                'best_time': data.get('best_time_ms'),
                'all_times': data.get('all_benchmark_times', [])
            }
    
    # Load concurrent results
    for f in concurrent_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            kernel_name = data.get('kernel_name', f)
            concurrent_results[kernel_name] = {
                'best_config': data.get('best_config'),
                'best_time': data.get('best_time_ms'),
                'all_times': data.get('all_benchmark_times', [])
            }
    
    # Compare results
    print("Auto-tuning Variance Analysis")
    print("=" * 60)
    
    for kernel in baseline_results:
        if kernel in concurrent_results:
            base_time = baseline_results[kernel]['best_time']
            conc_time = concurrent_results[kernel]['best_time']
            
            if base_time and conc_time:
                diff_pct = ((conc_time - base_time) / base_time) * 100
                
                print(f"\nKernel: {kernel}")
                print(f"  Baseline best time: {base_time:.3f} ms")
                print(f"  Concurrent best time: {conc_time:.3f} ms")
                print(f"  Difference: {diff_pct:+.1f}%")
                
                # Check if best configs are different
                if baseline_results[kernel]['best_config'] != concurrent_results[kernel]['best_config']:
                    print(f"  WARNING: Different optimal configurations found!")
                
                # Calculate variance in timing measurements
                base_times = baseline_results[kernel]['all_times']
                conc_times = concurrent_results[kernel]['all_times']
                
                if base_times and conc_times:
                    base_std = np.std(base_times)
                    conc_std = np.std(conc_times)
                    print(f"  Baseline std dev: {base_std:.3f} ms")
                    print(f"  Concurrent std dev: {conc_std:.3f} ms")
                    print(f"  Variance increase: {(conc_std/base_std - 1)*100:+.1f}%")

if __name__ == '__main__':
    # Usage: python analyze_variance.py baseline1.json,baseline2.json concurrent1.json,concurrent2.json
    baseline_files = sys.argv[1].split(',')
    concurrent_files = sys.argv[2].split(',')
    analyze_tuning_variance(baseline_files, concurrent_files)
```

### 2. Resource Correlation Analysis
```python
# correlate_resources.py
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def correlate_performance_with_resources(perf_file, gpu_monitor_file, system_monitor_file):
    """Correlate performance variations with resource utilization"""
    
    # Load performance data
    with open(perf_file, 'r') as f:
        perf_data = json.load(f)
    
    # Load GPU monitoring data
    with open(gpu_monitor_file, 'r') as f:
        gpu_data = json.load(f)
    
    # Load system monitoring data
    with open(system_monitor_file, 'r') as f:
        system_data = json.load(f)
    
    # Extract timing variations from performance data
    benchmark_times = perf_data.get('detailed_timings', [])
    
    # Align monitoring data with benchmark runs
    # This is simplified - in practice you'd need timestamp alignment
    
    # Calculate correlations
    correlations = {}
    
    # GPU utilization correlation
    gpu_utils = [s['gpus'][0]['gpu_util'] for s in gpu_data['samples']]
    if len(gpu_utils) >= len(benchmark_times):
        corr, p_value = stats.pearsonr(benchmark_times[:len(gpu_utils)], gpu_utils[:len(benchmark_times)])
        correlations['gpu_utilization'] = {'correlation': corr, 'p_value': p_value}
    
    # PCIe bandwidth correlation
    pcie_tx = [s['gpus'][0]['pcie_tx_mb'] for s in gpu_data['samples']]
    if len(pcie_tx) >= len(benchmark_times):
        corr, p_value = stats.pearsonr(benchmark_times[:len(pcie_tx)], pcie_tx[:len(benchmark_times)])
        correlations['pcie_bandwidth'] = {'correlation': corr, 'p_value': p_value}
    
    # CPU utilization correlation
    cpu_utils = [np.mean(s['cpu_percent']) for s in system_data['samples']]
    if len(cpu_utils) >= len(benchmark_times):
        corr, p_value = stats.pearsonr(benchmark_times[:len(cpu_utils)], cpu_utils[:len(benchmark_times)])
        correlations['cpu_utilization'] = {'correlation': corr, 'p_value': p_value}
    
    # Print results
    print("Resource Correlation Analysis")
    print("=" * 60)
    for resource, stats in correlations.items():
        print(f"{resource}:")
        print(f"  Correlation: {stats['correlation']:.3f}")
        print(f"  P-value: {stats['p_value']:.3e}")
        print(f"  Significant: {'Yes' if stats['p_value'] < 0.05 else 'No'}")
        print()
    
    return correlations

# Visualization function
def plot_resource_timeline(gpu_monitor_file, system_monitor_file, output_file='resource_timeline.png'):
    """Create a timeline visualization of resource usage"""
    
    with open(gpu_monitor_file, 'r') as f:
        gpu_data = json.load(f)
    
    with open(system_monitor_file, 'r') as f:
        system_data = json.load(f)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Extract data
    timestamps = [s['timestamp'] for s in gpu_data['samples']]
    
    # Plot GPU utilization
    for gpu_id in range(4):
        gpu_utils = [s['gpus'][gpu_id]['gpu_util'] for s in gpu_data['samples']]
        axes[0].plot(timestamps, gpu_utils, label=f'GPU {gpu_id}')
    axes[0].set_ylabel('GPU Utilization %')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot PCIe bandwidth
    for gpu_id in range(4):
        pcie_tx = [s['gpus'][gpu_id]['pcie_tx_mb'] for s in gpu_data['samples']]
        axes[1].plot(timestamps, pcie_tx, label=f'GPU {gpu_id}')
    axes[1].set_ylabel('PCIe TX (MB/s)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot memory usage
    for gpu_id in range(4):
        mem_used = [s['gpus'][gpu_id]['mem_used_gb'] for s in gpu_data['samples']]
        axes[2].plot(timestamps, mem_used, label=f'GPU {gpu_id}')
    axes[2].set_ylabel('Memory Used (GB)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot CPU utilization
    cpu_timestamps = [s['timestamp'] for s in system_data['samples']]
    cpu_utils = [np.mean(s['cpu_percent']) for s in system_data['samples']]
    axes[3].plot(cpu_timestamps, cpu_utils, 'k-', label='Avg CPU')
    axes[3].set_ylabel('CPU Utilization %')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"Resource timeline saved to {output_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 4:
        correlate_performance_with_resources(sys.argv[1], sys.argv[2], sys.argv[3])
        if len(sys.argv) >= 5:
            plot_resource_timeline(sys.argv[2], sys.argv[3], sys.argv[4])
```

## Experiment Execution Script

```bash
#!/bin/bash
# run_verification_experiment.sh

# Configuration
BENCHMARKS="flash_attention_v2 gemm_benchmark softmax"
GPUS="0 1 2 3"
DURATION=300  # 5 minutes per test

echo "Multi-GPU Resource Contention Verification Experiment"
echo "====================================================="

# Create results directory
mkdir -p verification_results
cd verification_results

# Step 1: Baseline measurements (isolated runs)
echo "Step 1: Running baseline measurements..."
for gpu in $GPUS; do
    for bench in $BENCHMARKS; do
        echo "  Running $bench on GPU $gpu (isolated)..."
        
        # Start monitors
        python ../gpu_monitor.py gpu_monitor_baseline_${gpu}_${bench}.json &
        GPU_MON_PID=$!
        python ../system_monitor.py system_monitor_baseline_${gpu}_${bench}.json &
        SYS_MON_PID=$!
        
        # Run benchmark
        CUDA_VISIBLE_DEVICES=$gpu python ../benchmarks/run.py \
            --benchmark $bench \
            --output baseline_${gpu}_${bench}.json \
            --duration $DURATION
        
        # Stop monitors
        kill $GPU_MON_PID $SYS_MON_PID
        wait
        
        # Cool down period
        sleep 30
    done
done

# Step 2: Concurrent execution test
echo "Step 2: Running concurrent execution test..."

# Start global monitors
python ../gpu_monitor.py gpu_monitor_concurrent.json &
GPU_MON_PID=$!
python ../system_monitor.py system_monitor_concurrent.json &
SYS_MON_PID=$!
../pcie_monitor.sh pcie_monitor_concurrent.log $DURATION &
PCIE_MON_PID=$!

# Launch benchmarks concurrently
PIDS=""
for gpu in $GPUS; do
    CUDA_VISIBLE_DEVICES=$gpu python ../benchmarks/run.py \
        --benchmark flash_attention_v2 \
        --output concurrent_gpu${gpu}.json \
        --duration $DURATION &
    PIDS="$PIDS $!"
done

# Wait for all benchmarks to complete
for pid in $PIDS; do
    wait $pid
done

# Stop monitors
kill $GPU_MON_PID $SYS_MON_PID $PCIE_MON_PID
wait

# Step 3: Process isolation test
echo "Step 3: Running process isolation test..."

# Start global monitors
python ../gpu_monitor.py gpu_monitor_isolated.json &
GPU_MON_PID=$!
python ../system_monitor.py system_monitor_isolated.json &
SYS_MON_PID=$!

# Launch with isolation
PIDS=""
for gpu in $GPUS; do
    ../isolated_benchmark.sh $gpu flash_attention_v2 isolated_gpu${gpu}.json &
    PIDS="$PIDS $!"
done

# Wait for completion
for pid in $PIDS; do
    wait $pid
done

# Stop monitors
kill $GPU_MON_PID $SYS_MON_PID
wait

# Step 4: Analysis
echo "Step 4: Analyzing results..."

# Compare baseline vs concurrent
python ../analyze_variance.py \
    baseline_0_flash_attention_v2.json,baseline_1_flash_attention_v2.json,baseline_2_flash_attention_v2.json,baseline_3_flash_attention_v2.json \
    concurrent_gpu0.json,concurrent_gpu1.json,concurrent_gpu2.json,concurrent_gpu3.json \
    > variance_analysis.txt

# Correlation analysis
for gpu in 0 1 2 3; do
    python ../correlate_resources.py \
        concurrent_gpu${gpu}.json \
        gpu_monitor_concurrent.json \
        system_monitor_concurrent.json \
        > correlation_gpu${gpu}.txt
done

# Generate visualizations
python ../correlate_resources.py \
    concurrent_gpu0.json \
    gpu_monitor_concurrent.json \
    system_monitor_concurrent.json \
    resource_timeline_concurrent.png

echo "Experiment complete! Results in verification_results/"
```

## Expected Results and Interpretation

### Indicators of Resource Contention:
1. **Higher variance** in auto-tuning times during concurrent runs
2. **Different optimal configurations** found in concurrent vs isolated runs  
3. **Positive correlation** between PCIe bandwidth usage and timing variance
4. **CPU utilization spikes** correlating with performance degradation
5. **Suboptimal auto-tuning convergence** in concurrent scenarios

### Mitigation Strategies:
1. **Process Isolation**: Run each GPU benchmark in a separate process
2. **CPU Affinity**: Pin processes to specific CPU cores
3. **NUMA Awareness**: Bind memory to local NUMA nodes
4. **GPU Topology**: Consider PCIe topology when assigning GPUs
5. **Persistent Tuning Cache**: Implement cross-run persistence to avoid re-tuning
6. **MIG/MPS**: Use NVIDIA MIG or MPS for better GPU isolation
7. **Exclusive Compute Mode**: Set GPUs to exclusive process mode