#!/usr/bin/env python3
"""
Simple examples showing different ways to run isolated GPU benchmarks.
These examples demonstrate the progression from naive to fully isolated execution.
"""

import subprocess
import os
import time
import sys

def example_1_naive_concurrent():
    """
    NAIVE APPROACH: Just setting CUDA_VISIBLE_DEVICES
    Problem: All processes share CPU, memory, and system resources
    """
    print("=" * 60)
    print("Example 1: Naive Concurrent Execution")
    print("=" * 60)
    
    processes = []
    
    # Launch benchmarks on each GPU
    for gpu_id in range(4):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        cmd = [
            sys.executable,
            'benchmarks/run.py',
            '--benchmark', 'flash_attention_v2',
            '--output', f'results_gpu{gpu_id}.json'
        ]
        
        print(f"Starting GPU {gpu_id} (naive mode)")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
    
    # Wait for completion
    for proc in processes:
        proc.wait()
    
    print("✓ Naive concurrent execution complete")
    print("⚠️  Problem: Shared CPU/memory resources cause contention\n")


def example_2_basic_isolation():
    """
    BASIC ISOLATION: Separate processes with CUDA_VISIBLE_DEVICES
    Better: Each process sees only one GPU
    Still problematic: CPU and memory contention
    """
    print("=" * 60)
    print("Example 2: Basic Process Isolation")
    print("=" * 60)
    
    for gpu_id in range(4):
        # Create a separate script for each GPU
        script_content = f"""
import subprocess
import os

# Only this GPU is visible to the process
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

# Run the benchmark
subprocess.run([
    '{sys.executable}',
    'benchmarks/run.py',
    '--benchmark', 'flash_attention_v2',
    '--output', 'results_gpu{gpu_id}_isolated.json'
])
"""
        
        # Write temporary script
        script_file = f'temp_gpu{gpu_id}.py'
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Run as completely separate process
        print(f"Starting GPU {gpu_id} in separate process")
        subprocess.run([sys.executable, script_file])
        
        # Cleanup
        os.remove(script_file)
    
    print("✓ Basic isolation complete")
    print("⚠️  Better but still has CPU contention\n")


def example_3_cpu_pinning():
    """
    CPU PINNING: Use taskset to pin each process to specific CPU cores
    This reduces CPU contention between processes
    """
    print("=" * 60)
    print("Example 3: CPU Pinning with taskset")
    print("=" * 60)
    
    # Assume 64 CPU cores, distribute evenly
    cores_per_gpu = 16
    
    processes = []
    
    for gpu_id in range(4):
        # Calculate CPU core range for this GPU
        start_core = gpu_id * cores_per_gpu
        end_core = start_core + cores_per_gpu - 1
        cpu_list = f"{start_core}-{end_core}"
        
        # Set up environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Build command with taskset
        cmd = [
            'taskset', '-c', cpu_list,  # Pin to specific CPU cores
            sys.executable,
            'benchmarks/run.py',
            '--benchmark', 'flash_attention_v2',
            '--output', f'results_gpu{gpu_id}_pinned.json'
        ]
        
        print(f"Starting GPU {gpu_id} on CPU cores {cpu_list}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
    
    # Wait for completion
    for proc in processes:
        proc.wait()
    
    print("✓ CPU-pinned execution complete")
    print("✓ Reduced CPU contention\n")


def example_4_numa_aware():
    """
    NUMA-AWARE ISOLATION: Pin to NUMA nodes for memory locality
    Best for multi-socket systems
    """
    print("=" * 60)
    print("Example 4: NUMA-Aware Isolation")
    print("=" * 60)
    
    # Example assumes 2 NUMA nodes, 2 GPUs per node
    numa_mapping = {
        0: 0,  # GPU 0 -> NUMA 0
        1: 0,  # GPU 1 -> NUMA 0
        2: 1,  # GPU 2 -> NUMA 1
        3: 1,  # GPU 3 -> NUMA 1
    }
    
    processes = []
    
    for gpu_id in range(4):
        numa_node = numa_mapping[gpu_id]
        
        # Set up environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Build command with numactl
        cmd = [
            'numactl',
            f'--cpunodebind={numa_node}',  # Bind to NUMA node CPUs
            f'--membind={numa_node}',       # Bind to NUMA node memory
            sys.executable,
            'benchmarks/run.py',
            '--benchmark', 'flash_attention_v2',
            '--output', f'results_gpu{gpu_id}_numa.json'
        ]
        
        print(f"Starting GPU {gpu_id} on NUMA node {numa_node}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
    
    # Wait for completion
    for proc in processes:
        proc.wait()
    
    print("✓ NUMA-aware execution complete")
    print("✓ Optimal memory locality\n")


def example_5_full_isolation():
    """
    FULL ISOLATION: Combine all techniques
    - CUDA_VISIBLE_DEVICES for GPU isolation
    - taskset for CPU pinning
    - numactl for NUMA binding
    - Process isolation
    """
    print("=" * 60)
    print("Example 5: Full Isolation (Recommended)")
    print("=" * 60)
    
    # Configuration
    gpu_config = {
        0: {'numa': 0, 'cpus': '0-15'},
        1: {'numa': 0, 'cpus': '16-31'},
        2: {'numa': 1, 'cpus': '32-47'},
        3: {'numa': 1, 'cpus': '48-63'},
    }
    
    processes = []
    
    for gpu_id, config in gpu_config.items():
        # Set up environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Also set thread counts to avoid oversubscription
        num_cpus = len(range(int(config['cpus'].split('-')[0]), 
                            int(config['cpus'].split('-')[1]) + 1))
        env['OMP_NUM_THREADS'] = str(num_cpus)
        env['MKL_NUM_THREADS'] = str(num_cpus)
        
        # Build command with full isolation
        cmd = [
            'taskset', '-c', config['cpus'],
            'numactl',
            f'--cpunodebind={config["numa"]}',
            f'--membind={config["numa"]}',
            sys.executable,
            'benchmarks/run.py',
            '--benchmark', 'flash_attention_v2',
            '--output', f'results_gpu{gpu_id}_full.json'
        ]
        
        print(f"Starting GPU {gpu_id}:")
        print(f"  NUMA node: {config['numa']}")
        print(f"  CPU cores: {config['cpus']}")
        print(f"  Threads: {num_cpus}")
        
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
    
    # Wait for completion
    for proc in processes:
        proc.wait()
    
    print("\n✓ Fully isolated execution complete")
    print("✓ Minimal resource contention")
    print("✓ Optimal performance for auto-tuning\n")


def example_6_using_helper_script():
    """
    USING THE HELPER SCRIPT: Easiest way with all optimizations
    """
    print("=" * 60)
    print("Example 6: Using the Isolated Runner Helper")
    print("=" * 60)
    
    # Run concurrent isolated benchmarks
    cmd = [
        sys.executable,
        'monitoring_scripts/isolated_benchmark_runner.py',
        '--benchmark', 'flash_attention_v2',
        '--mode', 'concurrent',
        '--gpus', '0,1,2,3',
        '--duration', '300'
    ]
    
    print("Running with automated isolation...")
    subprocess.run(cmd)
    
    print("\n✓ Helper script handles all isolation automatically")


def main():
    """Show progression from naive to fully isolated execution"""
    
    print("GPU BENCHMARK ISOLATION EXAMPLES")
    print("================================\n")
    
    print("These examples show different levels of process isolation")
    print("for running GPU benchmarks to avoid resource contention.\n")
    
    # Uncomment the example you want to run:
    
    # example_1_naive_concurrent()        # ❌ High contention
    # example_2_basic_isolation()         # ⚠️  Better but not optimal
    # example_3_cpu_pinning()            # ✓ Good CPU isolation
    # example_4_numa_aware()             # ✓ Good memory isolation
    example_5_full_isolation()         # ✅ Best: Full isolation
    # example_6_using_helper_script()    # ✅ Easiest: Automated
    
    print("\nRECOMMENDATION:")
    print("For accurate auto-tuning results, use Example 5 (full isolation)")
    print("or Example 6 (helper script) to minimize resource contention.")


if __name__ == '__main__':
    main()