#!/usr/bin/env python3
"""
Capture autotune min/mid/max from first two generations only.
Kills the process after seeing Generation 2 data to save time.
"""

import subprocess
import re
import time
import os
import sys
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import json
try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. CPU affinity detection will be limited.")
    psutil = None

class AutotuneCapture:
    def __init__(self):
        self.initial_pop = None
        self.pattern = re.compile(
            r'(?:Initial population|Initial generation):.*?'
            r'min=([\d.]+)\s+mid=([\d.]+)\s+max=([\d.]+)'
        )
        
    def parse_line(self, line):
        """Extract initial generation data from a line."""
        match = self.pattern.search(line)
        if match:
            data = {
                'min': float(match.group(1)),
                'mid': float(match.group(2)),
                'max': float(match.group(3))
            }
            
            if 'Initial population' in line or 'Initial generation' in line:
                self.initial_pop = data
                return 'initial'
        return None

def monitor_process_output(proc, gpu_id, capture):
    """Monitor process output and kill after initial generation."""
    print(f"[GPU {gpu_id}] Monitoring autotune output...")
    
    for line in proc.stdout:
        line = line.strip()
        if line:
            # Check for generation data
            result = capture.parse_line(line)
            
            if result == 'initial':
                print(f"[GPU {gpu_id}] Initial generation: min={capture.initial_pop['min']:.4f} "
                      f"mid={capture.initial_pop['mid']:.4f} max={capture.initial_pop['max']:.4f}")
                
                # Kill the process after initial generation
                print(f"[GPU {gpu_id}] Got initial generation data, terminating...")
                proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()
                break
    
    return capture

def get_numa_cpu_affinity(gpu_id, total_gpus=8):
    """Get NUMA node and CPU cores for a GPU."""
    import psutil
    
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # Simple heuristic: distribute GPUs across available CPUs
    cpus_per_gpu = max(4, cpu_count_logical // total_gpus)
    start_cpu = gpu_id * cpus_per_gpu
    end_cpu = min(start_cpu + cpus_per_gpu - 1, cpu_count_logical - 1)
    
    # Assume 2 NUMA nodes for simplicity
    numa_node = gpu_id // (total_gpus // 2)
    
    return numa_node, f"{start_cpu}-{end_cpu}"

def run_autotune_capture(gpu_id, kernel='gemm', log_file=None, use_isolation=True):
    """Run benchmark and capture initial generation of autotune data."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Enable Helion autotune logging
    env['HELION_AUTOTUNE_LOG_LEVEL'] = '10'  # DEBUG level
    
    # Set thread limits to prevent oversubscription
    numa_node, cpu_list = get_numa_cpu_affinity(gpu_id)
    num_cpus = len(range(int(cpu_list.split('-')[0]), int(cpu_list.split('-')[1]) + 1))
    env['OMP_NUM_THREADS'] = str(num_cpus)
    env['MKL_NUM_THREADS'] = str(num_cpus)
    
    cmd = []
    
    # Add isolation commands if requested
    if use_isolation:
        # CPU pinning with taskset
        cmd.extend(['taskset', '-c', cpu_list])
        
        # NUMA binding with numactl
        cmd.extend(['numactl', f'--cpunodebind={numa_node}', f'--membind={numa_node}'])
    
    # Python command
    cmd.extend([
        sys.executable,
        'benchmarks/run.py',
        '--kernel', kernel,
        '--num-inputs', '1'
    ])
    
    if use_isolation:
        print(f"[GPU {gpu_id}] Starting autotune capture with isolation:")
        print(f"  NUMA node: {numa_node}, CPU cores: {cpu_list}")
    else:
        print(f"[GPU {gpu_id}] Starting autotune capture...")
    
    capture = AutotuneCapture()
    
    # Start process
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Monitor output
    try:
        monitor_process_output(proc, gpu_id, capture)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        proc.kill()
    
    # Save to log file if specified
    if log_file and capture.initial_pop:
        with open(log_file, 'w') as f:
            f.write(f"GPU {gpu_id} Autotune Results\n")
            f.write("="*40 + "\n")
            f.write(f"Initial: min={capture.initial_pop['min']:.4f} "
                   f"mid={capture.initial_pop['mid']:.4f} "
                   f"max={capture.initial_pop['max']:.4f}\n")
    
    return capture

def compare_single_vs_concurrent(kernel='gemm', num_gpus=4, use_isolation=True):
    """Compare autotune results between single GPU and concurrent execution."""
    results_dir = Path('autotune_comparison')
    results_dir.mkdir(exist_ok=True)
    
    print("AUTOTUNE INITIAL GENERATION COMPARISON")
    print("="*60)
    print(f"Kernel: {kernel}")
    print(f"Capturing: Initial population/generation only")
    
    # Step 1: Single GPU baseline
    print(f"\nStep 1: Single GPU baseline (GPU 0)")
    print("-"*40)
    
    baseline_log = results_dir / 'baseline_gpu0.txt'
    baseline = run_autotune_capture(0, kernel, baseline_log, use_isolation=False)  # Single GPU doesn't need isolation
    
    if not baseline.initial_pop:
        print("ERROR: Failed to capture initial generation for baseline")
        return
    
    # Cool down
    print("\nCooling down for 30s...")
    time.sleep(30)
    
    # Step 2: Concurrent execution
    isolation_msg = "with process isolation" if use_isolation else "without isolation"
    print(f"\nStep 2: Concurrent execution on {num_gpus} GPUs {isolation_msg}")
    print("-"*40)
    
    concurrent_results = {}
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(
                run_autotune_capture, 
                gpu_id, 
                kernel, 
                results_dir / f'concurrent_gpu{gpu_id}.txt',
                use_isolation
            ): gpu_id
            for gpu_id in range(num_gpus)
        }
        
        for future in futures:
            gpu_id = futures[future]
            try:
                result = future.result()
                concurrent_results[gpu_id] = result
            except Exception as e:
                print(f"[GPU {gpu_id}] Failed: {e}")
    
    # Cool down
    print("\nCooling down for 30s...")
    time.sleep(30)

    # Analysis
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print("\nBaseline (Single GPU):")
    print(f"  Initial: min={baseline.initial_pop['min']:.4f} "
          f"mid={baseline.initial_pop['mid']:.4f} "
          f"max={baseline.initial_pop['max']:.4f}")
    
    print("\nConcurrent GPUs:")
    
    # Collect all concurrent min times
    concurrent_mins = []
    for gpu_id in sorted(concurrent_results.keys()):
        result = concurrent_results[gpu_id]
        if result.initial_pop:
            concurrent_mins.append(result.initial_pop['min'])
            print(f"  GPU {gpu_id} Initial: min={result.initial_pop['min']:.4f} "
                  f"mid={result.initial_pop['mid']:.4f} "
                  f"max={result.initial_pop['max']:.4f}")
    
    if concurrent_mins:
        # Compare min times
        baseline_min = baseline.initial_pop['min']
        avg_concurrent_min = sum(concurrent_mins) / len(concurrent_mins)
        
        degradation = ((avg_concurrent_min - baseline_min) / baseline_min) * 100
        
        print(f"\nInitial Generation Min Time Comparison:")
        print(f"  Baseline:        {baseline_min:.4f}")
        print(f"  Concurrent avg:  {avg_concurrent_min:.4f}")
        print(f"  Degradation:     {degradation:+.1f}%")
        
        if degradation > 5:
            print(f"\n⚠️  SIGNIFICANT CONTENTION DETECTED!")
            print(f"   Concurrent autotuning shows {degradation:.1f}% worse min times")
            print(f"   This indicates resource contention is affecting autotune quality")
        elif degradation > 2:
            print(f"\n⚡ MODERATE CONTENTION ({degradation:.1f}% degradation)")
        else:
            print(f"\n✅ MINIMAL CONTENTION ({degradation:.1f}% degradation)")
    
    # Save summary
    summary = {
        'kernel': kernel,
        'baseline': {
            'initial': baseline.initial_pop
        },
        'concurrent': {
            f'gpu_{gpu_id}': {
                'initial': result.initial_pop
            }
            for gpu_id, result in concurrent_results.items()
            if result.initial_pop
        }
    }
    
    import json
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to {results_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Capture autotune generations for contention analysis')
    parser.add_argument('--kernel', default='gemm', help='Kernel to test')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs for concurrent test')
    parser.add_argument('--no-isolation', action='store_true', help='Disable process isolation for concurrent runs')
    
    args = parser.parse_args()
    
    compare_single_vs_concurrent(args.kernel, args.num_gpus, use_isolation=not args.no_isolation)

if __name__ == '__main__':
    main()
