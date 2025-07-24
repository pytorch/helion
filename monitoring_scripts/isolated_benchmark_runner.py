#!/usr/bin/env python3
"""
Isolated benchmark runner that properly isolates each GPU benchmark in its own process
with CPU affinity, NUMA binding, and proper CUDA_VISIBLE_DEVICES settings.
"""

import subprocess
import os
import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
import signal
import psutil
import argparse
from datetime import datetime

class IsolatedBenchmarkRunner:
    def __init__(self, results_dir="isolated_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Detect system topology
        self.num_gpus = self._get_gpu_count()
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.numa_nodes = self._get_numa_topology()
        
        print(f"System topology detected:")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  CPU cores: {self.cpu_count} physical, {self.cpu_count_logical} logical")
        print(f"  NUMA nodes: {len(self.numa_nodes)}")
        
    def _get_gpu_count(self):
        """Get number of available GPUs"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], 
                capture_output=True, 
                text=True,
                check=True
            )
            return len(result.stdout.strip().split('\n'))
        except:
            return 0
    
    def _get_numa_topology(self):
        """Get NUMA topology mapping"""
        numa_nodes = {}
        try:
            # Parse lscpu output for NUMA topology
            result = subprocess.run(["lscpu"], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'NUMA node' in line and 'CPU(s):' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        node_match = line.split('NUMA node')[1].split()[0]
                        node_id = int(node_match)
                        
                        # Parse CPU list (e.g., "0-15,32-47" or "0,2,4,6")
                        cpu_list = []
                        cpu_spec = parts[1].strip()
                        for cpu_range in cpu_spec.split(','):
                            if '-' in cpu_range:
                                start, end = map(int, cpu_range.split('-'))
                                cpu_list.extend(range(start, end + 1))
                            else:
                                cpu_list.append(int(cpu_range))
                        
                        numa_nodes[node_id] = cpu_list
        except Exception as e:
            print(f"Warning: Could not parse NUMA topology: {e}")
            # Fallback: assume single NUMA node
            numa_nodes[0] = list(range(self.cpu_count_logical))
        
        return numa_nodes
    
    def _get_gpu_numa_affinity(self, gpu_id):
        """Get the best NUMA node for a given GPU"""
        # Try to get GPU NUMA affinity from nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True
            )
            # This is simplified - in practice you'd parse the topology matrix
            # For now, distribute GPUs across NUMA nodes
            numa_node = gpu_id % len(self.numa_nodes)
            return numa_node
        except:
            # Simple distribution
            return gpu_id % len(self.numa_nodes)
    
    def _get_cpu_affinity_for_gpu(self, gpu_id):
        """Get CPU cores to pin for a given GPU"""
        numa_node = self._get_gpu_numa_affinity(gpu_id)
        numa_cpus = self.numa_nodes.get(numa_node, list(range(self.cpu_count_logical)))
        
        # Allocate a subset of CPUs from the NUMA node
        cpus_per_gpu = max(4, len(numa_cpus) // max(1, self.num_gpus // len(self.numa_nodes)))
        
        # Calculate offset within NUMA node
        gpus_per_numa = max(1, self.num_gpus // len(self.numa_nodes))
        offset = (gpu_id % gpus_per_numa) * cpus_per_gpu
        
        # Select CPU cores
        selected_cpus = numa_cpus[offset:offset + cpus_per_gpu]
        if not selected_cpus:  # Fallback
            selected_cpus = numa_cpus[:cpus_per_gpu]
        
        return selected_cpus, numa_node
    
    def run_benchmark_isolated(self, gpu_id, benchmark, kernel=None, config=None, 
                             duration=300, output_dir=None, extra_args=None):
        """
        Run a benchmark in complete isolation on a specific GPU.
        
        Args:
            gpu_id: GPU index to use
            benchmark: Benchmark name (e.g., 'flash_attention_v2')
            kernel: Specific kernel to run (optional)
            config: Path to config file (optional)
            duration: Benchmark duration in seconds
            output_dir: Output directory for results
            extra_args: Additional arguments for the benchmark
        """
        
        if output_dir is None:
            output_dir = self.results_dir
        
        # Get CPU affinity and NUMA node
        cpu_list, numa_node = self._get_cpu_affinity_for_gpu(gpu_id)
        cpu_list_str = ','.join(map(str, cpu_list))
        
        # Prepare environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Optional: Set additional isolation variables
        env['OMP_NUM_THREADS'] = str(len(cpu_list))
        env['MKL_NUM_THREADS'] = str(len(cpu_list))
        env['NUMEXPR_NUM_THREADS'] = str(len(cpu_list))
        
        # Disable Python multiprocessing to avoid interference
        env['PYTHONUNBUFFERED'] = '1'
        
        # Build command
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{benchmark}_gpu{gpu_id}_{timestamp}.json"
        
        cmd = []
        
        # Add taskset for CPU pinning
        cmd.extend(['taskset', '-c', cpu_list_str])
        
        # Add numactl for NUMA binding
        cmd.extend(['numactl', f'--cpunodebind={numa_node}', f'--membind={numa_node}'])
        
        # Python command
        cmd.extend([sys.executable, 'benchmarks/run.py'])
        cmd.extend(['--benchmark', benchmark])
        cmd.extend(['--output', str(output_file)])
        
        if kernel:
            cmd.extend(['--kernel', kernel])
        if config:
            cmd.extend(['--config', config])
        if duration:
            cmd.extend(['--duration', str(duration)])
        if extra_args:
            cmd.extend(extra_args)
        
        # Log the command
        print(f"\nGPU {gpu_id}: Running isolated benchmark")
        print(f"  NUMA node: {numa_node}")
        print(f"  CPU cores: {cpu_list_str}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Output: {output_file}")
        
        # Run the benchmark
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Run from helion root
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  ✓ Completed in {elapsed:.1f}s")
                
                # Add metadata to the output file
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    data['isolation_metadata'] = {
                        'gpu_id': gpu_id,
                        'numa_node': numa_node,
                        'cpu_cores': cpu_list,
                        'elapsed_time': elapsed,
                        'command': ' '.join(cmd)
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                except:
                    pass
                
                return True, output_file
            else:
                print(f"  ✗ Failed with return code {result.returncode}")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return False, None
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            return False, None
    
    def run_concurrent_isolated(self, benchmark, gpus=None, **kwargs):
        """
        Run benchmarks concurrently on multiple GPUs, each properly isolated.
        
        Args:
            benchmark: Benchmark name
            gpus: List of GPU IDs to use (default: all GPUs)
            **kwargs: Additional arguments passed to run_benchmark_isolated
        """
        
        if gpus is None:
            gpus = list(range(self.num_gpus))
        
        print(f"\nRunning concurrent isolated benchmarks on GPUs: {gpus}")
        print("=" * 60)
        
        # Create a process pool
        processes = []
        
        # Define worker function
        def worker(gpu_id):
            return self.run_benchmark_isolated(gpu_id, benchmark, **kwargs)
        
        # Start processes
        with mp.Pool(processes=len(gpus)) as pool:
            results = pool.starmap(
                self.run_benchmark_isolated,
                [(gpu_id, benchmark) for gpu_id in gpus]
            )
        
        # Summary
        print("\n" + "=" * 60)
        print("CONCURRENT RUN SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for success, _ in results if success)
        print(f"Successful runs: {successful}/{len(gpus)}")
        
        output_files = [f for success, f in results if success and f]
        return output_files
    
    def run_sequential_isolated(self, benchmark, gpus=None, cooldown=30, **kwargs):
        """
        Run benchmarks sequentially on multiple GPUs with cooldown periods.
        
        Args:
            benchmark: Benchmark name
            gpus: List of GPU IDs to use (default: all GPUs)
            cooldown: Cooldown period between runs in seconds
            **kwargs: Additional arguments passed to run_benchmark_isolated
        """
        
        if gpus is None:
            gpus = list(range(self.num_gpus))
        
        print(f"\nRunning sequential isolated benchmarks on GPUs: {gpus}")
        print(f"Cooldown between runs: {cooldown}s")
        print("=" * 60)
        
        output_files = []
        
        for i, gpu_id in enumerate(gpus):
            success, output_file = self.run_benchmark_isolated(gpu_id, benchmark, **kwargs)
            
            if success and output_file:
                output_files.append(output_file)
            
            # Cooldown period (except after last run)
            if i < len(gpus) - 1 and cooldown > 0:
                print(f"\nCooldown period ({cooldown}s)...")
                time.sleep(cooldown)
        
        # Summary
        print("\n" + "=" * 60)
        print("SEQUENTIAL RUN SUMMARY")
        print("=" * 60)
        print(f"Successful runs: {len(output_files)}/{len(gpus)}")
        
        return output_files


def main():
    """Example usage of the isolated benchmark runner"""
    
    parser = argparse.ArgumentParser(description='Run isolated GPU benchmarks')
    parser.add_argument('--benchmark', required=True, help='Benchmark to run')
    parser.add_argument('--mode', choices=['sequential', 'concurrent'], default='concurrent',
                       help='Run mode: sequential or concurrent')
    parser.add_argument('--gpus', type=str, help='Comma-separated list of GPU IDs (default: all)')
    parser.add_argument('--kernel', help='Specific kernel to benchmark')
    parser.add_argument('--duration', type=int, default=300, help='Benchmark duration in seconds')
    parser.add_argument('--cooldown', type=int, default=30, help='Cooldown between sequential runs')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse GPU list
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(',')]
    else:
        gpus = None
    
    # Create runner
    runner = IsolatedBenchmarkRunner(results_dir=args.output_dir or "isolated_results")
    
    # Run benchmarks
    if args.mode == 'sequential':
        output_files = runner.run_sequential_isolated(
            args.benchmark,
            gpus=gpus,
            kernel=args.kernel,
            duration=args.duration,
            cooldown=args.cooldown
        )
    else:
        output_files = runner.run_concurrent_isolated(
            args.benchmark,
            gpus=gpus,
            kernel=args.kernel,
            duration=args.duration
        )
    
    print(f"\nResults saved to:")
    for f in output_files:
        print(f"  - {f}")


if __name__ == '__main__':
    main()