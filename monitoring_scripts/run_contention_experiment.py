#!/usr/bin/env python3
"""
Complete experiment runner for testing GPU resource contention in Helion auto-tuning.
This script orchestrates the entire experiment with proper process isolation.
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import signal
import argparse

class ContentionExperiment:
    def __init__(self, results_dir="contention_experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.results_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.baseline_dir = self.experiment_dir / "baseline"
        self.concurrent_dir = self.experiment_dir / "concurrent"
        self.isolated_dir = self.experiment_dir / "isolated"
        self.monitoring_dir = self.experiment_dir / "monitoring"
        
        for d in [self.baseline_dir, self.concurrent_dir, self.isolated_dir, self.monitoring_dir]:
            d.mkdir(exist_ok=True)
        
        # Paths to monitoring scripts
        self.script_dir = Path(__file__).parent
        self.gpu_monitor_script = self.script_dir / "gpu_monitor.py"
        self.system_monitor_script = self.script_dir / "system_monitor.py"
        self.isolated_runner_script = self.script_dir / "isolated_benchmark_runner.py"
        
    def run_monitoring(self, name_suffix, duration=None):
        """Start monitoring processes and return their PIDs"""
        pids = []
        
        # GPU monitoring
        gpu_monitor_file = self.monitoring_dir / f"gpu_monitor_{name_suffix}.json"
        gpu_cmd = [sys.executable, str(self.gpu_monitor_script), str(gpu_monitor_file)]
        if duration:
            gpu_cmd.extend(['--duration', str(duration)])
        
        gpu_proc = subprocess.Popen(gpu_cmd)
        pids.append(gpu_proc.pid)
        
        # System monitoring
        sys_monitor_file = self.monitoring_dir / f"system_monitor_{name_suffix}.json"
        sys_cmd = [sys.executable, str(self.system_monitor_script), str(sys_monitor_file)]
        if duration:
            sys_cmd.extend(['--duration', str(duration)])
        
        sys_proc = subprocess.Popen(sys_cmd)
        pids.append(sys_proc.pid)
        
        print(f"Started monitoring processes: GPU (PID {gpu_proc.pid}), System (PID {sys_proc.pid})")
        
        # Give monitors time to initialize
        time.sleep(2)
        
        return pids
    
    def stop_monitoring(self, pids):
        """Stop monitoring processes"""
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        
        # Wait a bit for processes to terminate
        time.sleep(2)
        
        # Force kill if still running
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    
    def run_baseline_benchmark(self, gpu_id, benchmark, kernel=None, duration=300):
        """Run a single baseline benchmark with full isolation"""
        
        print(f"\n{'='*60}")
        print(f"BASELINE: GPU {gpu_id}, Benchmark: {benchmark}")
        print(f"{'='*60}")
        
        # Start monitoring
        monitor_pids = self.run_monitoring(f"baseline_gpu{gpu_id}_{benchmark}", duration + 30)
        
        try:
            # Prepare environment - only this GPU visible
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # Output file
            output_file = self.baseline_dir / f"{benchmark}_gpu{gpu_id}.json"
            
            # Build command
            cmd = [
                sys.executable,
                'benchmarks/run.py',
                '--benchmark', benchmark,
                '--output', str(output_file),
                '--duration', str(duration)
            ]
            
            if kernel:
                cmd.extend(['--kernel', kernel])
            
            print(f"Running: {' '.join(cmd)}")
            print(f"CUDA_VISIBLE_DEVICES={gpu_id}")
            
            # Run benchmark
            start_time = time.time()
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)  # Run from helion root
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ Completed in {elapsed:.1f}s")
                return True
            else:
                print(f"✗ Failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return False
                
        finally:
            # Stop monitoring
            self.stop_monitoring(monitor_pids)
            
            # Cooldown
            print("Cooldown period (30s)...")
            time.sleep(30)
    
    def run_concurrent_naive(self, benchmark, gpus, kernel=None, duration=300):
        """Run benchmarks concurrently without proper isolation (naive approach)"""
        
        print(f"\n{'='*60}")
        print(f"CONCURRENT (NAIVE): Running on GPUs {gpus}")
        print(f"{'='*60}")
        
        # Start monitoring
        monitor_pids = self.run_monitoring("concurrent_naive", duration + 30)
        
        try:
            processes = []
            
            for gpu_id in gpus:
                # Prepare environment
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                # Output file
                output_file = self.concurrent_dir / f"{benchmark}_gpu{gpu_id}_naive.json"
                
                # Build command
                cmd = [
                    sys.executable,
                    'benchmarks/run.py',
                    '--benchmark', benchmark,
                    '--output', str(output_file),
                    '--duration', str(duration)
                ]
                
                if kernel:
                    cmd.extend(['--kernel', kernel])
                
                print(f"Starting GPU {gpu_id}: CUDA_VISIBLE_DEVICES={gpu_id}")
                
                # Start process
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=str(Path(__file__).parent.parent)
                )
                processes.append((gpu_id, proc))
            
            # Wait for all to complete
            print("\nWaiting for all processes to complete...")
            for gpu_id, proc in processes:
                proc.wait()
                print(f"GPU {gpu_id}: {'✓' if proc.returncode == 0 else '✗'}")
                
        finally:
            # Stop monitoring
            self.stop_monitoring(monitor_pids)
            
            # Cooldown
            print("Cooldown period (30s)...")
            time.sleep(30)
    
    def run_concurrent_isolated(self, benchmark, gpus, kernel=None, duration=300):
        """Run benchmarks concurrently with proper isolation"""
        
        print(f"\n{'='*60}")
        print(f"CONCURRENT (ISOLATED): Running on GPUs {gpus}")
        print(f"{'='*60}")
        
        # Start monitoring
        monitor_pids = self.run_monitoring("concurrent_isolated", duration + 30)
        
        try:
            # Use the isolated runner
            cmd = [
                sys.executable,
                str(self.isolated_runner_script),
                '--benchmark', benchmark,
                '--mode', 'concurrent',
                '--gpus', ','.join(map(str, gpus)),
                '--duration', str(duration),
                '--output-dir', str(self.isolated_dir)
            ]
            
            if kernel:
                cmd.extend(['--kernel', kernel])
            
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )
            
            if result.returncode == 0:
                print("✓ All isolated benchmarks completed")
            else:
                print(f"✗ Some benchmarks failed")
                print(f"stderr: {result.stderr}")
                
        finally:
            # Stop monitoring
            self.stop_monitoring(monitor_pids)
            
            # Cooldown
            print("Cooldown period (30s)...")
            time.sleep(30)
    
    def analyze_results(self):
        """Run variance analysis on the collected results"""
        
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        
        # Collect result files
        baseline_files = list(self.baseline_dir.glob("*.json"))
        concurrent_naive_files = [f for f in self.concurrent_dir.glob("*.json") if 'naive' in f.name]
        concurrent_isolated_files = list(self.isolated_dir.glob("*.json"))
        
        # Run variance analysis
        analyze_script = self.script_dir / "analyze_variance.py"
        
        # Analysis 1: Baseline vs Concurrent Naive
        if baseline_files and concurrent_naive_files:
            print("\n1. Baseline vs Concurrent (Naive):")
            cmd = [
                sys.executable,
                str(analyze_script),
                ','.join(map(str, baseline_files)),
                ','.join(map(str, concurrent_naive_files))
            ]
            
            subprocess.run(cmd)
        
        # Analysis 2: Baseline vs Concurrent Isolated
        if baseline_files and concurrent_isolated_files:
            print("\n2. Baseline vs Concurrent (Isolated):")
            cmd = [
                sys.executable,
                str(analyze_script),
                ','.join(map(str, baseline_files)),
                ','.join(map(str, concurrent_isolated_files))
            ]
            
            subprocess.run(cmd)
        
        # Save experiment summary
        summary = {
            'experiment_timestamp': self.timestamp,
            'experiment_dir': str(self.experiment_dir),
            'baseline_runs': len(baseline_files),
            'concurrent_naive_runs': len(concurrent_naive_files),
            'concurrent_isolated_runs': len(concurrent_isolated_files),
            'files': {
                'baseline': [f.name for f in baseline_files],
                'concurrent_naive': [f.name for f in concurrent_naive_files],
                'concurrent_isolated': [f.name for f in concurrent_isolated_files]
            }
        }
        
        with open(self.experiment_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment results saved to: {self.experiment_dir}")
    
    def run_full_experiment(self, benchmark='flash_attention_v2', gpus=None, kernel=None, duration=300):
        """Run the complete contention experiment"""
        
        if gpus is None:
            # Detect available GPUs
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-L"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                num_gpus = len(result.stdout.strip().split('\n'))
                gpus = list(range(num_gpus))
            except:
                print("Error: Could not detect GPUs")
                return
        
        print(f"Running contention experiment")
        print(f"Benchmark: {benchmark}")
        print(f"GPUs: {gpus}")
        print(f"Duration: {duration}s per run")
        
        # Phase 1: Baseline (sequential isolated runs)
        print(f"\n{'#'*60}")
        print("PHASE 1: BASELINE MEASUREMENTS")
        print(f"{'#'*60}")
        
        for gpu_id in gpus:
            self.run_baseline_benchmark(gpu_id, benchmark, kernel, duration)
        
        # Phase 2: Concurrent naive (without isolation)
        print(f"\n{'#'*60}")
        print("PHASE 2: CONCURRENT EXECUTION (NAIVE)")
        print(f"{'#'*60}")
        
        self.run_concurrent_naive(benchmark, gpus, kernel, duration)
        
        # Phase 3: Concurrent with isolation
        print(f"\n{'#'*60}")
        print("PHASE 3: CONCURRENT EXECUTION (ISOLATED)")
        print(f"{'#'*60}")
        
        self.run_concurrent_isolated(benchmark, gpus, kernel, duration)
        
        # Analysis
        print(f"\n{'#'*60}")
        print("PHASE 4: ANALYSIS")
        print(f"{'#'*60}")
        
        self.analyze_results()


def main():
    parser = argparse.ArgumentParser(
        description='Run GPU resource contention experiment for Helion auto-tuning'
    )
    parser.add_argument('--benchmark', default='rms_norm',
                       help='Benchmark to run (default: rms_norm)')
    parser.add_argument('--gpus', type=str,
                       help='Comma-separated list of GPU IDs (default: all)')
    parser.add_argument('--kernel', help='Specific kernel to benchmark')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration per benchmark run in seconds (default: 300)')
    parser.add_argument('--output-dir', default='contention_experiment_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse GPU list
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(',')]
    else:
        gpus = None
    
    # Run experiment
    experiment = ContentionExperiment(results_dir=args.output_dir)
    experiment.run_full_experiment(
        benchmark=args.benchmark,
        gpus=gpus,
        kernel=args.kernel,
        duration=args.duration
    )


if __name__ == '__main__':
    main()