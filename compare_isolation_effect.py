#!/usr/bin/env python3
"""
Compare autotune results with and without process isolation.
This helps verify that isolation reduces resource contention.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_comparison(kernel='gemm', num_gpus=4):
    """Run the autotune capture test with and without isolation."""
    
    results_dir = Path('isolation_comparison')
    results_dir.mkdir(exist_ok=True)
    
    print("PROCESS ISOLATION COMPARISON")
    print("="*70)
    print(f"Kernel: {kernel}")
    print(f"GPUs: {num_gpus}")
    print("This test compares concurrent GPU execution with and without isolation")
    print("="*70)
    
    # Test 1: Without isolation
    print("\nTest 1: Concurrent execution WITHOUT process isolation")
    print("-"*50)
    
    cmd1 = [
        sys.executable,
        'capture_autotune_generations.py',
        '--kernel', kernel,
        '--num-gpus', str(num_gpus),
        '--no-isolation'
    ]
    
    print("Running without isolation...")
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    # Save output
    with open(results_dir / 'without_isolation.txt', 'w') as f:
        f.write(result1.stdout)
        if result1.stderr:
            f.write("\nSTDERR:\n")
            f.write(result1.stderr)
    
    # Extract degradation percentage from output
    degradation_no_isolation = None
    for line in result1.stdout.split('\n'):
        if 'Degradation:' in line and '%' in line:
            try:
                degradation_no_isolation = float(line.split()[1].rstrip('%'))
                break
            except:
                pass
    
    print("Completed.")
    
    # Cool down
    print("\nCooling down for 60 seconds...")
    time.sleep(60)
    
    # Test 2: With isolation
    print("\nTest 2: Concurrent execution WITH process isolation")
    print("-"*50)
    
    cmd2 = [
        sys.executable,
        'capture_autotune_generations.py',
        '--kernel', kernel,
        '--num-gpus', str(num_gpus)
        # No --no-isolation flag, so isolation is enabled
    ]
    
    print("Running with isolation (taskset + numactl)...")
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    # Save output
    with open(results_dir / 'with_isolation.txt', 'w') as f:
        f.write(result2.stdout)
        if result2.stderr:
            f.write("\nSTDERR:\n")
            f.write(result2.stderr)
    
    # Extract degradation percentage from output
    degradation_with_isolation = None
    for line in result2.stdout.split('\n'):
        if 'Degradation:' in line and '%' in line:
            try:
                degradation_with_isolation = float(line.split()[1].rstrip('%'))
                break
            except:
                pass
    
    print("Completed.")
    
    # Summary
    print("\n" + "="*70)
    print("ISOLATION EFFECT SUMMARY")
    print("="*70)
    
    if degradation_no_isolation is not None and degradation_with_isolation is not None:
        print(f"\nPerformance degradation due to concurrent execution:")
        print(f"  Without isolation: {degradation_no_isolation:+.1f}%")
        print(f"  With isolation:    {degradation_with_isolation:+.1f}%")
        
        improvement = degradation_no_isolation - degradation_with_isolation
        print(f"\nIsolation improvement: {improvement:.1f} percentage points")
        
        if improvement > 2:
            print("\n✅ Process isolation significantly reduces contention!")
            print("   Recommendation: Always use isolation for multi-GPU runs")
        elif improvement > 0:
            print("\n⚡ Process isolation provides some benefit")
        else:
            print("\n🤔 Process isolation shows minimal benefit in this case")
            print("   (GPUs may already be well-isolated on this system)")
    else:
        print("\nERROR: Could not extract degradation percentages from output")
        print(f"Check the output files in {results_dir}/")
    
    # Save summary
    summary = {
        'kernel': kernel,
        'num_gpus': num_gpus,
        'degradation_no_isolation': degradation_no_isolation,
        'degradation_with_isolation': degradation_with_isolation,
        'improvement': degradation_no_isolation - degradation_with_isolation if (degradation_no_isolation is not None and degradation_with_isolation is not None) else None
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to {results_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare isolation effect on autotune')
    parser.add_argument('--kernel', default='gemm', help='Kernel to test')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs')
    
    args = parser.parse_args()
    
    # Check for required tools
    import shutil
    if not shutil.which('taskset'):
        print("ERROR: taskset not found. Install with: sudo apt-get install util-linux")
        sys.exit(1)
    if not shutil.which('numactl'):
        print("ERROR: numactl not found. Install with: sudo apt-get install numactl")
        sys.exit(1)
    
    run_comparison(args.kernel, args.num_gpus)

if __name__ == '__main__':
    main()