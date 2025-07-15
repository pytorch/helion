#!/usr/bin/env python
import sys
import subprocess

# Run the benchmark with specific parameters and capture output
cmd = [
    sys.executable,
    "benchmarks/run.py",
    "--kernel", "jagged_softmax",
    "--B", "32",
    "--M", "8", 
    "--seqlen", "64",
    "--sparsity", "0.5",
    "--metrics", "speedup,accuracy"
]

print(f"Running command: {' '.join(cmd)}")

# Run with a short timeout to see where it hangs
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired as e:
    print("Command timed out after 10 seconds")
    print("STDOUT so far:")
    print(e.stdout.decode() if e.stdout else "")
    print("\nSTDERR so far:")
    print(e.stderr.decode() if e.stderr else "")