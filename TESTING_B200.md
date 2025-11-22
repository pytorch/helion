# Testing OpenEvolve Autotuner on B200 GPUs

This guide provides instructions for testing the OpenEvolve autotuner on NVIDIA B200 (Blackwell) machines.

## Prerequisites

### 1. Environment Setup

```bash
# Ensure you're on a machine with B200 GPU
nvidia-smi

# Install required dependencies
pip install openevolve torch triton

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: Set Helion environment variables
export HELION_AUTOTUNE_LOG_LEVEL=INFO
export HELION_AUTOTUNE_PROGRESS_BAR=1
```

### 2. Verify GPU Detection

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_properties(0).name}')"
# Expected output: GPU: NVIDIA B200
```

## Quick Tests

### Test 1: Simple Vector Add (Baseline)

This test verifies the basic structure without full tuning:

```bash
cd /home/user/helion
python examples/helion_vector_add_tuning.py --simple
```

**Expected output:**
```
✓ Vector add kernel is working correctly!
```

### Test 2: Mock Tuning (No GPU/API Required)

Test the tuner logic without GPU or OpenAI API:

```bash
unset OPENAI_API_KEY
python examples/helion_vector_add_tuning.py
```

**Expected output:**
```
Running in MOCK MODE (no GPU evaluation)
...
Best configuration found: {'block_size': ..., 'num_warps': ...}
```

### Test 3: Real Tuning on B200

Run actual GPU benchmarking with OpenEvolve:

```bash
export OPENAI_API_KEY="sk-your-key"
python examples/helion_vector_add_tuning.py
```

**Expected output:**
```
Running in REAL MODE (GPU evaluation)
Starting OpenEvolve optimization...
Evaluation 1/50: config={'block_size': 128, 'num_warps': 4}, perf=450.23 GB/s
...
Best configuration: {'block_size': 256, 'num_warps': 4}
Best performance: 512.45 GB/s
```

## Advanced Tests

### Test 4: B200-Specific Attention Kernel

I've created a B200-optimized attention kernel tuning example:

```bash
python examples/helion_b200_attention_tuning.py
```

This will tune Blackwell-specific parameters like:
- Tensor descriptor indexing
- Warp specialization
- Register allocation (maxRegAutoWS)
- Multi-buffering strategies

### Test 5: Custom Kernel Tuning

Create your own tuning script:

```python
from helion.autotuner.openevolve_tuner import OpenEvolveTuner
import torch
import helion

# Define B200-specific config space
config_space = {
    'block_size_m': [64, 128, 256],
    'block_size_n': [64, 128, 256],
    'num_warps': [4, 8, 16],
    'num_stages': [2, 3, 4, 5],
    'maxreg': [128, 152, 192, 256],
}

def evaluate_config(config):
    # Your kernel evaluation here
    kernel = create_kernel(config)
    throughput = benchmark(kernel)
    return throughput

tuner = OpenEvolveTuner(config_space, evaluate_config, max_evaluations=100)
best_config = tuner.tune()
```

## Performance Expectations

### B200 Vector Add Baseline

| Configuration | Expected Throughput |
|--------------|---------------------|
| block_size=128, num_warps=4 | ~450-500 GB/s |
| block_size=256, num_warps=4 | ~500-550 GB/s |
| Optimal (tuned) | ~550-600 GB/s |

### Tuning Time Estimates

| Evaluations | Wall Time | Cost |
|------------|-----------|------|
| 20 | ~5-10 min | $0.01-0.02 |
| 50 | ~10-20 min | $0.03-0.05 |
| 100 | ~20-40 min | $0.05-0.10 |

*Times include GPU benchmarking + OpenAI API calls*

## Troubleshooting on B200

### Issue: "CUDA out of memory"

B200 has different memory than older GPUs. If you hit OOM:

```python
# Reduce problem size in evaluation
def evaluate_config(config):
    n = 512 * 1024  # Reduced from 1M
    x = torch.randn(n, device='cuda')
    ...
```

### Issue: "Triton compilation error"

Some configs may not work on Blackwell:

```python
def evaluate_config(config):
    try:
        kernel = create_kernel(config)
        return benchmark(kernel)
    except Exception as e:
        print(f"Config failed: {e}")
        return 0.0  # Let tuner try other configs
```

### Issue: Slow benchmarking

B200 kernels compile faster but may have different warmup needs:

```python
from triton.testing import do_bench

# Adjust warmup/rep for B200
time_ms = do_bench(
    lambda: kernel(x, y),
    warmup=50,  # More warmup for B200
    rep=100
)
```

## Monitoring During Tuning

### Terminal 1: Run tuning
```bash
python examples/helion_vector_add_tuning.py
```

### Terminal 2: Monitor GPU
```bash
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization should be high (>80%)
- Memory usage should be stable
- Temperature within limits

### Terminal 3: Monitor costs
```bash
# Check OpenAI API usage
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Collecting Results

### Save tuning history

```python
tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=evaluate_config,
    max_evaluations=100,
    verbose=True  # Prints progress
)

best_config = tuner.tune()

# Save results
import json
results = {
    'best_config': best_config,
    'best_score': tuner.best_score,
    'history': [(c, s) for c, s in tuner.history],
    'gpu': torch.cuda.get_device_properties(0).name
}

with open('b200_tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Visualize results

```python
import matplotlib.pyplot as plt

configs, scores = zip(*tuner.history)
plt.plot(scores)
plt.xlabel('Evaluation')
plt.ylabel('Throughput (GB/s)')
plt.title('OpenEvolve Tuning Progress on B200')
plt.savefig('b200_tuning_progress.png')
```

## B200-Specific Optimizations

### Tensor Descriptors

B200 has enhanced tensor descriptor support:

```python
config_space = {
    'indexing': ['default', 'tensor_descriptor'],
    'block_size': [128, 256],
    ...
}
```

### Warp Specialization

Blackwell benefits from warp specialization:

```python
config_space = {
    'range_warp_specializes': [[None, None], [True, None], [None, True]],
    ...
}
```

### Register Tuning

B200 has more registers available:

```python
config_space = {
    'maxreg': [128, 152, 192, 224, 256],
    ...
}
```

## Benchmarking Tips

### 1. Use Real Workload Sizes

```python
# Production-like sizes for B200
batch_size = 16
seq_len = 8192
hidden = 4096
```

### 2. Include Compilation Time

```python
def evaluate_config(config):
    kernel = create_kernel(config)
    # First call includes compilation
    _ = kernel(x, y)
    torch.cuda.synchronize()
    # Now benchmark
    time_ms = do_bench(lambda: kernel(x, y))
    return throughput
```

### 3. Test Multiple Input Sizes

```python
def evaluate_config(config):
    scores = []
    for size in [1024, 4096, 16384]:
        x = torch.randn(size, device='cuda')
        score = benchmark_with_size(kernel, x)
        scores.append(score)
    return sum(scores) / len(scores)  # Average performance
```

## Example: Full B200 Test Session

```bash
# 1. Verify environment
nvidia-smi | grep B200
echo $OPENAI_API_KEY | head -c 10
python -c "import openevolve; print('OpenEvolve OK')"

# 2. Run quick test
python examples/helion_vector_add_tuning.py --simple

# 3. Run small tuning (20 evals, ~5min, $0.01)
python -c "
from helion.autotuner.openevolve_tuner import OpenEvolveTuner
import torch

config_space = {
    'block_size': [64, 128, 256],
    'num_warps': [2, 4, 8]
}

def mock_eval(c):
    return 100.0 * (c['block_size'] / 128) * (c['num_warps'] / 4)

tuner = OpenEvolveTuner(config_space, mock_eval, max_evaluations=20)
best = tuner.tune()
print(f'Best config: {best}')
"

# 4. Run full tuning (100 evals, ~30min, $0.10)
python examples/helion_vector_add_tuning.py

# 5. Collect results
ls -lh b200_tuning_results.json
```

## Expected Cost Breakdown

For a typical B200 tuning session:

| Item | Cost |
|------|------|
| OpenAI API (gpt-4o-mini) | $0.05-0.10 |
| GPU time (B200 rental) | $2-5/hour |
| Total (1 hour session) | ~$2.05-5.10 |

**Tip:** Start with 20-50 evaluations to test, then scale up if needed.

## Next Steps

1. **Start small:** Run `--simple` test first
2. **Test mock mode:** Verify structure without API/GPU
3. **Small tuning run:** 20 evaluations to test full pipeline
4. **Full tuning:** 50-100 evaluations for real optimization
5. **Compare with baseline:** Measure improvement vs default configs
6. **Production:** Use best config in your actual kernels

## Support

If you encounter issues:

1. Check logs: Set `HELION_AUTOTUNE_LOG_LEVEL=DEBUG`
2. Verify GPU: `nvidia-smi` should show B200
3. Test API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`
4. Check examples: Look at working Helion examples in `examples/`

## Reporting Results

When sharing results, include:

```python
import torch
print(f"GPU: {torch.cuda.get_device_properties(0).name}")
print(f"Helion version: {helion.__version__}")
print(f"Best config: {tuner.best_config}")
print(f"Best score: {tuner.best_score}")
print(f"Total evaluations: {tuner.evaluation_count}")
```

Good luck with testing! 🚀
