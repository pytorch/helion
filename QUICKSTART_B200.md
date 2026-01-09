# Quick Start: Testing OpenEvolve Autotuner on B200

This guide gets you up and running with the OpenEvolve autotuner on NVIDIA B200 GPUs in under 10 minutes.

## ⚡ Fast Track (2 minutes)

```bash
# 1. Set up environment
export OPENAI_API_KEY="sk-your-api-key-here"

# 2. Run automated tests
./test_openevolve_b200.sh quick

# 3. Run a simple tuning example
python examples/helion_vector_add_tuning.py --simple
```

## 📋 Prerequisites

### Required
- ✅ NVIDIA B200 GPU (or any CUDA GPU for testing)
- ✅ Python 3.10+
- ✅ PyTorch with CUDA support
- ✅ Triton
- ✅ OpenEvolve: `pip install openevolve`

### Optional
- 🔑 OpenAI API key (for real tuning)

## 🚀 Installation

```bash
# Install OpenEvolve
pip install openevolve

# Verify installation
python -c "import openevolve; print(f'OpenEvolve {openevolve.__version__} installed')"
```

## 🧪 Run Tests

### Option 1: Automated Test Suite

```bash
# Quick tests (no GPU/API required, ~1 min)
./test_openevolve_b200.sh quick

# Full tests (with GPU, ~5-10 min)
export OPENAI_API_KEY="sk-your-key"
./test_openevolve_b200.sh full
```

### Option 2: Manual Tests

```bash
# Test 1: Verify kernel works (no tuning)
python examples/helion_vector_add_tuning.py --simple
# Expected: "✓ Vector add kernel is working correctly!"

# Test 2: Mock tuning (no API key needed)
unset OPENAI_API_KEY
python examples/helion_vector_add_tuning.py
# Expected: "Running in MOCK MODE..."

# Test 3: Real tuning (requires API key)
export OPENAI_API_KEY="sk-your-key"
python examples/helion_vector_add_tuning.py
# Expected: "Running in REAL MODE... Best performance: XXX GB/s"

# Test 4: B200 attention tuning
python examples/helion_b200_attention_tuning.py
# Expected: "B200-specific features tuned..."
```

## 📊 Example Output

### Successful Run

```
======================================================================
Helion Vector Add Kernel Tuning with OpenEvolve
======================================================================

GPU: NVIDIA B200
Mode: REAL

Configuration space:
  block_size: [32, 64, 128, 256, 512, 1024]
  num_warps: [1, 2, 4, 8]

Starting tuning with max_evaluations=50...

Evaluation 1/50: config={'block_size': 128, 'num_warps': 4}, perf=450.23 GB/s
Evaluation 2/50: config={'block_size': 256, 'num_warps': 4}, perf=512.45 GB/s
Evaluation 3/50: config={'block_size': 256, 'num_warps': 8}, perf=498.12 GB/s
...

==================================================================
TUNING COMPLETE
==================================================================
Best configuration: {'block_size': 256, 'num_warps': 4}
Best score: 512.45
Total evaluations: 50
==================================================================
```

## 🎯 Quick Examples

### Example 1: Tune Vector Add

```python
from helion.autotuner.openevolve_tuner import OpenEvolveTuner
import torch

config_space = {
    'block_size': [64, 128, 256, 512],
    'num_warps': [2, 4, 8]
}

def evaluate(config):
    # Your benchmarking code here
    kernel = create_kernel(config)
    return benchmark_throughput(kernel)

tuner = OpenEvolveTuner(config_space, evaluate, max_evaluations=30)
best = tuner.tune()
print(f"Best config: {best}")
```

### Example 2: Tune with B200 Features

```python
config_space = {
    'block_size': [128, 256],
    'num_warps': [4, 8, 16],
    'num_stages': [2, 3, 4],
    'maxreg': [128, 192, 256],  # B200-specific
    'indexing': ['default', 'tensor_descriptor'],  # B200 feature
}

tuner = OpenEvolveTuner(config_space, evaluate, max_evaluations=50)
best = tuner.tune()
```

## 💰 Cost Estimates

| Task | Evaluations | Time | Cost |
|------|-------------|------|------|
| Quick test | 10 | ~2 min | $0.01 |
| Standard tuning | 50 | ~15 min | $0.05 |
| Comprehensive | 100 | ~30 min | $0.10 |

## 🔧 Troubleshooting

### "No module named 'openevolve'"
```bash
pip install openevolve
```

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### "CUDA out of memory"
Reduce problem size:
```python
# In your evaluation function
n = 512 * 1024  # Instead of 1M
```

### "Tuning failed"
Check logs:
```bash
export HELION_AUTOTUNE_LOG_LEVEL=DEBUG
python examples/helion_vector_add_tuning.py
```

## 📈 Next Steps

1. ✅ **Run tests** - Verify everything works
2. 📖 **Read TESTING_B200.md** - Detailed testing guide
3. 🔧 **Integrate into your kernels** - Use OpenEvolveTuner in production
4. 🚀 **Optimize for B200** - Leverage Blackwell-specific features

## 🎓 Learning Resources

- **Basic Usage**: `helion/autotuner/openevolve_tuner_README.md`
- **B200 Testing**: `TESTING_B200.md`
- **Example Code**: `examples/helion_vector_add_tuning.py`
- **B200 Attention**: `examples/helion_b200_attention_tuning.py`

## 📞 Getting Help

If you encounter issues:

1. Check test output: `./test_openevolve_b200.sh quick`
2. Review logs: Set `HELION_AUTOTUNE_LOG_LEVEL=DEBUG`
3. Verify GPU: `nvidia-smi | grep B200`
4. Test API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

## ✅ Success Checklist

- [ ] B200 GPU detected
- [ ] OpenEvolve installed
- [ ] API key configured
- [ ] Test script passes: `./test_openevolve_b200.sh quick`
- [ ] Example runs: `python examples/helion_vector_add_tuning.py --simple`
- [ ] Real tuning works: `python examples/helion_vector_add_tuning.py`

Once all items are checked, you're ready to use OpenEvolve for production kernel tuning! 🎉

---

**Estimated Time**: 10 minutes to complete setup and first tuning run
**Estimated Cost**: $0.01-0.05 for initial testing
