# OpenEvolve-based Autotuner for Helion

This module implements an autotuner that uses [OpenEvolve](https://github.com/codelion/openevolve)'s evolutionary algorithm to find optimal Helion kernel configurations. It serves as an alternative to the differential evolution autotuner.

## Overview

The `OpenEvolveTuner` class uses Large Language Models (LLMs) to intelligently evolve kernel configurations, searching for the optimal parameters that maximize performance. Unlike traditional random search or grid search, OpenEvolve leverages AI to make informed decisions about which configurations to try next.

## Installation

### Prerequisites

1. **Install OpenEvolve:**
   ```bash
   pip install openevolve
   ```

2. **Set up OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   The tuner uses OpenAI's GPT models (specifically `gpt-4o-mini` by default) to evolve configurations. You'll need an OpenAI API account with credits.

### Cost Considerations

- Typical tuning cost: **$0.01 - $0.10** per tuning run
- Cost depends on:
  - Number of evaluations (`max_evaluations`)
  - Model used (gpt-4o-mini is cheapest)
  - Complexity of config space

## Quick Start

### Basic Usage

```python
from helion.autotuner.openevolve_tuner import OpenEvolveTuner

# Define configuration space
config_space = {
    'block_size': [32, 64, 128, 256, 512],
    'num_warps': [1, 2, 4, 8],
    'num_stages': [1, 2, 3, 4, 5]
}

# Define objective function (higher is better)
def evaluate_config(config):
    """
    Benchmark a kernel configuration and return throughput.
    Return 0.0 for invalid configs.
    """
    try:
        kernel = create_kernel(config)
        throughput = benchmark_kernel(kernel)
        return throughput  # GB/s or TFLOPS
    except Exception as e:
        print(f"Config failed: {e}")
        return 0.0

# Create tuner
tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=evaluate_config,
    max_evaluations=50,  # Number of configs to try
    verbose=True
)

# Run tuning
best_config = tuner.tune()
print(f"Best config: {best_config}")
```

### Complete Example

See `examples/helion_vector_add_tuning.py` for a complete working example that tunes a vector addition kernel.

```bash
# Run the example (requires GPU and torch)
python examples/helion_vector_add_tuning.py

# Run simple test without tuning
python examples/helion_vector_add_tuning.py --simple

# Run in mock mode (no GPU required)
unset OPENAI_API_KEY
python examples/helion_vector_add_tuning.py
```

## API Reference

### `OpenEvolveTuner`

```python
class OpenEvolveTuner:
    def __init__(
        self,
        config_space: Dict[str, List[Any]],
        objective: Callable[[Dict[str, Any]], float],
        max_evaluations: int = 100,
        population_size: int = 20,
        temperature: float = 0.8,
        verbose: bool = True,
    )
```

#### Parameters

- **`config_space`** (dict): Dictionary mapping parameter names to lists of valid values.
  - Keys are parameter names (e.g., `'block_size'`, `'num_warps'`)
  - Values are lists of allowed values (e.g., `[32, 64, 128, 256]`)
  - All parameters must have at least one valid value

- **`objective`** (callable): Function that evaluates a configuration.
  - Input: config dict (e.g., `{'block_size': 128, 'num_warps': 4}`)
  - Output: float score (higher is better, 0.0 or `-inf` for invalid configs)
  - Should handle errors gracefully and return 0.0 for failed configs

- **`max_evaluations`** (int, default=100): Number of configurations to evaluate.
  - More evaluations = better results but higher cost
  - Typical range: 20-100 for simple kernels, 100-200 for complex ones
  - Each evaluation costs ~$0.001-0.002 with gpt-4o-mini

- **`population_size`** (int, default=20): Population size per island in OpenEvolve.
  - Larger populations explore more diverse configurations
  - Typical range: 10-30

- **`temperature`** (float, default=0.8): LLM temperature for mutations.
  - Range: 0.0 (deterministic) to 1.0 (creative)
  - Higher values explore more aggressively
  - Lower values refine existing good configs

- **`verbose`** (bool, default=True): Whether to print progress information.

#### Methods

##### `tune() -> Dict[str, Any]`

Run the optimization and return the best configuration found.

**Returns:** Dictionary with the best configuration parameters.

**Raises:**
- `ImportError`: If OpenEvolve is not installed
- `RuntimeError`: If tuning fails or no valid configs are found

**Example:**
```python
best_config = tuner.tune()
# Returns: {'block_size': 256, 'num_warps': 4, 'num_stages': 3}
```

#### Attributes

- **`best_config`** (dict | None): Best configuration found (available after `tune()`)
- **`best_score`** (float | None): Best score achieved (available after `tune()`)
- **`evaluation_count`** (int): Number of configs evaluated
- **`history`** (list): List of `(config, score)` tuples for all evaluations

## How It Works

### 1. Initial Program Generation

The tuner generates a Python function that returns a kernel configuration:

```python
def get_kernel_config():
    config = {
        'block_size': 128,  # Valid values: [32, 64, 128, 256]
        'num_warps': 4,     # Valid values: [1, 2, 4, 8]
    }
    return config
```

### 2. Evolution Process

OpenEvolve uses an LLM to:
1. Analyze the current best configurations
2. Generate mutations (new values within the valid ranges)
3. Evaluate the new configurations
4. Keep the best-performing ones
5. Repeat for `max_evaluations` iterations

### 3. Evaluation

For each evolved configuration:
1. Extract the config dict from the evolved code
2. Validate it against the `config_space`
3. Call the `objective` function to get the score
4. Track the best configuration found so far

### 4. Result Selection

After all evaluations, the tuner returns the configuration with the highest score.

## Advanced Usage

### Custom System Message

The tuner provides a system message to guide the LLM. You can customize this by modifying the `_create_config_yaml` method.

### Fallback to Random Search

If OpenEvolve fails (e.g., API key not set, network issues), the tuner automatically falls back to random search to find a reasonable configuration.

### Logging and Debugging

Enable verbose mode to see detailed progress:

```python
tuner = OpenEvolveTuner(
    config_space=config_space,
    objective=evaluate_config,
    max_evaluations=50,
    verbose=True  # Print each evaluation
)
```

Output example:
```
Evaluation 1/50: config={'block_size': 128, 'num_warps': 4}, score=450.23
Evaluation 2/50: config={'block_size': 256, 'num_warps': 4}, score=512.45
...
```

### Error Handling

The objective function should handle errors gracefully:

```python
def evaluate_config(config):
    try:
        kernel = create_kernel(config)
        return benchmark_kernel(kernel)
    except torch.cuda.OutOfMemoryError:
        print(f"OOM with config: {config}")
        return 0.0
    except Exception as e:
        print(f"Error with config {config}: {e}")
        return 0.0
```

## Comparison with Differential Evolution

| Feature | OpenEvolveTuner | DifferentialEvolutionSearch |
|---------|----------------|----------------------------|
| **Search Strategy** | LLM-guided evolution | Genetic algorithm |
| **Intelligence** | High (uses AI reasoning) | Medium (random mutations) |
| **Cost** | ~$0.01-0.10 per run | Free |
| **Speed** | Moderate (API calls) | Fast (local computation) |
| **Requires API Key** | Yes (OpenAI) | No |
| **Best For** | Complex search spaces | Simple/medium search spaces |
| **Exploration** | Intelligent | Random with crossover |

### When to Use OpenEvolveTuner

**Use OpenEvolveTuner when:**
- You have a complex configuration space (5+ parameters)
- You want to minimize the number of evaluations
- Cost is not a primary concern ($0.01-0.10 is acceptable)
- You have an OpenAI API key

**Use DifferentialEvolutionSearch when:**
- You want a free, offline solution
- You have a simple configuration space (2-4 parameters)
- You can afford more evaluations
- You want deterministic, reproducible results

## Limitations

1. **Requires OpenAI API Key**: Must have valid OpenAI API access
2. **Cost**: Each tuning run costs money (though typically < $0.10)
3. **Network Dependency**: Requires internet connection for API calls
4. **Non-deterministic**: Results may vary between runs due to LLM sampling

## Troubleshooting

### "OPENAI_API_KEY not set"

Set your API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### "OpenEvolve is not installed"

Install OpenEvolve:
```bash
pip install openevolve
```

### "No valid configuration found"

- Check that your objective function works with at least one config
- Verify config_space has valid values
- Try increasing `max_evaluations`
- Check for CUDA errors in objective function

### High Costs

- Reduce `max_evaluations` (try 20-50 instead of 100)
- Use a cheaper model (gpt-4o-mini is already the cheapest)
- Pre-filter the config space to remove obviously bad configs

## Examples

### Example 1: Simple Vector Add

See `examples/helion_vector_add_tuning.py` for a complete example.

### Example 2: Matrix Multiplication

```python
config_space = {
    'block_size_m': [32, 64, 128, 256],
    'block_size_n': [32, 64, 128, 256],
    'block_size_k': [16, 32, 64],
    'num_warps': [2, 4, 8],
    'num_stages': [2, 3, 4, 5]
}

def evaluate_matmul_config(config):
    try:
        kernel = create_matmul_kernel(config)
        tflops = benchmark_matmul(kernel, m=1024, n=1024, k=1024)
        return tflops
    except:
        return 0.0

tuner = OpenEvolveTuner(config_space, evaluate_matmul_config, max_evaluations=100)
best_config = tuner.tune()
```

### Example 3: Attention Kernel

```python
config_space = {
    'block_m': [64, 128, 256],
    'block_n': [64, 128, 256],
    'num_warps': [4, 8, 16],
    'stages': [1, 2, 3],
    'use_tensor_cores': [True, False]
}

def evaluate_attention_config(config):
    try:
        kernel = create_attention_kernel(config)
        throughput = benchmark_attention(kernel)
        return throughput
    except:
        return 0.0

tuner = OpenEvolveTuner(config_space, evaluate_attention_config, max_evaluations=150)
best_config = tuner.tune()
```

## Contributing

To add support for new LLM providers or customize the evolution strategy, modify:
- `_create_config_yaml`: Configure OpenEvolve settings
- `_generate_initial_program`: Customize the initial configuration
- System message in `_create_config_yaml`: Guide the LLM's optimization strategy

## References

- [OpenEvolve GitHub](https://github.com/codelion/openevolve)
- [OpenEvolve Paper](https://arxiv.org/abs/2406.12832)
- [Helion Documentation](https://github.com/pytorch/helion)

## License

This code is part of the Helion project and follows the same license.
