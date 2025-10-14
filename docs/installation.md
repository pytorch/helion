# Installation

## System Requirements

Helion currently targets Linux systems and requires a recent Python and PyTorch environment:

### Operating System
- Linux-based OS
- Other Unix-like systems may work but are not officially supported

### Python Environment
- **Python 3.10–3.14**
- We recommend using [uv](https://docs.astral.sh/uv/) for lightweight, fast virtual environments

### Dependencies
- **[PyTorch](https://github.com/pytorch/pytorch) 2.9 or later**
- **[Triton](https://github.com/triton-lang/triton) 3.5 or later**

  *Note: Older versions may work, but will lack support for features like TMA on Hopper/Blackwell GPUs and may exhibit lower performance.*

## Installation Methods

### Method 1: Install via pip

The easiest way to install Helion is from PyPI:

```bash
pip install helion
```

We also publish [PyPI releases](https://pypi.org/project/helion/), but the GitHub version is recommended for the latest features and fixes.

### Method 2: Development Installation

For development purposes or if you want to modify Helion:

```bash
# Clone the repository
git clone https://github.com/pytorch/helion.git
cd helion

# Install in editable mode with development dependencies
pip install -e '.[dev]'
```

This installs Helion in "editable" mode so that changes to the source code take effect without needing to reinstall.

## Step-by-Step Setup Guide

### 1. Create and Activate a uv Virtual Environment

We recommend using uv to manage dependencies:

```bash
# Create a new virtual environment in .venv (one-time)
uv venv .venv

# Activate the environment
source .venv/bin/activate
```

### 2. Install PyTorch

Install PyTorch 2.9 or later:

```bash
# CUDA 12.8
pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128

# ROCm 7.0
pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/rocm7.0
```
see [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for other options.

### 3. Install Triton

Install Triton 3.5 or later using one of the options below.

#### Option A: Prebuilt wheel (recommended)

```bash
# Install Triton from PyPI (will pick the right wheel for your platform if available)
pip install "triton>=3.5"

# Verify
python -c "import triton; print('Triton version:', triton.__version__)"
```

If a suitable wheel is not available for your platform, build from source:

#### Option B: Build from source (Ubuntu/Debian)

```bash
# Install build dependencies (adjust versions as needed)
sudo apt-get update
sudo apt-get install -y git clang-14 clang++-14 zlib1g-dev python3-dev

# (Optional) set compilers explicitly
export CC=clang-14
export CXX=clang++-14

# Clone and install Triton
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt
MAX_JOBS=$(nproc) TRITON_PARALLEL_LINK_JOBS=2 pip install .

# Verify and clean up
python -c "import triton; print('Triton version:', triton.__version__)"
cd ..
```

### 4. Install Helion

Choose one of the installation methods above:

```bash
# From PyPI
pip install helion

# Development installation
git clone https://github.com/pytorch/helion.git
cd helion
pip install -e '.[dev]'
```

## Verification

Verify your installation by running a simple test:

```python
import torch
import helion
import helion.language as hl

@helion.kernel(autotune_effort="none")
def test_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * 2
    return out

x = torch.randn(100, device='cuda')
result = test_kernel(x)
torch.testing.assert_close(result, x * 2)
print("Verification successful!")
```

## Development Dependencies

If you installed with `[dev]`, you get additional development tools:

- **pytest** - Test runner
- **pre-commit** - Code formatting and linting hooks

Set up and use pre-commit for linting:

```bash
# Install pre-commit hooks into your local git repo (one-time)
pre-commit install

# Run all checks across the repository
pre-commit run --all-files
```

## Optional Dependencies

### Documentation Building

To build documentation locally:

```bash
pip install -e '.[docs]'
cd docs
make html
```

### GPU Requirements

Matches the requirements of [Triton](https://github.com/triton-lang/triton).  At the time of writing:
* NVIDIA GPUs (Compute Capability 8.0+)
* AMD GPUs (ROCm 6.2+)
* Via third-party forks: Intel XPU, CPU, and many other architectures

## Next Steps

Once installation is complete:

1. **Check out the {doc}`api/index` for complete API documentation**
2. **Explore the [examples/](https://github.com/pytorch/helion/tree/main/examples) folder for real-world patterns**
