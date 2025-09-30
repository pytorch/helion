#!/bin/bash
set -eux

# GitHub log group functions
start_group() {
    echo "::group::$1"
}

end_group() {
    echo "::endgroup::"
}

# Parameters
PYTHON_VERSION="$1"
RUNTIME_VERSION="$2"
NUM_SHARDS="$3"
SHARD="$4"

# Setup environment
export PATH="$HOME/.local/bin:$PATH"

start_group "Install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
end_group

start_group "Create virtual environment"
uv venv --python "$PYTHON_VERSION"
end_group

start_group "Install PyTorch"
source .venv/bin/activate
uv pip install -U --pre torch --index-url "https://download.pytorch.org/whl/nightly/$RUNTIME_VERSION"
end_group

start_group "Install Triton"
set -x
source .venv/bin/activate
apt-get update
apt-get install -y git
apt-get install -y clang-14 clang++-14 zlib1g-dev
export CC=clang-14
export CXX=clang++-14
mkdir -p /tmp/$USER
cd /tmp/$USER
uv pip uninstall triton pytorch-triton || true
rm -rf triton/ || true
git clone https://github.com/triton-lang/triton.git
cd triton/
uv pip install -r python/requirements.txt
MAX_JOBS=$(nproc) TRITON_PARALLEL_LINK_JOBS=2 uv pip install .
cd /tmp/$USER
rm -rf triton/
python -c "import triton; print(f'Triton version: {triton.__version__}')"
end_group

start_group "Install Helion"
source .venv/bin/activate
uv pip install -r requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" uv pip install -e .'[dev]' --no-deps
python -c "import helion; print(helion.__name__)"
end_group

start_group "Install Benchmark Requirements"
set -x
source .venv/bin/activate
uv pip install pip
uv pip install quack-kernels --no-deps
mkdir -p benchmarks/ && pushd benchmarks/
git clone https://github.com/pytorch-labs/tritonbench/
pushd tritonbench/
git submodule update --init --recursive
uv pip install -r requirements.txt
python install.py --liger
uv pip install -e . --no-deps
popd
popd
end_group

start_group "Run Benchmark"
set -eux

rm -rf /tmp/torchinductor_*/ || true

source .venv/bin/activate

KERNELS=("softmax" "geglu" "swiglu" "jsd" "welford" "kl_div" "int4_gemm" "layer_norm" "layer_norm-bwd" "rms_norm" "rms_norm-bwd" "cross_entropy")
NUMSHARDS="$NUM_SHARDS"
SHARD="$SHARD"

SHARD_KERNELS=()
for ((i=0; i<${#KERNELS[@]}; i++)); do
  if [ $((i % NUMSHARDS)) -eq $SHARD ]; then
    SHARD_KERNELS+=("${KERNELS[i]}")
  fi
done

KERNEL_LIST=$(IFS=','; echo "${SHARD_KERNELS[*]}")
echo "Running shard $SHARD of $NUMSHARDS with kernels: $KERNEL_LIST"

TEST_REPORTS_DIR=$(pwd)/test/test-reports
mkdir -p "$TEST_REPORTS_DIR"
echo "$TEST_REPORTS_DIR"

for kernel in "${SHARD_KERNELS[@]}"; do
  echo "=========================================="
  echo "Running benchmark for kernel: $kernel"
  echo "=========================================="

  # Get available implementations and baseline for this kernel
  KERNEL_INFO=$(python benchmarks/run.py --list-impls-for-benchmark-ci --op "$kernel" | grep "^$kernel:")
  IMPLS=$(echo "$KERNEL_INFO" | sed -n 's/.*impls=\([^ ]*\).*/\1/p')
  BASELINE=$(echo "$KERNEL_INFO" | sed -n 's/.*baseline=\([^ ]*\).*/\1/p')

  if [[ -z "$IMPLS" ]]; then
    echo "Warning: No implementations found for kernel $kernel, skipping..."
    continue
  fi
  if [[ -z "$BASELINE" ]]; then
    echo "Warning: No baseline found for kernel $kernel, skipping..."
    continue
  fi
  echo "Using baseline: $BASELINE"
  echo "Available implementations for $kernel: $IMPLS"

  # Do autotuning but do not record the results
  python benchmarks/run.py \
      --op "$kernel" \
      --metrics speedup,accuracy \
      --latency-measure-mode triton_do_bench \
      --cudagraph \
      --only "$IMPLS" \
      --only-match-mode prefix-with-baseline \
      --baseline "$BASELINE" \
      --exit-on-exception

  # Relax the GPU
  sleep 2m

  # Run again with cache and record results
  python benchmarks/run.py \
      --op "$kernel" \
      --metrics speedup,accuracy \
      --latency-measure-mode triton_do_bench \
      --cudagraph \
      --only "$IMPLS" \
      --only-match-mode prefix-with-baseline \
      --baseline "$BASELINE" \
      --output "$TEST_REPORTS_DIR/helionbench.json" \
      --append-to-output \
      --exit-on-exception

  echo "✅ Completed benchmark for kernel: $kernel"
done

if [[ ! -s "$TEST_REPORTS_DIR/helionbench.json" ]]; then
  echo "❌ helionbench.json is missing or empty"
  exit 1
fi
cat "$TEST_REPORTS_DIR/helionbench.json"
end_group
