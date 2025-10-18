#!/usr/bin/env bash

# Usage examples:
# 1) Run the 1st kernel shard on GPU 0 of an 8-GPU machine:
#    PYTHON_VERSION=3.12 RUNTIME_VERSION=cu130 CUDA_VISIBLE_DEVICES=0 KERNEL_SHARD=1/8 bash run_benchmarks.sh >output.log 2>&1
# 2) On a 1-GPU machine, run the 2nd shard:
#    PYTHON_VERSION=3.12 RUNTIME_VERSION=cu130 CUDA_VISIBLE_DEVICES=0 KERNEL_SHARD=2/8 bash run_benchmarks.sh >output.log 2>&1

set -euo pipefail

STEP_REQUEST=${1:-__all__}
DEFAULT_KERNELS="softmax,jsd,welford,kl_div,int4_gemm,layer_norm,layer_norm-bwd,rms_norm,rms_norm-bwd,cross_entropy"
PINNED_HELION_COMMIT="d7e69f99d381b1dc9c229e445c5dfd852312cf7b"

require_env() {
  local missing=0
  for var in "$@"; do
    if [[ -z ${!var:-} ]]; then
      echo "Missing required environment variable: $var" >&2
      missing=1
    fi
  done
  if ((missing)); then
    exit 1
  fi
}

# Optional environment prefix (e.g. CUDA_VISIBLE_DEVICES=0) and custom arguments.
# shellcheck disable=SC2206
env_prefix=(${ENV_VARS:-})
# shellcheck disable=SC2206
custom_args=(${CUSTOM_ARGS:-})

activate_venv() {
  if [[ ! -f .venv/bin/activate ]]; then
    echo "Virtual environment not found at .venv/bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

run_with_optional_env() {
  if ((${#env_prefix[@]})); then
    env "${env_prefix[@]}" "$@"
  else
    "$@"
  fi
}

lock_sm_clocks() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not available; skipping SM clock lock." >&2
    return 0
  fi

  local cuda_devices=${CUDA_VISIBLE_DEVICES:-}
  if [[ -z "${cuda_devices}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set when locking SM clocks." >&2
    exit 1
  fi

  IFS=',' read -r -a cuda_indices <<< "${cuda_devices}"
  if (( ${#cuda_indices[@]} == 0 )); then
    echo "No devices specified in CUDA_VISIBLE_DEVICES." >&2
    exit 1
  fi

  local idx=${cuda_indices[0]}
  idx=$(echo "${idx}" | xargs)
  if [[ -z "${idx}" ]]; then
    echo "Unable to parse first CUDA_VISIBLE_DEVICES entry." >&2
    exit 1
  fi

  local gpu_info
  gpu_info=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep "^${idx},")
  if [[ -z "${gpu_info}" ]]; then
    echo "GPU index ${idx} not found." >&2
    exit 1
  fi

  local model
  model=$(echo "${gpu_info}" | awk -F',' '{print $2}' | xargs | awk '{print $2}')

  echo "Locking SM clocks for GPU ${idx} (${model})"
  sudo nvidia-smi -pm 1 -i "${idx}"
  sudo nvidia-smi -lgc 1620,1620 -i "${idx}"

  local desired_power
  case ${model} in
    H100)
      desired_power=500
      ;;
    GB200)
      desired_power=1200
      ;;
    B200)
      desired_power=750
      ;;
    *)
      desired_power=500
      ;;
  esac

  sudo nvidia-smi --power-limit="${desired_power}" -i "${idx}"

  echo "NOTE: Use 'nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,clocks.video --format=csv' to verify clock settings."
}

unlock_sm_clocks() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not available; skipping SM clock unlock." >&2
    return 0
  fi

  local cuda_devices=${CUDA_VISIBLE_DEVICES:-}
  if [[ -z "${cuda_devices}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set when unlocking SM clocks." >&2
    exit 1
  fi

  IFS=',' read -r -a cuda_indices <<< "${cuda_devices}"
  if (( ${#cuda_indices[@]} == 0 )); then
    echo "No devices specified in CUDA_VISIBLE_DEVICES." >&2
    exit 1
  fi

  local idx=${cuda_indices[0]}
  idx=$(echo "${idx}" | xargs)
  if [[ -z "${idx}" ]]; then
    echo "Unable to parse first CUDA_VISIBLE_DEVICES entry." >&2
    exit 1
  fi

  local gpu_info
  gpu_info=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep "^${idx},")
  if [[ -z "${gpu_info}" ]]; then
    echo "GPU index ${idx} not found." >&2
    exit 1
  fi

  local model
  model=$(echo "${gpu_info}" | awk -F',' '{print $2}' | xargs | awk '{print $2}')

  echo "Unlocking SM clocks for GPU ${idx} (${model})"
  sudo nvidia-smi -rgc -i "${idx}"

  local power_cap
  case ${model} in
    H100)
      power_cap=650
      ;;
    GB200)
      power_cap=1200
      ;;
    B200)
      power_cap=750
      ;;
    *)
      power_cap=500
      ;;
  esac

  sudo nvidia-smi --power-limit="${power_cap}" -i "${idx}"
}

setup_toolchain() {
  set -x
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y git curl ca-certificates
    else
      apt-get update
      apt-get install -y git curl ca-certificates
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo dnf install -y git curl ca-certificates
    else
      dnf install -y git curl ca-certificates
    fi
  elif command -v yum >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo yum install -y git curl ca-certificates
    else
      yum install -y git curl ca-certificates
    fi
  else
    set +x
    echo "No supported package manager found. Please install git manually." >&2
    exit 1
  fi
  set +x
}

ensure_uv_available() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found, installing via official installer"
  local installer_url="https://astral.sh/uv/install.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "${installer_url}" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${installer_url}" | sh
  else
    echo "Neither curl nor wget available to download uv installer." >&2
    exit 1
  fi

  export PATH="${HOME}/.local/bin:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation failed; uv still not on PATH." >&2
    exit 1
  fi
}

create_virtualenv() {
  require_env PYTHON_VERSION
  ensure_uv_available
  if [[ ! -d .venv ]]; then
    uv venv --python "${PYTHON_VERSION}"
  fi
  activate_venv
}

install_pytorch() {
  require_env RUNTIME_VERSION
  ensure_uv_available
  activate_venv
  uv pip install -U "torch==2.9.*" --index-url "https://download.pytorch.org/whl/${RUNTIME_VERSION}"
}

install_helion() {
  ensure_uv_available
  activate_venv
  if [[ ! -f pyproject.toml || ! -d helion ]]; then
    if [[ ! -d helion ]]; then
      git clone https://github.com/pytorch/helion helion
    fi
    pushd helion >/dev/null
    git fetch origin "${PINNED_HELION_COMMIT}"
    git checkout "${PINNED_HELION_COMMIT}"
    uv pip install -r requirements.txt
    SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" uv pip install -e .'[dev]' --no-deps
    python -c "import helion; print(helion.__name__)"
    popd >/dev/null
  else
    uv pip install -r requirements.txt
    SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" uv pip install -e .'[dev]' --no-deps
    python -c "import helion; print(helion.__name__)"
  fi
}

install_benchmark_requirements() {
  set -x
  ensure_uv_available
  activate_venv
  uv pip install pip
  uv pip install quack-kernels --no-deps
  mkdir -p benchmarks/
  pushd benchmarks/ >/dev/null
  if [[ ! -d tritonbench ]]; then
    git clone https://github.com/pytorch-labs/tritonbench/
  fi
  pushd tritonbench/ >/dev/null
  git submodule update --init --recursive
  uv pip install -r requirements.txt
  python install.py --liger
  uv pip install -e . --no-deps
  popd >/dev/null
  popd >/dev/null
  set +x
}

run_benchmarks() {
  local kernels_input=${KERNELS:-${DEFAULT_KERNELS}}
  local selected_kernels
  selected_kernels=$(shard_kernels "${kernels_input}")

  if [[ -z "${selected_kernels}" ]]; then
    echo "❌ No kernels selected to run" >&2
    exit 1
  fi

  ensure_uv_available
  echo "Running kernels: ${selected_kernels}"

  activate_venv

  local test_reports_dir
  test_reports_dir=$(pwd)/test/test-reports
  mkdir -p "${test_reports_dir}"
  echo "${test_reports_dir}"

  IFS=',' read -r -a kernels <<< "${selected_kernels}"
  for kernel in "${kernels[@]}"; do
    kernel=$(echo "${kernel}" | xargs)
    if [[ -z "${kernel}" ]]; then
      continue
    fi

    echo "=========================================="
    echo "Running benchmark for kernel: ${kernel}"
    echo "=========================================="

    local kernel_info impls baseline
    kernel_info=$(python benchmarks/run.py --list-impls-for-benchmark-ci --op "${kernel}" | grep "^${kernel}:")
    impls=$(echo "${kernel_info}" | sed -n 's/.*impls=\([^ ]*\).*/\1/p')
    baseline=$(echo "${kernel_info}" | sed -n 's/.*baseline=\([^ ]*\).*/\1/p')

    if [[ -z "${impls}" ]]; then
      echo "Warning: No implementations found for kernel ${kernel}, skipping..."
      continue
    fi
    if [[ -z "${baseline}" ]]; then
      echo "Warning: No baseline found for kernel ${kernel}, skipping..."
      continue
    fi

    echo "Using baseline: ${baseline}"
    echo "Available implementations for ${kernel}: ${impls}"

    local -a base_args=(
      "benchmarks/run.py"
      "--op" "${kernel}"
      "--metrics" "speedup,accuracy"
      "--latency-measure-mode" "triton_do_bench"
      "--cudagraph"
      "--only" "${impls}"
      "--only-match-mode" "prefix-with-baseline"
      "--baseline" "${baseline}"
      "--atol" "1e-2"
      "--rtol" "1e-2"
      "--input-sample-mode" "equally-spaced-k"
      "--keep-going"
    )

    local -a first_run_args=("${base_args[@]}")

    local timestamp
    timestamp=$(date -u +%s)
    local sanitized_kernel
    sanitized_kernel=$(echo "${kernel}" | sed 's/[^A-Za-z0-9._-]/_/g')
    local output_file="${test_reports_dir}/${sanitized_kernel}_${timestamp}_helionbench.json"

    local -a second_run_args=("${base_args[@]}" "--output" "${output_file}")

    first_run_args+=("${custom_args[@]}")
    second_run_args+=("${custom_args[@]}")

    if ((${#env_prefix[@]})); then
      env "${env_prefix[@]}" HELION_FORCE_AUTOTUNE=1 python "${first_run_args[@]}"
    else
      HELION_FORCE_AUTOTUNE=1 python "${first_run_args[@]}"
    fi

    echo "Sleeping 120 seconds before the second run to allow GPU to cooldown..."
    sleep 120

    run_with_optional_env python "${second_run_args[@]}"

    echo "✅ Completed benchmark for kernel: ${kernel}"
  done

  shopt -s nullglob
  local result_files=("${test_reports_dir}"/*_helionbench.json)
  shopt -u nullglob

  if (( ${#result_files[@]} == 0 )); then
    echo "❌ No benchmark result files were generated"
    exit 1
  fi

  for result in "${result_files[@]}"; do
    if [[ ! -s "${result}" ]]; then
      echo "❌ Result file ${result} is empty"
      exit 1
    fi
    echo "--- Contents of ${result} ---"
    cat "${result}"
  done
}

shard_kernels() {
  local kernels=$1
  local shard=${KERNEL_SHARD:-}

  if [[ -z "${kernels}" ]]; then
    echo ""
    return
  fi

  if [[ -z "${shard}" ]]; then
    echo "${kernels}"
    return
  fi

  if [[ ! ${shard} =~ ^([0-9]+)/([0-9]+)$ ]]; then
    echo "Invalid KERNEL_SHARD format '${shard}', expected K/N" >&2
    exit 1
  fi

  local k=${BASH_REMATCH[1]}
  local n=${BASH_REMATCH[2]}

  if (( n <= 0 )); then
    echo "Shard denominator must be positive" >&2
    exit 1
  fi
  if (( k < 1 || k > n )); then
    echo "Shard index ${k} out of range for ${n} shard(s)" >&2
    exit 1
  fi

  IFS=',' read -r -a kernel_array <<< "${kernels}"
  local total=${#kernel_array[@]}

  if (( total == 0 )); then
    echo ""
    return
  fi

  local base=$(( total / n ))
  local remainder=$(( total % n ))

  local start count
  if (( k <= remainder )); then
    count=$(( base + 1 ))
    start=$(( (k - 1) * (base + 1) ))
  else
    count=${base}
    start=$(( remainder * (base + 1) + (k - 1 - remainder) * base ))
  fi

  if (( count <= 0 || start >= total )); then
    echo ""
    return
  fi

  local end=$(( start + count ))
  if (( end > total )); then
    end=${total}
  fi

  local -a selected=()
  local idx
  for ((idx = start; idx < end; idx++)); do
    local item
    item=$(echo "${kernel_array[idx]}" | xargs)
    if [[ -n "${item}" ]]; then
      selected+=("${item}")
    fi
  done

  if (( ${#selected[@]} == 0 )); then
    echo ""
    return
  fi

  local joined
  printf -v joined "%s," "${selected[@]}"
  joined=${joined%,}
  echo "${joined}"
}

run_steps_in_order() {
  local step
  for step in "Lock SM clocks" "Setup toolchain" "Create virtual environment" "Install PyTorch" "Install Helion" "Install Benchmark Requirements" "Run Benchmark" "Unlock SM clocks"; do
    execute_step "${step}"
  done
}

execute_step() {
  local step=$1
  echo "==> Running step: ${step}"
  case ${step} in
    "Lock SM clocks")
      lock_sm_clocks
      ;;
    "Setup toolchain")
      setup_toolchain
      ;;
    "Create virtual environment")
      create_virtualenv
      ;;
    "Install PyTorch")
      install_pytorch
      ;;
    "Install Helion")
      install_helion
      ;;
    "Install Benchmark Requirements")
      install_benchmark_requirements
      ;;
    "Run Benchmark")
      run_benchmarks
      ;;
    "Unlock SM clocks")
      unlock_sm_clocks
      ;;
    *)
      echo "Unknown step: ${step}" >&2
      exit 1
      ;;
  esac
}

main() {
  if [[ ${STEP_REQUEST} == "__all__" ]]; then
    run_steps_in_order
  else
    execute_step "${STEP_REQUEST}"
  fi
}

main
