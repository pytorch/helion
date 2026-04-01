#!/bin/bash
set -euxo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$SCRIPT_DIR/.build-wheels"
DIST_DIR="$REPO_DIR/dist"

mkdir -p "$BUILD_DIR" "$DIST_DIR"

BAZELISK="$BUILD_DIR/bazelisk"
if [ ! -x "$BAZELISK" ]; then
    curl -fSL -o "$BAZELISK" "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
    chmod +x "$BAZELISK"
fi

pushd "$BUILD_DIR"

if [ ! -d .venv ]; then
    uv venv -p 3.12 --managed-python
fi

if ! ls "$DIST_DIR"/torch-*.whl 1>/dev/null 2>&1; then
    uvx -p 3.12 pip download --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --no-deps -d "$DIST_DIR"
fi

. .venv/bin/activate
uv pip install "$DIST_DIR"/torch-*.whl

TORCH_TPU_COMMIT=$(cat "$REPO_DIR/.github/ci_commit_pins/torch_tpu.txt")
if [ ! -d torch_tpu ]; then
    git clone git@github.com:google-pytorch/torch_tpu.git torch_tpu
fi
pushd torch_tpu
git fetch origin
git checkout "${TORCH_TPU_COMMIT}"

export TORCH_SOURCE=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
"$BAZELISK" build -c opt //ci/wheel:torch_tpu_wheel --define WHEEL_VERSION=0.1.0 --define TORCH_SOURCE=local --action_env=PYTHONPATH=$TORCH_SOURCE:$SITE_PACKAGES --action_env=JAX_PLATFORMS=cpu
uv pip install bazel-bin/ci/wheel/*.whl
cp bazel-bin/ci/wheel/*.whl "$DIST_DIR"
popd
popd
