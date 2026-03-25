#!/bin/bash
set -euxo pipefail
cd ..
uv venv -p 3.12 --managed-python
rm -r ../dist || true
mkdir ../dist
uvx -p 3.12 pip download --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --no-deps -d ../dist
. .venv/bin/activate
uv pip install ../dist/torch-*.whl
TORCH_TPU_COMMIT=$(cat .github/ci_commit_pins/torch_tpu.txt)
# git clone git@github.com:google-ml-infra/torch_tpu.git ../torch_tpu
cd ../torch_tpu
git checkout "${TORCH_TPU_COMMIT}"
export TORCH_SOURCE=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
bazel build -c opt //ci/wheel:torch_tpu_wheel --define WHEEL_VERSION=0.1.0 --define TORCH_SOURCE=local --action_env=PYTHONPATH=$TORCH_SOURCE:$SITE_PACKAGES --action_env=JAX_PLATFORMS=cpu
uv pip install bazel-bin/ci/wheel/*.whl
cp bazel-bin/ci/wheel/*.whl ../dist
