#!/bin/bash
set -x
uv venv -p 3.12 --managed-python
. .venv/bin/activate
uv pip install --no-deps -r <(cat << EOF
arpeggio==2.0.3
asttokens==3.0.0
autopep8==2.3.2
caliper-reader==0.4.1
certifi==2025.10.5
cfgv==3.4.0
charset-normalizer==3.4.3
cmake==3.31.6
contourpy==1.3.3
cuda-bindings==12.9.2
cuda-pathfinder==1.3.0
cuda-python==12.9.2
cycler==0.12.1
decorator==5.2.1
dill==0.4.0
distlib==0.4.0
einops==0.8.1
execnet==2.1.1
executing==2.2.1
expecttest==0.3.0
filecheck==1.0.3
filelock==3.19.1
fonttools==4.60.1
fsspec==2025.9.0
hf-xet==1.1.10
huggingface-hub==0.35.3
identify==2.6.15
idna==3.10
iniconfig==2.1.0
ipdb==0.13.13
ipython==9.6.0
ipython-pygments-lexers==1.1.1
isort==6.1.0
jedi==0.19.2
jinja2==3.1.6
kiwisolver==1.4.9
lit==18.1.8
llnl-hatchet==2025.1.0
markdown-it-py==4.0.0
markupsafe==3.0.2
matplotlib==3.10.7
matplotlib-inline==0.1.7
mdurl==0.1.2
mpmath==1.3.0
multiprocess==0.70.18
networkx==3.5
ninja==1.13.0
nodeenv==1.9.1
numpy==2.3.3
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.1.0.dev0
nvidia-ml-py==13.580.82
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.24
nvidia-nvtx-cu12==12.8.90
packaging==25.0
pandas==2.3.3
parso==0.8.5
pexpect==4.9.0
pillow==11.3.0
pip==25.2
platformdirs==4.5.0
pluggy==1.6.0
pre-commit==4.3.0
prompt-toolkit==3.0.52
psutil==7.1.0
ptyprocess==0.7.0
pure-eval==0.2.3
py==1.11.0
pybind11==3.0.1
pycodestyle==2.14.0
pydot==4.0.1
pygments==2.19.2
pyparsing==3.2.5
pyright==1.1.406
pytest==8.4.2
pytest-forked==1.6.0
pytest-xdist==3.8.0
python-dateutil==2.9.0.post0
pytz==2025.2
pyyaml==6.0.3
regex==2025.9.18
requests==2.32.5
rich==14.2.0
ruff==0.14.0
safetensors==0.6.2
scipy==1.16.2
setuptools==78.1.0
six==1.17.0
stack-data==0.6.3
sympy==1.14.0
tabulate==0.9.0
textx==4.2.3
tokenizers==0.20.3
tqdm==4.67.1
traitlets==5.14.3
transformers==4.46.1
typing-extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
virtualenv==20.35.2
wcwidth==0.2.14
wheel==0.45.1
EOF
)
git clone https://github.com/facebookexperimental/triton.git
pushd triton
  git checkout 2f987ec37f7856f02b11de1c4a742975bdb77739
  make dev-install-llvm
popd
uv pip install --pre torch==2.10.0.dev20251008+cu128 torchvision==0.25.0.dev20251009+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps
git clone https://github.com/pytorch/helion.git
pushd helion
  git checkout f5ba06da5811f295d8c7373a47c7ee3c90d76a13
  uv pip install -e --no-deps .
  pushd benchmarks
    git clone https://github.com/meta-pytorch/tritonbench.git
    pushd tritonbench
      git checkout 9a4bbc7070b134fb274114018ac02b38fcfd4ba7
      uv pip install -e --no-deps .
    popd
  popd
popd
