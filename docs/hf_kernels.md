# Shipping Kernels with Hugging Face Kernels

This page shows how to package a Helion kernel and publish it on the Hugging Face Hub using [Kernels](https://huggingface.co/docs/kernels/). Once a kernel is published, anyone can load it with:

```python
# pip install -U kernels

from kernels import get_kernel

kernel = get_kernel("namespace/helion_kernel", version=1)
```

## How it works

A Helion kernel is in plain Python. It compiles itself on the user’s machine first time it runs, so there is nothing to compile *ahead of time*. We simply ship the source. [`kernel-builder`](https://huggingface.co/docs/kernels/main/en/builder/writing-kernels) packages the source and uploads it to the Hugging Face Hub. This in turn enables [`kernels`](https://huggingface.co/docs/kernels/main/en/index) to download it and load it for the user.

In the guide, we showcase the end-to-end process using a vector addition kernel. This is a very simple kernel, chosen specifically so that the kernel itself stays out of the way and the focus stays on the packaging and the distribution workflow.

## Prerequisites

A CUDA GPU. This guide was tested on an NVIDIA L4.

A Python environment with `torch`, `helion`, and `kernels` installed.

```sh
uv pip install torch --torch-backend "auto"
uv pip install kernels helion
```

A Hugging Face account with write access to the namespace you will push to.

```sh
uvx hf auth login
```

## 1. Create the project

`kernel-builder` is the CLI that scaffolds, builds, and uploads kernel repositories. One can install it with the handy one-liner:

```bash
curl -fsSL <https://raw.githubusercontent.com/huggingface/kernels/main/install.sh> | bash
```

Once `kernel-builder` is installed, we can build the kernel project. `kernle-builder` builds a template for us, a good starting point for anyone who is willing to build kernels.

```bash
kernel-builder init --name ariG23498/vector-add-helion --backends cuda -- vector-add-helion
cd vector-add-helion
```

One thing to note is the `--name ariG23498/vector-add-helion` flag. You should have either your Hugging Face organization's name, or your personal account name inplace of `ariG23498`.

The template is set up for a compiled kernels. It contains:

- CUDA source directories
- a C++ Torch binding (`torch-ext/torch_binding.{cpp,h}`)
- a Python package in `torch-ext/vector_add_helion/`
- `tests/`, `benchmarks/`, `example.py`, `CARD.md`, and a `flake.nix`

Helion kernels are plain Python that is JIT-compiled to Triton (and a bunch of other backends) at runtime, so none of the compiled pieces are needed. Remove them:

```bash
rm -rf vector_add_helion_cuda torch-ext/torch_binding.{cpp,h}
```

## 2. Configure `build.toml`

Edit `build.toml` so it describes a pure-Python (`noarch`) kernel that depends on `helion`:

```toml
[general]
name = "vector-add-helion"
license = "Apache-2.0"
backends = [
  "cuda",
]
version = 1
edition = 6

python-depends = ["helion"]

[general.hub]
repo-id = "ariG23498/vector-add-helion"

[torch-noarch]
```

The three settings that matter:

- `python-depends = ["helion"]` declares Helion as a runtime dependency of the kernel. When someone loads the kernel, `kernels` uses this to check that `helion` is available.
- `[torch-noarch]` marks the kernel as architecture-independent. The build produces a single `torch-cuda` variant that contains only Python files, instead of one binary per Torch/CUDA combination.
- `repo-id = "ariG23498/vector-add-helion"` here you should edit the name and point to your Hugging Face personal account or organisation.

## 3. Write the kernel and a layer

The Python package lives in `torch-ext/vector_add_helion/`. We put two things in it:

1. The kernel itself, as a function you can call directly.
2. A layer (`nn.Module`) that [`kernelize`](https://huggingface.co/docs/kernels/main/en/api/layers#kernels.kernelize) can use to replace the `forward` of a layer in someone else's model.

### Layers

In the `torch-ext/vector_add_helion/layers.py` we add:

```py
import torch
import torch.nn as nn

import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[1024], num_warps=4))
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out

class VectorAdd(nn.Module):
    """Drop-in `forward` replacement for layers annotated with
    `@use_kernel_forward_from_hub("VectorAdd")`.

    `kernelize` only borrows `forward` from this class, so it must stay
    stateless: no `__init__` and no extra members besides the two flags below.
    """

    # The Helion kernel has no autograd formula, so do not use it for training.
    has_backward = False
    can_torch_compile = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The Helion kernel is 1D, so flatten N-D inputs (e.g. residual
        # streams of shape [batch, seq, hidden]) and restore the shape after.
        out = vector_add(x.contiguous().view(-1), y.contiguous().view(-1))
        return out.view_as(x)
```

Here `hl.tile(x.size(0))` splits the vector into tiles, and each tile becomes one program instance. Passing an explicit `helion.Config` fixes the block size and warp count. Without it, Helion would *autotune* on the first call, which takes a long time.

### Init

In the `torch-ext/vector_add_helion/__init__.py` file we add:

```python
from . import layers
from .layers import vector_add

__all__ = ["vector_add", "layers"]
```

Both exports matter:

- **`vector_add`** is what users call with `kernel.vector_add(...)`.
- **`layers`** is where `kernelize` looks for layers. It resolves `kernel.layers.<layer_name>`. If the package does not import `layers`, `kernelize` fails with *"does not define any layers"*.

## 4. Build locally

Now that everything is ready, we will build the kernels locally.

```sh
kernel-builder check-config .
kernel-builder build-and-copy .
```

Here [`check-config`](https://huggingface.co/docs/kernels/main/en/builder-cli#kernel-builder-check-config) validates `build.toml`, while [`build-and-copy`](https://huggingface.co/docs/kernels/main/en/builder-cli#kernel-builder-build-and-copy) builds the kernel and copies the result into `build/`.

> [!NOTE]
> The build runs through a Nix flake, and Nix only sees files that git tracks. If you create `layers.py` and forget to `git add` it, the build succeeds but `layers.py` is silently missing from `build/torch-cuda/`. Run `git add` on new files before building. You don't have to commit them first.
> If you build with uncommitted changes, `kernels` prints a warning at load time that the kernel *"was built from a dirty git tree"*. Builds you publish should come from a clean commit.

For a noarch kernel, the output is small:

```txt
build/
├── CARD.md
├── benchmarks/
└── torch-cuda/
    ├── __init__.py
    ├── _ops.py
    ├── layers.py
    ├── metadata.json
    └── symbols.json
```

You can load the local build directly to check it before publishing:

```py
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(Path("build"))
```

## 5. Upload to the Hub

Log in Hugging Face, then [`build-and-upload`](https://huggingface.co/docs/kernels/main/en/builder-cli#kernel-builder-build-and-upload) in one step:

```bash
uvx hf auth login
kernel-builder build-and-upload .
```

This pushes the kernel to the `repo-id` from `build.toml` under the version given by `version`, and renders `CARD.md` into the repo's model card.

| ![Kernel Card](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/kernels/image.png) |
| :--: |
| The kernel card after publishing to the Hub |

## 6. Use the kernel from the Hub

After we have successfully publised the kernel on the Hugging Face Hub, we can use it with the `kernels` library.

### As a function

```py
import torch
from kernels import get_kernel

kernel = get_kernel("ariG23498/vector-add-helion", version=1, trust_remote_code=True)

x = torch.arange(10, dtype=torch.float16, device="cuda")
y = torch.arange(10, dtype=torch.float16, device="cuda")

z = kernel.vector_add(x, y)
print(z)
```

Expected output:

```py
tensor([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.], device='cuda:0',
       dtype=torch.float16)
```

`trust_remote_code=True` is required because the kernel is published under a personal namespace, not a trusted publisher organisation. Loading a kernel runs its Python code, so only pass this flag for repositories you trust.

### Inside a PyTorch model with `kernelize`

Calling `kernel.vector_add` directly means editing the model code. `kernelize` avoids that where the model author marks a layer as *extensible*, and kernels from the Hub are mapped onto it afterwards. The model's own PyTorch implementation stays as the reference and fallback.

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import (
    LayerRepository,
    Mode,
    get_kernelized_layers,
    kernelize,
    use_kernel_forward_from_hub,
    use_kernel_mapping,
)

# 1. Annotate the layer. Its forward is the reference implementation
#    and the fallback when no kernel is available.
@use_kernel_forward_from_hub("VectorAdd")
class VectorAdd(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

# 2. Build the registry mapping: layer name -> device -> kernel repository.
kernel_layer_mapping = {
    "VectorAdd": {
        "cuda": LayerRepository(
            repo_id="ariG23498/vector-add-helion",
            layer_name="VectorAdd",
            version=1,
            trust_remote_code=True,
        ),
    }
}

# 3. Build the model. The residual connection uses the extensible layer.
class ResidualMLP(nn.Module):
    def __init__(self, d_model=1024, d_hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.add = VectorAdd()

    def forward(self, x):
        return self.add(x, self.fc2(F.gelu(self.fc1(x))))

torch.manual_seed(0)
x = torch.randn(8, 128, 1024, device="cuda")

model = ResidualMLP().to("cuda").eval()
with torch.no_grad():
    ref_out = model(x)

# 4. Kernelize the model.
with use_kernel_mapping(kernel_layer_mapping):
    kernelized_model = kernelize(
        ResidualMLP().to("cuda").eval(), device="cuda", mode=Mode.INFERENCE
    )
kernelized_model.load_state_dict(model.state_dict())

print(get_kernelized_layers(kernelized_model).layers)

with torch.no_grad():
    hub_out = kernelized_model(x)

print("Hub kernel output matches original:", torch.allclose(hub_out, ref_out, atol=1e-5))
```

To know more about `kernelize` we recommend you reading the [official documentation](https://huggingface.co/docs/kernels/main/en/layers) and interacting with the [kernelize cookbook](https://colab.research.google.com/github/huggingface/kernels/blob/main/examples/cookbook/use_layers.ipynb).

## Further reading

- [Kernels documentation](https://huggingface.co/docs/kernels/en/index)
- [`kernel-builder` CLI reference](https://huggingface.co/docs/kernels/en/builder-cli)
- [`build.toml` reference](https://huggingface.co/docs/kernels/en/builder/writing-kernels#buildtoml)
- [Blog: Helion x HF Kernels](https://pytorch.org/blog/helion-x-%F0%9F%A4%97-hf-kernels-building-and-shipping-out-of-the-box-performant-kernels/)
