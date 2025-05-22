# Helion

**Helion** is a Python-embedded domain-specific language (DSL) for
authoring machine learning kernels, designed to compile down to [Triton],
a performant backend for programming GPUs and other devices. Helion aims
to raise the level of abstraction compared to Triton, making it easier
to write correct and efficient kernels while enabling more automation
in the autotuning process.

The name *Helion* refers to the nucleus of a helium-3 atom, while *Triton*
refers to hydrogen-3.

[Triton]: https://github.com/triton-lang/triton

> ⚠️ **Early Development Warning**
> Helion is currently in an experimental stage. You should expect bugs, incomplete features, and APIs that may change in future versions. Feedback and bug reports are welcome and appreciated!

## Example

A minimal matrix multiplication kernel in Helion looks like this:

```python
import torch, helion, helion.language as hl

@helion.kernel()
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out
```

The code outside the `for` loops is standard PyTorch code executed on
the CPU. It is typically used for tasks like allocating output tensors
and performing shape computations.

The code inside the `for` loops is compiled into a Triton kernel,
resulting in a single GPU kernel.  A single Helion kernel is always
compiled to exactly one GPU kernel.

The `hl.tile` function subdivides the iteration space (in this case `m` by
`n`) into tiles. These tiles are executed in parallel on the GPU. Tiling
details, such as dimensionality (1D vs 2D), tile sizes, and loop ordering,
are automatically determined by Helion's autotuner. Alternatively, these
details can be explicitly specified using the `config=` argument in
`helion.kernel`.

* The outer `for` loop is mapped onto the grid of the generated
kernel. The grid size is determined automatically based on the chosen
tile size.

* The inner `for` loop translates into a loop within the generated kernel,
and its tile size is also determined automatically.

Within a Helion kernel, standard PyTorch operators (like
`torch.addmm`) are automatically mapped to Triton operations using
[TorchInductor](https://github.com/pytorch/pytorch/tree/main/torch/_inductor).
Thus, familiarity with PyTorch means you already know most of
Helion. Helion supports a wide range of operations including pointwise
(`add`, `sigmoid`, etc.), reductions (`sum`, `softmax`, etc.), views,
and matrix multiplication operations.  Arbitrary function calls
within a Helion kernel are supported, but must be traceable with
[make_fx](https://pytorch.org/docs/stable/generated/torch.fx.experimental.proxy_tensor.make_fx.html).

## Autotuning

The above example can be executed with:

```python
out = matmul(torch.randn([2048, 2028], device="cuda"),
             torch.randn([2048, 2028], device="cuda"))
```

When a kernel runs for the first time, Helion initiates autotuning. A
typical autotuning session produces output similar to:

```
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
[20s] Initial population: failed=10 min=0.9677s mid=3.0013s max=22.1430s best=Config(block_sizes=[[64, 32], [32]], loop_orders=[[1, 0]], num_warps=2, num_stages=2, indexing='pointer', l2_grouping=1, use_yz_grid=False)
[52s] Generation 2: replaced=16 min=0.7731s mid=1.7203s max=3.1227s best=Config(block_sizes=[[32, 128], [16]], loop_orders=[[0, 1]], num_warps=4, num_stages=4, indexing='block_ptr', l2_grouping=16)
[85s] Generation 3: replaced=19 min=0.6256s mid=1.3916s max=2.7868s best=Config(block_sizes=[[64, 128], [16]], loop_orders=[[0, 1]], num_warps=4, num_stages=4, indexing='block_ptr', l2_grouping=16)
...
[593s] Generation 19: replaced=7 min=0.6072s mid=0.6626s max=0.7496s best=Config(block_sizes=[[64, 128], [16]], loop_orders=[[1, 0]], num_warps=4, num_stages=3, indexing='block_ptr', l2_grouping=32)
[593s] Autotuning complete in 593.1s after searching 1520 configs.
One can hardcode the best config and skip autotuning with:
    @helion.kernel(config=helion.Config(block_sizes=[[64, 128], [16]], loop_orders=[[1, 0]], num_warps=4, num_stages=3, indexing='block_ptr', l2_grouping=32))
```

Because autotuning can be time-consuming (around 10 minutes in the above
example), you may want to manually specify the best configuration found from
autotuning to avoid repeated tuning:

```python
@helion.kernel(config=helion.Config(
    block_sizes=[[64, 128], [16]],
    loop_orders=[[1, 0]],
    num_warps=4,
    num_stages=3,
    indexing='block_ptr',
    l2_grouping=32
))
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...
```

This explicit configuration skips autotuning on subsequent runs.

You can also specify multiple configurations, prompting Helion to perform
a more lightweight autotuning process:

```python
@helion.kernel(configs=[
    helion.Config(...),
    helion.Config(...),
])
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...
```

In this case, Helion evaluates the provided configurations and selects the fastest one.

Additionally, Helion provides programmatic APIs to manage autotuning
and configurations directly from your code.

## Configurations

Helion configurations include the following options:

* **block\_sizes** (`list[int | list[int]]`):
Controls tile sizes corresponding to each `hl.tile` invocation in the
kernel. For tiles with two or more dimensions, you can use either an
integer to flatten the iteration space into a single dimension or a list
of integers for multi-dimensional tiling.

* **loop\_orders** (`list[list[int]]`):
Contains one entry per `hl.tile` call with two or more dimensions,
allowing you to permute the iteration order of the tiles.

* **reduction\_loops** (`list[int | None]`):
Contains one entry per reduction dimension (see
`examples/softmax.py`). Using `None` triggers a persistent reduction,
where the entire reduction is processed in a single tile. Specifying an
integer block size converts the reduction into a loop, beneficial for
larger reductions that exceed the registers available.

* **l2\_grouping** (`int`):
Reorders the program IDs (PIDs) of the generated kernel for improved L2
cache behavior. A value of `1` disables this optimization, while higher
values specify the grouping size.

* **indexing** (`"pointer"`, `"tensor_descriptor"` or `"block_ptr"`):
Specifies the type of indexing code to generate. The `"tensor_descriptor"`
option uses Tensor Memory Accelerators (TMAs) but requires a Hopper or
newer GPU and the latest development version of Triton.

* **use\_yz\_grid** (`bool`):
  Determines if the `y` and `z` dimensions of the launch grid are utilized,
  or if only the `x` dimension is used. This option is ignored if `l2_grouping>1`.

* **num\_warps** (`int`):
Sets the number of warps the kernel will use.

* **num\_stages** (`int`):
Defines the number of pipeline stages to be passed to Triton.

Changing these options results in often significantly different
output Triton code, allowing the autotuner to explore a wide range of
implementations from a single Helion kernel.

## Logs and Debugging

`HELION_LOGS` is the recommended way to emit debugging information from Helion.

An example to this is
```
HELION_LOGS=helion.runtime.kernel python examples/add.py
```
will emit the generated Triton kernels at INFO level logging.
Adding `+` in front of path like `+helion.runtime.kernel` will emit logs at
DEBUG level.

## Requirements

Helion currently targets Linux systems and requires a recent Python and PyTorch environment:

- Linux-based OS
- Python 3.10, 3.11, or 3.12
- [PyTorch] 2.7 or newer
- A development version of [Triton], installed from source
  *(Older versions may work, but will lack support for features like
  TMA on Hopper/Blackwell GPUs and may exhibit lower performance.)*

[PyTorch]: https://github.com/pytorch/pytorch

## Installation

We recommend using a [conda] environment to manage dependencies. First,
install compatible versions of [PyTorch] and [Triton].

[conda]: https://www.anaconda.com/docs/getting-started/miniconda/install#linux

Once your environment is set up, you can install Helion directly from GitHub:

```bash
pip install git+https://github.com/pytorch-labs/helion.git
```

Alternatively, you may install from source for development purposes:
```bash
git clone https://github.com/pytorch-labs/helion.git
cd helion
# To install in editable w/ required dev packages
pip install -e .'[dev]'
````
This installs Helion in "editable" mode so that changes to the source
code take effect without needing to reinstall.

## License

Helion is BSD-style licensed, as found in the LICENSE file.
