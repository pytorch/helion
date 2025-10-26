<div align="center">
  <img src="docs/_static/helion_nobackground.png" alt="Helion Logo" width="250"/>
</div>

# News

- **Oct. 21, 2025**: Talk at [2025 Triton Developer Conference](https://tritonconference.eventbuilder.com/TritonDeveloperConference?ref=TritonDeveloperConference) - Helion: A Higher-level DSL for Kernel Authoring
- **Oct. 22, 2025**: [Meet the Developers of PyTorch Compiler and Helion](https://pytorchconference.sched.com/event/27QN9/meet-the-developers-of-pytorch-compiler-and-helion?iframe=no) at PyTorch Conference 2025
- **Oct. 23, 2025**: Talk at PyTorch Conference 2025 - [Helion: A High-level DSL for Kernel Authoring](https://pytorchconference.sched.com/event/27QDl/helion-a-high-level-dsl-for-kernel-authoring-jason-ansel-meta?iframe=no)
- **Dec. 11, 2025**: PyTorch Webinar - [Inside Helion: Live Q&A with the Developers](https://pytorch.org/event/inside-helion-live-qa/)

# About

📚 **[View Documentation](https://helionlang.com)** 📚 | 🎥 **[Watch Talk](https://youtu.be/MBOPzfl1JBo?si=DwAhgL-bpH1kFSt3)** 🎥 | 🚀 **[Try In Colab](https://colab.research.google.com/github/pytorch/helion/blob/main/notebooks/softmax.ipynb)** 🚀

**Helion** is a Python-embedded domain-specific language (DSL) for
authoring machine learning kernels, designed to compile down to [Triton],
a performant backend for programming GPUs and other devices. Helion aims
to raise the level of abstraction compared to Triton, making it easier
to write correct and efficient kernels while enabling more automation
in the autotuning process.

[Triton]: https://github.com/triton-lang/triton

The name *Helion* refers to the nucleus of a helium-3 atom, while *Triton*
refers to hydrogen-3.

Helion can be viewed either as *PyTorch with tiles* or as *a higher-level Triton*. Compared to
Triton, Helion reduces manual coding effort through autotuning. Helion spends more time (approx
10 min) autotuning as it evaluates hundreds of potential Triton implementations generated
from a single Helion kernel. This larger search space also makes kernels more performance
portable between different hardware. Helion automates and autotunes over:

1. **Tensor Indexing:**

   * Automatically calculates strides and indices.
   * Autotunes choices among various indexing methods (pointers, block pointers, TensorDescriptors).
   * Supports per-operation indexing strategies for fine-grained memory access control of loads and stores.

2. **Masking:**

   * Most masking is implicit in Helion, and is optimized away when not needed.

3. **Grid Sizes and PID Calculations:**

   * Automatically determines grid sizes.
   * Autotunes multiple mappings from Program IDs (PIDs) to data tiles.

4. **Implicit Search Space Definition:**

   * Eliminates the need to manually define search configurations.
   * Automatically generates configuration flags and exploration spaces.

5. **Kernel Arguments Management:**

   * Automates the handling of kernel arguments, including tensor sizes and strides.
   * Lifts global variables and (nested) closures into kernel arguments, allowing better templating.

6. **Looping Reductions:**

   * Can automatically convert large reductions into looped implementations.

7. **Automated Optimizations:**

   * PID swizzling for improved L2 cache reuse.
   * Loop reordering.
   * Persistent kernel strategies.
   * Warp specialization choices, unrolling, and more.

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
out = matmul(torch.randn([2048, 2048], device="cuda"),
             torch.randn([2048, 2048], device="cuda"))
```

When a kernel runs for the first time, Helion initiates autotuning. A
typical autotuning session produces output similar to:

```
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
[20s] Initial population: failed=4 min=0.0266 mid=0.1577 max=1.2390 best=Config(block_sizes=[64, 32, 64], loop_orders=[[1, 0]], l2_groupings=[8], range_unroll_factors=[3, 1], range_warp_specializes=[True, False], range_num_stages=[1, 0], range_multi_buffers=[True, True], range_flattens=[None, False], num_warps=4, num_stages=7, indexing='block_ptr', pid_type='persistent_blocked')
[51s] Generation 2: replaced=17 min=0.0266 mid=0.0573 max=0.1331 best=Config(block_sizes=[64, 32, 64], loop_orders=[[1, 0]], l2_groupings=[8], range_unroll_factors=[3, 1], range_warp_specializes=[True, False], range_num_stages=[1, 0], range_multi_buffers=[True, True], range_flattens=[None, False], num_warps=4, num_stages=7, indexing='block_ptr', pid_type='persistent_blocked')
[88s] Generation 3: replaced=18 min=0.0225 mid=0.0389 max=0.1085 best=Config(block_sizes=[64, 64, 16], loop_orders=[[0, 1]], l2_groupings=[4], range_unroll_factors=[0, 1], range_warp_specializes=[None, None], range_num_stages=[0, 0], range_multi_buffers=[None, False], range_flattens=[None, None], num_warps=4, num_stages=6, indexing='pointer', pid_type='flat')
...
[586s] Generation 19: replaced=3 min=0.0184 mid=0.0225 max=0.0287 best=Config(block_sizes=[64, 64, 64], loop_orders=[[0, 1]], l2_groupings=[4], range_unroll_factors=[0, 1], range_warp_specializes=[None, False], range_num_stages=[0, 3], range_multi_buffers=[None, False], range_flattens=[None, None], num_warps=8, num_stages=6, indexing='block_ptr', pid_type='flat')
[586s] Autotuning complete in 586.6s after searching 1520 configs.
One can hardcode the best config and skip autotuning with:
    @helion.kernel(config=helion.Config(block_sizes=[64, 64, 64], loop_orders=[[0, 1]], l2_groupings=[4], range_unroll_factors=[0, 1], range_warp_specializes=[None, False], range_num_stages=[0, 3], range_multi_buffers=[None, False], range_flattens=[None, None], num_warps=8, num_stages=6, indexing='block_ptr', pid_type='flat'))
```

Because autotuning can be time-consuming (around 10 minutes in the above
example), you may want to manually specify the best configuration found from
autotuning to avoid repeated tuning:

```python
@helion.kernel(config=helion.Config(
    block_sizes=[64, 64, 64],
    loop_orders=[[0, 1]],
    l2_groupings=[4],
    range_unroll_factors=[0, 1],
    range_warp_specializes=[None, False],
    range_num_stages=[0, 3],
    range_multi_buffers=[None, False],
    range_flattens=[None, None],
    num_warps=8,
    num_stages=6,
    indexing='block_ptr',
    pid_type='flat'
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

**For production deployment**, we recommend using ahead-of-time tuned configurations rather than relying on runtime autotuning. The autotuning process can be time-consuming and resource-intensive, making it unsuitable for production environments where predictable performance and startup times are critical.

### Static shapes and autotuning keys

By default Helion uses static shapes (`static_shapes=True`). This means each unique input shape/stride signature is treated as its own specialization and will be autotuned separately. This typically yields the best performance, but may increase autotuning time when many shapes are encountered.

If you want to reduce autotuning time by sharing configurations between different shapes, set `static_shapes=False`. In this mode, the autotuning key ignores exact sizes, allowing a single tuned config to be reused across multiple shapes. This can come with a performance penalty compared to fully specialized static shapes.

```python
@helion.kernel(static_shapes=False)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

## Configurations

Helion configurations include the following options:

* **block\_sizes** (`list[int]`):
Controls tile sizes corresponding to each dimension passed `hl.tile` or call
to `hl.register_block_size` in the kernel.

* **loop\_orders** (`list[list[int]]`):
Contains one entry per `hl.tile` call with two or more dimensions,
allowing you to permute the iteration order of the tiles.

* **flatten_loops** (`list[bool]`):
Contains one entry per `hl.tile` call with two or more dimensions,
allowing you to flatten the iteration space into a single dimension.

* **range\_unroll\_factors** (`list[int]`):
Contains one entry per loop dimension, specifying the unroll factor for
`tl.range()` calls. Values less than 1 omit the `loop_unroll_factor` parameter.

* **range\_num\_stages** (`list[int]`):
Contains one entry per loop dimension, specifying the number of stages for
`tl.range()` calls. Values less than 1 omit the `num_stages` parameter.

* **range\_multi\_buffers** (`list[bool | None]`):
Contains one entry per loop dimension, controlling the `disallow_acc_multi_buffer`
parameter for `tl.range()` calls. `True` allows multi-buffer (sets `disallow_acc_multi_buffer=False`),
`False` disallows multi-buffer (sets `disallow_acc_multi_buffer=True`), and `None` omits the parameter.

* **range\_flattens** (`list[bool | None]`):
Contains one entry per loop dimension, controlling the `flatten`
parameter for `tl.range()` calls. `True` sets `flatten=True`,
`False` sets `flatten=False`, and `None` omits the parameter.

* **range\_warp\_specializes** (`list[bool | None]`):
Contains one entry per loop dimension, controlling the `warp_specialize`
parameter for `tl.range()` calls. `True` sets `warp_specialize=True`,
`False` sets `warp_specialize=False`, and `None` omits the parameter.
Only available on CUDA devices with Blackwell or newer architectures
when `allow_warp_specialize` setting is enabled.

* **static\_ranges** (`list[bool]`):
Contains one entry per loop dimension with static bounds, controlling whether to use
`tl.static_range()` calls. `True` generates `tl.static_range()` and ignores range_* configs for that loop. `False` generates `tl.range()`.

* **reduction\_loops** (`list[int | None]`):
Contains one entry per reduction dimension (see
`examples/softmax.py`). Using `None` triggers a persistent reduction,
where the entire reduction is processed in a single tile. Specifying an
integer block size converts the reduction into a loop, beneficial for
larger reductions that exceed the registers available.

* **l2\_groupings** (`list[int]`):
Reorders the program IDs (PIDs) of the generated kernel for improved L2
cache behavior. A value of `1` disables this optimization, while higher
values specify the grouping size.

* **indexing** (`"pointer"`, `"tensor_descriptor"`, `"block_ptr"`, or a list of these):
Specifies the memory indexing strategy for load and store operations. Can be:
  - A single strategy (applies to all loads and stores): `indexing="block_ptr"`
  - A list of strategies (one per load/store in execution order): `indexing=["pointer", "pointer", "block_ptr"]`
  - Empty/omitted (defaults to `"pointer"` for all operations)
  - When using a list, provide strategies in order: `[load1, load2, ..., store1, store2, ...]`

  The `"tensor_descriptor"` option uses Tensor Memory Accelerators (TMAs) but
  requires a Hopper or newer GPU and the latest development version of Triton.

* **pid\_type** (`"flat"`, `"xyz"`, `"persistent_blocked"`, or `"persistent_interleaved"`):
  Specifies the program ID mapping strategy. `"flat"` uses only the x-dimension,
  `"xyz"` utilizes multiple grid dimensions, and persistent strategies enable
  persistent kernels for improved SM utilization.

* **num\_warps** (`int`):
Sets the number of warps the kernel will use.

* **num\_stages** (`int`):
Defines the number of pipeline stages to be passed to Triton.

* **load_eviction_policies** (`list[str]`):
Controls eviction policy used for loads discovered in device loops. Each entry
corresponds to a load site; allowed values are `""` (no policy), `"first"`
(maps to Triton `evict_first`), and `"last"` (maps to Triton `evict_last`).
Explicit `eviction_policy=...` on `hl.load` overrides this config.

Changing these options results in often significantly different
output Triton code, allowing the autotuner to explore a wide range of
implementations from a single Helion kernel.

## Settings for Development and Debugging

When developing kernels with Helion, you might prefer skipping autotuning for faster iteration. To
do this, set the environment variable `HELION_AUTOTUNE_EFFORT=none` or use the decorator argument
`@helion.kernel(autotune_effort="none")`. **Warning:** The default configuration is slow and not intended for
production or performance testing.

To view the generated Triton code, set the environment variable `HELION_PRINT_OUTPUT_CODE=1` or include
`print_output_code=True` in the `@helion.kernel` decorator. This prints the Triton code to `stderr`, which is
helpful for debugging and understanding Helion's compilation process.  One can also use
`foo_kernel.bind(args).to_triton_code(config)` to get the Triton code as a string.

Within an `hl.tile`/`hl.grid` device loop, if you want to print intermediate results using `print("x", ...)` syntax,
or pause execution using Python's built-in `breakpoint()`, set either `TRITON_INTERPRET=1` (runs Triton's CPU interpreter)
or `HELION_INTERPRET=1` (runs the Helion kernel in eager mode).

To force autotuning, bypassing provided configurations, set `HELION_FORCE_AUTOTUNE=1` or invoke `foo_kernel.autotune(args,
force=True)`.

Additional settings are available in
[settings.py](https://github.com/pytorch/helion/blob/main/helion/runtime/settings.py).  If both an environment
variable and a kernel decorator argument are set, the kernel decorator argument takes precedence, and the environment
variable will be ignored.

Enable logging by setting the environment variable `HELION_LOGS=all` for INFO-level logs, or `HELION_LOGS=+all`
for DEBUG-level logs. Alternatively, you can specify logging for specific modules using a comma-separated list
(e.g., `HELION_LOGS=+helion.runtime.kernel`).


## Requirements

Helion currently targets Linux systems and requires a recent Python and PyTorch environment:

- Linux-based OS
- Python 3.10–3.14
- [PyTorch] 2.9 or later
- [Triton] 3.5 or later
  *(Older versions may work, but will lack support for features like
  TMA on Hopper/Blackwell GPUs and may exhibit lower performance.)*

[PyTorch]: https://github.com/pytorch/pytorch

## Installation

We recommend using [uv] to manage an isolated virtual environment. First,
install compatible versions of [PyTorch] and [Triton].

[uv]: https://docs.astral.sh/uv/

Once your environment is set up, you can install Helion:

```bash
pip install helion
```

Alternatively, you may install from source for development purposes. If using `uv`, create and activate a virtual environment first:
```bash
git clone https://github.com/pytorch/helion.git
cd helion

# Create and activate a virtual environment with uv (one-time)
uv venv .venv
source .venv/bin/activate

# To install in editable w/ required dev packages
pip install -e .'[dev]'
```
This installs Helion in "editable" mode so that changes to the source
code take effect without needing to reinstall.

## Linting

We use `pre-commit` to run ruff, pyright, and other checks automatically.

– One-time setup (installs the git hook):
```bash
pip install pre-commit
pre-commit install
```

– Run all checks across the repository:
```bash
pre-commit run --all-files
```

Note: You can still run the underlying tools directly via `./lint.sh [fix|check|unsafe]`.

## Community

Questions or feedback? Join us on the [GPU MODE Discord](https://discord.gg/gpumode) in the `#helion` channel.

## License

Helion is BSD-style licensed, as found in the LICENSE file.
