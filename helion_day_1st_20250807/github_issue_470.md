# Issue #470: triton.runtime.errors.OutOfResources: out of resource: shared memory

## Metadata
- **State**: OPEN
- **Author**: [Lucaskabela](https://github.com/Lucaskabela)
- **Created**: August 08, 2025 at 16:28 UTC
- **Updated**: August 08, 2025 at 16:29 UTC

## Description

**Describe the bug**
When running with `use_default_config=True`, a helion kernel with some torch.matmul operation fails with memory issues 

**To Reproduce**
```
"""
Geglu Loss Example
======================

This example demonstrates how to implement a Geglu activation function using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations
import math
import torch

import helion
from helion._testing import run_example
import helion.language as hl



# self.bsz = 8
# self.hidden_size = 4096
# self.intermediate_size = 11008

# tanh approximation form of GELU is computed with:
# 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
class LlamaMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.mlp_bias = False

        # [h, i]
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        # [h, i]
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        # [i, h]
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=self.mlp_bias)

    # x is: [b, t, h]
    # output is: [b, t, h]
    def forward(self, x):
       actv = torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh")
       return self.down_proj(actv * self.up_proj(x))
    
    def forward_mm(self, x):
        return LlamaMLP.forward_instances(x, self.gate_proj.weight.T, self.up_proj.weight.T, self.down_proj.weight.T)
    
    @staticmethod
    def forward_instances(x, gate_proj, up_proj, down_proj):
        a = torch.matmul(x, gate_proj)
        b = torch.matmul(x, up_proj)
        actv = torch.nn.functional.gelu(a, approximate="tanh")
        b = torch.matmul(x, up_proj)
        c = actv * b
        return torch.matmul(c, down_proj)

# Implementation of GegluMLP has...
# - 4 input (x, gate_proj, up_proj, down_proj) and 1 output (down_proj)
# - Order of operations is:
#   - (a): matmul (x, gate_proj)
#   - (b): geglu activation 
#     - (1): a ** 3 * 0.044715
#     - (2): a + (1) (matrix add)
#     - (3): 1 + tanh(sqrt(2 / pi) * 3
#     - (4): 0.5 * a * (3) (matmul)
#   - (c): matmul (x, up_proj)
#   - (d): matmul (b * c)
#   - (e): matmul (down_proj * d)


# @helion.kernel(ref_mode=helion.RefMode.EAGER)
@helion.kernel(use_default_config=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def geglu_mlp(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:

    # Pt 1: Get Tensor sizes
    t, h = x.size()
    h1, i = gate_proj.size()
    i_red = hl.register_reduction_dim(i)
    h2, i2 = up_proj.size()
    i3, h3 = down_proj.size()
    h_red = hl.register_reduction_dim(h)
    assert i == i2 and i2 == i3, "intermediate size mismatch"
    assert h == h1 and h1 == h2 and h2 == h3, "hidden size mismatch"

    # Pt 2: Allocate Output Sizes 
    # out = torch.empty_like(x)
    temp = torch.empty([t, i], device=x.device)

    # quick constant precompute
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
    
    # Pt 3: Use helion to tile for actv!
    for tile_t in hl.tile(t):

        #   - (a): matmul for x, gate_proj
        #   - (c): matmul (x, up_proj)
        acc = hl.zeros([tile_t, i_red], dtype=torch.float32)
        ccc = hl.zeros([tile_t, i_red], dtype=torch.float32)
        for tile_h in hl.tile(h):
            x_tile = x[tile_t, tile_h]
            mm = torch.matmul(x_tile, gate_proj[tile_h, :])
            mm2 = torch.matmul(x_tile, up_proj[tile_h, :])
            acc += mm
            ccc += mm2
        # Acc and ccc should be tile_t, i_red
        # These constants pulled from
        # https://github.com/pytorch/pytorch/blob/e619c6bb90b9dedaccd3cbeed86a288993a4e33f/torch/_refs/nn/functional/__init__.py#L1066

        # Now acc is finished, so (b)
        #     - (1/2): a ** 3 * 0.044715 + a
        acc_2 = kBeta * ((acc ** 3) * 0.044715 + acc)
        #     - (3): 1 + tanh(sqrt(2 / pi) * 3
        acc_3 = 1 + torch.tanh(acc_2)
        #     - (4): 0.5 * a * (3) (matmul)
        acc_4 = 0.5 * acc * acc_3

        # Now we have finished gelu (or have b) - three more matmul
        #   - (d): matmul (b * c)
        dcc = acc_4 * ccc
        temp[tile_t, :] = dcc

    return torch.matmul(temp, down_proj)

# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the geglu kernel verification.
    Tests with a batch size of 128 and vocabulary size of 1000.
    """
    # Test with moderate size
    # n, v = 128, 1000
    # logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    # labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)
    # batch = 8
    hidden = 512
    intermediate = 1524
    T = 100
    x = torch.randn(T, hidden, device="cuda", dtype=torch.float32)
    gated_proj = torch.randn(hidden, intermediate, device="cuda", dtype=torch.float32)
    up_proj = torch.randn(hidden, intermediate, device="cuda", dtype=torch.float32)
    down_proj = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float32)

    run_example(
        geglu_mlp,
        LlamaMLP.forward_instances,
        (x, gated_proj, up_proj, down_proj),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
```

**Expected behavior**
Expected kernel to run (perhaps less efficiently since one matmul is outside of the tiling code)

**Additional context**
While trying to implement the LLaMaMLP, I ran into an issue with the dependent matmuls, so wanted to rewrite the kernel to ensure the first matmuls are done correctly, so I used temp output with an explicit torch.matmul.  My expectation is that this would run, but less efficiently, OR if this is not supported I would get an error message mentioning this.  Instead the program runs and seems to do autotuning with triton, but fails with the memory error:
```
raceback (most recent call last):
  File "/home/lucaskabela/helion/examples/geglu.py", line 177, in <module>
    main()
  File "/home/lucaskabela/helion/examples/geglu.py", line 165, in main
    run_example(
  File "/home/lucaskabela/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 583, in __call__
    return self._run(*args)
           ^^^^^^^^^^^^^^^^
  File "/tmp/torchinductor_lucaskabela/o2/co2vvojuyhkif5i6ppf5xbrxwtu7mv3tur7bdemo5sfxufo7cz2a.py", line 76, in geglu_mlp
    _launcher(_geglu_mlp_kernel, (triton.cdiv(t, _BLOCK_SIZE_2),), x, gate_proj, up_proj, temp, down_proj, out, down_proj.stride(0), down_proj.stride(1), gate_proj.stride(0), gate_proj.stride(1), out.stride(0), out.stride(1), temp.stride(0), temp.stride(1), up_proj.stride(0), up_proj.stride(1), x.stride(0), x.stride(1), t, i2, h, _BLOCK_SIZE_2, _RDIM_SIZE_0, _RDIM_SIZE_1, _BLOCK_SIZE_3, _BLOCK_SIZE_4, num_warps=4, num_stages=3)
  File "/home/lucaskabela/helion/helion/runtime/__init__.py", line 63, in default_launcher
    return triton_kernel.run(
           ^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/runtime/jit.py", line 617, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
    ^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 498, in __getattribute__
    self._init_handles()
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 483, in _init_handles
    raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 526336, Hardware limit: 232448. Reducing block sizes or `num_stages` may help.
(helion) [lucaskabela@devgpu003.pci5 ~/helion/examples (main)]$ python geglu.py
Passed simple check that we have implemented geglu correctly
/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/torch/__init__.py:1605: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  _C._set_float32_matmul_precision(precision)
Testing helion correctness...
Traceback (most recent call last):
  File "/home/lucaskabela/helion/examples/geglu.py", line 177, in <module>
    main()
  File "/home/lucaskabela/helion/examples/geglu.py", line 165, in main
    run_example(
  File "/home/lucaskabela/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 583, in __call__
    return self._run(*args)
           ^^^^^^^^^^^^^^^^
  File "/tmp/torchinductor_lucaskabela/o2/co2vvojuyhkif5i6ppf5xbrxwtu7mv3tur7bdemo5sfxufo7cz2a.py", line 76, in geglu_mlp
    _launcher(_geglu_mlp_kernel, (triton.cdiv(t, _BLOCK_SIZE_2),), x, gate_proj, up_proj, temp, down_proj, out, down_proj.stride(0), down_proj.stride(1), gate_proj.stride(0), gate_proj.stride(1), out.stride(0), out.stride(1), temp.stride(0), temp.stride(1), up_proj.stride(0), up_proj.stride(1), x.stride(0), x.stride(1), t, i2, h, _BLOCK_SIZE_2, _RDIM_SIZE_0, _RDIM_SIZE_1, _BLOCK_SIZE_3, _BLOCK_SIZE_4, num_warps=4, num_stages=3)
  File "/home/lucaskabela/helion/helion/runtime/__init__.py", line 63, in default_launcher
    return triton_kernel.run(
           ^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/runtime/jit.py", line 617, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
    ^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 498, in __getattribute__
    self._init_handles()
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 483, in _init_handles
    raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 526336, Hardware limit: 232448. Reducing block sizes or `num_stages` may help.
(helion) [lucaskabela@devgpu003.pci5 ~/helion/examples (main)]$ python geglu.py
Passed simple check that we have implemented geglu correctly
/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/torch/__init__.py:1605: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  _C._set_float32_matmul_precision(precision)
Testing helion correctness...
WARNING[TensorOperationInWrapper]: A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused in the generated kernel.
Use @helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper]) to suppress this warning.
If this is not a tensor operation, please report this as a bug.
While processing:
  File "/home/lucaskabela/helion/examples/geglu.py", line 131, in geglu_mlp
    return torch.matmul(temp, down_proj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^CTraceback (most recent call last):
  File "/home/lucaskabela/helion/examples/geglu.py", line 173, in <module>
    main()
  File "/home/lucaskabela/helion/examples/geglu.py", line 161, in main
    run_example(
  File "/home/lucaskabela/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 583, in __call__
    return self._run(*args)
           ^^^^^^^^^^^^^^^^
  File "/tmp/torchinductor_lucaskabela/ar/carfdafhyrfmb65gglmenxcvww3stfb2iuzv3uwo5tsmqcd5coc7.py", line 61, in geglu_mlp
    _launcher(_geglu_mlp_kernel, (triton.cdiv(t, _BLOCK_SIZE_2),), x, gate_proj, up_proj, temp, gate_proj.stride(0), gate_proj.stride(1), temp.stride(0), temp.stride(1), up_proj.stride(0), up_proj.stride(1), x.stride(0), x.stride(1), t, i3, h, _BLOCK_SIZE_2, _RDIM_SIZE_0, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
  File "/home/lucaskabela/helion/helion/runtime/__init__.py", line 63, in default_launcher
    return triton_kernel.run(
           ^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/runtime/jit.py", line 594, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 359, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 461, in <lambda>
    stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 424, in make_cubin
    subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/subprocess.py", line 550, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/subprocess.py", line 1201, in communicate
    self.wait()
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/subprocess.py", line 1264, in wait
    return self._wait(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/subprocess.py", line 2053, in _wait
    (pid, sts) = self._try_wait(0)
                 ^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/subprocess.py", line 2011, in _try_wait
    (pid, sts) = os.waitpid(self.pid, wait_flags)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
^C
(helion) [lucaskabela@devgpu003.pci5 ~/helion/examples (main)]$ ^C
(helion) [lucaskabela@devgpu003.pci5 ~/helion/examples (main)]$ python geglu.py
Passed simple check that we have implemented geglu correctly
/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/torch/__init__.py:1605: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  _C._set_float32_matmul_precision(precision)
Testing helion correctness...
Traceback (most recent call last):
  File "/home/lucaskabela/helion/examples/geglu.py", line 173, in <module>
    main()
  File "/home/lucaskabela/helion/examples/geglu.py", line 161, in main
    run_example(
  File "/home/lucaskabela/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/helion/helion/runtime/kernel.py", line 583, in __call__
    return self._run(*args)
           ^^^^^^^^^^^^^^^^
  File "/tmp/torchinductor_lucaskabela/ar/carfdafhyrfmb65gglmenxcvww3stfb2iuzv3uwo5tsmqcd5coc7.py", line 61, in geglu_mlp
    _launcher(_geglu_mlp_kernel, (triton.cdiv(t, _BLOCK_SIZE_2),), x, gate_proj, up_proj, temp, gate_proj.stride(0), gate_proj.stride(1), temp.stride(0), temp.stride(1), up_proj.stride(0), up_proj.stride(1), x.stride(0), x.stride(1), t, i3, h, _BLOCK_SIZE_2, _RDIM_SIZE_0, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
  File "/home/lucaskabela/helion/helion/runtime/__init__.py", line 63, in default_launcher
    return triton_kernel.run(
           ^^^^^^^^^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/runtime/jit.py", line 617, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
    ^^^^^^^^^^
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 498, in __getattribute__
    self._init_handles()
  File "/home/lucaskabela/.conda/envs/helion/lib/python3.12/site-packages/triton/compiler/compiler.py", line 483, in _init_handles
    raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 1056768, Hardware limit: 232448. Reducing block sizes or `num_stages` may help.
```


## Comments

*No comments yet.*
