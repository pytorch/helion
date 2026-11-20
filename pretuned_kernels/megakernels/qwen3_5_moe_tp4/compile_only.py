# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202
"""Codegen-only build of the Qwen3.5 TP4 MoE megakernel on one GPU (no launch): a
fake 4-rank process group stands in for the TP4 group, so get_remote_tensors
traces 4 peers. With --warm, also compiles the generated Gluon kernel
(kernel.warmup, no launch) and prints its shared memory, registers and spills.

CUDA_VISIBLE_DEVICES=N PYTHONHASHSEED=0 python -m \
    pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only [--warm] [out.py]

``generate()`` and ``fake_group()`` serve the other Qwen3.5 TP4 megakernels'
compile_only scripts.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import qwen3_5_moe_tp4
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

TOKENS = 7


@contextlib.contextmanager
def fake_group():
    """A fake 4-rank default process group; yields its name."""
    dist.init_process_group("fake", rank=0, world_size=4, store=FakeStore())
    try:
        yield dist.group.WORLD.group_name
    finally:
        dist.destroy_process_group()


def moe_args(device, group_name, tokens=TOKENS) -> tuple:
    """qwen3_5_moe_tp4's arguments on uninitialized tensors."""
    shape = _setup.Shape()

    def bf(*size):
        return torch.empty(size, dtype=torch.bfloat16, device=device)

    return (
        bf(tokens, shape.hidden),  # hidden_input
        *_setup.empty_weight_args(device, shape),
        bf(tokens, shape.hidden),  # residual
        bf(shape.hidden),  # norm_weight
        bf(tokens, shape.hidden),  # symmetric_partial
        group_name,
    )


def generate(kernel, args, out, *, warm=False) -> str:
    """Write the kernel's code for its pinned config to ``out``; with ``warm``,
    also compile it and print its resources."""
    bound = kernel.bind(args)
    config = kernel.configs[0]
    code = bound.to_triton_code(config)
    pathlib.Path(out).write_text(code)
    print(f"{kernel.name}: {len(code.splitlines())} lines -> {out}")
    if warm:
        from helion.runtime.triton.launcher import compile_only_launch_args

        fn = bound.compile_config(config)

        def launcher(kernel, grid, *a, num_warps, num_stages, **kw):
            a, kw = compile_only_launch_args(*a, **kw)
            c = kernel.warmup(
                *a, grid=grid, num_warps=num_warps, num_stages=num_stages, **kw
            )
            c._init_handles()
            print(
                f"compiled: grid {grid} shared {c.metadata.shared} (limit 232448) "
                f"regs {getattr(c, 'n_regs', None)} spills {getattr(c, 'n_spills', None)}"
            )

        # The fake group has no symmetric memory: the host's peer list is unused here.
        real = torch.ops.symm_mem.get_remote_tensors
        torch.ops.symm_mem.get_remote_tensors = lambda t, g: [t] * 4
        try:
            fn(*args, _launcher=launcher)
        finally:
            torch.ops.symm_mem.get_remote_tensors = real
    return code


def main():
    argv = [a for a in sys.argv[1:] if a != "--warm"]
    out = argv[0] if argv else "/tmp/qwen3_5_moe_tp4_gen.py"
    with fake_group() as group_name:
        args = moe_args(torch.device("cuda"), group_name)
        generate(qwen3_5_moe_tp4, args, out, warm="--warm" in sys.argv)


if __name__ == "__main__":
    main()
