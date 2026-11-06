"""Shared runners for the ``cute_mma_dag`` runtime tests.

``test/test_cute_mma_dag_gpu.py`` (the warp mode) and
``test/test_cute_mma_dag_tcgen05_gpu.py`` (the tensor-memory mode) compile
the same kernels at pinned configs and compare them with the Triton backend.
One runner renders, checks and launches the cute kernel for either mode, and
the Triton reference of a (kernel, inputs, block sizes) is compiled once per
process.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

LAUNCHES = 2
"""Launches per cute compile whose outputs must agree bitwise: a race or an
uninitialized read shows up as a relaunch that differs."""

DETERMINISM_LAUNCHES = 8
"""The launch count of the rows that keep the long relaunch sweep: the tests
written around races between launches, and one schedule per module with the
most barrier phases per launch."""


def cast_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def tensors(result: object) -> tuple[torch.Tensor, ...]:
    """A kernel's outputs as a tuple (a single tensor or a tuple of them)."""
    return result if isinstance(result, tuple) else (cast_tensor(result),)


RUNTIMES = {"warp": "_helion_mma_dag", "tcgen05": "_helion_mma_dag_tc"}
"""The generated module's runtime name for each ``cute_mma_dag`` mode."""


def run_cute(
    fn: Callable[..., object],
    args: tuple[object, ...],
    config: helion.Config,
    *,
    mode: str,
    launches: int = LAUNCHES,
) -> tuple[str, tuple[torch.Tensor, ...]]:
    """Render, check and launch ``fn(*args)`` under the cute backend.

    ``mode`` is the ``cute_mma_dag`` value of the test module (``"warp"`` or
    ``"tcgen05"``): the source must use that mode's runtime MMA
    (``RUNTIMES``) and no fallback lowering, and the tensor-memory mode must
    survive normalization.  ``launches`` launches must agree bitwise; returns
    the source and the first launch's outputs.
    """
    runtime = RUNTIMES[mode]
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    if mode == "tcgen05":
        assert normalized.config["cute_mma_dag"] == "tcgen05", normalized
    source = bound.to_code(normalized)
    if mode == "tcgen05":
        assert f"{runtime}.prims.tcgen05_mma(" in source, source
        assert "_helion_mma_dag.mma_" not in source
    else:
        assert f"{runtime}.mma_" in source, source
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    compiled = bound.compile_config(normalized)
    outputs = [tensors(compiled(*args)) for _ in range(launches)]
    torch.cuda.synchronize()
    for other in outputs[1:]:
        for lhs, rhs in zip(outputs[0], other, strict=True):
            assert torch.equal(lhs, rhs)
    return source, outputs[0]


_TRITON_REFERENCES: dict[tuple[object, ...], tuple[torch.Tensor, ...]] = {}


def _argument_key(value: object) -> tuple[object, ...]:
    """A hashable identity of a kernel argument.

    A tensor is its shape, dtype, strides, storage offset and a digest of
    all of its bytes, so two inputs of one shape never share a reference and
    an input the kernel writes in place gets a fresh run once its contents
    changed; other arguments are their type and value.
    """
    if isinstance(value, torch.Tensor):
        raw = value.detach().reshape(-1).contiguous().view(torch.uint8).cpu()
        digest = hashlib.blake2b(raw.numpy().tobytes(), digest_size=16).hexdigest()
        return (
            "tensor",
            tuple(value.shape),
            value.dtype,
            tuple(value.stride()),
            value.storage_offset(),
            digest,
        )
    return ("value", type(value).__name__, repr(value))


def run_triton(
    fn: Callable[..., object],
    args: tuple[object, ...],
    block_sizes: Sequence[int],
    *,
    cache: bool = True,
) -> tuple[torch.Tensor, ...]:
    """The Triton backend's outputs of ``fn(*args)`` at ``block_sizes``.

    With ``cache`` the reference is compiled and launched once per process
    for each (kernel, inputs, block sizes) and kept for the process's life:
    callers compare against the returned tensors and must not write into
    them. Pass ``cache=False`` when the kernel writes an output the caller
    passed in ``args`` and reads afterwards: a cached hit would skip the
    launch that fills it.
    """
    key = None
    if cache:
        key = (fn, tuple(block_sizes), *(_argument_key(value) for value in args))
    reference = _TRITON_REFERENCES.get(key) if key is not None else None
    if reference is None:
        kernel = helion.kernel(fn, backend="triton", static_shapes=True)
        bound = kernel.bind(args)
        compiled = bound.compile_config(helion.Config(block_sizes=list(block_sizes)))
        reference = tensors(compiled(*args))
        torch.cuda.synchronize()
        if key is not None:
            # Kept for the worker's life: tens of MB of references at most.
            _TRITON_REFERENCES[key] = reference
    return reference


def assert_close(
    outs: tuple[torch.Tensor, ...],
    refs: tuple[torch.Tensor, ...],
    *,
    scaled: bool = False,
) -> None:
    """``scaled``: the absolute tolerance follows the tensor's magnitude (a
    recurrence of cancelling fp32 sums rounded to bf16 leaves single
    elements a few bf16 ulps of the large terms apart between backends; the
    SIMT cute path shows the same elements against Triton)."""
    for out, ref in zip(outs, refs, strict=True):
        assert out.dtype == ref.dtype and out.shape == ref.shape
        if out.dtype == torch.float32:
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
        else:
            atol = 2e-2 * max(1.0, ref.float().abs().max().item()) if scaled else 2e-2
            torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=2e-2)
