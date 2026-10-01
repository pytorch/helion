from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.cute.input_view_layout import input_view_copy_facts
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion.runtime.kernel import Kernel


def _env(
    inputs: list[torch.Tensor], *, metadata: bool = True, static: bool = True
) -> CompileEnvironment:
    return cast(
        "CompileEnvironment",
        SimpleNamespace(
            settings=SimpleNamespace(static_shapes=static),
            compiler_fact_specialization_facts=frozenset({"input_tensor_metadata"})
            if metadata
            else frozenset(),
            input_sources=dict.fromkeys(inputs),
            tensor_input_source=lambda value: (
                "input" if any(value is item for item in inputs) else None
            ),
            backend=SimpleNamespace(
                dtype_str=lambda value: {
                    torch.float32: "cutlass.Float32",
                    torch.float16: "cutlass.Float16",
                }[value]
            ),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("offset", [0, 16, 32])
def test_view_uses_relative_input_pointer_displacement(
    dtype: torch.dtype, offset: int
) -> None:
    allocation = torch.empty(256, dtype=dtype)
    # A base can itself have a nonzero storage offset. Its alignment is the
    # guarded runtime data_ptr alignment, not the allocation's alignment.
    base = allocation[offset : offset + 128]
    view = base.view(8, 16).T
    with (
        patch(
            "helion._compiler.cute.input_view_layout.tensor_has_specialized_base_alignment",
            return_value=True,
        ) as guard,
        patch(
            "helion._compiler.cute.input_view_layout._pointer_preserving_views",
            return_value={view: base},
        ),
    ):
        facts = input_view_copy_facts(_env([base]), view)
    assert facts is not None
    assert facts.strides == (1, 16) and facts.alignment_bytes == 16
    assert guard.call_args is not None
    assert guard.call_args.args[1] is base


@pytest.mark.parametrize(
    "change",
    [
        "unaligned_view",
        "unaligned_input",
        "missing_metadata",
        "dynamic",
        "fresh",
        "ambiguous_alias",
        "reinterpret",
    ],
)
def test_view_proof_rejects_missing_or_ambiguous_facts(change: str) -> None:
    base = torch.empty(256)
    view = base[16:144].view(8, 16)
    inputs = [base]
    if change == "unaligned_view":
        view = base[1:129].view(8, 16)
    elif change == "fresh":
        view = torch.empty((8, 16))
    elif change == "ambiguous_alias":
        inputs.append(base[:])
    elif change == "reinterpret":
        view = base.view(torch.float16)
    env = _env(
        inputs, metadata=change != "missing_metadata", static=change != "dynamic"
    )
    with (
        patch(
            "helion._compiler.cute.input_view_layout.tensor_has_specialized_base_alignment",
            return_value=change != "unaligned_input",
        ),
        patch(
            "helion._compiler.cute.input_view_layout._pointer_preserving_views",
            return_value={view: base},
        ),
    ):
        assert input_view_copy_facts(env, view) is None


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _known_view(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    out = torch.empty_like(flat)
    for row in hl.tile(flat.numel()):
        out[row] = flat[row]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _absolute_storage_view(x: torch.Tensor) -> torch.Tensor:
    flat = torch.as_strided(x, (x.numel(),), (1,), storage_offset=0)
    out = torch.empty_like(flat)
    for row in hl.tile(flat.numel()):
        out[row] = flat[row]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mutated_view(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    x.add_(1)
    out = torch.empty_like(flat)
    for row in hl.tile(flat.numel()):
        out[row] = flat[row]
    return out


@pytest.mark.parametrize("kernel", [_known_view, _absolute_storage_view, _mutated_view])
@skipUnlessBackends(["cute"])
def test_host_view_origin_and_effects_are_checked(kernel: Kernel[torch.Tensor]) -> None:
    bound = _cpu_bind(kernel, (torch.empty((3, 4)),))
    host = bound.host_function
    assert host is not None
    source = next(
        node.args[0].meta["val"]
        for graph in host.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is memory_ops.load
    )
    bound.env.compiler_fact_specialization_facts |= frozenset({"input_tensor_metadata"})
    with (
        bound.env,
        host,
        patch(
            "helion._compiler.cute.input_view_layout.tensor_has_specialized_base_alignment",
            return_value=True,
        ),
    ):
        facts = input_view_copy_facts(bound.env, source)
    assert (facts is not None) == (kernel is _known_view)
