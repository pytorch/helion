"""A cloned input's aligned base does not prove contiguous lanes to Inductor."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.compile_environment import tensor_input_sources
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_fragment_index_domains import nested_windowed_cumsum
from helion._testing.cute_fragment_index_domains import windowed_cumsum
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("scope", "begin", "end", "packets"),
    [
        ("grid", 0, 61, False),
        ("grid", 1, 64, False),
        ("grid", 4, 60, True),
        ("inner", 0, 61, False),
        ("inner", 1, 64, False),
        ("inner", 4, 60, True),
        ("runtime-end", 0, 60, False),
    ],
)
def test_tile_packets_require_aligned_start_and_limit(
    scope: str, begin: int, end: int, packets: bool
) -> None:
    x = torch.empty(2, 16, 64)
    args = (x, torch.empty_like(x), begin, end)
    with cpu_render_target(fake_device_sm_count=148):
        bound = (
            cpu_bind(windowed_cumsum, args)
            if scope == "grid"
            else cpu_bind(nested_windowed_cumsum, (*args, scope != "runtime-end"))
        )
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config()) | {"cute_fragment_vector_io": True}
        )
        source = bound.to_code(config)
    tree = ast.parse(source)
    if scope == "runtime-end":
        # The trip count may use a named, clipped end. Follow the bound
        # into the actual device loop instead of requiring a counter spelling.
        device = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
        )
        dynamic_end = {"end"}
        for stmt in device.body:
            if isinstance(stmt, ast.Assign):
                targets = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                reads_end = any(
                    isinstance(n, ast.Name) and n.id in dynamic_end
                    for n in ast.walk(stmt.value)
                )
                dynamic_end.difference_update(targets)
                if reads_end:
                    dynamic_end.update(targets)
            if (
                isinstance(stmt, ast.For)
                and any(
                    isinstance(n, ast.Name) and n.id in dynamic_end
                    for n in ast.walk(stmt.iter)
                )
                and any(
                    isinstance(n, ast.Attribute)
                    and n.attr == "iterator"
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "out"
                    for child in stmt.body
                    for n in ast.walk(child)
                )
            ):
                break
        else:
            pytest.fail("the device loop does not depend on the runtime end")
    # Match the actual global load and store separately: a register or shared
    # copy elsewhere does not establish that this host window uses a packet.
    packet_homes = {
        attr.value.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "cute.copy"
        for attr in ast.walk(call)
        if isinstance(attr, ast.Attribute)
        and attr.attr == "iterator"
        and isinstance(attr.value, ast.Name)
    }
    assert {"x", "out"} & packet_homes == ({"x", "out"} if packets else set())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _overlapping_gram(x: torch.Tensor) -> torch.Tensor:
    ready = x.clone()
    batches, _, width = x.shape
    out = torch.empty((batches, 16, 16), dtype=torch.float32, device=x.device)
    for batch in hl.tile(batches, block_size=1):
        rows = hl.arange(16)
        cols = hl.arange(16)
        features = hl.arange(width)
        a = ready[batch, rows, features]
        b = ready[batch, 16 + rows, features]
        out[batch, rows, cols] = hl.dot(a.float() * 2, b.float().transpose(-2, -1))
    return out


@pytest.mark.parametrize("slab", [False, True])
def test_cloned_input_packets_require_guarded_strides(slab: bool) -> None:
    x = torch.empty(2, 32, 16, dtype=torch.bfloat16)
    # A graph of the same shape can pass a transposed dense input: clone
    # preserves its layout and the consecutive feature addresses are no
    # longer consecutive elements. Both inputs have aligned storage.
    transposed = torch.empty(2, 16, 32, dtype=x.dtype).transpose(1, 2)
    assert x.shape == transposed.shape and transposed.clone().stride(-1) == 32

    def has_input_packets(source: str) -> bool:
        return any(
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "cute.copy"
            and len(call.args) > 1
            and "ready.iterator" in ast.unparse(call.args[1])
            for call in ast.walk(ast.parse(source))
        )

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_overlapping_gram, (x,))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_vector_io": True, "cute_fragment_slab_loads": slab}
        )
        with bound.env.without_dispatch_guards():
            assert has_input_packets(bound.to_code(config))
        with bound.env.without_dispatch_guards(
            symbolic_strides=tensor_input_sources({"x": x}, [x])
        ):
            assert not has_input_packets(bound.to_code(config))
