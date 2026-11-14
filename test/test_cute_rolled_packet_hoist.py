"""CPU codegen tests: which statements a rolled reduction's packet load may pass.

A looped reduction lowers each 16- or 32-bit row it reads along the
reduction dim as one packet per lane, hoisted above the lane's constexpr
V-loop (``memory_ops._cute_register_unroll_vec_hoist``), so the packet loads
before every statement the V-loop body already holds.  The proof
(``_cute_looped_unroll_uniform_definitions``) admits register computations
and loads there, and:

- a packet store's per-lane append: the store flushes after the V-loop, so the
  packet stays ahead of it either way;
- a read-after-write barrier ordering another tensor's stores, when no write
  in the kernel can alias the loaded tensor.

Under ``cute_bool_byte_packets``, bool rows load as byte packets
(``_CUTE_VECTOR_UNROLL_BYTE_DTYPES``): one integer of V bytes per lane, each
element a byte compared against zero.

``scaled_workspace_reload`` is T5's softmax-dropout backward in miniature: its
first sweep stores the workspace before it reads ``grad`` and ``keep``, its
second reads the workspace back behind the barrier and ``grad`` and ``keep``
again.
"""

from __future__ import annotations

import ast
import functools
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.memory_ops import bool_reduction_row_blocks
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_sweep_kernels import scaled_workspace_config
from helion._testing.cute_sweep_kernels import scaled_workspace_reload
from helion.autotuner.config_spec import ReductionLoopSpec
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

_ROWS, _N = 8, 4096


def _inplace_scaled_reload(
    x: torch.Tensor, grad: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """``scaled_workspace_reload`` writing its result over ``grad``."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        ws[tile_r, :] = v * v
        scaled = grad[tile_r, :].to(torch.float32) * (
            keep[tile_r, :].to(torch.float32) * 1.1111111111111112
        )
        total = torch.sum(scaled * v, dim=-1, keepdim=True)
        grad[tile_r, :] = (ws[tile_r, :] * scaled - total).to(grad.dtype)
    return grad


@functools.cache
def _bound(fn: Callable[..., torch.Tensor]) -> BoundKernel:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    args = (
        torch.empty(_ROWS, _N, dtype=torch.bfloat16),
        torch.empty(_ROWS, _N, dtype=torch.bfloat16),
        torch.empty(_ROWS, _N, dtype=torch.bool),
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args)


@functools.cache
def _code(
    fn: Callable[..., torch.Tensor] = scaled_workspace_reload,
    reloads: str = "auto",
    bool_byte_packets: bool = False,
) -> str:
    """The kernel at 1024-element chunks, 128 threads, V2."""
    bound = _bound(fn)
    config = scaled_workspace_config(
        bound.config_spec, reloads, bool_byte_packets=bool_byte_packets
    )
    with mock_cuda_unavailable():
        return bound.to_code(config)


def _sweeps(
    fn: Callable[..., torch.Tensor] = scaled_workspace_reload,
    reloads: str = "auto",
    bool_byte_packets: bool = False,
) -> tuple[str, str]:
    """The reduce and consume sweeps of ``_code``."""
    tree = ast.parse(_code(fn, reloads, bool_byte_packets))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    reduce, consume = (
        node
        for node in kernel.body
        if isinstance(node, ast.For) and ast.unparse(node.target) == "roffset_1"
    )
    return ast.unparse(reduce), ast.unparse(consume)


def _loads(sweep: str, tensor: str) -> tuple[int, int]:
    """(packet loads, scalar loads) of ``tensor`` in one sweep's code."""
    packets = scalars = 0
    for node in ast.walk(ast.parse(sweep)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if ast.unparse(func) == "cute.arch.load":
            packets += f"{tensor}.iterator" in ast.unparse(node.args[0])
        elif isinstance(func, ast.Attribute) and func.attr == "load":
            scalars += f"{tensor}.iterator" in ast.unparse(func.value)
    return packets, scalars


def test_packet_passes_a_packet_store_append() -> None:
    # The workspace store's append precedes ``grad`` in the V-loop; ``grad``
    # still loads as one packet per lane above it.
    reduce, _consume = _sweeps()
    assert "_reduction_store_vals_0.append(" in reduce
    assert _loads(reduce, "grad") == (1, 0)


def test_read_only_packet_passes_the_raw_barrier() -> None:
    # The barrier orders the workspace stores before its reload, which stays
    # per element behind it; ``grad``, which nothing writes, loads as packets.
    _reduce, consume = _sweeps()
    assert "cute.arch.sync_threads()" in consume
    assert _loads(consume, "grad") == (1, 0)
    assert _loads(consume, "ws") == (0, 1)


def test_written_tensor_stays_behind_the_raw_barrier() -> None:
    # Over a ``grad`` the kernel writes, the consume sweep keeps its loads per
    # element behind the barrier; the reduce sweep, where no barrier precedes
    # them, still loads packets.
    reduce, consume = _sweeps(_inplace_scaled_reload)
    assert _loads(reduce, "grad") == (1, 0)
    assert "cute.arch.sync_threads()" in consume
    assert _loads(consume, "grad") == (0, 1)


def test_register_reloads_serve_the_consume_sweep_from_the_packets() -> None:
    # With register reloads the consume sweep reads the reduce sweep's
    # ``grad`` packets, and its ``keep`` byte packets in their ``Uint16``
    # carrier, from fragments instead of loading them again.
    reduce, consume = _sweeps(reloads="register", bool_byte_packets=True)
    assert _loads(reduce, "grad") == (1, 0)
    assert _loads(reduce, "keep") == (1, 0)
    assert _loads(consume, "grad") == (0, 0)
    assert _loads(consume, "keep") == (0, 0)
    assert "_fuse_cache_" in consume
    # The consume sweep's ``keep`` packet is a read of a fragment holding one
    # 2-byte packet per lane (4 chunks of 4 lanes).
    match = re.search(r"_unroll_vec_\d+ = (_fuse_cache_\d+)\[", consume)
    assert match is not None, consume
    assert f"{match.group(1)} = cute.make_rmem_tensor(16, cutlass.Uint16)" in _code(
        scaled_workspace_reload, "register", True
    )


@functools.cache
def _mask_sum_config(
    spec: ConfigSpec, vec_width: int, reduction_loop: int | None
) -> helion.Config:
    """64 threads on the row of ``_mask_sum``, rolled in
    ``reduction_loop``-element chunks or persistent, under
    ``cute_bool_byte_packets``."""
    (rdim,) = spec.reduction_loops.valid_block_ids()
    return helion.Config(
        block_sizes=[1],
        reduction_loops=[reduction_loop],
        num_threads=[
            64 if block_id == rdim else 1
            for block_id in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            vec_width if block_id == rdim else 1
            for block_id in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_bool_byte_packets=True,
    )


def _mask_sum_code(n: int, vec_width: int, reduction_loop: int | None) -> str:
    """``mask[tile_r, :].to(torch.float32).sum(-1)`` over 8 x ``n`` bools
    (``_mask_sum_config``)."""
    bound = _mask_sum_bound(n)
    config = _mask_sum_config(bound.config_spec, vec_width, reduction_loop)
    with mock_cuda_unavailable():
        return bound.to_code(config)


def _mask_sum(mask: torch.Tensor) -> torch.Tensor:
    out = torch.empty([mask.size(0)], dtype=torch.float32, device=mask.device)
    for tile_r in hl.tile(mask.size(0)):
        out[tile_r] = mask[tile_r, :].to(torch.float32).sum(-1)
    return out


@functools.cache
def _mask_sum_bound(n: int) -> BoundKernel:
    kernel = helion.kernel(_mask_sum, backend="cute", static_shapes=True)
    with mock_cuda_unavailable():
        return cpu_bind(kernel, (torch.empty(_ROWS, n, dtype=torch.bool),))


@pytest.mark.parametrize(
    ("n", "vec_width", "reduction_loop", "packet"),
    [
        pytest.param(_N, 8, 1024, "cutlass.Uint64", id="rolled-v8"),
        # A row that ends inside its last chunk masks that chunk: its byte
        # packets are predicated on the chunk, as an integer carrier's are.
        pytest.param(3000, 8, 1024, "cutlass.Uint64", id="rolled-v8-ragged"),
        # 16 bytes is wider than one packed integer: scalar loads, as before.
        pytest.param(_N, 16, 1024, None, id="rolled-v16"),
        # A persistent row (one 8-byte vector per thread) keeps its
        # per-element loads.
        pytest.param(512, 8, None, None, id="persistent-v8"),
    ],
)
def test_bool_byte_packets_only_where_one_integer_holds_them(
    n: int, vec_width: int, reduction_loop: int | None, packet: str | None
) -> None:
    code = _mask_sum_code(n, vec_width, reduction_loop)
    if reduction_loop is None:
        assert "roffset_" not in code, code
    packets, scalars = _loads(code, "mask")
    if packet is None:
        assert packets == 0 and scalars > 0, code
    else:
        assert packets == 1 and scalars == 0, code
        assert f"{packet})" in code, code


@pytest.mark.parametrize(
    ("n", "vec_width", "reduction_loop", "kept"),
    [
        pytest.param(_N, 8, 1024, True, id="rolled-v8"),
        pytest.param(_N, 16, 1024, False, id="rolled-v16"),
        pytest.param(_N, 1, 1024, False, id="rolled-v1"),
        pytest.param(512, 8, None, False, id="persistent-v8"),
    ],
)
def test_an_inert_bool_byte_packets_knob_turns_off(
    n: int, vec_width: int, reduction_loop: int | None, kept: bool
) -> None:
    # Where no bool row rolls at a width one packed integer holds, the knob
    # changes no code, so normalization turns it off: the search would
    # otherwise compile and measure the same code twice.
    spec = _mask_sum_bound(n).config_spec
    config = spec.normalized_config(_mask_sum_config(spec, vec_width, reduction_loop))
    assert config.config["cute_bool_byte_packets"] is kept
    assert spec.normalized_config(config) == config


def test_bool_rows_load_as_byte_packets() -> None:
    # ``keep`` loads one 2-byte integer per lane at V2 in both sweeps, the
    # consume sweep's behind the barrier like ``grad``'s.
    reduce, consume = _sweeps(bool_byte_packets=True)
    for sweep in (reduce, consume):
        assert _loads(sweep, "keep") == (1, 0)
        assert "cutlass.Uint16)" in sweep
        assert "& 255) != cutlass.Uint8(0)" in sweep
    # Off by default: one byte load per element, in the V-loop.
    for sweep in _sweeps():
        assert _loads(sweep, "keep") == (0, 1)


def _row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(x.size(0)):
        out[tile_r] = x[tile_r, :].sum(-1)
    return out


def test_bool_byte_packets_are_offered_for_a_rollable_bool_row() -> None:
    # The knob opens on structure alone: the mask sum offers it at every
    # width (its search fingerprint does not change), a sum without a bool
    # row does not and rejects an explicit value.
    specs = [_mask_sum_bound(n).config_spec for n in (_N, 3000, 600)]
    assert all(spec.cute_bool_byte_packet_sites for spec in specs)
    assert len({spec.structural_fingerprint() for spec in specs}) == 1
    kernel = helion.kernel(_row_sum, backend="cute", static_shapes=True)
    with mock_cuda_unavailable():
        spec = cpu_bind(kernel, (torch.empty(_ROWS, _N),)).config_spec
    assert not spec.cute_bool_byte_packet_sites
    with pytest.raises(exc.InvalidConfig, match="cute_bool_byte_packets requires"):
        spec.normalized_config(helion.Config(cute_bool_byte_packets=True))
    # An explicit False, as a config tuned where the knob applies carries it,
    # is dropped.
    off = spec.normalized_config(
        helion.Config(**spec.default_config().config, cute_bool_byte_packets=False)
    )
    assert off == spec.default_config()


def test_a_pinned_reduction_slot_offers_no_bool_byte_packets() -> None:
    # A pinned slot's dim stays persistent (``ReductionLoopSpec.pinned``), so
    # the bool row it reads never rolls.
    kernel = helion.kernel(_mask_sum, backend="cute", static_shapes=True)
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(_ROWS, 2048, dtype=torch.bool),))
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    (slot,) = spec.reduction_loops
    with bound.env:
        assert bool_reduction_row_blocks(device_ir, spec) == {slot.block_id}
        spec.reduction_loops[0] = ReductionLoopSpec(
            block_id=slot.block_id,
            size_hint=slot.size_hint,
            persistent_reason="rolled by older versions",
        )
        assert not bool_reduction_row_blocks(device_ir, spec)


def test_bool_byte_packets_are_covered_on_rolled_seeds() -> None:
    # Off by default, the knob is offered to the search through a coverage
    # group: up to four of the first compiler seeds that roll the row at a
    # width one packed integer holds, each with the knob on.
    spec = _bound(scaled_workspace_reload).config_spec
    (group,) = (
        group
        for group in spec.compiler_coverage_groups
        if group.key == "cute_bool_byte_packets"
    )
    assert group.legacy is False and group.domain == (False, True)
    assert 1 <= len(group.witnesses) <= 4
    (rdim,) = spec.reduction_loops.valid_block_ids()
    for witness in group.witnesses:
        assert witness.value is True
        assert (
            spec.reduction_loops.config_get(
                list(witness.carrier.config["reduction_loops"]), rdim, None
            )
            is not None
        )
