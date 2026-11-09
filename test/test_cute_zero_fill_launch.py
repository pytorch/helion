"""The host zero fills of a ``cute_pdl`` kernel launch as dependent fill
kernels (``helion/_compiler/cute/host_zero_fill_launches.py``).

``acc = torch.zeros(...)`` ahead of a device loop that accumulates into
``acc`` is a memset node in the stream; between the previous kernel and a
``cute_pdl`` kernel it costs 1.1-1.4 us of device time in a CUDA graph and
defeats the dependent launch (a kernel behind a memset node launches as a
plain edge).  With the knob on, the fill becomes ``empty`` plus the launch
of a small fill kernel of the same module that waits at entry and releases
its dependents right away; off, the memset stays (a plain fill kernel is
slower than the memset).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.host_zero_fill_launches import MAX_FILL_NUMEL
from helion._compiler.cute.pdl_release import ZERO_FILL_KERNEL_PREFIX
from helion._compiler.cute.pdl_release import generic_release_rendered
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")

_WAIT = "cute.arch.griddepcontrol_wait()"
_RELEASE = "cute.arch.griddepcontrol_launch_dependents()"


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def atomic_column_sum(x: torch.Tensor) -> torch.Tensor:
    """Column sums through atomics into a zero-filled fp32 accumulator: the
    fill cannot be elided (no CTA writes a whole column)."""
    m, n = x.size()
    out = torch.zeros([n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        hl.atomic_add(out, [tile_n], x[tile_m, tile_n].to(torch.float32).sum(0))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scaled_copy(x: torch.Tensor) -> torch.Tensor:
    """A zero-filled output every element of which the grid stores: the
    dead-fill elision turns the fill into ``empty`` ahead of this pass."""
    m, n = x.size()
    out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


_COLUMN_SUM = {"block_sizes": [64, 64], "num_threads": [4, 64]}


def _kernel_defs(code: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    }


def _host_lines(code: str, name: str) -> list[str]:
    (host,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return [ast.unparse(statement) for statement in host.body]


def _codes(kernel: helion.Kernel, args: tuple, **config: object) -> tuple[str, str]:
    bound = cpu_bind(kernel, args)
    spec = bound.config_spec
    on = bound.to_code(
        spec.normalized_config(helion.Config.from_dict({**config, "cute_pdl": True}))
    )
    off = bound.to_code(spec.normalized_config(helion.Config.from_dict(config)))
    return on, off


def test_the_accumulator_fill_launches_as_a_dependent_kernel(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    on, off = _codes(atomic_column_sum, (x,), **_COLUMN_SUM)
    # Knob off: the memset stays, no fill kernel.
    assert "out = torch.zeros([n], dtype=torch.float32, device=x.device)" in off
    assert ZERO_FILL_KERNEL_PREFIX not in off
    # Knob on: a second kernel of the module, waiting and releasing at entry,
    # 16-byte packets of fp32 zeros over the flat view and a scalar tail.
    kernels = _kernel_defs(on)
    (fill_name,) = [k for k in kernels if k.startswith(ZERO_FILL_KERNEL_PREFIX)]
    assert fill_name == f"{ZERO_FILL_KERNEL_PREFIX}0_helion_atomic_column_sum"
    statements = [ast.unparse(s) for s in kernels[fill_name].body]
    assert statements[:2] == [_WAIT, _RELEASE]
    assert "zf_zero = cute.make_rmem_tensor((4,), cutlass.Float32)" in statements
    assert "for zf_i in range(zf_tid, zf_n // 4, zf_stride):" in ast.unparse(
        kernels[fill_name]
    )
    assert f"{fill_name}._helion_cute_use_pdl = True" in on
    # The host allocates with ``empty`` and launches the fill where the
    # ``zeros`` stood, ahead of the kernel's own launch: 1024 fp32 elements
    # fill from one 256-thread CTA (4 elements per thread), capped at four
    # CTAs per SM (148 SMs under the CPU target).
    lines = _host_lines(on, "atomic_column_sum")
    allocation = lines.index(
        "out = torch.empty([n], dtype=torch.float32, device=x.device)"
    )
    assert lines[allocation + 1] == (
        f"_launcher({fill_name}, (max(1, min((out.numel() + 1023) // 1024, 592)),), "
        "out.view(-1), block=(256, 1, 1))"
    )
    (launch,) = [
        index
        for index, line in enumerate(lines)
        if line.startswith("_launcher(_helion_atomic_column_sum,")
    ]
    assert allocation + 1 < launch
    # The kernel proper releases at entry and, behind the fill kernel, waits
    # late: below its row loop, right before the atomic into ``out``.
    body = kernels["_helion_atomic_column_sum"].body
    main = [ast.unparse(s) for s in body]
    assert main[0] == _RELEASE
    wait = main.index(_WAIT)
    assert any(line.startswith("for ") for line in main[:wait])

    def touches_out(statement: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id == "out"
            for node in ast.walk(statement)
        )

    assert not any(touches_out(statement) for statement in body[:wait])
    assert touches_out(body[wait + 1])
    # The fill kernel never stands in for the kernel's own early release.
    assert generic_release_rendered(on)
    bound = cpu_bind(atomic_column_sum, (x,))
    late = bound.to_code(
        bound.config_spec.normalized_config(
            helion.Config.from_dict(
                {**_COLUMN_SUM, "cute_pdl": True, "cute_pdl_release": "none"}
            )
        )
    )
    assert ZERO_FILL_KERNEL_PREFIX in late
    assert not generic_release_rendered(late)


def test_a_dead_fill_stays_an_empty_without_a_fill_kernel(cpu_only: None) -> None:
    x = torch.empty(512, 512, dtype=torch.bfloat16)
    on, off = _codes(scaled_copy, (x,), block_sizes=[64, 64], num_threads=[4, 64])
    for code in (on, off):
        assert "out = torch.empty([m, n], dtype=x.dtype, device=x.device)" in code
        assert ZERO_FILL_KERNEL_PREFIX not in code
    assert _WAIT in on and _WAIT not in off


def test_the_accumulator_reads_zeros_on_every_call() -> None:
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
    code, out = code_and_output(atomic_column_sum, (x,), **_COLUMN_SUM, cute_pdl=True)
    assert ZERO_FILL_KERNEL_PREFIX in code
    expected = x.to(torch.float32).sum(0)
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-3)
    # A second call (another empty + fill, no memset) accumulates from zero.
    again = atomic_column_sum(x)
    torch.testing.assert_close(again, expected, atol=1e-1, rtol=1e-3)


def test_the_backend_sees_the_fill_kernels_of_a_module() -> None:
    """The persistent compile cache, the module annotation and the source
    hash reach the fill kernels through the host function's globals."""
    from types import SimpleNamespace

    from helion._compiler.cute.backend import CuteBackend

    main = SimpleNamespace(_helion_cute_source_hash="h")
    fill0 = SimpleNamespace(_helion_cute_source_hash="h")
    fill1 = SimpleNamespace(_helion_cute_source_hash="h")
    other = SimpleNamespace(_helion_cute_source_hash="other")

    def host() -> None:
        pass

    host.__globals__.update(  # type: ignore[attr-defined]
        {
            "_helion_host": main,
            f"{ZERO_FILL_KERNEL_PREFIX}1_helion_host": fill1,
            f"{ZERO_FILL_KERNEL_PREFIX}0_helion_host": fill0,
            f"{ZERO_FILL_KERNEL_PREFIX}0_helion_other": other,
        }
    )
    try:
        assert CuteBackend._compiled_kernels(host, "host") == (main, fill0, fill1)
        assert CuteBackend().generated_source_hash(host) == "h"
        fill1._helion_cute_source_hash = "stale"
        assert CuteBackend().generated_source_hash(host) is None
    finally:
        for key in (
            "_helion_host",
            f"{ZERO_FILL_KERNEL_PREFIX}1_helion_host",
            f"{ZERO_FILL_KERNEL_PREFIX}0_helion_host",
            f"{ZERO_FILL_KERNEL_PREFIX}0_helion_other",
        ):
            host.__globals__.pop(key, None)  # type: ignore[attr-defined]


def _prefix_copy(x: torch.Tensor) -> torch.Tensor:
    """Stores from element 8 on: the first eight elements are the fill's."""
    n = x.size(0)
    out = torch.zeros([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(8, n):
        out[tile] = x[tile]
    return out


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
    ],
)
@pytest.mark.parametrize("n", [13, 1001, 4096])
@pytest.mark.parametrize("static_shapes", [True, False])
def test_every_fill_dtype_and_tail(
    dtype: torch.dtype, n: int, static_shapes: bool
) -> None:
    kernel = helion.kernel(
        _prefix_copy,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    if dtype.is_floating_point:
        x = torch.randn(n, device=DEVICE).to(dtype)
    else:
        x = torch.randint(-100, 100, (n,), device=DEVICE, dtype=dtype)
    code, out = code_and_output(kernel, (x,), block_sizes=[64], cute_pdl=True)
    assert ZERO_FILL_KERNEL_PREFIX in code
    expected = x.clone()
    expected[:8] = 0
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


_SPLIT_K = {
    "block_sizes": [32, 64, 64],
    "num_threads": [2, 64, 1],
    "cute_collective_mma": True,
    "cute_collective_compute": "warp",
    "cute_collective_copy": "async",
    "cute_collective_stages": 4,
    "cute_collective_epilogue": "vector_unrolled",
    "split_k": 16,
}


def _split_k_kernel() -> helion.Kernel:
    matmul_split_k = import_example("matmul_split_k").matmul_split_k

    return helion.kernel(
        matmul_split_k.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )


def test_the_split_k_kernel_waits_after_its_operand_loop(cpu_only: None) -> None:
    """Behind the fill kernel of its atomic output the split-K kernel waits
    late: the entry release leads the body, the K loop (operand loads, MMAs)
    runs ahead of the wait, and the wait precedes the first statement that
    touches ``out`` (the packed atomic epilogue). The fill kernel releases
    its dependents only after its own wait, so every kernel ahead of it has
    completed when this grid starts; only the fill's own writes, confined to
    ``out``, are still in flight."""
    x = torch.empty((64, 1024), dtype=torch.float16)
    y = torch.empty((1024, 128), dtype=torch.float16)
    on, off = _codes(_split_k_kernel(), (x, y), **_SPLIT_K)
    kernels = _kernel_defs(on)
    (fill_name,) = [k for k in kernels if k.startswith(ZERO_FILL_KERNEL_PREFIX)]
    body = kernels["_helion_matmul_split_k"].body
    main = [ast.unparse(s) for s in body]
    # The entry release leads the body: plain at one wave, or the CTA index
    # and the guard of ``last_wave`` past it (``cute_pdl_release`` auto).
    release = 1 if main[0] == _RELEASE else 2
    if release == 2:
        assert main[0].startswith("pdl_cta = ") and _RELEASE in main[1]
    assert main.count(_WAIT) == 1
    wait = main.index(_WAIT)
    assert wait > release

    def touches_out(statement: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id == "out"
            for node in ast.walk(statement)
        )

    # A memory-touching loop (the K loop's copies and MMAs) runs ahead of the
    # wait; nothing ahead of it touches ``out``; the next statement does.
    assert any(
        isinstance(s, ast.For) and "cute.gemm(" in ast.unparse(s) for s in body[:wait]
    )
    assert not any(touches_out(s) for s in body[:wait])
    assert touches_out(body[wait + 1])
    assert "_cute_red_add_f16_vec(" in main[wait + 1]
    assert generic_release_rendered(on)
    # Knob off: no fill kernel, no wait, the memset stays.
    assert ZERO_FILL_KERNEL_PREFIX not in off and _WAIT not in off
    assert "out = torch.zeros(" in off
    assert f"{fill_name}._helion_cute_use_pdl = True" in on


def test_the_split_k_output_is_right_on_every_call_behind_the_late_wait() -> None:
    """GPU: the late-waiting split-K kernel behind its fill kernel, run
    back to back (eager and as a CUDA graph of several calls with inputs
    that change between calls) matches the reference on every call."""
    from helion.autotuner.benchmarking import _make_cudagraph_replay

    torch.manual_seed(11)
    x = torch.randn((64, 4096), device=DEVICE, dtype=torch.float16) * 0.05
    y = torch.randn((4096, 128), device=DEVICE, dtype=torch.float16) * 0.05
    kernel = _split_k_kernel()
    bound = kernel._bind_isolated((x, y))
    config = bound.config_spec.normalized_config(
        helion.Config.from_dict({**_SPLIT_K, "cute_pdl": True})
    )
    code = bound.to_code(config)
    assert ZERO_FILL_KERNEL_PREFIX in code
    main = [ast.unparse(s) for s in _kernel_defs(code)["_helion_matmul_split_k"].body]
    assert _RELEASE in main[0] + main[1] and main.index(_WAIT) > 2
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(x, y)

    def expected() -> torch.Tensor:
        return (x.float() @ y.float()).to(torch.float16)

    torch.testing.assert_close(run(), expected(), atol=1e-2, rtol=2e-2)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        x.uniform_(-0.05, 0.05)
        y.uniform_(-0.05, 0.05)
        torch.testing.assert_close(run(), expected(), atol=1e-2, rtol=2e-2)
        torch.testing.assert_close(replay(), expected(), atol=1e-2, rtol=2e-2)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alias_and_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The zero-filled output is read through a host view (``flat``, a
    kernel argument of its own) at the cell the same loop accumulates into."""
    m, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    flat = out.view(m, n)
    for tile_m, tile_n in hl.tile([m, n]):
        seen = flat[tile_m, tile_n]
        hl.atomic_add(out, [tile_m, tile_n], y[tile_m, tile_n].to(torch.float32) + seen)
    return out


def test_every_argument_aliasing_the_filled_tensor_bounds_the_late_wait(
    cpu_only: None,
) -> None:
    """The late wait is bounded by every kernel argument that shares the
    filled tensor's storage, not only the one carrying its name: the host
    view ``flat`` of ``out`` is a separate argument and must count as a
    touch of ``out`` (it has to see the fill's zeros)."""
    from helion._compiler.cute.passes import pdl

    recorded: list[set[str]] = []
    original = pdl._late_pdl_wait_names

    def record(fn: DeviceFunction) -> set[str]:
        names = original(fn)
        recorded.append(names)
        return names

    x = torch.empty(64, dtype=torch.float32)
    y = torch.empty(64, 128, dtype=torch.float32)
    with patch.object(pdl, "_late_pdl_wait_names", record):
        on, off = _codes(
            _alias_and_atomic, (x, y), block_sizes=[16, 64], num_threads=[4, 64]
        )
    kernels = _kernel_defs(on)
    assert any(k.startswith(ZERO_FILL_KERNEL_PREFIX) for k in kernels)
    kernel = kernels["_helion__alias_and_atomic"]
    arguments = [a.arg for a in kernel.args.args]
    assert "out" in arguments and "flat" in arguments
    assert recorded and recorded[-1] == {"out", "flat"}
    # One loop touches both: nothing precedes it, the kernel waits first.
    assert ast.unparse(kernel.body[0]) == _WAIT
    assert _WAIT not in off and ZERO_FILL_KERNEL_PREFIX not in off


def test_an_opaque_descriptor_parameter_keeps_the_wait_first(cpu_only: None) -> None:
    """A kernel with a parameter the argument names cannot account for (a
    lowering's own tensor map, as the flash backward's ``_fbwd_tma_dq``
    that reduces into dq without naming it) keeps the plain wait-first
    form even where the late wait would apply otherwise."""
    from helion._compiler.device_function import DeviceFunction

    original = DeviceFunction.codegen_function_def

    def with_descriptor(self: DeviceFunction, *args: object, **kwargs: object):
        self.wrapper_only_params.append("_test_tma_desc")
        return original(self, *args, **kwargs)

    x = torch.empty((64, 1024), dtype=torch.float16)
    y = torch.empty((1024, 128), dtype=torch.float16)
    with patch.object(DeviceFunction, "codegen_function_def", with_descriptor):
        on, _off = _codes(_split_k_kernel(), (x, y), **_SPLIT_K)
    kernels = _kernel_defs(on)
    assert any(k.startswith(ZERO_FILL_KERNEL_PREFIX) for k in kernels)
    main = [ast.unparse(s) for s in kernels["_helion_matmul_split_k"].body]
    assert main[0] == _WAIT
    assert "_test_tma_desc" in [
        a.arg for a in kernels["_helion_matmul_split_k"].args.args
    ]


def _wide_prefix_copy(x: torch.Tensor) -> torch.Tensor:
    """``_prefix_copy`` into an output 65536 times longer than the input: an
    input of 32768 elements makes it exactly 2**31 elements."""
    n = x.size(0)
    out = torch.zeros([n * 65536], dtype=x.dtype, device=x.device)
    for tile in hl.tile(8, n):
        out[tile] = x[tile]
    return out


_WIDE_ROWS = MAX_FILL_NUMEL // 65536  # 32768 rows: 2**31 elements
_LAUNCH_GUARD = f"if out.numel() < {MAX_FILL_NUMEL}:"


def test_a_static_fill_of_two_to_the_31_elements_keeps_its_memset(
    cpu_only: None,
) -> None:
    # The fill kernel counts in Int32: a static element count at the bound
    # or past it declines the launch form (the memset stays, no fill
    # kernel); one element short of it launches, unguarded.
    kernel = helion.kernel(
        _wide_prefix_copy, backend="cute", static_shapes=True, autotune_effort="none"
    )
    x = torch.empty(_WIDE_ROWS, dtype=torch.int16)
    on, off = _codes(kernel, (x,), block_sizes=[64])
    assert "out = torch.zeros([n * 65536], dtype=x.dtype, device=x.device)" in on
    assert ZERO_FILL_KERNEL_PREFIX not in on
    assert _WAIT in on and _WAIT not in off  # the kernel proper is a dependent
    x = torch.empty(_WIDE_ROWS - 1, dtype=torch.int16)
    on, _ = _codes(kernel, (x,), block_sizes=[64])
    assert ZERO_FILL_KERNEL_PREFIX in on
    assert _LAUNCH_GUARD not in on


def test_a_runtime_sized_fill_launches_behind_a_bound_on_its_element_count(
    cpu_only: None,
) -> None:
    kernel = helion.kernel(
        _prefix_copy, backend="cute", static_shapes=False, autotune_effort="none"
    )
    x = torch.empty(1001, dtype=torch.bfloat16)
    on, off = _codes(kernel, (x,), block_sizes=[64])
    assert "out = torch.zeros([n], dtype=x.dtype, device=x.device)" in off
    lines = _host_lines(on, "_prefix_copy")
    allocation = lines.index("out = torch.empty([n], dtype=x.dtype, device=x.device)")
    guard = ast.parse(lines[allocation + 1]).body[0]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == _LAUNCH_GUARD[3:-1]
    (launch,) = guard.body
    assert ast.unparse(launch).startswith(
        f"_launcher({ZERO_FILL_KERNEL_PREFIX}0_helion__prefix_copy, "
    )
    assert [ast.unparse(s) for s in guard.orelse] == ["out.zero_()"]


@pytest.mark.parametrize("static_shapes", [True, False])
def test_a_fill_of_two_to_the_31_elements_reads_zero(static_shapes: bool) -> None:
    """Static: the memset stays.  Dynamic: the host bound sends the 2**31
    elements to the memset instead of the Int32 fill kernel."""
    rows = _WIDE_ROWS
    needed = 2 * rows * 65536 * 2  # the output, plus the reference's slack
    if torch.cuda.mem_get_info()[0] < needed:
        pytest.skip("needs 8 GB of free device memory")
    kernel = helion.kernel(
        _wide_prefix_copy,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    x = torch.randint(1, 100, (rows,), device=DEVICE, dtype=torch.int16)
    code, out = code_and_output(kernel, (x,), block_sizes=[64], cute_pdl=True)
    if static_shapes:
        assert ZERO_FILL_KERNEL_PREFIX not in code
    else:
        assert _LAUNCH_GUARD in code
    assert out.numel() == MAX_FILL_NUMEL
    assert torch.equal(out[:8], torch.zeros(8, device=DEVICE, dtype=torch.int16))
    assert torch.equal(out[8:rows], x[8:])
    # Past the stored prefix the fill's zeros: sample the far end.
    assert int(out[rows:].count_nonzero()) == 0
    del out
