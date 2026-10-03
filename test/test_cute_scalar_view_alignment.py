from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import launcher

B, N, C, D = 4, 3, 64, 128

requires_cute = skipUnlessBackends(["cute"])
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _decayed_rows_kernel() -> helion.Kernel:
    # Mirrors the gated-delta state pass: a per-chunk scalar decay read from
    # ``decay_last = g_cs[:, :, -1]``, an fp32 view only 4-byte aligned.
    @helion.kernel(backend="cute", autotune_effort="none")
    def decayed_rows(x: torch.Tensor, decay_last: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile_b, tile_n in hl.tile([x.size(0), x.size(1)], block_size=[1, 1]):
            scale = torch.exp(decay_last[tile_b, tile_n])
            out[tile_b, tile_n, :] = (x[tile_b, tile_n, :] * scale[:, :, None]).to(
                out.dtype
            )
        return out

    return decayed_rows


def _inputs(seed: int, batch: int = B) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(batch, N, D, device=DEVICE, generator=generator).bfloat16()
    g = -torch.rand(batch, N, C, device=DEVICE, generator=generator) * 0.05
    return x, g.cumsum(-1)


def _placed(values: torch.Tensor, offset: int) -> torch.Tensor:
    """``values`` copied into a view with ``g_cs[:, :, -1]`` strides at ``offset``."""
    batch = values.size(0)
    storage = torch.zeros(batch * N * C + 4, device=DEVICE)
    view = storage.as_strided((batch, N), (N * C, C), offset)
    view.copy_(values)
    return view


def _expected(x: torch.Tensor, decay_last: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.exp(decay_last.float())[:, :, None]).bfloat16()


def _cute_kernel(kernel: helion.Kernel, args: tuple[object, ...]) -> Any:
    bound = kernel.bind(args)
    compiled = bound._compile_cache[bound._require_implicit_config()]
    return compiled.__globals__[f"_helion_{bound.kernel.name}"]


def _param_index(cute_kernel: Any, name: str) -> int:
    return list(inspect.signature(inspect.unwrap(cute_kernel)).parameters).index(name)


def _compiled_alignments(cute_kernel: Any, name: str) -> set[int]:
    """Pointer alignment of argument ``name`` in every compiled launcher."""
    index = _param_index(cute_kernel, name)
    return {
        launcher._cute_schema_pointer_alignment(key[0][index])
        for key in cute_kernel._helion_cute_compiled_launchers
    }


@requires_cute
@requires_cuda
def test_scalar_route_accepts_element_aligned_view() -> None:
    kernel = _decayed_rows_kernel()
    x, g_cs = _inputs(0)
    decay_last = g_cs[:, :, -1]
    assert decay_last.data_ptr() % 16 == 12
    code, out = code_and_output(kernel, (x, decay_last))
    # The ordinary scalar route has no kernel-wide alignment override or
    # wrapper plan; the launcher binds the view at its actual alignment.
    assert "ptp_thread" not in code
    assert "_helion_cute_pointer_alignment" not in code
    assert "_helion_cute_wrapper_plans" not in code
    torch.testing.assert_close(out, _expected(x, decay_last))
    cute_kernel = _cute_kernel(kernel, (x, decay_last))
    assert _compiled_alignments(cute_kernel, "decay_last") == {4}
    aligned = _placed(decay_last, 0)
    torch.testing.assert_close(out, kernel(x, aligned), rtol=0, atol=0)
    # Same generated source: both launches share one CuTe kernel and its caches.
    assert _cute_kernel(kernel, (x, aligned)) is cute_kernel


@requires_cute
@requires_cuda
def test_low_alignment_fastpath_serves_stronger_views() -> None:
    kernel = _decayed_rows_kernel()
    # A batch size of its own gives this test a distinct generated module, so
    # no other test has touched the shared CuTe kernel caches.
    x, g_cs = _inputs(1, batch=5)
    views = {4: g_cs[:, :, -1], 8: g_cs[:, :, -2]}
    assert {view.data_ptr() % 16 for view in views.values()} == {12, 8}
    torch.testing.assert_close(kernel(x, views[4]), _expected(x, views[4]))
    cute_kernel = _cute_kernel(kernel, (x, views[4]))
    fastpath = cute_kernel._helion_cute_fastpath
    assert isinstance(fastpath, launcher._CuteFastRelaunch)
    guards = {guard[0]: guard[-1] for guard in fastpath.tensor_guards}
    assert guards[_param_index(cute_kernel, "decay_last")] == 4
    assert guards[_param_index(cute_kernel, "x")] == 16
    # A weaker compiled pointer type is valid for every stronger address.
    for decay_last in (views[8], _placed(views[4], 0), views[4]):
        torch.testing.assert_close(kernel(x, decay_last), _expected(x, decay_last))
    assert _compiled_alignments(cute_kernel, "decay_last") == {4}


@requires_cute
@requires_cuda
def test_aligned_fastpath_specializes_each_weaker_view() -> None:
    kernel = _decayed_rows_kernel()
    x, g_cs = _inputs(4, batch=6)
    views = {4: g_cs[:, :, -1], 8: g_cs[:, :, -2]}
    aligned = _placed(views[4], 0)
    torch.testing.assert_close(kernel(x, aligned), _expected(x, aligned))
    cute_kernel = _cute_kernel(kernel, (x, aligned))
    fastpath = cute_kernel._helion_cute_fastpath
    assert isinstance(fastpath, launcher._CuteFastRelaunch)
    # Weaker addresses miss the aligned fast path before any pointer is patched
    # and compile their own launcher; the aligned fast path is kept.
    for decay_last in (views[4], views[8], views[4]):
        torch.testing.assert_close(kernel(x, decay_last), _expected(x, decay_last))
    assert cute_kernel._helion_cute_fastpath is fastpath
    assert _compiled_alignments(cute_kernel, "decay_last") == {16, 4, 8}


@requires_cute
@requires_cuda
def test_cuda_graph_replays_misaligned_view() -> None:
    kernel = _decayed_rows_kernel()
    x, g_cs = _inputs(2)
    decay_last = g_cs[:, :, -1]
    kernel(x, decay_last)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = kernel(x, decay_last)
    torch.cuda.current_stream().wait_stream(stream)
    g_cs.copy_(_inputs(3)[1])
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, _expected(x, decay_last))
    torch.testing.assert_close(
        captured, kernel(x, _placed(decay_last, 0)), rtol=0, atol=0
    )
