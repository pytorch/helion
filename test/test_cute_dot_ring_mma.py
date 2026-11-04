from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])


def gemv(w: Tensor, x: Tensor) -> Tensor:
    """``w @ x`` as the scalar dot fallback: one row tile, a K loop, N = 1."""
    M, K = w.size()
    x2 = x.view(K, 1)
    out = torch.empty([M, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(w[tile_m, tile_k], x2[tile_k, :], acc=acc)
        out[tile_m, :] = acc
    return out.view(M)


def gemv_scaled(w: Tensor, x: Tensor) -> Tensor:
    """A computed row operand: the element loop is not the plain fold."""
    M, K = w.size()
    x2 = x.view(K, 1)
    out = torch.empty([M, 1], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(w[tile_m, tile_k] * 2, x2[tile_k, :], acc=acc)
        out[tile_m, :] = acc
    return out.view(M)


def gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """examples/gather_gemv.py: a gathered row index and ``acc +=``."""
    B, S, _ = w.size()
    N = idx.size(0)
    w_view = w.view(B * S, S)
    x2 = x.view(S, 1)
    out = torch.empty([N * S, 1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(N * S):
        acc = hl.zeros([tile, 1], dtype=torch.float32)
        idx_gather = idx[tile.index // S]
        for tile_k in hl.tile(S):
            gathered = w_view[idx_gather * S + tile.index % S, tile_k]
            acc += hl.dot(gathered, x2[tile_k, :])
        out[tile, :] = acc
    return out.view(N, S)


def masked_gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """A gathered row index under an extra mask: the odd rows read matrix 0."""
    B, S, _ = w.size()
    N = idx.size(0)
    w_view = w.view(B * S, S)
    x2 = x.view(S, 1)
    out = torch.empty([N * S, 1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(N * S):
        acc = hl.zeros([tile, 1], dtype=torch.float32)
        idx_gather = hl.load(idx, [tile.index // S], extra_mask=tile.index % 2 == 0)
        for tile_k in hl.tile(S):
            gathered = w_view[idx_gather * S + tile.index % S, tile_k]
            acc += hl.dot(gathered, x2[tile_k, :])
        out[tile, :] = acc
    return out.view(N, S)


def _ring_config(
    rows: int = 64,
    chunk: int = 256,
    threads: int = 128,
    stages: int = 3,
    split: str = "k",
    **extra: Any,
) -> helion.Config:
    return helion.Config(
        block_sizes=[rows, chunk],
        num_threads=[1, threads],
        cute_vector_widths=[1, max(2, min(8, chunk // threads))],
        cute_dot_ring_mma=True,
        cute_dot_ring_stages=stages,
        cute_dot_ring_split=split,
        **extra,
    )


def _cpu_code(
    fn: Callable[..., object], args: tuple[object, ...], config: helion.Config
) -> str:
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        kernel = helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        bound = cpu_bind(kernel, args)
        return bound.to_code(config)


def _cpu_spec(fn: Callable[..., object], args: tuple[object, ...]) -> Any:
    with mock_cuda_unavailable():
        kernel = helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        return cpu_bind(kernel, args).config_spec


@pytest.mark.parametrize(
    ("dtype", "helper"),
    [(torch.float16, "_cute_mma_f16"), (torch.bfloat16, "_cute_mma_bf16")],
)
def test_gemv_dot_streams_through_the_ring(dtype: torch.dtype, helper: str) -> None:
    code = _cpu_code(
        gemv,
        (torch.empty(256, 4096, dtype=dtype), torch.empty(4096, dtype=dtype)),
        _ring_config(),
    )
    # 64 rows x 256 K per chunk, 3 ring slots of row tile + vector chunk, the
    # four warps' partial sums: (64 + 1) * 512 * 3 + 4 * 64 * 4 bytes.
    assert "cute.arch.alloc_smem(cutlass.Uint8, 100864, alignment=128)" in code
    # The row tile moves by cp.async (16 packets per thread per chunk), the
    # ring waits with STAGES - 2 groups outstanding, and the warps consume it
    # with ldmatrix + mma.sync; no element converts, no shuffle sums.
    assert "_cute_ring_cp_async_16(" in code
    assert "cute.arch.cp_async_wait_group(1)" in code
    assert "_cute_ring_ldmatrix_x4(" in code
    assert f"{helper}(" in code
    assert "_cute_ring_sts_f32(" in code  # the K split's cross-warp sum
    assert "_cute_grouped_reduce" not in code
    assert "warp_reduction" not in code
    assert "bitcast(cutlass.Float16)" not in code
    assert "bitcast(cutlass.BFloat16)" not in code
    assert "_helion_vfold_acc" not in code
    # K = 4096 is 16 whole chunks: no zero-filled tail packets.
    assert "_cute_ring_cp_async_16_zfill(" not in code


def test_pdl_release_follows_the_last_mma() -> None:
    code = _cpu_code(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
        _ring_config(cute_pdl=True),
    )
    # The generic entry release is dropped; the ring releases its dependents
    # once after its last MMA (released at entry, the dependents' CTAs take
    # the slots the ring leaves free: 55 us instead of 46 at 7 CTAs per SM).
    assert code.count("cute.arch.griddepcontrol_wait()") == 1
    assert code.count("cute.arch.griddepcontrol_launch_dependents()") == 1
    assert code.rindex("_cute_mma_f16(") < code.index(
        "cute.arch.griddepcontrol_launch_dependents()"
    )
    assert code.index("cute.arch.griddepcontrol_launch_dependents()") < code.index(
        ".store("
    )


def test_m_split_puts_the_warps_on_row_atoms() -> None:
    code = _cpu_code(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
        _ring_config(rows=64, chunk=256, threads=64, stages=3, split="m"),
    )
    assert "_cute_ring_ldmatrix_x4(" in code
    assert "_cute_mma_f16(" in code
    # No partial sums to combine: each warp owns whole rows.
    assert "_cute_ring_sts_f32(" not in code
    assert "cute.arch.alloc_smem(cutlass.Uint8, 99840, alignment=128)" in code


def test_k_tail_is_zero_filled() -> None:
    code = _cpu_code(
        gemv,
        (
            torch.empty(256, 4160, dtype=torch.float16),
            torch.empty(4160, dtype=torch.float16),
        ),
        _ring_config(),
    )
    # 4160 = 16 chunks of 256 plus 64: the last chunk's packets past K are
    # zero-filled through cp.async's src-size (on both operands).
    assert "_cute_ring_cp_async_16_zfill(" in code
    assert "_cute_mma_f16(" in code


def test_gathered_row_operand_takes_the_ring() -> None:
    code = _cpu_code(
        gather_gemv,
        (
            torch.empty(8, 2048, 2048, dtype=torch.float16),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2048, dtype=torch.float16),
        ),
        _ring_config(),
    )
    assert "_cute_mma_f16(" in code
    assert "_cute_ring_cp_async_16(" in code
    # The gathered row index is re-instantiated per packet row of the thread
    # (16 rows: 64 rows x 32 packets over 128 threads) and per stored row,
    # with the bounds-masked gather turned into an address select (no branch).
    assert code.count("idx.iterator") >= 16
    assert "else idx.iterator + cutlass.Int32(0)" in code
    assert ").load() if" not in code
    # A row past the tile reads row 0 instead (never stored).
    assert "else cutlass.Int32(0)" in code


def test_masked_gather_keeps_its_value_select() -> None:
    code = _cpu_code(
        masked_gather_gemv,
        (
            torch.empty(8, 2048, 2048, dtype=torch.float16),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2048, dtype=torch.float16),
        ),
        _ring_config(),
    )
    assert "_cute_mma_f16(" in code
    # The masked gather loads unconditionally from a selected address (the
    # tensor base when masked) and keeps the lowering's value select, so an
    # in-bounds row under the extra mask still reads 0, as it did.
    assert ").load() if" not in code
    assert (
        "else idx.iterator + cutlass.Int32(0) * cutlass.Int32(idx.layout.stride[0])).load()"
        in code
    )
    assert "= _ring_gather" in code
    assert code.count("else cutlass.Int32(0)") >= 16


def test_knob_needs_one_k_packet_per_thread() -> None:
    spec = _cpu_spec(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
    )
    # A 512 chunk on 128 threads with 2-element packets is two packets per
    # thread (a blocked lane base the matcher declines): the knob is inert,
    # so the search does not carry a duplicate identity of the knob-off code.
    two_packets = _ring_config(rows=64, chunk=512, threads=128)
    two_packets.config["cute_vector_widths"] = [1, 2]
    assert spec.normalized_config(two_packets).config["cute_dot_ring_mma"] is False
    one_packet = _ring_config(rows=64, chunk=512, threads=128)
    one_packet.config["cute_vector_widths"] = [1, 4]
    assert spec.normalized_config(one_packet).config["cute_dot_ring_mma"] is True


def test_pdl_release_is_pinned_under_the_ring() -> None:
    spec = _cpu_spec(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
    )
    normalized = spec.normalized_config(
        _ring_config(cute_pdl=True, cute_pdl_release="entry")
    ).config
    assert normalized["cute_dot_ring_mma"] is True
    # The ring places the release itself (after its last MMA): the key is inert.
    assert normalized["cute_pdl_release"] == "auto"


def test_computed_row_operand_regenerates_knob_off() -> None:
    code = _cpu_code(
        gemv_scaled,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
        _ring_config(),
    )
    # The element loop scales the operand first: not the plain fold, so the
    # kernel is generated again with the knob off and keeps the reduction.
    assert "_cute_mma_f16(" not in code
    assert "_cute_ring_cp_async_16(" not in code
    assert "_cute_grouped_reduce" in code or "warp_reduction" in code


def test_knob_needs_rolled_rows_and_whole_warps() -> None:
    spec = _cpu_spec(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
    )
    on = spec.normalized_config(_ring_config()).config
    assert on["cute_dot_ring_mma"] is True
    assert on["cute_dot_ring_stages"] == 3
    assert on["cute_dot_ring_split"] == "k"
    # Threads on the rows (the fragment layout), a scalar K width or fewer
    # than 16 rows cannot take the ring: the knob is inert.
    off = _ring_config()
    off.config["num_threads"] = [8, 4]
    assert spec.normalized_config(off).config["cute_dot_ring_mma"] is False
    off = _ring_config()
    off.config["cute_vector_widths"] = [1, 1]
    assert spec.normalized_config(off).config["cute_dot_ring_mma"] is False
    off = _ring_config(rows=8)
    assert spec.normalized_config(off).config["cute_dot_ring_mma"] is False
    # Inert knobs pin their companions to the defaults.
    assert spec.normalized_config(off).config["cute_dot_ring_stages"] == 3
    assert spec.normalized_config(off).config["cute_dot_ring_split"] == "k"


def test_stage_and_split_knobs_are_repaired() -> None:
    spec = _cpu_spec(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float16),
            torch.empty(4096, dtype=torch.float16),
        ),
    )
    # 128 x 256 x 8 stages is 528 KB: the ring shrinks to the 3 stages that fit.
    repaired = spec.normalized_config(
        _ring_config(rows=128, chunk=256, threads=128, stages=8, split="m")
    ).config
    assert repaired["cute_dot_ring_mma"] is True
    assert repaired["cute_dot_ring_stages"] == 3
    # 32 rows are two atoms: four warps cannot split them, the K steps can.
    repaired = spec.normalized_config(
        _ring_config(rows=32, chunk=256, threads=128, stages=4, split="m")
    ).config
    assert repaired["cute_dot_ring_mma"] is True
    assert repaired["cute_dot_ring_split"] == "k"
    with pytest.raises(helion.exc.InvalidConfig):
        spec.normalized_config(_ring_config(stages=5))


def test_fp32_dot_has_no_ring_knob() -> None:
    spec = _cpu_spec(
        gemv,
        (
            torch.empty(256, 4096, dtype=torch.float32),
            torch.empty(4096, dtype=torch.float32),
        ),
    )
    with pytest.raises(helion.exc.InvalidConfig):
        spec.normalized_config(_ring_config())
    assert "cute_dot_ring_mma" not in spec._flat_fields()


# ---- the decline contract, on the generic kernel text itself: the pass must
# refuse (and leave the body untouched) whenever the lowering it replaces is
# not the plain streaming dot.


def _generic_gather_text() -> str:
    """The knob-off code of ``gather_gemv`` (rolled rows, a K loop, the K lane
    loop with the plain lane base and the two masked packet loads)."""
    config = _ring_config()
    config.config["cute_dot_ring_mma"] = False
    return _cpu_code(
        gather_gemv,
        (
            torch.empty(8, 2048, 2048, dtype=torch.float16),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2048, dtype=torch.float16),
        ),
        config,
    )


def _run_ring_pass(text: str) -> tuple[bool, bool]:
    """``(fired, untouched)`` of the ring pass on the kernel of ``text``."""
    from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
    from helion._compiler.cute.ring_mma_dot import ring_mma_dot

    module = ast.parse(text)
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    for node in ast.walk(kernel):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("lane_")
        ):
            setattr(node, HELION_LANE_LOOP_VAR_ATTR, node.target.id)
    before = "\n".join(ast.unparse(stmt) for stmt in kernel.body)
    s = 2048
    shapes = {
        "w_view": torch.empty(8 * s, s, dtype=torch.float16),
        "x2": torch.empty(s, dtype=torch.float16).view(s, 1),
        "idx": torch.empty(2, dtype=torch.int32),
        "out": torch.empty(2 * s, 1, dtype=torch.float16),
    }
    tensors = {arg.arg: shapes[arg.arg] for arg in kernel.args.args}
    counter = [0]

    def new_var(name: str) -> str:
        counter[0] += 1
        return f"{name}_{counter[0]}"

    body, fired = ring_mma_dot(
        kernel.body,
        thread_block_dims=(128, 1, 1),
        rows=64,
        chunk=256,
        stages=3,
        split="k",
        k_block_var="_BLOCK_SIZE_1",
        tensors=tensors,
        aligned=set(tensors),
        new_var=new_var,
    )
    # Untouched: the returned statements are the kernel's own, and the
    # kernel's lists were not spliced (a declined pass must not mutate).
    after = "\n".join(ast.unparse(stmt) for stmt in body)
    own = "\n".join(ast.unparse(stmt) for stmt in kernel.body)
    return fired, after == before and own == before


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new, 1)


def test_generic_text_fixture_takes_the_ring() -> None:
    fired, untouched = _run_ring_pass(_generic_gather_text())
    assert fired and not untouched


def test_declines_a_shifted_k_coordinate() -> None:
    # The vector packet reads K shifted by 8: not the streaming order.
    text = _replace_once(
        _generic_gather_text(),
        "cutlass.Int32(lane_base_1) * cutlass.Int32(x2.layout.stride[0])",
        "cutlass.Int32(lane_base_1 + 8) * cutlass.Int32(x2.layout.stride[0])",
    )
    assert _run_ring_pass(text) == (False, True)


def test_declines_a_scaled_lane_base() -> None:
    # Every other K element: the lane base is not the plain K position.
    text = _replace_once(
        _generic_gather_text(),
        "lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 2 + cutlass.Int32(lane_1) * 2",
        "lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 4 + cutlass.Int32(lane_1) * 2",
    )
    assert _run_ring_pass(text) == (False, True)


def test_declines_a_k_mask_that_is_not_the_loop_extent() -> None:
    # The vector packet is masked at 1024 while the K loop runs to 2048: the
    # replaced lowering zeroed half of K, the ring would not.
    text = _generic_gather_text()
    vector_line = next(line for line in text.split("\n") if "x2.iterator" in line)
    assert "lane_base_1 < 2048" in vector_line
    text = text.replace(
        vector_line, vector_line.replace("lane_base_1 < 2048", "lane_base_1 < 1024")
    )
    assert _run_ring_pass(text) == (False, True)


def test_declines_a_store_reading_a_lane_body_value() -> None:
    text = _replace_once(
        _generic_gather_text(),
        "cutlass.Float16(acc)",
        "cutlass.Float16(acc + cutlass.Float32(load_1))",
    )
    assert _run_ring_pass(text) == (False, True)


def test_declines_two_candidate_row_loops_untouched() -> None:
    text = _generic_gather_text()
    module = ast.parse(text)
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    row_loop = next(
        node
        for node in kernel.body
        if isinstance(node, ast.For) and ast.unparse(node.target) == "lane_0"
    )
    kernel.body.append(ast.parse(ast.unparse(row_loop)).body[0])
    assert _run_ring_pass(ast.unparse(module)) == (False, True)


_needs_mma = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="mma.sync m16n8k16 and cp.async need compute capability 8.0",
)


@_needs_mma
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "chunk", "threads", "stages", "split"),
    [(64, 256, 128, 3, "k"), (32, 128, 64, 4, "k"), (64, 256, 64, 3, "m")],
)
def test_gemv_ring_matches_torch(
    dtype: torch.dtype, rows: int, chunk: int, threads: int, stages: int, split: str
) -> None:
    torch.manual_seed(0)
    w = torch.randn(256, 4096, device=DEVICE, dtype=dtype)
    x = torch.randn(4096, device=DEVICE, dtype=dtype)
    kernel = helion.kernel(
        gemv,
        backend="cute",
        static_shapes=True,
        config=_ring_config(rows, chunk, threads, stages, split),
    )
    code = kernel.bind((w, x)).to_triton_code(kernel.configs[0])
    assert "_cute_ring_cp_async_16(" in code
    out = kernel(w, x)
    expected = (w.float() @ x.float()).to(dtype)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-1)


@_needs_mma
@pytest.mark.parametrize(
    ("m", "k"),
    [(256, 4160), (100, 4096), (48, 1000), (17, 1000), (1, 1000), (48, 1004)],
)
def test_gemv_ring_ragged_extents_match_torch(m: int, k: int) -> None:
    """K past the last whole chunk (zero-filled packets, a partial MMA step),
    rows past the last whole tile (clamped rows, guarded stores); a tile under
    16 rows cannot form an MMA atom and a row stride that is not 16-byte
    aligned cannot be a cp.async packet: the knob is inert and the fallback
    runs."""
    torch.manual_seed(0)
    w = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    x = torch.randn(k, device=DEVICE, dtype=torch.float16)
    kernel = helion.kernel(
        gemv,
        backend="cute",
        static_shapes=True,
        config=_ring_config(rows=32, chunk=256, threads=128, stages=3, split="k"),
    )
    code = kernel.bind((w, x)).to_triton_code(kernel.configs[0])
    assert ("_cute_ring_cp_async_16" in code) == (m >= 16 and (k * 2) % 16 == 0)
    out = kernel(w, x)
    expected = (w.float() @ x.float()).to(torch.float16)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-1)


@_needs_mma
def test_masked_gather_gemv_ring_matches_torch() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 2048, 2048, device=DEVICE, dtype=torch.float16)
    idx = torch.randint(1, 8, (2,), device=DEVICE, dtype=torch.int32)
    x = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    kernel = helion.kernel(
        masked_gather_gemv,
        backend="cute",
        static_shapes=True,
        config=_ring_config(rows=64, chunk=256, threads=128, stages=3, split="k"),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    code = kernel.bind((w, idx, x)).to_triton_code(kernel.configs[0])
    assert "_cute_ring_cp_async_16(" in code
    out = kernel(w, idx, x)
    rows = torch.arange(2 * 2048, device=DEVICE)
    gathered = torch.where(rows % 2 == 0, idx[rows // 2048].long(), 0)
    expected = (
        w.view(8 * 2048, 2048)[gathered * 2048 + rows % 2048].float() @ x.float()
    ).to(torch.float16)
    torch.testing.assert_close(out.view(-1), expected, rtol=1e-2, atol=1e-1)


@_needs_mma
def test_gather_gemv_ring_matches_torch() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 2048, 2048, device=DEVICE, dtype=torch.float16)
    idx = torch.randint(0, 8, (2,), device=DEVICE, dtype=torch.int32)
    x = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    kernel = helion.kernel(
        gather_gemv,
        backend="cute",
        static_shapes=True,
        config=_ring_config(rows=64, chunk=256, threads=128, stages=3, split="k"),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    out = kernel(w, idx, x)
    expected = (w[idx].float() @ x.float()).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)
