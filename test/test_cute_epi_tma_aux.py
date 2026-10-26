"""``tcgen05_aux_load_mode="epi_tma"``: exact-shape auxiliary tiles staged through
the per-subtile SMEM aux rings by the epilogue warps themselves.

CPU renders (fake sm100) pin the structure -- the first ring stages are
issued before the accumulator wait, every later stage one subtile ahead of its
read, no producer warp -- and the SMEM model / normalizer that keeps the AB
ring deep beside the rings. The GPU cases check numerics against torch on the
one-CTA and the two-CTA pair families.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_aux import _cpu_codegen

import helion
from helion import exc
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._compiler.cute.pipeline_smem import max_pipeline_ab_stages
from helion._compiler.cute.pipeline_smem import pipeline_smem_bytes
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_EPI_TMA
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

MODE = TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
EPI_TMA = TCGEN05_AUX_LOAD_MODE_EPI_TMA


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gated_gemm(
    a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Two exact-shape auxiliary tiles: ``c = g * d + a @ b.T``."""
    m, k = a.size()
    n = b.size(0)
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_n, tile_k].T)
        c[tile_m, tile_n] = g[tile_m, tile_n] * d[tile_m, tile_n] + acc.to(a.dtype)
    return c


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_gemm(a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """One exact-shape auxiliary tile: ``c = a @ b + r``."""
    m, k = a.size()
    n = b.size(1)
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        c[tile_m, tile_n] = (acc + r[tile_m, tile_n].to(torch.float32)).to(a.dtype)
    return c


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = b.size(1)
    c = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        c[tile_m, tile_n] = acc.to(a.dtype)
    return c


def _gated_args(m: int = 4096, n: int = 1024, k: int = 512) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=torch.float16),
        torch.empty((n, k), dtype=torch.float16),
        torch.empty((m, n), dtype=torch.float16),
        torch.empty((m, n), dtype=torch.float16),
    )


def _one_cta(**overrides: object) -> dict[str, object]:
    return {
        "block_sizes": [128, 128, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 4,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        MODE: EPI_TMA,
        **overrides,
    }


def _pair(**overrides: object) -> dict[str, object]:
    return {
        "block_sizes": [256, 128, 64],
        "pid_type": "persistent_blocked",
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 6,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        MODE: EPI_TMA,
        **overrides,
    }


def _producer_warp_tma(**overrides: object) -> dict[str, object]:
    """The pre-existing producer-warp ring (``tma``): with_scheduler + C-input warp."""
    return {
        "block_sizes": [256, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cta_group": "auto",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 1,
        "tcgen05_c_stages": 4,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_warp_spec_c_input_warps": 1,
        "tcgen05_persistence_model": "static_persistent",
        MODE: TCGEN05_AUX_LOAD_MODE_TMA,
        **overrides,
    }


def _illegal_aux_views(
    m: int, n: int, *, device: torch.device | str = "cpu", randn: bool = False
) -> dict[str, torch.Tensor]:
    """Rank-2 (m, n) fp16 views no row-major TensorMap may address."""

    def make(*shape: int) -> torch.Tensor:
        if randn:
            return torch.randn(*shape, device=device, dtype=torch.float16)
        return torch.empty(shape, device=device, dtype=torch.float16)

    return {
        # A 2-byte base offset (the row stride stays a 16-byte multiple).
        "offset_base": make(m, n + 8)[:, 1 : n + 1],
        # No unit stride.
        "inner_stride_2": make(m, 2 * n)[:, ::2],
        # Aligned, but the contiguous axis is the first one.
        "column_major": make(n, m).T,
        # Unit last stride, row stride (n + 4) * 2 bytes off a 16-byte multiple.
        "row_stride_off_16": make(m, n + 4)[:, :n],
    }


def _render(
    kernel: object, args: tuple[torch.Tensor, ...], config: dict[str, object]
) -> str:
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(helion.Config.from_dict(config))  # type: ignore[attr-defined]


def _normalize(
    kernel: object,
    args: tuple[torch.Tensor, ...],
    config: dict[str, object],
    *,
    fix: bool,
) -> dict[str, object]:
    normalized = dict(config)
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)  # type: ignore[attr-defined]
        with bound.env:
            bound.config_spec.normalize(normalized, _fix_invalid=fix)
    return normalized


def _epilogue(code: str) -> str:
    """The epilogue role's text: from the first store-side setup line on."""
    start = code.index("tcgen05_aux_gmem_tile_0 = cute.local_tile(")
    return code[start:]


def test_one_cta_render_prefetches_before_the_accumulator_wait() -> None:
    code = _render(_gated_gemm, _gated_args(), _one_cta())
    # The ring and its TMA pipeline, sized by the epilogue tile, two rings.
    assert (
        "tcgen05_aux_pipeline = cutlass.pipeline.PipelineTmaAsync.create(num_stages=2"
        in code
    )
    assert "tcgen05_aux_smem_layout_1 = " in code
    # No producer warp: the epilogue warps own the ring.
    assert "tcgen05_c_input_warp_valid" not in code
    assert "tcgen05_warp_spec_c_input_warps" not in code
    epilogue = _epilogue(code)
    prefetch = epilogue.index(
        "for _tcgen05_aux_prefetch in cutlass.range(min(2, tcgen05_subtile_count), unroll_full=True):"
    )
    acc_wait = epilogue.index(
        "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
    )
    consumer = epilogue.index(
        "tcgen05_aux_pipeline.consumer_wait(tcgen05_aux_pipeline_consumer_state)"
    )
    release = epilogue.index(
        "tcgen05_aux_pipeline.consumer_release(tcgen05_aux_pipeline_consumer_state)"
    )
    refill = epilogue.index("if _tcgen05_subtile + 2 < tcgen05_subtile_count:")
    assert prefetch < acc_wait < consumer < release < refill
    # Warp 0 issues both rings' loads of the same subtile into one stage.
    assert (
        epilogue.count(
            "tcgen05_aux_pipeline.producer_acquire(tcgen05_aux_pipeline_producer_state)"
        )
        == 2
    )
    assert (
        "tcgen05_aux_tma_gmem_part_1[None, cutlass.Int32(_tcgen05_subtile + 2)]"
        in epilogue
    )
    assert (
        "tcgen05_aux_tma_gmem_part_0[None, cutlass.Int32(_tcgen05_aux_prefetch)]"
        in epilogue
    )
    assert (
        "(128, 128), (tile_offset_0 // cutlass.Int32(128), tile_offset_1 // cutlass.Int32(128))"
        in epilogue
    )
    assert code.count("'kind': 'tcgen05_aux_tma'") == 2


def test_pair_render_loads_the_per_cta_half_of_the_tile() -> None:
    code = _render(_gated_gemm, _gated_args(), _pair())
    epilogue = _epilogue(code)
    assert "CtaGroup.TWO" in code
    assert (
        "(128, 128), (tile_offset_0 // cutlass.Int32(256) * cutlass.Int32(2) + "
        "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) % cutlass.Int32(2), "
        "tile_offset_1 // cutlass.Int32(128))"
    ) in epilogue
    prefetch = epilogue.index("for _tcgen05_aux_prefetch in cutlass.range(")
    acc_wait = epilogue.index(
        "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
    )
    assert prefetch < acc_wait


def test_pre_acc_wait_placement_reads_the_ring_before_the_accumulator() -> None:
    code = _render(
        _gated_gemm, _gated_args(), _pair(tcgen05_aux_load_placement="pre_acc_wait")
    )
    epilogue = _epilogue(code)
    consumer = epilogue.index("tcgen05_aux_pipeline.consumer_wait(")
    acc_wait = epilogue.index(
        "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
    )
    assert consumer < acc_wait


def test_single_ring_residual_renders_one_tma_plan() -> None:
    args = (
        torch.empty((4096, 512), dtype=torch.float16),
        torch.empty((512, 1024), dtype=torch.float16),
        torch.empty((4096, 1024), dtype=torch.float16),
    )
    code = _render(_residual_gemm, args, _pair())
    assert code.count("'kind': 'tcgen05_aux_tma'") == 1
    assert "tcgen05_aux_smem_layout_1" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _gated_pair(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``examples/se_block.py``'s forward: two outputs, ``out`` reads ``x``."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        x_tile = x[tile_m, tile_n].to(torch.float32)
        out[tile_m, tile_n] = (2.0 * x_tile * sigmoid_result).to(x.dtype)
    return out, s


def test_smem_model_prices_the_rings() -> None:
    facts = Tcgen05PipelineSmemFacts(2, 2, 232448, exact_aux_count=2)
    kwargs = {"bm": 256, "bn": 128, "bk": 64, "c_stages": 2, "acc_stages": 2}
    simt = pipeline_smem_bytes(facts, ab_stages=6, **kwargs)
    staged = pipeline_smem_bytes(facts, ab_stages=6, aux_load_mode=EPI_TMA, **kwargs)
    # Two rings of two (128, 32) fp16 subtiles (8 KiB each) and their barriers.
    assert staged - simt == 2 * 2 * 8192 + 2 * 2 * 8
    assert max_pipeline_ab_stages(facts, **kwargs) == 8
    assert max_pipeline_ab_stages(facts, aux_load_mode=EPI_TMA, **kwargs) == 7
    assert (
        max_pipeline_ab_stages(facts, aux_load_mode=EPI_TMA, aux_stages=4, **kwargs)
        == 6
    )
    wide = {"bm": 256, "bn": 256, "bk": 64, "c_stages": 2, "acc_stages": 2}
    assert max_pipeline_ab_stages(facts, **wide) == 6
    assert max_pipeline_ab_stages(facts, aux_load_mode=EPI_TMA, **wide) == 4
    # A single ring on the same tile leaves one more stage.
    single = Tcgen05PipelineSmemFacts(2, 2, 232448, exact_aux_count=1)
    assert max_pipeline_ab_stages(single, aux_load_mode=EPI_TMA, **wide) == 5


def test_normalizer_keeps_the_ring_at_the_deepest_fitting_depth() -> None:
    args = _gated_args()
    # The paired pipeline's complete proof clamps the sampled depth ...
    fixed = _normalize(_gated_gemm, args, _pair(tcgen05_ab_stages=8), fix=True)
    assert fixed["tcgen05_ab_stages"] == 7 and fixed[MODE] == EPI_TMA
    # ... and refuses an explicit one with the limit.
    with pytest.raises(exc.InvalidConfig, match="maximum AB stages: 7"):
        _normalize(_gated_gemm, args, _pair(tcgen05_ab_stages=8), fix=False)
    assert (
        _normalize(_gated_gemm, args, _pair(tcgen05_ab_stages=7), fix=False)[MODE]
        == EPI_TMA
    )
    # The one-CTA arena model does the same for the 128x256 tile (48 KiB per
    # AB stage, 32 KiB of C ring, 64 KiB of aux rings).
    one = _one_cta(block_sizes=[128, 256, 64], tcgen05_ab_stages=4)
    fixed = _normalize(_gated_gemm, args, one, fix=True)
    assert fixed["tcgen05_ab_stages"] == 2 and fixed[MODE] == EPI_TMA
    with pytest.raises(exc.InvalidConfig, match="at most 2 AB stages"):
        _normalize(_gated_gemm, args, one, fix=False)
    # The register hoist is a SIMT-only knob: dropped beside the ring.
    fixed = _normalize(
        _gated_gemm,
        args,
        _pair(
            tcgen05_aux_load_placement="pre_acc_wait", tcgen05_aux_fragment_hoist=True
        ),
        fix=True,
    )
    assert fixed[MODE] == EPI_TMA and fixed["tcgen05_aux_fragment_hoist"] is False
    # The ring replaces the producer warp: a sample carrying both keeps the
    # ring and drops the warp (ahead of the aux-TMA full-tile projection,
    # which would otherwise take a repaired SIMT pair sample onto the
    # producer-warp ``tma`` regime); an explicit config is refused.
    for key in ("tcgen05_warp_spec_c_input_warps", "tcgen05_warp_spec_store_warps"):
        fixed = _normalize(_gated_gemm, args, _pair(**{key: 1}), fix=True)
        assert fixed[MODE] == EPI_TMA and fixed[key] == 0
        assert fixed["tcgen05_ab_stages"] == 6
        fixed = _normalize(_gated_gemm, args, _one_cta(**{key: 1}), fix=True)
        assert fixed[MODE] == EPI_TMA and fixed[key] == 0
        with pytest.raises(exc.InvalidConfig, match="replaces the aux producer warp"):
            _normalize(_gated_gemm, args, _one_cta(**{key: 1}), fix=False)


def test_normalizer_refuses_the_ring_without_an_exact_shape_operand() -> None:
    args = (
        torch.empty((4096, 512), dtype=torch.float16),
        torch.empty((512, 1024), dtype=torch.float16),
    )
    with pytest.raises(exc.InvalidConfig, match="exact-shape rank-2 auxiliary"):
        _normalize(_plain_gemm, args, _one_cta(), fix=False)
    fixed = _normalize(_plain_gemm, args, _one_cta(), fix=True)
    assert fixed[MODE] == TCGEN05_AUX_LOAD_MODE_SIMT
    # Deeper ring depths are priced only by the epilogue-issued ring.
    gated = _gated_args()
    with pytest.raises(exc.InvalidConfig, match="only available with"):
        _normalize(
            _gated_gemm,
            gated,
            _pair(tcgen05_aux_stages=4, **{MODE: TCGEN05_AUX_LOAD_MODE_SIMT}),
            fix=False,
        )
    assert (
        _normalize(_gated_gemm, gated, _pair(tcgen05_aux_stages=4), fix=False)[
            "tcgen05_aux_stages"
        ]
        == 4
    )


def test_search_surface_and_seeds() -> None:
    with _cpu_codegen():
        bound = _gated_gemm._bind_isolated(_gated_args())
        state = bound.config_spec._cute_tcgen05_config
        facts = state.pipeline_smem_facts
        assert facts is not None and facts.exact_aux_count == 2
        assert state.rowvec_aux_facts is not None
        assert state.rowvec_aux_facts.exact_aux_count == 2
        assert state.exact_aux_tma_provable
        assert state.epi_tma_search_enabled()
        fragment = state.aux_load_mode_autotune_fragments()[MODE]
        assert EPI_TMA in fragment.choices  # type: ignore[attr-defined]
        seeds = [
            seed
            for seed in state._paired_pipeline_seed_configs()
            if seed.config.get(MODE) == EPI_TMA
        ]
    # One twin per 256-row pair tile and placement, at the deepest fitting
    # ring beside the aux rings; the 128-row pair tile stays on SIMT reads.
    assert all(seed.block_sizes[0] == 256 for seed in seeds)
    depths = {
        (
            tuple(seed.block_sizes),
            seed.config["tcgen05_aux_load_placement"],
        ): seed.config["tcgen05_ab_stages"]
        for seed in seeds
    }
    assert depths[((256, 128, 64), "post_acc_wait")] == 7
    assert depths[((256, 128, 64), "pre_acc_wait")] == 7
    assert depths[((256, 256, 64), "post_acc_wait")] == 4
    assert depths[((256, 256, 128), "post_acc_wait")] == 2


@pytest.mark.parametrize("view", sorted(_illegal_aux_views(1, 1)))
def test_tma_illegal_aux_views_keep_the_staged_modes_off(view: str) -> None:
    """Both staged modes bulk-load the aux tile through a row-major TensorMap.

    An operand without a 16-byte base, 16-byte outer strides and a contiguous
    last axis is proved illegal at bind time (the search surface, the seeds and
    the normalizer keep both modes off) and refused by codegen regardless.
    """
    a, b, _g, d = _gated_args()
    g = _illegal_aux_views(4096, 1024)[view]
    args = (a, b, g, d)
    with _cpu_codegen():
        bound = _gated_gemm._bind_isolated(args)
        state = bound.config_spec._cute_tcgen05_config
        assert state.exact_shape_aux_kernel_detected
        assert not state.exact_aux_tma_provable
        assert not state.epi_tma_search_enabled()
        assert not state._aux_tma_search_enabled()
        assert state.aux_load_mode_autotune_fragments() == {}
        assert all(
            seed.config.get(MODE, TCGEN05_AUX_LOAD_MODE_SIMT)
            == TCGEN05_AUX_LOAD_MODE_SIMT
            for seed in bound.config_spec.compiler_seed_configs
        )
    with pytest.raises(exc.InvalidConfig, match="TMA-legal auxiliary tensors"):
        _normalize(_gated_gemm, args, _one_cta(), fix=False)
    assert _normalize(_gated_gemm, args, _one_cta(), fix=True)[MODE] == (
        TCGEN05_AUX_LOAD_MODE_SIMT
    )
    # Codegen proves the staged descriptors itself (fails closed even if the
    # bind-time fact were wrong), on the epilogue-issued and the
    # producer-warp ring alike.
    for config in (_one_cta(), _producer_warp_tma()):
        with _cpu_codegen():
            bound = _gated_gemm._bind_isolated(args)
            bound.config_spec._cute_tcgen05_config.exact_aux_tma_provable = True
            with pytest.raises(
                exc.BackendUnsupported, match="TMA-legal auxiliary tensors"
            ):
                bound.to_code(helion.Config.from_dict(config))


def test_shared_fanout_takes_the_ring() -> None:
    """The two-output fanout family (``se_block_fwd``) keeps the shared
    schedule with the epilogue-issued ring; its one ``x`` ring costs the
    pair's fourth AB stage at bk=128 (two C rings beside it)."""
    args = (
        torch.empty((4096, 1024), dtype=torch.float16),
        torch.empty((1024, 1024), dtype=torch.float16),
    )
    config = {
        "block_sizes": [256, 128, 128],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 4,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_epilogue_fanout": "shared",
        "tcgen05_aux_load_placement": "pre_acc_wait",
        MODE: EPI_TMA,
    }
    fixed = _normalize(_gated_pair, args, config, fix=True)
    assert fixed["tcgen05_epilogue_fanout"] == "shared"
    assert fixed[MODE] == EPI_TMA and fixed["tcgen05_ab_stages"] == 3
    code = _render(_gated_pair, args, dict(config, tcgen05_ab_stages=3))
    epilogue = _epilogue(code)
    assert code.count("'kind': 'tcgen05_aux_tma'") == 1
    prefetch = epilogue.index("for _tcgen05_aux_prefetch in cutlass.range(")
    loop = epilogue.index("for _tcgen05_subtile in cutlass.range(")
    acc_wait = epilogue.index(
        "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
    )
    assert prefetch < loop < acc_wait


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="tcgen05 requires an sm_100 GPU",
)
@pytest.mark.parametrize(
    "config",
    (
        _one_cta(),
        _one_cta(tcgen05_aux_load_placement="pre_acc_wait"),
        _pair(),
        _pair(tcgen05_aux_load_placement="pre_acc_wait", tcgen05_aux_stages=3),
    ),
    ids=("one_cta", "one_cta_pre", "pair", "pair_pre_s3"),
)
def test_gpu_numerics(config: dict[str, object]) -> None:
    torch.manual_seed(0)
    # Several output tiles per CTA (and per CTA pair) on 148 SMs, so the ring
    # state is carried across tiles.
    m, n, k = 4096, 2048, 256
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(n, k, device=DEVICE, dtype=torch.float16)
    g = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    d = torch.rand(m, n, device=DEVICE, dtype=torch.float16)
    bound = _gated_gemm.bind((a, b, g, d))
    bound.set_config(helion.Config.from_dict(config))
    out = bound(a, b, g, d)
    expected = torch.addmm(g * d, a, b.T)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="tcgen05 requires an sm_100 GPU",
)
@pytest.mark.parametrize("view", sorted(_illegal_aux_views(1, 1)))
def test_gpu_tma_illegal_aux_views_are_refused(view: str) -> None:
    torch.manual_seed(0)
    m, n, k = 4096, 1024, 256
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(n, k, device=DEVICE, dtype=torch.float16)
    d = torch.rand(m, n, device=DEVICE, dtype=torch.float16)
    g = _illegal_aux_views(m, n, device=DEVICE, randn=True)[view]
    bound = _gated_gemm.bind((a, b, g, d))
    assert not bound.config_spec._cute_tcgen05_config.exact_aux_tma_provable
    for config in (_one_cta(), _producer_warp_tma()):
        with pytest.raises(
            (exc.InvalidConfig, exc.BackendUnsupported),
            match="TMA-legal auxiliary tensors",
        ):
            bound.set_config(helion.Config.from_dict(config))
    # The repaired sample reads the view through the register path.
    fixed = _one_cta()
    with bound.env:
        bound.config_spec.normalize(fixed, _fix_invalid=True)
    assert fixed[MODE] == TCGEN05_AUX_LOAD_MODE_SIMT
    bound.set_config(helion.Config.from_dict(fixed))
    out = bound(a, b, g, d)
    torch.testing.assert_close(out, torch.addmm(g * d, a, b.T), rtol=2e-2, atol=2e-1)
