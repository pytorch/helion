"""Multi-output tcgen05 epilogues (CPU only): ``pre_acc_wait`` legality, the
per-output C ring in the SMEM model, the whole-fragment auxiliary hoist and the
fanout / hoist coverage carriers.

The kernels mirror ``examples/se_block.py``'s forward after the full-slice
stripmine: one accumulator feeding two fresh outputs where only the second
reads an exact-shape auxiliary operand (``out = 2 * x * sigmoid(x @ w)``,
``s = sigmoid(x @ w)``).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch._subclasses.fake_tensor import FakeTensor

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.autotuner_heuristics.cute_epilogue_fanout import (
    _synthesized_pair_fanout_carriers,
)
from helion._compiler.autotuner_heuristics.cute_l2_row_swizzle import row_swizzle_size
from helion._compiler.autotuner_heuristics.cute_launch_bounds import MIN_BLOCKS_KEY
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_PAIR_BLOCK_NS
from helion._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxFacts
from helion._compiler.cute.tcgen05_constants import (
    tcgen05_aux_fragment_bytes_per_thread,
)
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])

PLACEMENT = TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
HOIST = TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
SWIZZLE = TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
PRE = {PLACEMENT: "pre_acc_wait"}
SHARED = {FANOUT_CONFIG_KEY: "shared"}
ON = {HOIST: True}


def gated_pair(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """se_block_fwd after the stripmine: ``s`` is aux-free, ``out`` reads ``x``."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        x_tile = x[tile_m, tile_n]
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        acc2 = 2.0 * x_tile.to(torch.float32) * sigmoid_result
        out[tile_m, tile_n] = acc2.to(x.dtype)
    return out, s


def residual_single(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One output whose chain reads the exact-shape operand: aux on every store."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + x[tile_m, tile_n].to(torch.float32)).to(x.dtype)
    return out


def pair_no_aux(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two outputs, neither reading an auxiliary operand."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        out[tile_m, tile_n] = (2.0 * sigmoid_result).to(x.dtype)
    return out, s


def plain_single(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One output, no auxiliary operand: the plain narrow-subtile epilogue."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@pytest.fixture(autouse=True)
def cuda_trace() -> Generator[None, None, None]:
    """Real static converter, with only the traced device replaced by CUDA."""
    old_init = CompileEnvironment.__init__
    old_convert = CompileEnvironment._to_fake_tensor
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def initialize(env: Any, device: torch.device, *args: Any, **kwargs: Any) -> None:
        assert device.type in ("cpu", "cuda")
        old_init(env, torch.device("cuda:0"), *args, **kwargs)

    def convert(env: Any, tensor: torch.Tensor, origin: Any) -> FakeTensor:
        result = old_convert(env, tensor, origin)
        assert isinstance(result, FakeTensor) and result.fake_mode is env.fake_mode
        result.fake_device = torch.device("cuda:0")
        return result

    try:
        with ExitStack() as stack:
            stack.enter_context(_target())
            stack.enter_context(
                patch.object(CompileEnvironment, "__init__", initialize)
            )
            stack.enter_context(
                patch.object(CompileEnvironment, "_to_fake_tensor", convert)
            )
            stack.enter_context(_mock_cuda_unavailable())
            stack.enter_context(
                patch(
                    "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
                )
            )
            stack.enter_context(
                patch("torch.cuda.get_device_capability", return_value=(10, 0))
            )
            stack.enter_context(
                patch.object(
                    DeviceProperties,
                    "create",
                    return_value=DeviceProperties(
                        type="cuda",
                        index=0,
                        multi_processor_count=148,
                        cc=100,
                        major=10,
                        regs_per_multiprocessor=65536,
                        max_threads_per_multi_processor=2048,
                        max_threads_per_block=1024,
                        warp_size=32,
                    ),
                )
            )
            yield
    finally:
        torch.set_num_threads(old_threads)


def bind(
    fn: Callable[..., object] = gated_pair,
    *,
    rows: int = 4096,
    columns: int = 1024,
) -> BoundKernel[Any]:
    args = (
        torch.empty((rows, columns), dtype=torch.float16),
        torch.empty((columns, columns), dtype=torch.float16),
    )
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    return kernel._bind_isolated(args)


def state_of(bound: BoundKernel[Any]) -> Any:
    return bound.config_spec._cute_tcgen05_config


def base_config(*parts: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "block_sizes": [128, 128, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 5,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 4,
        "tcgen05_num_epi_warps": 4,
    }
    for part in parts:
        config.update(part)
    config.update(overrides)
    return config


def normalized(
    bound: BoundKernel[Any], config: dict[str, Any], *, fix_invalid: bool = False
) -> dict[str, Any]:
    config = dict(config)
    with bound.env:
        bound.config_spec.normalize(config, _fix_invalid=fix_invalid)
    return config


def render(bound: BoundKernel[Any], config: dict[str, Any]) -> str:
    return bound.to_code(helion.Config(**config))


def test_facts_distinguish_aux_on_every_store() -> None:
    gated = state_of(bind(gated_pair))
    assert gated.aux_kernel_detected and gated.exact_shape_aux_kernel_detected
    assert not gated.aux_loads_on_every_store
    assert gated.rowvec_aux_facts == Tcgen05RowvecAuxFacts(
        output_itemsize=2, rows=(), output_count=2, exact_aux_count=1
    )
    assert gated.pre_acc_wait_admissible_via_fanout()

    single = state_of(bind(residual_single))
    assert single.aux_kernel_detected and single.aux_loads_on_every_store
    assert single.rowvec_aux_facts is not None
    assert single.rowvec_aux_facts.output_count == 1

    plain = state_of(bind(pair_no_aux))
    assert not plain.aux_kernel_detected and not plain.aux_loads_on_every_store
    assert plain.rowvec_aux_facts is not None
    assert plain.rowvec_aux_facts.output_count == 2
    assert not plain.pre_acc_wait_admissible_via_fanout()


def test_pre_acc_wait_is_refused_at_an_aux_free_store_and_repaired_in_search() -> None:
    bound = bind(gated_pair)
    # Explicit: refused before any compile (the store lowering would refuse
    # it at ``s``).
    with pytest.raises(InvalidConfig, match="no per-subtile auxiliary loads"):
        normalized(bound, base_config(PRE))
    # Sampled: repaired to the default placement.
    repaired = normalized(bound, base_config(PRE), fix_invalid=True)
    assert repaired[PLACEMENT] == "post_acc_wait"
    # The shared fanout merges the chains: the placement stands.
    shared = normalized(bound, base_config(PRE, SHARED))
    assert shared[PLACEMENT] == "pre_acc_wait"
    assert shared[FANOUT_CONFIG_KEY] == "shared"
    # Aux on every store: the placement needs no fanout.
    single = normalized(bind(residual_single), base_config(PRE))
    assert single[PLACEMENT] == "pre_acc_wait"
    # No aux anywhere: refused even with the fanout, and under the search's
    # repairing normalization too (the placement is never offered to such a
    # kernel, so it is a config error rather than a sample to repair).
    plain = bind(pair_no_aux)
    for fix_invalid in (False, True):
        with pytest.raises(InvalidConfig, match="no per-subtile auxiliary loads"):
            normalized(plain, base_config(PRE, SHARED), fix_invalid=fix_invalid)
        with pytest.raises(InvalidConfig, match="no per-subtile auxiliary loads"):
            normalized(plain, base_config(PRE), fix_invalid=fix_invalid)


def test_placement_and_hoist_leave_the_surface_where_nothing_renders() -> None:
    gated = bind(gated_pair).config_spec._flat_fields()
    assert PLACEMENT in gated and HOIST in gated
    assert gated[PLACEMENT].choices == ("post_acc_wait", "pre_acc_wait")  # type: ignore[attr-defined]
    single = bind(residual_single).config_spec._flat_fields()
    assert PLACEMENT in single and HOIST in single
    assert single[PLACEMENT].choices == ("post_acc_wait", "pre_acc_wait")  # type: ignore[attr-defined]
    plain = bind(pair_no_aux).config_spec._flat_fields()
    assert PLACEMENT not in plain and HOIST not in plain


def test_placement_collapses_to_the_default_where_pre_acc_wait_is_unsampled() -> None:
    # A carrierless geometry: the fanout has no search coordinate, so no
    # sampled config can render ``pre_acc_wait`` at the aux-free ``s`` store.
    # The key stays on the surface with the default as its only choice: a
    # coverage carrier normalized on another binding holds ``post_acc_wait``,
    # and a coverage transfer must never change a supplied field.
    bound = bind(gated_pair, rows=4096 + 64)
    assert not state_of(bound).epilogue_fanout_search_enabled
    fields = bound.config_spec._flat_fields()
    assert FANOUT_CONFIG_KEY not in fields
    assert fields[PLACEMENT].choices == ("post_acc_wait",)  # type: ignore[attr-defined]
    # A complete carrier of another coverage group, the placement supplied.
    carrier = _group(bound, MIN_BLOCKS_KEY).witnesses[0].carrier
    carrier.config[MIN_BLOCKS_KEY] = 0
    carrier.config[PLACEMENT] = "post_acc_wait"
    expected = bound._normalized_config_copy(carrier)
    assert expected.config[PLACEMENT] == "post_acc_wait"
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        flat, selected = generation.strict_config_pair(carrier)
        assert selected == expected
        assert generation.unflatten(flat) == expected
        # Random draws fill the slot with the default rather than a repair.
        (index,), _ = generation._key_to_flat_indices[PLACEMENT]
        assert all(
            generation.random_flat()[index] == "post_acc_wait" for _ in range(16)
        )
    # The search's repair and the strict refusal of ``pre_acc_wait`` stand.
    assert normalized(bound, base_config(PRE), fix_invalid=True)[PLACEMENT] == (
        "post_acc_wait"
    )
    with pytest.raises(InvalidConfig, match="no per-subtile auxiliary loads"):
        normalized(bound, base_config(PRE))


def test_hoist_admission_is_strict_and_budgeted() -> None:
    bound = bind(gated_pair)
    # Needs the pre-wait placement (so, here, the fanout too).
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(bound, base_config(ON))
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(bound, base_config(ON, SHARED))
    assert normalized(bound, base_config(ON, SHARED), fix_invalid=True)[HOIST] is False
    assert normalized(bound, base_config(ON, PRE, SHARED))[HOIST] is True
    # Register budget: a 128x256 fp16 fragment is 512 B per epilogue thread.
    assert (
        tcgen05_aux_fragment_bytes_per_thread(bm=128, bn=256, itemsize=2, epi_warps=4)
        == 2 * TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
    )
    wide = base_config(
        ON,
        PRE,
        SHARED,
        block_sizes=[128, 256, 64],
        tcgen05_ab_stages=3,
        tcgen05_c_stages=2,
    )
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(bound, wide)
    # A CTA pair's epilogue owns half of the 256-row pair tile per CTA.
    state = state_of(bound)
    pair = base_config(
        ON,
        PRE,
        SHARED,
        block_sizes=[256, 128, 64],
        tcgen05_cluster_m=2,
        tcgen05_c_stages=2,
    )
    assert state.aux_fragment_hoist_admissible(pair)
    pair["tcgen05_cluster_m"] = 1
    assert not state.aux_fragment_hoist_admissible(pair)
    # Typed: a non-boolean is refused.
    with pytest.raises(InvalidConfig, match="must be a boolean"):
        normalized(bound, base_config(PRE, SHARED, **{HOIST: 1}))


def test_hoist_mirrors_the_static_full_cta_pair_tma_store_epilogue() -> None:
    # The store lowering renders the hoist only on the static-full TMA-store
    # epilogue; the normalizer refuses the knob wherever that is not proved
    # from the config, so no sample reaches the render worker to be refused.
    bound = bind(residual_single, rows=1024, columns=512)
    state = state_of(bound)
    # 16-bit bm=128 with cluster_m=2 is the clustered single-CTA family: no
    # role-local TMA-store epilogue, hence no hoist.
    clustered = base_config(PRE, ON, tcgen05_cluster_m=2, tcgen05_c_stages=2)
    assert not state.aux_fragment_hoist_admissible(clustered)
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(bound, clustered)
    assert normalized(bound, clustered, fix_invalid=True)[HOIST] is False
    # The bm=256 CTA pair and the explicit two-CTA group are the pair MMA.
    pair = base_config(
        PRE, ON, block_sizes=[256, 128, 64], tcgen05_cluster_m=2, tcgen05_c_stages=2
    )
    assert state.aux_fragment_hoist_admissible(pair)
    assert normalized(bound, pair)[HOIST] is True
    assert state._config_is_two_cta_mma(
        {"tcgen05_cta_group": "two"}, bm=128, cluster_m=2
    )
    assert not state._config_is_two_cta_mma({}, bm=128, cluster_m=2)
    assert not state._config_is_two_cta_mma({}, bm=256, cluster_m=1)
    # Whole tiles on every axis: a tile that does not divide the static
    # extents takes an edge-tile store the hoist does not render on.
    assert state._static_full_tiles(bm=128, bn=128, bk=64)
    assert not state._static_full_tiles(bm=128, bn=96, bk=64)
    odd = bind(residual_single, rows=1024 + 64, columns=512)
    assert not state_of(odd).aux_fragment_hoist_admissible(base_config(PRE, ON))
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(odd, base_config(PRE, ON))
    assert state_of(odd).aux_fragment_hoist_admissible(
        base_config(PRE, ON, block_sizes=[64, 128, 64])
    )


def test_hoist_at_a_narrow_epilogue_tile_relays_the_slice_onto_the_carrier() -> None:
    # At bm=64 each epilogue thread's fragment spans two M rows and keeps the
    # nested T2R profile; the hoisted subtile slice is re-laid out onto the
    # accumulator carrier's layout (as the per-subtile global read is) instead
    # of being loaded directly, which the CuTe DSL refuses to combine.
    narrow = base_config(PRE, ON, block_sizes=[64, 128, 64], tcgen05_c_stages=2)
    for fn, parts in ((residual_single, ()), (gated_pair, (SHARED,))):
        bound = bind(fn, rows=1024, columns=512)
        config = dict(narrow)
        for part in parts:
            config.update(part)
        assert normalized(bound, config)[HOIST] is True
        code = render(bound, config)
        assert "_BLOCK_SIZE_0 = 64" in code
        role = _epilogue_role(code)
        assert role.index(
            "cute.autovec_copy(tcgen05_tTR_gAux_grouped_0, tcgen05_aux_rmem_full_0)"
        ) < role.index("tcgen05_acc_pipeline.consumer_wait(")
        slice_line = (
            "tcgen05_tTR_gAux_subtile_0 = tcgen05_aux_rmem_full_0"
            "[None, None, None, cutlass.Int32(_tcgen05_subtile)]"
        )
        assert slice_line in role
        after = role[role.index(slice_line) :]
        assert after.index(
            "tcgen05_aux_rmem_0 = cute.make_rmem_tensor(cute.make_layout("
        ) < (after.index("tcgen05_aux_loaded_0 = tcgen05_aux_rmem_0.load()"))
        assert (
            "tcgen05_aux_rmem_full_0[None, None, None, cutlass.Int32(_tcgen05_subtile)].load()"
            not in role
        )
        # No per-subtile global read remains.
        assert (
            "cute.autovec_copy(tcgen05_tTR_gAux_subtile_0, tcgen05_aux_rmem_0)" in role
        )
        assert "tcgen05_tTR_gAux_subtile_0 = tcgen05_tTR_gAux_grouped_0" not in role
    # A 128-row tile keeps the direct register load of the slice.
    wide = render(bind(residual_single, rows=1024, columns=512), base_config(PRE, ON))
    assert (
        "tcgen05_aux_loaded_0 = tcgen05_aux_rmem_full_0"
        "[None, None, None, cutlass.Int32(_tcgen05_subtile)].load()"
    ) in _epilogue_role(wide)


def _epilogue_role(code: str) -> str:
    start = code.index("tcgen05_c_pipeline.producer_acquire()")
    return code[start:]


def test_hoist_renders_the_fragment_read_before_the_accumulator_wait() -> None:
    bound = bind(gated_pair)
    plain = render(bound, base_config(PRE, SHARED))
    hoisted = render(bound, base_config(PRE, SHARED, ON))
    assert "tcgen05_aux_rmem_full_0" not in plain
    role = _epilogue_role(hoisted)
    hoist = role.index(
        "cute.autovec_copy(tcgen05_tTR_gAux_grouped_0, tcgen05_aux_rmem_full_0)"
    )
    wait = role.index("tcgen05_acc_pipeline.consumer_wait(")
    assert hoist < wait
    assert (
        "tcgen05_aux_loaded_0 = tcgen05_aux_rmem_full_0"
        "[None, None, None, cutlass.Int32(_tcgen05_subtile)].load()"
    ) in role
    # The per-subtile global read is gone.
    per_subtile = "cute.autovec_copy(tcgen05_tTR_gAux_subtile_0, tcgen05_aux_rmem_0)"
    assert per_subtile in plain and per_subtile not in hoisted
    # Off: byte-identical to a config without the key.
    assert render(bound, base_config(PRE, SHARED, **{HOIST: False})) == plain
    # Aux on every store: no fanout needed for the hoist.
    role = _epilogue_role(render(bind(residual_single), base_config(PRE, ON)))
    assert role.index("tcgen05_aux_rmem_full_0)") < role.index(
        "tcgen05_acc_pipeline.consumer_wait("
    )


def test_smem_model_charges_one_c_ring_per_output() -> None:
    bound = bind(gated_pair)
    state = state_of(bound)
    facts = state.rowvec_aux_facts
    assert facts is not None and facts.output_count == 2
    kwargs: dict[str, Any] = {
        "bm": 128,
        "bn": 128,
        "bk": 64,
        "cluster_m": 1,
        "ab_stages": 5,
        "c_stages": 4,
        "stage_rows": False,
    }
    two_rings = state.default_layout_smem_bytes(**kwargs)
    state.rowvec_aux_facts = Tcgen05RowvecAuxFacts(
        output_itemsize=2, rows=(), output_count=1
    )
    one_ring = state.default_layout_smem_bytes(**kwargs)
    state.rowvec_aux_facts = facts
    assert two_rings is not None and one_ring is not None
    # The (128, 32) fp16 subtile: 8 KiB per stage, 32 KiB per 4-stage ring.
    assert two_rings - one_ring == 4 * 128 * 32 * 2
    # ab=6 + c=4 fits one ring (224 KiB) but not two (256 KiB): the sampled
    # rings are shallowed to the deepest AB depth the two rings leave.
    sample = base_config(tcgen05_ab_stages=6, tcgen05_c_stages=4)
    state._fix_multi_output_ring_search_config(sample)
    assert (sample["tcgen05_ab_stages"], sample["tcgen05_c_stages"]) == (5, 4)
    sample = base_config(tcgen05_ab_stages=6, tcgen05_c_stages=2)
    state._fix_multi_output_ring_search_config(sample)
    assert (sample["tcgen05_ab_stages"], sample["tcgen05_c_stages"]) == (6, 2)
    # A single-output kernel is untouched by this gate.
    single = state_of(bind(residual_single))
    sample = base_config(tcgen05_ab_stages=6, tcgen05_c_stages=4)
    single._fix_multi_output_ring_search_config(sample)
    assert (sample["tcgen05_ab_stages"], sample["tcgen05_c_stages"]) == (6, 4)
    # An explicit config is refused where its two rings provably overflow
    # (128x128x128: ab=3 is 192 KiB, two 4-stage C rings 64 KiB more -- the
    # NVVM backend used to reject it after normalization accepted it); the
    # shallow C ring fits, and the search repairs the same sample to it.
    deep = base_config(
        SHARED,
        block_sizes=[128, 128, 128],
        tcgen05_ab_stages=3,
        tcgen05_c_stages=4,
    )
    with pytest.raises(InvalidConfig, match="overflows the per-CTA SMEM budget"):
        normalized(bound, deep)
    assert normalized(bound, deep | {"tcgen05_c_stages": 2})["tcgen05_c_stages"] == 2
    repaired = normalized(bound, deep, fix_invalid=True)
    assert (repaired["tcgen05_ab_stages"], repaired["tcgen05_c_stages"]) == (3, 2)


def _group(bound: BoundKernel[Any], key: str) -> Any:
    groups = [g for g in bound.config_spec.compiler_coverage_groups if g.key == key]
    return groups[0] if groups else None


def _carriers(group: Any) -> list[dict[str, Any]]:
    return [dict(witness.carrier.config) for witness in group.witnesses]


def test_fanout_and_hoist_coverage_for_a_seedless_two_output_kernel() -> None:
    bound = bind(gated_pair)
    spec = bound.config_spec
    # The only compiler seeds are the 256x256 CTA pairs, which the fanout
    # does not admit; the carriers are synthesized from the geometry.
    seeds = spec.compiler_seed_configs
    assert seeds and all(seed.block_sizes[:2] == [256, 256] for seed in seeds)
    assert state_of(bound).epilogue_fanout_search_enabled
    fanout = _group(bound, FANOUT_CONFIG_KEY)
    assert fanout is not None and not fanout.dependencies
    carriers = _carriers(fanout)
    assert [w.value for w in fanout.witnesses] == ["off", "shared", "off", "shared"]
    assert [c["tcgen05_c_stages"] for c in carriers] == [4, 4, 2, 2]
    for carrier in carriers:
        assert carrier["block_sizes"] == [128, 128, 64]
        assert carrier["pid_type"] == "persistent_interleaved"
        assert carrier["tcgen05_cluster_m"] == 1
        assert carrier["tcgen05_ab_stages"] == 5
        assert carrier.get(PLACEMENT, "post_acc_wait") == "post_acc_wait"
        assert FANOUT_CONFIG_KEY not in carrier
    hoist = _group(bound, HOIST)
    assert hoist is not None
    assert [(d.mechanism, d.key, d.value) for d in hoist.dependencies] == [
        ("cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared")
    ]
    assert [w.value for w in hoist.witnesses] == [False, True, False, True]
    hoist_carriers = _carriers(hoist)
    for carrier in hoist_carriers:
        assert carrier[FANOUT_CONFIG_KEY] == "shared"
        assert carrier[PLACEMENT] == "pre_acc_wait"
        assert HOIST not in carrier
    # The hoist group carries two output-tile geometries: the CTA pair the
    # fanout admits (256x128, the deepest AB ring the two-ring arena fits at
    # K tile 128: 4 x 48 KiB + 32 KiB) before the fanout group's one-CTA
    # carrier (its deep-C twin).
    for carrier in hoist_carriers[:2]:
        assert carrier["block_sizes"] == [256, 128, 128]
        assert carrier["tcgen05_cluster_m"] == 2
        assert carrier["tcgen05_ab_stages"] == 4
        assert carrier["tcgen05_c_stages"] == 2
    for carrier in hoist_carriers[2:]:
        assert carrier["block_sizes"] == [128, 128, 64]
        assert carrier["tcgen05_cluster_m"] == 1
        assert carrier["tcgen05_c_stages"] == 4
    # The row swizzle rides the hoist carriers: eight N tiles of 128 at
    # N = 1024, so the row-covering swizzle is 8, declared beside the
    # legacy 1 with the fanout and the hoist as dependencies.
    swizzle = _group(bound, SWIZZLE)
    assert swizzle is not None and swizzle.mechanism == "cute.l2_row_swizzle"
    assert [(d.mechanism, d.key, d.value) for d in swizzle.dependencies] == [
        ("cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared"),
        ("cute.aux_fragment_hoist", HOIST, True),
    ]
    assert swizzle.domain == (1, 2, 4, 8) and swizzle.legacy == 1
    assert [w.value for w in swizzle.witnesses] == [1, 8, 1, 8]
    swizzle_carriers = _carriers(swizzle)
    for carrier, source in zip(swizzle_carriers, hoist_carriers, strict=True):
        assert SWIZZLE not in carrier
        assert carrier.pop(HOIST) is True
        assert carrier == {k: v for k, v in source.items() if k != SWIZZLE}
    # Every witness is a complete, strictly valid config that keeps its mode.
    for group in (fanout, hoist, swizzle):
        for witness in group.witnesses:
            config = normalized(
                bound, dict(witness.carrier.config) | {group.key: witness.value}
            )
            assert config.get(group.key, group.legacy) == witness.value
    # Geometries no 128x128 tile divides stay carrierless.
    odd = bind(gated_pair, rows=4096 + 64)
    assert not state_of(odd).epilogue_fanout_search_enabled
    assert _group(odd, FANOUT_CONFIG_KEY) is None
    assert _group(odd, HOIST) is None


def test_pair_fanout_sample_keeps_its_geometry_in_the_search() -> None:
    bound = bind(gated_pair)
    pair = base_config(
        SHARED,
        PRE,
        ON,
        block_sizes=[256, 128, 128],
        tcgen05_cluster_m=2,
        tcgen05_ab_stages=4,
        tcgen05_c_stages=2,
        tcgen05_l2_swizzle_size=8,
    )
    # Strictly valid, and a fixed point of the search's repairing pass: the
    # cluster_m=2 projections (the canonical 256x256 tile, the aux-TMA
    # producer regime) leave a sample whose proved fanout admits its tile
    # alone, so the pair recipe is a config the search can measure.
    strict = normalized(bound, pair)
    repaired = normalized(bound, pair, fix_invalid=True)
    for config in (strict, repaired):
        assert config["block_sizes"] == [256, 128, 128]
        assert config["tcgen05_cluster_m"] == 2
        assert config["tcgen05_ab_stages"] == 4
        assert config["tcgen05_c_stages"] == 2
        assert config["tcgen05_l2_swizzle_size"] == 8
        assert config[FANOUT_CONFIG_KEY] == "shared"
        assert config[PLACEMENT] == "pre_acc_wait"
        assert config[HOIST] is True
        assert config.get("tcgen05_aux_load_mode", "simt") == "simt"
        assert config.get("tcgen05_strategy", "role_local_monolithic") == (
            "role_local_monolithic"
        )
    # Without the fanout the same pair sample takes the projections as before.
    plain = normalized(
        bound,
        base_config(
            block_sizes=[256, 128, 128],
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=4,
            tcgen05_c_stages=2,
        ),
        fix_invalid=True,
    )
    assert plain["tcgen05_cluster_m"] == 2
    assert plain["block_sizes"][:2] == [256, 256]
    assert plain.get(FANOUT_CONFIG_KEY, "off") == "off"


def _pair(bn: int, bk: int, ab: int, *parts: dict[str, Any]) -> dict[str, Any]:
    return base_config(
        SHARED,
        PRE,
        {HOIST: False},
        *parts,
        block_sizes=[256, bn, bk],
        tcgen05_cluster_m=2,
        tcgen05_ab_stages=ab,
        tcgen05_c_stages=2,
    )


def test_pair_fanout_is_admitted_by_budget_at_every_pair_geometry() -> None:
    # The fanout's CTA pair is admitted by its TMEM / SMEM / C-ring budget at
    # every pair N tile the backend runs (256, 128, 64), not at one fixed
    # geometry: a strictly valid sample is a fixed point of the search's
    # repairing pass with its tile, rings and recipe intact.
    bound = bind(gated_pair)
    state = state_of(bound)
    for bn, bk, ab in (
        (256, 64, 4),
        (256, 64, 5),
        (256, 128, 2),
        (128, 64, 8),
        (64, 64, 9),
    ):
        sample = _pair(bn, bk, ab)
        assert state.epilogue_fanout_refusal(sample) is None
        strict = normalized(bound, sample)
        repaired = normalized(bound, sample, fix_invalid=True)
        for config in (strict, repaired):
            assert config["block_sizes"] == [256, bn, bk]
            assert config["tcgen05_cluster_m"] == 2
            assert config["tcgen05_ab_stages"] == ab
            assert config["tcgen05_c_stages"] == 2
            assert config[FANOUT_CONFIG_KEY] == "shared"
            assert config[PLACEMENT] == "pre_acc_wait"
            assert config.get("tcgen05_aux_load_mode", "simt") == "simt"
            assert config.get("tcgen05_strategy", "role_local_monolithic") == (
                "role_local_monolithic"
            )
        assert normalized(bound, repaired) == repaired
    # The ring depth is the budget's: two (128, 64) fp16 C rings of the
    # 256-wide pair tile take 64 KiB, so a 32 KiB-per-stage bk=64 AB ring
    # holds five stages and a 64 KiB bk=128 ring two; the refusal names the
    # overflow and the search shallows the sample to the deepest fit.
    for bk, ab, deepest in ((64, 6, 5), (128, 3, 2)):
        with pytest.raises(InvalidConfig, match="overflows the per-CTA SMEM budget"):
            normalized(bound, _pair(256, bk, ab))
        repaired = normalized(bound, _pair(256, bk, ab), fix_invalid=True)
        assert repaired["tcgen05_ab_stages"] == deepest
        assert repaired["block_sizes"] == [256, 256, bk]
        assert repaired[FANOUT_CONFIG_KEY] == "shared"
    # The hoist keeps its own budget: a 128x256 fp16 fragment is 512 bytes
    # per epilogue thread, so the 256-wide pair takes pre_acc_wait without it.
    hoisted = _pair(256, 64, 4, ON)
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True requires"):
        normalized(bound, hoisted)
    repaired = normalized(bound, hoisted, fix_invalid=True)
    assert repaired[HOIST] is False
    assert repaired[FANOUT_CONFIG_KEY] == "shared"
    assert repaired[PLACEMENT] == "pre_acc_wait"
    assert repaired["block_sizes"] == [256, 256, 64]
    # Refusals name their budget, through the public refusal: the pair's
    # role-local TMA-store epilogue stores each CTA's 128-row half of the
    # 256-row pair tile (the fact the codegen store protocol demands), so a
    # 128-row tile under cluster_m=2 -- the clustered one-CTA family, or the
    # pair MMA under tcgen05_cta_group='two' -- is refused with that fact
    # rather than reaching the codegen gate; an N tile the pair MMA does not
    # run is refused before SMEM.
    half = (
        f"{TCGEN05_TWO_CTA_BLOCK_M // 2}-row half of the {TCGEN05_TWO_CTA_BLOCK_M}-row"
    )
    for family in ({}, {"tcgen05_cta_group": "two"}):
        clustered = _pair(128, 64, 4) | {"block_sizes": [128, 128, 64]} | family
        reason = state.epilogue_fanout_refusal(clustered)
        assert reason is not None and half in reason
    with pytest.raises(InvalidConfig, match=half):
        normalized(bound, _pair(128, 64, 4) | {"block_sizes": [128, 128, 64]})
    assert state.epilogue_fanout_refusal(_pair(32, 64, 4)) == (
        f"the CTA pair runs its MMA at N tiles "
        f"{sorted(TCGEN05_TWO_CTA_PAIR_BLOCK_NS)}, not 32"
    )
    # Every pair tile of the fanout renders through the one store protocol.
    assert "_BLOCK_SIZE_1 = 256" in render(bound, _pair(256, 64, 5))
    # A search neighbour of the 256x128 hoist witness whose pid type the
    # fanout refuses is projected onto the canonical 256x256 pair after the
    # hoist was judged at bn=128; the admitted fanout keeps the recipe and
    # the hoist is re-judged on the final tile (the autotuner compiled such
    # samples and lost them to the strict refusal).
    for pid_type in ("persistent_blocked", "flat"):
        neighbour = _pair(128, 128, 4, ON) | {"pid_type": pid_type}
        repaired = normalized(bound, neighbour, fix_invalid=True)
        assert repaired["block_sizes"] == [256, 256, 128]
        assert repaired[FANOUT_CONFIG_KEY] == "shared"
        assert repaired[PLACEMENT] == "pre_acc_wait"
        assert repaired[HOIST] is False
        assert normalized(bound, repaired) == repaired


def test_search_repair_rejudges_the_hoist_after_the_geometry_projections() -> None:
    # The re-judge is not the fanout's: a single-output kernel's neighbour of
    # the 256x128 pair hoist witness is projected onto the canonical 256x256
    # pair in the aux-TMA regime (scheduler warps, tcgen05_aux_load_mode=tma)
    # after the hoist was judged at bn=128 on the SIMT recipe, so without the
    # re-judge the repaired config kept tcgen05_aux_fragment_hoist=True on a
    # 512 B/thread fragment with a TMA aux load, which strict normalization
    # refuses (verified on the unpatched tree: the strict pass raised the
    # hoist's own InvalidConfig on the repaired config).
    bound = bind(residual_single)
    for pid_type in ("persistent_blocked", "flat"):
        neighbour = _pair(128, 128, 4, ON) | {"pid_type": pid_type}
        neighbour.pop(FANOUT_CONFIG_KEY)
        repaired = normalized(bound, neighbour, fix_invalid=True)
        assert repaired["block_sizes"] == [256, 256, 128]
        assert repaired["tcgen05_cluster_m"] == 2
        assert repaired.get(FANOUT_CONFIG_KEY, "off") == "off"
        assert repaired["tcgen05_aux_load_mode"] == "tma"
        assert repaired[HOIST] is False
        assert normalized(bound, repaired) == repaired


def test_smem_model_keeps_the_with_source_subtile_for_two_outputs() -> None:
    # The plain (128, 32) subtile of the 256-wide pair tile belongs to a
    # single-output epilogue without auxiliary operands; two outputs (or an
    # aux operand) keep CuTe's with-source (128, 64) subtile, so each fp16 C
    # ring is 32 KiB and the pair's arena holds 32 KiB less AB ring. The old
    # model admitted 256x256x64 ab=6 with two outputs; NVVM rejected it.
    bound = bind(gated_pair)
    state = state_of(bound)
    kwargs: dict[str, Any] = {
        "bm": 256,
        "bn": 256,
        "bk": 64,
        "cluster_m": 2,
        "ab_stages": 1,
        "c_stages": 2,
        "stage_rows": False,
    }
    two_outputs = state.default_layout_smem_bytes(**kwargs)
    facts = state.rowvec_aux_facts
    assert facts is not None and facts.output_count == 2
    state.rowvec_aux_facts = Tcgen05RowvecAuxFacts(
        output_itemsize=2, rows=(), output_count=1
    )
    one_output_with_aux = state.default_layout_smem_bytes(**kwargs)
    state.aux_kernel_detected = False
    plain = state.default_layout_smem_bytes(**kwargs)
    state.aux_kernel_detected = True
    state.rowvec_aux_facts = facts
    assert two_outputs is not None and one_output_with_aux is not None
    assert plain is not None
    # Two (128, 64) rings of two stages: 64 KiB; one: 32 KiB; plain: 16 KiB.
    assert two_outputs - one_output_with_aux == 2 * 128 * 64 * 2
    assert one_output_with_aux - plain == 2 * 128 * 64 * 2 - 2 * 128 * 32 * 2
    fits = [
        ab
        for ab in range(1, 9)
        if state.default_layout_smem_fits(**(kwargs | {"ab_stages": ab}))
    ]
    assert fits == [1, 2, 3, 4, 5]


def test_codegen_and_the_arena_model_agree_on_the_pair_subtile() -> None:
    # Codegen takes the plain (128, 32) subtile on the 256-wide 16-bit pair
    # tile for one output store without auxiliary descriptors
    # (``tcgen05_plain_narrow_subtile``); the arena model's predicate keeps
    # both conditions (``output_count == 1`` and no auxiliary operand), so
    # it prices a 16 KiB C ring for the plain kernel, a 32 KiB with-source
    # ring for one output with an aux operand and two of them for two
    # outputs: at 256x256x64 six 32 KiB AB stages fit beside one ring of
    # either size and five beside two -- the depths that render.
    narrow = "(cute.make_layout(128), cute.make_layout(32))"
    kwargs: dict[str, Any] = {
        "bm": 256,
        "bn": 256,
        "bk": 64,
        "cluster_m": 2,
        "c_stages": 2,
        "stage_rows": False,
    }
    priced: dict[str, int] = {}
    for fn, deepest, drop in (
        (plain_single, 6, (FANOUT_CONFIG_KEY, PLACEMENT, HOIST)),
        (residual_single, 6, (FANOUT_CONFIG_KEY,)),
        (gated_pair, 5, ()),
    ):
        bound = bind(fn)
        state = state_of(bound)
        fits = [
            ab
            for ab in range(1, 9)
            if state.default_layout_smem_fits(**kwargs, ab_stages=ab)
        ]
        assert fits == list(range(1, deepest + 1)), fn.__name__
        size = state.default_layout_smem_bytes(**kwargs, ab_stages=1)
        assert size is not None
        priced[fn.__name__] = size
        config = _pair(256, 64, deepest)
        for key in drop:
            config.pop(key)
        strict = normalized(bound, config)
        assert strict["tcgen05_ab_stages"] == deepest, fn.__name__
        code = render(bound, config)
        assert (narrow in code) is (fn is plain_single), fn.__name__
        assert ("compute_epilogue_tile_shape(" in code) is not (fn is plain_single), (
            fn.__name__
        )
        # One stage deeper overflows: the multi-output ring judge names the
        # rings, the single-output kernels' arena check the budget.
        with pytest.raises(InvalidConfig, match="SMEM budget"):
            normalized(bound, config | {"tcgen05_ab_stages": deepest + 1})
    # The same AB ring and fixed allocations: the C rings alone differ.
    assert priced["residual_single"] - priced["plain_single"] == 2 * 128 * 32 * 2
    assert priced["gated_pair"] - priced["residual_single"] == 2 * 128 * 64 * 2


def test_pair_carriers_enumerate_the_budgeted_geometries() -> None:
    # One carrier per pair geometry that walks whole output tiles, the
    # widest N tile first and per K tile, each at the deepest AB ring the two
    # C rings leave; the hoist group still takes one pair geometry (the first
    # the hoist admits: the 256-wide fragment is over its budget) beside the
    # fanout group's one-CTA carrier.
    bound = bind(gated_pair)
    carriers = _synthesized_pair_fanout_carriers(bound.config_spec, state_of(bound))
    assert [
        (tuple(c.block_sizes), c.config["tcgen05_ab_stages"]) for c in carriers
    ] == [
        ((256, 256, 128), 2),
        ((256, 256, 64), 5),
        ((256, 128, 128), 4),
        ((256, 128, 64), 8),
        ((256, 64, 128), 4),
        ((256, 64, 64), 9),
    ]
    for carrier in carriers:
        config = dict(carrier.config)
        assert config["tcgen05_cluster_m"] == 2
        assert config["tcgen05_c_stages"] == 2
        assert FANOUT_CONFIG_KEY not in config
        shared = normalized(bound, config | SHARED)
        assert shared[FANOUT_CONFIG_KEY] == "shared"
        assert shared["block_sizes"] == carrier.block_sizes
    hoist = _group(bound, HOIST)
    assert hoist is not None
    geometries = [
        (tuple(c["block_sizes"]), c["tcgen05_cluster_m"]) for c in _carriers(hoist)
    ]
    assert geometries == [((256, 128, 128), 2)] * 2 + [((128, 128, 64), 1)] * 2


def test_pair_carrier_needs_whole_pair_tiles() -> None:
    # 4224 rows: whole 128-row tiles but not whole 256-row pair tiles, so the
    # hoist group keeps its one-CTA C-ring twins and the swizzle group follows.
    bound = bind(gated_pair, rows=4096 + 128)
    hoist = _group(bound, HOIST)
    assert hoist is not None
    carriers = _carriers(hoist)
    assert [w.value for w in hoist.witnesses] == [False, True, False, True]
    assert [c["block_sizes"] for c in carriers] == [[128, 128, 64]] * 4
    assert [c["tcgen05_c_stages"] for c in carriers] == [4, 4, 2, 2]
    assert all(c["tcgen05_cluster_m"] == 1 for c in carriers)
    swizzle = _group(bound, SWIZZLE)
    assert swizzle is not None
    assert [w.value for w in swizzle.witnesses] == [1, 8, 1, 8]
    for carrier in _carriers(swizzle):
        assert carrier["block_sizes"] == [128, 128, 64]
        assert carrier[HOIST] is True
    for group in (hoist, swizzle):
        for witness in group.witnesses:
            normalized(bound, dict(witness.carrier.config) | {group.key: witness.value})


def test_row_swizzle_is_the_deepest_legal_size_within_one_row_of_n_tiles() -> None:
    legal = (1, 2, 4, 8)
    assert row_swizzle_size(8, legal) == 8
    assert row_swizzle_size(16, legal) == 8
    assert row_swizzle_size(4, legal) == 4
    assert row_swizzle_size(3, legal) == 2
    assert row_swizzle_size(1, legal) == 1
    # Narrower outputs: the witnesses pair the legacy 1 with the row-covering
    # swizzle of each carrier's own N tile (four tiles of 128 at N = 512; the
    # small-grid seeds' narrow N tiles at N = 128), every one strictly valid.
    for rows, columns in ((1024, 512), (256, 128)):
        bound = bind(gated_pair, rows=rows, columns=columns)
        swizzle = _group(bound, SWIZZLE)
        assert swizzle is not None
        n_index = state_of(bound)._matmul_block_indices()[1]
        values = [w.value for w in swizzle.witnesses]
        assert values[::2] == [1] * (len(values) // 2)
        for witness in swizzle.witnesses:
            tiles = columns // witness.carrier.block_sizes[n_index]
            assert witness.value in (1, row_swizzle_size(tiles, swizzle.domain))
            config = normalized(
                bound, dict(witness.carrier.config) | {SWIZZLE: witness.value}
            )
            assert config[SWIZZLE] == witness.value
        assert all(value != 1 for value in values[1::2])


def test_hoist_coverage_without_fanout_where_every_store_has_aux() -> None:
    bound = bind(residual_single, rows=256, columns=128)
    hoist = _group(bound, HOIST)
    assert hoist is not None and not hoist.dependencies
    assert _group(bound, FANOUT_CONFIG_KEY) is None
    assert len(hoist.witnesses) in (2, 4)
    for witness in hoist.witnesses:
        carrier = dict(witness.carrier.config)
        assert carrier[PLACEMENT] == "pre_acc_wait"
        assert carrier.get(FANOUT_CONFIG_KEY, "off") == "off"
        normalized(bound, carrier | {HOIST: witness.value})


def test_hoist_key_is_inert_or_refused_without_an_exact_shape_operand() -> None:
    # A recorded config of an exact-shape kernel carries the key at its
    # default; reused on a bind without such an operand it must not fail
    # loudly, while a requested hoist is refused (nothing could render it).
    bound = bind(pair_no_aux)
    off = normalized(bound, base_config(**{HOIST: False}))
    assert off.get(HOIST, False) is False
    with pytest.raises(InvalidConfig):
        normalized(bound, base_config(ON))
    assert (
        normalized(bound, base_config(ON), fix_invalid=True).get(HOIST, False) is False
    )


@pytest.mark.parametrize(
    "schedule",
    (
        {"pid_type": "persistent_blocked"},
        {"pid_type": "flat", "tcgen05_persistence_model": "non_persistent"},
        {"tcgen05_strategy": "role_local_with_scheduler"},
        {"block_sizes": [128, 128, 128], "tcgen05_ab_stages": 3},
    ),
)
def test_search_repair_drops_the_placement_and_hoist_with_the_fanout(
    schedule: dict[str, Any],
) -> None:
    # A hoist witness whose neighbour moved to a schedule the fanout refuses:
    # the search-time repair drops the fanout and, with it, the placement the
    # fanout alone admitted and the hoist that needs the placement, so the
    # repaired config is a fixed point of strict normalization instead of a
    # render-worker refusal.
    bound = bind(gated_pair)
    sample = base_config(PRE, SHARED, ON, tcgen05_c_stages=2, **schedule)
    repaired = normalized(bound, sample, fix_invalid=True)
    if repaired.get(FANOUT_CONFIG_KEY, "off") == "shared":
        assert repaired[PLACEMENT] == "pre_acc_wait"
        assert repaired[HOIST] is True
    else:
        assert repaired[PLACEMENT] == "post_acc_wait"
        assert repaired[HOIST] is False
    assert normalized(bound, repaired) == repaired
