from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.triton import _B200_MATMUL_HEURISTICS_PATH
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200FormulaMatmulHeuristic,
)
from helion._compiler.autotuner_heuristics.triton import TritonB200MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonH100MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_bucket
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_config_spec
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import MatmulFact

_SHAPE_BUCKET_KEYS = {
    "dtype",
    "k_bucket",
    "m_bucket",
    "n_bucket",
    "k_value",
    "m_value",
    "n_value",
}


def _matmul_fact(
    *,
    static_m: int = 1024,
    static_n: int = 1024,
    static_k: int = 1024,
    lhs_ndim: int = 2,
    rhs_ndim: int = 2,
) -> MatmulFact:
    return MatmulFact(
        lhs_ndim=lhs_ndim,
        rhs_ndim=rhs_ndim,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )


def _matmul_config_spec(
    *,
    matmul_facts: list[MatmulFact] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        matmul_facts=[] if matmul_facts is None else matmul_facts,
        block_sizes=[object(), object(), object()],
        allowed_pid_types=("flat",),
        _base_default_config=lambda: helion.Config(
            block_sizes=[1, 1, 1],
            l2_groupings=[1],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=3),
            "l2_groupings": ListOf(IntegerFragment(1, 64, 1), length=1),
            "num_warps": PowerOfTwoFragment(1, 32, 4),
            "num_stages": IntegerFragment(1, 8, 1),
            "pid_type": EnumFragment(("flat",)),
        },
        normalize=lambda raw, _fix_invalid=False: None,
        _shrink_for_numel_constraints=lambda config: None,
    )


def _bucket(m: int, n: int, k: int) -> dict[str, object]:
    return {
        "dtype": "fp16_bf16",
        "m_value": m,
        "n_value": n,
        "k_value": k,
    }


def test_matmul_heuristic_rules_have_unique_shape_buckets() -> None:
    data = json.loads(_B200_MATMUL_HEURISTICS_PATH.read_text())
    keys = [json.dumps(rule["shape_bucket"], sort_keys=True) for rule in data["rules"]]

    assert set(data) == {"rules"}
    assert len(keys) == len(set(keys))
    for rule in data["rules"]:
        assert set(rule) == {"shape_bucket", "templates"}
        assert set(rule["shape_bucket"]).issubset(_SHAPE_BUCKET_KEYS)
        for key in ("k_bucket", "m_bucket", "n_bucket"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, str) for item in values)
                assert all(item.startswith("(") for item in values)
                assert all(item.endswith(("]", ")")) for item in values)
        for key in ("k_value", "m_value", "n_value"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, int) for item in values)
        assert rule["templates"]
        assert all("template" not in template for template in rule["templates"])


def test_matmul_bucket_matching_generates_seed_config() -> None:
    seed = _seed_config_for_bucket(
        _bucket(1024, 1024, 1024),
        config_spec=_matmul_config_spec(),
    )

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]
    assert dict(seed)["l2_groupings"] == [2]

    assert (
        _seed_config_for_bucket(
            _bucket(128, 128, 128),
            config_spec=_matmul_config_spec(),
        )
        is None
    )


def test_matmul_fact_generates_compiler_seed_config() -> None:
    config_spec = _matmul_config_spec(matmul_facts=[_matmul_fact()])

    seed = _seed_config_for_config_spec(config_spec)

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    config_spec = _matmul_config_spec(
        matmul_facts=[_matmul_fact(), _matmul_fact()],
    )

    assert _seed_config_for_config_spec(config_spec) is None


def test_triton_b200_matmul_heuristic_gates_on_hardware() -> None:
    env = SimpleNamespace(device=None, config_spec=_matmul_config_spec())
    env.config_spec.matmul_facts.append(_matmul_fact())
    b200 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA B200",
        compute_capability="sm100",
    )
    h100 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA H100",
        compute_capability="sm90",
    )

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=b200,
    ):
        assert TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())
        seed = TritonB200MatmulHeuristic.get_seed_config(env, SimpleNamespace())

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=h100,
    ):
        assert not TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())


def test_b200_formula_subsumes_table_promotion_wiring() -> None:
    # The sm100 FORMULA owns the compiler default; the TABLE is demoted to a search seed.
    assert TritonB200FormulaMatmulHeuristic.promote_seed_to_default is True
    assert TritonB200MatmulHeuristic.promote_seed_to_default is False
    assert TritonB200FormulaMatmulHeuristic.HARDWARE_TARGETS == (("cuda", "sm100"),)
    # The formula is a subclass of the H100 budget formula (inherits _matmul_tile).
    assert issubclass(TritonB200FormulaMatmulHeuristic, TritonH100MatmulHeuristic)
    # Registered AFTER the table so it wins the last-promote-wins default loop.
    from helion._compiler.autotuner_heuristics import get_heuristics

    order = [h.__name__ for h in get_heuristics("triton")]
    assert order.index("TritonB200FormulaMatmulHeuristic") > order.index(
        "TritonB200MatmulHeuristic"
    )


def test_h100_base_tile_is_unchanged_by_tmem_budget() -> None:
    # TMEM_BUDGET is None on the sm90 base (no tensor memory), so the TMEM growth step is skipped and
    # the H100 formula stays byte-identical: a big compute-bound cube keeps the register-budget wide-N
    # [128, 256, 64] w8 s4 tile (num_sm=132).
    assert TritonH100MatmulHeuristic.TMEM_BUDGET is None
    assert TritonH100MatmulHeuristic._matmul_tile(4096, 4096, 4096, 2, 132, 1) == (
        128,
        256,
        64,
        8,
        4,
        1,
    )


def test_b200_tile_grows_to_fill_tmem_budget() -> None:
    # On sm100 the tile is grown against the TENSOR-MEMORY budget (the accumulator lives there, not in
    # registers): double whichever axis still fits, N first since it is the coalesced store axis. The
    # reservation includes the A operand at the largest bk the formula can emit, so the [256,256] fp32
    # accumulator -- which alone exactly fills tensor memory -- can never be reached, and the wide
    # [128,256] tile is the largest that fits.
    cls = TritonB200FormulaMatmulHeuristic
    sm = 148
    for m, n, k in ((2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)):
        assert cls._matmul_tile(m, n, k, 2, sm, 1)[:2] == (128, 256), (m, n, k)
    # Non-saturated batched dots grow too -- tensor memory is theirs to use as well.
    assert cls._matmul_tile(4096, 4096, 4096, 2, sm, 4)[:2] == (128, 256)
    # ...but a SATURATED batched dot keeps the small occupancy tile step (2.5) chose for it; growing it
    # back would undo that (measured: it inflates mamba-shaped tiles from [32,64] to [32,1024]).
    bm, bn = cls._matmul_tile(32, 1024, 4096, 2, sm, 1000)[:2]
    assert bm <= cls.SAT_TILE_BM and bn <= cls.SAT_TILE_BN, (bm, bn)


def test_b200_tmem_budget_matches_measured_hardware() -> None:
    # _tmem_bytes is a deliberate OVER-estimate in BYTES, checked against TMEM_BUDGET exactly like
    # _smem_bytes is checked against SMEM_BUDGET. Hardware reports tensor memory in COLUMNS (limit
    # 512), so the ground truth below is measured tmem_size in columns and what must agree is the
    # VERDICT: fits / does not fit. Ground truth is tmem_size read off compiled sm100 kernel metadata
    # in the worst case, i.e. with the A operand promoted into tensor memory.
    cls = TritonB200FormulaMatmulHeuristic
    hw_columns = {  # (bm, bn, bk) bf16 -> measured tmem columns, A promoted; hardware limit is 512
        (64, 64, 16): 128,
        (64, 128, 16): 256,
        (64, 256, 16): 512,
        (64, 512, 16): 512,
        (64, 1024, 16): 520,
        (128, 128, 16): 256,
        (128, 256, 16): 512,
        (128, 512, 16): 520,
        (128, 512, 32): 528,
        (128, 512, 64): 544,
        (128, 1024, 16): 1032,
        (256, 128, 16): 512,
        (256, 256, 16): 528,
        (
            256,
            256,
            32,
        ): 544,  # the CI failure: 512 for the acc + 32 for the promoted A operand
        (256, 256, 64): 576,
        (256, 256, 128): 640,
        (256, 512, 16): 1040,
    }
    for (bm, bn, bk), cols in hw_columns.items():
        fits_model = cls._tmem_bytes(bm, bn, bk, 2) <= cls.TMEM_BUDGET
        fits_hardware = cols <= 512
        assert fits_model == fits_hardware, (bm, bn, bk, cols)
    # Specifically: the [256,256] square is rejected, the wide [128,256] tile accepted.
    assert cls._tmem_bytes(256, 256, 16, 2) > cls.TMEM_BUDGET
    assert cls._tmem_bytes(128, 256, 64, 2) <= cls.TMEM_BUDGET
    # The budget is the real capacity: 128 lanes x 512 columns x 32 bit.
    assert cls.TMEM_BUDGET == 128 * 512 * 4
    # ...and a [256,256] fp32 accumulator alone exactly fills it, which is why no promoted A operand
    # can ever coexist with that square.
    assert cls.TMEM_BUDGET == 256 * 256 * 4
    # A tile too small for tcgen05 uses no tensor memory at all (measured tmem_size == 0), so it must
    # be charged nothing -- otherwise tiny decode tiles get rejected for a resource they never touch.
    assert cls._tmem_bytes(32, 4096, 16, 2) == 0
    assert cls._tmem_bytes(16, 512, 32, 4) == 0


def test_b200_smem_bytes_bounds_measured_hardware() -> None:
    # The epilogue's accumulator staging buffer (bm*bn*4) is invisible to the operand-ring formula and
    # is independent of num_stages, so a tile can fit the ring and still exceed SMEM. Values are
    # measured `shared` from compiled sm100 metadata; the model must be an UPPER bound on each.
    cls = TritonB200FormulaMatmulHeuristic
    # (bm, bn, bk, itemsize, num_stages) -> measured shared bytes (worst case over epilogue variants)
    measured = {
        (64, 64, 64, 2, 6): 32784,
        (128, 128, 64, 2, 6): 65552,
        (128, 256, 64, 2, 6): 131072,
        (256, 128, 32, 2, 6): 131072,
        (256, 256, 32, 2, 6): 262144,
        (
            256,
            256,
            16,
            2,
            6,
        ): 262144,  # ring shrinks with bk, the epilogue term does NOT
        (128, 128, 128, 4, 6): 786448,
    }
    for (bm, bn, bk, itemsize, ns), shared in measured.items():
        assert cls._smem_bytes(bm, bn, bk, itemsize, ns) >= shared, (bm, bn, bk, ns)
    # [256,256] is rejected by the epilogue term alone, at ANY bk -- the ring cannot rescue it.
    for bk in (16, 32, 64):
        assert cls._smem_bytes(256, 256, bk, 2, 1) > cls.SMEM_BUDGET
    # The slack is load-bearing: without it an otherwise-exact bound misses by the 16-byte mbarriers.
    assert cls.SMEM_SLACK >= 16


def test_sm90_conservative_accounting_is_inert() -> None:
    # sm90 must stay byte-identical, but NOT because the epilogue term is Blackwell-specific -- it is
    # not. Measured on an sm90 target, a [128,256,64] bf16 dot with an fp32 output reports
    # shared=131072 == bm*bn*4, exactly as sm100 does, so EPILOGUE_ACC_ITEMSIZE is set on the base.
    # What is sm100-only is ENFORCING the budget by shrinking the tile.
    cls = TritonH100MatmulHeuristic
    assert cls.TMEM_BUDGET is None  # no tensor memory on sm90
    assert cls._tmem_bytes(256, 256, 32, 2) == 0
    assert cls.EPILOGUE_ACC_ITEMSIZE == 4  # the term is arch-independent...
    assert (
        cls.ENFORCE_SMEM_BUDGET is False
    )  # ...but sm90 does not enforce it (model over-estimates)
    assert cls.SMEM_SLACK == 0

    # The epilogue term is a NO-OP on sm90 for an arithmetic reason: the accumulator lives in the
    # register file, so ACC_BUDGET caps the emittable tile at bm*bn == ACC_BUDGET. Binding would need
    # bm*bn * 4 > SMEM_BUDGET -- unreachable. Pin both halves of that argument so a future ACC_BUDGET
    # bump cannot silently make the term start binding on sm90 unnoticed.
    assert cls.ACC_BUDGET * cls.EPILOGUE_ACC_ITEMSIZE <= cls.SMEM_BUDGET
    emitted = {
        cls._matmul_tile(m, n, k, itemsize, 132, pinned_grid)[:2]
        for m in (1, 16, 256, 4096, 65536)
        for n in (1, 16, 256, 4096, 65536)
        for k in (16, 256, 4096)
        for itemsize in (1, 2, 4)
        for pinned_grid in (1, 4, 1000)
    }
    assert max(bm * bn for bm, bn in emitted) <= cls.ACC_BUDGET

    # And the sm90 tile is unchanged: still the register-budget [128, 256].
    assert cls._matmul_tile(4096, 4096, 4096, 2, 132, 1)[:3] == (128, 256, 64)
