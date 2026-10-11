"""The grouped-worklist heuristic's promoted default seed runs exactly.

The seed's root body reads ``tile_m.index`` only for the masks of the operand
load and the store, both taken over by the grouped tcgen05 matmul.  With the
root lane loops suppressed, the generic lowering of those masks
(``v = operator.lt(indices_0, store_m)``) would read a coordinate nothing
defines, so dead assignments computed from lane-loop definitions are deleted
and the kernel launches (the heuristic tests only generate the code).  A side
store of a worklist entry checks the residual statements of the grouped
persistent schedule's shared loop; one of ``store_m``, which the schedule
rewrites to a constant, is refused.
"""

from __future__ import annotations

import pytest
import torch

from test.test_autotuner_heuristics import _grouped_worklist_args
from test.test_autotuner_heuristics import _grouped_worklist_bind_patches

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl


def _worklist_matmul_fn(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
    side: torch.Tensor,
    side_of_store_m: hl.constexpr,
) -> torch.Tensor:
    m_total, k = a_packed.shape
    _groups, n, _k = b_grouped.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64, 128)
    out = torch.empty([m_total, n], dtype=a_packed.dtype, device=a_packed.device)
    for work_tile, tile_m, tile_n in hl.tile(
        [worklist.size(0), 256, n], block_size=[1, block_m, block_n]
    ):
        work_id = work_tile.begin
        group_id = worklist[work_id, 0]
        start = worklist[work_id, 1]
        valid_m = worklist[work_id, 2]
        store_m = worklist[work_id, 3]
        local_m = tile_m.index
        row = start + local_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_block = hl.load(
                a_packed,
                [row, tile_k],
                extra_mask=(local_m < valid_m)[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(acc, a_block, b_grouped[group_id, tile_n, tile_k].T)
        hl.store(
            out,
            [row, tile_n],
            acc.to(out.dtype),
            extra_mask=(local_m < store_m)[:, None],  # pyrefly: ignore[bad-index]
        )
        if side_of_store_m:
            side[work_tile.id] = store_m.to(torch.float32)
        else:
            side[work_tile.id] = worklist[work_id, 3].to(torch.float32)
    return out


_worklist_matmul = helion.kernel(
    _worklist_matmul_fn, backend="cute", static_shapes=True
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("source_m_tile", "b_major"), [(32, "k"), (256, "n")])
@skipUnlessBackends(["cute"])
def test_promoted_grouped_worklist_seed_runs_exactly(
    source_m_tile: int, b_major: str
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(0)
    with _grouped_worklist_bind_patches(analysis_runtime_n_ptx=True):
        a, b, worklist = _grouped_worklist_args(
            row_extent=source_m_tile, b_major=b_major
        )
        a.copy_(torch.randn(a.shape, device=a.device))
        b.copy_(torch.randn(b.shape, device=b.device))
        side = torch.zeros(worklist.size(0), device=a.device)
        args = (a, b, worklist, side, False)
        bound = _worklist_matmul.bind(args)
        spec = bound.config_spec
        # As the heuristic test does for the cluster_m=2 seeds.
        spec._cute_tcgen05_config.cluster_m2_search_constraints = None
        config = spec.default_config()
        assert config.config["tcgen05_grouped_mode"] == "worklist_nm"
        assert str(config.config["pid_type"]).startswith("persistent")
        code = bound.to_code(config)
    assert "indices_0" not in code
    bound.set_config(config)
    out = bound(*args)
    expected = torch.empty_like(out)
    for group, start, _valid_m, store_m in worklist.tolist():
        rows = slice(start, start + store_m)
        expected[rows] = (a[rows].float() @ b[group].float().T).to(out.dtype)
    torch.testing.assert_close(out, expected, atol=0.25, rtol=1e-2)
    torch.testing.assert_close(side, worklist[:, 3].float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@skipUnlessBackends(["cute"])
def test_side_store_of_rewritten_worklist_scaffolding_is_refused() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    with _grouped_worklist_bind_patches(analysis_runtime_n_ptx=True):
        a, b, worklist = _grouped_worklist_args(row_extent=32, b_major="k")
        side = torch.zeros(worklist.size(0), device=a.device)
        bound = _worklist_matmul.bind((a, b, worklist, side, True))
        config = bound.config_spec.default_config()
        config = helion.Config(
            **{**config.config, "tcgen05_grouped_mode": "worklist_nm"}
        )
        with pytest.raises(helion.exc.BackendUnsupported, match="store_m"):
            bound.to_code(config)
