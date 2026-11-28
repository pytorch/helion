"""A four-warp local selection may hand off only complete, privately aliased rows."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.analysis.selection_handoff import match_selection_handoff
from helion._compiler.cute.selection_handoff import _local_buffers
from helion._compiler.cute.selection_handoff import _local_effects
from helion._compiler.cute.selection_handoff import normalize
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_selection_handoff import repeated_selections
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def test_replacing_a_local_fill_requires_private_registers() -> None:
    def holds(alloc: str, write: str = "keys.fill(0)", suffix: str = "") -> bool:
        f = ast.parse(
            f"def kernel(audit):\n    for i in range(4):\n        keys = {alloc}\n        {write}\n    {suffix or 'pass'}"
        ).body[0]
        assert isinstance(f, ast.FunctionDef)
        loop = f.body[0]
        assert isinstance(loop, ast.For)
        return _local_effects(loop, [], _local_buffers(f, loop))

    rmem = "cute.make_rmem_tensor(1, cutlass.Int64)"
    assert holds(rmem)
    assert holds(rmem, "keys[0] = 1")
    assert not holds("audit")
    assert not holds(rmem, "audit[i] = 1")
    assert not holds(rmem, "keys[0] = 1", "audit[0] = keys[0]")


@helion.kernel(backend="cute", static_shapes=True)
def _two_selections(x: torch.Tensor, other: torch.Tensor, mode: hl.constexpr):
    width = x.size(1)
    count = width // 32
    scores = torch.empty_like(x)
    cv = torch.empty((1, count, 8), dtype=torch.float32, device=x.device)
    ci = torch.empty((1, count, 8), dtype=torch.int32, device=x.device)
    cf = cv.view(1, count * 8)
    fi = ci.view(1, count * 8)
    outv = torch.empty((1, 8), dtype=torch.float32, device=x.device)
    outi = torch.empty((1, 8), dtype=torch.int32, device=x.device)
    y = torch.empty_like(outv)
    for r in hl.tile(width, block_size=32):
        scores[0, r] = x[0, r] * 2
    for row, group in hl.tile([1, count], block_size=[1, 1]):
        j = group.begin * 32 + hl.arange(32)
        q, ix = torch.topk(scores[row.begin, j], 8)
        cv[row.begin, group.begin, :] = q
        ci[row.begin, group.begin, :] = ix.to(torch.int32) + group.begin * 32
        if mode == "extra_store":
            scores[row.begin, j] = torch.zeros_like(scores[row.begin, j])
    for row in hl.tile(1, block_size=1):
        v, i = torch.topk(cf[row.begin, :], 8)
        if mode == "unrelated":
            outi[row.begin, :] = other[row.begin, i]
        else:
            outi[row.begin, :] = fi[row.begin, i]
        outv[row.begin, :] = v
    for r in hl.tile(8, block_size=8):
        y[0, r] = outv[0, r] + outi[0, r].to(torch.float32)
    return y, outi, cv, ci


@pytest.mark.parametrize(
    ("width", "mode", "expected"),
    [
        (128, "normal", True),
        (96, "normal", False),
        (128, "unrelated", False),
        (128, "extra_store", False),
    ],
)
def test_selection_handoff_requires_both_views_and_complete_effects(
    width, mode, expected
):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            _two_selections,
            (
                torch.empty(1, width),
                torch.empty(1, width // 4, dtype=torch.int32),
                mode,
            ),
        )
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            plan = match_selection_handoff(bound.env, ir)
            assert bound.config_spec.cute_selection_handoff_structure
            assert (plan is not None) == expected
            cfg = {
                "cute_cooperative_selection_merge": True,
                "cute_fragment_topk_network": True,
                "cross_loop_pipeline": "barrier",
                "pid_type": "persistent_interleaved",
            }
            if expected:
                assert plan is not None and plan.producer != plan.consumer
                normalize(bound.config_spec, cfg, fix_invalid=False)
                # A requested but unavailable network is dropped during repair.
                # The dependent merge must be dropped in the same pass.
                original_roots = bound.config_spec.cute_fragment_topk_network_root_ids
                try:
                    bound.config_spec.cute_fragment_topk_network_root_ids = frozenset()
                    repaired = helion.Config.from_dict(
                        dict(bound.config_spec.default_config()) | cfg
                    )
                    bound.config_spec.normalize(repaired, _fix_invalid=True)
                    assert not repaired.get("cute_fragment_topk_network")
                    assert not repaired.get("cute_cooperative_selection_merge")
                    bound.config_spec.normalize(repaired)
                finally:
                    bound.config_spec.cute_fragment_topk_network_root_ids = (
                        original_roots
                    )
                for incompatible in (
                    {"cross_loop_pipeline": "off"},
                    {"cute_phase_participation": True},
                    {"cute_fragment_topk_network": False},
                ):
                    with pytest.raises(
                        exc.InvalidConfig, match="adjacent selection pair"
                    ):
                        normalize(
                            bound.config_spec, cfg | incompatible, fix_invalid=False
                        )
            else:
                with pytest.raises(exc.InvalidConfig, match="adjacent selection pair"):
                    normalize(bound.config_spec, cfg, fix_invalid=False)


@pytest.mark.parametrize(
    ("static", "first_largest"), [(False, False), (False, True), (True, False)]
)
def test_shared_buffers_select_only_the_proved_pair(static, first_largest):
    with cpu_render_target(fake_device_sm_count=148):
        k = helion.kernel(backend="cute", static_shapes=static)(repeated_selections)
        bound = cpu_bind(k, (torch.empty(1, 128), torch.empty(1, 117), first_largest))
        cfg = helion.Config(
            pid_type="persistent_interleaved",
            cross_loop_pipeline="barrier",
            cute_fragment_topk_network=True,
            num_threads=[128, 1],
        )
        candidate = helion.Config.from_dict(
            cfg.config | {"cute_cooperative_selection_merge": True}
        )
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            pair = match_selection_handoff(bound.env, ir)
            assert pair is not None
            ordinal = ir.root_ids.index(pair.producer)
            assert ordinal == (1 if first_largest else 4)
        off = bound.to_code(cfg)
        barrier = "_cute_grid_barrier(x_grid_sem.iterator)"
        assert off.count(barrier) == 6
        if not static and not first_largest:
            # The valid selection is after the dynamic root. Do not silently
            # substitute the earlier smallest-first producer whose buffers fit.
            with pytest.raises(
                exc.InvalidConfig,
                match="static dispatch intervals for the proved FX roots",
            ):
                bound.to_code(candidate)
            return
        on = bound.to_code(candidate)
        assert on.count(barrier) == 5
        assert on.partition("selection_slot =")[0].count(barrier) == ordinal
