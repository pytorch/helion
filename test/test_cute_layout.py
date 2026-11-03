"""CuTe index proofs, the thread budget, lane-guard lifting and tile-layout regressions."""

from __future__ import annotations

import ast
import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.indexing import is_cute_direct_iota_index
from helion._compiler.cute.indexing import is_cute_unit_stride_iota_index
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl


def test_unit_stride_iota_index_proof() -> None:
    graph = torch.fx.Graph()
    iota = graph.call_function(
        torch.ops.prims.iota.default,
        args=(32,),
        kwargs={"start": 0, "step": 1, "dtype": torch.int32, "device": "cpu"},
    )
    shifted = graph.call_function(torch.ops.aten.add.Tensor, args=(iota, 4))
    nonzero_start = graph.call_function(
        torch.ops.prims.iota.default,
        args=(32,),
        kwargs={"start": 4, "step": 1, "dtype": torch.int32, "device": "cpu"},
    )
    gathered = graph.call_function(torch.ops.aten.div.Tensor, args=(iota, 4))
    scaled = graph.call_function(
        torch.ops.aten.add.Tensor, args=(4, iota), kwargs={"alpha": 2}
    )

    assert is_cute_direct_iota_index(iota)
    assert is_cute_unit_stride_iota_index(iota)
    assert not is_cute_direct_iota_index(shifted)
    assert is_cute_unit_stride_iota_index(shifted)
    assert not is_cute_direct_iota_index(nonzero_start)
    assert is_cute_unit_stride_iota_index(nonzero_start)
    assert not is_cute_unit_stride_iota_index(gathered)
    assert not is_cute_unit_stride_iota_index(scaled)


@helion.kernel(
    backend="cute",
    static_shapes=False,
    config=helion.Config(
        # V=32 consumes one thread axis, forcing the static K=128 reduction
        # into a 32-thread x four-lane loop.  The four dependent reductions in
        # the body must therefore be split into ordered lane passes.
        block_sizes=[32],
        num_warps=1,
        num_stages=1,
        indexing="pointer",
    ),
)
def _state_update_and_project(
    state: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Small recurrent-state update with reduction and broadcast fan-out."""
    V = hl.specialize(state.size(0))
    K = hl.specialize(state.size(1))
    block_v = hl.register_block_size(1, V)
    out = torch.empty([V], dtype=q.dtype, device=q.device)
    for tile_v in hl.tile(V, block_size=block_v):
        if active[0] == 0:
            out[tile_v] = 0.0
        else:
            k_offsets = hl.arange(K)
            state_tile = state[tile_v, k_offsets]
            k_tile = k[k_offsets].float()
            k_tile = k_tile / torch.sqrt((k_tile * k_tile).sum() + 1e-6)
            residual = v[tile_v].float() - (state_tile * k_tile[None, :]).sum(-1)
            updated = state_tile + residual[:, None] * k_tile[None, :]
            q_tile = q[k_offsets].float()
            q_tile = q_tile / torch.sqrt((q_tile * q_tile).sum() + 1e-6)
            out[tile_v] = (updated * q_tile[None, :]).sum(-1).to(q.dtype)
            state[tile_v, k_offsets] = updated
    return out


def test_reduction_and_broadcast_fan_out_codegen_accepts_same_lane_update() -> None:
    """Dynamic exact strides make the in-place state recurrence splittable."""
    args = (
        torch.randn(128, 128),
        torch.randn(128, dtype=torch.bfloat16),
        torch.randn(128, dtype=torch.bfloat16),
        torch.randn(128, dtype=torch.bfloat16),
        torch.ones(1, dtype=torch.int32),
    )
    _state_update_and_project.reset()
    bound = _state_update_and_project.bind(args)
    code = bound.to_code(bound.config_spec.default_config())

    assert "_helion_lane_reduce" not in code
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts

    zero_stride_state = torch.as_strided(
        torch.randn(128),
        (128, 128),
        (1, 0),
    )
    zero_stride_bound = _state_update_and_project.bind((zero_stride_state, *args[1:]))
    assert zero_stride_bound is not bound
    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="potentially aliasing write",
    ):
        zero_stride_bound.to_code(zero_stride_bound.config_spec.default_config())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reduction_and_broadcast_fan_out() -> None:
    """The in-place state recurrence (reduction + broadcast fan-out, split into
    ordered lane passes) matches the reference for the active and the inactive
    branch."""
    state = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
    k = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
    q = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
    active = torch.ones(1, device=DEVICE, dtype=torch.int32)
    state_ref = state.clone()

    k_ref = torch.nn.functional.normalize(k.float(), dim=0)
    q_ref = torch.nn.functional.normalize(q.float(), dim=0)
    residual = v.float() - (state_ref * k_ref[None, :]).sum(-1)
    state_ref += residual[:, None] * k_ref[None, :]
    expected = (state_ref * q_ref[None, :]).sum(-1).to(q.dtype)

    actual = _state_update_and_project(state, k, q, v, active)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, state_ref, rtol=2e-4, atol=2e-4)

    inactive_state = torch.randn_like(state)
    inactive_ref = inactive_state.clone()
    inactive = torch.zeros_like(active)
    inactive_out = _state_update_and_project(inactive_state, k, q, v, inactive)
    torch.testing.assert_close(inactive_out, torch.zeros_like(inactive_out))
    torch.testing.assert_close(inactive_state, inactive_ref)


@onlyBackends(["cute"])
class TestThreadBudget(unittest.TestCase):
    """Test thread budget validation."""

    def test_under_limit(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit

        # Should not raise
        check_thread_limit(1024)
        check_thread_limit(512)
        check_thread_limit(1)

    def test_over_limit(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit
        from helion.exc import BackendUnsupported

        with patch(
            "helion._compiler.compile_environment.CompileEnvironment.current"
        ) as current:
            current.return_value.backend.name = "cute"
            with self.assertRaises(BackendUnsupported):
                check_thread_limit(1025)

    def test_over_limit_message(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit
        from helion.exc import BackendUnsupported

        with patch(
            "helion._compiler.compile_environment.CompileEnvironment.current"
        ) as current:
            current.return_value.backend.name = "cute"
            with self.assertRaises(BackendUnsupported) as ctx:
                check_thread_limit(2048, context="(64, 64)")
        self.assertIn("(64, 64)", str(ctx.exception))


@onlyBackends(["cute"])
class TestLaneGuardLifting(unittest.TestCase):
    """Lifting a lane-invariant guard out of a lane loop around a collective."""

    @staticmethod
    def _guarded_lane_reduction(*, thread_dependent: bool = False) -> ast.For:
        from helion._compiler.cute.lane_loops.markers import _create_lane_loop

        condition = (
            "cutlass.Int32(cute.arch.thread_idx()[0]) == 0"
            if thread_dependent
            else "external_flag"
        )
        prefix = ast.parse(f"lane_value = lane + 1\ncondition = {condition}").body
        marker = ast.parse(
            "reduced = _helion_lane_reduce("
            "lane_value, 'sum', cutlass.Float32(0), threads_in_group=32)"
        ).body
        guard = ast.If(
            test=ast.Name(id="condition", ctx=ast.Load()),
            body=marker,
            orelse=[],
        )
        return _create_lane_loop("lane", 4, [*prefix, guard])

    def test_lane_guard_requires_uniform_external_dependencies(self) -> None:
        """A guard is not lifted when an external value is not proven uniform."""
        from helion import exc
        from helion._compiler.cute.lane_loops.split import _lift_lane_invariant_if
        from helion._compiler.cute.lane_loops.split import split_lane_loop_reductions

        loop = self._guarded_lane_reduction()
        self.assertIsNone(_lift_lane_invariant_if(loop, "lane", {"cutlass"}))
        with self.assertRaisesRegex(exc.BackendUnsupported, "thread uniformity"):
            split_lane_loop_reductions([loop])

    def test_lane_guard_rejects_thread_dependent_condition(self) -> None:
        """A CUDA-thread-dependent guard cannot enclose a collective."""
        from helion._compiler.cute.lane_loops.split import _lift_lane_invariant_if

        loop = self._guarded_lane_reduction(thread_dependent=True)
        self.assertIsNone(_lift_lane_invariant_if(loop, "lane", {"cutlass", "cute"}))

    def test_lane_guard_empty_branch_preserves_lane_prefix(self) -> None:
        """Lifting an empty branch keeps unconditional per-lane statements."""
        from helion._compiler.cute.lane_loops.split import _lift_lane_invariant_if

        loop = self._guarded_lane_reduction()
        lifted = _lift_lane_invariant_if(
            loop,
            "lane",
            {"cutlass", "external_flag"},
        )
        self.assertIsNotNone(lifted)
        assert lifted is not None
        outer_if = lifted[-1]
        self.assertIsInstance(outer_if, ast.If)
        assert isinstance(outer_if, ast.If)
        self.assertEqual(len(outer_if.orelse), 1)
        self.assertIsInstance(outer_if.orelse[0], ast.For)
        self.assertIn("lane_value = lane + 1", ast.unparse(outer_if.orelse[0]))


def _recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = state.float() * alpha.float()[:, :, None]
    kv = k.float()[:, :, None]
    if beta is not None:
        kts = (kv * s).sum(1)
        s = s - beta.float()[:, None, None] * kv * kts[:, None, :]
        s = s + beta.float()[:, None, None] * kv * v.float()[:, None, :]
    else:
        s = s + kv * v.float()[:, None, :]
    out = (q.float()[:, :, None] * s).sum(1)
    return out, s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("correction", [False, True])
def test_unit_tile_beside_a_sliced_dim_matches_reference(correction: bool) -> None:
    """``hl.tile([BH, DV], block_size=[1, None])`` at ``block_sizes=[1]``: every
    value is a ``[D]`` or ``[D, 1]`` tile spread over the D threads.  The
    ``a[:, None]`` broadcast used to get a one-float shared-memory round trip
    written by all D threads (a seed that only saw the unit tile), returning
    garbage for the state and the output."""
    from helion.experimental.linear_attention import recurrent_step_correction_fused
    from helion.experimental.linear_attention import recurrent_step_fused

    source = recurrent_step_correction_fused if correction else recurrent_step_fused
    kernel = helion.kernel(
        source.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    BH, D, DV = 64, 128, 128
    torch.manual_seed(0)
    q = torch.randn(BH, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(BH, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(BH, DV, device=DEVICE, dtype=torch.bfloat16)
    initial = torch.randn(BH, D, DV, device=DEVICE)
    alpha = torch.rand(BH, D, device=DEVICE)
    beta = torch.rand(BH, device=DEVICE) if correction else None
    state = initial.clone()
    args = (q, k, v, state, alpha, *(() if beta is None else (beta,)))
    bound = kernel.bind(args)
    config = bound.config_spec.normalized_config(
        helion.Config(block_sizes=[1], reduction_loops=[None])
    )
    assert config.block_sizes == [1]
    compiled = bound.compile_config(config)
    out, new_state = compiled(*args)
    torch.cuda.synchronize()
    expected_out, expected_state = _recurrent_reference(q, k, v, initial, alpha, beta)
    torch.testing.assert_close(new_state, expected_state, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.float(), expected_out, rtol=2e-2, atol=2e-1)
    assert torch.equal(state, initial)
