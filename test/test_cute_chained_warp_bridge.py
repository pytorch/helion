from __future__ import annotations

from dataclasses import replace
import importlib
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_matmul import _plain_chain
from .test_cute_chained_matmul import _scan_cached_chain
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_warp_stage as roots
from helion._compiler.cute import chained_warp_bridge as bridge
from helion._compiler.cute import chained_warp_stage as shared
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _right_bridge(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, m, q = v.shape
    r, k = a.shape[1:]
    out = torch.empty((batches, m, r), device=a.device, dtype=a.dtype)
    for batch, row in hl.tile([batches, m], block_size=[1, 16]):
        bi = batch.begin
        kk, qq, rr = hl.arange(k), hl.arange(q), hl.arange(r)
        first = hl.dot(a[bi, rr, kk], b[bi, qq, kk].T)
        weights = torch.where(rr[:, None] >= qq[None, :], first * 0.5, 0.0)
        result = hl.dot(v[bi, row, qq], weights.to(a.dtype).T)
        out[bi, row, rr] = result.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _escaped_bridge(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, seed: hl.constexpr
) -> torch.Tensor:
    m, k = a.shape
    q = b.shape[0]
    out = torch.empty((m, q), device=a.device, dtype=torch.float32)
    for row in hl.tile(m, block_size=16):
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[row, kk], b[qq, kk].T)
        if seed:
            result = hl.dot(first.to(a.dtype), v[qq, :], acc=first)
        else:
            result = hl.dot(first.to(a.dtype), v[qq, :]) + first
        out[row, qq] = result
    return out


def _code(dtype=torch.bfloat16, scan=False, warps=4, mode="direct"):
    with _cpu_codegen():
        args = tuple(
            torch.empty(shape, dtype=dtype)
            for shape in (
                (2, 128, 64),
                (2, 128, 64),
                (2, 128, 32),
            )
        )
        kernel = _plain_chain
        if scan:
            kernel = _scan_cached_chain
            args = (
                *args,
                torch.empty((2, 160), dtype=dtype),
                torch.empty((2, 160), dtype=dtype),
                torch.empty((128,), dtype=torch.int64),
                mode,
            )
        return kernel._bind_isolated(args).to_code(
            helion.Config(
                num_warps=warps,
                cute_chained_mma_schedule=(
                    "cp_async_register_reuse_scan" if scan else "cp_async_register"
                ),
            )
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("scan", (False, True))
@pytest.mark.parametrize("warps", (4, 8))
def test_original_source_and_both_shared_stages(dtype, scan, warps):
    initialized = torch.cuda.is_initialized()
    with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
        old = _code(dtype, scan, warps)
    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as stages:
        new = _code(dtype, scan, warps)
    assert old == new
    assert stages.call_count == 2
    first, last = [call.args[3] for call in stages.call_args_list]
    assert isinstance(first.result, shared.WarpRegisterResult)
    assert isinstance(last.result, shared.WarpRootResult)
    assert first.result.fragment is not None
    assert first.result.fragment._consumed
    assert first.threads == first.completion.execution.threads
    assert "chain_0_c_ptr =" not in new
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("mode", ("shift", "gather", "multiple", "scan_only"))
def test_original_scan_cache_and_mask_variants(mode):
    with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
        old = _code(scan=True, mode=mode)
    assert _code(scan=True, mode=mode) == old


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("change", ("map", "cg", "config", "graph", "bridge"))
def test_actual_context_rejects_before_stage(stage, change):
    original = bridge.RootWarpBridgeSequence.emit

    def changed(self, cg, plan, boundaries, staged, scratch, prefix, current):
        if current == stage:
            if change == "map":
                boundaries = {**boundaries, plan.dots[current].args[0]: "unapproved"}
            elif change == "cg":
                cg = object()
            elif change == "config":
                cg.device_function.config.config["block_sizes"].append(16)
            elif change == "graph":
                plan.dots[current].kwargs = {"unapproved": True}
            else:
                object.__setattr__(self.bridge, "lines", ())
        return original(self, cg, plan, boundaries, staged, scratch, prefix, current)

    with (
        patch.object(bridge.RootWarpBridgeSequence, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code()


@pytest.mark.parametrize(
    "change", ("map", "cg", "probes", "layout", "stride", "inner", "order")
)
def test_actual_bridge_arguments_reject_without_consumption(change):
    original = bridge.WarpBridgeInput.emit
    checked = []

    def changed(self, cg, plan, boundaries, probes, stage, layout, selected):
        if change == "map":
            boundaries = {**boundaries, selected.source: "unapproved"}
        elif change == "cg":
            cg = object()
        elif change == "probes":
            probes = {**probes, selected.source: "unapproved"}
        elif change == "layout":
            layout = replace(layout, shape=(16, 16))
        elif change == "stride":
            layout = replace(layout, stride=(1, 1))
        elif change == "inner":
            layout = replace(layout, inner=1 - layout.inner)
        else:
            stage = 0
        with pytest.raises(chain._UnsupportedChain):
            original(self, cg, plan, boundaries, probes, stage, layout, selected)
        assert not self.fragment._consumed
        assert not self._emitted
        checked.append(True)
        raise chain._UnsupportedChain("rejected invalid warp bridge")

    with (
        patch.object(bridge.WarpBridgeInput, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code()
    assert checked == [True]


@pytest.mark.parametrize(
    "change",
    (
        "thread",
        "warp",
        "threads",
        "sync",
        "a_workspace",
        "b_workspace",
        "tmem",
        "barriers",
    ),
)
def test_completed_fragment_execution_is_deeply_checked(change):
    original = bridge.RootWarpBridgeSequence.emit

    def changed(self, cg, plan, boundaries, staged, scratch, prefix, stage):
        if stage == 1:
            fragment = self._state.fragment
            assert fragment is not None
            execution = fragment.prepared.completion.execution
            object.__setattr__(execution, change, 32 if change == "threads" else "bad")
        return original(self, cg, plan, boundaries, staged, scratch, prefix, stage)

    with (
        patch.object(bridge.RootWarpBridgeSequence, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code()


@pytest.mark.parametrize("change", ("lost", "prefix", "early_consumed", "obligation"))
def test_retained_completion_obligation_cannot_disappear(change):
    original = bridge.RootWarpBridgeSequence.emit

    def changed(self, cg, plan, boundaries, staged, scratch, prefix, stage):
        if stage == 1:
            if change == "lost":
                self._state.fragment = None
            elif change == "prefix":
                prefix = prefix[:-1]
            elif change == "obligation":
                object.__setattr__(self, "obligation", ())
            else:
                assert self._state.fragment is not None
                object.__setattr__(self._state.fragment, "_consumed", True)
        return original(self, cg, plan, boundaries, staged, scratch, prefix, stage)

    with (
        patch.object(bridge.RootWarpBridgeSequence, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code()


def test_failed_rhs_does_not_consume_fragment_or_publish_c():
    original = roots.emit_original_warp_operands
    checked = []

    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        selected = kwargs.get("bridge_input")
        if selected is not None:
            assert selected._emitted
            assert not selected.fragment._consumed
            assert selected.bridge.source not in args[2]
            checked.append(True)
            raise chain._UnsupportedChain("failed original warp rhs")
        return result

    with (
        patch.object(roots, "emit_original_warp_operands", failed),
        pytest.raises(exc.BackendUnsupported, match="failed original warp rhs"),
    ):
        _code()
    assert checked == [True]


def test_original_fragment_and_sequence_are_consumed_once():
    original = bridge.RootWarpBridgeSequence.emit
    checked = []

    def checked_emit(self, cg, plan, boundaries, staged, scratch, prefix, stage):
        prior = dict(boundaries)
        result = original(self, cg, plan, boundaries, staged, scratch, prefix, stage)
        with pytest.raises(chain._UnsupportedChain):
            original(self, cg, plan, prior, staged, scratch, prefix, stage)
        if stage == 1:
            selected = self._state.bridge_input
            assert selected is not None
            with pytest.raises(chain._UnsupportedChain):
                selected.complete(cg, plan, prior)
        checked.append(stage)
        return result

    with patch.object(bridge.RootWarpBridgeSequence, "emit", checked_emit):
        _code()
    assert checked == [0, 1]


def test_no_new_bridge_selected_when_original_proof_declines():
    with patch.object(chain, "_register_bridges", return_value={}):
        with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
            old = _code()
        with patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as stages:
            new = _code()
    assert old == new and stages.call_count == 0
    assert "chain_0_c_ptr =" in new


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_original_b_role_coordinate_bridge(dtype):
    with _cpu_codegen():
        args = tuple(
            torch.empty(shape, dtype=dtype)
            for shape in (
                (2, 16, 64),
                (2, 128, 64),
                (2, 32, 128),
            )
        )
        config = helion.Config(
            num_warps=4, cute_chained_mma_schedule="cp_async_register"
        )
        with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
            old = _right_bridge._bind_isolated(args).to_code(config)
        with patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as stages:
            new = _right_bridge._bind_isolated(args).to_code(config)
    assert old == new
    assert stages.call_count == 2
    assert stages.call_args_list[0].args[3].result.bridge.role == "b"


@pytest.mark.parametrize("seed", (False, True))
def test_escaping_fp32_and_explicit_accumulator_keep_original(seed):
    with _cpu_codegen():
        args = (
            *tuple(
                torch.empty(shape, dtype=torch.bfloat16)
                for shape in (
                    (32, 64),
                    (128, 64),
                    (128, 128),
                )
            ),
            seed,
        )
        config = helion.Config(
            num_warps=4, cute_chained_mma_schedule="cp_async_register"
        )
        with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
            old = _escaped_bridge._bind_isolated(args).to_code(config)
        with patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as stages:
            new = _escaped_bridge._bind_isolated(args).to_code(config)
    assert old == new
    assert stages.call_count == 0
    assert "chain_0_c_ptr =" in new


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
@pytest.mark.parametrize("role", ("a", "b"))
@pytest.mark.parametrize("inner", (0, 1))
@pytest.mark.parametrize("threads", (128, 256))
def test_actual_bridge_partition_payload_and_ordered_consumer_slots(
    dtype_name, role, inner, threads
):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = cute.make_mma_atom(
                cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
            )
            producer = cute.make_tiled_mma(atom, atom_layout_mnk=(1, threads // 32, 1))
            shape = (16, 128)
            stride = (136, 1) if inner == 1 else (1, 24)
            layout = cute.make_layout(shape, stride=stride)
            image = cute.make_tensor(0, layout)
            values = {}
            for thread in range(threads):
                part = producer.get_slice(thread)
                coords = part.partition_C(cute.make_identity_tensor(shape))
                target = part.partition_C(image)
                source = cute.make_rmem_tensor(
                    producer.partition_shape_C(shape), cutlass.Float32
                )
                narrowed = cute.make_rmem_tensor(source.shape, dtype)
                assert cute.size(source) == cute.size(narrowed) == cute.size(target)
                for slot in range(cute.size(source)):
                    coordinate = tuple(map(int, coords[slot]))
                    address = int(target[slot])
                    assert address == int(layout(coordinate))
                    assert address not in values
                    values[address] = coordinate
            assert set(values.values()) == {
                (r, k) for r in range(16) for k in range(128)
            }
            # The original bridge publishes with the producer's partition_C;
            # the next MMA has a different team and A/B partition, not an RMEM alias.
            consumer_threads = 64
            consumer = cute.make_tiled_mma(
                atom, atom_layout_mnk=(1, consumer_threads // 32, 1)
            )
            read = set()
            for thread in range(consumer_threads):
                part = consumer.get_slice(thread)
                partition = part.partition_A if role == "a" else part.partition_B
                source = partition(image)
                coords = partition(cute.make_identity_tensor(shape))
                assert cute.size(source, mode=[2]) == 8
                for k in range(8):
                    panel = source[None, None, k]
                    coordinates = coords[None, None, k]
                    for slot in range(cute.size(panel)):
                        coordinate = tuple(map(int, coordinates[slot]))
                        assert values[int(panel[slot])] == coordinate
                        assert k * 16 <= coordinate[1] < (k + 1) * 16
                        read.add(coordinate)
            assert read == set(values.values())
        assert module.operation.verify()


@pytest.mark.parametrize(
    "change", ("map", "prefix", "lost", "config", "staged", "aliases")
)
def test_final_consumption_rechecks_complete_original_sequence(change):
    original = bridge.RootWarpBridgeSequence.validate

    def changed(self, cg, plan, boundaries, staged, prefix):
        if change == "map":
            boundaries = {**boundaries, self.bridge.source: "unallocated"}
        elif change == "prefix":
            prefix = prefix[:-1]
        elif change == "lost":
            self._state.bridge_input = None
        elif change == "config":
            cg.device_function.config.config["block_sizes"].append(16)
        elif change == "aliases":
            original_names = dict(self._state.fragment.aliases)
            late = [key for key in plan.tensor_aliases if key not in original_names]
            assert late
            plan.tensor_aliases[late[0]] = "unapproved_after_stage1"
        else:
            staged = []
        return original(self, cg, plan, boundaries, staged, prefix)

    with (
        patch.object(bridge.RootWarpBridgeSequence, "validate", changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code(scan=True)


@pytest.mark.parametrize("phase", ("input", "fragment", "final"))
@pytest.mark.parametrize("field", ("storage", "operand0", "operand1"))
def test_effective_typed_plan_change_rejects_at_every_phase(phase, field):
    def mutate(plan):
        if field == "storage":
            object.__setattr__(plan, "dtype", torch.float16)
        else:
            assert plan.region is not None
            stage = 0 if field == "operand0" else 1
            object.__setattr__(
                plan.region.contractions[stage],
                "operand_dtypes",
                (torch.float16, torch.float16),
            )

    if phase == "input":
        owner, name = shared, "emit_prepared_warp_stage"
        emit_stage = shared.emit_prepared_warp_stage

        def changed(cg, plan, boundaries, prepared, prefix):
            mutate(plan)
            return emit_stage(cg, plan, boundaries, prepared, prefix)
    elif phase == "fragment":
        owner, name = bridge.RootWarpBridgeSequence, "emit"
        emit_sequence = bridge.RootWarpBridgeSequence.emit

        def changed(self, cg, plan, boundaries, staged, scratch, prefix, stage):
            if stage == 1:
                mutate(plan)
            return emit_sequence(
                self, cg, plan, boundaries, staged, scratch, prefix, stage
            )
    else:
        owner, name = bridge.RootWarpBridgeSequence, "validate"
        validate_sequence = bridge.RootWarpBridgeSequence.validate

        def changed(self, cg, plan, boundaries, staged, prefix):
            mutate(plan)
            return validate_sequence(self, cg, plan, boundaries, staged, prefix)

    with (
        patch.object(owner, name, changed),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _code()
