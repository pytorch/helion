from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute import chained_warp_pair
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_warp_mma import emit_warp_mma
from helion._compiler.cute.chained_warp_pair import native_pair_copies
from helion._compiler.cute.chained_warp_pair import native_pair_layout
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pair(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    fanout: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
):
    output = torch.empty(
        (a.size(0), a.size(1), c.size(2)), dtype=torch.float32, device=a.device
    )
    for batch, row in hl.tile([a.size(0), a.size(1)], block_size=[1, None]):
        first = hl.dot(a[batch.id, row, :], b[batch.id, :, :])
        operand = (-first).to(a.dtype)
        second = hl.dot(operand, c[batch.id, :, :])
        if fanout:
            second = second + first
        output[batch.id, row, :] = second
    return output


def _args(dtype, rows=16, middle=16, reduction=16, *, device="cpu", transpose=False):
    views, backing = [], []
    for m, n in ((rows, reduction), (reduction, middle), (middle, middle)):
        raw = torch.full((2, m + 2, n + 4), -19, dtype=dtype, device=device)
        if transpose:
            raw = torch.full((2, n + 4, m + 2), -19, dtype=dtype, device=device)
            view = raw[:, 2 : n + 2, 1 : m + 1].transpose(1, 2)
        else:
            view = raw[:, 1 : m + 1, 2 : n + 2]
        view.normal_(0, 0.0625)
        views.append(view)
        backing.append(raw)
    return tuple(views), tuple(backing)


def _config(rows=16, schedule="coalesced", **kwargs):
    return helion.Config(
        block_sizes=[rows], num_warps=4, cute_chained_mma_schedule=schedule, **kwargs
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 8, 8), (16, 16, 16), (32, 32, 32)])
def test_actual_native_storage_and_every_k_panel(dtype, shape) -> None:
    proof = native_pair_layout(shape, shape, dtype, dtype)
    assert proof is not None
    assert proof.threads == shape[0] * 2
    assert proof.slots * proof.threads == shape[0] * shape[1]
    assert proof.panels == max(1, shape[2] // 16)


@pytest.mark.parametrize(
    "producer,consumer",
    [
        ((8, 8, 8), (8, 8, 8)),
        ((16, 7, 8), (16, 8, 7)),
        ((16, 16, 16), (32, 16, 16)),
        ((32, 64, 32), (32, 32, 64)),
    ],
)
def test_incomplete_or_unsupported_native_ownership_declines(
    producer, consumer
) -> None:
    assert native_pair_layout(producer, consumer, torch.float16, torch.float16) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 8, 8), (16, 16, 16), (19, 32, 32)])
def test_original_source_and_typed_pair_codegen(dtype, shape) -> None:
    args, _ = _args(dtype, *shape, transpose=shape[0] == 19)
    rows = 32 if shape[0] > 16 else 16
    with _cpu_codegen():
        bound = _pair._bind_isolated(args)
        config = _config(rows)
        original = bound.to_code(config)
        assert (
            bound.to_code(_config(rows, cute_chained_warp_register_a=False)) == original
        )
        changed = bound.to_code(_config(rows, cute_chained_warp_register_a=True))
    assert "chain_pair_0_a" in changed
    assert "chain_1_copy_a =" not in changed
    assert "chain_1_ra[None, None, chain_1_kk]" in changed
    for call in ("sync_threads()", "alloc_smem(", "autovec_copy("):
        assert changed.count(call) == original.count(call)
    tree = ast.parse(changed)
    declarations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "chain_pair_0_a" for t in node.targets
        )
    ]
    assert len(declarations) == 1
    assert "if chain_thread < " in changed
    assert "chain_0_c = cute.make_tensor" in changed
    assert "chain_1_c = cute.make_tensor" in changed


def test_fanout_and_configuration_exclusions_fail_closed() -> None:
    args, _ = _args(torch.float16)
    with _cpu_codegen():
        bound = _pair._bind_isolated((*args, True))
        with pytest.raises(BackendUnsupported, match="exclusive adjacent native pair"):
            bound.to_code(_config(cute_chained_warp_register_a=True))
        bound = _pair._bind_isolated(args)
        spec = bound.config_spec
        assert "cute_chained_warp_register_a" not in spec._flat_fields()
        for bad in (1, None, "true"):
            with pytest.raises(InvalidConfig, match="must be bool"):
                spec.normalize(_config(cute_chained_warp_register_a=bad))
        for bad in (
            {"cute_chained_mma_schedule": "cp_async_register"},
            {"cute_chained_warp_result_transport": "packed"},
            {"cute_chained_preparation_pipeline": True},
        ):
            cfg = {**_config(cute_chained_warp_register_a=True).config, **bad}
            with pytest.raises(InvalidConfig):
                spec.normalize(helion.Config.from_dict(cfg))


def test_attempt_identity_reordered_issue_and_seed_are_rejected() -> None:
    attempts = []
    original = chained_warp_pair.plan_warp_pairs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        attempts.extend(result)
        return result

    args, _ = _args(torch.float16)
    with _cpu_codegen(), patch.object(chained_warp_pair, "plan_warp_pairs", capture):
        _pair._bind_isolated(args).to_code(_config(cute_chained_warp_register_a=True))
        (pair,) = attempts
        assert pair.phase == "consumed"
        with pytest.raises(_UnsupportedChain):
            pair.declare([])
        with pytest.raises(_UnsupportedChain):
            pair.fill()
        with pytest.raises(_UnsupportedChain):
            emit_warp_mma(
                "chain_1",
                "cutlass.Float16",
                (16, 16, 16),
                32,
                {"a": 1, "b": 1},
                ["chain_1_acc.fill(1.0)"],
                pair=pair.consumer,
            )
        old = pair.layout
        pair.layout = replace(old, threads=64)
        with pytest.raises(_UnsupportedChain, match="layout changed"):
            pair.check()
        pair.layout = old
        pair.phase = "bound"
        with pytest.raises(_UnsupportedChain, match="completion changed"):
            pair.complete(1, ["cute.arch.sync_threads()"], [], {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 8, 8), (16, 16, 16), (19, 32, 32)])
def test_gpu_pair_original_math_masks_and_input_guards(dtype, shape) -> None:
    torch.manual_seed(684)
    args, raw = _args(dtype, *shape, device="cuda", transpose=shape[0] == 19)
    before = tuple(t.clone() for t in raw)
    a, b, c = args
    expected = (-(a.double() @ b.double())).to(dtype).double() @ c.double()
    rows = 32 if shape[0] > 16 else 16
    bound = _pair._bind_isolated(args)
    config = _config(rows, "cp_async")
    old = bound.compile_config(config)(*args)
    changed = helion.Config.from_dict(
        {**config.config, "cute_chained_warp_register_a": True}
    )
    source = bound.to_code(changed)
    assert "chain_pair_0_a" in source
    actual = bound.compile_config(changed)(*args)
    torch.testing.assert_close(old.double(), expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual, old, atol=0, rtol=0)
    for tensor, snapshot in zip(raw, before, strict=True):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 8, 8), (16, 16, 16), (32, 32, 32)])
@pytest.mark.parametrize("axes", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_actual_ldmatrix_packet_delivery_under_m_teams(dtype, shape, axes) -> None:
    assert native_pair_copies(shape, dtype, axes)


@pytest.mark.parametrize("fault", ["missing", "conditional", "reorder"])
def test_original_enclosing_declaration_is_required(fault) -> None:
    original = chained_warp_pair.WarpPairAttempt.declare

    def changed(self, prefix):
        lines = original(self, prefix)
        if fault == "missing":
            return []
        if fault == "conditional":
            return ["if chain_thread < 32:", *["    " + line for line in lines]]
        return list(reversed(lines))

    with (
        _cpu_codegen(),
        patch.object(chained_warp_pair.WarpPairAttempt, "declare", changed),
    ):
        args, _ = _args(torch.float16)
        with pytest.raises(BackendUnsupported, match="before-issue declaration"):
            _pair._bind_isolated(args).to_code(
                _config(cute_chained_warp_register_a=True)
            )


@pytest.mark.parametrize("fault", ["stride", "role", "node", "prefix", "alias"])
def test_original_operand_and_prefix_cannot_be_rebound(fault) -> None:
    original = chained_warp_pair.WarpPairAttempt.bind_operand

    def changed(self, cg, plan, boundaries, stage, layout, prefix):
        if fault == "stride":
            layout = replace(layout, stride=(16, 1))
        elif fault == "role":
            layout = replace(layout, role="b")
        elif fault == "node":
            layout = replace(layout, node=plan.dots[stage].args[1])
        elif fault == "prefix":
            prefix = prefix[1:]
        original(self, cg, plan, boundaries, stage, layout, prefix)
        if fault == "alias":
            name = next(iter(plan.tensor_aliases))
            plan.tensor_aliases[name] = "changed_alias"

    with (
        _cpu_codegen(),
        patch.object(chained_warp_pair.WarpPairAttempt, "bind_operand", changed),
    ):
        args, _ = _args(torch.float16)
        with pytest.raises(BackendUnsupported, match="warp register A changed"):
            _pair._bind_isolated(args).to_code(
                _config(cute_chained_warp_register_a=True)
            )


@pytest.mark.parametrize("fault", ["token", "expression", "source", "config", "layout"])
def test_attempt_selection_is_immutable(fault) -> None:
    original = chained_warp_pair.WarpPairAttempt.declare

    def changed(self, prefix):
        if fault == "token":
            self.producer = replace(self.producer)
        elif fault == "expression":
            self.expression = replace(self.expression)
        elif fault == "source":
            self.source = 1
        elif fault == "config":
            self.codegen.device_function.config.config[
                "cute_chained_operand_coordinates"
            ] = True
        else:
            self.layout = replace(self.layout, slots=self.layout.slots + 1)
        return original(self, prefix)

    with (
        _cpu_codegen(),
        patch.object(chained_warp_pair.WarpPairAttempt, "declare", changed),
    ):
        args, _ = _args(torch.float16)
        with pytest.raises(
            BackendUnsupported, match="graph, config, storage or native layout"
        ):
            _pair._bind_isolated(args).to_code(
                _config(cute_chained_warp_register_a=True)
            )


def test_compute_must_precede_fill_and_original_join() -> None:
    original = chained_warp_pair.WarpPairAttempt.complete

    def changed(self, stage, prefix, publication, boundaries):
        if stage == self.source:
            prefix = [*prefix[:-2], "cute.arch.sync_threads()", prefix[-2], prefix[-1]]
        return original(self, stage, prefix, publication, boundaries)

    with (
        _cpu_codegen(),
        patch.object(chained_warp_pair.WarpPairAttempt, "complete", changed),
    ):
        args, _ = _args(torch.float16)
        with pytest.raises(BackendUnsupported, match="reordered"):
            _pair._bind_isolated(args).to_code(
                _config(cute_chained_warp_register_a=True)
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 8, 8), (32, 32, 32)])
def test_exact_private_a_emission_lowers_to_native_mma_ir(dtype, shape) -> None:
    import importlib

    import cutlass
    import cutlass.cute as cute

    attempts = []
    original = chained_warp_pair.plan_warp_pairs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        attempts.extend(result)
        return result

    with _cpu_codegen(), patch.object(chained_warp_pair, "plan_warp_pairs", capture):
        args, _ = _args(dtype, *shape)
        _pair._bind_isolated(args).to_code(
            _config(shape[0], cute_chained_warp_register_a=True)
        )
    (pair,) = attempts
    ir = importlib.import_module("cutlass._mlir.ir")
    manager = importlib.import_module("cutlass._mlir.passmanager")
    kind = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scope = {
                "cutlass": cutlass,
                "cute": cute,
                "chain_thread": 0,
                "range": range,
            }
            for token in (pair.producer, pair.consumer):
                rows, columns, reduction = pair.plan.shapes[token.stage]
                axes = dict(token._inner_axes)
                for role, extent in (("a", rows), ("b", columns)):
                    stride = (reduction + 8, 1) if axes[role] else (1, extent + 8)
                    scope[f"chain_{token.stage}_{role}"] = cute.make_tensor(
                        cute.make_ptr(
                            kind,
                            token.stage * 8192 + (4096 if role == "b" else 0),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((extent, reduction), stride=stride),
                    )
            assert pair.producer._issue is not None
            assert pair.consumer._issue is not None
            lines = [
                *pair._declaration,
                *pair.producer._issue,
                *pair._fill,
                *pair.consumer._issue,
            ]
            tree = ast.parse("\n".join(lines))
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    assert isinstance(node.iter, ast.Call)
                    assert ast.unparse(node.iter.func) == "cutlass.range_constexpr"
                    node.iter.func = ast.Name(id="range", ctx=ast.Load())
            exec(
                compile(ast.fix_missing_locations(tree), "<native-warp-pair>", "exec"),
                scope,
            )
        manager.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        lowered = str(module)
        assert "nvvm.ldmatrix" in lowered
        assert "nvvm.mma" in lowered
