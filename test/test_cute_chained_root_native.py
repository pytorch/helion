from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.chained_native_destination import _native_layout
from helion._compiler.cute.chained_root_native_emission import RootNativeEmission
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_root_native_emission import RootNativeCompletion
    from helion._compiler.cute.chained_root_typed_publication import (
        RootTypedPublication,
    )
    from helion._compiler.cute.chained_root_typed_publication import (
        TypedPublicationDraft,
    )
    from helion._compiler.generate_ast import GenerateAST

if TYPE_CHECKING:
    from torch._inductor.ir import ComputedBuffer


def _typed_publication_graph(dtype=torch.float16, shape=(64, 64)):
    from torch.fx import Graph

    from helion._compiler.inductor_lowering import PointwiseLowering

    graph = Graph()
    source = graph.placeholder("native_fp32_result")
    source.meta["val"] = torch.empty(shape, dtype=torch.float32)
    target = graph.call_function(
        torch.ops.aten._to_copy.default, (source,), {"dtype": dtype}
    )
    target.meta["val"] = torch.empty(shape, dtype=dtype)
    data = SimpleNamespace(
        dtype=dtype, device=torch.device("cpu"), ranges=shape, inner_fn=None
    )
    target.meta["lowering"] = PointwiseLowering(
        cast("ComputedBuffer", SimpleNamespace(data=data)), ["source"]
    )
    graph.output(target)
    region = SimpleNamespace(nodes=tuple(graph.nodes), scans=(), reductions=())
    stage = SimpleNamespace(
        node=source, geometry=SimpleNamespace(logical=(*shape, 128))
    )
    plan = SimpleNamespace(
        region=region, dots=(source,), root_native=SimpleNamespace(stages=(stage,))
    )
    return graph, source, target, plan


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 64), (128, 32)])
def test_typed_native_target_all_use_cut_and_capacity(dtype, shape):
    from helion._compiler.cute.chained_root_typed_publication import (
        typed_publication_bytes,
    )
    from helion._compiler.cute.chained_root_typed_publication import typed_target

    graph, source, target, plan = _typed_publication_graph(dtype, shape)
    assert typed_target(cast("ChainedMatmulPlan", plan), source) is target
    assert (
        typed_publication_bytes(cast("ChainedMatmulPlan", plan))
        == shape[0] * shape[1] * 2
    )
    # A retained raw FP32 export is a real reader, even if another typed value exists.
    graph.output(source)
    plan.region.nodes = tuple(graph.nodes)
    assert typed_target(cast("ChainedMatmulPlan", plan), source) is None
    assert typed_publication_bytes(cast("ChainedMatmulPlan", plan)) == 0


@pytest.mark.parametrize(
    "mutation", ["dtype", "shape", "foreign", "reduction", "extra_use"]
)
def test_typed_native_target_rejects_unproved_descendants(mutation):
    from helion._compiler.cute.chained_root_typed_publication import typed_target

    graph, source, target, plan = _typed_publication_graph()
    if mutation == "dtype":
        target.meta["val"] = torch.empty((64, 64), dtype=torch.float32)
    elif mutation == "shape":
        target.meta["val"] = torch.empty((32, 128), dtype=torch.float16)
    elif mutation == "foreign":
        target.args = (graph.placeholder("unrelated"),)
    elif mutation == "reduction":
        middle = graph.call_function(torch.ops.aten.sum.dim_IntList, (source, [1]))
        middle.meta["val"] = torch.empty((64,), dtype=torch.float32)
        target.args = (middle,)
        plan.region.reductions = (middle,)
    else:
        graph.call_function(torch.ops.aten.clone.default, (source,))
    plan.region.nodes = tuple(graph.nodes)
    assert typed_target(cast("ChainedMatmulPlan", plan), source) is None


def test_typed_native_retirement_preserves_waits_and_declines_extra_reads():
    from helion._compiler.cute.chained_root_typed_publication import _retired_body

    allocation = ("chain_0_c_ptr = allocate()", "chain_0_c = tensor(chain_0_c_ptr)")
    publication = ("for i in range(8):\n    chain_0_c[i] = values[i]",)
    receipt = SimpleNamespace(
        completion=SimpleNamespace(
            owner="chain_0_c",
            allocation=allocation,
            publication=publication,
        )
    )
    lines = [
        *allocation,
        "issue()",
        "wait()",
        "typed_store()",
        *publication,
        "join()",
        "use_typed()",
    ]
    assert _retired_body(cast("RootTypedPublication", receipt), lines) == (
        "issue()",
        "wait()",
        "typed_store()",
        "join()",
        "use_typed()",
    )
    for extra in ("consume(chain_0_c)", "alias = chain_0_c_ptr", "chain_0_c[0] = 1"):
        assert (
            _retired_body(cast("RootTypedPublication", receipt), [*lines, extra])
            is None
        )
    assert (
        _retired_body(cast("RootTypedPublication", receipt), [*allocation, *lines])
        is None
    )
    changed = [text.replace("values[i]", "-values[i]") for text in lines]
    assert _retired_body(cast("RootTypedPublication", receipt), changed) is None


@pytest.mark.parametrize("operand_domain", [False, True])
@pytest.mark.parametrize("explicit_none_scope", [False, True])
def test_typed_native_binder_default_path_unchanged(
    operand_domain, explicit_none_scope
):
    from torch.fx import Graph

    from helion._compiler.cute import chained_matmul as chain
    from helion._compiler.cute import chained_root_vector_cache as vectors
    from helion._compiler.cute.chained_fragment_epilogue import bind_fragment_expression

    graph = Graph()
    source = graph.placeholder("completed_fp32")
    source.meta["val"] = torch.empty((3, 5), dtype=torch.float32, device="cpu")
    target = graph.call_function(torch.ops.aten.permute.default, (source, (1, 0)))
    target.meta["val"] = torch.empty((5, 3), dtype=torch.float32, device="cpu")
    graph.output(target)
    plan = cast(
        "ChainedMatmulPlan",
        SimpleNamespace(
            dots=(source,),
            prepared_widenings=None,
            axes=(),
            loop=None,
        ),
    )
    cg = cast("GenerateAST", object())
    coordinates = ("row", "col")
    target_coordinates = ("col", "row")
    boundaries = {source: "unused_shared_image"}
    scans = []
    options = {"root_read_scope": None} if explicit_none_scope else {}
    # Shape/domain planning is tested separately. Exercise the real expression,
    # view mapping, register binding and receipt here, without source generation.
    with (
        patch.object(
            chain, "_shape", side_effect=lambda node: tuple(node.meta["val"].shape)
        ),
        patch.object(chain, "_operand_domain", return_value=["col < 5"]) as domain,
        patch.object(
            vectors,
            "bind_read_scope",
            side_effect=AssertionError("default path gained native scope"),
        ),
    ):
        result = bind_fragment_expression(
            cg,
            plan,
            boundaries,
            scans,
            source,
            target,
            coordinates,
            target_coordinates,
            "acc[slot]",
            "cutlass.Float32",
            operand_domain=operand_domain,
            **options,
        )
        assert isinstance(result.expression, chain._Expression)
        assert result.expression.scan_inputs is scans
        assert result.expression.boundaries is boundaries
        assert result.expression.context is plan
        assert result.expression.root_vector_scope is None
        assert result.expression.fragments == {source: (coordinates, "acc[slot]")}
        assert result.expression.memo == {
            (source, coordinates): "acc[slot]",
            (target, target_coordinates): "acc[slot]",
        }
        assert result.expression.lines == []
        assert result.value == "acc[slot]"
        assert result.domain == (("col < 5",) if operand_domain else ())
        assert result.masked_value() == (
            "(cutlass.Float32(acc[slot]) if (col < 5) else cutlass.Float32(0))"
            if operand_domain
            else "cutlass.Float32(acc[slot])"
        )
        if operand_domain:
            domain.assert_called_once_with(cg, target, target_coordinates, plan)
        else:
            domain.assert_not_called()
        # A transpose must preserve the original per-thread source coordinates.
        with pytest.raises(chain._UnsupportedChain, match="cross-coordinate"):
            bind_fragment_expression(
                cg,
                plan,
                boundaries,
                scans,
                source,
                target,
                coordinates,
                coordinates,
                "acc[slot]",
                "cutlass.Float32",
                operand_domain=False,
            )
        foreign = Graph().placeholder("foreign")
        with pytest.raises(chain._UnsupportedChain, match="foreign fragment"):
            bind_fragment_expression(
                cg,
                plan,
                boundaries,
                scans,
                source,
                foreign,
                coordinates,
                coordinates,
                "acc[slot]",
                "cutlass.Float32",
                operand_domain=False,
            )
        result.expression.fragments[source] = (coordinates, "other_acc[slot]")
        with pytest.raises(chain._UnsupportedChain, match="expression changed"):
            result.check()


def test_typed_native_completion_requires_actual_publication_and_join():
    from dataclasses import replace

    from helion._compiler.cute.chained_matmul import _UnsupportedChain
    from helion._compiler.cute.chained_root_typed_publication import (
        RootTypedPublication,
    )

    source, target = object(), object()
    stage = SimpleNamespace(node=source)
    body = SimpleNamespace(
        plan=object(), execution=SimpleNamespace(sync="join()"), typed_publications={}
    )
    draft = SimpleNamespace(
        check=lambda: None,
        scope=SimpleNamespace(body=body, stage=stage),
        target=target,
        owner="typed",
        allocation=("allocate_typed()",),
        publication=("typed_store()",),
    )
    phase = (
        "issue()",
        "wait()",
        "allocate_typed()",
        "typed_store()",
        "fp32_store()",
        "join()",
    )
    completion = SimpleNamespace(
        body=body,
        stage=stage,
        owner="fp32",
        phase=phase,
        publication=("fp32_store()",),
        check=lambda *args: None,
    )
    receipt = RootTypedPublication(
        cast("TypedPublicationDraft", draft),
        cast("RootNativeCompletion", completion),
        phase,
    )
    receipt = replace(receipt, selection=receipt.facts())
    body.typed_publications[target] = receipt
    receipt.check()
    for changed in (
        phase[:-1],
        tuple(line for line in phase if line != "typed_store()"),
        tuple(line.replace("typed_store()", "wrong_cast()") for line in phase),
    ):
        invalid = replace(receipt, phase=changed)
        invalid = replace(invalid, selection=invalid.facts())
        completion.phase = changed
        body.typed_publications[target] = invalid
        with pytest.raises(_UnsupportedChain):
            invalid.check()


def test_typed_native_decline_restores_speculative_vector_reads():
    from .test_cute_root_vector_cache import _native_vector_scope_fixture
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.cute import chained_fragment_epilogue
    from helion._compiler.cute import chained_root_typed_publication as typed
    from helion._compiler.cute.chained_matmul import _UnsupportedChain

    scope, expression, read_scope, binding, node = _native_vector_scope_fixture()
    body = scope.body
    object.__setattr__(
        body.cg,
        "device_function",
        SimpleNamespace(
            config=SimpleNamespace(config={"cute_chained_completed_image": True})
        ),
    )
    target = node.graph.call_function(
        torch.ops.aten._to_copy.default, (scope.stage.node,), {"dtype": torch.float16}
    )
    target.meta["val"] = torch.empty((64, 64), dtype=torch.float16)
    prior = read_scope
    binding.publication.reads.append(prior)

    def decline(*args, **kwargs):
        assert kwargs["root_read_scope"].coordinates == scope.coordinates
        binding.publication.reads.append(read_scope)
        raise _UnsupportedChain("later unsupported descendant")

    env = SimpleNamespace(
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float16")
    )
    with (
        patch.object(typed, "typed_target", return_value=target),
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(chained_fragment_epilogue, "bind_fragment_expression", decline),
    ):
        assert (
            typed.prepare_typed_publication(
                body, scope.stage, {node: binding}, [], scope.lines
            )
            is None
        )
    assert binding.publication.reads == [prior]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_stage(a, b, rhs):
    out = torch.empty(
        (a.shape[0], a.shape[1], rhs.shape[2]), device=a.device, dtype=torch.float32
    )
    for batch in hl.tile(a.shape[0], block_size=1):
        first = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :].T)
        out[batch.begin, :, :] = hl.dot(first.to(a.dtype), rhs[batch.begin, :, :])
    return out


def config(**kwargs):
    return helion.Config.from_dict(
        {"num_warps": 4, "cute_chained_mma_schedule": "cp_async", **kwargs}
    )


def args(m=64, n=64, k=128, dtype=torch.bfloat16):
    return tuple(
        torch.empty(shape, dtype=dtype) for shape in ((2, m, k), (2, n, k), (2, n, 16))
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(64, 64, 128), (64, 32, 32), (128, 64, 128)])
def test_actual_raw_first_and_warp_second(m, n, k, dtype):
    seen = []
    original = RootNativeEmission.finish

    def finish(self, lines):
        original(self, lines)
        assert self.finished and set(self.completed) == {0}
        receipt = self.completed[0]
        assert receipt.stage.geometry.logical == (m, n, k)
        assert receipt.owner == "chain_0_c"
        assert receipt.phase[-1] == "cute.arch.sync_threads()"
        assert receipt.phase.index(
            "cute.arch.fence_view_async_shared()"
        ) < receipt.phase.index("cute.arch.mbarrier_wait(chain_bars + 0, 0)")
        assert receipt.phase.index(
            "cute.arch.mbarrier_wait(chain_bars + 0, 0)"
        ) < receipt.phase.index("cute.arch.fence_view_async_tmem_load()")
        seen.append(receipt)

    with _cpu_codegen(), patch.object(RootNativeEmission, "finish", finish):
        bound = two_stage._bind_isolated(args(m, n, k, dtype))
        original_source = bound.to_code(config())
        assert bound.to_code(config(cute_chained_root_native=False)) == original_source
        source = bound.to_code(config(cute_chained_root_native=True))
    assert len(seen) == 1
    assert "make_trivial_tiled_mma" in source and "MmaF16BF16Op" in source
    assert source.count("mbarrier_wait(chain_bars + 0, 0)") == 1
    assert source.count("chain_allocator.free(chain_tptr)") == 1


@pytest.mark.parametrize("shape", [(64, 32), (64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actual_swizzle_packets_cover_exact_original_words(shape, dtype):
    with _cpu_codegen():
        capacity, cells, complete = _native_layout(shape, dtype, 128)
    assert capacity == shape[0] * shape[1] and complete
    assert sorted(cells) == list(range(capacity))


def test_actual_bound_receipts_reject_owner_layout_and_lifetime_changes():
    from dataclasses import replace

    from helion._compiler.cute.chained_matmul import _UnsupportedChain

    checked = []
    original = RootNativeEmission.finish

    def finish(body, lines):
        original(body, lines)
        receipt = body.completed[0]

        def verify():
            receipt.check(body.plan, receipt.stage.node, receipt.owner, receipt.phase)

        def field(label, obj, name, value):
            saved = getattr(obj, name)
            object.__setattr__(obj, name, value)
            try:
                with pytest.raises(_UnsupportedChain):
                    verify()
            finally:
                object.__setattr__(obj, name, saved)
            verify()
            checked.append(label)

        field("phase_source", receipt, "phase", ("wrong()", *receipt.phase))
        field("allocation_capacity", receipt, "allocation", ("wrong_capacity()",))
        field("input_cut", receipt, "input_prefix", ())
        field("FP32_publication", receipt, "publication", ("wrong_cast()",))
        field("owner", receipt, "owner", "foreign_c")
        field("root_phase", body.native.root, "phase_index", body.native.phase + 1)
        field("boolean_phase", body.native.root, "phase_index", False)
        field("boolean_stage", receipt.stage, "index", False)
        field("barrier_count", body.native, "barrier_count", 1)
        field("TMEM_capacity", body.native, "tmem_columns", 32)
        field(
            "execution_team",
            body.native,
            "execution",
            replace(body.execution, threads=256),
        )
        field(
            "geometry",
            receipt.stage,
            "geometry",
            replace(receipt.stage.geometry, native_rows=128),
        )
        target = receipt.operands.native_destinations[0]
        field("copy_role", target, "role", "b")
        field("reference_stride", target, "reference_stride", (128, 2))
        capacity, cells, complete = target.layout
        field(
            "swizzle_cells",
            target,
            "layout",
            (capacity, (cells[1], cells[0], *cells[2:]), complete),
        )
        field(
            "copy_program",
            receipt.operands,
            "lines",
            ("wrong_load()", *receipt.operands.lines),
        )
        source = receipt.stage.leaves[0].args[0]
        value = source.meta["val"]
        source.meta["val"] = value.detach()
        try:
            with pytest.raises(_UnsupportedChain):
                verify()
        finally:
            source.meta["val"] = value
        checked.append("tensor_identity")
        verify()
        key = next(iter(body.plan.tensor_aliases))
        alias = body.plan.tensor_aliases[key]
        body.plan.tensor_aliases[key] = "foreign_owner"
        try:
            with pytest.raises(_UnsupportedChain):
                verify()
        finally:
            body.plan.tensor_aliases[key] = alias
        checked.append("bound_alias")
        verify()
        cfg = body.cg.device_function.config.config
        cfg["cute_chained_root_native"] = False
        try:
            with pytest.raises(_UnsupportedChain):
                verify()
        finally:
            cfg["cute_chained_root_native"] = True
        checked.append("config_intent")
        verify()
        saved = body.cursor
        body.cursor = 0
        body.finished = False
        try:
            with pytest.raises(_UnsupportedChain):
                original(body, list(lines))
        finally:
            body.cursor = saved
            body.finished = True
        checked.append("incomplete_stage_visit")
        verify()

    with _cpu_codegen(), patch.object(RootNativeEmission, "finish", finish):
        two_stage._bind_isolated(args()).to_code(config(cute_chained_root_native=True))
    assert len(checked) == 20


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n", [(64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)]
)
def test_existing_native_drain_delivers_every_full_FP32_word(dtype, m, n):
    from helion._compiler.cute.chained_completed_members import _native_map

    with _cpu_codegen():
        coordinates, addresses, selected = _native_map(
            (m, n), (m, n), 0, n, False, dtype, "SMEM"
        )
    result = {}
    native_words = set()
    for points, words, slots in zip(coordinates, addresses, selected, strict=True):
        for slot in slots:
            row, column = points[slot]
            target = row * (n + 4) + column
            assert target not in result and words[slot] not in native_words
            result[target] = (row, column)
            native_words.add(words[slot])
    assert result == {r * (n + 4) + c: (r, c) for r in range(m) for c in range(n)}
    assert max(result) < m * (n + 4)


@pytest.mark.parametrize(
    "change",
    [
        {"num_warps": 8},
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_root_native": 1},
        {"cute_chained_warp_register_a": True},
    ],
)
def test_unsupported_explicit_policy_cannot_silently_fall_back(change):
    from helion import exc

    with _cpu_codegen():
        bound = two_stage._bind_isolated(args())
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(config(**{"cute_chained_root_native": True, **change}))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_two_stage(a, b, rhs):
    out = torch.empty(
        (a.shape[0], a.shape[1], 16), device=a.device, dtype=torch.float32
    )
    for batch, rows in hl.tile([a.shape[0], a.shape[1]], block_size=[1, 64]):
        rr = hl.arange(64) + rows.begin
        red = hl.arange(128)
        left = hl.load(a, [batch.begin, rr, red], extra_mask=rr[:, None] < a.shape[1])
        right = hl.load(b, [batch.begin, hl.arange(64), red])
        first = hl.dot(left, right.T)
        out[batch.begin, rows, :] = hl.dot(first.to(a.dtype), rhs[batch.begin, :, :])
    return out


@pytest.mark.parametrize("physical_k", [127, 128])
def test_masked_copy_preserves_every_original_guard_and_scalar_fallback(physical_k):
    from helion._compiler.cute.chained_root_warp_stage import (
        emit_original_warp_operands,
    )

    checked = []
    original = RootNativeEmission.finish

    def finish(body, lines):
        original(body, lines)
        receipt = body.completed[0]
        reference = emit_original_warp_operands(
            body.cg,
            body.plan,
            {},
            [],
            {},
            [],
            0,
            "cp_async",
            8,
            body.dtype(receipt.stage),
        )
        # Only the target declaration changes. The exact old pointer/domain,
        # integer overflow, source layout/alignment and fallback code remains.
        native = list(receipt.operands.lines)
        old = list(reference.lines)
        for target, layout in zip(
            receipt.operands.native_destinations, reference.layouts, strict=True
        ):
            declaration = target.declaration()
            begin = native.index(declaration[0])
            assert native[begin : begin + len(declaration)] == declaration
            native[begin : begin + len(declaration)] = [
                f"{target.name} = cute.make_tensor({target.name}_ptr, cute.make_layout({layout.shape!r}, stride={layout.stride!r}))"
            ]
            for tag in (f"{target.name}_masked_async", f"{target.name}_async"):
                for i, line in enumerate(native):
                    if f"{tag}_target = {target.name}" in line:
                        native[i] = line.replace(
                            f"{tag}_target = {target.name}",
                            f"{tag}_target = cute.make_tensor({target.name}_ptr, cute.make_layout({layout.shape!r}, stride=({layout.stride[0]}, 1)))",
                        )
        import ast
        import re

        def normalized(program):
            tree = ast.parse("\n".join(program))
            definitions = [
                node.targets[0].id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and re.fullmatch(r"(?:chain_value|v)(?:_[0-9]+)?", node.targets[0].id)
            ]
            assert len(definitions) == len(set(definitions))
            mapping = {name: f"bound_value_{i}" for i, name in enumerate(definitions)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in mapping:
                    node.id = mapping[node.id]
            return ast.dump(tree, include_attributes=False)

        assert normalized(native) == normalized(old)
        assert normalized(
            [
                x.replace("input_tensor.iterator", "foreign_tensor.iterator")
                for x in native
            ]
        ) != normalized(old)
        assert len(receipt.operands.copies) == (2 if physical_k == 128 else 0)
        assert sum(
            "_masked_async_copied = cutlass.Boolean(False)" in block for block in native
        ) == (1 if physical_k == 128 else 0)
        checked.append(True)

    inputs = tuple(
        torch.empty(s, dtype=torch.bfloat16)
        for s in ((2, 96, physical_k), (2, 64, physical_k), (2, 64, 16))
    )
    with _cpu_codegen(), patch.object(RootNativeEmission, "finish", finish):
        masked_two_stage._bind_isolated(inputs).to_code(
            config(cute_chained_root_native=True)
        )
    assert checked == [True]


def test_explicit_native_request_rejects_root_without_full_native_raw_stage():
    from helion import exc

    with _cpu_codegen(), pytest.raises(exc.BackendUnsupported):
        two_stage._bind_isolated(args(32, 32, 64)).to_code(
            config(cute_chained_root_native=True)
        )
