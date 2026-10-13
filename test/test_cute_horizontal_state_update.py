from __future__ import annotations

from collections import Counter
import dataclasses
from itertools import product
from itertools import starmap
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.horizontal_state_update import HorizontalGeometry
from helion._compiler.cute.horizontal_state_update import InapplicableStateUpdate
from helion._compiler.cute.horizontal_state_update import bind_state_task_domain
from helion._compiler.cute.horizontal_state_update import (
    discover_pointwise_state_region,
)
import helion.language as hl
from helion.language import view_ops


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", fast_math=False
)
def _affine_rows(history, forcing, weights, index, result):
    batch, heads, rows = result.shape
    features = history.size(3)
    for bi, hi, ri in hl.tile([batch, heads, rows], block_size=[1, 1, None]):
        ki = hl.arange(features)
        slot = index[bi.id]
        old = history[slot, hi.id, ri.index, ki].float()
        update = (
            old * weights[ki][None, :].float()
            + forcing[bi.id, hi.id, ri][:, None].float()
        )
        result[bi.id, hi.id, ri] = (update * weights[ki][None, :].float()).sum(-1)
        history[slot, hi.id, ri.index, ki] = update


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", fast_math=False
)
def _pointwise_rows(history, forcing, weights, index, result):
    batch, heads, rows = result.shape
    features = history.size(3)
    for bi, hi, ri in hl.tile([batch, heads, rows], block_size=[1, 1, None]):
        ki = hl.arange(features)
        slot = index[bi.id]
        old = history[slot, hi.id, ri.index, ki].float()
        update = torch.sin(old) + forcing[bi.id, hi.id, ri][:, None].float()
        result[bi.id, hi.id, ri] = (update * weights[ki][None, :].float()).sum(-1)
        history[slot, hi.id, ri.index, ki] = update


def _args(alias=False):
    state = torch.empty((3, 2, 64, 32), dtype=torch.bfloat16)
    forcing = (
        state.reshape(-1)[: 3 * 2 * 64].view(3, 2, 64)
        if alias
        else torch.empty((3, 2, 64), dtype=torch.bfloat16)
    )
    return (
        state,
        forcing,
        torch.empty(32, dtype=torch.bfloat16),
        torch.tensor([2, 0, 1], dtype=torch.int64),
        torch.empty((3, 2, 64), dtype=torch.bfloat16),
    )


class _Captured(Exception):
    pass


def _with_region(callback, *, kernel=_affine_rows, alias=False):
    def capture(backend, graphs, config, strategy):
        callback(discover_pointwise_state_region(graphs), graphs)
        raise _Captured

    with _cpu_codegen(), patch.object(CuteBackend, "_pre_codegen", capture):
        args = _args(alias)
        bound = kernel._bind_isolated(args)
        with pytest.raises(_Captured):
            bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("kernel", [_affine_rows, _pointwise_rows])
def test_actual_graph_keeps_original_pointwise(kernel):
    def check(region, graphs):
        assert region is not None
        region.check()
        region.check_readonly_owners()
        domain = bind_state_task_domain(region)
        domain.check()
        assert domain.grid_extents == (3, 2, 64)
        assert domain.state_shape == (3, 2, 64, 32)
        assert region.state_load.args[1] == region.state_store.args[1]
        assert region.update is region.state_store.args[2].args[0]
        assert (
            region.reduction in region.output_store.args[2].all_input_nodes
            or region.output_store.args[2] is region.reduction
        )

    _with_region(check, kernel=kernel)


def test_current_alias_declines_without_new_index_contract():
    def check(region, graphs):
        assert region is not None
        with pytest.raises(InapplicableStateUpdate, match="alias"):
            region.check_readonly_owners()

    _with_region(check, alias=True)


@pytest.mark.parametrize(
    "mutation", ["store_index", "mask", "source", "dtype", "view", "extra_write"]
)
def test_live_graph_mutations_reject(mutation):
    def check(region, graphs):
        assert region is not None
        node = region.state_store
        if mutation == "store_index":
            args = list(node.args)
            selectors = list(args[1])
            selectors[0] = 0
            args[1] = selectors
            node.args = tuple(args)
        elif mutation == "mask":
            node.args = (*node.args[:3], True)
        elif mutation == "source":
            node.args = (region.output_store.args[0], *node.args[1:])
        elif mutation == "dtype":
            region.update.meta["val"] = torch.empty((64, 32), dtype=torch.float64)
        elif mutation == "view":
            views = [n for n in graphs[0].graph.nodes if n.target is view_ops.subscript]
            # No import/name heuristics: mutate the actual view target in the graph.
            views[0].args = (views[0].args[0], [slice(None, None, -1)])
        else:
            with graphs[0].graph.inserting_before(node):
                graphs[0].graph.call_function(node.target, node.args)
        with pytest.raises(InapplicableStateUpdate):
            region.check()

    _with_region(check)


@pytest.mark.parametrize(
    "warps,lanes,slab,features,size",
    [
        (8, 2, 32, 128, 2),
        (4, 2, 32, 64, 2),
        (4, 4, 64, 128, 4),
        (2, 1, 32, 32, 2),
    ],
)
def test_geometry_all_words_once(warps, lanes, slab, features, size):
    geometry = HorizontalGeometry(warps, lanes, slab, features, size)
    geometry.check(232448)
    words = list(
        starmap(
            geometry.logical_origin,
            product(
                range(warps), range(32), range(geometry.stages), range(slab // lanes)
            ),
        )
    )
    assert Counter(words) == Counter(product(range(geometry.rows), range(features)))
    assert geometry.arrivals == warps * 32 + 1
    assert geometry.state_bytes == len(words) * size


@pytest.mark.parametrize(
    "updates",
    [
        {"consumer_warps": 32},
        {"lanes_per_row": 3},
        {"stage_features": 48},
        {"feature_extent": 0},
        {"itemsize": 1},
        {"consumer_warps": True},
    ],
)
def test_invalid_geometry_rejects(updates):
    with pytest.raises(InapplicableStateUpdate):
        dataclasses.replace(HorizontalGeometry(8, 2, 32, 128, 2), **updates).check(
            232448
        )


def test_shared_budget_is_resource_bound():
    geometry = HorizontalGeometry(8, 2, 32, 128, 2)
    with pytest.raises(InapplicableStateUpdate):
        geometry.check(geometry.state_bytes)


class _ScalarCaptured(BaseException):
    pass


def _with_scalar_program(callback, *, kernel=_affine_rows):
    from helion._compiler.cute.horizontal_state_expression import (
        lower_state_scalar_program,
    )
    from helion._compiler.generate_ast import GenerateAST

    holder = {}
    original = CuteBackend._pre_codegen

    def capture(backend, graphs, config, strategy):
        region = discover_pointwise_state_region(graphs)
        assert region is not None
        holder["domain"] = bind_state_task_domain(region)
        return original(backend, graphs, config, strategy)

    def emit(cg):
        domain = holder["domain"]
        program = lower_state_scalar_program(
            cg,
            domain,
            outer_coordinates=("outer_batch", "outer_head"),
            coordinates=("logical_row", "logical_feature"),
        )
        callback(cg, program)
        raise _ScalarCaptured

    with (
        _cpu_codegen(),
        patch.object(CuteBackend, "_pre_codegen", capture),
        patch.object(GenerateAST, "_try_codegen_block_scaled_root", emit),
    ):
        args = _args()
        bound = kernel._bind_isolated(args)
        with pytest.raises(_ScalarCaptured):
            bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("kernel", [_affine_rows, _pointwise_rows])
def test_scalar_program_uses_original_lowering_and_narrowed_mask(kernel):
    import ast

    from helion.language.memory_ops import _cute_scalar_pointer_expr

    def check(cg, program):
        program.check(cg)
        assert program.old_state.pointer == _cute_scalar_pointer_expr(
            program.old_state.tensor, list(program.old_state.indices)
        )
        assert "cutlass.Int32(" in program.old_state.predicate
        assert " < 3" in program.old_state.predicate
        assert ">=" not in program.old_state.predicate
        assert "Int64" not in program.old_state.predicate
        assert program.old_state.dtype is torch.bfloat16
        assert program.storage_value != program.update
        assert all(read.node in program.domain.region.loads for read in program.reads)
        body = ast.parse("\n".join(program.lines))
        assert any(isinstance(n, ast.IfExp) for n in ast.walk(body))
        assert "cutlass.BFloat16" in "\n".join(program.lines)

    _with_scalar_program(check, kernel=kernel)


def test_scalar_program_rechecks_original_lowering_and_config():
    def check(cg, program):
        node = program.domain.region.update
        original = node.meta["lowering"]
        node.meta["lowering"] = None
        with pytest.raises(InapplicableStateUpdate, match="lowering"):
            program.check(cg)
        node.meta["lowering"] = original
        program.check(cg)
        config = cg.device_function.config.config
        original_fast = config.get("cute_lane_unroll")
        config["cute_lane_unroll"] = "changed"
        with pytest.raises(InapplicableStateUpdate, match="configuration"):
            program.check(cg)
        if original_fast is None:
            config.pop("cute_lane_unroll")
        else:
            config["cute_lane_unroll"] = original_fast
        program.check(cg)

    _with_scalar_program(check)


def test_feature_invariance_follows_zero_stride_and_current_predicates():
    from helion._compiler.cute.horizontal_state_invariance import feature_invariance

    def check(region, graphs):
        proof = feature_invariance(region)
        proof.check()
        assert region.state_load not in proof.invariant
        assert region.update not in proof.invariant
        assert region.feature_selector not in proof.invariant
        row_loads = [
            n
            for n in region.loads
            if n.meta["val"].ndim == 1
            and region.feature_selector not in n.all_input_nodes
        ]
        assert row_loads and all(n in proof.invariant for n in row_loads)
        for node, axes in proof.axes:
            assert len(axes) == node.meta["val"].ndim
        original = region.state_store.args
        region.state_store.args = (*original[:3], False)
        with pytest.raises(InapplicableStateUpdate):
            proof.check()
        region.state_store.args = original

    _with_region(check)


def test_tensor_snapshot_retains_underlying_storage_across_set():
    from helion._compiler.cute.horizontal_state_update import _freeze

    for _ in range(1000):
        value = torch.empty(8)
        before = _freeze(value)
        owner = value.untyped_storage()._cdata
        replacement = torch.empty(8)
        value.set_(replacement)
        assert owner != value.untyped_storage()._cdata
        assert before != _freeze(value)
        assert _freeze(value) == _freeze(value)


def test_graph_snapshot_rejects_same_abi_storage_replacement():
    def check(region, graphs):
        node = region.state_load.args[0]
        tensor = node.meta["val"]
        old_owner = tensor.untyped_storage()._cdata
        with tensor.fake_mode:
            replacement = torch.empty_strided(
                tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
            )
            tensor.set_(replacement)
        assert old_owner != tensor.untyped_storage()._cdata
        with pytest.raises(InapplicableStateUpdate, match="graph changed"):
            region.check()

    _with_region(check)


def _horizontal_code(*, rows=64, features=32, coefficient_offset=0, forcing_rows=None):
    values = list(_args())
    values[0] = torch.empty((3, 2, rows, features), dtype=torch.bfloat16)
    values[1] = torch.empty(
        (3, 2, rows if forcing_rows is None else forcing_rows), dtype=torch.bfloat16
    )
    owner = torch.empty(features + coefficient_offset, dtype=torch.bfloat16)
    values[2] = owner[coefficient_offset:]
    values[4] = torch.empty((3, 2, rows), dtype=torch.bfloat16)
    with _cpu_codegen():
        bound = _affine_rows._bind_isolated(values)
        original = bound.config_spec.default_config()
        assert "cute_state_update_schedule" not in original.config
        selected = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_state_update_schedule") == "horizontal_tma"
        ]
        assert len(selected) == 1
        return bound.to_code(selected[0]), values


@pytest.mark.parametrize("rows,features", [(16, 32), (64, 64), (65, 64), (128, 128)])
def test_public_structural_seed_emits_finite_transport(rows, features):
    import ast

    source, _ = _horizontal_code(rows=rows, features=features)
    tree = ast.parse(source)
    assert "horizontal_state_tma" in source
    assert "_cute_store_u16_vec" in source
    assert "horizontal_consumers" in source
    assert "cute.arch.cp_async_bulk_wait_group(0, read=False)" in source
    # A partial final row tile still reaches all cohort barriers; it does not
    # emit an early return, break or continue inside the device body.
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert not any(
        isinstance(node, (ast.Return, ast.Break, ast.Continue))
        for node in ast.walk(device)
    )


def _host_guard_fixture(offset=0):
    tensors = _args()
    if offset:
        copies = []
        for tensor in tensors:
            owner = torch.empty(tensor.numel() + offset + 64, dtype=tensor.dtype)
            copies.append(owner.as_strided(tensor.shape, tensor.stride(), offset))
        tensors = tuple(copies)
    plan = {
        "kind": "horizontal_state_tma",
        "metadata_indices": [0, 1, 2, 3, 4],
        "metadata": [
            (tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in tensors
        ],
        "write_indices": [0, 4],
        "readonly_indices": [1, 2, 3],
        "aligned_indices": [2],
        "state_idx": 0,
        "state_shape": list(tensors[0].shape),
        "state_strides": list(tensors[0].stride()),
    }
    return plan, tensors


@pytest.mark.parametrize("offset", [0, 64])
def test_runtime_current_storage_and_canary_offsets(offset):
    from helion.runtime.cute.horizontal_state import validate_horizontal_state_arguments

    plan, tensors = _host_guard_fixture(offset)
    validate_horizontal_state_arguments([plan], tensors)
    # New storage with the same legal layout is supported; no stale pointer is
    # retained in the descriptor plan.
    tensors[0].set_(torch.empty_like(tensors[0]))
    validate_horizontal_state_arguments([plan], tensors)


@pytest.mark.parametrize(
    "mutation",
    [
        "state_alias",
        "readonly_alias",
        "state_alignment",
        "packet_alignment",
        "dtype",
        "stride",
        "metadata",
        "bool_index",
        "duplicate_index",
        "wrong_state",
        "write_owner",
        "packet_owner",
    ],
)
def test_runtime_rejects_changed_current_storage(mutation):
    from helion.runtime.cute.horizontal_state import validate_horizontal_state_arguments

    plan, tensors = _host_guard_fixture()
    tensors = list(tensors)
    if mutation in ("state_alias", "readonly_alias"):
        destination = 4 if mutation == "state_alias" else 1
        tensors[destination] = (
            tensors[0]
            .view(-1)[: tensors[destination].numel()]
            .view(tensors[destination].shape)
        )
    elif mutation in ("state_alignment", "packet_alignment"):
        index = 0 if mutation == "state_alignment" else 2
        owner = torch.empty(tensors[index].numel() + 1, dtype=tensors[index].dtype)
        tensors[index] = owner[1:].view(tensors[index].shape)
    elif mutation == "dtype":
        tensors[1] = tensors[1].float()
    elif mutation == "stride":
        tensors[0] = tensors[0].transpose(1, 2)
    elif mutation == "metadata":
        plan["metadata"][0] = (plan["state_shape"],)
    elif mutation == "bool_index":
        plan["state_idx"] = False
    elif mutation == "duplicate_index":
        plan["metadata_indices"][1] = 0
    elif mutation == "wrong_state":
        plan["state_idx"] = 2
    elif mutation == "write_owner":
        plan["write_indices"] = [0, 1]
    else:
        plan["aligned_indices"] = [0]
    with pytest.raises(ValueError):
        validate_horizontal_state_arguments([plan], tuple(tensors))


def test_scheduled_program_receipt_rejects_nested_access_mutation():
    from helion._compiler.cute.horizontal_state_expression import (
        lower_scheduled_scalar_program,
    )

    def check(cg, original):
        program = lower_scheduled_scalar_program(
            cg,
            original.domain,
            outer_coordinates=("outer_batch", "outer_head"),
            coordinates=("logical_row", "logical_feature"),
            reduced_value="row_sum",
        )
        program.check(cg)
        object.__setattr__(program.state_access, "pointer", "foreign.iterator")
        with pytest.raises(InapplicableStateUpdate, match="scheduled scalar"):
            program.check(cg)

    _with_scalar_program(check)


def test_horizontal_declines_narrower_original_row_bound():
    with pytest.raises(helion.exc.BackendUnsupported, match="narrower original bound"):
        _horizontal_code(forcing_rows=63)


def test_unaligned_coefficient_uses_scalar_publication():
    source, _ = _horizontal_code(coefficient_offset=1)
    assert "horizontal_coefficient" in source
    assert "_cute_load_u16_vec((weights.iterator" not in source


@pytest.mark.parametrize(
    "mutation", ["body", "wrapper", "publication", "publication_list", "program"]
)
def test_installed_receipt_rejects_mutations(mutation):
    import ast

    from helion._compiler.cute import horizontal_state_codegen as module

    original = module.codegen_horizontal_state
    checked = []

    def probe(cg):
        accepted = original(cg)
        if not accepted:
            return accepted
        receipt = cg.device_function.cute_state.horizontal_state_body
        receipt.check(cg)
        if mutation == "body":
            cg.device_function.body.append(ast.parse("foreign = 1").body[0])
        elif mutation == "wrapper":
            cg.cute_wrapper_plans[0]["rows"] += 16
        elif mutation == "publication":
            assert receipt.publications
            object.__setattr__(
                receipt.publications[0].access, "pointer", "foreign.iterator"
            )
        elif mutation == "publication_list":
            object.__setattr__(receipt, "publications", ())
        else:
            object.__setattr__(receipt.program, "element", ("foreign = 1",))
        with pytest.raises(InapplicableStateUpdate):
            receipt.check(cg)
        checked.append(True)
        raise _ScalarCaptured

    with (
        patch.object(module, "codegen_horizontal_state", probe),
        pytest.raises(_ScalarCaptured),
    ):
        _horizontal_code()
    assert checked == [True]


@pytest.mark.parametrize("removed", ["body", "plan", "both"])
def test_selected_receipt_cannot_be_removed_before_finalization(removed):
    from helion._compiler.cute import horizontal_state_codegen as module

    original = module.codegen_horizontal_state

    def corrupt(cg):
        accepted = original(cg)
        if accepted:
            state = cg.device_function.cute_state
            if removed in ("body", "both"):
                state.horizontal_state_body = None
            if removed in ("plan", "both"):
                state.horizontal_state_plan = None
        return accepted

    with (
        patch.object(module, "codegen_horizontal_state", corrupt),
        pytest.raises(helion.exc.InternalError, match="receipt was removed"),
    ):
        _horizontal_code()


@pytest.mark.parametrize("other", [1, 4])
def test_runtime_rejects_adjacent_views_of_mutable_owner(other):
    from helion.runtime.cute.horizontal_state import validate_horizontal_state_arguments

    plan, tensors = _host_guard_fixture()
    tensors = list(tensors)
    prefix = 4096  # one original slot; slot=-1 can address this owned prefix
    owner = torch.empty(prefix + tensors[0].numel(), dtype=tensors[0].dtype)
    tensors[0] = owner.as_strided(tensors[0].shape, tensors[0].stride(), prefix)
    tensors[other] = owner[: tensors[other].numel()].view(tensors[other].shape)
    assert (
        tensors[other].data_ptr() + tensors[other].numel() * 2 <= tensors[0].data_ptr()
    )
    with pytest.raises(ValueError, match="storage owners overlap"):
        validate_horizontal_state_arguments([plan], tuple(tensors))


def test_current_graph_rejects_disjoint_views_of_one_mutable_owner():
    args = list(_args())
    prefix = 4096
    owner = torch.empty(prefix + args[0].numel(), dtype=args[0].dtype)
    args[0] = owner.as_strided(args[0].shape, args[0].stride(), prefix)
    args[1] = owner[: args[1].numel()].view(args[1].shape)

    def check(region, graphs):
        with pytest.raises(InapplicableStateUpdate, match="alias"):
            region.check_readonly_owners()

    with patch(__name__ + "._args", return_value=tuple(args)):
        _with_region(check)


def test_runtime_rejects_distinct_storage_wrappers_over_one_allocation():
    from helion.runtime.cute.horizontal_state import validate_horizontal_state_arguments

    plan, tensors = _host_guard_fixture()
    tensors = list(tensors)
    prefix = 4096
    buffer = bytearray(2 * (prefix + tensors[0].numel()))
    first = torch.frombuffer(buffer, dtype=torch.bfloat16)
    second = torch.frombuffer(buffer, dtype=torch.bfloat16, count=tensors[1].numel())
    tensors[0] = first.as_strided(tensors[0].shape, tensors[0].stride(), prefix)
    tensors[1] = second.view(tensors[1].shape)
    assert tensors[0].untyped_storage()._cdata != tensors[1].untyped_storage()._cdata
    with pytest.raises(ValueError, match="storage owners overlap"):
        validate_horizontal_state_arguments([plan], tuple(tensors))


@pytest.mark.parametrize(
    "mutation", ["delete", "empty", "kind", "metadata", "writes", "reads", "alignment"]
)
def test_runtime_selected_module_contract_survives_metadata_mutation(mutation):
    from helion.runtime.cute.horizontal_state import bind_horizontal_state_kernel
    from helion.runtime.cute.horizontal_state import validate_horizontal_state_kernel

    plan, args = _host_guard_fixture()

    def kernel():
        pass

    kernel._helion_cute_wrapper_plans = [plan]
    bind_horizontal_state_kernel(kernel)
    validate_horizontal_state_kernel(kernel, args)
    bind_horizontal_state_kernel(kernel)
    if mutation == "delete":
        del kernel._helion_cute_wrapper_plans
    elif mutation == "empty":
        kernel._helion_cute_wrapper_plans = []
    elif mutation == "kind":
        plan["kind"] = "ordinary"
    elif mutation == "metadata":
        plan["metadata"][0] = plan["metadata"][1]
    elif mutation == "writes":
        plan["write_indices"] = [0, 1]
    elif mutation == "reads":
        plan["readonly_indices"] = [2, 3, 4]
    else:
        plan["aligned_indices"] = []
    with pytest.raises(ValueError, match="removed or changed"):
        validate_horizontal_state_kernel(kernel, args)
    with pytest.raises(ValueError):
        bind_horizontal_state_kernel(kernel)


def test_runtime_selected_contract_is_weak_and_ordinary_is_unchanged():
    import gc
    import weakref

    from helion.runtime.cute import horizontal_state as runtime

    plan, args = _host_guard_fixture()

    def kernel():
        pass

    # Ordinary modules have no new contract or argument validation.
    runtime.validate_horizontal_state_kernel(kernel, ())
    kernel._helion_cute_wrapper_plans = [plan]
    with pytest.raises(ValueError, match="not bound"):
        runtime.validate_horizontal_state_kernel(kernel, args)
    identity = id(kernel)
    runtime.bind_horizontal_state_kernel(kernel)
    reference = weakref.ref(kernel)
    del kernel
    gc.collect()
    assert reference() is None and identity not in runtime._contracts


def test_actual_selected_module_binds_before_any_launch():
    import types

    from helion.runtime.cute import horizontal_state as runtime

    source, args = _horizontal_code()
    module = types.ModuleType("horizontal_current_contract_test")
    exec(compile(source, "<horizontal current module>", "exec"), module.__dict__)
    kernel = module._helion__affine_rows
    assert runtime._contracts[id(kernel)][0]() is kernel
    # The actual host function supplies the original reordered launch tuple.
    captured = []

    def capture(fn, grid, *values, block):
        assert fn is kernel
        captured.append(values)

    module._affine_rows(*args, _launcher=capture)
    runtime.validate_horizontal_state_kernel(kernel, captured[0])
    kernel._helion_cute_wrapper_plans = []
    with pytest.raises(ValueError, match="removed or changed"):
        runtime.validate_horizontal_state_kernel(kernel, captured[0])
