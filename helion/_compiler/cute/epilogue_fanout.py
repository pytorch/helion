"""Share a typed, rounded epilogue prefix between two fresh output stores.

The proof consumes the original typed host and complete FX fanout. The emitter
consumes structured pieces from the ordinary store lowering; it does not match
generated variable names or search generated source for an arithmetic pattern.
Both output arenas remain distinct. One TMA commit group owns both reads of the
same ring slot, and only the final shared TMEM traversal releases the producer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import textwrap
from typing import TYPE_CHECKING

import torch

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from .cute_epilogue import Tcgen05UnaryEpilogueChain
    from .cute_epilogue import _AuxiliaryTensorLoadExpr
    from .device_state import CuteTcgen05StoreValue


FANOUT_CONFIG_KEY = "tcgen05_epilogue_fanout"
FANOUT_MODES = ("off", "shared")


def _fresh_returned_tensors(host: HostFunction) -> frozenset[torch.Tensor]:
    """Prove direct fresh factories and a host without intervening alias effects.

    Fake storage identity alone is insufficient: an opaque host helper or a
    later ``set_`` could change ownership on the next call. Admit only direct
    known factories, metadata assignments, device roots and a literal return of
    names. This also works for the typed stage views of materialized fission,
    which retain the owning host's allocation prelude.
    """
    from ... import language as hl
    from ...language.constexpr import specialize
    from ...language.tunable_ops import register_block_size
    from ..ast_extension import ExtendedAST
    from ..ast_extension import LoopType
    from ..type_info import CallableType
    from ..type_info import LiteralType
    from ..type_info import PythonModuleType
    from ..type_info import SequenceType
    from ..type_info import SymIntType
    from ..type_info import TensorType
    from .promote_output_axis import _FRESH_FACTORIES

    tensors = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    occupied = set(host.params.arguments)
    allocations: dict[str, torch.Tensor] = {}
    metadata_names: set[str] = set()
    seen_root = False

    def literal(node: ast.AST) -> bool:
        return (
            isinstance(node, ExtendedAST)
            and isinstance(info := node._type_info, LiteralType)
            and type(info.value)
            in (
                int,
                float,
                bool,
                str,
                type(None),
                torch.dtype,
                torch.device,
                torch.layout,
                torch.memory_format,
            )
        )

    def torch_attribute(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and isinstance(node.value, ExtendedAST)
            and isinstance(info := node.value._type_info, PythonModuleType)
            and info.value is torch
        )

    def integer_metadata_type(info: object, *, sequence: bool) -> bool:
        if isinstance(info, SymIntType):
            return True
        if isinstance(info, LiteralType):
            return type(info.value) in (int, bool)
        return (
            sequence
            and isinstance(info, SequenceType)
            and all(
                integer_metadata_type(element, sequence=True)
                for element in info.element_types
            )
        )

    def builtin_metadata_call(node: ast.AST, *, register: bool) -> bool:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ExtendedAST)
            and isinstance(callee := node.func._type_info, CallableType)
            and (
                isinstance(node.func, ast.Name)
                or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and isinstance(node.func.value, ExtendedAST)
                    and isinstance(
                        module := node.func.value._type_info, PythonModuleType
                    )
                    and module.value is hl
                )
            )
            and not node.keywords
        ):
            return False
        if callee.value is specialize:
            if len(node.args) != 1:
                return False
            sequence = True
        elif register and callee.value is register_block_size:
            if len(node.args) not in (1, 2):
                return False
            sequence = False
        else:
            return False
        return all(
            isinstance(arg, ExtendedAST)
            and integer_metadata_type(arg._type_info, sequence=sequence)
            and metadata(arg)
            for arg in node.args
        )

    def metadata(node: ast.AST) -> bool:
        # A sampled LiteralType is not an effect proof: an opaque callback or
        # property can return an int while changing a fresh output's storage.
        # Track only pure local definitions, immutable scalar names, standard
        # tensor metadata and direct torch constants, recursively.
        if isinstance(node, ast.Name):
            return node.id in tensors or node.id in metadata_names or literal(node)
        if isinstance(node, ast.Constant):
            return type(node.value) in (int, float, bool, str, type(None))
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(metadata(item) for item in node.elts)
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name)
                and node.value.id in tensors
                and node.attr in ("shape", "dtype", "device", "ndim")
            ) or (torch_attribute(node) and literal(node))
        if isinstance(node, ast.Call):
            if builtin_metadata_call(node, register=True):
                return True
            return (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in tensors
                and node.func.attr in ("size", "stride", "dim", "ndimension")
                and not node.keywords
                and all(metadata(arg) for arg in node.args)
            )
        if isinstance(node, ast.Subscript):
            return metadata(node.value) and metadata(node.slice)
        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
                and metadata(node.left)
                and metadata(node.right)
            )
        return False

    def returned(node: ast.AST | None) -> set[str] | None:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for item in node.elts:
                child = returned(item)
                if child is None:
                    return None
                names.update(child)
            return names
        return None

    for index, stmt in enumerate(host.body):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if (
            not seen_root
            and isinstance(stmt, ast.Expr)
            and builtin_metadata_call(stmt.value, register=False)
        ):
            continue
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt, ExtendedAST)
            and stmt._loop_type is LoopType.GRID
        ):
            seen_root = True
            continue
        if isinstance(stmt, ast.Return) and index == len(host.body) - 1:
            returned_names = returned(stmt.value)
            if not seen_root or returned_names is None:
                return frozenset()
            return frozenset(
                value for name, value in allocations.items() if name in returned_names
            )
        if seen_root or not isinstance(stmt, ast.Assign):
            return frozenset()
        names: list[str] = []
        for target in stmt.targets:
            elements = (
                target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            )
            if any(not isinstance(element, ast.Name) for element in elements):
                return frozenset()
            names.extend(
                element.id for element in elements if isinstance(element, ast.Name)
            )
        if len(set(names)) != len(names) or occupied.intersection(names):
            return frozenset()
        occupied.update(names)
        value = stmt.value
        if isinstance(value, ExtendedAST) and isinstance(value._type_info, TensorType):
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ExtendedAST)
                and (isinstance(value.func, ast.Name) or torch_attribute(value.func))
                and isinstance(callee := value.func._type_info, CallableType)
                and any(callee.value is factory for factory in _FRESH_FACTORIES)
                and len(names) == 1
                and all(metadata(arg) for arg in value.args)
                and all(
                    kw.arg not in (None, "out") and metadata(kw.value)
                    for kw in value.keywords
                )
            ):
                return frozenset()
            allocations[names[0]] = value._type_info.proxy()
            tensors.add(names[0])
        else:
            if not metadata(value):
                return frozenset()
            metadata_names.update(names)
    return frozenset()


@dataclass(frozen=True)
class PairedFanoutPlan:
    stores: tuple[torch.fx.Node, torch.fx.Node]
    chains: tuple[Tcgen05UnaryEpilogueChain, Tcgen05UnaryEpilogueChain]
    output_dtype: torch.dtype
    block_ids: tuple[int, int]
    shape: tuple[int, int]
    prefix_steps: int
    # True: the second chain consumes the first output's rounded value. False:
    # both chains are identical and the second output copies that value directly.
    rounded_suffix: bool
    pre_wait_aux_safe: bool = False


def shared_prefix(
    first: Tcgen05UnaryEpilogueChain,
    second: Tcgen05UnaryEpilogueChain,
    dtype: torch.dtype,
) -> tuple[int, bool] | None:
    """Compare complete typed expression nodes, including rounding templates."""
    from .cute_epilogue import _FLOAT_CAST_TYPES
    from .cute_epilogue import _AuxiliaryTensorExprStep
    from .cute_epilogue import _CurrentTensorExpr
    from .cute_epilogue import _round_epilogue_expression
    from .cute_epilogue import _UnaryOp
    from .cute_epilogue import _UnaryTensorExpr

    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    count = len(first.steps)
    if not count or first.auxiliary_tensor_loads or second.steps[:count] != first.steps:
        return None
    if len(second.steps) == count:
        return count, False
    cast_template = (
        "{inner}.to(cutlass.Float32)"
        if dtype is torch.float32
        else _round_epilogue_expression("{inner}", dtype)
    )
    cast_step = _AuxiliaryTensorExprStep(
        _UnaryTensorExpr(_UnaryOp(f"to_{dtype}", cast_template), _CurrentTensorExpr())
    )
    assert dtype in _FLOAT_CAST_TYPES
    if second.steps[count] != cast_step:
        return None
    return count, True


def prove_paired_fanout(
    env: CompileEnvironment,
    host: HostFunction,
    graphs: Sequence[GraphInfo],
    stores: Sequence[torch.fx.Node],
    anchors: set[torch.fx.Node],
) -> PairedFanoutPlan | None:
    from ...language import _tracing_ops
    from ...language import memory_ops
    from ..indexing_strategy import exact_tile_block_ids
    from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
    from .cute_fx_walk import build_inner_outputs_index_from_graphs
    from .promote_output_axis import _access_tensor
    from .promote_output_axis import _ordinary_operation

    if (
        len(stores) != 2
        or stores[0] is stores[1]
        or stores[0].graph is not stores[1].graph
    ):
        return None
    graph = stores[0].graph
    ordered = tuple(node for node in graph.nodes if node in stores)
    if len(ordered) != 2:
        return None
    fresh = _fresh_returned_tensors(host)
    tensors: list[torch.Tensor] = []
    axes: list[tuple[int, ...]] = []
    chains: list[Tcgen05UnaryEpilogueChain] = []
    producer: torch.fx.Node | None = None
    inner_outputs = build_inner_outputs_index_from_graphs(graphs)
    for store in ordered:
        if (
            store.target is not memory_ops.store
            or store.kwargs
            or len(store.args) != 4
            or store.args[3] is not None
        ):
            return None
        tensor = _access_tensor(store)
        indices, value = store.args[1:3]
        if (
            tensor is None
            or tensor not in fresh
            or tensor.ndim != 2
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or any(type(size) is not int or size <= 0 for size in tensor.shape)
            or not env.known_equal(tensor.stride(1), 1)
            or not env.known_equal(tensor.stride(0), tensor.shape[1])
            or not isinstance(indices, (list, tuple))
            or not isinstance(value, torch.fx.Node)
        ):
            return None
        block_ids = exact_tile_block_ids(env, indices)
        if block_ids is None or len(block_ids) != 2:
            return None
        analyzed = analyze_tcgen05_unary_epilogue_chain(
            None,
            value,
            output_global_shape=tuple(tensor.shape),
            target_fx_nodes=anchors,
            inner_outputs_by_graph_id=inner_outputs,
        )
        if analyzed is None or producer is not None and analyzed[1] is not producer:
            return None
        chain, producer = analyzed
        tensors.append(tensor)
        axes.append(block_ids)
        chains.append(chain)
    first, second = tensors
    if (
        first.dtype is not second.dtype
        or first.shape != second.shape
        or first.stride() != second.stride()
        or axes[0] != axes[1]
        or first.untyped_storage()._cdata == second.untyped_storage()._cdata
    ):
        return None
    common = shared_prefix(chains[0], chains[1], first.dtype)
    if common is None:
        return None
    output_storage = {tensor.untyped_storage()._cdata for tensor in tensors}
    writes: set[int] = set()
    between = False
    for node in graph.nodes:
        if node is ordered[0]:
            between = True
        if node is ordered[1]:
            between = False
        if node.target is memory_ops.store:
            target = _access_tensor(node)
            if target is None or between and node not in ordered:
                return None
            writes.add(target.untyped_storage()._cdata)
        elif node.target is memory_ops.load:
            target = _access_tensor(node)
            if target is None or target.untyped_storage()._cdata in output_storage:
                return None
        elif between and (
            not _ordinary_operation(node)
            or _tracing_ops.is_for_loop_target(node.target)
            or node.target is _tracing_ops._if
        ):
            return None
    for load in chains[1].auxiliary_tensor_loads:
        target = _access_tensor(load.load_node)
        if (
            load.broadcast_axis is not None
            or target is None
            or target.untyped_storage()._cdata in writes
        ):
            return None
    assert len(axes[0]) == 2
    return PairedFanoutPlan(
        (ordered[0], ordered[1]),
        (chains[0], chains[1]),
        first.dtype,
        (axes[0][0], axes[0][1]),
        (int(first.shape[0]), int(first.shape[1])),
        common[0],
        common[1],
        # Moving a read ahead of the ACC wait must not cross an aliased writer.
        # Fresh factories are disjoint on every call, unlike sampled argument
        # storage IDs. Reject other writers throughout the enclosing regions.
        all(
            _ordinary_operation(node)
            and (node.target is not memory_ops.store or _access_tensor(node) in fresh)
            for info in graphs
            for node in info.graph.nodes
        ),
    )


def schedule_supported(config: dict[str, object]) -> bool:
    """Standard role-local stores, with one CTA or an M-paired CTA group."""
    expected = {
        "pid_type": "persistent_interleaved",
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_cluster_n": 1,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_warp_spec_mma_warps": 1,
        "tcgen05_warp_spec_ab_load_warps": 1,
        "tcgen05_warp_spec_scheduler_warps": 0,
        "tcgen05_warp_spec_c_input_warps": 0,
        "tcgen05_warp_spec_store_warps": 0,
        "tcgen05_layout_strategy": "default",
        "tcgen05_epilogue_layout": "normal",
        "tcgen05_c_store_mode": "normal",
        "tcgen05_acc_wait_placement": "subtile_loop",
        "tcgen05_aux_load_mode": "simt",
    }
    return (
        all(
            type(config.get(key, default)) is type(default)
            and config.get(key, default) == default
            for key, default in expected.items()
        )
        and type(config.get("tcgen05_cluster_m", 1)) is int
        and config.get("tcgen05_cluster_m", 1) in (1, 2)
        and config.get("tcgen05_cta_group", "auto")
        in (
            ("auto", "two")
            if config.get("tcgen05_cluster_m", 1) == 2
            else ("auto", "one")
        )
        and config.get("tcgen05_c_acquire_placement", "pre_loop")
        in ("pre_loop", "before_store")
        and config.get("tcgen05_aux_load_placement", "post_acc_wait")
        in ("post_acc_wait", "pre_acc_wait")
        and config.get("tcgen05_c_stages", 2) in (2, 4)
        and (
            config.get("tcgen05_cluster_m", 1) == 1
            or (
                config.get("tcgen05_c_stages", 2) == 2
                and config.get("tcgen05_acc_stages", 2) == 2
            )
        )
        and not config.get("tcgen05_grouped_mode")
        and not config.get("tcgen05_flat_role_coordinates")
        and not config.get("tcgen05_diagnostic_invalid_output")
        and all(
            value is None
            for key, value in config.items()
            if key.startswith("tcgen05_layout_overrides_")
        )
    )


def store_protocol_supported(value: CuteTcgen05StoreValue) -> bool:
    from .device_state import Tcgen05Orientation

    return (
        value.use_role_local_epi
        and value.use_tma_store_epilogue
        and not value.tma_store_full_tiles_only
        and not value.partial_output_tma_store
        and not value.pure_matmul_role_lifecycle
        and (
            not value.lifecycle_context.is_two_cta
            or value.bm == 256
            and value.bn == 128
            and value.c_stage_count == 2
        )
        and value.orientation is Tcgen05Orientation.MN
        and not value.output_column_major
        and not value.segment_store_m_offset
        and not value.has_explicit_epilogue_tile
        and bool(value.role_local_tile_counter)
        and value.epi_warp_count == 4
        and value.c_stage_count in (2, 4)
    )


@dataclass(frozen=True)
class RenderedChain:
    chain: Tcgen05UnaryEpilogueChain
    source: str
    result: str
    # Every old name is allocated in its old order, even for a shared-away step.
    # Later regions, wrapper plans and unaffected setup consequently keep names.
    step_variables: tuple[tuple[tuple[str, str], ...], ...]


def render_chain(
    chain: Tcgen05UnaryEpilogueChain,
    carrier: str,
    new_var: Callable[[str], str],
    indent: str,
    aux: dict[_AuxiliaryTensorLoadExpr, str],
) -> RenderedChain:
    source: list[str] = []
    all_variables: list[tuple[tuple[str, str], ...]] = []
    for step in chain.steps:
        variables: list[tuple[str, str]] = []

        def record(prefix: str, variables: list[tuple[str, str]] = variables) -> str:
            name = new_var(prefix)
            variables.append((prefix, name))
            return name

        prelude, carrier = step.render_prelude_and_expr(carrier, aux, record, indent)
        source.append(prelude)
        all_variables.append(tuple(variables))
    return RenderedChain(chain, "".join(source), carrier, tuple(all_variables))


def _chain_key(chain: Tcgen05UnaryEpilogueChain) -> tuple[object, ...]:
    """Aux leaves are rebuilt by each walker; their FX load identity is stable."""
    from .cute_epilogue import _AuxiliaryTensorLoadExpr
    from .cute_epilogue import _BinaryTensorExpr
    from .cute_epilogue import _CurrentTensorExpr
    from .cute_epilogue import _UnaryTensorExpr

    def key(expr: object) -> object:
        if isinstance(expr, _CurrentTensorExpr):
            return ("current",)
        if isinstance(expr, _AuxiliaryTensorLoadExpr):
            return ("aux", expr.load_node, expr.broadcast_axis, expr.template)
        if isinstance(expr, _UnaryTensorExpr):
            return ("unary", expr.step, key(expr.operand))
        if isinstance(expr, _BinaryTensorExpr):
            return (
                "binary",
                expr.op_name,
                expr.op_template,
                key(expr.lhs),
                key(expr.rhs),
            )
        raise AssertionError(f"unexpected typed epilogue expression {type(expr)}")

    return tuple(key(step.expr) for step in chain.steps)


def render_shared_suffix(
    plan: PairedFanoutPlan,
    rendered: RenderedChain,
    first_value: str,
    indent: str,
    aux: dict[_AuxiliaryTensorLoadExpr, str],
) -> tuple[str, str]:
    """Replay the old suffix names, replacing only its proved redundant cast."""
    assert _chain_key(rendered.chain) == _chain_key(plan.chains[1])
    if not plan.rounded_suffix:
        return "", first_value
    from .cute_epilogue import _AuxiliaryTensorExprStep
    from .cute_epilogue import _CurrentTensorExpr
    from .cute_epilogue import _UnaryOp
    from .cute_epilogue import _UnaryTensorExpr

    source: list[str] = []
    carrier = first_value
    for index in range(plan.prefix_steps, len(rendered.chain.steps)):
        step = rendered.chain.steps[index]
        if index == plan.prefix_steps:
            step = _AuxiliaryTensorExprStep(
                _UnaryTensorExpr(
                    _UnaryOp("promote_rounded_output", "{inner}.to(cutlass.Float32)"),
                    _CurrentTensorExpr(),
                )
            )
        variables = iter(rendered.step_variables[index])

        def reuse(prefix: str, variables: Iterator[tuple[str, str]] = variables) -> str:
            expected, name = next(variables)
            assert expected == prefix
            return name

        prelude, carrier = step.render_prelude_and_expr(carrier, aux, reuse, indent)
        assert next(variables, None) is None
        source.append(prelude)
    return "".join(source), carrier


@dataclass(frozen=True)
class FanoutIteration:
    tmem_read: str
    auxiliary_loads: str
    arithmetic: str
    assignment: str
    value_name: str
    target_dtype: str
    release: str
    register_store: str
    chain: RenderedChain
    auxiliary_names: dict[_AuxiliaryTensorLoadExpr, str]
    pre_wait_aux: bool = False


@dataclass(frozen=True)
class FanoutStore:
    plan: PairedFanoutPlan
    site: torch.fx.Node
    main: ast.If
    setup: tuple[str, ...]
    setup_without_acquire: tuple[str, ...]
    iteration: FanoutIteration
    subtile_count: str
    acquire: str
    wait: str
    barrier: str
    buffer: str
    buffer_expr: str
    c_stages: int
    r2s: str
    tma: str
    commit: str
    advance: str
    late_acquire: str = ""


def combine_stores(first: FanoutStore, second: FanoutStore) -> None:
    """Replace the first marked body with one ordered two-output traversal."""
    plan = first.plan
    if (
        first.site is not plan.stores[0]
        or second.site is not plan.stores[1]
        or second.plan.stores != plan.stores
        or first.c_stages != second.c_stages
        or first.iteration.release
        or not second.iteration.release
        or first.advance
        or not second.advance
        or first.iteration.pre_wait_aux != second.iteration.pre_wait_aux
        or bool(first.late_acquire) != bool(second.late_acquire)
        or first.iteration.pre_wait_aux
        and not plan.pre_wait_aux_safe
    ):
        raise exc.BackendUnsupported("cute", "incomplete shared epilogue lifetime")
    left, right = first.iteration, second.iteration
    assert _chain_key(left.chain.chain) == _chain_key(plan.chains[0])
    suffix, result = render_shared_suffix(
        plan, right.chain, left.value_name, "        ", right.auxiliary_names
    )
    pre_wait_aux = left.pre_wait_aux
    body = (
        (left.auxiliary_loads + right.auxiliary_loads if pre_wait_aux else "")
        + first.acquire
        + first.wait
        + left.tmem_read
        + ("" if pre_wait_aux else left.auxiliary_loads)
        + left.arithmetic
        + left.assignment
        + ("" if pre_wait_aux else right.auxiliary_loads)
        + suffix
        + f"        {right.value_name} = ({result}).to({right.target_dtype})\n"
        + right.release
        + left.register_store
        + right.register_store
        + first.late_acquire
        + f"        {first.barrier}.arrive_and_wait()\n"
        + f"        {first.buffer} = ({first.buffer_expr}) % cutlass.Int32({first.c_stages})\n"
        + f"        {second.buffer} = ({second.buffer_expr}) % cutlass.Int32({second.c_stages})\n"
        + first.r2s
        + second.r2s
        + "        cute.arch.fence_view_async_shared()\n"
        + f"        {first.barrier}.arrive_and_wait()\n"
        + first.tma
        + second.tma
        + first.commit
    )
    # Both TMA strings include their shared warp-0 predicate only in `first.tma`.
    loop = (
        f"assert cutlass.const_expr({first.subtile_count} == {second.subtile_count})\n"
        f"for _tcgen05_subtile in cutlass.range({first.subtile_count}, unroll_full=True):\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
        + second.advance
    )
    source = "\n".join((*first.setup, *second.setup_without_acquire, loop))
    first.main.body[:] = ast.parse(textwrap.dedent(source)).body
