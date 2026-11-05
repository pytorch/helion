"""Bounded same-layout cross-store epilogue DAG sharing.

The old paired rounded-prefix recipe remains first and unchanged. This fallback
shares only typed original FX values and proven-readonly loads in one common
subtile; every output retains its own C arena, pipeline and visibility protocol.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
import textwrap
from typing import TYPE_CHECKING

import torch

from ... import exc
from . import cute_epilogue as epi
from .epilogue_fanout import PairedFanoutPlan
from .epilogue_fanout import _fresh_returned_tensors
from .epilogue_fanout import prove_paired_fanout
from .epilogue_fanout import schedule_supported as paired_schedule_supported
from .epilogue_fanout import store_protocol_supported
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from .device_state import CuteTcgen05StoreValue
    from .epilogue_fanout import FanoutStore


@dataclass(frozen=True)
class SharedDagFanoutPlan:
    stores: tuple[torch.fx.Node, ...]
    chains: tuple[epi.Tcgen05UnaryEpilogueChain, ...]
    output_dtype: torch.dtype
    block_ids: tuple[int, int]
    shape: tuple[int, int]
    pre_wait_aux_safe: bool


FanoutPlan = PairedFanoutPlan | SharedDagFanoutPlan


def _atom(node: epi._TensorExpr) -> object:
    if isinstance(node, epi._CurrentTensorExpr):
        return ("current",)
    if isinstance(node, epi._AuxiliaryTensorLoadExpr):
        return (type(node), node.load_node, node.broadcast_axis, node.template)
    if isinstance(node, epi._RuntimeScalarExpr):
        return ("scalar", node.source)
    if isinstance(node, epi._ScalarConstantExpr):
        return ("constant", node.literal, node.dtype)
    if isinstance(node, epi._UnaryTensorExpr):
        return ("unary", node.step)
    if isinstance(node, epi._BinaryTensorExpr):
        return ("binary", node.op_name, node.op_template)
    if isinstance(node, epi._WhereTensorExpr):
        return ("where", node.op_template)
    raise ValueError("nested or unknown cross-store DAG expression")


def dag_keys(dag: epi._PointwiseDagExpr) -> tuple[object, ...]:
    """Bounded keys: typed original FX origins, operators and child references.

    Child references are shallow typed-origin/operator identities, not recursive
    expanded text/trees. Reconvergence consequently stays O(nodes + edges).
    """
    if len(dag.nodes) != len(dag.provenance):
        raise ValueError("cross-store DAG has no complete typed FX provenance")
    references = {
        id(node): (origin, dtype, _atom(node))
        for node, (origin, dtype) in zip(dag.nodes, dag.provenance, strict=True)
    }
    keys: list[object] = []
    seen: set[int] = set()
    for node in dag.nodes:
        children: tuple[epi._TensorExpr, ...] = ()
        if isinstance(node, epi._UnaryTensorExpr):
            children = (node.operand,)
        elif isinstance(node, epi._BinaryTensorExpr):
            children = (node.lhs, node.rhs)
        elif isinstance(node, epi._WhereTensorExpr):
            children = (node.predicate, node.lhs, node.rhs)
        if any(id(child) not in seen for child in children):
            raise ValueError("cross-store DAG is not dependency ordered")
        keys.append((references[id(node)], tuple(references[id(c)] for c in children)))
        seen.add(id(node))
    return tuple(keys)


def raw_aux_key(leaf: epi._AuxiliaryTensorLoadExpr) -> object:
    return (leaf.load_node, leaf.broadcast_axis, epi._node_tensor_dtype(leaf.load_node))


def _store_interval_is_pure(
    graph: torch.fx.Graph, stores: Sequence[torch.fx.Node]
) -> bool:
    """Moving stores into the first body cannot cross control or other effects."""
    from torch.utils._pytree import tree_leaves

    from ...language import _tracing_ops
    from .promote_output_axis import _ordinary_operation

    if not stores:
        return False
    between = False
    for node in graph.nodes:
        if node is stores[0]:
            between = True
        if node is stores[-1]:
            return between
        if (
            between
            and node not in stores
            and (
                node.graph is not graph
                or any(
                    isinstance(argument, torch.fx.Node) and argument.graph is not graph
                    for argument in tree_leaves((node.args, node.kwargs))
                )
                or not (
                    _ordinary_operation(node) or node.target in epi._ZERO_ARG_TARGETS
                )
                or _tracing_ops.is_for_loop_target(node.target)
                or node.target is _tracing_ops._if
            )
        ):
            return False
    return False


def prove_shared_dag_fanout(
    env: CompileEnvironment,
    host: HostFunction,
    graphs: Sequence[GraphInfo],
    stores: Sequence[torch.fx.Node],
    anchors: set[torch.fx.Node],
    *,
    allow_unbound: bool = False,
) -> SharedDagFanoutPlan | None:
    from ...language import memory_ops
    from ..indexing_strategy import exact_tile_block_ids
    from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
    from .promote_output_axis import _access_tensor
    from .register_loads import host_load_is_readonly

    if not 2 <= len(stores) <= 4 or len(set(stores)) != len(stores):
        return None
    graph = stores[0].graph
    if any(store.graph is not graph for store in stores):
        return None
    ordered = tuple(node for node in graph.nodes if node in stores)
    if len(ordered) != len(stores) or {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    } != set(ordered):
        return None
    if not _store_interval_is_pure(graph, ordered):
        return None
    outputs: list[torch.Tensor] = []
    eligible_host_tensors = _fresh_returned_tensors(host, allow_external=True)
    chains: list[epi.Tcgen05UnaryEpilogueChain] = []
    axes: tuple[int, int] | None = None
    anchor: torch.fx.Node | None = None
    keys: set[object] = set()
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
            or tensor not in eligible_host_tensors
            or tensor.ndim != 2
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or any(type(size) is not int or size <= 0 for size in tensor.shape)
            or not env.known_equal(tensor.stride(1), 1)
            or not env.known_equal(tensor.stride(0), tensor.shape[1])
            or not isinstance(indices, (list, tuple))
            or not isinstance(value, torch.fx.Node)
        ):
            return None
        blocks = exact_tile_block_ids(env, indices)
        if blocks is None or len(blocks) != 2:
            return None
        current_axes = (blocks[0], blocks[1])
        if axes is not None and axes != current_axes:
            return None
        axes = current_axes
        if outputs and (
            tensor.dtype is not outputs[0].dtype
            or tensor.shape != outputs[0].shape
            or tensor.stride() != outputs[0].stride()
        ):
            return None
        analyzed = epi._analyze_tcgen05_pointwise_dag(
            value,
            graphs,
            anchors,
            tuple(tensor.shape),
            allow_unbound=allow_unbound,
        )
        if analyzed is None or anchor is not None and anchor is not analyzed[1]:
            return None
        chain, anchor = analyzed
        dag = chain.steps[0].expr
        if not isinstance(dag, epi._PointwiseDagExpr):
            return None
        try:
            keys.update(dag_keys(dag))
        except ValueError:
            return None
        if len(keys) > 64 or any(
            leaf.broadcast_axis not in (None, 1)
            for leaf in chain.auxiliary_tensor_loads
        ):
            return None
        outputs.append(tensor)
        chains.append(chain)
    # Only already registered, dispatch/cache-backed alias facts may prove
    # external destinations. No classifier registration or publication here.
    allow_unbound = allow_unbound or env.early_mma_metadata_scope_active
    for left, right in itertools.combinations(outputs, 2):
        if left.untyped_storage() == right.untyped_storage() or not (
            env.tensor_storage_is_compiler_allocated(left)
            or env.tensor_storage_is_compiler_allocated(right)
            or runtime_tensors_are_proven_disjoint(
                env, left, right, allow_unbound=allow_unbound
            )
        ):
            return None
    pre_wait_aux_safe = all(
        host_load_is_readonly(node, env, graphs, allow_unbound=allow_unbound)
        for info in graphs
        for node in info.graph.nodes
        if node.target is memory_ops.load
    )
    if not pre_wait_aux_safe:
        return None
    assert axes is not None
    return SharedDagFanoutPlan(
        ordered,
        tuple(chains),
        outputs[0].dtype,
        axes,
        (int(outputs[0].shape[0]), int(outputs[0].shape[1])),
        pre_wait_aux_safe,
    )


def prove_epilogue_fanout(
    env: CompileEnvironment,
    host: HostFunction,
    graphs: Sequence[GraphInfo],
    stores: Sequence[torch.fx.Node],
    anchors: set[torch.fx.Node],
    *,
    allow_unbound: bool = False,
) -> FanoutPlan | None:
    paired = prove_paired_fanout(env, host, graphs, stores, anchors)
    return (
        paired
        if paired is not None
        else prove_shared_dag_fanout(
            env, host, graphs, stores, anchors, allow_unbound=allow_unbound
        )
    )


def schedule_supported(plan: FanoutPlan, config: dict[str, object]) -> bool:
    if isinstance(plan, PairedFanoutPlan):
        return paired_schedule_supported(config)
    expected = {
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_num_epi_warps": 4,
        "tcgen05_epi_warp_groups": 1,
        "tcgen05_warp_spec_mma_warps": 1,
        "tcgen05_warp_spec_ab_load_warps": 1,
        "tcgen05_warp_spec_c_input_warps": 0,
        "tcgen05_warp_spec_store_warps": 0,
        "tcgen05_layout_strategy": "default",
        "tcgen05_epilogue_layout": "normal",
        "tcgen05_c_store_mode": "normal",
        "tcgen05_acc_wait_placement": "subtile_loop",
        "tcgen05_aux_load_mode": "simt",
        "tcgen05_aux_fragment_hoist": False,
        "tcgen05_aux_prefetch_subtiles": 0,
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_c_acquire_placement": "pre_loop",
    }
    strategy = config.get("tcgen05_strategy", "role_local_monolithic")
    return (
        all(
            type(config.get(key, default)) is type(default)
            and config.get(key, default) == default
            for key, default in expected.items()
        )
        and config.get("pid_type") in ("persistent_interleaved", "persistent_blocked")
        and strategy in ("role_local_monolithic", "role_local_with_scheduler")
        and config.get("tcgen05_warp_spec_scheduler_warps", 0)
        == (1 if strategy == "role_local_with_scheduler" else 0)
        and config.get("tcgen05_c_stages", 2) in (2, 4)
        and config.get("tcgen05_cta_group", "auto") in ("auto", "one")
        and config.get("tcgen05_aux_load_placement", "post_acc_wait")
        in ("post_acc_wait", "pre_acc_wait")
        and not config.get("tcgen05_grouped_mode")
        and not config.get("tcgen05_flat_role_coordinates")
        and not config.get("tcgen05_diagnostic_invalid_output")
        and all(
            value is None
            for key, value in config.items()
            if key.startswith("tcgen05_layout_overrides_")
        )
    )


def protocol_supported(plan: FanoutPlan, value: CuteTcgen05StoreValue) -> bool:
    return store_protocol_supported(value) and (
        isinstance(plan, PairedFanoutPlan)
        or not value.lifecycle_context.is_two_cta
        and value.bm <= TCGEN05_ONE_CTA_MAX_BLOCK_M
        and not value.has_explicit_epilogue_tile
    )


def render_dag(
    dag: epi._PointwiseDagExpr,
    carrier: str,
    aux: epi._LeafBindings,
    factory: Callable[[str], str],
    memo: dict[object, str],
    indent: str,
) -> tuple[str, str]:
    names: dict[int, str] = {}
    lines: list[str] = []
    for node, key in zip(dag.nodes, dag_keys(dag), strict=True):
        if isinstance(node, epi._UnaryTensorExpr):
            expression = node.step.template.format(inner=names[id(node.operand)])
        elif isinstance(node, epi._BinaryTensorExpr):
            expression = node.op_template.format(
                lhs=names[id(node.lhs)], rhs=names[id(node.rhs)]
            )
        elif isinstance(node, epi._WhereTensorExpr):
            expression = node.op_template.format(
                predicate=names[id(node.predicate)],
                lhs=names[id(node.lhs)],
                rhs=names[id(node.rhs)],
            )
        else:
            source, name = epi._render_auxiliary_tensor_expr(
                node, carrier, aux, factory, indent
            )
            if key not in memo:
                lines.append(source)
                memo[key] = name
            names[id(node)] = memo[key]
            continue
        # Consume the original name even when the node is shared away. This
        # preserves allocation order and rejects missing renderer provenance.
        name = factory("tcgen05_dag_step")
        if key not in memo:
            lines.append(f"{indent}{name} = {expression}\n")
            memo[key] = name
        names[id(node)] = memo[key]
    return "".join(lines), names[id(dag.nodes[-1])]


def combine_dag_stores(stores: Sequence[FanoutStore]) -> None:
    if not stores:
        raise exc.BackendUnsupported("cute", "empty shared-DAG fanout stores")
    plan = stores[0].plan
    if (
        not isinstance(plan, SharedDagFanoutPlan)
        or tuple(s.site for s in stores) != plan.stores
    ):
        raise exc.BackendUnsupported("cute", "unordered shared-DAG fanout stores")
    first, last = stores[0], stores[-1]
    if (
        any(
            not isinstance(s.plan, SharedDagFanoutPlan)
            or s.plan.stores != plan.stores
            or s.plan.output_dtype is not plan.output_dtype
            or s.plan.block_ids != plan.block_ids
            or s.plan.shape != plan.shape
            or s.c_stages != first.c_stages
            or not s.setup
            or not s.acquire
            or not s.barrier
            or not s.buffer
            or not s.buffer_expr
            or not s.r2s
            or not s.tma
            or not s.commit
            or (s is not first and s.wait)
            or s.body_indent
            or s.buffer_prefix
            or s.pre_loop
            or s.pre_loop_wait
            or s.post_loop
            or s.late_acquire
            or s.iteration.pre_wait_aux != first.iteration.pre_wait_aux
            or (s is not last and (s.iteration.release or s.advance))
            for s in stores
        )
        or not last.iteration.release
        or not last.advance
        or not first.wait
        or not first.iteration.tmem_read
        or not first.iteration.chain.carrier
        or not first.iteration.chain.carrier_prelude
        or len({s.commit for s in stores}) != len(stores)
    ):
        raise exc.BackendUnsupported("cute", "incomplete shared-DAG full-tile lifetime")
    memo: dict[object, str] = {}
    aux_names: dict[object, str] = {}
    chunks: list[list[str]] = []
    bindings: list[epi._LeafBindings] = []
    for store in stores:
        loads: list[str] = []
        local = dict(store.iteration.auxiliary_names)
        if (
            tuple(leaf for leaf, _ in store.iteration.auxiliary_load_sources)
            != store.iteration.chain.chain.auxiliary_tensor_loads
        ):
            raise exc.BackendUnsupported(
                "cute", "incomplete shared-DAG auxiliary load proof"
            )
        for leaf, source in store.iteration.auxiliary_load_sources:
            if not source or leaf not in local:
                raise exc.BackendUnsupported(
                    "cute", "incomplete shared-DAG auxiliary load source"
                )
            key = raw_aux_key(leaf)
            if key not in aux_names:
                aux_names[key] = local[leaf]
                loads.append(source)
            local[leaf] = aux_names[key]
        chunks.append(loads)
        bindings.append(local)
    pre_wait = first.iteration.pre_wait_aux
    body = "".join(chunk for group in chunks for chunk in group) if pre_wait else ""
    body += (
        "".join(store.acquire for store in stores)
        + first.wait
        + first.iteration.tmem_read
        + first.iteration.chain.carrier_prelude
    )
    carrier = first.iteration.chain.carrier
    for index, store in enumerate(stores):
        chain = plan.chains[index]
        dag = chain.steps[0].expr
        assert isinstance(dag, epi._PointwiseDagExpr)
        if (
            len(store.iteration.chain.chain.steps) != 1
            or len(store.iteration.chain.step_variables) != 1
        ):
            raise exc.BackendUnsupported("cute", "shared-DAG source/proof disagreement")
        actual = store.iteration.chain.chain.steps[0].expr
        if (
            not isinstance(actual, epi._PointwiseDagExpr)
            or len(actual.nodes) != len(actual.provenance)
            or dag_keys(actual) != dag_keys(dag)
        ):
            raise exc.BackendUnsupported("cute", "shared-DAG source/proof disagreement")
        variables = iter(store.iteration.chain.step_variables[0])

        def original_name(
            prefix: str, variables: Iterator[tuple[str, str]] = variables
        ) -> str:
            expected, name = next(variables)
            assert prefix == expected
            return name

        arithmetic, result = render_dag(
            actual, carrier, bindings[index], original_name, memo, "        "
        )
        assert next(variables, None) is None
        if not pre_wait:
            body += "".join(chunks[index])
        body += (
            arithmetic
            + f"        {store.iteration.value_name} = ({result}).to({store.iteration.target_dtype})\n"
            + store.iteration.release
            + store.iteration.register_store
            + f"        {store.barrier}.arrive_and_wait()\n"
            + f"        {store.buffer} = ({store.buffer_expr}) % cutlass.Int32({store.c_stages})\n"
            + store.r2s
            + "        cute.arch.fence_view_async_shared()\n"
            + f"        {store.barrier}.arrive_and_wait()\n"
            + store.tma
            + store.commit
        )
    checks = "".join(
        f"assert cutlass.const_expr({first.subtile_count} == {s.subtile_count})\n"
        for s in stores[1:]
    )
    loop = (
        checks
        + (
            first.loop_header
            or f"for _tcgen05_subtile in cutlass.range({first.subtile_count}, unroll_full=True):\n"
        )
        + textwrap.indent(textwrap.dedent(body), "    ")
        + last.advance
    )
    source = "\n".join(line for store in stores for line in store.setup) + "\n" + loop
    first.main.body[:] = ast.parse(textwrap.dedent(source)).body
