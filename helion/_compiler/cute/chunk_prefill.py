"""Semantic proof for a fused local-factor chunk recurrence.

The arithmetic matcher proves the complete rounding DAG before Inductor prunes
duplicate operands. The region planner additionally proves loop bounds,
addressing, aliases and the initial/final state ABI before selecting an
asynchronous schedule. Keeping those obligations separate prevents an
arithmetic match from authorizing unsafe memory or loop rewrites.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import cast

import torch

from ...autotuner.config_spec import CUTE_CHUNK_PREFILL_SCHEDULE_KEY
from ...autotuner.config_spec import CUTE_CHUNK_PREFILL_TASK_ORDER_KEY
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _phi
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.scan_ops import _associative_scan
from ...language.tile_ops import tile_id
from ...language.tile_ops import tile_index
from ...language.view_ops import subscript
from .chunk_prepare import _packed_workspace_sources
from .chunk_prepare import _static_extent
from .chunk_prepare import _static_float
from .chunk_prepare import _zero_scalar
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _linear_offsets_fit_i32
from .fx_matcher import _load_ref
from .fx_matcher import _store_ref
from .fx_matcher import _TensorRef
from .fx_matcher import _xyz_grid_fits

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


_STORAGE_KEY = "cute_fused_chunk_disjoint_storage_v1"


def capture_chunk_prefill_semantics(device_ir: DeviceIR) -> None:
    """Retain original operands only for a candidate arithmetic region."""
    if any(match_chunk_prefill_step(g.graph) is not None for g in device_ir.graphs):
        device_ir.cute_semantic_graphs = tuple(g.copy() for g in device_ir.graphs)


def _disjoint_storage(values: Sequence[object]) -> bool:
    from torch._subclasses import FakeTensor

    if len(values) != 11 or any(not isinstance(v, torch.Tensor) for v in values):
        return False
    tensors = cast("tuple[torch.Tensor, ...]", tuple(values))
    if any(not t.is_contiguous() for t in tensors):
        return False
    spans = []
    for tensor in tensors:
        if isinstance(tensor, FakeTensor):
            base = tensor.untyped_storage()._cdata
            start = tensor.storage_offset() * tensor.element_size()
        else:
            base = 0
            start = tensor.data_ptr()
            if start % 16:
                return False
        spans.append((base, start, start + tensor.numel() * tensor.element_size()))
    for output_index in (9, 10):
        output = spans[output_index]
        for other_index, other in enumerate(spans):
            if output_index == other_index:
                continue
            if output[0] == other[0] and output[1] < other[2] and other[1] < output[2]:
                return False
    return True


def finalize_chunk_prefill_semantics(device_ir: DeviceIR) -> None:
    """Install the immutable region proof and a guard over runtime addresses."""
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import RuntimeInputSpecialization

    if not device_ir.cute_semantic_graphs:
        return
    region = match_chunk_prefill_region(device_ir, device_ir.cute_semantic_graphs)
    if region is None:
        return
    env = CompileEnvironment.current()
    refs = (
        *region.inputs,
        region.initial_state,
        region.cu_seqlens,
        region.output,
        region.final_state,
    )
    sources = _packed_workspace_sources(env, tuple(r.fake for r in refs))
    if sources is None:
        return
    env.register_runtime_input_specialization(
        _STORAGE_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(_STORAGE_KEY, tuple(map(repr, sources))),
            classifier=_disjoint_storage,
            reusable_tensor_properties=frozenset(("data_ptr", "storage_span")),
        ),
    )
    device_ir.cute_chunk_prefill_region = region


def _chunk_prefill_static_eligible(
    env: CompileEnvironment, region: CuteChunkPrefillRegion | None
) -> bool:
    """Share complete-root admission between search setup and device planning."""
    target = env.config_spec.target_device_capability
    if (
        region is None
        or env.backend_name != "cute"
        or not env.settings.fast_math
        or target is None
        or target < (10, 0)
        or target >= (11, 0)
        or region.key_width != 128
        or region.value_width != 128
        or "flat" not in env.config_spec.allowed_pid_types
        or env.settings.autotune_config_overrides.get("pid_type", "flat") != "flat"
    ):
        return False
    refs = (
        *region.inputs,
        region.initial_state,
        region.cu_seqlens,
        region.output,
        region.final_state,
    )
    return _linear_offsets_fit_i32(
        region.total_tokens * region.heads, tuple(ref.fake.numel() for ref in refs)
    ) and _xyz_grid_fits((region.heads, region.sequences, 1))


def register_chunk_prefill_search(env: CompileEnvironment, device_ir: DeviceIR) -> None:
    """Restrict tuning only when this bind can use the complete-root emitter."""
    from ..compile_environment import _replay_tensor_input_source

    region = device_ir.cute_chunk_prefill_region
    if not _chunk_prefill_static_eligible(env, region):
        return
    # Seed registration runs with real arguments, before immutable bound cache
    # facts are published. Replay the exact registered sources: flattened
    # operand views need not themselves be replayable inputs. Planning still
    # requires the corresponding captured cache-key result below.
    specialization = env.runtime_input_specializations[_STORAGE_KEY]
    values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in specialization.sources
    )
    if specialization.classifier(values) is True:
        env.config_spec.enable_cute_chunk_prefill_task_order_search()


def plan_chunk_prefill() -> None:
    """Select the available SM100 schedule using capability and storage facts."""
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    region = HostFunction.current().device_ir.cute_chunk_prefill_region
    if (
        not _chunk_prefill_static_eligible(env, region)
        or df.config.pid_type != "flat"
        or not env.runtime_input_specialization_matches_bound(_STORAGE_KEY, True)
    ):
        return
    df.cute_state.chunk_prefill_plan = region


def codegen_chunk_prefill(cg: GenerateAST) -> bool:
    """Emit a wrapper-owned host launch for the proven complete root."""
    df = cg.device_function
    region = df.cute_state.chunk_prefill_plan
    root = cg.current_root_graph_info
    if region is None or root is None or root.graph_id != region.root_graph_id:
        return False
    refs = (
        *region.inputs,
        region.initial_state,
        region.cu_seqlens,
        region.output,
        region.final_state,
    )
    names = tuple(df.tensor_arg(r.fake, prefer_name=r.name).name for r in refs)
    output_scale = df.literal_expr(region.step.output_scale.meta["val"]._sympy_())
    gate_scale = df.literal_expr(region.step.gate_scale.meta["val"]._sympy_())
    if not output_scale.isidentifier() or not gate_scale.isidentifier():
        return False
    keys = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "a_log",
        "dt",
        "initial_state",
        "cu_seqlens",
        "out",
        "final_state",
    )
    schedule = cast("str", df.config.get(CUTE_CHUNK_PREFILL_SCHEDULE_KEY, "single"))
    cg.cute_wrapper_plans.append(
        {
            "kind": "chunk_prefill_sm100",
            **{f"{key}_name": name for key, name in zip(keys, names, strict=True)},
            "scale_name": output_scale,
            "gate_scale_name": gate_scale,
            "heads": region.heads,
            "sequences": region.sequences,
            "total_tokens": region.total_tokens,
            "threads": 512,
            "task_order": df.config.get(CUTE_CHUNK_PREFILL_TASK_ORDER_KEY, "identity"),
            "schedule": schedule,
            "prefix_count": {"single": 0, "prefix_tail_2": 2, "prefix_tail_4": 4}[
                schedule
            ],
            "sequence_groups": min(4, region.sequences),
            "device_abi": 1,
        }
    )
    df.placeholder_args.update((*names, output_scale, gate_scale))
    df.preamble = []
    df.body = [ast.Pass()]
    cg.cute_uses_matmul = True
    return True


@dataclass(frozen=True)
class _Capture:
    name: str
    predicate: Callable[[object], bool] | None = None
    equivalent: Callable[[object, object], bool] | None = None


@dataclass(frozen=True)
class _Call:
    target: object
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()
    commutative: bool = False


@dataclass(frozen=True)
class _Either:
    choices: tuple[object, ...]


_Bindings: TypeAlias = dict[str, Any]


def _match(pattern: object, value: object, bindings: _Bindings) -> bool:
    """Match an explicit operator DAG, including casts and shared producers.

    Captures bind identity, so two equal-looking computations do not count as
    one producer.  Commutative matching is opt-in only for scalar multiplication
    and addition; contractions and rounding are never reassociated.
    """
    if isinstance(pattern, _Either):
        for choice in pattern.choices:
            trial = dict(bindings)
            if _match(choice, value, trial):
                bindings.clear()
                bindings.update(trial)
                return True
        return False
    if isinstance(pattern, _Capture):
        if pattern.predicate is not None and not pattern.predicate(value):
            return False
        if not pattern.name:
            return True
        if pattern.name in bindings:
            previous = bindings[pattern.name]
            return previous is value or (
                pattern.equivalent is not None and pattern.equivalent(previous, value)
            )
        bindings[pattern.name] = value
        return True
    if isinstance(pattern, _Call):
        if not (
            isinstance(value, torch.fx.Node)
            and value.op == "call_function"
            and value.target == pattern.target
            and len(value.args) == len(pattern.args)
            and tuple(value.kwargs) == tuple(k for k, _ in pattern.kwargs)
        ):
            return False
        orders = [value.args]
        if pattern.commutative and len(value.args) == 2:
            orders.append(tuple(reversed(value.args)))
        for args in orders:
            trial = dict(bindings)
            if _match(pattern.args, args, trial) and all(
                _match(p, value.kwargs[k], trial) for k, p in pattern.kwargs
            ):
                bindings.clear()
                bindings.update(trial)
                return True
        return False
    if isinstance(pattern, (tuple, list)):
        return (
            isinstance(value, (tuple, list))
            and len(pattern) == len(value)
            and all(_match(p, v, bindings) for p, v in zip(pattern, value, strict=True))
        )
    if isinstance(value, torch.fx.Node):
        return False
    return type(pattern) is type(value) and pattern == value


def _call(target: object, *args: object, commutative: bool = False) -> _Call:
    return _Call(target, args, commutative=commutative)


def _cast(value: object, dtype: torch.dtype) -> _Call:
    return _call(torch.ops.prims.convert_element_type.default, value, dtype)


def _round(value: object, dtype: torch.dtype) -> _Call:
    return _cast(_cast(value, dtype), torch.float32)


def _transpose(value: object) -> _Call:
    return _call(torch.ops.aten.permute.default, value, [1, 0])


def _column(value: object) -> _Call:
    return _call(subscript, value, [slice(None), None])


def _row(value: object) -> _Call:
    return _call(subscript, value, [None, slice(None)])


def _mul(left: object, right: object) -> _Call:
    return _call(torch.ops.aten.mul.Tensor, left, right, commutative=True)


def _add(left: object, right: object) -> _Call:
    return _call(torch.ops.aten.add.Tensor, left, right, commutative=True)


def _dot(left: object, right: object, acc: object = None) -> _Call:
    return _call(dot, left, right, acc, torch.float32)


def _where(mask: object, value: object) -> _Call:
    return _call(
        torch.ops.aten.where.self,
        mask,
        value,
        _Capture("", _zero_scalar),
    )


def _ancestors(value: object) -> set[torch.fx.Node]:
    if not isinstance(value, torch.fx.Node):
        return set()
    result: set[torch.fx.Node] = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if node not in result:
            result.add(node)
            pending.extend(node.all_input_nodes)
    return result


def _dtype(value: object, dtype: torch.dtype) -> bool:
    if not isinstance(value, torch.fx.Node):
        return False
    fake = value.meta.get("val")
    return isinstance(fake, torch.Tensor) and fake.dtype == dtype


def _host_float_scalar(value: object) -> bool:
    """Only host symbolic floats can become uniform wrapper scalar arguments."""
    return (
        isinstance(value, torch.fx.Node)
        and isinstance(value.meta.get("val"), torch.SymFloat)
        and _match(
            _call(_get_symnode, _Capture("", lambda expr: isinstance(expr, str))),
            value,
            {},
        )
    )


def _same_tile_index(left: object, right: object) -> bool:
    """Compare repeated pure tile-index views, never ordinary memory loads."""
    if left is right:
        return True
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            starmap(_same_tile_index, zip(left, right, strict=True))
        )
    if not isinstance(left, torch.fx.Node) or not isinstance(right, torch.fx.Node):
        return type(left) is type(right) and left == right
    if (
        left.op != "call_function"
        or right.op != "call_function"
        or left.target != right.target
    ):
        return False
    if left.target is load:
        if not all(
            isinstance(n.args[0], torch.fx.Node) and n.args[0].target is tile_index
            for n in (left, right)
        ):
            return False
    elif left.target not in (tile_index, torch.ops.aten.sym_size.int):
        return False
    return (
        not left.kwargs and not right.kwargs and _same_tile_index(left.args, right.args)
    )


def _block_inverse(lower: object) -> object:
    """The actual blockwise matrix inverse, without implicit RHS transposes."""
    index = _Capture(
        "inverse_index",
        lambda n: bool(
            isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target is torch.ops.prims.iota.default
            and n.args == (16,)
            and n.kwargs.get("start") == 0
            and n.kwargs.get("step") == 1
        ),
    )
    row = _call(torch.ops.aten.unsqueeze.default, index, 1)
    col = _call(torch.ops.aten.unsqueeze.default, index, 0)
    same_half = _call(
        torch.ops.aten.eq.Tensor,
        _call(torch.ops.aten.lt.Scalar, row, 8),
        _call(torch.ops.aten.lt.Scalar, col, 8),
    )
    lower_half = _call(
        torch.ops.aten.bitwise_and.Tensor,
        _call(torch.ops.aten.ge.Scalar, row, 8),
        _call(torch.ops.aten.lt.Scalar, col, 8),
    )
    diagonal = _where(same_half, lower)
    coupling = _where(lower_half, lower)
    diagonal2 = _dot(_cast(diagonal, torch.float16), _cast(diagonal, torch.float16))
    diagonal4 = _dot(_cast(diagonal2, torch.float16), _cast(diagonal2, torch.float16))
    eye = _cast(_call(torch.ops.aten.eq.Tensor, row, col), torch.float32)
    inverse0 = _call(torch.ops.aten.sub.Tensor, eye, _round(diagonal, torch.float16))
    inverse1 = _add(
        _round(inverse0, torch.float16),
        _dot(_cast(inverse0, torch.float16), _cast(diagonal2, torch.float16)),
    )
    inverse2 = _add(
        _round(inverse1, torch.float16),
        _dot(_cast(inverse1, torch.float16), _cast(diagonal4, torch.float16)),
    )
    first = _dot(_cast(inverse2, torch.float16), _cast(coupling, torch.float16))
    lower_left = _dot(_cast(first, torch.float16), _cast(inverse2, torch.float16))
    return _cast(
        _call(
            torch.ops.aten.where.self,
            lower_half,
            _call(torch.ops.aten.neg.default, lower_left),
            inverse2,
        ),
        torch.bfloat16,
    )


@dataclass(frozen=True)
class CuteChunkPrefillStep:
    """Exact arithmetic subgraph with explicit memory and loop boundaries."""

    state_input: torch.fx.Node
    state_output: torch.fx.Node
    output_store: torch.fx.Node
    q_load: torch.fx.Node
    k_load: torch.fx.Node
    value_load: torch.fx.Node
    gate_load: torch.fx.Node
    beta_load: torch.fx.Node
    a_log_load: torch.fx.Node
    bias_load: torch.fx.Node
    inverse: torch.fx.Node
    gate_scale: torch.fx.Node
    output_scale: torch.fx.Node
    gate_scan: torch.fx.Node
    row_coordinate: torch.fx.Node
    token_coordinate: torch.fx.Node
    feature_coordinate: torch.fx.Node
    value_coordinate: torch.fx.Node
    valid: torch.fx.Node
    arithmetic_nodes: frozenset[torch.fx.Node]


@dataclass(frozen=True)
class CuteChunkPrefillRegion:
    """Arithmetic and address proof, pending runtime storage guards.

    A backend may synthesize a full-value-width CTA only after registering
    guards that output and final state do not alias any input or each other.
    Sequence boundaries remain runtime inputs with the usual packed contract.
    """

    root_graph_id: int
    loop_graph_id: int
    step: CuteChunkPrefillStep
    sequences: int
    heads: int
    key_width: int
    value_width: int
    total_tokens: int
    initial_state: _TensorRef
    final_state: _TensorRef
    output: _TensorRef
    cu_seqlens: _TensorRef
    inputs: tuple[_TensorRef, ...]


def _is_iota(node: torch.fx.Node, size: int) -> bool:
    return bool(
        node.op == "call_function"
        and node.target is torch.ops.prims.iota.default
        and node.args == (size,)
        and node.kwargs.get("start") == 0
        and node.kwargs.get("step") == 1
    )


def match_chunk_prefill_region(
    device_ir: DeviceIR, graphs: Sequence[GraphInfo]
) -> CuteChunkPrefillRegion | None:
    """Prove a whole sequential root against pre-Inductor semantic graphs.

    ``graphs`` must preserve pointwise operands before Inductor's input-pruning
    rewrite. ``device_ir`` supplies the finalized logical grid domain. This
    function has no device launch, compilation side effect or name dispatch.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import HelperFunctionGraphInfo
    from ..device_ir import RootGraphInfo

    roots = [g for g in graphs if isinstance(g, RootGraphInfo)]
    loops = [g for g in graphs if isinstance(g, ForLoopGraphInfo)]
    helpers = [g for g in graphs if isinstance(g, HelperFunctionGraphInfo)]
    if len(graphs) != 3 or len(roots) != 1 or len(loops) != 1 or len(helpers) != 1:
        return None
    root, loop, helper = roots[0], loops[0], helpers[0]
    if len(loop.block_ids) != 1:
        return None
    step = match_chunk_prefill_step(loop.graph)
    if step is None or step.gate_scan.args[0] != helper.graph_id:
        return None
    helper_inputs = list(helper.graph.find_nodes(op="placeholder"))
    helper_outputs = list(helper.graph.find_nodes(op="output"))
    if len(helper_inputs) != 2 or len(helper_outputs) != 1:
        return None
    bindings: _Bindings = {"lhs": helper_inputs[0], "rhs": helper_inputs[1]}
    if not all(_dtype(n, torch.float32) for n in helper_inputs) or not _match(
        _add(_Capture("lhs"), _Capture("rhs")), helper_outputs[0].args[0], bindings
    ):
        return None
    if set(helper.graph.nodes) != set(helper_inputs) | _ancestors(helper_outputs[0]):
        return None
    root_stores = [
        n for n in root.graph.nodes if n.op == "call_function" and n.target is store
    ]
    if len(root_stores) != 1:
        return None
    final_store = root_stores[0]
    initial = _Capture("initial", lambda n: _dtype(n, torch.float32))
    loop_call = _Capture("loop")
    bindings = {}
    if (
        len(final_store.args) != 4
        or final_store.args[3] is not None
        or not _match(
            _call(_phi, initial, _call(operator.getitem, loop_call, 0)),
            final_store.args[2],
            bindings,
        )
    ):
        return None
    sequence, head, value = (
        _Capture("sequence"),
        _Capture("head"),
        _Capture("value", equivalent=_same_tile_index),
    )
    if not _match(
        _call(
            load,
            _Capture("initial_tensor"),
            [sequence, head, value, slice(None)],
            None,
            None,
        ),
        bindings["initial"],
        bindings,
    ) or not _match(
        [sequence, head, value, slice(None)], final_store.args[1], bindings
    ):
        return None
    begin, end = _Capture("begin"), _Capture("end")
    if not _match(
        _call(
            _for_loop,
            loop.graph_id,
            [0],
            [_call(torch.ops.aten.sub.Tensor, end, begin)],
            [end, begin, initial],
        ),
        bindings["loop"],
        bindings,
    ):
        return None
    cu_tensor = _Capture("cu_tensor")
    if not _match(
        _call(load, cu_tensor, [sequence], None, None), bindings["begin"], bindings
    ):
        return None
    if not _match(
        _call(load, cu_tensor, [_call(operator.add, sequence, 1)], None, None),
        bindings["end"],
        bindings,
    ):
        return None
    placeholders = list(loop.graph.find_nodes(op="placeholder"))
    if len(placeholders) != 3:
        return None
    if not _match(
        _call(_new_var, _Capture("state_placeholder")), step.state_input, bindings
    ):
        return None
    if bindings["state_placeholder"] is not placeholders[2]:
        return None
    refs = [
        _load_ref(n)
        for n in (
            step.q_load,
            step.k_load,
            step.value_load,
            step.gate_load,
            step.beta_load,
            step.a_log_load,
            step.bias_load,
        )
    ]
    initial_ref = _load_ref(bindings["initial"])
    final_ref = _store_ref(final_store)
    output_ref = _store_ref(step.output_store)
    cu_ref = _load_ref(bindings["begin"])
    if any(r is None for r in (*refs, initial_ref, final_ref, output_ref, cu_ref)):
        return None
    assert (
        initial_ref is not None
        and final_ref is not None
        and output_ref is not None
        and cu_ref is not None
    )
    if (
        len(initial_ref.fake.shape) != 4
        or final_ref.fake.shape != initial_ref.fake.shape
    ):
        return None
    sizes = tuple(_static_extent(s) for s in initial_ref.fake.shape)
    if any(s is None or s <= 0 for s in sizes):
        return None
    sequences, heads, value_width, key_width = sizes
    assert (
        sequences is not None
        and heads is not None
        and value_width is not None
        and key_width is not None
    )
    q_ref, k_ref, v_ref, gate_ref, beta_ref, a_ref, bias_ref = refs
    assert (
        q_ref is not None
        and k_ref is not None
        and v_ref is not None
        and gate_ref is not None
        and beta_ref is not None
        and a_ref is not None
        and bias_ref is not None
    )
    refs = [q_ref, k_ref, v_ref, gate_ref, beta_ref, a_ref, bias_ref]
    rows = _static_extent(q_ref.fake.size(0))
    if rows is None or rows % heads != 0 or q_ref.fake.shape != (rows, key_width):
        return None
    if k_ref.fake.shape != q_ref.fake.shape or gate_ref.fake.shape != q_ref.fake.shape:
        return None
    if (
        v_ref.fake.shape != (rows, value_width)
        or output_ref.fake.shape != v_ref.fake.shape
    ):
        return None
    if (
        beta_ref.fake.shape != (rows,)
        or a_ref.fake.shape != (heads,)
        or bias_ref.fake.shape != (heads, key_width)
    ):
        return None
    if cu_ref.fake.shape != (sequences + 1,) or cu_ref.fake.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return None
    if (
        final_ref.fake.dtype is not torch.float32
        or output_ref.fake.dtype is not torch.bfloat16
    ):
        return None
    if not all(
        r.fake.is_contiguous()
        for r in (*refs, initial_ref, final_ref, output_ref, cu_ref)
    ):
        return None
    # Tie every raw tensor row to the same sequence-local chunk coordinate.
    bindings["begin_placeholder"], bindings["end_placeholder"] = (
        placeholders[1],
        placeholders[0],
    )
    bindings["token_lane"] = step.token_coordinate
    bindings["head_coordinate"] = cast(
        "Sequence[torch.fx.Node]", step.a_log_load.args[1]
    )[0]
    chunk_id = _Capture("chunk_id")
    token = _add(
        _add(
            _call(_new_var, _Capture("begin_placeholder")),
            _call(operator.mul, chunk_id, 16),
        ),
        _Capture("token_lane"),
    )
    if not _match(
        _add(_mul(token, heads), _Capture("head_coordinate")),
        step.row_coordinate,
        bindings,
    ):
        return None
    if not _match(
        _call(
            torch.ops.aten.lt.Tensor,
            token,
            _call(_new_var, _Capture("end_placeholder")),
        ),
        step.valid,
        bindings,
    ):
        return None
    if not _match(_call(tile_id, _call(_get_symnode, "16")), bindings["chunk_id"], {}):
        return None
    if not _is_iota(step.token_coordinate, 16) or not _is_iota(
        step.feature_coordinate, key_width
    ):
        return None
    value_size = _call(torch.ops.aten.sym_size.int, _Capture("state_placeholder"), 0)
    if not _match(
        _Either(
            (
                value_size,
                _call(
                    load,
                    _call(tile_index, value_size),
                    [None, slice(None)],
                    None,
                    None,
                ),
            )
        ),
        step.value_coordinate,
        bindings,
    ):
        return None
    # Inner and outer head IDs must denote the same logical grid axis.
    inner_head, outer_head = bindings["head_coordinate"], bindings["head"]
    if not (
        isinstance(inner_head, torch.fx.Node)
        and isinstance(outer_head, torch.fx.Node)
        and inner_head.target is tile_id
        and outer_head.target is tile_id
        and cast("torch.fx.Node", inner_head.args[0]).meta.get("val")
        == cast("torch.fx.Node", outer_head.args[0]).meta.get("val")
    ):
        return None
    value_coordinate = bindings["value"]
    if not (
        isinstance(bindings["sequence"], torch.fx.Node)
        and isinstance(value_coordinate, torch.fx.Node)
        and value_coordinate.op == "call_function"
        and value_coordinate.target is tile_index
        and len(value_coordinate.args) == 1
        and isinstance(value_coordinate.args[0], torch.fx.Node)
    ):
        return None
    if (
        _canonical_root_axis_ids(
            device_ir,
            root_phase_index=root.phase_index,
            coordinates=(
                bindings["sequence"],
                bindings["head"],
                value_coordinate.args[0],
            ),
            expected_extents=(sequences, heads, value_width),
        )
        is None
    ):
        return None
    if any(
        n not in _ancestors(final_store) and n.op != "output" for n in root.graph.nodes
    ):
        return None
    return CuteChunkPrefillRegion(
        root.graph_id,
        loop.graph_id,
        step,
        sequences,
        heads,
        key_width,
        value_width,
        rows // heads,
        initial_ref,
        final_ref,
        output_ref,
        cu_ref,
        tuple(refs),
    )


def match_chunk_prefill_step(graph: torch.fx.Graph) -> CuteChunkPrefillStep | None:
    """Prove local-factor arithmetic without using names or workload shapes.

    Supported semantics retain the authoritative state in FP32.  State MMA
    operands, gate/key factors, projection, residual, beta multiplication and
    solved update each have distinct, mandatory BF16 rounding boundaries.
    The inverse helper separately proves every FP16 operation and its masks.
    """
    nodes = list(graph.nodes)
    outputs = [n for n in nodes if n.op == "output"]
    stores = [n for n in nodes if n.op == "call_function" and n.target is store]
    if len(outputs) != 1 or len(stores) != 1:
        return None
    returned = outputs[0].args[0]
    if not isinstance(returned, (tuple, list)) or len(returned) != 1:
        return None
    state_output = returned[0]
    output_store = stores[0]
    bindings: _Bindings = {}
    state = _Capture("state", lambda n: _dtype(n, torch.float32))
    gamma = _Capture("gamma")
    solved = _Capture("solved")
    key_restore = _Capture("key_restore")
    if not _match(
        _dot(_transpose(solved), key_restore, _mul(state, _row(gamma))),
        state_output,
        bindings,
    ):
        return None
    inverse = _Capture("inverse")
    rhs = _Capture("rhs")
    if not _match(
        _cast(_dot(inverse, rhs), torch.bfloat16), bindings["solved"], bindings
    ):
        return None
    q_decay, k_decay = _Capture("q_decay"), _Capture("k_decay")
    qk = _Capture("qk")
    scale = _Capture("scale", _host_float_scalar)
    result = _dot(qk, solved, _dot(q_decay, _cast(_transpose(state), torch.bfloat16)))
    if len(output_store.args) != 4 or not _match(
        _cast(_mul(result, scale), torch.bfloat16), output_store.args[2], bindings
    ):
        return None
    beta, raw_value = _Capture("beta"), _Capture("value")
    projection = _dot(k_decay, _cast(_transpose(state), torch.bfloat16))
    residual = _round(
        _call(
            torch.ops.aten.sub.Tensor,
            _cast(raw_value, torch.float32),
            _round(projection, torch.bfloat16),
        ),
        torch.bfloat16,
    )
    if not _match(
        _cast(_mul(_column(_round(beta, torch.bfloat16)), residual), torch.bfloat16),
        bindings["rhs"],
        bindings,
    ):
        return None
    token = _Capture("token")
    key_inverse = _Capture("key_inverse")
    causal = _call(torch.ops.aten.ge.Tensor, _column(token), _row(token))
    if not _match(
        _cast(_where(causal, _dot(q_decay, _transpose(key_inverse))), torch.bfloat16),
        bindings["qk"],
        bindings,
    ):
        return None
    # Constants created by distinct torch.where calls need not share storage.
    bindings.pop("zero", None)
    lower_pattern = _round(
        _where(
            _call(torch.ops.aten.gt.Tensor, _column(token), _row(token)),
            _mul(_dot(k_decay, _transpose(key_inverse)), _column(beta)),
        ),
        torch.bfloat16,
    )
    inverse_node = bindings["inverse"]
    if not _match(_block_inverse(lower_pattern), inverse_node, bindings):
        return None
    exp_gate = _Capture("exp_gate")
    normalized_q, normalized_k = _Capture("normalized_q"), _Capture("normalized_k")
    ki_fp32 = _mul(normalized_k, _call(torch.ops.aten.reciprocal.default, exp_gate))
    for pattern, name in (
        (_cast(_mul(normalized_q, exp_gate), torch.bfloat16), "q_decay"),
        (_cast(_mul(normalized_k, exp_gate), torch.bfloat16), "k_decay"),
        (_cast(ki_fp32, torch.bfloat16), "key_inverse"),
        (_cast(_mul(ki_fp32, _row(gamma)), torch.bfloat16), "key_restore"),
    ):
        if not _match(pattern, bindings[name], bindings):
            return None
    for name, raw_name in (("normalized_q", "q"), ("normalized_k", "k")):
        raw = _cast(_Capture(raw_name), torch.float32)
        norm = _call(
            torch.ops.aten.rsqrt.default,
            _call(
                torch.ops.aten.clamp_min.default,
                _call(torch.ops.aten.sum.dim_IntList, _mul(raw, raw), [-1]),
                1e-24,
            ),
        )
        if not _match(_mul(raw, _column(norm)), bindings[name], bindings):
            return None
    bindings.pop("zero", None)
    gamma_pattern = _call(
        torch.ops.aten.sum.dim_IntList,
        _where(_column(_call(torch.ops.aten.eq.Scalar, token, 15)), exp_gate),
        [0],
    )
    if not _match(gamma_pattern, bindings["gamma"], bindings):
        return None
    bindings.pop("zero", None)
    gate, bias, a_log = _Capture("gate"), _Capture("bias"), _Capture("a_log")
    log2_e = _Capture("log2_e", lambda n: _static_float(n) == 1.4426950408889634)
    gate_scale = _Capture("gate_scale", _host_float_scalar)
    valid = _Capture("valid")
    increment = _mul(
        gate_scale,
        _add(
            _mul(
                _call(
                    torch.ops.aten.tanh.default,
                    _mul(
                        _mul(
                            _call(torch.ops.aten.exp2.default, _mul(a_log, log2_e)),
                            _add(_cast(gate, torch.float32), _row(bias)),
                        ),
                        0.5,
                    ),
                ),
                0.5,
            ),
            0.5,
        ),
    )
    scan_id = _Capture("scan_id")
    scan = _call(
        _associative_scan, scan_id, _where(_column(valid), increment), 0, False, False
    )
    if not _match(
        _call(torch.ops.aten.exp2.default, scan), bindings["exp_gate"], bindings
    ):
        return None
    if not _match(
        _call(
            torch.ops.aten.sigmoid.default, _cast(_Capture("beta_raw"), torch.float32)
        ),
        bindings["beta"],
        bindings,
    ):
        return None
    # Memory boundary proof: factors above are local SSA values.  The caller
    # must prove row/feature coordinates against its enclosing sequential loop.
    row, feature = _Capture("row"), _Capture("feature")
    for name in ("q", "k", "gate"):
        if not _match(
            _call(
                load,
                _Capture(name + "_tensor"),
                _Either(([row, feature], [_column(row), _row(feature)])),
                _column(valid),
                None,
            ),
            bindings[name],
            bindings,
        ):
            return None
    if not _match(
        _call(load, _Capture("beta_tensor"), [row], valid, None),
        bindings["beta_raw"],
        bindings,
    ):
        return None
    value_coordinate = _Capture("value_coordinate", equivalent=_same_tile_index)
    if (
        not _match(
            _call(
                load,
                _Capture("value_tensor"),
                _Either(([row, value_coordinate], [_column(row), value_coordinate])),
                _column(valid),
                None,
            ),
            bindings["value"],
            bindings,
        )
        or not _match(
            _Either(([row, value_coordinate], [_column(row), value_coordinate])),
            output_store.args[1],
            bindings,
        )
        or not _match(_column(valid), output_store.args[3], bindings)
    ):
        return None
    head = _Capture("head")
    if not _match(
        _call(load, _Capture("bias_tensor"), [head, feature], None, None),
        bindings["bias"],
        bindings,
    ):
        return None
    if not _match(
        _call(load, _Capture("a_log_tensor"), [head], None, None),
        bindings["a_log"],
        bindings,
    ):
        return None
    required_nodes = (
        "q",
        "k",
        "value",
        "gate",
        "beta_raw",
        "a_log",
        "bias",
        "gate_scale",
        "scale",
        "row",
        "token",
        "feature",
        "valid",
    )
    if any(not isinstance(bindings[name], torch.fx.Node) for name in required_nodes):
        return None
    if any(
        not _dtype(bindings[name], torch.bfloat16)
        for name in ("q", "k", "value", "gate", "beta_raw")
    ):
        return None
    if any(not _dtype(bindings[name], torch.float32) for name in ("a_log", "bias")):
        return None
    arithmetic = _ancestors(state_output) | _ancestors(output_store)
    # No extra side effects or live numerical work may escape the proved DAG.
    if any(
        n not in arithmetic and n.op not in ("output", "placeholder") for n in nodes
    ):
        return None
    return CuteChunkPrefillStep(
        bindings["state"],
        state_output,
        output_store,
        bindings["q"],
        bindings["k"],
        bindings["value"],
        bindings["gate"],
        bindings["beta_raw"],
        bindings["a_log"],
        bindings["bias"],
        inverse_node,
        bindings["gate_scale"],
        bindings["scale"],
        bindings["exp_gate"].args[0],
        bindings["row"],
        bindings["token"],
        bindings["feature"],
        bindings["value_coordinate"],
        bindings["valid"],
        frozenset(arithmetic),
    )
