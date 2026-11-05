"""Prove the result-fission schedule of one native GEMM epilogue.

The schedule splits the ordinary fused kernel into two ordered launches on one
stream: a native TCGEN05 producer GEMM that stores the accumulator into the
original output ``V`` (an output the program already returns), and a CuTe SIMT
consumer that reads ``V`` back and computes every other original store with the
original statements, casts and FP32 operation order.

Eligibility (every check fails closed):

* the complete shared-DAG fanout proof of the original program
  (``prove_shared_dag_fanout``): one graph, one MMA anchor, 2-4 full unmasked
  same-layout stores, a tagged store-only effect interval, every load
  read-only and every output pairwise disjoint through the registered
  dispatch/cache classifiers;
* exactly one of those stores is the pure current accumulator (a FP32 store of
  the FP32 loop-carried value, through same-dtype identity steps only), all
  outputs are FP32, and every other store reaches the accumulator only through
  that value (``V`` dominates every accumulator use);
* BF16/F16 dense row-major operands loaded directly (or through an exact
  widen-then-narrow round trip) over the whole K extent;
* the source root is ``acc = hl.zeros(...)``, one K loop, then straight-line
  ``name = expr`` / ``out[tile_m, tile_n] = expr`` statements whose consumer
  part references only kernel parameters, its own locals, the tile names,
  ``torch`` and ``hl`` -- never the accumulator except through ``V``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import linecache
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..device_ir import ForLoopGraphInfo
from .split_k_workspace import _direct_load
from .split_k_workspace import _getitem
from .split_k_workspace import _is_call
from .split_k_workspace import _output
from .split_k_workspace import _shape
from .split_k_workspace import _static
from .split_k_workspace import _strides
from .split_k_workspace import _strip_new_vars
from .split_k_workspace import _tensor

if TYPE_CHECKING:
    import types

    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..host_function import HostFunction


CONSUMER_FUNCTION = "_helion_result_fission_consumer"
_ALLOWED_GLOBALS = frozenset(("torch", "hl"))
_ALLOWED_HL = frozenset(("full", "zeros"))


@dataclass(frozen=True)
class ResultFissionProof:
    root_graph_id: int
    identity_index: int
    v_output: str
    outputs: tuple[str, ...]
    lhs: torch.Tensor
    rhs: torch.Tensor
    v_tensor: torch.Tensor
    lhs_expression: str
    rhs_expression: str
    lhs_strides: tuple[int, int]
    rhs_strides: tuple[int, int]
    operand_dtype: torch.dtype
    m: int
    n: int
    k: int
    m_block_id: int
    n_block_id: int
    k_block_id: int
    consumer_source: str
    consumer_params: tuple[str, ...]
    consumer_statements: tuple[str, ...]
    v_name: str


def _exact_operand(node: object, env: CompileEnvironment) -> object:
    """Strip one exact ``x.to(float32).to(x.dtype)`` round trip of a half load."""
    convert = torch.ops.prims.convert_element_type.default
    if (
        _is_call(node, convert)
        and isinstance(node, torch.fx.Node)
        and len(node.args) == 2
        and _is_call(node.args[0], convert)
    ):
        inner = node.args[0]
        assert isinstance(inner, torch.fx.Node)
        source = inner.args[0]
        source_value = (
            source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        )
        if (
            len(inner.args) == 2
            and inner.args[1] is torch.float32
            and isinstance(source_value, torch.Tensor)
            and source_value.dtype in (torch.float16, torch.bfloat16)
            and node.args[1] is source_value.dtype
            and len(inner.users) == 1
        ):
            return source
    return node


def _reaches(start: object, targets: set[Node], stop: set[Node]) -> bool:
    stack = [start]
    seen: set[Node] = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in seen or node in stop:
            continue
        if node in targets:
            return True
        seen.add(node)
        stack.extend(node.all_input_nodes)
    return False


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _alias_names(root: ast.AST) -> tuple[str, ...]:
    """Epilogue names bound as a plain alias ``name = acc`` of the accumulator."""
    assert isinstance(root, ast.For)
    body = root.body
    if not (
        body
        and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1
        and isinstance(body[0].targets[0], ast.Name)
    ):
        return ()
    acc = body[0].targets[0].id
    return tuple(
        stmt.targets[0].id
        for stmt in body[2:]
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == acc
    )


def _check_consumer_expression(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if isinstance(
            node,
            (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.Starred),
        ):
            return False
        if isinstance(node, ast.Call):
            func = node.func
            text = ast.unparse(func)
            if text.startswith("hl."):
                if text[3:] not in _ALLOWED_HL:
                    return False
            elif isinstance(func, ast.Attribute) and func.attr in ("to", "reshape"):
                continue
            elif not text.startswith("torch."):
                return False
    return True


def _consumer_program(
    host: HostFunction,
    root: ast.For,
    outputs: tuple[str, ...],
    v_output: str,
    m: int,
    n: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str] | None:
    params = tuple(host.params.arguments)
    target = root.target
    if not (
        isinstance(target, ast.Tuple)
        and len(target.elts) == 2
        and all(isinstance(e, ast.Name) for e in target.elts)
    ):
        return None
    tiles = cast("tuple[str, str]", tuple(cast("ast.Name", e).id for e in target.elts))
    body = root.body
    if (
        len(body) < 3
        or not isinstance(body[0], ast.Assign)
        or len(body[0].targets) != 1
        or not isinstance(body[0].targets[0], ast.Name)
        or not isinstance(body[1], ast.For)
    ):
        return None
    acc = body[0].targets[0].id
    loop_names = {
        t.id
        for s in ast.walk(body[1])
        if isinstance(s, ast.Assign)
        for t in s.targets
        if isinstance(t, ast.Name)
    } | ({body[1].target.id} if isinstance(body[1].target, ast.Name) else set())
    epilogue: list[ast.Assign] = []
    defs: dict[str, int] = {}
    stores: dict[str, int] = {}
    order: list[str] = []
    for index, stmt in enumerate(body[2:]):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        epilogue.append(stmt)
        lhs = stmt.targets[0]
        if (_names(stmt.value) & loop_names) - {acc}:
            return None
        if isinstance(lhs, ast.Name):
            if lhs.id in defs or lhs.id in params or lhs.id in (acc, *tiles):
                return None
            defs[lhs.id] = index
        elif (
            isinstance(lhs, ast.Subscript)
            and isinstance(lhs.value, ast.Name)
            and lhs.value.id in outputs
            and lhs.value.id not in stores
            and isinstance(lhs.slice, ast.Tuple)
            and [ast.unparse(e) for e in lhs.slice.elts] == list(tiles)
        ):
            stores[lhs.value.id] = index
            order.append(lhs.value.id)
        else:
            return None
    if tuple(order) != outputs:
        return None
    v_value = epilogue[stores[v_output]].value
    if isinstance(v_value, ast.Name):
        v_name = v_value.id
    elif (
        isinstance(v_value, ast.Call)
        and isinstance(v_value.func, ast.Attribute)
        and v_value.func.attr == "to"
        and isinstance(v_value.func.value, ast.Name)
        and len(v_value.args) == 1
        and not v_value.keywords
        and ast.unparse(v_value.args[0]) == "torch.float32"
    ):
        v_name = v_value.func.value.id
    else:
        return None
    if v_name not in defs:
        return None
    v_def = epilogue[defs[v_name]].value
    if not (
        (isinstance(v_def, ast.Name) and v_def.id == acc)
        or ast.unparse(v_def) == f"{acc}.to(torch.float32)"
    ):
        return None
    allowed = set(params) | set(defs) | set(tiles) | _ALLOWED_GLOBALS
    lines = [f"{v_name} = {v_output}[{tiles[0]}, {tiles[1]}]"]
    for index, stmt in enumerate(epilogue):
        if index in (defs[v_name], stores[v_output]):
            continue
        names = _names(stmt)
        if acc in names or not names <= allowed or _names(stmt.value) & set(outputs):
            return None
        if not _check_consumer_expression(stmt.value):
            return None
        lines.append(ast.unparse(stmt))
    used: set[str] = set()
    for line in lines:
        used |= _names(ast.parse(line))
    consumer_params = tuple(p for p in params if p in used)
    for name in consumer_params:
        if not isinstance(host.params.arguments[name], torch.Tensor):
            return None
    signature = ", ".join(f"{p}: torch.Tensor" for p in consumer_params)
    inner = "\n".join(f"        {line}" for line in lines)
    source = (
        "import torch\n"
        "import helion\n"
        "import helion.language as hl\n\n"
        f"def {CONSUMER_FUNCTION}({signature}) -> None:\n"
        f"    for {tiles[0]}, {tiles[1]} in hl.tile([{m}, {n}]):\n"
        f"{inner}\n"
    )
    module = ast.parse(source)
    if acc in _names(module):
        return None
    return source, consumer_params, tuple(lines), v_name


def consumer_function(source: str) -> types.FunctionType:
    """Materialize the consumer as a Python function with retrievable source."""
    import helion
    import helion.language as hl

    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    filename = f"<helion_result_fission_consumer_{digest}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"torch": torch, "helion": helion, "hl": hl}
    exec(compile(source, filename, "exec"), namespace)
    function = namespace[CONSUMER_FUNCTION]
    assert callable(function)
    return cast("types.FunctionType", function)


def analyze_result_fission(
    env: CompileEnvironment, ir: DeviceIR, *, allow_unbound: bool = False
) -> ResultFissionProof | None:
    from . import cute_epilogue as epi
    from .shared_epilogue_dag import prove_shared_dag_fanout

    host = ir.host_function
    capability = env.config_spec.target_device_capability
    if (
        env.backend_name != "cute"
        or host is None
        or env._is_distributed
        or env.cute_fission_plan is not None
        or len(ir.root_ids) != 1
        or len(ir.grid_block_ids[0]) != 2
        or capability is None
        or capability[0] != 10
    ):
        return None
    grid = [
        statement
        for statement in host.body
        if isinstance(statement, ast.For)
        and isinstance(statement, ExtendedAST)
        and statement._loop_type is LoopType.GRID
    ]
    if len(grid) != 1:
        return None
    root = ir.graphs[ir.root_ids[0]]
    calls = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.op == "call_function"
    ]
    stores = [node for node in calls if node.target is memory_ops.store]
    contractions = [
        node
        for node in calls
        if node.target in (torch.ops.aten.addmm.default, torch.ops.aten.mm.default)
    ]
    loops = [node for node in root.graph.nodes if node.target is _tracing_ops._for_loop]
    if (
        not 2 <= len(stores) <= 4
        or len(contractions) != 1
        or contractions[0].target is not torch.ops.aten.addmm.default
        or len(loops) != 1
        or _output(root) is not None
        or any(store.graph is not root.graph for store in stores)
    ):
        return None
    mm, loop = contractions[0], loops[0]
    if loop.kwargs or len(loop.args) != 4 or type(loop.args[0]) is not int:
        return None
    inner = ir.graphs[loop.args[0]]
    if (
        not isinstance(inner, ForLoopGraphInfo)
        or len(inner.block_ids) != 1
        or mm.graph is not inner.graph
        or mm.kwargs
        or len(mm.args) != 3
    ):
        return None
    (k_axis,) = inner.block_ids
    lhs = _direct_load(_exact_operand(mm.args[1], env), env)
    rhs = _direct_load(_exact_operand(mm.args[2], env), env)
    if lhs is None or rhs is None:
        return None
    a, a_expression, a_axes = lhs
    b, b_expression, b_axes = rhs
    a_shape, b_shape = _shape(env, a), _shape(env, b)
    a_strides, b_strides = _strides(env, a), _strides(env, b)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.dtype not in (torch.float16, torch.bfloat16)
        or b.dtype is not a.dtype
        or a_shape is None
        or b_shape is None
        or a_strides is None
        or b_strides is None
        or a_strides[1] != 1
        or b_strides[1] != 1
        or a_strides[0] < a_shape[1]
        or b_strides[0] < b_shape[1]
        or any(
            stride != 1 and stride * a.element_size() % 16
            for stride in (*a_strides, *b_strides)
        )
    ):
        return None
    m_axis, k_a = a_axes
    k_b, n_axis = b_axes
    m, k = a_shape
    k2, n = b_shape
    if (
        k_a != k_axis
        or k_b != k_axis
        or m_axis == n_axis
        or set(ir.grid_block_ids[0]) != {m_axis, n_axis}
        or k != k2
        or env.specialize_expr(env.block_sizes[m_axis].numel) != m
        or env.specialize_expr(env.block_sizes[n_axis].numel) != n
        or env.specialize_expr(env.block_sizes[k_axis].numel) != k
        or max(m, n, k, m * n) >= 1 << 31
    ):
        return None
    start, stop, carried = loop.args[1:]
    if any(
        not isinstance(values, (list, tuple)) or len(values) != 1
        for values in (start, stop, carried)
    ):
        return None
    begin, end, initial = start[0], stop[0], carried[0]
    if (
        _static(env, begin) != 0
        or _static(env, end) != k
        or not _is_call(initial, creation_ops.full)
        or not isinstance(initial, torch.fx.Node)
        or len(initial.args) != 4
        or type(initial.args[1]) not in (int, float)
        or initial.args[1] != 0
        or math.copysign(1, initial.args[1]) != 1
        or initial.args[2:] != (torch.float32, None)
        or inner.node_args != [initial]
    ):
        return None
    placeholders = tuple(inner.graph.find_nodes(op="placeholder"))
    matched: set[Node] = set()
    if (
        len(placeholders) != 1
        or _strip_new_vars(mm.args[0], matched) is not placeholders[0]
        or _output(inner) != [mm]
    ):
        return None
    current = [
        node
        for node in root.graph.nodes
        if _is_call(node, _tracing_ops._phi)
        and len(node.args) == 2
        and node.args[0] is initial
        and _getitem(node.args[1], loop, 0, matched)
    ]
    if len(current) != 1:
        return None
    accumulator = current[0]
    plan = prove_shared_dag_fanout(
        env, host, ir.graphs, stores, {mm}, allow_unbound=allow_unbound
    )
    if plan is None or plan.output_dtype is not torch.float32:
        return None
    # Quotient-read transport is proved only for the ordinary scalar native
    # epilogue. Do not silently extend RF's separate consumer/vector protocol.
    if any(
        isinstance(leaf, epi._QuotientAuxiliaryTensorLoadExpr)
        for chain in plan.chains
        for leaf in chain.auxiliary_tensor_loads
    ):
        return None
    identities = [
        index
        for index, chain in enumerate(plan.chains)
        if isinstance(chain.steps[0].expr, epi._PointwiseDagExpr)
        and isinstance(chain.steps[0].expr.nodes[-1], epi._CurrentTensorExpr)
    ]
    if len(identities) != 1:
        return None
    identity = identities[0]
    names: list[str] = []
    for store in plan.stores:
        source = store.args[0]
        if not (
            _is_call(source, _tracing_ops._host_tensor)
            and isinstance(source, torch.fx.Node)
            and len(source.args) == 1
            and isinstance(source.args[0], str)
        ):
            return None
        names.append(source.args[0])
    outputs = tuple(names)
    v_store = plan.stores[identity]
    value = _strip_new_vars(v_store.args[2], matched)
    convert = torch.ops.prims.convert_element_type.default
    while (
        _is_call(value, convert)
        and isinstance(value, torch.fx.Node)
        and value.args[1] is torch.float32
    ):
        value = _strip_new_vars(value.args[0], matched)
    accumulator_value = accumulator.meta.get("val")
    v_tensor = _tensor(v_store.args[0])
    if (
        value is not accumulator
        or not isinstance(accumulator_value, torch.Tensor)
        or accumulator_value.dtype is not torch.float32
        or v_tensor is None
        or v_tensor.dtype is not torch.float32
        # The producer template stores a dense row-major (n, 1) output.
        or _shape(env, v_tensor) != (m, n)
        or _strides(env, v_tensor) != (n, 1)
    ):
        return None
    # V dominates every accumulator use of the other stores: below the phi,
    # only the loop result, the MMA and the carried initial value lead back
    # to the accumulator.
    upstream = {
        loop,
        mm,
        initial,
        *(n for n in root.graph.nodes if _getitem(n, loop, 0, set())),
    }
    for index, store in enumerate(plan.stores):
        if index == identity:
            continue
        if _reaches(store.args[2], upstream, {accumulator}):
            return None
    # No floating multiply may feed an add/sub anywhere downstream of V: no
    # contraction can differ between the fused epilogue and the consumer.
    region: set[Node] = set()
    stack = [store.args[2] for i, store in enumerate(plan.stores) if i != identity]
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in region or node is accumulator:
            continue
        region.add(node)
        stack.extend(node.all_input_nodes)
    multiplies = {torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar}
    additions = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.rsub.Tensor,
        torch.ops.aten.rsub.Scalar,
    }
    fused = {
        torch.ops.aten.addcmul.default,
        torch.ops.aten.addcdiv.default,
        torch.ops.aten.lerp.Scalar,
        torch.ops.aten.lerp.Tensor,
    }
    convert = torch.ops.prims.convert_element_type.default
    passthrough = {
        torch.ops.aten.neg.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.permute.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        _tracing_ops._new_var,
    }

    def value_source(operand: object) -> object:
        """Look through value-preserving unary steps (a contraction can see
        ``fadd(fneg(fmul))`` or a same-dtype convert just as a direct edge)."""
        seen: set[Node] = set()
        while isinstance(operand, torch.fx.Node) and operand not in seen:
            seen.add(operand)
            if operand.target in passthrough and operand.args:
                operand = operand.args[0]
                continue
            if operand.target is convert and len(operand.args) == 2:
                source = operand.args[0]
                before = (
                    source.meta.get("val")
                    if isinstance(source, torch.fx.Node)
                    else None
                )
                after = operand.meta.get("val")
                if (
                    isinstance(before, torch.Tensor)
                    and isinstance(after, torch.Tensor)
                    and before.dtype is after.dtype
                ):
                    operand = source
                    continue
            break
        return operand

    for node in region:
        if node.target in fused:
            return None
        if node.target in additions and any(
            value_source(arg) in region
            and getattr(value_source(arg), "target", None) in multiplies
            for arg in node.args
        ):
            return None
    output_tensors = {id(_tensor(store.args[0])) for store in plan.stores}
    for node in calls:
        if (
            node.target is memory_ops.load
            and id(_tensor(node.args[0])) in output_tensors
        ):
            return None
    program = _consumer_program(
        host, cast("ast.For", grid[0]), outputs, outputs[identity], m, n
    )
    if program is None:
        return None
    source, consumer_params, statements, v_name = program
    # Source/graph identity: V is the loop-carried accumulator node, and every
    # consumer local names an original root-graph node.
    # (``V = acc`` binds the phi under the accumulator's own name.)
    root_names = {node.name for node in root.graph.nodes}
    if v_name not in (accumulator.name, *_alias_names(grid[0])) or any(
        line.split(" = ", 1)[0] not in root_names
        for line in statements[1:]
        if not line.split(" = ", 1)[0].endswith("]")
    ):
        return None
    return ResultFissionProof(
        root_graph_id=ir.root_ids[0],
        identity_index=identity,
        v_output=outputs[identity],
        outputs=outputs,
        lhs=a,
        rhs=b,
        v_tensor=v_tensor,
        lhs_expression=a_expression,
        rhs_expression=b_expression,
        lhs_strides=cast("tuple[int, int]", a_strides),
        rhs_strides=cast("tuple[int, int]", b_strides),
        operand_dtype=a.dtype,
        m=m,
        n=n,
        k=k,
        m_block_id=m_axis,
        n_block_id=n_axis,
        k_block_id=k_axis,
        consumer_source=source,
        consumer_params=consumer_params,
        consumer_statements=statements,
        v_name=v_name,
    )
