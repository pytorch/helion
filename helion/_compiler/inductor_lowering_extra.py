from __future__ import annotations

import contextlib
import functools
from typing import Callable
from typing import Generator
from typing import cast

import sympy
import torch
import torch.fx
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import MultiOutput
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import Pointwise
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import register_lowering
from torch._inductor.lowering import to_dtype
from torch._inductor.virtualized import V

from ._inductor.template_buffer import HelionTemplateBuffer
from helion._compiler._dynamo.higher_order_ops import get_helion_kernel
from helion._compiler._dynamo.higher_order_ops import (
    helion_kernel_wrapper_mutation as _helion_hop,
)

inductor_lowering_dispatch: dict[
    Callable[..., object] | str, Callable[..., object]
] = {}


def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
        if isinstance(x, TensorBox) and (original_dtype := x.get_dtype()) in (
            torch.float16,
            torch.bfloat16,
        ):
            x_fp32 = to_dtype(x, torch.float32)
            result_fp32 = original_op(x_fp32)
            assert isinstance(result_fp32, TensorBox)
            return to_dtype(result_fp32, original_dtype)
        return original_op(x)

    return fp32_fallback_lowering


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.log.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.exp.default,
]

# Register fp32 fallback lowerings for ops that don't support fp16/bfloat16
for op in FP32_FALLBACK_OPS_UNARY:
    inductor_lowering_dispatch[op] = create_fp16_to_fp32_unary_fallback_lowering(
        original_lowerings[op]
    )


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, None, None]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    # pyrefly: ignore [implicit-import]
    original_lowerings = torch._inductor.lowering.lowerings.copy()
    try:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)
        yield
    finally:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings = original_lowerings


# pyrefly: ignore [implicit-import]
register_inductor_lowering = torch._inductor.lowering.register_lowering


def var_mean_helper_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    from torch._inductor.lowering import var_mean_sum_
    from torch._prims_common import get_computation_dtype

    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)

    x = to_dtype(x, compute_dtype, copy=False)

    kwargs = {
        "x": x,
        "axis": axis,
        "correction": correction,
        "keepdim": keepdim,
        "return_mean": return_mean,
    }
    # TODO(yf225): support Welford reduction in Helion, then switch back to use Inductor `var_mean_helper_()`.
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(o, out_dtype, copy=False) for o in output)
    # pyrefly: ignore [bad-return]
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )


# View operations that create aliases (don't copy data)
_VIEW_OP_NAMES = {
    "permute",
    "transpose",
    "view",
    "reshape",
    "squeeze",
    "unsqueeze",
    "expand",
    "t",
    "slice",
    "narrow",
    "select",
    "as_strided",
    "unfold",
    "flatten",
}


def _get_op_name(target: object) -> str:
    """Extract the base operation name from a target."""
    if hasattr(target, "__name__"):
        # Handle cases like 'permute.default' -> 'permute'
        return target.__name__.split(".")[0]
    elif hasattr(target, "name"):
        # Handle torch._ops.OpOverload like 'aten::permute' -> 'permute'
        return target.name().split("::")[-1].split(".")[0]
    elif isinstance(target, str):
        return target
    return ""


def _is_clone_op(fx_node: torch.fx.Node) -> bool:
    """Check if an FX node is a clone operation."""
    if fx_node.op == "call_method" and fx_node.target == "clone":
        return True
    if fx_node.op == "call_function":
        op_name = _get_op_name(fx_node.target)
        if op_name == "clone":
            return True
    return False


def _trace_to_base(fx_node: torch.fx.Node) -> tuple[torch.fx.Node, bool]:
    """Trace back through view and clone operations to find the base tensor.

    Returns:
        (base_tensor, has_clone): The ultimate base tensor and whether a clone
        operation was encountered in the path.
    """
    current = fx_node
    has_clone = False

    while True:
        # Check for clone operation (call_method or call_function)
        if _is_clone_op(current):
            has_clone = True
            if len(current.args) > 0 and isinstance(current.args[0], torch.fx.Node):
                current = current.args[0]
                continue
            break

        # Check for view operations (call_function only)
        if current.op == "call_function":
            op_name = _get_op_name(current.target)
            if op_name in _VIEW_OP_NAMES:
                if len(current.args) > 0 and isinstance(current.args[0], torch.fx.Node):
                    current = current.args[0]
                    continue
            break

        # Check for getattr (like x.T which becomes getattr(x, "T"))
        if current.op == "call_function":
            target_name = _get_op_name(current.target)
            # getattr is used for .T accessor
            if target_name == "getattr" and len(current.args) >= 2:
                attr_name = current.args[1]
                if attr_name in ("T", "mT", "mH", "H"):
                    if isinstance(current.args[0], torch.fx.Node):
                        current = current.args[0]
                        continue
            break

        break

    return current, has_clone


def _output_depends_on(
    output_nodes: set[torch.fx.Node],
    target: torch.fx.Node,
    graph: torch.fx.Graph,
) -> bool:
    """Check if any output transitively depends on target."""
    # Build reverse dependency map: node -> nodes that use it
    # Then check if target can reach any output

    # Use forward search from target to see if we reach any output
    visited: set[torch.fx.Node] = set()
    queue = [target]

    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)

        if node in output_nodes:
            return True

        # Add all users of this node to the queue
        for user in node.users:
            if user not in visited:
                queue.append(user)

    return False


def _clone_mutated_graph_output_inputs(
    mutated_names: set[str],
    realized: dict[str, IRNode],
    realize_fn: Callable[[TensorBox], IRNode],
    cloned_inputs: set[str],
) -> None:
    """Clone realized inputs that are mutated AND whose base is used by outputs.

    When a user writes clone-then-mutate patterns like:
        x_clone = x.clone()  # or x.T.clone(), x.view().clone()
        result = kernel(x_clone)  # mutates x_clone
        return result, x  # or x + 1, x.sum(), etc.

    The clone may be eliminated by AOT autograd optimization. This function
    restores clone semantics by using the cloned_inputs info passed from Dynamo:

    1. cloned_inputs contains parameters that were derived from a clone operation
       at the Dynamo level (before AOT optimization eliminates the clone).
    2. For inputs in cloned_inputs that are also mutated, we check if the base
       tensor (what was cloned from) is used by graph outputs.
    3. If both conditions are true, we clone the buffer to prevent mutation
       from affecting the original base tensor.

    When an input is NOT in cloned_inputs (user intentionally mutates directly),
    we don't clone because the mutation should be visible in the original tensor.
    """
    current_node: torch.fx.Node | None = getattr(V.graph, "current_node", None)
    if current_node is None:
        return
    fx_tensor_args = current_node.kwargs.get("tensor_args", {})
    if not fx_tensor_args:
        return

    # Collect FX nodes that are direct graph outputs
    output_fx_nodes: set[torch.fx.Node] = set()
    graph = V.graph.module.graph
    for fx_node in graph.nodes:
        if fx_node.op == "output":

            def _collect(arg: object) -> None:
                if isinstance(arg, torch.fx.Node):
                    output_fx_nodes.add(arg)
                elif isinstance(arg, (tuple, list)):
                    for a in arg:
                        _collect(a)

            _collect(fx_node.args)
            break

    if not output_fx_nodes:
        return

    # Track cloned FX nodes so multiple args pointing to the same node
    # (identical aliased inputs) share a single clone buffer.
    cloned: dict[torch.fx.Node, IRNode] = {}
    for name in mutated_names:
        fx_arg_node = fx_tensor_args.get(name)
        if not isinstance(fx_arg_node, torch.fx.Node) or name not in realized:
            continue

        # Check if this input was cloned at the Dynamo level.
        # This info survives AOT autograd optimization.
        was_cloned_at_dynamo = name in cloned_inputs

        # Trace back through views to find the base tensor.
        # Note: At the Inductor level, the clone is already eliminated,
        # so we only trace through views here.
        base_tensor, _ = _trace_to_base(fx_arg_node)

        # Check if any output depends on the base tensor (directly or transitively)
        base_used_by_output = _output_depends_on(output_fx_nodes, base_tensor, graph)

        # Clone if: the input was originally cloned AND the base is used by outputs
        should_clone = was_cloned_at_dynamo and base_used_by_output

        if should_clone:
            if fx_arg_node in cloned:
                realized[name] = cloned[fx_arg_node]
            else:
                orig = realized[name]
                clone_tb = Pointwise.create(
                    device=orig.get_device(),
                    dtype=orig.get_dtype(),
                    inner_fn=orig.make_loader(),
                    ranges=list(orig.get_size()),
                )
                assert isinstance(clone_tb, TensorBox)
                clone_ir = realize_fn(clone_tb)
                realized[name] = clone_ir
                cloned[fx_arg_node] = clone_ir


@register_lowering(_helion_hop, type_promotion_kind=None)
def lower_helion_kernel(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
) -> tuple[TensorBox | int | float | None, ...]:
    """Lower a Helion kernel call to HelionTemplateBuffer."""
    kernel = get_helion_kernel(kernel_idx)

    # Extract output spec components
    num_outputs = cast("int", output_spec.get("num_outputs", 0))
    specs = cast("list[dict[str, object] | None]", output_spec.get("output_specs", []))
    aliases = cast("list[str | None]", output_spec.get("output_aliases", []))
    direct_flags = cast("list[bool]", output_spec.get("output_alias_is_direct", []))

    # Realize inputs: convert TensorBox to buffer/ReinterpretView
    def realize(tb: TensorBox) -> IRNode:
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(getattr(result, "layout", None), FlexibleLayout):
            result.freeze_layout()
        return result

    realized = {
        n: realize(tb) for n, tb in tensor_args.items() if isinstance(tb, TensorBox)
    }

    # Clone mutated inputs that are also graph outputs.  After AOT autograd
    # eliminates a user's clone, the mutated input and graph output share
    # the same buffer.  By giving the HOP a fresh copy, the original buffer
    # (still referenced via V.graph.env) stays unmutated for the graph output.
    mutated_names = set(cast("list[str]", output_spec.get("mutated_inputs", [])))
    # cloned_inputs contains parameter names that were derived from a clone
    # operation at the Dynamo level (before AOT optimization eliminates them)
    cloned_inputs = set(cast("list[str]", output_spec.get("cloned_inputs", [])))
    if mutated_names:
        _clone_mutated_graph_output_inputs(mutated_names, realized, realize, cloned_inputs)

    # Build ordered arg_names and inputs lists from realized
    arg_names = list(realized.keys())
    inputs = list(realized.values())

    # Build fake tensors for kernel binding
    fake_tensors = []
    for name, param in kernel.signature.parameters.items():
        if name in realized:
            inp = realized[name]
            size = [
                int(s) if isinstance(s, (int, sympy.Integer)) else 64
                for s in inp.get_size()
            ]
            stride = [
                int(s) if isinstance(s, (int, sympy.Integer)) else 1
                for s in inp.get_stride()
            ]
            fake_tensors.append(
                torch.empty_strided(
                    size, stride, dtype=inp.get_dtype(), device=inp.get_device()
                )
            )
        elif name in constant_args:
            fake_tensors.append(constant_args[name])
        elif param.default is not param.empty:
            fake_tensors.append(param.default)
    bound = kernel.bind(tuple(fake_tensors))

    def make_layout(idx: int) -> FixedLayout | None:
        """Create FixedLayout from output spec at given index, or None for scalars."""
        spec = specs[idx] if idx < len(specs) else None
        if spec is None or "shape" not in spec:
            return None
        return FixedLayout(
            device=torch.device(  # pyrefly: ignore[no-matching-overload]
                spec["device"]
            ),
            dtype=spec["dtype"],  # pyrefly: ignore[bad-argument-type]
            size=cast("list[sympy.Expr]", spec["shape"]),
        )

    # Determine buffer layout
    if num_outputs == 1:
        layout = make_layout(0)
        if layout is None:
            raise ValueError("Single-output kernel must return a tensor, not a scalar")
    else:
        device = next(
            (
                torch.device(s["device"])  # pyrefly: ignore[no-matching-overload]
                for s in specs
                if s and "device" in s
            ),
            torch.device("cuda"),
        )
        layout = MultiOutputLayout(device=device)

    # Build HelionTemplateBuffer
    mutated = [
        inputs[arg_names.index(n)]
        for n in cast("list[str]", output_spec.get("mutated_inputs", []))
        if n in arg_names
    ]
    buf = HelionTemplateBuffer(
        layout=layout,
        inputs=inputs,
        kernel=kernel,
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_arg_names=arg_names,
        bound_kernel=bound,
        mutated_inputs=mutated or None,
        output_aliases=aliases,
        output_alias_is_direct=direct_flags,
        autotune_args=tuple(fake_tensors),
    )
    V.graph.no_fuse_buffer_names.add(buf.get_name())  # Disable fusion for now

    # Build output results
    results: list[TensorBox | int | float | None] = []
    multi_output_nodes: list[MultiOutput] = []

    for i in range(num_outputs):
        spec = specs[i]
        alias_name = aliases[i] if i < len(aliases) else None

        # Handle aliased outputs
        if alias_name and alias_name in arg_names:
            alias_inp = inputs[arg_names.index(alias_name)]
            if isinstance(alias_inp, IRNode):
                is_direct = i < len(direct_flags) and direct_flags[i]
                if is_direct:
                    results.append(TensorBox.create(alias_inp))
                    continue
                # Indirect alias: create ReinterpretView
                if spec and "stride" in spec:
                    alias_device = alias_inp.get_device()
                    assert alias_device is not None
                    view_layout = FixedLayout(
                        device=alias_device,
                        dtype=alias_inp.get_dtype(),
                        size=[
                            sympy.Integer(s)
                            for s in spec["shape"]  # pyrefly: ignore[not-iterable]
                        ],
                        stride=[
                            sympy.Integer(s)
                            for s in spec["stride"]  # pyrefly: ignore[not-iterable]
                        ],
                        offset=sympy.Integer(spec.get("storage_offset", 0)),
                    )
                    storage = (
                        alias_inp
                        if isinstance(alias_inp, StorageBox)
                        else StorageBox(alias_inp)
                    )
                    results.append(
                        TensorBox.create(
                            ReinterpretView(data=storage, layout=view_layout)
                        )
                    )
                    continue

        # Handle non-alias tensor outputs
        out_layout = make_layout(i)
        if out_layout is not None:
            if num_outputs == 1:
                results.append(TensorBox(StorageBox(buf)))
            else:
                mo = MultiOutput(layout=out_layout, input=buf, indices=[(tuple, i)])
                multi_output_nodes.append(mo)
                results.append(TensorBox.create(mo))
        else:
            # Scalar output
            # pyrefly: ignore[bad-argument-type]
            results.append(spec.get("scalar_value") if spec else None)

    if num_outputs > 1:
        buf.multi_output_nodes = multi_output_nodes
        if not multi_output_nodes:
            fallback = make_layout(0)
            if fallback:
                buf.layout = fallback

    return tuple(results)
