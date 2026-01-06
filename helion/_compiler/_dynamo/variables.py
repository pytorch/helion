from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

import torch
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import TupleVariable

from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.host_function import HostFunction


def _find_mutated_inputs(
    host_function: HostFunction,
    device_ir: DeviceIR,
    input_names: set[str],
) -> list[str]:
    """Find input parameters that are mutated (store/atomic ops) in the kernel."""
    from helion._compiler.type_propagation import TensorType
    from helion.language import atomic_ops
    from helion.language import memory_ops
    from helion.language._tracing_ops import _host_tensor
    from helion.language.inline_triton_ops import inline_triton
    from helion.language.inline_triton_ops import triton_kernel
    from helion.language.signal_wait import signal
    from helion.language.signal_wait import wait

    mutating_ops = {memory_ops.store, signal} | {
        getattr(atomic_ops, n)
        for n in atomic_ops.__all__
        if callable(getattr(atomic_ops, n, None))
    }
    unknown_mutation_ops = {inline_triton, triton_kernel}
    mutated: set[str] = set()
    local_types = host_function.local_types or {}

    def trace_to_host_tensor(node: torch.fx.Node) -> str | None:
        visited: set[torch.fx.Node] = set()

        def _trace(n: torch.fx.Node) -> str | None:
            if n in visited:
                return None
            visited.add(n)
            if n.op == "call_function" and n.target is _host_tensor:
                return n.args[0] if n.args and isinstance(n.args[0], str) else None
            for arg in n.args:
                if isinstance(arg, torch.fx.Node) and (r := _trace(arg)):
                    return r
            return None

        return _trace(node)

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in unknown_mutation_ops
                and input_names
            ):
                return list(input_names)
            if node.op == "call_function" and node.target is wait and input_names:
                update = (
                    node.args[3] if len(node.args) > 3 else node.kwargs.get("update")
                )
                if update is None:
                    continue
                target_node = node.args[0] if node.args else None
                if isinstance(target_node, torch.fx.Node):
                    param_name = trace_to_host_tensor(target_node)
                    if param_name is None:
                        return list(input_names)
                    if param_name in input_names:
                        mutated.add(param_name)
                    elif isinstance(lt := local_types.get(param_name), TensorType):
                        for inp in input_names:
                            if isinstance(it := local_types.get(inp), TensorType):
                                # pyrefly: ignore[missing-attribute]
                                if torch._C._is_alias_of(lt.fake_value, it.fake_value):
                                    mutated.add(inp)
                continue
            if node.op == "call_function" and node.target in mutating_ops:
                if node.args and isinstance(node.args[0], torch.fx.Node):
                    param_name = trace_to_host_tensor(node.args[0])
                    if param_name is None:
                        continue
                    if param_name in input_names:
                        mutated.add(param_name)
                    elif isinstance(lt := local_types.get(param_name), TensorType):
                        for inp in input_names:
                            if isinstance(it := local_types.get(inp), TensorType):
                                # pyrefly: ignore[missing-attribute]
                                if torch._C._is_alias_of(lt.fake_value, it.fake_value):
                                    mutated.add(inp)
    return list(mutated)


def _expand_mutated_inputs_with_aliases(
    mutated_inputs: Sequence[str],
    param_names: Sequence[str],
    example_args: Sequence[object],
) -> list[str]:
    """Expand mutated inputs to include any aliased tensor arguments."""
    if not mutated_inputs:
        return []
    tensors = {
        n: a
        for n, a in zip(param_names, example_args, strict=True)
        if isinstance(a, torch.Tensor)
    }
    if len(tensors) < 2:
        return list(mutated_inputs)
    mutated = set(mutated_inputs)
    for name in list(mutated):
        if name not in tensors:
            continue
        for other, t in tensors.items():
            if other != name and (
                # pyrefly: ignore[missing-attribute]
                t is tensors[name] or torch._C._is_alias_of(t, tensors[name])
            ):
                mutated.add(other)
    return [n for n in param_names if n in mutated]


class HelionKernelSideTable:
    """Side table for storing Helion Kernel objects during Dynamo tracing."""

    _kernels: ClassVar[list[Any]] = []
    _kernel_ids: ClassVar[dict[int, int]] = {}

    @classmethod
    def add_kernel(cls, kernel: Any) -> int:  # noqa: ANN401
        kid = id(kernel)
        if kid in cls._kernel_ids:
            return cls._kernel_ids[kid]
        cls._kernels.append(kernel)
        cls._kernel_ids[kid] = len(cls._kernels) - 1
        return cls._kernel_ids[kid]

    @classmethod
    def get_kernel(cls, idx: int) -> Any:  # noqa: ANN401
        return cls._kernels[idx]


helion_kernel_side_table = HelionKernelSideTable()


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    _kernel: Any
    _kernel_idx: int

    def __init__(
        self,
        kernel: Any,  # noqa: ANN401
        kernel_idx: int | None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self._kernel = kernel
        self._kernel_idx = (
            kernel_idx
            if kernel_idx is not None
            else helion_kernel_side_table.add_kernel(kernel)
        )

    def call_function(
        self,
        tx: InstructionTranslator,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        param_names = list(self._kernel.signature.parameters.keys())
        combined = {
            variables.ConstantVariable.create(param_names[i]): arg
            for i, arg in enumerate(args)
            if i < len(param_names)
        }
        combined.update(
            {variables.ConstantVariable.create(k): v for k, v in kwargs.items()}
        )
        param_vars = {k.as_python_constant(): v for k, v in combined.items()}

        constant_args, tensor_args = {}, {}
        for k, v in combined.items():
            ks = k.as_python_constant()
            if isinstance(v, VariableTracker) and v.is_python_constant():
                constant_args[ks] = v.as_python_constant()
            else:
                tensor_args[k] = v

        output_spec = self._infer_output_spec(args)
        hop_proxy = tx.output.create_proxy(
            "call_function",
            helion_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": self._kernel_idx,
                "constant_args": constant_args,
                "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(),
                "output_spec": output_spec,
            },
        )
        result = wrap_fx_proxy(tx, hop_proxy)

        # Use num_outputs from output_spec (determined by kernel AST analysis)
        num_outputs = output_spec["num_outputs"]
        output_aliases = output_spec.get("output_aliases", [])
        output_alias_is_direct = output_spec.get("output_alias_is_direct", [])
        if num_outputs > 1:
            return TupleVariable(
                [
                    (
                        param_vars[output_aliases[i]]
                        if (
                            i < len(output_alias_is_direct)
                            and output_alias_is_direct[i]
                            and i < len(output_aliases)
                            and output_aliases[i] in param_vars
                        )
                        else result.call_method(
                            tx,
                            "__getitem__",
                            [variables.ConstantVariable.create(i)],
                            {},
                        )
                    )
                    for i in range(num_outputs)
                ]
            )
        if (
            output_alias_is_direct
            and output_alias_is_direct[0]
            and output_aliases
            and output_aliases[0] in param_vars
        ):
            return param_vars[output_aliases[0]]
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(0)], {}
        )

    def _infer_output_spec(self, args: Sequence[VariableTracker]) -> dict[str, Any]:
        from helion._compiler.type_propagation import TensorType

        fake_args = [self._get_example_value(a) for a in args]
        param_names = list(self._kernel.signature.parameters.keys())
        # Default to -1, will be updated from kernel AST analysis
        num_outputs = -1
        num_tiled = None
        output_specs: list[dict[str, Any] | None] = []

        bound = self._kernel.bind(tuple(fake_args))
        if bound.host_function:
            num_tiled = sum(1 for bs in bound.env.block_sizes if not bs.reduction)
            input_names = set(param_names)
            local_types = bound.host_function.local_types or {}

            # Parse return statement to get output info
            # Each element is either: variable name (str), constant value, or None
            return_elts: list[tuple[str | None, Any]] = []  # (name, constant_value)
            for stmt in reversed(bound.host_function.body):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    if isinstance(stmt.value, ast.Tuple):
                        num_outputs = len(stmt.value.elts)
                        for elt in stmt.value.elts:
                            if isinstance(elt, ast.Name):
                                return_elts.append((elt.id, None))
                            elif isinstance(elt, ast.Constant):
                                return_elts.append((None, elt.value))
                            else:
                                return_elts.append((None, None))
                    else:
                        num_outputs = 1
                        if isinstance(stmt.value, ast.Name):
                            return_elts.append((stmt.value.id, None))
                        elif isinstance(stmt.value, ast.Constant):
                            return_elts.append((None, stmt.value.value))
                        else:
                            return_elts.append((None, None))
                    break

            # Build per-output specs by looking up each return value's type
            for name, const_value in return_elts:
                if name is not None and name in local_types:
                    vtype = local_types[name]
                    if isinstance(vtype, TensorType):
                        t = vtype.fake_value
                        output_specs.append(
                            {
                                "shape": list(t.shape),
                                "dtype": t.dtype,
                                "device": str(t.device),
                            }
                        )
                    else:
                        # Non-tensor variable - try to get its value
                        output_specs.append({"scalar_value": None})
                elif const_value is not None:
                    # Constant value in return statement
                    output_specs.append({"scalar_value": const_value})
                else:
                    output_specs.append(None)

            # Track outputs that alias inputs (direct return or via local alias)
            output_aliases: list[str | None] = []
            output_alias_is_direct: list[bool] = []
            for name, _ in return_elts:
                alias_name: str | None = None
                is_direct = False
                if name is not None:
                    if name in input_names:
                        alias_name = name
                        is_direct = True
                    elif isinstance(lt := local_types.get(name), TensorType):
                        for inp in input_names:
                            if isinstance(it := local_types.get(inp), TensorType):
                                # pyrefly: ignore[missing-attribute]
                                if torch._C._is_alias_of(lt.fake_value, it.fake_value):
                                    alias_name = inp
                                    if list(lt.fake_value.shape) == list(
                                        it.fake_value.shape
                                    ) and list(lt.fake_value.stride()) == list(
                                        it.fake_value.stride()
                                    ):
                                        is_direct = True
                                    break
                output_aliases.append(alias_name)
                output_alias_is_direct.append(is_direct)

            # Detect mutated inputs by analyzing the device IR
            mutated_inputs = _find_mutated_inputs(
                bound.host_function,
                bound.host_function.device_ir,
                input_names,
            )
            mutated_inputs = _expand_mutated_inputs_with_aliases(
                mutated_inputs,
                param_names,
                fake_args,
            )

            return {
                "num_outputs": num_outputs,
                "num_tiled_dims": num_tiled,
                "output_specs": output_specs,
                "mutated_inputs": mutated_inputs,
                "output_aliases": output_aliases,
                "output_alias_is_direct": output_alias_is_direct,
            }

        raise AssertionError("kernel.bind() succeeded but host_function is None")

    def _get_example_value(self, arg: VariableTracker) -> Any:  # noqa: ANN401
        if arg.is_python_constant():
            return arg.as_python_constant()
        proxy = arg.as_proxy()
        if proxy is not None and hasattr(proxy, "node"):
            return proxy.node.meta.get("example_value")
        return None
