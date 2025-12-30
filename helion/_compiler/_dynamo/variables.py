"""Helion kernel variable tracking for Dynamo."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Sequence

from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import TupleVariable

from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


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
        if num_outputs > 1:
            return TupleVariable(
                [
                    result.call_method(
                        tx, "__getitem__", [variables.ConstantVariable.create(i)], {}
                    )
                    for i in range(num_outputs)
                ]
            )
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(0)], {}
        )

    def _infer_output_spec(self, args: Sequence[VariableTracker]) -> dict[str, Any]:
        from helion._compiler.type_propagation import TensorType

        fake_args = [self._get_example_value(a) for a in args]
        # Default to -1, will be updated from kernel AST analysis
        num_outputs = -1
        num_tiled = None
        output_specs: list[dict[str, Any] | None] = []

        bound = self._kernel.bind(tuple(fake_args))
        if bound.host_function:
            num_tiled = sum(1 for bs in bound.env.block_sizes if not bs.reduction)
            input_names = set(self._kernel.signature.parameters.keys())
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

            return {
                "num_outputs": num_outputs,
                "num_tiled_dims": num_tiled,
                "output_specs": output_specs,
            }

        raise AssertionError("kernel.bind() succeeded but host_function is None")

    def _get_example_value(self, arg: VariableTracker) -> Any:  # noqa: ANN401
        if arg.is_python_constant():
            return arg.as_python_constant()
        proxy = arg.as_proxy()
        if proxy is not None and hasattr(proxy, "node"):
            return proxy.node.meta.get("example_value")
        return None
