"""Helion kernel variable tracking for Dynamo."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from torch._dynamo.variables.base import VariableTracker

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class HelionKernelSideTable:
    """Side table for storing Helion Kernel objects during Dynamo tracing."""
    _kernels: list[Any] = []
    _kernel_ids: dict[int, int] = {}

    @classmethod
    def add_kernel(cls, kernel: Any) -> int:
        kid = id(kernel)
        if kid in cls._kernel_ids: return cls._kernel_ids[kid]
        cls._kernels.append(kernel); cls._kernel_ids[kid] = len(cls._kernels) - 1
        return cls._kernel_ids[kid]

    @classmethod
    def get_kernel(cls, idx: int) -> Any: return cls._kernels[idx]


helion_kernel_side_table = HelionKernelSideTable()


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""
    kernel: Any
    kernel_idx: int

    def __init__(self, kernel: Any, kernel_idx: int | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.kernel_idx = kernel_idx if kernel_idx is not None else helion_kernel_side_table.add_kernel(kernel)

    def call_function(self, tx: "InstructionTranslator", args: Sequence[VariableTracker],
                      kwargs: dict[str, VariableTracker]) -> VariableTracker:
        from torch._dynamo import variables
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.dicts import ConstDictVariable
        from helion._dynamo.higher_order_ops import helion_kernel_wrapper_mutation

        param_names = list(self.kernel.signature.parameters.keys())
        combined = {variables.ConstantVariable.create(param_names[i]): arg for i, arg in enumerate(args) if i < len(param_names)}
        combined.update({variables.ConstantVariable.create(k): v for k, v in kwargs.items()})

        constant_args, tensor_args = {}, {}
        for k, v in combined.items():
            ks = k.as_python_constant()
            if isinstance(v, VariableTracker) and v.is_python_constant(): constant_args[ks] = v.as_python_constant()
            else: tensor_args[k] = v

        hop_proxy = tx.output.create_proxy("call_function", helion_kernel_wrapper_mutation, (),
            {"kernel_idx": self.kernel_idx, "constant_args": constant_args,
             "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(), "output_spec": self._infer_output_spec(args)})
        result = wrap_fx_proxy(tx, hop_proxy)

        num_outputs = self._get_num_outputs()
        if num_outputs > 1:
            from torch._dynamo.variables.lists import TupleVariable
            return TupleVariable([result.call_method(tx, "__getitem__", [variables.ConstantVariable.create(i)], {}) for i in range(num_outputs)])
        return result.call_method(tx, "__getitem__", [variables.ConstantVariable.create(0)], {})

    def _get_num_outputs(self) -> int:
        import inspect, typing, torch
        ann = self.kernel.signature.return_annotation
        if ann is inspect.Parameter.empty: return 1
        if isinstance(ann, str):
            try: ann = typing.get_type_hints(self.kernel.fn, globalns=getattr(self.kernel.fn, '__globals__', {}),
                                             localns={'torch': torch, 'Tensor': torch.Tensor}).get('return', ann)
            except Exception: return 1
        if typing.get_origin(ann) is tuple:
            args = typing.get_args(ann)
            if args and not (len(args) == 2 and args[1] is ...): return len(args)
        return 1

    def _infer_output_spec(self, args: Sequence[VariableTracker]) -> dict[str, Any]:
        import torch; from helion._compiler.type_propagation import TensorType
        fake_args, num_outputs, num_tiled = [self._get_example_value(a) for a in args], self._get_num_outputs(), None
        try:
            bound = self.kernel.bind(tuple(fake_args))
            if bound.host_function:
                num_tiled = sum(1 for bs in bound.env.block_sizes if not bs.reduction)
                input_names = set(self.kernel.signature.parameters.keys())
                for name, vtype in (bound.host_function.local_types or {}).items():
                    if isinstance(vtype, TensorType) and name not in input_names:
                        t = vtype.fake_value
                        return {"shape": list(t.shape), "dtype": t.dtype, "device": str(t.device), "num_tiled_dims": num_tiled or len(t.shape), "num_outputs": num_outputs}
        except Exception: pass
        for arg in fake_args:
            if isinstance(arg, torch.Tensor):
                return {"shape": list(arg.shape), "dtype": arg.dtype, "device": str(arg.device), "num_tiled_dims": num_tiled or len(arg.shape), "num_outputs": num_outputs}
        return {"num_outputs": num_outputs}

    def _get_example_value(self, arg: VariableTracker) -> Any:
        try: return arg.as_python_constant()
        except Exception: pass
        try: return arg.as_proxy().node.meta.get("example_value")
        except Exception: return None
