"""HelionTemplate - IR node for Helion kernels in Inductor."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import sympy

from torch._inductor.ir import (
    Layout,
    MultiOutput,
    MultiOutputLayout,
    TritonTemplateBuffer,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .context import EpilogueSpec, PrologueSpec

if TYPE_CHECKING:
    from torch._inductor.ir import IRNode


class HelionTemplate(TritonTemplateBuffer):
    """Helion kernel IR node with fusion support."""

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence["IRNode"],
        kernel: Any,
        kernel_idx: int,
        constant_args: dict[str, Any],
        tensor_arg_names: list[str],
        bound_kernel: Any,
    ) -> None:
        self.helion_kernel = kernel
        self.kernel_idx = kernel_idx
        self.constant_args_dict = constant_args
        self.tensor_arg_names = tensor_arg_names
        self.bound_kernel = bound_kernel

        # Layout info
        self.output_ranges = list(layout.size) if hasattr(layout, "size") else []

        # Fusion specs
        self.epilogue_specs: dict[str, list[EpilogueSpec]] = {}
        self.prologue_specs: dict[str, list[PrologueSpec]] = {}
        self._uses_atomics: bool | None = None

        # Codegen state
        self.epilogue_closures: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs: set[str] = set()
        self.prologue_fused_inputs_preserve_zero: set[str] = set()
        self.removed_buffers: set[str] = set()
        self.inplaced_to_remove: set[str] = set()
        self.named_input_nodes: dict[str, Any] = {
            name: inputs[i]
            for i, name in enumerate(tensor_arg_names)
            if i < len(inputs)
        }
        self._triton_code: str | None = None
        self.cse = _HelionCSE()
        self._subgraph_bodies: dict[str, list] = {}

        # Build allowed prologue inputs
        from .helpers import safe_get_name
        allowed_prologue = OrderedSet(
            name for inp in inputs if (name := safe_get_name(inp)) is not None
        )

        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=self._make_render,
            mutated_inputs=None,
            allowed_prologue_inps=allowed_prologue,
        )

        # If kernel uses atomics, mark as non-fusable
        if self.uses_atomics():
            V.graph.no_fuse_buffer_names.add(self.get_name())

    def _make_render(
        self, template_buffer: "HelionTemplate", hint_override: int | None = None
    ):
        from .codegen import HelionRender

        return self, HelionRender(self)

    @property
    def template_buffer(self) -> "HelionTemplate":
        return self

    def add_epilogue(self, output_name: str, spec: EpilogueSpec) -> None:
        self.epilogue_specs.setdefault(output_name, []).append(spec)

    def add_prologue(self, input_name: str, spec: PrologueSpec) -> None:
        self.prologue_specs.setdefault(input_name, []).append(spec)

    def prepare_epilogues(self, epilogue_nodes: list, template_name: str) -> None:
        if not epilogue_nodes or not self.supports_epilogue_fusion():
            return
        template_outputs = self.output_names | {template_name}
        by_output: dict[str, list] = {}
        for ep in epilogue_nodes:
            if not (hasattr(ep, "read_writes") and ep.read_writes):
                continue
            for d in ep.read_writes.reads:
                dep_name = getattr(d, "name", None)
                if dep_name in template_outputs:
                    by_output.setdefault(dep_name, []).append(ep)
                    break
        for acc_name, nodes in by_output.items():
            self.add_epilogue(acc_name, EpilogueSpec(epilogue_nodes=nodes, accumulator_name=acc_name))

    def prepare_prologues(self, buf_name_to_prologue: dict[str, list]) -> None:
        for input_name, buffer in self.named_input_nodes.items():
            buf_name = buffer.get_name()
            if prologue := buf_name_to_prologue.get(buf_name):
                self.add_prologue(input_name, PrologueSpec(prologue_nodes=list(prologue), input_name=buf_name))

    def uses_atomics(self) -> bool:
        if self._uses_atomics is not None:
            return self._uses_atomics
        if self.bound_kernel is None:
            self._uses_atomics = True
            return True
        try:
            from ..generate_ast import generate_ast
            from ..ast_extension import unparse
            from ...runtime.config import Config
            from .context import fusion_context
            config = self.bound_kernel._require_implicit_config()
            if not isinstance(config, Config):
                config = Config(**config)
            self.bound_kernel.env.config_spec.normalize(config)
            with self.bound_kernel.env:
                with fusion_context({}, {}) as _:
                    root = generate_ast(self.bound_kernel.host_function, config, emit_repro_caller=False)
                self._uses_atomics = "tl.atomic_" in unparse(root, output_origin_lines=False)
        except Exception:
            self._uses_atomics = True
        return self._uses_atomics

    def supports_epilogue_fusion(self) -> bool:
        return not self.uses_atomics()

    def get_layout(self) -> Layout:
        if isinstance(self.layout, MultiOutputLayout):
            mo_nodes = getattr(self, "multi_output_nodes", None)
            assert mo_nodes and hasattr(mo_nodes[0], "layout"), "MultiOutputLayout without multi_output_nodes"
            return mo_nodes[0].layout
        return super().get_layout()

    @property
    def is_multi_output(self) -> bool:
        return isinstance(self.layout, MultiOutputLayout)

    @property
    def output_names(self) -> set[str]:
        names = {self.get_name()}
        for out in getattr(self, "multi_output_nodes", []):
            if hasattr(out, "get_name"):
                names.add(out.get_name())
        return names

    def __enter__(self) -> "HelionTemplate":
        import contextlib
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.__enter__()
        self._exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, *args: Any) -> None:
        self._exit_stack.__exit__(*args)

    def set_current_node(self, node: Any) -> "_NodeContext":
        return _NodeContext(self, node)

    def set_subgraph_body(self, name: str) -> "_SubgraphContext":
        return _SubgraphContext(self, name)

    def get_store_output_count(self) -> int:
        return 1

    def _get_store_output_subgraph_name(self, index: int) -> str:
        return f"<STORE_OUTPUT_{index}>"

    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]) -> list[list[sympy.Expr]]:
        result, idx = [], 0
        for group in lengths:
            vars = [sympy.Symbol(f"idx{idx + i}") for i in range(len(group))]
            result.append(vars)
            idx += len(group)
        return result

    def can_fuse_epilogue(self, node1: Any, node2: Any, why: Callable[[str], None]) -> bool:
        output_names = self.output_names
        if len(output_names) <= 1 or not (hasattr(node2, "read_writes") and node2.read_writes):
            return True
        output_reads = {getattr(d, "name", None) for d in node2.read_writes.reads} & output_names
        if len(output_reads) > 1:
            why("epilogue reads from multiple outputs (circular dependency)")
            return False
        return True

    def can_fuse_multi_output(self, node2: Any) -> bool:
        return (isinstance(self.layout, MultiOutputLayout) and isinstance(node2.node, MultiOutput)
                and len(node2.node.inputs) == 1 and node2.node.inputs[0].get_name() == self.name)

    def separate_fusable_prologues(self, prologue_nodes: list) -> tuple[list, list]:
        """Separate prologues into fusable and non-fusable groups."""
        from .can_fuse import prepare_prologues_for_template
        template_reads = OrderedSet(inp.get_name() for inp in self.inputs if hasattr(inp, "get_name"))
        buf_name_to_prologue_group: dict[str, list] = {}
        prologue_group: list = []
        for prologue in prologue_nodes:
            names = prologue.get_buffer_names()
            prologue_group.append(prologue)
            if names & template_reads:
                assert len(names) == 1
                buf_name_to_prologue_group[next(iter(names))] = prologue_group
                prologue_group = []
        stray_prologues = prologue_group
        fusable_group, non_fusable = prepare_prologues_for_template(self, buf_name_to_prologue_group)
        fusable_prologues = [n for nodes in fusable_group.values() for n in nodes]
        non_fusable.extend(stray_prologues)
        return fusable_prologues, non_fusable

    def separate_fusable_epilogues(self, epilogue_nodes: list) -> tuple[list, list]:
        """Separate epilogues into fusable and non-fusable groups."""
        if not epilogue_nodes:
            return [], []
        if not self.supports_epilogue_fusion():
            return [], list(epilogue_nodes)
        from .can_fuse import prepare_epilogues_for_template
        fusable_by_output, non_fusable = prepare_epilogues_for_template(self, epilogue_nodes)
        fusable_epilogues = [n for nodes in fusable_by_output.values() for n in nodes]
        return fusable_epilogues, non_fusable

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:
        from .codegen import HelionCodegen
        HelionCodegen(self).emit_call(V.graph.wrapper_code, kernel_name)

    def emit_kernel(self, wrapper: Any, src_code: str, kernel_name: str,
                    node_schedule: list, kernel_path: str, get_kernel_metadata_fn: Any) -> bool:
        from .codegen import HelionCodegen
        return HelionCodegen(self).emit_to_wrapper(wrapper, src_code, kernel_name, node_schedule,
                                                    kernel_path, get_kernel_metadata_fn)

    def codegen_template(self, scheduling: Any, kernel: Any, template_node: Any, epilogue_nodes: list,
                         prologue_nodes: list, buf_name_to_prologue_group: dict,
                         prologue_preserves_zero_mask_fn: Any, render: Callable, only_gen_src_code: bool) -> Any:
        from .codegen import HelionCodegen
        return HelionCodegen(self).codegen_template(scheduling, kernel, template_node, epilogue_nodes,
            prologue_nodes, buf_name_to_prologue_group, prologue_preserves_zero_mask_fn, render, only_gen_src_code)


class _HelionCSE:
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
    def invalidate(self, names: OrderedSet) -> None:
        for name in names:
            self._cache.pop(name, None)


class _NodeContext:
    def __init__(self, kernel: HelionTemplate, node: Any):
        self.kernel, self.node = kernel, node
    def __enter__(self):
        self._prev = getattr(self.kernel, "_current_node", None)
        self.kernel._current_node = self.node
        return self
    def __exit__(self, *args):
        self.kernel._current_node = self._prev


class _SubgraphContext:
    def __init__(self, kernel: HelionTemplate, name: str):
        self.kernel, self.name = kernel, name
    def __enter__(self):
        self._prev = getattr(self.kernel, "_current_subgraph", None)
        self.kernel._current_subgraph = self.name
        return self
    def __exit__(self, *args):
        self.kernel._current_subgraph = self._prev


# --- Inductor Lowering ---

def _realize_input(tensor_box) -> Any:
    from torch._inductor.ir import Buffer, ComputedBuffer, InputBuffer, ReinterpretView, StorageBox, TensorBox, TemplateBuffer
    _BUFFER_TYPES = (ComputedBuffer, InputBuffer, ReinterpretView, TemplateBuffer, Buffer)
    if not isinstance(tensor_box, TensorBox):
        return tensor_box
    data = tensor_box.data
    if isinstance(data, StorageBox):
        if not isinstance(data.data, _BUFFER_TYPES):
            data.realize()
        return data.data
    if isinstance(data, _BUFFER_TYPES):
        return data
    tensor_box.realize()
    return tensor_box.data.data if isinstance(tensor_box.data, StorageBox) else tensor_box.data


def _get_helion_kernel(kernel_idx: int) -> Any:
    from helion._dynamo.higher_order_ops import get_helion_kernel
    return get_helion_kernel(kernel_idx)


def _try_bind_kernel(kernel, tensor_args, constant_args, arg_names, dtype, device) -> Any:
    import torch
    try:
        fake_tensors = []
        sig = kernel.signature.parameters
        for name in sig:
            if name in arg_names:
                tb = list(tensor_args.values())[arg_names.index(name)]
                size = [int(s) if isinstance(s, (int, sympy.Integer)) else 64 for s in tb.get_size()] if hasattr(tb, "get_size") else [1]
                fake_tensors.append(torch.empty(size, dtype=dtype, device=device))
            elif name in constant_args:
                fake_tensors.append(constant_args[name])
            elif sig[name].default is not sig[name].empty:
                fake_tensors.append(sig[name].default)
        return kernel.bind(tuple(fake_tensors))
    except Exception:
        return None


from helion._dynamo.higher_order_ops import helion_kernel_wrapper_mutation as _helion_hop
from torch._inductor.lowering import register_lowering


@register_lowering(_helion_hop, type_promotion_kind=None)
def lower_helion_kernel(*, kernel_idx, constant_args, tensor_args, output_spec):
    import torch
    from torch._inductor.ir import FixedLayout, StorageBox, TensorBox
    kernel = _get_helion_kernel(kernel_idx)
    inputs, arg_names = [], []
    for name, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            inputs.append(_realize_input(tb))
            arg_names.append(name)
    dtype = output_spec.get("dtype", torch.float32)
    device = torch.device(output_spec.get("device", "cuda"))
    num_outputs = output_spec.get("num_outputs", 1)
    shape = output_spec.get("shape", [])
    bound = _try_bind_kernel(kernel, tensor_args, constant_args, arg_names, dtype, device)
    if num_outputs == 1:
        buf = HelionTemplate(layout=FixedLayout(device=device, dtype=dtype, size=shape), inputs=inputs,
            kernel=kernel, kernel_idx=kernel_idx, constant_args=constant_args, tensor_arg_names=arg_names, bound_kernel=bound)
        return (TensorBox(StorageBox(buf)),)
    buf = HelionTemplate(layout=MultiOutputLayout(device=device), inputs=inputs, kernel=kernel,
        kernel_idx=kernel_idx, constant_args=constant_args, tensor_arg_names=arg_names, bound_kernel=bound)
    layout = FixedLayout(device=device, dtype=dtype, size=shape)
    mos = [MultiOutput(layout=layout, input=buf, indices=[(tuple, i)]) for i in range(num_outputs)]
    buf.multi_output_nodes = mos
    return tuple(TensorBox.create(mo) for mo in mos)


def register_helion_lowerings() -> None:
    pass
