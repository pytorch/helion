"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import sympy

from torch._inductor.ir import (
    Layout,
    MultiOutput,
    MultiOutputLayout,
    TritonTemplateBuffer,
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .convert import EpilogueSpec, PrologueSpec, fusion_context

if TYPE_CHECKING:
    from torch._inductor.ir import IRNode


class HelionTemplateBuffer(TritonTemplateBuffer):
    """Helion kernel IR node with fusion support."""

    def __init__(self, layout: Layout, inputs: Sequence["IRNode"], kernel: Any, kernel_idx: int,
                 constant_args: dict[str, Any], tensor_arg_names: list[str], bound_kernel: Any) -> None:
        self.helion_kernel, self.kernel_idx = kernel, kernel_idx
        self.constant_args_dict, self.tensor_arg_names, self.bound_kernel = constant_args, tensor_arg_names, bound_kernel
        self.output_ranges = list(layout.size) if hasattr(layout, "size") else []
        self.epilogue_specs: dict[str, list[EpilogueSpec]] = {}
        self.prologue_specs: dict[str, list[PrologueSpec]] = {}
        self.deferred_epilogue_specs: list[EpilogueSpec] = []
        self._uses_atomics: bool | None = None
        self.epilogue_closures: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs: set[str] = set()
        self.prologue_fused_inputs_preserve_zero: set[str] = set()
        self.removed_buffers: set[str] = set()
        self.inplaced_to_remove: set[str] = set()
        self.named_input_nodes = {name: inputs[i] for i, name in enumerate(tensor_arg_names) if i < len(inputs)}
        self._triton_code: str | None = None
        self.cse, self.kernel_name = _HelionCSE(), None
        super().__init__(layout=layout, inputs=inputs, make_kernel_render=self._make_render, mutated_inputs=None,
                        allowed_prologue_inps=OrderedSet(n for inp in inputs if (n := _safe_get_name(inp))))
        if self.uses_atomics(): V.graph.no_fuse_buffer_names.add(self.get_name())

    def _make_render(self, template_buffer: "HelionTemplateBuffer", hint_override: int | None = None): return self, self.render
    def render(self) -> PartialRender: return PartialRender(self._generate_code(), {})

    def _generate_code(self) -> str:
        """Generate Triton code with fusion applied."""
        from torch._inductor.utils import Placeholder
        from ..ast_extension import unparse; from ..generate_ast import generate_ast
        from ..output_header import get_needed_imports; from ...runtime.config import Config
        from .convert import inject_closure_params, rename_function
        if not self.bound_kernel: return ""
        cfg = self.bound_kernel._require_implicit_config()
        if not isinstance(cfg, Config): cfg = Config(**cfg)
        self.bound_kernel.env.config_spec.normalize(cfg)
        host_fn, triton_fn = self.helion_kernel.name, f"_helion_{self.helion_kernel.name}"
        epilogues = {} if self.uses_atomics() else self.epilogue_specs
        store_map = {i: n for i, mo in enumerate(getattr(self, "multi_output_nodes", [])) if (n := _safe_get_name(mo))} if isinstance(self.layout, MultiOutputLayout) else {}

        with self.bound_kernel.env, fusion_context(epilogues, self.prologue_specs, store_map) as ctx:
            ctx.multi_output_epilogues = self.deferred_epilogue_specs
            root = generate_ast(self.bound_kernel.host_function, cfg, emit_repro_caller=False)
        if ctx.all_closures:
            inject_closure_params(root, ctx.all_closures, triton_fn, host_fn)
            self.epilogue_closures.update(ctx.all_closures.keys())
        rename_function(root, host_fn, str(Placeholder.KERNEL_NAME))
        self._triton_code = get_needed_imports(root) + unparse(root, output_origin_lines=self.bound_kernel.settings.output_origin_lines)
        return self._triton_code

    # Benchmark stubs (not used for Helion)
    def estimate_kernel_num_bytes(self) -> int: return 0
    def imports_for_benchmark_kernel(self) -> str: return ""
    def codegen_kernel_benchmark(self, num_gb: float) -> Any:
        from torch._inductor.codecache import IndentedBuffer; return IndentedBuffer()

    def codegen_template_override(self, scheduling, template_node, epilogue_nodes, prologue_nodes,
                                   buf_name_to_prologue_group, prologue_preserves_zero_mask_fn, render, only_gen_src_code):
        return self.codegen_template(scheduling, self, template_node, epilogue_nodes, prologue_nodes,
                                     buf_name_to_prologue_group, prologue_preserves_zero_mask_fn, render, only_gen_src_code)

    def emit_kernel_override(self, wrapper, src_code, kernel_name, node_schedule, kernel_path, get_kernel_metadata):
        return self.emit_kernel(wrapper, src_code, kernel_name, node_schedule, kernel_path, get_kernel_metadata)

    def add_epilogue(self, name: str, spec: EpilogueSpec) -> None: self.epilogue_specs.setdefault(name, []).append(spec)
    def add_prologue(self, name: str, spec: PrologueSpec) -> None: self.prologue_specs.setdefault(name, []).append(spec)

    def prepare_epilogues(self, epilogue_nodes: list, template_name: str) -> None:
        """Prepare epilogue specs from epilogue nodes."""
        if not epilogue_nodes or not self.supports_epilogue_fusion(): return
        outputs, by_output, deferred = self.output_names | {template_name}, {}, []
        for ep in epilogue_nodes:
            if not (hasattr(ep, "read_writes") and ep.read_writes): continue
            reads = {getattr(d, "name", None) for d in ep.read_writes.reads if getattr(d, "name", None) in outputs}
            if len(reads) > 1: deferred.append(EpilogueSpec(epilogue_nodes=[ep], accumulator_name=reads))
            elif len(reads) == 1: by_output.setdefault(next(iter(reads)), []).append(ep)
        for acc, nodes in by_output.items():
            self.add_epilogue(acc, EpilogueSpec(epilogue_nodes=nodes, accumulator_name=acc))
        self.deferred_epilogue_specs = deferred

    def prepare_prologues(self, buf_to_prologue: dict[str, list]) -> None:
        for name, buf in self.named_input_nodes.items():
            if p := buf_to_prologue.get(buf.get_name()): self.add_prologue(name, PrologueSpec(prologue_nodes=list(p), input_name=buf.get_name()))

    def uses_atomics(self) -> bool:
        if self._uses_atomics is not None: return self._uses_atomics
        if not self.bound_kernel: self._uses_atomics = True; return True
        try:
            from ..generate_ast import generate_ast; from ..ast_extension import unparse; from ...runtime.config import Config; from .convert import fusion_context
            cfg = self.bound_kernel._require_implicit_config()
            if not isinstance(cfg, Config): cfg = Config(**cfg)
            self.bound_kernel.env.config_spec.normalize(cfg)
            with self.bound_kernel.env, fusion_context({}, {}):
                self._uses_atomics = "tl.atomic_" in unparse(generate_ast(self.bound_kernel.host_function, cfg, emit_repro_caller=False), output_origin_lines=False)
        except Exception: self._uses_atomics = True
        return self._uses_atomics

    def supports_epilogue_fusion(self) -> bool: return not self.uses_atomics()

    def get_layout(self) -> Layout:
        if isinstance(self.layout, MultiOutputLayout):
            mo = getattr(self, "multi_output_nodes", None)
            assert mo and hasattr(mo[0], "layout"), "MultiOutputLayout without multi_output_nodes"
            return mo[0].layout
        return super().get_layout()

    @property
    def is_multi_output(self) -> bool: return isinstance(self.layout, MultiOutputLayout)
    def can_fuse_multi_output(self, node2: Any) -> bool:
        return isinstance(self.layout, MultiOutputLayout) and isinstance(node2.node, MultiOutput) and len(node2.node.inputs) == 1 and node2.node.inputs[0].get_name() == self.get_name()
    def supports_multi_outputs(self) -> bool: return isinstance(self.layout, MultiOutputLayout)
    @property
    def output_names(self) -> set[str]:
        return {self.get_name()} | {o.get_name() for o in getattr(self, "multi_output_nodes", []) if hasattr(o, "get_name")}

    def __enter__(self) -> "HelionTemplateBuffer":
        import contextlib; self._exit_stack = contextlib.ExitStack(); self._exit_stack.__enter__()
        self._exit_stack.enter_context(V.set_kernel_handler(self)); return self
    def __exit__(self, *args: Any) -> None: self._exit_stack.__exit__(*args)
    def set_current_node(self, node: Any) -> "_NodeContext": return _NodeContext(self, node)
    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]) -> list[list[sympy.Expr]]:
        result, idx = [], 0
        for g in lengths: result.append([sympy.Symbol(f"idx{idx + i}") for i in range(len(g))]); idx += len(g)
        return result

    def _find_matching_closure(self, fused_input, closures) -> str | None:
        """Find a closure arg with matching shape to substitute for fused input."""
        if not closures:
            return None
        try:
            fused_size = fused_input.get_size()
            for buf_name in closures:
                try:
                    buf = V.graph.get_buffer(buf_name)
                    if buf and hasattr(buf, "get_size") and buf.get_size() == fused_size:
                        return buf_name
                except Exception:
                    continue
        except Exception:
            pass
        return next(iter(closures), None)

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:
        """Emit the kernel call site."""
        wrapper, output_name = V.graph.wrapper_code, self.get_name()

        # Build input names, replacing fused inputs with closure args
        input_names = [self._find_matching_closure(inp, self.epilogue_closures) or inp.get_name()
                       if inp.get_name() in self.prologue_fused_inputs else inp.get_name() for inp in self.inputs]

        # Build args from signature
        args, input_idx, sig = [], 0, self.helion_kernel.signature.parameters
        for name in sig:
            if name in self.tensor_arg_names and input_idx < len(input_names): args.append(input_names[input_idx]); input_idx += 1
            elif name in self.constant_args_dict: args.append(repr(self.constant_args_dict[name]))
            elif sig[name].default is not sig[name].empty: args.append(repr(sig[name].default))
        args.extend(self.epilogue_closures)
        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Handle multi-output unpacking
        mo_names, mo_list = set(), []
        if isinstance(self.layout, MultiOutputLayout):
            for mo in getattr(self, "multi_output_nodes", []):
                if hasattr(mo, "get_name") and hasattr(mo, "indices") and mo.indices:
                    n = mo.get_name(); mo_names.add(n); mo_list.append(n)
                    idx_str = output_name
                    for _, idx in mo.indices: idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{n} = {idx_str}")
        for i, a in enumerate(a for a in getattr(self, "_helion_epilogue_aliases", []) if a != output_name and a not in mo_names):
            wrapper.writeline(f"{a} = {mo_list[i] if i < len(mo_list) else output_name}")

    def emit_kernel(self, wrapper: Any, src_code: str, kernel_name: str,
                    node_schedule: list, kernel_path: str, get_kernel_metadata_fn: Any) -> bool:
        """Emit Helion kernel code to wrapper header."""
        wrapper.add_import_once("import triton"); wrapper.add_import_once("import triton.language as tl")
        wrapper.add_import_once("from helion.runtime import default_launcher as _default_launcher")
        if "libdevice." in src_code: wrapper.add_import_once("from torch._inductor.runtime.triton_compat import libdevice")
        if "tl_math." in src_code: wrapper.add_import_once("from torch._inductor.runtime.triton_helpers import math as tl_math")

        origins, detailed = get_kernel_metadata_fn(node_schedule, wrapper)
        wrapper.header.writeline(f"# kernel path: {kernel_path}\n{origins}\n{detailed}")

        in_imports = True
        for line in src_code.split("\n"):
            s = line.strip()
            if in_imports and (s.startswith(("from __future__", "import ", "from ")) or not s): continue
            in_imports = False
            wrapper.header.writeline(line)
        wrapper.header.writeline("")

        _name = lambda n: (n.node if hasattr(n, "node") else n).get_name() if hasattr(n.node if hasattr(n, "node") else n, "get_name") else None
        self._helion_epilogue_aliases = [n for sn in node_schedule if (n := _name(sn)) and n != self.get_name() and n not in self.epilogue_closures]
        return True

    def codegen_template(self, scheduling: Any, kernel: Any, template_node: Any, epilogue_nodes: list,
                         prologue_nodes: list, buf_name_to_prologue_group: dict,
                         prologue_preserves_zero_mask_fn: Any, render: Callable, only_gen_src_code: bool) -> Any:
        """Complete codegen for Helion templates with fusion."""
        from torch._inductor import config
        multi_output_nodes, fusable_epilogue_nodes = partition_multi_output(epilogue_nodes or [])

        epilogues_by_output = {}
        if fusable_epilogue_nodes:
            self.prepare_epilogues(fusable_epilogue_nodes, self.get_name())
            for acc, specs in self.epilogue_specs.items():
                epilogues_by_output.setdefault(acc, []).extend(n for s in specs for n in s.epilogue_nodes)
        if buf_name_to_prologue_group: self.prepare_prologues(buf_name_to_prologue_group)
        fusable_prologue_nodes = [n for nodes in buf_name_to_prologue_group.values() for n in nodes]

        with kernel:
            if not only_gen_src_code:
                for node in [template_node, *fusable_epilogue_nodes, *multi_output_nodes]: node.mark_run()
            self._helion_epilogue_aliases = [ep.node.get_name() for eps in epilogues_by_output.values()
                                             for ep in eps if hasattr(ep, "node") and hasattr(ep.node, "get_name")]
            partial_code = render()
            for _, buffer in kernel.named_input_nodes.items():
                pg = buf_name_to_prologue_group.get(buffer.get_name(), [])
                if not pg: continue
                with config.patch("triton.codegen_upcast_to_fp32", not all(p.can_codegen_without_upcasts() for p in pg)):
                    for pn in pg:
                        if len(pn.get_buffer_names()) == 1 and len(pg) == 1 and prologue_preserves_zero_mask_fn(pn):
                            kernel.prologue_fused_inputs_preserve_zero |= pn.get_buffer_names()
                        pn.codegen(kernel.split_and_set_ranges(pn.get_ranges()))
                    kernel.cse.invalidate(OrderedSet())

        with V.set_kernel_handler(self):
            src_code = partial_code if isinstance(partial_code, str) else partial_code.finalize_remaining()
            node_schedule = [*fusable_prologue_nodes, template_node, *fusable_epilogue_nodes]
            if config.benchmark_kernel:
                src_code = f"{kernel.imports_for_benchmark_kernel()}\n{src_code}\n{kernel.codegen_kernel_benchmark(kernel.estimate_kernel_num_bytes() / 1e9).getvalue()}"
            if only_gen_src_code: return src_code
            kernel.kernel_name = scheduling.define_kernel(src_code, node_schedule, kernel)
        return kernel


class _HelionCSE:
    def __init__(self): self._cache = {}
    def invalidate(self, names): [self._cache.pop(n, None) for n in names]

class _NodeContext:
    def __init__(self, kernel, node): self.kernel, self.node = kernel, node
    def __enter__(self): self._prev, self.kernel._current_node = getattr(self.kernel, "_current_node", None), self.node; return self
    def __exit__(self, *a): self.kernel._current_node = self._prev


# --- Helper Functions ---
def _safe_get_name(node: Any) -> str | None:
    return node.get_name() if hasattr(node, "get_name") else None

def _is_multi_output_node(node: Any) -> bool:
    return isinstance(getattr(node, "node", node), MultiOutput)

def partition_multi_output(nodes: list) -> tuple[list, list]:
    """Partition nodes into MultiOutput and non-MultiOutput groups."""
    mo, other = [], []
    for n in nodes:
        (mo if _is_multi_output_node(n) else other).append(n)
    return mo, other

safe_get_name, is_multi_output_node = _safe_get_name, _is_multi_output_node  # backward compat


# --- Inductor Lowering ---
def _realize_input(tensor_box):
    from torch._inductor.ir import Buffer, ComputedBuffer, InputBuffer, ReinterpretView, StorageBox, TensorBox, TemplateBuffer
    BUF_TYPES = (ComputedBuffer, InputBuffer, ReinterpretView, TemplateBuffer, Buffer)
    if not isinstance(tensor_box, TensorBox):
        return tensor_box
    data = tensor_box.data
    if isinstance(data, StorageBox):
        if not isinstance(data.data, BUF_TYPES): data.realize()
        return data.data
    if isinstance(data, BUF_TYPES):
        return data
    tensor_box.realize()
    return tensor_box.data.data if isinstance(tensor_box.data, StorageBox) else tensor_box.data

def _get_helion_kernel(kernel_idx: int):
    from helion._dynamo.higher_order_ops import get_helion_kernel
    return get_helion_kernel(kernel_idx)

def _try_bind_kernel(kernel, tensor_args, constant_args, arg_names, dtype, device):
    import torch
    try:
        fake_tensors, sig = [], kernel.signature.parameters
        for name in sig:
            if name in arg_names:
                tb = list(tensor_args.values())[arg_names.index(name)]
                size = [int(s) if isinstance(s, (int, sympy.Integer)) else 64 for s in tb.get_size()] if hasattr(tb, "get_size") else [1]
                fake_tensors.append(torch.empty(size, dtype=dtype, device=device))
            elif name in constant_args: fake_tensors.append(constant_args[name])
            elif sig[name].default is not sig[name].empty: fake_tensors.append(sig[name].default)
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
    inputs, arg_names = zip(*[((_realize_input(tb), name)) for name, tb in tensor_args.items() if isinstance(tb, TensorBox)]) if tensor_args else ([], [])
    inputs, arg_names = list(inputs), list(arg_names)
    dtype, device = output_spec.get("dtype", torch.float32), torch.device(output_spec.get("device", "cuda"))
    num_outputs, shape = output_spec.get("num_outputs", 1), output_spec.get("shape", [])
    bound = _try_bind_kernel(kernel, tensor_args, constant_args, arg_names, dtype, device)
    if num_outputs == 1:
        buf = HelionTemplateBuffer(layout=FixedLayout(device=device, dtype=dtype, size=shape), inputs=inputs,
            kernel=kernel, kernel_idx=kernel_idx, constant_args=constant_args, tensor_arg_names=arg_names, bound_kernel=bound)
        return (TensorBox(StorageBox(buf)),)
    buf = HelionTemplateBuffer(layout=MultiOutputLayout(device=device), inputs=inputs, kernel=kernel,
        kernel_idx=kernel_idx, constant_args=constant_args, tensor_arg_names=arg_names, bound_kernel=bound)
    layout = FixedLayout(device=device, dtype=dtype, size=shape)
    buf.multi_output_nodes = [MultiOutput(layout=layout, input=buf, indices=[(tuple, i)]) for i in range(num_outputs)]
    return tuple(TensorBox.create(mo) for mo in buf.multi_output_nodes)
