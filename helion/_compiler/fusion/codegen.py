"""Code generation for Helion templates."""
from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any, Callable

from torch._inductor.ir import MultiOutputLayout
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .context import fusion_context

if TYPE_CHECKING:
    from .template import HelionTemplate

log = logging.getLogger(__name__)


class HelionRender:
    """Generates Triton code for a HelionTemplate."""

    def __init__(self, template: "HelionTemplate") -> None:
        self.template = template

    def __call__(self) -> "HelionRender":
        self._generate()
        return self

    @property
    def code(self) -> str:
        return self.template._triton_code or ""

    def finalize_hook(self, hook_name: str, strict: bool = True) -> None:
        """No-op: Helion generates complete code in _generate()."""
        pass

    def finalize_remaining(self) -> str:
        """Return generated code. Required for Inductor interface."""
        return self.code

    def finalize_all(self) -> str:
        """Return generated code. Required for Inductor interface."""
        return self.code

    def _generate(self) -> None:
        """Generate Triton code with fusion applied."""
        from torch._inductor.utils import Placeholder
        from ..generate_ast import generate_ast
        from ..ast_extension import unparse
        from ..output_header import get_needed_imports
        from ...runtime.config import Config

        t = self.template
        if t.bound_kernel is None:
            return

        config = t.bound_kernel._require_implicit_config()
        if not isinstance(config, Config):
            config = Config(**config)
        t.bound_kernel.env.config_spec.normalize(config)

        host_fn = t.helion_kernel.name
        triton_fn = f"_helion_{host_fn}"
        placeholder = str(Placeholder.KERNEL_NAME)

        # Skip epilogue for atomics
        epilogues = t.epilogue_specs
        if t.uses_atomics() and epilogues:
            log.debug("Skipping epilogue for atomics: %s", host_fn)
            epilogues = {}

        # Build store index map for multi-output
        store_map = {}
        if isinstance(t.layout, MultiOutputLayout):
            from .helpers import safe_get_name
            for i, mo in enumerate(getattr(t, "multi_output_nodes", [])):
                name = safe_get_name(mo)
                if name:
                    store_map[i] = name

        # Generate code with fusion applied
        # Note: can_fuse checks are comprehensive - if we get here with epilogues/prologues,
        # they have been pre-validated and fusion MUST succeed. No fallback needed.
        with t.bound_kernel.env:
            with fusion_context(
                epilogues, t.prologue_specs, store_map
            ) as ctx:
                root = generate_ast(
                    t.bound_kernel.host_function,
                    config,
                    emit_repro_caller=False,
                )

            all_closures = ctx.all_closures
            if all_closures:
                inject_closure_params(root, all_closures, triton_fn, host_fn)
                t.epilogue_closures.update(all_closures.keys())

            rename_function(root, host_fn, placeholder)
            t._triton_code = get_needed_imports(root) + unparse(
                root,
                output_origin_lines=t.bound_kernel.settings.output_origin_lines,
            )


class HelionCodegen:
    """Handles code generation and emission for HelionTemplate."""

    def __init__(self, template: "HelionTemplate") -> None:
        self.template = template

    def _find_matching_closure(self, fused_input, closures) -> str | None:
        """Find a closure arg with matching shape to substitute for fused input."""
        if not closures:
            return None
        try:
            fused_size = fused_input.get_size()
        except Exception:
            return next(iter(closures), None)

        # Look for a closure with matching shape
        for buf_name in closures:
            try:
                buf = V.graph.get_buffer(buf_name)
                if buf is not None and hasattr(buf, "get_size") and buf.get_size() == fused_size:
                    return buf_name
            except Exception:
                continue

        return next(iter(closures), None)

    def emit_call(self, wrapper: Any, kernel_name: str) -> None:
        """Emit the kernel call site."""
        t = self.template

        output_name = t.get_name()

        # Build input names list, replacing fused inputs with closure args
        # When prologue is fused, the fused input buffer doesn't exist.
        # We need to pass a tensor with matching shape for size() calls.
        input_names = []
        for inp in t.inputs:
            inp_name = inp.get_name()
            if inp_name in t.prologue_fused_inputs:
                # Find a closure arg with matching shape
                substitute = self._find_matching_closure(inp, t.epilogue_closures)
                if substitute:
                    input_names.append(substitute)
                # else skip - shouldn't happen if fusion is correct
            else:
                input_names.append(inp_name)

        # Build args from signature
        args, input_idx = [], 0
        sig = t.helion_kernel.signature.parameters
        for name in sig:
            if name in t.tensor_arg_names and input_idx < len(input_names):
                args.append(input_names[input_idx])
                input_idx += 1
            elif name in t.constant_args_dict:
                args.append(repr(t.constant_args_dict[name]))
            elif sig[name].default is not sig[name].empty:
                args.append(repr(sig[name].default))

        # Add closure args (includes prologue closures)
        for buf_name in t.epilogue_closures:
            args.append(buf_name)

        wrapper.writeline(f"{output_name} = {kernel_name}({', '.join(args)})")

        # Handle multi-output unpacking
        multi_output_names_list = []
        multi_output_names = set()
        if isinstance(t.layout, MultiOutputLayout):
            for mo in getattr(t, "multi_output_nodes", []):
                if hasattr(mo, "get_name") and hasattr(mo, "indices") and mo.indices:
                    mo_name = mo.get_name()
                    multi_output_names.add(mo_name)
                    multi_output_names_list.append(mo_name)
                    idx_str = output_name
                    for _, idx in mo.indices:
                        idx_str = f"{idx_str}[{idx}]"
                    wrapper.writeline(f"{mo_name} = {idx_str}")

        # Epilogue aliases - map to multi-output names or output_name
        remaining_aliases = [
            a for a in getattr(t, "_helion_epilogue_aliases", [])
            if a != output_name and a not in multi_output_names
        ]
        for i, alias in enumerate(remaining_aliases):
            src = multi_output_names_list[i] if i < len(multi_output_names_list) else output_name
            wrapper.writeline(f"{alias} = {src}")

    def emit_to_wrapper(
        self,
        wrapper: Any,
        src_code: str,
        kernel_name: str,
        node_schedule: list,
        kernel_path: str,
        get_kernel_metadata_fn: Any,
    ) -> bool:
        """Emit Helion kernel code to wrapper header."""
        t = self.template

        # Add required imports
        wrapper.add_import_once("import triton")
        wrapper.add_import_once("import triton.language as tl")
        wrapper.add_import_once(
            "from helion.runtime import default_launcher as _default_launcher"
        )

        # Add optional imports based on code content
        if "libdevice." in src_code:
            wrapper.add_import_once(
                "from torch._inductor.runtime.triton_compat import libdevice"
            )
        if "tl_math." in src_code:
            wrapper.add_import_once(
                "from torch._inductor.runtime.triton_helpers import math as tl_math"
            )

        # Write metadata comment
        origins, detailed_origins = get_kernel_metadata_fn(node_schedule, wrapper)
        metadata_comment = f"# kernel path: {kernel_path}\n{origins}\n{detailed_origins}"
        wrapper.header.writeline(metadata_comment)

        # Strip imports and emit the kernel code
        lines = src_code.split("\n")
        in_imports = True
        for line in lines:
            stripped = line.strip()
            if in_imports:
                if stripped.startswith("from __future__"):
                    continue
                if stripped.startswith("import ") or stripped.startswith("from "):
                    continue
                if stripped == "":
                    continue
                in_imports = False
            wrapper.header.writeline(line)

        wrapper.header.writeline("")

        # Collect epilogue buffer names
        def _get_node_name(n):
            inner = n.node if hasattr(n, "node") else n
            return inner.get_name() if hasattr(inner, "get_name") else None

        template_name = t.get_name()
        t._helion_epilogue_aliases = [
            name for n in node_schedule
            if (name := _get_node_name(n)) and name != template_name
        ]
        return True

    def codegen_template(
        self,
        scheduling: Any,
        kernel: "HelionTemplate",
        template_node: Any,
        epilogue_nodes: list,
        prologue_nodes: list,
        buf_name_to_prologue_group: dict,
        prologue_preserves_zero_mask_fn: Any,
        render: Callable,
        only_gen_src_code: bool,
    ) -> Any:
        """Complete codegen for Helion templates with fusion."""
        from torch._inductor import config, ir
        from .helpers import partition_multi_output

        template_name = kernel.get_name()
        fusable_prologue_group = buf_name_to_prologue_group

        # Separate MultiOutput nodes (tuple unpacking) from actual epilogues
        multi_output_nodes, fusable_epilogue_nodes = partition_multi_output(epilogue_nodes or [])

        epilogues_by_output: dict = {}
        if fusable_epilogue_nodes:
            kernel.prepare_epilogues(fusable_epilogue_nodes, template_name)
            for acc_name, specs in kernel.epilogue_specs.items():
                for spec in specs:
                    epilogues_by_output.setdefault(acc_name, []).extend(spec.epilogue_nodes)

        if fusable_prologue_group:
            kernel.prepare_prologues(fusable_prologue_group)

        fusable_prologue_nodes = []
        for nodes in fusable_prologue_group.values():
            fusable_prologue_nodes.extend(nodes)

        with kernel:
            if not only_gen_src_code:
                for node in [template_node, *fusable_epilogue_nodes, *multi_output_nodes]:
                    node.mark_run()

            kernel._helion_epilogue_aliases = [
                ep.node.get_name()
                for eps in epilogues_by_output.values() for ep in eps
                if hasattr(ep, "node") and hasattr(ep.node, "get_name")
            ]

            partial_code = render()

            for input_name, buffer in kernel.named_input_nodes.items():
                    subgraph_name = f"<LOAD_INPUT_{input_name}>"
                    prologue_group = fusable_prologue_group.get(
                        buffer.get_name(), []
                    )
                    if not prologue_group:
                        continue

                    can_codegen_without_upcast = all(
                        p_n.can_codegen_without_upcasts() for p_n in prologue_group
                    )
                    with config.patch(
                        "triton.codegen_upcast_to_fp32", not can_codegen_without_upcast
                    ):
                        with kernel.set_subgraph_body(subgraph_name):
                            for prologue_node in prologue_group:
                                if (
                                    len(prologue_node.get_buffer_names()) == 1
                                    and len(prologue_group) == 1
                                ):
                                    if prologue_preserves_zero_mask_fn(prologue_node):
                                        kernel.prologue_fused_inputs_preserve_zero |= (
                                            prologue_node.get_buffer_names()
                                        )
                                prologue_node.codegen(
                                    kernel.split_and_set_ranges(prologue_node.get_ranges())
                                )
                            kernel.cse.invalidate(OrderedSet())

        with V.set_kernel_handler(kernel):
            if not isinstance(partial_code, str):
                with ir.IRNode.current_origins(template_node.node.origins):
                    partial_code.finalize_hook("<DEF_KERNEL>")
                partial_code.finalize_hook("<ARGDEFS>", strict=False)

            for input_name in kernel.named_input_nodes:
                subgraph_name = f"<LOAD_INPUT_{input_name}>"
                partial_code.finalize_hook(subgraph_name, strict=False)

            num_store_subgraphs = kernel.get_store_output_count()
            for i in range(num_store_subgraphs):
                subgraph_name = kernel._get_store_output_subgraph_name(i)
                partial_code.finalize_hook(subgraph_name)

            if isinstance(partial_code, str):
                src_code = partial_code
            else:
                src_code = partial_code.finalize_remaining()

            node_schedule = [*fusable_prologue_nodes, template_node, *fusable_epilogue_nodes]

            if config.benchmark_kernel:
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb).getvalue()}"
                )

            if only_gen_src_code:
                return src_code

            kernel.kernel_name = scheduling.define_kernel(
                src_code, node_schedule, kernel
            )

        return kernel


# --- AST Transform Utilities ---

class FusionError(Exception):
    """Raised when fusion cannot be applied."""
    pass


class _ASTSubstituter(ast.NodeTransformer):
    """Substitute a placeholder name with a replacement AST node."""

    def __init__(self, placeholder: str, replacement: ast.AST):
        self.placeholder = placeholder
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.placeholder:
            return self.replacement
        return node


def apply_fusion_ast(
    value: ast.AST,
    subscript_names: list[str],
    specs: list,
    register_closure_fn: Callable[[str], str],
) -> ast.AST:
    """Apply epilogue/prologue fusion to a value via AST manipulation."""
    from torch._inductor.ir import ComputedBuffer, Pointwise
    from torch._inductor.virtualized import OpsValue, V
    from .ops_handler import FusionOpsHandler
    from .context import EpilogueSpec, PrologueSpec

    if not specs:
        return value

    current = value

    for spec in specs:
        if isinstance(spec, EpilogueSpec):
            nodes = spec.epilogue_nodes
            acc_names = spec.accumulator_names
        else:
            nodes = spec.prologue_nodes
            acc_names = {spec.input_name}

        if not nodes:
            continue

        placeholder = "_hl_fusion_val_"
        handler = FusionOpsHandler(acc_names, placeholder, subscript_names, register_closure_fn)

        try:
            with V.set_ops_handler(handler):
                for s_node in nodes:
                    node = s_node.node
                    if isinstance(node, ComputedBuffer) and isinstance(node.data, Pointwise):
                        result = node.data.inner_fn(node.data._index(node.data.ranges))
                        if isinstance(result, OpsValue):
                            handler.final_value = result.value
                        else:
                            handler.final_value = str(result)
        except ValueError as e:
            raise FusionError(str(e)) from e

        current = _ASTSubstituter(placeholder, current).visit(handler.to_ast())

    return current


def inject_closure_params(root: ast.Module, closures: dict[str, str], triton_fn: str, host_fn: str) -> None:
    """Inject closure parameters into kernel AST."""
    if not closures:
        return

    params = list(closures.values())

    def _insert_before_block_size(args, new_args, make_arg):
        """Insert new_args before any _BLOCK_SIZE parameters."""
        existing = {(a.arg if hasattr(a, "arg") else a.id) for a in args if hasattr(a, "arg") or hasattr(a, "id")}
        to_add = [make_arg(p) for p in new_args if p not in existing]
        if not to_add:
            return
        idx = next((i for i, a in enumerate(args) if (getattr(a, "arg", None) or getattr(a, "id", "")).startswith("_BLOCK_SIZE")), len(args))
        args[idx:idx] = to_add

    tensor_ann = ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="Tensor", ctx=ast.Load())

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            if node.name == triton_fn:
                _insert_before_block_size(node.args.args, params, lambda p: ast.arg(arg=p))
            elif node.name == host_fn:
                _insert_before_block_size(node.args.args, params, lambda p: ast.arg(arg=p, annotation=tensor_ann))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_launcher":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == triton_fn:
                _insert_before_block_size(node.args, params, lambda p: ast.Name(id=p, ctx=ast.Load()))


def rename_function(root: ast.Module, old_name: str, new_name: str) -> None:
    """Rename a function in the AST."""
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
            break
