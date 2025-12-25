"""HelionTemplateBuffer - IR node for Helion kernels in Inductor."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Sequence

import sympy
import torch

from torch._inductor.ir import (
    Layout,
    MultiOutput,
    MultiOutputLayout,
    TritonTemplateBuffer,
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .convert import EpilogueSpec, PrologueSpec, FusionError, fx_to_pytorch_epilogue, EpilogueOp, get_torch_op

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
        # Epilogue metadata: {accumulator_name: (None, external_buffer_names, external_buffer_dims, accumulator_names_list, op_fn, epilogue_ops)}
        # First element is None (unused), external_buffer_dims maps buffer name to dimension indices
        self.epilogue_callables: dict[str, tuple[None, list[str], dict[str, list[int]], list[str], Callable | None, list[Callable]]] = {}
        # Multi-output epilogue metadata: {frozenset(accumulator_names): (None, ext_bufs, dims, acc_names, op_fn, ops)}
        self.multi_output_epilogue_callables: dict[frozenset[str], tuple[None, list[str], dict[str, list[int]], list[str], Callable | None, list[Callable]]] = {}
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

    def _make_render(self, template_buffer: "HelionTemplateBuffer", hint_override: int | None = None):
        return self, self.render
    def render(self) -> PartialRender:
        return PartialRender(self._generate_code(), {})

    def _generate_code(self) -> str:
        """Generate Triton code with fusion applied."""
        from torch._inductor.utils import Placeholder
        from ..ast_extension import unparse; from ..generate_ast import generate_ast
        from ..output_header import get_needed_imports; from ...runtime.config import Config
        from ..host_function import HostFunction
        from .convert import inject_closure_params, rename_function

        if not self.bound_kernel:
            return ""
        cfg = self.bound_kernel._require_implicit_config()
        if not isinstance(cfg, Config): cfg = Config(**cfg)
        self.bound_kernel.env.config_spec.normalize(cfg)
        host_fn, triton_fn = self.helion_kernel.name, f"_helion_{self.helion_kernel.name}"

        # Check if we should use re-tracing for epilogues
        host_function = self.bound_kernel.host_function
        env_to_use = self.bound_kernel.env  # Default to original env
        used_retrace = False  # Track whether we successfully used re-tracing
        # Use re-tracing for both single-output and multi-output epilogues/prologues.
        # Multi-output epilogues (like out + residual) are fused by:
        # 1. Skipping intermediate stores (they write the same value)
        # 2. At the last store, computing the combined result (e.g., value + value)
        # 3. Storing the combined result to the epilogue's output buffer
        # Prologues (like relu) transform input loads.
        has_epilogues = bool(self.epilogue_specs) or bool(self.multi_output_epilogue_callables) or bool(self.prologue_specs)
        if has_epilogues and not self.uses_atomics():
            try:
                host_function, env_to_use = self._create_retraced_host_function(cfg)
                used_retrace = True
                # Extend the config's indexing list for additional loads from epilogue buffers
                # Each external buffer adds one load operation
                if (num_extra := len(self.epilogue_closures)) > 0:
                    _l = lambda x: list(x) if x else None  # Helper to convert to list or None
                    cfg = Config(
                        block_sizes=list(cfg.block_sizes), indexing=list(cfg.indexing) + ['pointer'] * num_extra,
                        l2_groupings=_l(cfg.l2_groupings), load_eviction_policies=_l(cfg.load_eviction_policies),
                        loop_orders=_l(cfg.loop_orders), num_stages=cfg.num_stages, num_warps=cfg.num_warps,
                        pid_type=cfg.pid_type, range_flattens=_l(cfg.range_flattens), range_multi_buffers=_l(cfg.range_multi_buffers),
                        range_num_stages=_l(cfg.range_num_stages), range_unroll_factors=_l(cfg.range_unroll_factors),
                        range_warp_specializes=_l(cfg.range_warp_specializes), reduction_loops=_l(cfg.reduction_loops),
                        flatten_loops=getattr(cfg, 'flatten_loops', None), static_ranges=getattr(cfg, 'static_ranges', None),
                    )
            except Exception as e:
                # Fall back to original host function (no epilogue fusion)
                host_function = self.bound_kernel.host_function
                env_to_use = self.bound_kernel.env

        with env_to_use:
            root = generate_ast(host_function, cfg, emit_repro_caller=False)

        # Build closure name map for epilogue buffer parameters
        # When using re-trace, use the epilogue_closures that were populated during retracing
        closure_name_map: dict[str, str] = {}
        for buf_name in self.epilogue_closures:
            closure_name_map[buf_name] = f"_ext_{buf_name}"

        # Inject closure parameters into both Triton kernel and host function
        if closure_name_map:
            inject_closure_params(root, closure_name_map, triton_fn, host_fn)

        rename_function(root, host_fn, str(Placeholder.KERNEL_NAME))
        self._triton_code = get_needed_imports(root) + unparse(root, output_origin_lines=self.bound_kernel.settings.output_origin_lines)
        return self._triton_code

    def _create_retraced_host_function(self, cfg: Any) -> tuple["HostFunction", "CompileEnvironment"]:
        """Create HostFunction with epilogue ops traced (pure tracing approach)."""
        from ..host_function import HostFunction
        from ..compile_environment import CompileEnvironment
        from ..variable_origin import ArgumentOrigin
        from .convert import create_helion_traceable_epilogue, epilogue_tracing_context

        # Collect buffer info from epilogues
        ext_buf_info: dict[str, tuple[list[int], list[int] | None, torch.dtype, torch.device]] = {}
        ext_buf_dims: dict[str, int] = {}
        for _, (_, ext_buf_names, acc_ext_buf_dims, *_) in list(self.epilogue_callables.items()) + list(self.multi_output_epilogue_callables.items()):
            ext_buf_dims.update(acc_ext_buf_dims)
            for buf_name in ext_buf_names:
                if buf_name in ext_buf_info: continue
                try:
                    buf = V.graph.get_buffer(buf_name)
                    if buf is None: raise FusionError(f"Buffer {buf_name} not found")
                    size = [int(s) if hasattr(s, '__int__') else 64 for s in buf.get_size()]
                    stride = [int(s) if hasattr(s, '__int__') else 1 for s in buf.get_stride()] if hasattr(buf, 'get_stride') else None
                    ext_buf_info[buf_name] = (size, stride, buf.get_dtype() if hasattr(buf, 'get_dtype') else torch.float32, torch.device("cuda"))
                    self.epilogue_closures.add(buf_name)
                except FusionError: raise
                except Exception as e: raise FusionError(f"Failed to get external buffer {buf_name}: {e}") from e

        # Create a FRESH CompileEnvironment for re-tracing
        fresh_env = CompileEnvironment(
            self.bound_kernel.env.device,
            self.bound_kernel.settings,
            index_dtype=self.bound_kernel.env.index_dtype,
        )

        target_device = self.bound_kernel.env.device

        # Create real tensors for original args OUTSIDE the env context
        param_names = list(self.bound_kernel.kernel.signature.parameters.keys())
        real_tensors_to_convert: list[tuple[torch.Tensor | None, str | None]] = []
        for i, orig_fake in enumerate(self.bound_kernel.fake_args):
            if hasattr(orig_fake, 'shape') and hasattr(orig_fake, 'dtype'):
                real_tensor = torch.empty(
                    list(orig_fake.shape),
                    dtype=orig_fake.dtype,
                    device=target_device,
                )
                origin_name = param_names[i] if i < len(param_names) else None
                real_tensors_to_convert.append((real_tensor, origin_name))
            else:
                real_tensors_to_convert.append((None, None))

        # Create real tensors for external buffers
        ext_buf_real_tensors: dict[str, torch.Tensor] = {}
        for buf_name, (size, stride, dtype, device) in ext_buf_info.items():
            if stride is not None:
                ext_buf_real_tensors[buf_name] = torch.empty_strided(size, stride, dtype=dtype, device=target_device)
            else:
                ext_buf_real_tensors[buf_name] = torch.empty(size, dtype=dtype, device=target_device)

        # Collect prologue external buffer info BEFORE the fresh_env context
        # so we can create real tensors outside the fake mode context
        prologue_ext_buf_info: dict[str, tuple[list[int], list[int] | None, torch.dtype]] = {}
        if self.prologue_specs:
            from .convert import extract_prologue_ops

            # First pass: gather all external buffer names from all prologues
            prologue_ext_buf_names: set[str] = set()
            for param_name, spec_list in self.prologue_specs.items():
                for spec in spec_list:
                    # Call without ext_buf_fakes to just get buffer names
                    _, ext_buf_names = extract_prologue_ops(spec, None)
                    prologue_ext_buf_names.update(ext_buf_names)

            # Collect buffer info for prologue external buffers
            for buf_name in prologue_ext_buf_names:
                if buf_name in ext_buf_info: continue
                try:
                    buf = V.graph.get_buffer(buf_name)
                    if buf is None and hasattr(V.graph, 'graph_inputs_original'):
                        buf = V.graph.graph_inputs_original.get(buf_name)
                    if buf is not None:
                        size = [int(s) if hasattr(s, '__int__') else 64 for s in buf.get_size()]
                        stride = [int(s) if hasattr(s, '__int__') else 1 for s in buf.get_stride()] if hasattr(buf, 'get_stride') else None
                        prologue_ext_buf_info[buf_name] = (size, stride, buf.get_dtype() if hasattr(buf, 'get_dtype') else torch.float32)
                        self.epilogue_closures.add(buf_name)
                except Exception: pass
        # Create real tensors for prologue external buffers
        prologue_ext_buf_real_tensors: dict[str, torch.Tensor] = {}
        for buf_name, (size, stride, dtype) in prologue_ext_buf_info.items():
            prologue_ext_buf_real_tensors[buf_name] = torch.empty_strided(size, stride, dtype=dtype, device=target_device) if stride else torch.empty(size, dtype=dtype, device=target_device)

        with fresh_env:
            # Convert original args to fake tensors
            new_fake_args = []
            for i, (real_tensor, origin_name) in enumerate(real_tensors_to_convert):
                if real_tensor is not None:
                    origin = ArgumentOrigin(origin_name) if origin_name else None
                    new_fake = fresh_env.to_fake(real_tensor, origin)
                    new_fake_args.append(new_fake)
                else:
                    new_fake_args.append(self.bound_kernel.fake_args[i])

            # Convert external buffers to fake tensors (these will be captured as closures)
            ext_buf_fakes: dict[str, torch.Tensor] = {}
            for buf_name, real_tensor in ext_buf_real_tensors.items():
                origin = ArgumentOrigin(f"_ext_{buf_name}")
                ext_buf_fakes[buf_name] = fresh_env.to_fake(real_tensor, origin)
            # Build store_map and epilogue callables
            store_map = {i: acc for i, acc in enumerate(sorted(self.epilogue_callables.keys()))}
            store_idx = len(store_map)

            epilogue_callables_map: dict[str, Callable] = {}
            for acc_name in self.epilogue_callables:
                spec_list = self.epilogue_specs.get(acc_name, [])
                if spec_list:
                    try:
                        epilogue_callables_map[acc_name] = create_helion_traceable_epilogue(spec_list[0], acc_name, ext_buf_fakes, ext_buf_dims)
                    except FusionError: pass

            # Handle multi-output epilogues
            multi_output_epilogue_buffer, multi_output_op_fn, multi_output_acc_names = None, None, []
            if self.multi_output_epilogue_callables:
                _, (_, _, _, acc_names_mo, op_fn, _) = next(iter(self.multi_output_epilogue_callables.items()))
                existing_names = set(store_map.values())
                for acc_name in sorted(acc_names_mo):
                    if acc_name not in existing_names: store_map[store_idx] = acc_name; store_idx += 1
                multi_output_acc_names, multi_output_op_fn = list(sorted(acc_names_mo)), op_fn
                epilogue_output_name = getattr(self, 'multi_output_epilogue_output', None)
                if epilogue_output_name:
                    try:
                        buf = V.graph.get_buffer(epilogue_output_name)
                        if buf:
                            size = [int(s) if hasattr(s, '__int__') else 64 for s in buf.get_size()]
                            real_tensor = torch.empty(size, dtype=buf.get_dtype() if hasattr(buf, 'get_dtype') else torch.float32, device=target_device)
                            multi_output_epilogue_buffer = fresh_env.to_fake(real_tensor, ArgumentOrigin(f"_ext_{epilogue_output_name}"))
                            self.epilogue_closures.add(epilogue_output_name)
                    except Exception: pass

            # Convert prologue buffers to fake tensors
            prologue_ext_buf_fakes = {buf_name: fresh_env.to_fake(rt, ArgumentOrigin(f"_ext_{buf_name}")) for buf_name, rt in prologue_ext_buf_real_tensors.items()}
            all_ext_buf_fakes = {**ext_buf_fakes, **prologue_ext_buf_fakes}

            # Build prologue_info_map
            prologue_info_map: dict[int, "PrologueInfo"] = {}
            if self.prologue_specs:
                from .convert import extract_prologue_ops, PrologueInfo
                for param_name, spec_list in self.prologue_specs.items():
                    if param_name in param_names:
                        idx = param_names.index(param_name)
                        if idx < len(new_fake_args):
                            for spec in spec_list:
                                prologue_ops, _ = extract_prologue_ops(spec, all_ext_buf_fakes)
                                if prologue_ops: prologue_info_map[id(new_fake_args[idx])] = PrologueInfo(ops=prologue_ops); break

            constexpr_args = dict(getattr(self.bound_kernel.host_function, 'constexpr_args', {}))

            with epilogue_tracing_context(
                store_map,
                multi_output_op_fn=multi_output_op_fn,
                multi_output_acc_names=multi_output_acc_names,
                multi_output_epilogue_buffer=multi_output_epilogue_buffer,
                prologue_info=prologue_info_map,
                epilogue_callables=epilogue_callables_map,
            ):
                new_host_function = HostFunction(
                    self.helion_kernel.fn,  # ORIGINAL function, not wrapped!
                    tuple(new_fake_args),   # ORIGINAL args, no ext_buf params!
                    constexpr_args,
                )
        return new_host_function, fresh_env

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
        """Prepare epilogue specs from epilogue nodes and convert to callables."""
        if not epilogue_nodes or not self.supports_epilogue_fusion(): return
        outputs, by_output, multi_output_specs = self.output_names | {template_name}, {}, []
        for ep in epilogue_nodes:
            if not (hasattr(ep, "read_writes") and ep.read_writes):
                continue
            all_reads = {getattr(d, "name", None) for d in ep.read_writes.reads}
            reads = {getattr(d, "name", None) for d in ep.read_writes.reads if getattr(d, "name", None) in outputs}
            if len(reads) > 1:
                # Multi-output epilogue: collect for special handling
                multi_output_specs.append((ep, reads))
            elif len(reads) == 1: by_output.setdefault(next(iter(reads)), []).append(ep)

        # Process single-output epilogues
        for acc, nodes in by_output.items():
            spec = EpilogueSpec(epilogue_nodes=nodes, accumulator_name=acc)
            self.add_epilogue(acc, spec)
            # Convert to callable for tracing-level fusion
            try:
                make_fn, ext_bufs, ext_buf_dims, acc_names, op_fn, epilogue_ops = fx_to_pytorch_epilogue(spec, acc)
                self.epilogue_callables[acc] = (make_fn, ext_bufs, ext_buf_dims, acc_names, op_fn, epilogue_ops)
                # Register external buffers as closures
                for buf_name in ext_bufs:
                    self.epilogue_closures.add(buf_name)
            except FusionError as e:
                # Fall back to no fusion for this epilogue
                pass

        # Process multi-output epilogues (e.g., out + residual)
        for ep, reads in multi_output_specs:
            try:
                epilogue_node = ep.node
                acc_names = sorted(reads)
                from torch._inductor.ir import ComputedBuffer, Pointwise
                if not isinstance(epilogue_node, ComputedBuffer) or not isinstance(epilogue_node.data, Pointwise): continue

                pw = epilogue_node.data
                from torch._inductor.loop_body import LoopBody
                from torch._inductor import dependencies
                (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(pw.ranges, (), prefix="i")
                lb = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, reduce_vars)

                # Find binary operation in FX graph
                op_fn = None
                for node in lb.root_block.graph.nodes:
                    if node.op == "call_method" and isinstance(node.target, str) and node.target not in ("load", "store", "constant"):
                        try:
                            op_fn = get_torch_op(node.target)
                            break
                        except ValueError:
                            continue
                if op_fn is None: continue

                self.multi_output_epilogue_callables[frozenset(acc_names)] = (None, [], {}, acc_names, op_fn, [])
                self.multi_output_epilogue_output = epilogue_node.get_name()
            except Exception: continue

    def prepare_prologues(self, buf_to_prologue: dict[str, list]) -> None:
        for name, buf in self.named_input_nodes.items():
            buf_name = buf.get_name()
            if not (p := buf_to_prologue.get(buf_name)): continue
            self.add_prologue(name, PrologueSpec(prologue_nodes=list(p), input_name=buf_name))
            self.prologue_fused_inputs.add(buf_name)
            # Find original input the prologue reads from
            for pn in p:
                if hasattr(pn, 'read_writes') and pn.read_writes:
                    for dep in pn.read_writes.reads:
                        if (orig_name := getattr(dep, 'name', None)) and orig_name != buf_name:
                            if not hasattr(self, 'prologue_input_mapping'): self.prologue_input_mapping: dict[str, str] = {}
                            self.prologue_input_mapping[buf_name] = orig_name; break
            self.removed_buffers.add(buf_name)
    def uses_atomics(self) -> bool:
        if self._uses_atomics is not None: return self._uses_atomics
        if not self.bound_kernel: self._uses_atomics = True; return True
        try:
            from ..generate_ast import generate_ast; from ..ast_extension import unparse; from ...runtime.config import Config
            cfg = self.bound_kernel._require_implicit_config()
            if not isinstance(cfg, Config): cfg = Config(**cfg)
            self.bound_kernel.env.config_spec.normalize(cfg)
            with self.bound_kernel.env:
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

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:
        """Emit the kernel call site."""
        wrapper, output_name = V.graph.wrapper_code, self.get_name()
        # Build input names, replacing fused prologue inputs with their originals
        prologue_mapping = getattr(self, 'prologue_input_mapping', {})
        input_names = [(prologue_mapping.get(n, n) if n in self.prologue_fused_inputs else n) for inp in self.inputs for n in [inp.get_name()]]
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

        # Codegen unfused epilogues as separate kernels AFTER the template kernel call
        # These are epilogues that we couldn't fuse (e.g., multi-output epilogues like out + residual)
        if hasattr(self, "_unfused_epilogues") and self._unfused_epilogues:
            for node in self._unfused_epilogues:
                self._scheduling.codegen_node(node)

    def emit_kernel(self, wrapper: Any, src_code: str, kernel_name: str,
                    node_schedule: list, kernel_path: str, get_kernel_metadata_fn: Any) -> bool:
        """Emit Helion kernel code to wrapper header."""
        wrapper.add_import_once("import triton"); wrapper.add_import_once("import triton.language as tl")
        wrapper.add_import_once("from helion.runtime import default_launcher as _default_launcher")
        if "libdevice." in src_code: wrapper.add_import_once("from torch._inductor.runtime.triton_compat import libdevice")
        if "tl_math." in src_code: wrapper.add_import_once("from torch._inductor.runtime.triton_helpers import math as tl_math")
        if "triton_helpers." in src_code: wrapper.add_import_once("from torch._inductor.runtime import triton_helpers")

        # Handle _source_module import
        import re
        if "_source_module" in src_code:
            match = re.search(r'^import\s+(\S+)\s+as\s+_source_module\s*$', src_code, re.MULTILINE)
            module_name = match.group(1) if match else self.helion_kernel.fn.__module__
            wrapper.add_import_once(f"import {module_name} as _source_module")

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
        # Only set if not already set by codegen_template (None check needed - empty list is valid)
        if getattr(self, "_helion_epilogue_aliases", None) is None:
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

        # Collect actually fused nodes (single + multi-output epilogues)
        actually_fused_epilogues = {ep for nodes in epilogues_by_output.values() for ep in nodes}
        if kernel.multi_output_epilogue_callables:
            for ep in fusable_epilogue_nodes:
                if hasattr(ep, 'read_writes') and ep.read_writes:
                    reads = {getattr(d, "name", None) for d in ep.read_writes.reads if getattr(d, "name", None) in kernel.output_names}
                    if len(reads) > 1: actually_fused_epilogues.add(ep)

        with kernel:
            if not only_gen_src_code:
                # Only mark nodes that are actually fused - not ones we skipped (like multi-output epilogues)
                nodes_to_mark = [template_node, *multi_output_nodes]
                nodes_to_mark.extend(n for n in fusable_epilogue_nodes if n in actually_fused_epilogues)
                for node in nodes_to_mark: node.mark_run()
            self._helion_epilogue_aliases = [ep.node.get_name() for eps in epilogues_by_output.values()
                                             for ep in eps if hasattr(ep, "node") and hasattr(ep.node, "get_name")]
            # Add multi-output epilogue alias
            if hasattr(kernel, 'multi_output_epilogue_output') and kernel.multi_output_epilogue_output:
                self._helion_epilogue_aliases.append(kernel.multi_output_epilogue_output)
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

        # Store unfused epilogues for codegen in call_kernel
        unfused = [n for n in fusable_epilogue_nodes if n not in actually_fused_epilogues]
        if unfused: self._unfused_epilogues, self._scheduling = unfused, scheduling

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
            elif name in constant_args:
                fake_tensors.append(constant_args[name])
            elif sig[name].default is not sig[name].empty:
                fake_tensors.append(sig[name].default)
        result = kernel.bind(tuple(fake_tensors))
        return result
    except Exception as e:
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
