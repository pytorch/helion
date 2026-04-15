from __future__ import annotations

import ast
from contextlib import nullcontext
import dataclasses
import functools
import hashlib
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._dynamo.utils import ExactWeakKeyDictionary
from torch._inductor.codecache import PyCodeCache
from torch._inductor.ir import Buffer
from torch._inductor.ir import FinalizeCodegenResult
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import clone
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.select_algorithm import (
    ExternalTritonTemplateKernel,  # pyrefly: ignore[missing-module-attribute]
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._inductor.utils import convert_shape_to_symint
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _get_flat_output
from ..ast_extension import unparse
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import _active_library_imports
from ..output_header import get_needed_import_lines
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from typing import Any
    from typing import Iterable

    from torch._inductor.ir import MultiOutput

    from ..inductor_lowering import CodegenState
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


class _CodeExpr(str):
    """A str whose repr() returns itself, for embedding variable names in generated code.

    When generating a kernel call like ``kernel(x, (a, b))``, container args are
    rebuilt via pytree into e.g. ``(_CodeExpr("a"), _CodeExpr("b"))``.  Python's
    built-in ``repr()`` on that tuple then produces ``(a, b)`` instead of
    ``('a', 'b')``, giving us correct code for free.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


@dataclasses.dataclass
class _FusionMetadata:
    epilogue_idx_by_param: dict[str, int]
    epilogue_keep_store: set[str]
    prologue_vars: dict[str, dict[str, str]]
    prologue_fused_params: set[str]
    prologue_has_source: set[str]


class _FusionAutotuneAdapter:
    """Adapter for fusion-aware autotuning.

    ``compile_config`` returns the unfused kernel (for accuracy checking),
    ``bench_compile_config`` returns the fused kernel (for benchmarking).
    All other attributes are delegated to the wrapped ``BoundKernel`` via
    ``__getattr__``.
    """

    def __init__(
        self,
        bound_kernel: BoundKernel,
        bench_kernel_factory: Callable[
            [Config], tuple[Callable[..., object], str | None]
        ],
        fusion_context_hash: str,
    ) -> None:
        self._bound_kernel = bound_kernel
        self._factory = bench_kernel_factory
        self._fusion_context_hash = fusion_context_hash
        self._bench_cached_paths: dict[Config, str] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._bound_kernel, name)

    def bench_compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        if not isinstance(config, Config):
            config = (
                Config.from_dict(config)
                if config is not None
                else self._bound_kernel.config_spec.default_config()
            )
        fn, path = self._factory(config)
        if path is not None:
            self._bench_cached_paths[config] = path
        return fn

    def get_cached_path(self, config: Config | None = None) -> str | None:
        if config is None:
            return None
        return self._bench_cached_paths.get(config)

    def extra_cache_key(self) -> str:
        return self._fusion_context_hash


class HelionTemplateBuffer(TemplateBuffer):
    """Inductor template buffer for Helion kernel."""

    _fusion_config_cache: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        *,
        kernel: Kernel,
        bound_kernel: BoundKernel,
        constant_args: dict[str, object],
        autotune_args: tuple[object, ...] | None = None,
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
        named_inputs: dict[str, IRNode] | None = None,
    ) -> None:
        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args
        self._fusion_metadata: _FusionMetadata | None = None

        tb_self = self  # capture for closure

        def _make_kernel_render(
            out_node: TemplateBuffer, hint_override: object = None
        ) -> tuple[object, Callable[[], PartialRender]]:
            kernel = ExternalTritonTemplateKernel(out_node)

            def render() -> PartialRender:
                return tb_self._render_with_hooks(kernel)

            return kernel, render

        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=_make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
            named_inputs=named_inputs,  # pyrefly: ignore[unexpected-keyword]
        )

    def _render_with_hooks(self, kernel: Any) -> PartialRender:  # noqa: ANN401
        """Set up fusion hooks, read metadata, return a placeholder PartialRender.

        Called as the ``render()`` function from the standard
        ``codegen_template_body`` path.  Always returns a placeholder —
        autotuning and AST generation are deferred to
        ``_finalize_codegen`` (called after Inductor has
        processed epilogue/prologue subgraphs).
        """
        # 1. Set up fusion hooks (requires V.kernel context).
        kernel._setup_fusion_hooks()

        # 2. Read pre-computed fusion metadata from kernel into a single object.
        prologue_fused_params = set(kernel._prologue_vars.keys())
        self._fusion_metadata = _FusionMetadata(
            epilogue_idx_by_param=kernel._epilogue_idx_by_param,
            epilogue_keep_store=kernel._epilogue_keep_store,
            prologue_vars=kernel._prologue_vars,
            prologue_fused_params=prologue_fused_params,
            prologue_has_source={
                param_name
                for param_name in prologue_fused_params
                if kernel._prologue_source_buffers.get(param_name) is not None
            },
        )
        self._prologue_source_buffers = dict(kernel._prologue_source_buffers)

        # 3. Build extra_params list from extra inputs and store targets.
        extra_params_dict: dict[str, str] = {}
        for buf, param in kernel._extra_inputs.items():
            extra_params_dict.setdefault(param, buf)
        for buf, param in kernel._extra_store_targets.items():
            if param in extra_params_dict and extra_params_dict[param] != buf:
                raise RuntimeError(
                    f"extra_params conflict: param '{param}' maps to "
                    f"'{extra_params_dict[param]}' (from _extra_inputs) and "
                    f"'{buf}' (from _extra_store_targets)"
                )
            extra_params_dict.setdefault(param, buf)
        self._extra_params = sorted(extra_params_dict.items())

        # Pre-compute benchmark example values while V.graph is available.
        # Only needed when fusion-aware autotuning will actually run.
        if (
            self._autotune_args is not None
            and self._bound_kernel.settings.autotune_with_torch_compile_fusion
        ):
            self._extra_param_examples = [
                AlgorithmSelectorCache.benchmark_example_value(V.graph.get_buffer(buf))
                for _, buf in self._extra_params
            ]
        else:
            self._extra_param_examples = []

        # Return placeholder — autotuning and AST generation happen in _finalize_codegen.
        return PartialRender("", kernel.render_hooks)

    def _finalize_codegen(
        self,
        hook_outputs: dict[str, str],
    ) -> FinalizeCodegenResult | None:
        """Generate final kernel code after subgraph codegen.

        Receives hook outputs (rendered epilogue/prologue code fragments)
        from the kernel.  Returns a FinalizeCodegenResult with the final
        source code and call metadata.
        """
        bk = self._bound_kernel
        use_fusion_autotune = (
            self._autotune_args is not None
            and hook_outputs
            and bk.settings.autotune_with_torch_compile_fusion
        )
        if use_fusion_autotune:
            fusion_key = tuple(sorted(hook_outputs.items()))
            bk_cache = self._fusion_config_cache.get(bk)
            if bk_cache is not None and fusion_key in bk_cache:
                cfg = bk_cache[fusion_key]
            else:
                cfg = bk._user_provided_config()
                if cfg is None:
                    fusion_context_hash = hashlib.sha256(
                        repr(fusion_key).encode()
                    ).hexdigest()
                    adapter = _FusionAutotuneAdapter(
                        bk,
                        self._make_bench_kernel_factory(hook_outputs),
                        fusion_context_hash,
                    )
                    cfg = bk.env.backend.autotune(
                        adapter,  # pyrefly: ignore[bad-argument-type]
                        self._autotune_args,  # pyrefly: ignore[bad-argument-type]
                        force=False,
                    )
                if bk_cache is None:
                    bk_cache = {}
                    self._fusion_config_cache[bk] = bk_cache
                bk_cache[fusion_key] = cfg
        else:
            if self._autotune_args is not None:
                bk.ensure_config_exists(
                    self._autotune_args,  # pyrefly: ignore[bad-argument-type]
                )
            cfg = bk._config

        # Generate final AST with chosen config.
        assert cfg is not None, (
            "No config available: autotune_args was not provided and "
            "no prior compilation has set a config on BoundKernel"
        )
        root, source = self._build_and_unparse(cfg, rename_with_placeholders=True)

        call_order, constant_repr = self._call_order_and_constant_repr()
        call_preamble, call_args = self._build_call_args(
            call_order, constant_repr, self._prologue_source_buffers, self._extra_params
        )
        imports = get_needed_import_lines(root)

        return FinalizeCodegenResult(
            source=source,
            imports=imports,
            call_preamble=call_preamble,
            call_args=call_args,
        )

    def _make_bench_kernel_factory(
        self,
        hook_outputs: dict[str, str],
    ) -> Callable[[Config], tuple[Callable[..., object], str | None]]:
        """Build a closure that compiles the fused kernel for a given config.

        The returned closure captures ``self`` (for ``_build_and_unparse``).
        This is safe because the factory is only used synchronously within
        ``_finalize_codegen`` and goes out of scope when autotuning
        completes — no lifetime extension occurs.

        Returns ``(kernel_fn, cached_path)`` — the callable plus the
        file path of the compiled module (for diagnostic logging).
        """
        extra_tensors = self._extra_param_examples
        kernel_name = self._bound_kernel.kernel.name

        def _compile_for_config(
            config: Config,
        ) -> tuple[Callable[..., object], str | None]:
            _root, source = self._build_and_unparse(config)

            def _const(v: str) -> Callable[[], str]:
                return lambda: v

            hooks: dict[str, Callable[[], str] | None] = {
                k: _const(v) for k, v in hook_outputs.items()
            }
            finalized = PartialRender(source, hooks).finalize_all()
            # Import all active library imports unconditionally — the benchmark
            # kernel is ephemeral and never shown to users, so unused imports
            # are harmless.
            lib = _active_library_imports()
            all_imports = (
                "from __future__ import annotations\n\n"
                + "\n".join(lib.values())
                + "\n\n"
            )
            fused_code = all_imports + finalized
            module = PyCodeCache.load(fused_code)
            compiled_fn = getattr(module, kernel_name)
            return lambda *args: compiled_fn(*args, *extra_tensors), module.__file__

        return _compile_for_config

    # ------------------------------------------------------------------ #
    # TemplateBuffer overrides for multi-output layout                   #
    # ------------------------------------------------------------------ #

    def should_allocate(self) -> bool:
        return False

    def get_size(self) -> Sequence[sympy.Expr]:
        children = self._multi_output_children  # pyrefly: ignore[missing-attribute]
        if children:
            first_child = next(iter(children.values()))
            return first_child.get_size()
        return []

    def get_outputs(self) -> list[Buffer]:
        return [self, *self.mutation_outputs]

    def set_current_node(self, node: object) -> AbstractContextManager[None]:
        return nullcontext()

    def _build_call_args(
        self,
        call_order: list[str],
        constant_repr: dict[str, str],
        prologue_source_buffers: dict[str, str | None],
        extra_params: list[tuple[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Compute ``(call_preamble, call_args)`` for the kernel invocation."""
        preamble: list[str] = []
        # Sourceless prologues (e.g. ones_like) are fully inlined by the
        # prologue hook — their param should be dropped from the call args
        # and removed from the host function signature.
        sourceless_prologues: set[str] = {
            param_name
            for param_name, source in prologue_source_buffers.items()
            if source is None
        }

        def resolve_param(param_name: str) -> str | None:
            named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
            node = named_inputs.get(param_name)
            if node is None:
                result = constant_repr.get(param_name)
                if result is None and param_name not in sourceless_prologues:
                    raise RuntimeError(
                        f"param '{param_name}' not found in named_inputs or constant_repr"
                    )
                return result

            if param_name in sourceless_prologues:
                return None

            source_buf = prologue_source_buffers.get(param_name)

            if isinstance(node, ReinterpretView):
                base = source_buf if source_buf is not None else node.data.get_name()
                name = f"reinterp_{len(preamble)}"
                preamble.append(
                    f"{name} = reinterpret_tensor("
                    f"{base}, {tuple(node.get_size())}, {tuple(node.get_stride())}, {node.layout.offset})"
                )
                return name

            if source_buf is not None:
                return source_buf
            return node.get_name()  # type: ignore[union-attr]

        call_args: list[str] = [
            resolved
            for param in call_order
            if (resolved := resolve_param(param)) is not None
        ]
        call_args.extend(buf_name for _, buf_name in extra_params)
        return preamble, call_args

    @classmethod
    def create(
        cls,
        realized_inputs: dict[str, IRNode],
        structured_outputs: object,
        mutated_input_names: list[str],
        direct_aliases: dict[int, IRNode],
        *,
        on_tensor_leaf: Callable[[str, Any, list[tuple[type, int]], int], None]
        | None = None,
        fusion_enabled: bool = False,
        **buffer_kwargs: Any,  # noqa: ANN401
    ) -> tuple[HelionTemplateBuffer, tuple[TensorBox, ...]]:
        """Build a HelionTemplateBuffer and return ``(buf, outputs)``."""
        inputs = list(realized_inputs.values())
        dev = inputs[0].get_device() if inputs else torch.device("cuda")

        mutated_nodes = [
            realized_inputs[n] for n in mutated_input_names if n in realized_inputs
        ]
        mutated_inp_names = {n.get_name() for n in mutated_nodes}
        # Exclude container-flattened inputs (names with dots like "tensors.0")
        # from prologue fusion — the parameter remapping doesn't handle them.
        container_inp_names = {
            inp.get_name()  # type: ignore[union-attr]
            for param_name, inp in realized_inputs.items()
            if "." in param_name
        }
        buf = cls(
            layout=MultiOutputLayout(device=dev),  # pyrefly: ignore[bad-argument-type]
            inputs=inputs,
            mutated_inputs=mutated_nodes or None,
            allowed_prologue_inps=OrderedSet(
                inp.get_name()
                for inp in inputs  # type: ignore[union-attr]
                if inp.get_name() not in mutated_inp_names
                and inp.get_name() not in container_inp_names
            ),
            named_inputs=realized_inputs,
            **buffer_kwargs,
        )
        buf.allow_prologue_fusion = fusion_enabled
        buf.allow_epilogue_fusion = fusion_enabled
        for inp in mutated_nodes:
            V.graph.never_reuse_buffers.add(inp.get_name())

        flat, _ = (
            pytree.tree_flatten(structured_outputs)
            if structured_outputs is not None
            else ([], None)
        )
        if not any(isinstance(leaf, torch.Tensor) for leaf in flat):
            return buf, ()

        result = (
            TemplateBuffer.build_multi_outputs(  # pyrefly: ignore[missing-attribute]
                buf,
                structured_outputs,
                direct_alias_at_leaf=direct_aliases,
                on_tensor_leaf=on_tensor_leaf,
            )
        )
        return buf, result

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                   #
    # ------------------------------------------------------------------ #

    def _call_order_and_constant_repr(self) -> tuple[list[str], dict[str, str]]:
        """Compute the kernel call order and pre-repr'd non-tensor args.

        ``call_order`` lists every parameter name in signature order.
        ``constant_repr`` maps non-tensor param names to their ``repr()``-ready
        strings (scalars, defaults, and rebuilt container args) so the inherited
        ``call_kernel`` can emit them without calling back into this class.
        """
        # Both tensor inputs AND constant args must be combined before
        # _rebuild_container_args so it can pop 'param.0', 'param.1' etc.
        named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
        all_args: dict[str, object] = {
            n: _CodeExpr(inp.get_name())  # type: ignore[union-attr]
            for n, inp in named_inputs.items()
        }
        for n, v in self._constant_args_dict.items():
            if n not in all_args:
                all_args[n] = v if n == "__container_specs" else _CodeExpr(repr(v))
        _rebuild_container_args(all_args)

        tensor_flat_params = frozenset(named_inputs.keys())
        sig = self._helion_kernel.signature.parameters
        order: list[str] = []
        const_repr: dict[str, str] = {}
        for n, p in sig.items():
            if n in all_args:
                order.append(n)
                if n not in tensor_flat_params:
                    const_repr[n] = repr(all_args[n])
            elif p.default is not p.empty:
                order.append(n)
                const_repr[n] = repr(p.default)
        return order, const_repr

    # ------------------------------------------------------------------ #
    # Private Helion-specific helpers                                    #
    # ------------------------------------------------------------------ #

    def _rename_with_placeholders(self, root: ast.Module) -> None:
        """Rename functions and module-level vars with Placeholder suffixes.

        This allows the kernel to coexist with others in Inductor's output.
        """
        host_fn = self._helion_kernel.name
        inner_fn = f"_helion_{host_fn}"
        inner_fn_placeholder = f"{inner_fn}_{Placeholder.KERNEL_NAME}"

        module_level_vars: dict[str, str] = {
            target.id: f"{target.id}_{Placeholder.KERNEL_NAME}"
            for node in root.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        # Rename functions, module-level vars, and all references to them.
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == host_fn:
                    node.name = str(Placeholder.KERNEL_NAME)
                elif node.name == inner_fn:
                    node.name = inner_fn_placeholder
            elif isinstance(node, ast.Name):
                if node.id == inner_fn:
                    node.id = inner_fn_placeholder
                elif node.id in module_level_vars:
                    node.id = module_level_vars[node.id]

    def _build_and_unparse(
        self,
        config: Config,
        *,
        rename_with_placeholders: bool = False,
    ) -> tuple[ast.Module, str]:
        """Build AST for config, optionally rename for Inductor, and unparse."""
        fm = self._fusion_metadata
        assert fm is not None
        # Work on a copy so the caller's Config is not mutated with
        # normalize defaults specific to this BoundKernel's config_spec.
        config = Config(**config.config)  # pyrefly: ignore[bad-argument-type]
        self._bound_kernel.env.config_spec.normalize(config)
        extra_params = [p for p, _ in self._extra_params]
        # Prologue deduplication tracking scoped to this codegen pass.
        prologue_first_indexing: dict[str, str] = {}

        with self._bound_kernel.env:
            host_function = self._bound_kernel.host_function
            root = generate_ast(
                host_function,  # pyrefly: ignore[bad-argument-type]
                config,
                emit_repro_caller=False,
                store_transform=self._codegen_epilogue_fusion
                if fm.epilogue_idx_by_param
                else None,
                load_transform=functools.partial(
                    self._codegen_prologue_fusion,
                    prologue_first_indexing=prologue_first_indexing,
                )
                if fm.prologue_fused_params
                else None,
                extra_params=extra_params,
            )
        if rename_with_placeholders:
            self._rename_with_placeholders(root)
        source = unparse(
            root,
            output_origin_lines=self._bound_kernel.settings.output_origin_lines,
        )
        return root, source

    def _codegen_epilogue_fusion(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        value: ast.expr,
        extra_mask: ast.expr | None,
        codegen_store: Callable[..., ast.expr],
    ) -> ast.expr:
        """Emit per-epilogue index definitions + ``<STORE_OUTPUT_{i}>`` placeholder.

        For non-epilogue params, delegates to ``codegen_store`` unchanged.
        For epilogue params, emits the placeholder via ``add_statement`` and
        returns the placeholder name (dropped by ``run_node`` for 0-user nodes).

        **Single-store vs two-store mode:**

        *Single-store mode* (default): the epilogue completely replaces the
        original ``tl.store``.  The fused epilogue reads the kernel value
        from ``_kernel_val_{i}`` and writes the final result itself.
        No separate store is needed.  Example::

            # matmul result is only consumed by the epilogue (bias-add + store)
            y = matmul(a, b) + bias

        *Two-store mode* (``param_name in self._fusion_metadata.epilogue_keep_store``):
        the output buffer has downstream consumers beyond the epilogue,
        so the original ``tl.store`` is emitted *in addition to* the
        ``<STORE_OUTPUT_{i}>`` placeholder.  Example::

            # y is both epilogue-fused (for relu) and returned directly,
            # so it must be stored to memory for the second consumer
            y = matmul(a, b)
            return relu(y), y
        """
        assert self._fusion_metadata is not None
        param_name = state.device_function.tensor_arg(tensor).name
        epilogue_idx = self._fusion_metadata.epilogue_idx_by_param.get(param_name)
        if epilogue_idx is None:
            return codegen_store(state, tensor, [*subscript], value, extra_mask)

        kernel_val_name = f"_kernel_val_{epilogue_idx}"

        # 1. Assign original value to unique temp variable, upcasting to float32
        #    for narrow float types (float16, bfloat16, float8_*).  Inductor's
        #    epilogue codegen expects the kernel value in float32, so narrow
        #    floats must be widened to match.  In two-store mode the original
        #    (non-upcast) value is also needed for the keep-store.
        value_str = ast.unparse(value)
        needs_upcast = tensor.dtype.is_floating_point and tensor.dtype.itemsize < 4
        if needs_upcast:
            original_val_name = f"_kernel_output_{epilogue_idx}"
            state.add_statement(f"{original_val_name} = {value_str}")
            state.add_statement(
                f"{kernel_val_name} = {original_val_name}.to(tl.float32)"
            )
        else:
            original_val_name = kernel_val_name
            state.add_statement(f"{kernel_val_name} = {value_str}")

        # 2. Validate subscript before compute_per_dim_indexing (which has
        #    side effects on state), then emit per-dimension index definitions
        #    with per-epilogue unique names.
        assert all(s is not None for s in subscript), (
            f"Epilogue fusion does not support None (unsqueeze) indices "
            f"in store subscript for param {param_name}"
        )
        #    dim_index_exprs gives us the individual index expressions *before*
        #    stride multiplication (e.g. "(indices_0)[:, None]"), which we assign
        #    to x_epilogue{i}_{d} variables.  These names match what Inductor's
        #    store_output() sets on range tree entries.  The x_ prefix ensures
        #    get_block_shape recognizes them as XBLOCK-sized.
        per_dim = SubscriptIndexing.compute_per_dim_indexing(
            state, tensor, [*subscript], extra_mask
        )
        assert not per_dim.needs_broadcast(), (
            f"Epilogue fusion does not support broadcasted stores for param "
            f"{param_name}. Broadcast dims: {per_dim.broadcast_dims}"
        )
        for d, dim_str in enumerate(per_dim.dim_index_exprs):
            state.add_statement(f"x_epilogue{epilogue_idx}_{d} = {dim_str}")

        # 3. Emit per-epilogue mask alias (unique name avoids cross-epilogue collision).
        mask_str = ast.unparse(per_dim.mask_expr) if per_dim.has_mask() else "None"
        state.add_statement(f"_tile_mask_{epilogue_idx} = {mask_str}")

        # 4. Emit single placeholder statement.
        state.add_statement(
            ast.Expr(
                value=ast.Name(id=f"<STORE_OUTPUT_{epilogue_idx}>", ctx=ast.Load())
            )
        )

        # 5. In two-store mode the output buffer has downstream users beyond
        #    the epilogue, so also emit the original tl.store with the temp
        #    variable as the value.  In single-store mode this is skipped —
        #    the epilogue placeholder handles the store entirely.
        #
        #    Note: codegen_store internally calls SubscriptIndexing.create again
        #    for the same tensor+subscript.  This is safe because create's side
        #    effects (tensor_arg, tensor_stride, lift, allocate_reduction_dimension)
        #    are all idempotent — results are cached by DeviceFunction.  We call
        #    through codegen_store rather than building the store ourselves so
        #    the user's indexing strategy (pointer, block_ptr, tensor_descriptor)
        #    is respected.
        if param_name in self._fusion_metadata.epilogue_keep_store:
            store_val = ast.Name(id=original_val_name, ctx=ast.Load())
            state.add_statement(
                ast.Expr(
                    value=codegen_store(
                        state, tensor, [*subscript], store_val, extra_mask
                    )
                )
            )

        # The store's return value is unused (dropped by run_node for 0-user
        # nodes), so return None rather than a placeholder identifier.
        return ast.Constant(value=None)

    def _codegen_prologue_fusion(
        self,
        state: CodegenState,
        tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.expr | None,
        eviction_policy: ast.AST | None,
        codegen_load: Callable[..., ast.expr],
        *,
        prologue_first_indexing: dict[str, str],
    ) -> ast.expr:
        """Emit prologue variables + single ``<LOAD_INPUT_{param_name}>`` placeholder.

        For non-prologue params, delegates to ``codegen_load`` unchanged.
        For prologue params, emits index/mask variable definitions and a
        ``<LOAD_INPUT_{param_name}>`` placeholder (expanded at finalize time
        by the hook closure), then returns a reference to the result variable.

        ``prologue_first_indexing`` tracks which params have already been
        emitted in this codegen pass (for multi-output deduplication).
        """
        assert self._fusion_metadata is not None
        param_name = state.device_function.tensor_arg(tensor).name
        if param_name not in self._fusion_metadata.prologue_fused_params:
            return codegen_load(
                state, tensor, [*subscript], extra_mask, eviction_policy
            )

        # Read prologue variable names from kernel (set by _setup_prologue_hook).
        prologue_vars = self._fusion_metadata.prologue_vars[param_name]
        result_name = prologue_vars["result"]

        # For multi-output kernels the same input may be loaded at multiple
        # store sites.  The prologue hook is only registered once per input,
        # so only emit the placeholder + variable definitions on the first
        # encounter; subsequent references just reuse the result variable.
        # Prologue variables emitted once; reuse is safe because all loads
        # of the same fused input use the same subscript (same tile indices).
        if param_name not in prologue_first_indexing:
            xindex_name = prologue_vars["xindex"]
            xmask_name = prologue_vars["xmask"]

            # Compute linearized offset + mask from SubscriptIndexing
            indexing = SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
            offset_str = ast.unparse(indexing.index_expr)
            prologue_first_indexing[param_name] = ", ".join(indexing.dim_index_exprs)
            mask_str = (
                ast.unparse(indexing.mask_expr) if indexing.has_mask() else "True"
            )

            # Emit prologue variable definitions as AST statements (prevents DCE of
            # referenced variables like indices_0, indices_1).
            state.add_statement(f"{xindex_name} = {offset_str}")
            state.add_statement(f"{xmask_name} = {mask_str}")

            # Emit single placeholder statement (preamble + result assignment).
            state.add_statement(
                ast.Expr(
                    value=ast.Name(id=f"<LOAD_INPUT_{param_name}>", ctx=ast.Load())
                )
            )

            # Exempt param from DCE only if it has a source buffer — the
            # tensor pointer appears only in finalize_hook string substitutions.
            # Sourceless prologues (e.g. ones_like) are fully inlined by the
            # hook, so the param should be DCE'd away and also removed from
            # the host function signature.
            if param_name in self._fusion_metadata.prologue_has_source:
                state.device_function.placeholder_args.add(param_name)
            else:
                state.device_function.sourceless_prologue_params.add(param_name)
        else:
            # Validate that the second load uses the same subscript as the first.
            per_dim = SubscriptIndexing.compute_per_dim_indexing(
                state, tensor, [*subscript], extra_mask
            )
            new_offset = ", ".join(per_dim.dim_index_exprs)
            first_offset = prologue_first_indexing[param_name]
            assert new_offset == first_offset, (
                f"Prologue param {param_name} loaded with different subscripts "
                f"at multiple store sites — fusion requires identical tile "
                f"indices. First: {first_offset}, second: {new_offset}"
            )

        # Return variable reference (hook will assign fused value to this name).
        return ast.Name(id=result_name, ctx=ast.Load())


def _flatten_return_ast(
    ast_node: ast.expr | None,
    structured: object,
) -> list[ast.expr | None]:
    """Get the per-leaf AST nodes in DFS order matching build_multi_outputs traversal.

    Walks ``structured`` in the same order as ``build_multi_outputs`` to produce
    a flat list mapping ``leaf_idx`` to the corresponding AST node from the
    kernel's return statement.  Used to extract kernel parameter names
    (``ast.Name`` nodes) and detect symbolic (non-constant) non-tensor returns.
    """
    result: list[ast.expr | None] = []

    def walk(node: ast.expr | None, out: object) -> None:
        if isinstance(out, dict):
            raise AssertionError(
                "Dict outputs are not yet supported by build_multi_outputs — "
                "traversal order must be validated before enabling dict fusion"
            )
        if isinstance(out, (list, tuple)):
            elts = node.elts if isinstance(node, (ast.Tuple, ast.List)) else None
            if elts is not None and len(elts) != len(out):
                raise RuntimeError(
                    f"AST tuple/list length ({len(elts)}) != structured "
                    f"output length ({len(out)})"
                )
            for i, item in enumerate(out):
                walk(
                    elts[i] if elts is not None and i < len(elts) else None,
                    item,
                )
        else:
            result.append(node)  # leaf (tensor or non-tensor)

    walk(ast_node, structured)
    return result


@register_lowering(helion_kernel_wrapper_mutation, type_promotion_kind=None)
def lower_helion_kernel(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
) -> tuple[TensorBox, ...]:
    """Lower a Helion kernel HOP to a ``HelionTemplateBuffer``."""
    kernel = get_helion_kernel(kernel_idx)
    mutated_inputs_list = cast("list[str]", output_spec.get("mutated_inputs", []))

    # Realize inputs: convert TensorBox to buffer / ReinterpretView
    _realize = (
        TemplateBuffer.realize_template_input  # pyrefly: ignore[missing-attribute]
    )
    realized: dict[str, IRNode] = {}
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = _realize(tb)

    # Build fake tensors for kernel binding (sympy exprs to concrete ints)
    def as_int(x: object, default: int) -> int:
        return int(x) if isinstance(x, (int, sympy.Integer)) else default

    has_symbolic_shapes = any(
        not isinstance(s, (int, sympy.Integer))
        for r in realized.values()
        for s in (*r.get_size(), *r.get_stride())
    )

    if has_symbolic_shapes and kernel.settings.static_shapes:
        raise RuntimeError(
            f"Helion kernel '{kernel.fn.__name__}' has static_shapes=True but is "
            f"being compiled inside torch.compile(dynamic=True). "
            f"static_shapes=True would bake incorrect placeholder sizes into the "
            f"generated Triton code, producing wrong results at runtime. "
            f"Set static_shapes=False on the kernel, e.g.: "
            f"@helion.kernel(static_shapes=False)"
        )

    all_args: dict[str, object] = {**constant_args}
    for n, r in realized.items():
        all_args[n] = torch.empty_strided(
            [as_int(s, 64) for s in r.get_size()],
            [as_int(s, 1) for s in r.get_stride()],
            dtype=r.get_dtype(),
            device=r.get_device(),
        )
    _rebuild_container_args(all_args)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bound = kernel.bind(tuple(fake_tensors))

    # Derive output structure from bound kernel using inductor-time input layouts.
    # This gives correct strides even when inductor changes input memory layouts.
    host_function = bound.host_function
    assert host_function is not None
    flat_leaves, tree_spec, return_ast = _get_flat_output(host_function)

    if not flat_leaves:
        # No outputs — create still creates the buffer for mutations.
        buf, _ = HelionTemplateBuffer.create(
            realized_inputs=realized,
            structured_outputs=None,
            mutated_input_names=mutated_inputs_list,
            direct_aliases={},
            kernel=kernel,
            bound_kernel=bound,
            constant_args=constant_args,
            autotune_args=tuple(fake_tensors),
        )
        buf.epilogue_fusable_outputs = {}
        return ()

    # Fix flat_leaves for dynamic shapes: kernel.bind() creates concrete
    # placeholder tensors (size 64 for symbolic dims), but downstream
    # build_multi_outputs needs symbolic sizes for correct IR layouts.
    # Replace concrete leaves with FakeTensors carrying the real symbolic sizes.
    leaf_specs_list = cast("list[dict[str, object]]", output_spec.get("leaf_specs", []))
    if leaf_specs_list:
        for i, (leaf, spec) in enumerate(
            zip(flat_leaves, leaf_specs_list, strict=True)
        ):
            if not isinstance(leaf, torch.Tensor) or spec.get("type") != "tensor":
                continue
            spec_shape = cast("list[int | sympy.Expr]", spec.get("shape", []))
            if not any(not isinstance(s, int) for s in spec_shape):
                continue  # all static, no fixup needed
            spec_stride = cast("list[int | sympy.Expr]", spec.get("stride", []))
            sym_shape = convert_shape_to_symint(spec_shape)
            sym_stride = convert_shape_to_symint(spec_stride)
            with V.fake_mode:
                flat_leaves[i] = torch.empty_strided(
                    sym_shape, sym_stride, dtype=leaf.dtype, device=leaf.device
                )

    # Reconstruct structured output and create MultiOutput nodes.
    assert tree_spec is not None
    structured = pytree.tree_unflatten(flat_leaves, tree_spec)

    # Flatten return_ast to index by leaf_idx (same traversal as build_multi_outputs).
    flat_ast = _flatten_return_ast(return_ast, structured)
    assert len(flat_ast) == len(flat_leaves), (
        f"flat_ast length ({len(flat_ast)}) != flat_leaves length ({len(flat_leaves)})"
    )

    # Identify leaves with symbolic (non-constant) non-tensor returns.  These
    # outputs cannot be epilogue-fused because their AST representation is not
    # a simple parameter name.  Built as a set for per-output checking rather
    # than a global flag, so a single symbolic return doesn't disable fusion
    # for unrelated tensor outputs.
    symbolic_return_indices: set[int] = {
        i
        for i, leaf in enumerate(flat_leaves)
        if not isinstance(leaf, torch.Tensor)
        and not isinstance(flat_ast[i], ast.Constant)
    }

    # {mo_name: (kernel_param_name | None, skip_fusion)}
    output_fusion_meta: dict[str, tuple[str | None, bool]] = {}

    def on_tensor_leaf(
        mo_name: str,
        mo: MultiOutput,
        _indices: list[tuple[type, int]],
        leaf_idx: int,
    ) -> None:
        ast_node = flat_ast[leaf_idx]
        leaf = flat_leaves[leaf_idx]
        # Skip fusion for: non-tensor/non-constant returns (symbolic), and
        # tensors with dynamic sizes/strides (epilogue index expressions would
        # reference kernel-internal size variables like ks0).
        skip_fusion = leaf_idx in symbolic_return_indices or (
            isinstance(leaf, torch.Tensor) and leaf._has_symbolic_sizes_strides
        )
        param = ast_node.id if isinstance(ast_node, ast.Name) else None
        output_fusion_meta[mo_name] = (param, skip_fusion)

    buf, result = HelionTemplateBuffer.create(
        realized_inputs=realized,
        structured_outputs=structured,
        mutated_input_names=mutated_inputs_list,
        direct_aliases={
            i: realized[name]
            for i, name in cast(
                "dict[int, str]", output_spec.get("direct_aliases", {})
            ).items()
            if name in realized
        },
        on_tensor_leaf=on_tensor_leaf,
        fusion_enabled=kernel.settings.torch_compile_fusion,
        kernel=kernel,
        bound_kernel=bound,
        constant_args=constant_args,
        autotune_args=tuple(fake_tensors),
    )

    # Validate on_tensor_leaf was called during build_multi_outputs.
    # The callback captures flat_ast/flat_leaves by closure, so if
    # build_multi_outputs ever delays it, the captures would be stale.
    # Direct aliases and deduped tensors skip the callback, so we only
    # check that the callback fired when non-aliased tensor leaves exist.
    direct_alias_indices = set(
        cast("dict[int, str]", output_spec.get("direct_aliases", {})).keys()
    )
    has_non_aliased_tensors = any(
        isinstance(leaf, torch.Tensor) and i not in direct_alias_indices
        for i, leaf in enumerate(flat_leaves)
    )
    assert bool(output_fusion_meta) == has_non_aliased_tensors, (
        f"Expected on_tensor_leaf to be called for each non-aliased tensor "
        f"output during build_multi_outputs, but got "
        f"{len(output_fusion_meta)} callback(s) while "
        f"has_non_aliased_tensors={has_non_aliased_tensors}. "
        f"Direct alias indices: {direct_alias_indices}"
    )

    # Compute epilogue_fusable_outputs.  A return leaf is fusable when its AST
    # resolves to a kernel parameter name and it isn't skipped for other reasons
    # (symbolic shapes, dynamic sizes).  Non-aliased tensor returns that reach
    # on_tensor_leaf are always backed by an hl.store (direct aliases — inputs
    # returned unchanged — are filtered out by build_multi_outputs).  Derived
    # outputs like out.sum(dim=1) currently have param=None because their AST
    # isn't an ast.Name; reduction fusion may relax this in the future.
    seen_params: set[str] = set()
    epilogue_fusable_outputs: dict[str, str] = {}
    for mo_name, (param, skip_fusion) in output_fusion_meta.items():
        if (
            param is not None
            and not skip_fusion
            and param not in seen_params  # dedup: same param in multiple outputs
        ):
            epilogue_fusable_outputs[mo_name] = param
            seen_params.add(param)

    buf.epilogue_fusable_outputs = epilogue_fusable_outputs
    return result


@register_lowering(helion_kernel_wrapper_functional, type_promotion_kind=None)
def lower_helion_kernel_functional(
    *,
    kernel_idx: int,
    constant_args: dict[str, object],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, object],
    tensors_to_clone: list[str],
) -> tuple[tuple[TensorBox, ...], dict[str, TensorBox]]:
    cloned = {
        n: clone(tb) if n in tensors_to_clone and isinstance(tb, TensorBox) else tb
        for n, tb in tensor_args.items()
    }
    outputs = lower_helion_kernel(
        kernel_idx=kernel_idx,
        constant_args=constant_args,
        tensor_args=cloned,
        output_spec=output_spec,
    )
    return (outputs, {n: cloned[n] for n in tensors_to_clone if n in cloned})
