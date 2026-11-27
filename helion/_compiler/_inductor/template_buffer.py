from __future__ import annotations

import ast
from contextlib import contextmanager
from contextlib import nullcontext
import dataclasses
import hashlib
import itertools
import logging
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import ExactWeakKeyDictionary
from torch._inductor.codecache import PyCodeCache
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import Buffer
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import DtypeView
from torch._inductor.ir import FinalizeCodegenResult
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import FlexibleLayout
from torch._inductor.ir import IRNode
from torch._inductor.ir import Layout
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.ir import MutationOutput
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.ir import is_unaligned
from torch._inductor.lowering import clone
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.select_algorithm import (
    ExternalTritonTemplateKernel,  # pyrefly: ignore[missing-module-attribute]
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import GPU_ALIGN_BYTES
from torch._inductor.utils import Placeholder
from torch._inductor.utils import convert_shape_to_symint
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree
from torch.utils._sympy.symbol import SymT
from torch.utils._sympy.symbol import free_symbol_is_type

from ...language import memory_ops
from ...language.loops import static_range
from .._dynamo.higher_order_ops import _rebuild_container_args
from .._dynamo.higher_order_ops import get_helion_kernel
from .._dynamo.higher_order_ops import helion_kernel_wrapper_functional
from .._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from .._dynamo.variables import _HOST_SEMANTIC_FINGERPRINT
from .._dynamo.variables import _HOST_SEMANTIC_INPUT_NORMALIZATION
from .._dynamo.variables import _REQUIRES_ISOLATED_LOWERING
from .._dynamo.variables import _get_flat_output
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import unparse
from ..ast_read_writes import ReadWrites
from ..compile_environment import HELION_STATIC_DIMS_ATTR
from ..compile_environment import tensor_input_sources
from ..generate_ast import generate_ast
from ..indexing_strategy import SubscriptIndexing
from ..output_header import _active_library_imports
from ..type_info import CallableType
from ..type_info import SequenceType
from ..variable_origin import FlattenOrigin
from helion._compat import torch_uses_template_producer_fusion
from helion.runtime.config import Config
from helion.runtime.config_cache import ConfigCache

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterator
    from collections.abc import Sequence
    from contextlib import AbstractContextManager
    from typing import Any
    from typing import Iterable

    from torch._guards import Source
    from torch._inductor.ir import MultiOutput
    from torch._inductor.scheduler import BaseSchedulerNode

    from ..inductor_lowering import CodegenState
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


log = logging.getLogger(__name__)


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


def _input_expr(node: IRNode, base: str | None = None) -> str:
    """Wrapper-code expression of a kernel input: its buffer name, or for a
    ``ReinterpretView`` (a slice, an offset or bitcast view) the view, whose
    buffer name alone would drop its offset, strides and dtype. ``base`` (a
    prologue's source buffer) stands in for the view's buffer; bitcast views
    are never fused (see ``create``), so it keeps its own dtype. For a node
    that is not a view, ``base`` (when given) is the expression."""
    if not isinstance(node, ReinterpretView):
        return base if base is not None else node.get_name()
    if base is None:
        return node.codegen_reference()
    wrapper = V.graph.wrapper_code
    return (
        f"reinterpret_tensor({base}, "
        f"{wrapper.codegen_python_shape_tuple(node.get_size())}, "
        f"{wrapper.codegen_python_shape_tuple(node.get_stride())}, "
        f"{wrapper.codegen_sizevar(node.layout.offset)})"
    )


@dataclasses.dataclass
class _FusionMetadata:
    epilogue_idx_by_param: dict[str, int]
    epilogue_keep_store: set[str]
    prologue_vars: dict[str, dict[str, str]]
    prologue_fused_params: set[str]
    prologue_has_source: set[str]


@contextmanager
def inductor_call_facts(
    bound_kernel: BoundKernel,
    *,
    symbolic_input_layouts: bool,
    symbolic_strides: Collection[Source] = (),
) -> Iterator[None]:
    """The facts every call of ``bound_kernel``'s code holds when Inductor
    runs it (``render_inductor_ast``): inputs aligned only to
    ``GPU_ALIGN_BYTES``, and, in a graph with symbolic sizes or strides, none
    of the binding's dispatch guards, nor the traced strides of its inputs in
    ``symbolic_strides``.  Tuning a config for the graph sees them too, so it
    times the code the render emits and keeps no config the render would
    refuse.  A binding whose code reads none of these facts
    (``BoundKernel.reads_inductor_call_facts``) renders under its own."""
    env = bound_kernel.env
    assert symbolic_input_layouts or not symbolic_strides
    if not bound_kernel.reads_inductor_call_facts:
        yield
        return
    with (
        env.limit_input_alignment(GPU_ALIGN_BYTES),
        env.without_dispatch_guards(symbolic_strides)
        if symbolic_input_layouts
        else nullcontext(),
    ):
        yield


def render_inductor_ast(
    bound_kernel: BoundKernel,
    config: Config,
    *,
    symbolic_input_layouts: bool,
    symbolic_strides: Collection[Source] = (),
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
) -> ast.Module:
    """Generate ``bound_kernel``'s code for ``config`` as Inductor runs it.

    Inductor calls the code with its graph's inputs and buffers, not through
    the bound kernel's dispatch, and aligns them only to ``GPU_ALIGN_BYTES``:
    ``lower_helion_kernel``'s placeholder inputs reproduce base residues only
    modulo that, so no wider base alignment is proven from them.  A graph
    with symbolic sizes or strides (``symbolic_input_layouts``) passes every
    size it accepts, so nothing the binding's dispatch guards check holds,
    and its inputs in ``symbolic_strides`` come at any strides.
    """
    # Work on a copy so the caller's Config is not mutated with
    # normalize defaults specific to this BoundKernel's config_spec.
    config = Config(**config.config)  # pyrefly: ignore[bad-argument-type]
    env = bound_kernel.env
    env.config_spec.normalize(config)
    with (
        env,
        inductor_call_facts(
            bound_kernel,
            symbolic_input_layouts=symbolic_input_layouts,
            symbolic_strides=symbolic_strides,
        ),
    ):
        return generate_ast(
            bound_kernel.host_function,  # pyrefly: ignore[bad-argument-type]
            config,
            emit_repro_caller=False,
            store_transform=store_transform,
            load_transform=load_transform,
            extra_params=extra_params,
        )


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
        self._bench_cached_paths: ConfigCache[str] = ConfigCache()

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
        # The bound kernel's key holds the render's facts (``_call_facts``).
        return self._bound_kernel.extra_cache_key() + self._fusion_context_hash

    def supports_subprocess_benchmark(self) -> bool:
        return False


class HelionTemplateBuffer(TemplateBuffer):
    """Inductor template buffer for Helion kernel."""

    _fusion_config_cache: ExactWeakKeyDictionary = ExactWeakKeyDictionary()
    # Configs tuned on a binding of its own (``_tuning_kernel``) for a cached
    # binding, by the render's facts, so its later lowerings tune once.
    _tuned_config_cache: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        *,
        kernel: Kernel,
        bound_kernel: BoundKernel,
        constant_args: dict[str, object],
        autotune_args: tuple[object, ...] | None = None,
        symbolic_input_layouts: bool = False,
        symbolic_strides: Collection[Source] = (),
        mutated_inputs: Iterable[IRNode] | None = None,
        allowed_prologue_inps: OrderedSet[str] | None = None,
        named_inputs: dict[str, IRNode] | None = None,
        stand_in_fields: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self._helion_kernel = kernel
        self._bound_kernel = bound_kernel
        self._constant_args_dict = constant_args
        self._autotune_args = autotune_args
        # The graph passes inputs of dynamic sizes or strides, which
        # ``lower_helion_kernel`` bound to placeholders: the code serves every
        # size and stride the graph accepts, never checking the binding's
        # cache key.  The inputs in ``symbolic_strides`` come at any strides.
        self._symbolic_input_layouts = symbolic_input_layouts
        self._symbolic_strides = frozenset(symbolic_strides)
        self._tuning_bound_kernel: BoundKernel | None = None
        self._fusion_metadata: _FusionMetadata | None = None
        # What the generated code reads, beyond the device loads a fused
        # prologue replaces, of the parameters each input buffer is passed
        # for: what a prologue source must match to be passed in its place
        # (see ``_reads_besides_loads``).
        self._stand_in_fields = stand_in_fields or {}

        tb_self = self  # capture for closure

        def _make_kernel_render(
            out_node: TemplateBuffer, hint_override: object = None
        ) -> tuple[object, Callable[[], PartialRender]]:
            kernel = ExternalTritonTemplateKernel(out_node)

            def render() -> PartialRender:
                return tb_self._render_with_hooks(kernel)

            return kernel, render

        prologue_input_key = (
            "load_input_fusion_allowed_inputs"
            if torch_uses_template_producer_fusion()
            else "allowed_prologue_inps"
        )
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=_make_kernel_render,
            mutated_inputs=mutated_inputs,
            named_inputs=named_inputs,  # pyrefly: ignore[unexpected-keyword]
            **{prologue_input_key: allowed_prologue_inps},
        )

    @staticmethod
    def _refresh_range_tree_node_symbols(kernel: Any) -> None:  # noqa: ANN401
        """Keep Inductor range-tree lookups in sync after symbol renames."""

        def refresh(nodes: dict[sympy.Symbol, Any]) -> None:
            for old_symbol, entry in list(nodes.items()):
                new_symbol = entry.symbol()
                if old_symbol != new_symbol and old_symbol in entry.parent.var_list:
                    entry.parent.var_list[entry.parent.var_list.index(old_symbol)] = (
                        new_symbol
                    )
                if old_symbol in entry.parent.var_ranges:
                    entry.parent.var_ranges.setdefault(
                        new_symbol, entry.parent.var_ranges[old_symbol]
                    )
                for symbol in (
                    new_symbol,
                    sympy.Symbol(str(new_symbol), integer=True),
                    sympy.Symbol(str(new_symbol)),
                ):
                    nodes[symbol] = entry

        refresh(kernel.range_tree_nodes)
        for subgraph in kernel.subgraph_bodies.values():
            if subgraph.range_tree_nodes is not None:
                refresh(subgraph.range_tree_nodes)

    def _render_with_hooks(self, kernel: Any) -> PartialRender:  # noqa: ANN401
        """Set up fusion hooks, read metadata, return a placeholder PartialRender.

        Called as the ``render()`` function from the standard
        ``codegen_template_body`` path.  Always returns a placeholder —
        autotuning and AST generation are deferred to
        ``_finalize_codegen`` (called after Inductor has
        processed epilogue/prologue subgraphs).
        """
        # 1. Set up fusion hooks (requires V.kernel context).  The hooks pass
        #    the first source of each prologue for every parameter the buffer
        #    is passed for and the rest as extra inputs.  Inductor collects the
        #    sources in a frozenset, so order them (``_stand_ins_first``): one
        #    the generated code can use in the parameters' place first, so the
        #    choice does not follow the string hash seed.  A source an earlier
        #    kernel mutated is read under its MutationOutput's name, which
        #    names no tensor: pass the mutated buffer, as Inductor's
        #    ``KernelArgs.input`` does.
        real_name = V.graph.scheduler.mutation_real_name
        kernel._prologue_sources = {
            buf_name: _stand_ins_first(
                OrderedSet(real_name.get(s, s) for s in sources),
                V.graph.get_buffer(buf_name),
                self._stand_in_fields.get(buf_name, ()),
            )
            for buf_name, sources in kernel._prologue_sources.items()
        }
        kernel._setup_fusion_hooks()
        named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
        for param_name, source in kernel._prologue_source_buffers.items():
            buf_name = named_inputs[param_name].get_name()
            fields = self._stand_in_fields.get(buf_name)
            if fields and (
                source is None
                or not _can_stand_in_for(source, V.graph.get_buffer(buf_name), fields)
            ):
                raise RuntimeError(
                    f"Helion kernel {self._helion_kernel.name!r}: the prologue "
                    f"fused into parameter {param_name!r} has no source buffer "
                    "the host code can use in its place"
                )
        self._refresh_range_tree_node_symbols(kernel)

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

    def _call_facts(self, bound_kernel: BoundKernel) -> AbstractContextManager[None]:
        """``inductor_call_facts`` of this buffer's render, for ``bound_kernel``."""
        return inductor_call_facts(
            bound_kernel,
            symbolic_input_layouts=self._symbolic_input_layouts,
            symbolic_strides=self._symbolic_strides,
        )

    def _tunes_on_own_binding(self) -> bool:
        """Whether the lowering tunes through a binding of its own
        (``_tuning_kernel``): its binding is the kernel's cached one, which
        also serves eager calls, and its code reads facts the render's calls
        may not hold (``BoundKernel.reads_inductor_call_facts``).  Another
        binding is tuned itself, for the render and eager calls alike."""
        bk = self._bound_kernel
        return bk._cache_managed and bk.reads_inductor_call_facts

    def _tuning_kernel(self) -> BoundKernel:
        """The bound kernel the lowering tunes on, under ``_call_facts``.

        Tuning compiles its candidates into a bound kernel's compile cache,
        which is keyed by the config alone, and sets the kernel's config and
        run function.  The binding ``lower_helion_kernel`` takes from the
        kernel's cache also serves eager calls: they would run code compiled
        under the render's facts, which are weaker than theirs, and the tuner
        would time code an eager call compiled there instead of the render's.
        So where the binding's code reads those facts, a cached binding tunes
        through a binding of its own (``_tunes_on_own_binding``)."""
        bk = self._bound_kernel
        if not self._tunes_on_own_binding():
            return bk
        if self._tuning_bound_kernel is None:
            assert self._autotune_args is not None
            self._tuning_bound_kernel = self._helion_kernel._bind_isolated(
                self._autotune_args
            )
        return self._tuning_bound_kernel

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
        # A config tuned under the render's facts is cached by them, on disk
        # too (``BoundKernel.extra_cache_key``): one tuned for another render,
        # or for the binding's own calls, may be one this render refuses.  A
        # binding whose code reads none of them keys by none.
        facts = (
            (self._symbolic_input_layouts, self._symbolic_strides)
            if bk.reads_inductor_call_facts
            else ()
        )
        if use_fusion_autotune:
            fusion_key = (tuple(sorted(hook_outputs.items())), facts)
            bk_cache = self._fusion_config_cache.get(bk)
            if bk_cache is not None and fusion_key in bk_cache:
                cfg = bk_cache[fusion_key]
            else:
                # A config the user gave holds whatever the facts, so it binds
                # no tuner.
                cfg = bk._user_provided_config()
                if cfg is None:
                    tuner = self._tuning_kernel()
                    with self._call_facts(tuner):
                        fusion_context_hash = hashlib.sha256(
                            repr(fusion_key[0]).encode()
                        ).hexdigest()
                        adapter = _FusionAutotuneAdapter(
                            tuner,
                            self._make_bench_kernel_factory(hook_outputs),
                            fusion_context_hash,
                        )
                        cfg = tuner.env.backend.autotune(
                            adapter,  # pyrefly: ignore[bad-argument-type]
                            self._autotune_args,  # pyrefly: ignore[bad-argument-type]
                            force=False,
                        )
                if bk_cache is None:
                    bk_cache = {}
                    self._fusion_config_cache[bk] = bk_cache
                bk_cache[fusion_key] = cfg
        else:
            # A config the binding already has (an eager call tuned or set it)
            # stays, as before, though the render may refuse it.
            cfg = bk._config
            tuned = self._tuned_config_cache.get(bk)
            if cfg is None and tuned is not None:
                cfg = tuned.get(facts)
            if cfg is None and self._autotune_args is not None:
                # A config the user gave holds whatever the facts, so it
                # binds no tuner.
                cfg = (
                    bk._user_provided_config() if self._tunes_on_own_binding() else None
                )
                if cfg is None:
                    tuner = self._tuning_kernel()
                    with self._call_facts(tuner):
                        tuner.ensure_config_exists(
                            self._autotune_args,  # pyrefly: ignore[bad-argument-type]
                        )
                    cfg = tuner._config
                if self._tunes_on_own_binding():
                    if tuned is None:
                        tuned = {}
                        self._tuned_config_cache[bk] = tuned
                    tuned[facts] = cfg

        # Generate final AST with chosen config.
        assert cfg is not None, (
            "No config available: autotune_args was not provided and "
            "no prior compilation has set a config on BoundKernel"
        )
        root, source = self._build_and_unparse(cfg, rename_with_placeholders=True)

        host_def = _function_def(root, str(Placeholder.KERNEL_NAME))
        call_order, constant_repr = self._call_order_and_constant_repr(
            ReadWrites.from_list(host_def.body).reads
        )
        call_preamble, call_args = self._build_call_args(
            call_order,
            constant_repr,
            self._prologue_source_buffers,
            self._extra_params,
            host_params=_param_names(host_def),
        )

        return FinalizeCodegenResult(
            source=source,
            # generate_ast already put the backend's imports at the top of
            # the source, which Inductor splices into the wrapper whole.
            imports=[],
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

            def kernel_fn(*args: object) -> object:
                return compiled_fn(*args, *extra_tensors)

            # The timers read how many kernels a call launches from the host
            # function (benchmarking._launches_several_kernels), which this
            # wrapper hides.
            launches = getattr(compiled_fn, "_helion_cute_launches", None)
            if launches is not None:
                kernel_fn._helion_cute_launches = launches  # pyrefly: ignore[missing-attribute]
            return kernel_fn, module.__file__

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

    def has_aliasing_or_mutation_for_prologue_fusion(
        self,
        scheduler_node: object,
    ) -> bool:
        """Return the Inductor fusion-blocking alias/mutation state.

        Inductor's prologue-fusion gate asks templates this as a whole-template
        question before checking ``allowed_prologue_inps``.  Helion mutating
        templates can still safely fuse producers for independent inputs, so
        ignore only safe synthetic MutationOutput buffers here.  The
        MutationOutputs remain in the scheduler graph for dependencies and
        memory planning.
        """
        # cast (type-checker only) to the fuller scheduler-node interface the body
        # needs; no runtime isinstance, so structurally-compatible nodes (real
        # Inductor nodes and test fakes) all work.
        node = cast("BaseSchedulerNode", scheduler_node)
        mutation_names: set[str] = set()
        for output in node.get_outputs():
            if output.get_aliases():
                return True

            mutations = output.get_mutations()
            if not mutations:
                continue
            if not isinstance(output.node, MutationOutput):
                return True
            mutation_names.update(mutations)

        if not mutation_names:
            return False

        allowed_prologue_inps = self.get_allowed_prologue_inps()
        if any(name in allowed_prologue_inps for name in mutation_names):
            return True

        for inp in cast("Sequence[IRNode]", self.inputs):
            if (
                inp.get_name() in allowed_prologue_inps
                and not mutation_names.isdisjoint(inp.get_read_names())
            ):
                return True
        return False

    def get_allowed_prologue_inps(self) -> OrderedSet[str]:
        """Expose a single accessor across the PyTorch template API rename."""
        if torch_uses_template_producer_fusion():
            return self.load_input_fusion_allowed_inputs  # pyrefly: ignore[missing-attribute]
        return super().get_allowed_prologue_inps()  # pyrefly: ignore[missing-attribute]

    def has_aliasing_or_mutation_for_producer_fusion(
        self, scheduler_node: object
    ) -> bool:
        # Keep both scheduler entrypoints on Helion's mutation-safety policy.
        return self.has_aliasing_or_mutation_for_prologue_fusion(scheduler_node)

    def _build_call_args(
        self,
        call_order: list[str],
        constant_repr: dict[str, str],
        prologue_source_buffers: dict[str, str | None],
        extra_params: list[tuple[str, str]],
        host_params: Collection[str],
    ) -> tuple[list[str], list[str]]:
        """Compute ``(call_preamble, call_args)`` for the kernel invocation.

        ``host_params`` are the generated host function's parameters.
        """
        preamble: list[str] = []
        # Sourceless prologues (e.g. ones_like) are fully inlined by the
        # prologue hook — their param is dropped from the call args and from
        # the host function signature.  One whose device load turned out dead
        # never reached the hook and kept its parameter; it reads nothing else
        # (see ``_reads_besides_loads``), so it is passed ``None``.
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
                return "None" if param_name in host_params else None

            source_buf = prologue_source_buffers.get(param_name)

            if isinstance(node, ReinterpretView):
                name = f"reinterp_{len(preamble)}"
                preamble.append(f"{name} = {_input_expr(node, source_buf)}")
                return name
            return _input_expr(node, source_buf)

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
        bound_kernel: BoundKernel,
        on_tensor_leaf: Callable[[str, Any, list[tuple[type, int]], int], None]
        | None = None,
        fusion_enabled: bool = False,
        pre_realize_bitcast_names: frozenset[str] = frozenset(),
        **buffer_kwargs: Any,  # noqa: ANN401
    ) -> tuple[HelionTemplateBuffer, tuple[TensorBox, ...]]:
        """Build a HelionTemplateBuffer and return ``(buf, outputs)``.
        ``pre_realize_bitcast_names`` names inputs whose view chain held a
        bitcast before realization (see ``lower_helion_kernel``); they are
        kept unfused like a dtype-changing ReinterpretView."""
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
        mutation_dependent_inp_names = {
            inp.get_name()  # type: ignore[union-attr]
            for inp in inputs  # type: ignore[union-attr]
            if inp.get_read_names() & mutated_inp_names
        }
        # A fused producer would run on its source buffer, which the kernel
        # would then read through the bitcast view's dtype: keep bitcast views
        # unfused.
        bitcast_inp_names = {
            inp.get_name()
            for inp in inputs
            if isinstance(inp, ReinterpretView)
            and inp.layout.dtype != inp.data.get_dtype()
        } | pre_realize_bitcast_names
        size_indexed_inp_names = {
            inp.get_name()
            for inp in inputs
            if isinstance(inp, ComputedBuffer)
            and any(
                isinstance(dep, MemoryDep) and _read_offset_uses_size_vars(dep)
                for dep in inp.get_read_writes().reads
            )
        }
        # A producer can replace a parameter whose size, stride, dtype, device
        # or values the generated code reads only when one of its sources can
        # be passed in its place (see ``_reads_besides_loads``).
        # One source stands in for every parameter a buffer is passed for.
        stand_in_fields: dict[str, frozenset[str]] = {}
        if fusion_enabled:
            reads = _reads_besides_loads(bound_kernel, realized_inputs.keys())
            for param_name, fields in reads.items():
                buf_name = realized_inputs[param_name].get_name()
                stand_in_fields[buf_name] = (
                    stand_in_fields.get(buf_name, frozenset()) | fields
                )
        # Checked only for buffers no other rule keeps unfused, as the check
        # may freeze a source's layout.
        unfused_inp_names = (
            mutated_inp_names
            | container_inp_names
            | bitcast_inp_names
            | mutation_dependent_inp_names
            | size_indexed_inp_names
        )
        no_stand_in_inp_names = {
            buf_name
            for buf_name, fields in stand_in_fields.items()
            if buf_name not in unfused_inp_names
            and isinstance(producer := V.graph.get_buffer(buf_name), ComputedBuffer)
            and not _ensure_stand_in(producer, fields)
        }
        buf = cls(
            layout=MultiOutputLayout(device=dev),  # pyrefly: ignore[bad-argument-type]
            inputs=inputs,
            mutated_inputs=mutated_nodes or None,
            allowed_prologue_inps=OrderedSet(
                inp.get_name()
                for inp in inputs  # type: ignore[union-attr]
                if inp.get_name() not in unfused_inp_names
                and inp.get_name() not in no_stand_in_inp_names
            ),
            named_inputs=realized_inputs,
            bound_kernel=bound_kernel,
            stand_in_fields=stand_in_fields,
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

        result = TemplateBuffer.build_multi_outputs(  # pyrefly: ignore[missing-attribute]
            buf,
            structured_outputs,
            direct_alias_at_leaf=direct_aliases,
            on_tensor_leaf=on_tensor_leaf,
        )
        return buf, result

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                   #
    # ------------------------------------------------------------------ #

    def _call_order_and_constant_repr(
        self, host_reads: Collection[str]
    ) -> tuple[list[str], dict[str, str]]:
        """Compute the kernel call order and pre-repr'd non-tensor args.

        ``call_order`` lists every parameter name in signature order.
        ``constant_repr`` maps non-tensor param names to their ``repr()``-ready
        strings (scalars, defaults, and rebuilt container args) so the inherited
        ``call_kernel`` can emit them without calling back into this class.
        ``host_reads`` are the names the generated host function's body reads.
        """
        # Both tensor inputs AND constant args must be combined before
        # _rebuild_container_args so it can pop 'param.0', 'param.1' etc.
        named_inputs = self._named_inputs  # pyrefly: ignore[missing-attribute]
        all_args: dict[str, object] = {
            n: _CodeExpr(_input_expr(inp)) for n, inp in named_inputs.items()
        }
        for n, v in self._constant_args_dict.items():
            if n in all_args:
                continue
            if n == "__container_specs":
                all_args[n] = v
            elif _is_traced_callable(v):
                # The kernel traces a callable argument (a function, an
                # fx.GraphModule) into its device code, so the host function
                # does not read it and the call passes None: a callable's repr
                # is not Python source.
                param = n.partition(".")[0]
                if param in host_reads:
                    raise RuntimeError(
                        f"Helion kernel {self._helion_kernel.name!r} reads its "
                        f"callable argument {param!r} in host code; torch.compile "
                        "can only lower a callable argument used inside device "
                        "loops (hl.tile / hl.grid)"
                    )
                all_args[n] = _CodeExpr("None")
            else:
                all_args[n] = _CodeExpr(repr(v))
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
        root = render_inductor_ast(
            self._bound_kernel,
            config,
            symbolic_input_layouts=self._symbolic_input_layouts,
            symbolic_strides=self._symbolic_strides,
            store_transform=self._codegen_epilogue_fusion
            if fm.epilogue_idx_by_param
            else None,
            load_transform=self._codegen_prologue_fusion
            if fm.prologue_fused_params
            else None,
            extra_params=[p for p, _ in self._extra_params],
        )
        _strip_host_annotations(root, self._helion_kernel.name)
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
        cache_modifier: ast.expr | None,
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
            return codegen_store(
                state, tensor, [*subscript], value, extra_mask, cache_modifier
            )

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
            state, tensor, [*subscript], extra_mask, for_store=True
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
                        state,
                        tensor,
                        [*subscript],
                        store_val,
                        extra_mask,
                        cache_modifier,
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
        cache_modifier: ast.AST | None,
        codegen_load: Callable[..., ast.expr],
    ) -> ast.expr:
        """Emit prologue variables + a ``<LOAD_INPUT_{param_name}>`` placeholder.

        For non-prologue params, delegates to ``codegen_load`` unchanged.
        For prologue params, emits index/mask variable definitions and a
        ``<LOAD_INPUT_{param_name}>`` placeholder (expanded at finalize time
        by the hook closure, at every site it is emitted at) where no earlier
        emission is in scope, then returns a copy of the result variable.

        ``state.codegen.prologue_first_indexing`` and ``prologue_blocks``
        track what this codegen pass has emitted. They belong to the pass: a
        kernel ``generate_ast`` regenerates after a rejected first pass emits
        every placeholder again.
        """
        assert self._fusion_metadata is not None
        param_name = state.device_function.tensor_arg(tensor).name
        if param_name not in self._fusion_metadata.prologue_fused_params:
            return codegen_load(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                cache_modifier,
            )

        # Read prologue variable names from kernel (set by _setup_prologue_hook).
        prologue_vars = self._fusion_metadata.prologue_vars[param_name]
        result_name = prologue_vars["result"]

        # The same input may be loaded at several sites: the store sites of a
        # multi-output kernel, or the copies of one load that reduction loops
        # emit, one per loop. The placeholder expands at every site it is
        # emitted at (Inductor replaces each occurrence), so it is emitted
        # again wherever no earlier emission is in scope; a site inside the
        # block of an earlier one reuses its result variable. All sites use
        # the same subscript (``load_form`` keeps inputs loaded in two forms
        # unfused).
        codegen = state.codegen
        prologue_first_indexing = codegen.prologue_first_indexing
        blocks = codegen.prologue_blocks.setdefault(param_name, [])
        in_scope = any(
            block is enclosing
            for block in blocks
            for enclosing in codegen.statements_stack
        )
        if param_name not in prologue_first_indexing:
            # Exempt param from DCE only if it has a source buffer — the
            # tensor pointer appears only in finalize_hook string substitutions.
            # Sourceless prologues (e.g. ones_like) are fully inlined by the
            # hook, so the param should be DCE'd away and also removed from
            # the host function signature.
            if param_name in self._fusion_metadata.prologue_has_source:
                state.device_function.placeholder_args.add(param_name)
            else:
                state.device_function.sourceless_prologue_params.add(param_name)
        # The linearized offset + mask where the placeholder is emitted;
        # elsewhere only the per-dimension offsets, to check them.
        indexing = (
            None
            if in_scope
            else SubscriptIndexing.create(state, tensor, [*subscript], extra_mask)
        )
        dims = (
            SubscriptIndexing.compute_per_dim_indexing(
                state, tensor, [*subscript], extra_mask
            )
            if indexing is None
            else indexing
        ).dim_index_exprs
        offset = ", ".join(dims)
        first_offset = prologue_first_indexing.setdefault(param_name, offset)
        assert offset == first_offset, (
            f"Prologue param {param_name} loaded with different subscripts "
            f"at several sites — fusion requires identical tile indices. "
            f"First: {first_offset}, second: {offset}"
        )
        if indexing is not None:
            mask_str = (
                ast.unparse(indexing.mask_expr) if indexing.has_mask() else "True"
            )
            # Emit prologue variable definitions as AST statements (prevents DCE of
            # referenced variables like indices_0, indices_1).
            state.add_statement(
                f"{prologue_vars['xindex']} = {ast.unparse(indexing.index_expr)}"
            )
            state.add_statement(f"{prologue_vars['xmask']} = {mask_str}")
            # Emit single placeholder statement (preamble + result assignment).
            state.add_statement(
                ast.Expr(
                    value=ast.Name(id=f"<LOAD_INPUT_{param_name}>", ctx=ast.Load())
                )
            )
            blocks.append(codegen.statements_stack[-1])

        # A copy of the result (which the hook assigns): code that rebinds the
        # loaded value (``a = x[tile]; a = a + 1``) must not rebind the result
        # a later load of the input reuses.
        loaded = state.device_function.new_var(f"{result_name}_load", dce=True)
        state.add_statement(f"{loaded} = {result_name}")
        return ast.Name(id=loaded, ctx=ast.Load())


def _function_def(root: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in root.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _param_names(function: ast.FunctionDef) -> list[str]:
    """The parameters of ``function`` in positional order, keyword-only last."""
    args = function.args
    return [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]


def _strip_host_annotations(root: ast.Module, host_name: str) -> None:
    """Drop the parameter annotations of the generated host function.

    Inductor's output module imports only the libraries the generated code
    reads, so an annotation naming anything else (``Callable``, a ``Tensor``
    imported ``from torch``, a user type) raises ``NameError`` when the module
    loads; annotations carry nothing at runtime.  The ``ast.arg`` nodes are
    shared with the bound kernel's host function, so they are copied.
    """
    args = _function_def(root, host_name).args

    def strip(arg: ast.arg | None) -> ast.arg | None:
        if arg is None or arg.annotation is None:
            return arg
        return cast("ast.arg", cast("ExtendedAST", arg).copy(annotation=None))

    args.posonlyargs = [cast("ast.arg", strip(a)) for a in args.posonlyargs]
    args.args = [cast("ast.arg", strip(a)) for a in args.args]
    args.kwonlyargs = [cast("ast.arg", strip(a)) for a in args.kwonlyargs]
    args.vararg = strip(args.vararg)
    args.kwarg = strip(args.kwarg)


def _is_traced_callable(value: object) -> bool:
    """Whether a constant kernel argument is a callable the kernel traces."""
    return callable(value) and not isinstance(value, type)


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


def _bind_kernel_for_lowering(kernel: Kernel, args: tuple[object, ...]) -> BoundKernel:
    """Reuse eager state without waiting for a concurrent eager bind."""
    # Inductor owns Dynamo's compile lock here. Eager binding can hold
    # _bind_lock while waiting for that lock, so blocking would form an ABBA
    # deadlock. Contention is rare; keep the fallback private and diagnosable.
    if not kernel._bind_lock.acquire(blocking=False):
        log.debug(
            "kernel bind lock busy while lowering %s; using an isolated bind",
            kernel.name,
        )
        return kernel._bind_isolated(args)
    try:
        return kernel._bind(args)
    finally:
        kernel._bind_lock.release()


def _read_offset_uses_size_vars(dep: MemoryDep) -> bool:
    """Whether a producer's read differs from its own contiguous iteration by
    a size-variable term (a dynamic slice's offset, or a symbolic stride).

    A fused prologue renders that term as a size variable (``ks0``) which the
    Helion kernel does not define, so such a producer stays unfused.
    """
    flat = sympy.Add(
        *(sympy.Mul(var, *dep.size[i + 1 :]) for i, var in enumerate(dep.var_names))
    )
    return free_symbol_is_type(
        sympy.expand(sympy.Add(dep.index, sympy.Mul(-1, flat))),
        (SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.UNBACKED_INT),
    )


# What the generated code can read of a kernel's tensor argument, beyond the
# device loads a fused prologue replaces (see ``_reads_besides_loads``).
_SIZE = "size"
_STRIDE = "stride"
_DTYPE = "dtype"
_DEVICE = "device"
_VALUES = "values"
# Named at all (``x is None``): any source will do, but not a dropped parameter.
_REFERENCED = "referenced"
_LAYOUT_FIELDS = frozenset({_SIZE, _STRIDE, _DTYPE, _DEVICE})
_METADATA_ATTRIBUTES = {
    "shape": frozenset({_SIZE}),
    "ndim": frozenset({_SIZE}),
    "dtype": frozenset({_DTYPE}),
    "itemsize": frozenset({_DTYPE}),
    "device": frozenset({_DEVICE}),
    "is_cuda": frozenset({_DEVICE}),
}
_METADATA_METHODS = {
    "size": frozenset({_SIZE}),
    "dim": frozenset({_SIZE}),
    "numel": frozenset({_SIZE}),
    "stride": frozenset({_STRIDE}),
    "is_contiguous": frozenset({_SIZE, _STRIDE}),
    "element_size": frozenset({_DTYPE}),
    "new_empty": frozenset({_DTYPE, _DEVICE}),
    "new_zeros": frozenset({_DTYPE, _DEVICE}),
    "new_ones": frozenset({_DTYPE, _DEVICE}),
    "new_full": frozenset({_DTYPE, _DEVICE}),
}
_LIKE_FACTORIES = (
    torch.empty_like,
    torch.zeros_like,
    torch.ones_like,
    torch.full_like,
)


def _reads_besides_loads(
    bound_kernel: BoundKernel, params: Iterable[str]
) -> dict[str, frozenset[str]]:
    """What the generated code reads of each of ``params`` other than through
    the device loads (``x[tile]``, ``hl.load(x, ...)``) a fused prologue
    replaces: the size, stride, dtype, device, or values of the tensor.

    A fused prologue's output buffer is never allocated: the kernel call passes
    one of the prologue's source buffers in its place, or drops the parameter
    when the prologue reads none, and everything but those device loads sees
    the substituted buffer (``torch.empty_like(x)``, ``x.size(0)``,
    ``x.clone()``, the strides a dynamic-shape kernel indexes with).  Dtypes,
    devices and ``x is None`` in device code are compile-time constants, and
    under static shapes so are sizes and strides in device code and grid loop
    bounds.  A parameter without a device load, or whose device code
    generation reads (``device=x.device`` for a host factory call,
    ``get_num_sm(x.device)`` for a persistent kernel), must at least stay in
    the kernel call, which a prologue that reads no buffer does not allow.
    """
    host_function = bound_kernel.host_function
    assert host_function is not None
    params = set(params)
    static_shapes = bound_kernel.settings.static_shapes
    reads: dict[str, set[str]] = {name: set() for name in params}
    # The forms (see ``load_form``) each parameter is loaded in.
    loaded: dict[str, set[tuple[object, ...]]] = {}
    visits = itertools.count()
    # The tensors whose device code generation may read, whatever the config.
    origin = host_function.first_tensor_origin(bound_kernel.env.device)
    for name in (
        host_function.first_tensor_param(),
        None if origin is None else origin.host_str(),
    ):
        if name in reads:
            reads[name].add(_DEVICE)

    def static_value(node: ast.AST) -> object:
        """The callable or class ``node`` names, if any."""
        node_type = cast("ExtendedAST", node)._type_info
        return node_type.value if isinstance(node_type, CallableType) else None

    def callee(node: ast.Call) -> object:
        return static_value(node.func)

    def param_name(node: ast.AST) -> str | None:
        return node.id if isinstance(node, ast.Name) and node.id in params else None

    def metadata_read(node: ast.AST) -> tuple[str, frozenset[str], ast.AST] | None:
        """``(param, fields, child holding the param)`` for a metadata read."""
        if isinstance(node, ast.Attribute) and node.attr in _METADATA_ATTRIBUTES:
            if (name := param_name(node.value)) is not None:
                return name, _METADATA_ATTRIBUTES[node.attr], node.value
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _METADATA_METHODS
                and (name := param_name(func.value)) is not None
            ):
                return name, _METADATA_METHODS[func.attr], func
            if node.args and (name := param_name(node.args[0])) is not None:
                fn = callee(node)
                if fn in _LIKE_FACTORIES:
                    return name, _LAYOUT_FIELDS, node.args[0]
                if fn is len:
                    return name, frozenset({_SIZE}), node.args[0]
                # Any source is a tensor; other type checks (nn.Parameter)
                # may tell a source from the producer, so they read values.
                if (
                    fn is isinstance
                    and len(node.args) == 2
                    and static_value(node.args[1]) is torch.Tensor
                ):
                    return name, frozenset({_REFERENCED}), node.args[0]
        if (
            isinstance(node, ast.Compare)
            and all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
            and all(
                isinstance(c, ast.Constant) and c.value is None
                for c in node.comparators
            )
            and (name := param_name(node.left)) is not None
        ):
            return name, frozenset({_REFERENCED}), node.left
        return None

    def loaded_param(node: ast.AST) -> ast.AST | None:
        """The parameter ``Name`` a device load reads, if any."""
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            base = node.value
        elif (
            isinstance(node, ast.Call) and node.args and callee(node) is memory_ops.load
        ):
            base = node.args[0]
        else:
            return None
        return base if param_name(base) is not None else None

    def load_form(
        node: ast.AST, block: tuple[object, ...], targets: frozenset[str]
    ) -> tuple[object, ...]:
        """The form of a device load: its index and enclosing blocks (loop
        bodies, ``if`` and ``else`` branches, outermost first) when the index
        names only enclosing loop variables, constants and constant slices
        (``x[i, j]`` and ``hl.load(x, [i, j])`` alike), else the load itself,
        as a mask or a computed index may differ between loads written the
        same way (see ``form_count``)."""

        def plain(element: ast.AST | None) -> bool:
            if isinstance(element, ast.Slice):
                parts = (element.lower, element.upper, element.step)
                return all(p is None or isinstance(p, ast.Constant) for p in parts)
            return isinstance(element, ast.Constant) or (
                isinstance(element, ast.Name) and element.id in targets
            )

        if isinstance(node, ast.Subscript):
            index, rest = node.slice, []
        else:
            call = cast("ast.Call", node)
            index = call.args[1] if len(call.args) > 1 else None
            rest = [*call.args[2:], *call.keywords]
        elements = index.elts if isinstance(index, (ast.Tuple, ast.List)) else [index]
        if rest or not all(plain(e) for e in elements):
            # One per visit: an unrolled tuple loop repeats its body's nodes.
            return ("load", next(visits))
        return ("plain", block, tuple(ast.dump(cast("ast.AST", e)) for e in elements))

    def form_count(forms: set[tuple[object, ...]]) -> int:
        """How many forms of load ``forms`` are: loads at one index are one
        when one of them is in a block around all the others' (the generated
        code reuses that load's prologue in the inner blocks, or emits it
        again where it comes first); any other load is its own."""
        chains_by_index: dict[object, list[tuple[object, ...]]] = {}
        count = 0
        for form in forms:
            if form[0] != "plain":
                count += 1
                continue
            chains_by_index.setdefault(form[2], []).append(
                cast("tuple[object, ...]", form[1])
            )
        for chains in chains_by_index.values():
            enclosed = any(
                all(chain[: len(outer)] == outer for chain in chains)
                for outer in chains
            )
            count += 1 if enclosed else 2
        return count

    def visit(
        node: ast.AST,
        on_device: bool,
        constant_shapes: bool,
        block: tuple[object, ...] = (),
        targets: frozenset[str] = frozenset(),
        unrolled: frozenset[str] = frozenset(),
    ) -> None:
        if isinstance(node, ast.For):
            grid = cast("ExtendedAST", node)._loop_type is LoopType.GRID
            iter_shapes = static_shapes if grid else constant_shapes
            visit(node.target, on_device, constant_shapes, block, targets, unrolled)
            visit(node.iter, on_device, iter_shapes, block, targets, unrolled)
            names = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            # hl.static_range, and a loop over a tuple the device IR unrolls
            # (``cols = (1, 2); for c in cols``), unroll into one block, where
            # a load indexed by the variable is a load per iteration at
            # another index.
            if (
                isinstance(node.iter, ast.Call) and callee(node.iter) is static_range
            ) or isinstance(cast("ExtendedAST", node.iter)._type_info, SequenceType):
                inner, inner_unrolled = targets, unrolled | names
            else:
                inner, inner_unrolled = targets | names, unrolled
            for i, stmts in enumerate((node.body, node.orelse)):
                for stmt in stmts:
                    visit(
                        stmt,
                        on_device or grid,
                        iter_shapes,
                        (*block, (id(node), i)),
                        frozenset(inner),
                        frozenset(inner_unrolled),
                    )
            return
        if isinstance(node, (ast.If, ast.While)):
            visit(node.test, on_device, constant_shapes, block, targets, unrolled)
            for i, stmts in enumerate((node.body, node.orelse)):
                for stmt in stmts:
                    visit(
                        stmt,
                        on_device,
                        constant_shapes,
                        (*block, (id(node), i)),
                        targets,
                        unrolled,
                    )
            return
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            # The device IR unrolls a comprehension: a load indexed by one of
            # its variables is a load per element.
            names = {
                n.id
                for gen in node.generators
                for n in ast.walk(gen.target)
                if isinstance(n, ast.Name)
            }
            inner_unrolled = unrolled | names
            for child in ast.iter_child_nodes(node):
                visit(child, on_device, constant_shapes, block, targets, inner_unrolled)
            return
        skip: ast.AST | None = None
        if (read := metadata_read(node)) is not None:
            name, fields, skip = read
            if on_device:
                fields -= {_DTYPE, _DEVICE, _REFERENCED}
            if constant_shapes:
                fields -= {_SIZE, _STRIDE}
            reads[name].update(fields)
        elif (name := param_name(node)) is not None:
            reads[name].add(_VALUES)
        elif on_device and (skip := loaded_param(node)) is not None:
            name = cast("ast.Name", skip).id
            forms = loaded.setdefault(name, set())
            form = load_form(node, block, targets)
            forms.add(form)
            # In an hl.static_range body, a load indexed by its variable, or
            # by anything else not plain (``j = i + 1; x[tile, j]``), is a
            # load per unrolled iteration.
            if unrolled and (
                form[0] == "load"
                or any(
                    isinstance(n, ast.Name) and n.id in unrolled for n in ast.walk(node)
                )
            ):
                forms.add(("unrolled", next(visits)))
        for child in ast.iter_child_nodes(node):
            if child is not skip:
                visit(child, on_device, constant_shapes, block, targets, unrolled)

    for stmt in host_function.body:
        visit(stmt, False, False)
    # merge_reduction_dims loads a parameter through a flattened view
    # (``x.flatten(1, 2)``) the device code names instead of the parameter, so
    # its prologue would not replace those loads.
    for tensor_origin in host_function.tensor_to_origin.values():
        if isinstance(tensor_origin, FlattenOrigin):
            name = tensor_origin.value.host_str()
            if name in reads:
                reads[name].add(_VALUES)
    for name in params:
        if name not in loaded:
            reads[name].add(_REFERENCED)
            continue
        if form_count(loaded[name]) > 1:
            # A fused prologue replaces one form of load; the others, with
            # another index, mask or block, would need the parameter's own
            # values.
            reads[name].add(_VALUES)
        if not static_shapes:
            # Device loads index with the strides and sizes of the host tensor.
            reads[name].update((_SIZE, _STRIDE))
    return {name: frozenset(fields) for name, fields in reads.items() if fields}


# How to read each field of ``_reads_besides_loads`` off a buffer.
_FIELD_GETTERS: dict[str, Callable[[IRNode], object]] = {
    _SIZE: lambda node: list(node.get_size()),
    _STRIDE: lambda node: list(node.get_stride()),
    _DTYPE: lambda node: node.get_dtype(),
    _DEVICE: lambda node: node.get_device(),
}


def _can_stand_in_for(
    source_name: str, producer: IRNode, fields: Collection[str]
) -> bool:
    """Whether buffer ``source_name`` can be passed in place of ``producer`` to
    code that reads ``fields`` of it (see ``_reads_besides_loads``)."""
    if _VALUES in fields:
        return False
    source = V.graph.try_get_buffer(source_name)
    # A mutation's output (NoneLayout) is no tensor to pass.
    if source is None or isinstance(source, MutationOutput):
        return False
    if _STRIDE in fields and not isinstance(source.get_layout(), FixedLayout):
        return False  # Inductor may still change it.
    return _fields_match(source, producer, fields)


def _fields_match(source: IRNode, producer: IRNode, fields: Collection[str]) -> bool:
    # ``_REFERENCED`` asks only for a source, which this is.
    return all(
        _FIELD_GETTERS[field](source) == _FIELD_GETTERS[field](producer)
        for field in fields
        if field != _REFERENCED
    )


def _ensure_stand_in(producer: ComputedBuffer, fields: Collection[str]) -> bool:
    """Whether a buffer ``producer`` reads can stand in for it to code reading
    ``fields``, freezing the layout of one that needs it: a source whose
    layout Inductor has not decided yet (an intermediate buffer, during
    lowering) but matches as it stands is frozen, so it stays a match."""
    sources = [
        dep.name
        for dep in producer.get_read_writes().reads
        if isinstance(dep, MemoryDep)
    ]
    if any(_can_stand_in_for(name, producer, fields) for name in sources):
        return True
    if _VALUES in fields:
        return False
    for name in sources:
        source = V.graph.try_get_buffer(name)
        if (
            isinstance(source, Buffer)
            and isinstance(source.get_layout(), FlexibleLayout)
            and _fields_match(source, producer, fields)
        ):
            # Exactly the strides that matched: as_fixed() may pad them.
            source.freeze_layout_with_exact_strides(
                source.get_stride()  # pyrefly: ignore[bad-argument-type]
            )
            return True
    return False


def _stand_ins_first(
    sources: Iterable[str], producer: IRNode, fields: Collection[str]
) -> tuple[str, ...]:
    """``sources`` in a fixed order: those that can stand in for ``producer``
    to code reading ``fields`` first, then those with its whole layout, then
    by name."""
    return tuple(
        sorted(
            sources,
            key=lambda name: (
                not _can_stand_in_for(name, producer, fields),
                not _can_stand_in_for(name, producer, _LAYOUT_FIELDS),
                name,
            ),
        )
    )


def _example_storage_offset(node: IRNode) -> int:
    """Storage offset that gives ``node``'s example input its runtime alignment.

    Backends specialize on address residues (CuTe vector packets, Triton's
    16-byte TMA base), and Inductor promises 16-byte alignment only for inputs
    it has not recorded as unaligned.  Example storage comes from the caching
    allocator, so an offset within one 16-byte window reproduces a known
    residue, and a nonzero offset stays nonzero for zero-offset checks.  An
    unknown residue (unaligned base or symbolic offset) becomes a one-element
    shift, the weakest alignment an element offset can express; 16-byte
    elements cannot be misaligned by element offsets in the first place.
    """
    offset = node.get_layout().offset
    window = max(GPU_ALIGN_BYTES // node.get_dtype().itemsize, 1)
    if not is_unaligned(node):
        return 0 if V.graph.sizevars.statically_known_equals(offset, 0) else window
    if (
        isinstance(node, ReinterpretView)
        and not is_unaligned(node.data)
        and isinstance(offset, (int, sympy.Integer))
    ):
        return int(offset) % window
    return 1


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

    def realize(tb: TensorBox) -> IRNode:
        # A bitcast becomes a view of its source's buffer, as Inductor makes it
        # where DtypeView can reindex (newer PyTorch; see
        # pre_realize_bitcast_names below for PyTorch 2.13).
        if isinstance(tb.data, DtypeView):
            inner = tb.data.data
            source = _realize(
                inner if isinstance(inner, TensorBox) else TensorBox(inner)
            )
            return DtypeView.create(source, tb.data.get_dtype())
        return _realize(tb)

    def through_bitcast(node: IRNode) -> bool:
        """Whether ``node``'s view chain holds a bitcast (DtypeView)."""
        while isinstance(node, (TensorBox, BaseView)):
            if isinstance(node, DtypeView):
                return True
            node = node.data
        return False

    realized: dict[str, IRNode] = {}
    # Inputs whose view chain holds a bitcast. Newer PyTorch realizes them as
    # dtype-changing ReinterpretViews, which create() keeps unfused anyway; on
    # PyTorch 2.13, whose DtypeView cannot reindex, a view of a bitcast (a
    # top-level one goes through realize() above) is copied into a buffer of
    # its own, which create() cannot tell from a bitcast-free producer. Fused
    # as one prologue with its source's producer, such a copy left the
    # producer's input out of the kernel (NameError), so name them all.
    pre_realize_bitcast_names: set[str] = set()
    for n, tb in tensor_args.items():
        if isinstance(tb, TensorBox):
            realized[n] = realize(tb)
            if through_bitcast(tb):
                pre_realize_bitcast_names.add(realized[n].get_name())

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
    has_static_dims = False
    symbolic_stride_examples: list[torch.Tensor] = []
    for n, r in realized.items():
        # Reused as autotune_args; uninitialized memory may contain NaN
        # bytes that spuriously fail accuracy checks. rand_strided
        # matches inductor's benchmark_example_value.
        device = r.get_device()
        assert device is not None
        size = [as_int(s, 64) for s in r.get_size()]
        stride = [as_int(s, 1) for s in r.get_stride()]
        offset = _example_storage_offset(r)
        example = rand_strided(
            size,
            stride,
            dtype=r.get_dtype(),
            device=device,
            extra_size=offset,
        ).as_strided(size, stride, offset)
        if not kernel.settings.static_shapes:
            # Capture bound Dynamo's FakeTensors, whose static dims are
            # literal ints; keep them literal here too, or this bind traces
            # them as symbols (a size-1 one guarded to 1) and its host trace
            # no longer matches the captured one.
            static_dims = frozenset(
                i
                for i, s in enumerate(r.get_size())
                if isinstance(s, (int, sympy.Integer))
            )
            if static_dims:
                setattr(example, HELION_STATIC_DIMS_ATTR, static_dims)
                has_static_dims = True
            # The placeholder stride of 1 proves no layout fact
            # (``CompileEnvironment.traced_strides_hold``).
            if not all(isinstance(s, (int, sympy.Integer)) for s in r.get_stride()):
                symbolic_stride_examples.append(example)
        all_args[n] = example
    _rebuild_container_args(all_args)
    symbolic_strides = tensor_input_sources(all_args, symbolic_stride_examples)

    fake_tensors: list[object] = [
        all_args.get(n, p.default)
        for n, p in kernel.signature.parameters.items()
        if n in all_args or p.default is not p.empty
    ]
    bind_args = tuple(fake_tensors)
    # Static dims are not part of a dynamic-shape kernel's cache key, so a
    # bound that keeps them literal must not be shared through the cache.
    if output_spec.get(_REQUIRES_ISOLATED_LOWERING) or has_static_dims:
        bound = kernel._bind_isolated(bind_args)
    else:
        bound = _bind_kernel_for_lowering(kernel, bind_args)
    expected_semantic_fingerprint = output_spec.get(_HOST_SEMANTIC_FINGERPRINT)
    input_normalization = output_spec.get(_HOST_SEMANTIC_INPUT_NORMALIZATION)
    if not isinstance(input_normalization, tuple):
        input_normalization = ()
    host_function = bound.host_function
    assert host_function is not None
    flat_leaves, tree_spec, return_ast = _get_flat_output(host_function)
    actual_semantic_fingerprint: str | None = None
    needs_isolated_bind = bound._cache_managed and bool(host_function.global_imports)
    if isinstance(expected_semantic_fingerprint, str) and not needs_isolated_bind:
        actual_semantic_fingerprint = bound._get_host_semantic_fingerprint(
            flat_leaves,
            input_normalization=input_normalization,
        )
        needs_isolated_bind = (
            bound._cache_managed
            and actual_semantic_fingerprint != expected_semantic_fingerprint
        )
    if needs_isolated_bind:
        log.debug(
            "cached bound for %s depends on external state or does not match "
            "the Dynamo host trace; using an isolated bind",
            kernel.name,
        )
        bound = kernel._bind_isolated(bind_args)
        host_function = bound.host_function
        assert host_function is not None
        flat_leaves, tree_spec, return_ast = _get_flat_output(host_function)
        actual_semantic_fingerprint = None

    if isinstance(expected_semantic_fingerprint, str):
        if actual_semantic_fingerprint is None:
            actual_semantic_fingerprint = bound._get_host_semantic_fingerprint(
                flat_leaves,
                input_normalization=input_normalization,
            )
        if actual_semantic_fingerprint != expected_semantic_fingerprint:
            raise RuntimeError(
                "Helion kernel trace or compile environment differed between "
                "Dynamo capture and Inductor lowering for kernel "
                f"{kernel.fn.__name__!r}; this configuration cannot be lowered safely"
            )

    # Derive output structure from bound kernel using inductor-time input layouts.
    # This gives correct strides even when inductor changes input memory layouts.
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
            symbolic_input_layouts=has_symbolic_shapes,
            symbolic_strides=symbolic_strides,
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
        fusion_enabled=kernel.settings.torch_compile_fusion
        and bound.env.backend.supports_inductor_fusion,
        pre_realize_bitcast_names=frozenset(pre_realize_bitcast_names),
        kernel=kernel,
        bound_kernel=bound,
        constant_args=constant_args,
        autotune_args=tuple(fake_tensors),
        symbolic_input_layouts=has_symbolic_shapes,
        symbolic_strides=symbolic_strides,
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
