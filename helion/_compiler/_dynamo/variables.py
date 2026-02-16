from __future__ import annotations

import ast
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import cast

import torch

log = logging.getLogger(__name__)
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.higher_order_ops import OutputSpec as _HopOutputSpec
from torch._dynamo.variables.higher_order_ops import _call_function_and_unflatten_output
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable
import torch.utils._pytree as pytree

from helion._compat import requires_torch_version
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion.runtime.kernel import BoundKernel


def _validate_return(
    body: list[ast.stmt], flat_leaves: list[object]
) -> None:
    """Validate return statement for torch.compile fusion compatibility."""
    # Check return not in control flow
    for stmt in body:
        for node in ast.walk(stmt):
            if node is not stmt and isinstance(node, ast.Return):
                raise RuntimeError(
                    "Return statements inside control flow (if/else/for/while) "
                    "are not supported with torch.compile fusion. "
                    "Please restructure the kernel to have a single return at the end."
                )

    # Check shared storage among distinct output tensors
    seen_objs: set[int] = set()
    seen_storages: set[int] = set()
    for leaf in flat_leaves:
        if not isinstance(leaf, torch.Tensor) or id(leaf) in seen_objs:
            continue
        seen_objs.add(id(leaf))
        sid = id(leaf.untyped_storage())
        if sid in seen_storages:
            raise RuntimeError(
                "Returning multiple outputs that share storage is not yet supported "
                "with torch.compile fusion. Please return independent tensors."
            )
        seen_storages.add(sid)


def _get_flat_output(
    host_function: object,
) -> tuple[list[object], pytree.TreeSpec | None]:
    """Get flattened output leaves and tree spec from a host function."""
    body = getattr(host_function, "body", None)
    if body is None:
        return [], None
    ret = next((s for s in reversed(body) if isinstance(s, ast.Return)), None)
    if ret is not None and ret.value is not None:
        type_info = getattr(ret.value, "_type_info", None)
        if type_info is not None:
            return pytree.tree_flatten(type_info.proxy())
    return [], None


def _guard_syms(syms: Iterable[torch.SymInt]) -> None:
    """Clear ``RelaxedUnspecConstraint`` and call ``guard_int`` for each SymInt.

    Before calling ``guard_int()`` on a SymInt that may have been
    ``mark_dynamic``'d, we must remove its ``RelaxedUnspecConstraint`` from
    the corresponding ``StatefulSymbolicContext`` entry.  Otherwise
    ``produce_guards()`` raises ``ConstraintViolationError``.

    The ``source_to_ctx`` lookup is built once for all symbols.
    """
    from torch._dynamo.source import TensorProperty
    from torch._dynamo.source import TensorPropertySource
    from torch.fx.experimental.symbolic_shapes import StatefulSymbolicContext
    from torch.fx.experimental.symbolic_shapes import guard_int

    tracing_ctx = torch._guards.TracingContext.try_get()
    if tracing_ctx is None or not hasattr(tracing_ctx, "tensor_to_context"):
        # No tracing context or missing internals — just guard without clearing constraints.
        for sym in syms:
            guard_int(sym)
        return

    # Build lookup from tensor_source -> StatefulSymbolicContext once.
    source_to_ctx = {
        ctx_val.tensor_source: ctx_val
        for ctx_val in tracing_ctx.tensor_to_context.values()
        if isinstance(ctx_val, StatefulSymbolicContext)
        and hasattr(ctx_val, "tensor_source")
    }

    _prop_attr = {TensorProperty.SIZE: "constraint_sizes", TensorProperty.STRIDE: "constraint_strides"}
    for sym in syms:
        for sym_var in sym.node.expr.free_symbols:
            for src in sym.node.shape_env.var_to_sources.get(sym_var, []):
                if isinstance(src, TensorPropertySource) and (ctx := source_to_ctx.get(src.base)):
                    constraints = getattr(ctx, _prop_attr.get(src.prop, ""), None)
                    if constraints is not None and src.idx is not None and src.idx < len(constraints):
                        constraints[src.idx] = None
        guard_int(sym)


def _guard_specialized_syms(
    bound: BoundKernel,
    names: list[str],
    args: tuple[Any, ...],
) -> None:
    """Add Dynamo guards for SymInts used by ``hl.specialize()``.

    We look up each specialized var via ``var_to_sources`` to find the
    corresponding Dynamo SymInt, because ``hl.specialize()`` concretizes the
    helion-internal symbols.
    """
    from torch._dynamo.source import LocalSource
    from torch._dynamo.source import TensorProperty
    from torch._dynamo.source import TensorPropertySource

    name_to_arg = dict(zip(names, args, strict=False))
    syms_to_guard: list[torch.SymInt] = []
    for v in bound.env.specialized_vars:
        for src in bound.env.shape_env.var_to_sources.get(v, []):
            if not isinstance(src, TensorPropertySource) or not isinstance(src.base, LocalSource):
                continue
            arg = name_to_arg.get(src.base.local_name)
            if not isinstance(arg, torch.Tensor) or src.idx is None or src.idx >= arg.ndim:
                continue
            attr = {TensorProperty.SIZE: "size", TensorProperty.STRIDE: "stride"}.get(src.prop)
            if attr is not None:
                sym = getattr(arg, attr)(src.idx)
                if isinstance(sym, torch.SymInt):
                    syms_to_guard.append(sym)
    if syms_to_guard:
        _guard_syms(syms_to_guard)


def _build_dynamo_sym_remap(
    args: tuple[Any, ...], fake_args: tuple[Any, ...], helion_shape_env: object
) -> dict[Any, Any]:
    """Build a mapping from Helion's Dynamo-level SymInts to the caller's values.

    kernel.bind() creates FakeTensors in Helion's own ShapeEnv, so output
    SymInts aren't tracked by external tracers (make_fx, Dynamo).  This maps
    Helion's input symbols to the caller's original values (which may be tracer
    SymInts or concrete ints).
    """
    sym_remap: dict[Any, Any] = {}
    for orig_leaf, fake_leaf in zip(pytree.tree_leaves(args), pytree.tree_leaves(fake_args), strict=True):
        if isinstance(orig_leaf, torch.Tensor) and isinstance(fake_leaf, torch.Tensor):
            for orig_s, fake_s in zip(orig_leaf.shape, fake_leaf.shape, strict=True):
                if isinstance(fake_s, torch.SymInt) and fake_s.node.shape_env is helion_shape_env:
                    sym_remap.setdefault(fake_s.node.expr, orig_s)
        elif isinstance(fake_leaf, torch.SymInt) and fake_leaf.node.shape_env is helion_shape_env:
            sym_remap[fake_leaf.node.expr] = orig_leaf
    return sym_remap


def infer_output_spec(
    kernel: Kernel,
    args: tuple[Any, ...],
) -> dict[str, Any]:
    """Infer the HOP output_spec by binding the kernel and analyzing its outputs.

    Remaps helion-internal SymInts back to the caller's values so that symbolic
    shape relationships are preserved for external tracers (make_fx, Dynamo).
    """
    names = list(kernel.signature.parameters.keys())
    param_tensors: dict[str, torch.Tensor] = {}
    for n, v in zip(names, args, strict=True):
        if isinstance(v, torch.Tensor):
            param_tensors[n] = v
        elif isinstance(v, (tuple, list, dict)):
            for i, leaf in enumerate(pytree.tree_leaves(v)):
                if isinstance(leaf, torch.Tensor):
                    param_tensors[f"{n}.{i}"] = leaf

    # Guard SymInt dimensions when static_shapes=True so Dynamo retraces on shape change.
    if kernel.settings.static_shapes:
        syms = [s for t in param_tensors.values() for s in (*t.shape, *t.stride()) if isinstance(s, torch.SymInt)]
        if syms:
            _guard_syms(syms)

    bound = kernel.bind(args)
    assert bound.host_function, "kernel.bind() succeeded but host_function is None"

    # Guard SymInt dimensions used by hl.specialize() so Dynamo retraces on shape change.
    has_specialized_strides = bool(bound.env.specialized_strides)
    if bound.env.specialized_vars and not kernel.settings.static_shapes:
        _guard_specialized_syms(bound, names, args)

    # Detect mutations via tensor identity tracked during type propagation.
    # TensorType.propagate_setitem records id(fake_value) in
    # CompileEnvironment.mutated_tensor_ids, which correctly handles all
    # forms of container unpacking/subscripting without AST pattern matching.
    mutated_tensor_ids = bound.env.mutated_tensor_ids

    flat_leaves, tree_spec = _get_flat_output(bound.host_function)

    if tree_spec is None:
        return {
            "leaf_specs": [],
            "tree_spec_str": None,
            "mutated_inputs": bound.host_function.mutated_param_names(
                mutated_tensor_ids
            ),
        }

    _validate_return(bound.host_function.body, flat_leaves)

    # Detect output-to-input aliases via tensor identity and storage checks.
    # Uses the bound kernel's internal FakeTensors (from host_function.params)
    # which share identity/storage with pass-through and view outputs.
    bind_param_tensors = bound.host_function.flat_tensor_params()
    id_to_name = {id(v): n for n, v in bind_param_tensors.items()}
    storage_to_name = {
        id(v.untyped_storage()): n for n, v in bind_param_tensors.items()
    }
    direct_aliases: dict[int, str] = {}
    all_alias_names: set[str] = set()
    for i, leaf in enumerate(flat_leaves):
        if not isinstance(leaf, torch.Tensor):
            continue
        name = id_to_name.get(id(leaf))
        if name is not None:
            direct_aliases[i] = name
            all_alias_names.add(name)
        else:
            name = storage_to_name.get(id(leaf.untyped_storage()))
            if name is not None:
                all_alias_names.add(name)

    leaf_specs: list[dict[str, Any]] = []
    for leaf in flat_leaves:
        if isinstance(leaf, torch.Tensor):
            leaf_specs.append({
                "type": "tensor",
                "shape": list(leaf.shape),
                "dtype": leaf.dtype,
                "device": str(leaf.device),
            })
        elif isinstance(leaf, (torch.SymInt, int, float, bool, str)) or leaf is None:
            leaf_specs.append({"type": "scalar", "scalar_value": leaf})
        else:
            raise RuntimeError(
                f"Returning {type(leaf).__name__} values from a Helion kernel "
                f"is not supported with torch.compile fusion."
            )

    helion_shape_env = bound.env.shape_env
    sym_remap = _build_dynamo_sym_remap(args, bound.fake_args, helion_shape_env)

    def _remap_or_resolve(val: object) -> object:
        if isinstance(val, torch.SymInt) and val.node.shape_env is helion_shape_env:
            mapped = sym_remap.get(val.node.expr)
            return mapped if mapped is not None else int(helion_shape_env.size_hint(val.node.expr))
        return val

    stride_relaxed = not kernel.settings.static_shapes
    for spec, leaf in zip(leaf_specs, flat_leaves, strict=True):
        if spec["type"] == "tensor":
            spec["shape"] = [_remap_or_resolve(s) for s in spec["shape"]]
            if stride_relaxed:
                # Strides are unknown at trace time; the fake function
                # creates unbacked stride symbols so downstream ops like
                # as_strided are not folded away.
                spec["stride"] = None
                # Pass actual stride hints from the kernel's FakeTensor
                # output.  Using contiguous hints would cause
                # propagate_real_tensors to derive wrong contiguity
                # constraints when the kernel output is non-contiguous
                # (e.g. from empty_like of a transposed input).
                assert isinstance(leaf, torch.Tensor)
                spec["stride_hints"] = [
                    int(s) if not isinstance(s, torch.SymInt)
                    else int(helion_shape_env.size_hint(s.node.expr))
                    for s in leaf.stride()
                ]
            else:
                # Use actual strides from the kernel's FakeTensor output.
                # make_contiguous_strides_for would be wrong when the output
                # is non-contiguous (e.g. from empty_like of a transposed input).
                assert isinstance(leaf, torch.Tensor)
                spec["stride"] = [_remap_or_resolve(s) for s in leaf.stride()]
        elif spec["type"] == "scalar":
            sv = spec.get("scalar_value")
            if isinstance(sv, torch.SymInt):
                spec["scalar_value"] = _remap_or_resolve(sv)

    mutated = bound.host_function.mutated_param_names(mutated_tensor_ids, bind_param_tensors)
    for alias_name in all_alias_names:
        if alias_name not in mutated:
            mutated.append(alias_name)

    if len(mutated) > 1:
        storage_to_names: dict[int, list[str]] = {}
        for mut_name in mutated:
            if mut_name in param_tensors:
                sid = id(param_tensors[mut_name].untyped_storage())
                storage_to_names.setdefault(sid, []).append(mut_name)
        for shared_names in storage_to_names.values():
            if len(shared_names) > 1:
                raise RuntimeError(
                    f"torch.compile fusion does not support multiple mutated arguments "
                    f"that share storage ({', '.join(shared_names)}) in a Helion kernel"
                )

    return {
        "leaf_specs": leaf_specs,
        "tree_spec_str": pytree.treespec_dumps(tree_spec),
        "mutated_inputs": mutated,
        "direct_aliases": direct_aliases,
        "has_specialized_strides": has_specialized_strides,
    }


def _relax_stride_guards(
    tx: InstructionTranslator,
    tensor_args: dict[VariableTracker, VariableTracker],
) -> None:
    """Prevent Dynamo from guarding on stride order for kernel tensor inputs.

    Three layers of stride guards are relaxed:

    1. **TENSOR_MATCH** (C++ guard): Sets strides to ``None`` in
       ``input_source_to_sizes_strides`` so the C++ tensor-match guard
       skips stride checks.

    2. **SHAPE_ENV tracked_fakes**: Replaces tensor entries in
       ``tracked_fakes`` with per-dimension SymInt entries so that
       ``produce_guards_verbose`` only calls ``track_symint`` for size
       values (no stride iteration).

    3. **SHAPE_ENV stride symbols**: Concretizes stride-only symbols
       via ``shape_env._set_replacement`` so that any guard expressions
       referencing them (existing or created later by ``evaluate_expr``)
       simplify to concrete values and are automatically skipped by
       ``_maybe_evaluate_static``.
    """
    from torch._dynamo.source import TensorProperty
    from torch._dynamo.source import TensorPropertySource
    from torch.fx.experimental.symbolic_shapes import TrackedFake

    tensor_sources = {s for var in tensor_args.values() if (s := getattr(var, "source", None)) is not None}
    if not tensor_sources:
        return

    # Layer 1: Relax TENSOR_MATCH stride guards.
    if (sizes_strides := getattr(tx.output, "input_source_to_sizes_strides", None)) is not None:
        for source in tensor_sources:
            metadata = sizes_strides.get(source)
            if metadata is not None and "stride" in metadata:
                metadata["stride"] = (None,) * len(metadata["stride"])

    # Layer 2: Relax SHAPE_ENV stride guards.
    if (tracked_fakes := getattr(tx.output, "tracked_fakes", None)) is not None:
        new_tracked_fakes: list[TrackedFake] = []
        for tf in tracked_fakes:
            if tf.source in tensor_sources and isinstance(tf.fake, torch.Tensor):
                new_tracked_fakes.extend(
                    TrackedFake(tf.fake.size(i), TensorPropertySource(tf.source, TensorProperty.SIZE, i), None)
                    for i in range(tf.fake.dim())
                )
            else:
                new_tracked_fakes.append(tf)
        tracked_fakes[:] = new_tracked_fakes

    # Layer 3: Concretize stride-only symbols so that any guard expressions
    # referencing them (existing or created later by evaluate_expr)
    # simplify to concrete values and are automatically skipped by
    # _maybe_evaluate_static.
    tracing_ctx = torch._guards.TracingContext.try_get()
    if not tracing_ctx or not tracing_ctx.fake_mode:
        return
    shape_env = tracing_ctx.fake_mode.shape_env

    if hasattr(shape_env, "_set_replacement") and hasattr(shape_env, "backed_var_to_val") and hasattr(shape_env, "replacements"):
        import sympy

        for sym, sources in shape_env.var_to_sources.items():
            if (sources
                    and all(isinstance(s, TensorPropertySource) and s.prop is TensorProperty.STRIDE
                            and s.base in tensor_sources for s in sources)
                    and sym not in shape_env.replacements
                    and sym in shape_env.backed_var_to_val):
                shape_env._set_replacement(
                    sym, sympy.Integer(int(shape_env.backed_var_to_val[sym])), "helion_stride_relaxation")


def _flatten_variable_items(var: VariableTracker) -> list[VariableTracker]:
    """Recursively flatten container VariableTracker to leaf items (matching pytree depth)."""
    if isinstance(var, (TupleVariable, ListVariable, ConstDictVariable)):
        items = var.items.values() if isinstance(var, ConstDictVariable) else var.items
        return [r for item in items for r in _flatten_variable_items(item)]
    return [var]


def _unwrap_arg(a: VariableTracker) -> object:
    """Extract a concrete/fake value from a single Dynamo VariableTracker."""
    if a.is_python_constant():
        return a.as_python_constant()
    return pytree.tree_map(
        lambda p: p.node.meta.get("example_value") if hasattr(p, "node") else p,
        a.as_proxy(),
    )


def _replace_direct_aliases(
    result: VariableTracker,
    output_spec: dict[str, object],
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Replace direct-alias outputs with the original input variables."""
    direct_aliases = cast("dict[int, str]", output_spec.get("direct_aliases", {}))
    replacements = {
        i: param_vars[name] for i, name in direct_aliases.items() if name in param_vars
    }
    if not replacements:
        return result

    # Walk the variable tree, replacing leaves at aliased positions
    counter = [0]

    def walk(vt: VariableTracker) -> VariableTracker:
        if isinstance(vt, (TupleVariable, ListVariable)):
            new_items = [walk(item) for item in vt.items]
            return type(vt)(new_items)
        idx = counter[0]
        counter[0] += 1
        return replacements.get(idx, vt)

    return walk(result)


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    def __init__(
        self, kernel: Kernel, kernel_idx: int | None, **kwargs: object
    ) -> None:  # pyrefly: ignore[bad-argument-type]
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table

        super().__init__(**kwargs)  # pyrefly: ignore[bad-argument-type]
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
        """Handle a call to a Helion kernel during Dynamo tracing."""
        # Lazy import: higher_order_ops requires PyTorch >= 2.11 (checked in wrap_helion_kernel)
        from helion._compiler._dynamo.higher_order_ops import (
            helion_kernel_wrapper_mutation,
        )

        sig_params = self._kernel.signature.parameters

        # Map positional args and kwargs to parameter names, partition into constants vs tensors
        param_vars = dict(zip(sig_params.keys(), args, strict=False))
        param_vars.update(kwargs)
        constant_args: dict[str, object] = {}
        tensor_args: dict[VariableTracker, VariableTracker] = {}
        container_specs: dict[str, str] = {}
        for name, var in param_vars.items():
            if var.is_python_constant():
                constant_args[name] = var.as_python_constant()
            elif isinstance(var, (TupleVariable, ListVariable, ConstDictVariable)):
                # Flatten container elements into individual tensor_args/constant_args
                flat_items = _flatten_variable_items(var)
                container_specs[name] = pytree.treespec_dumps(pytree.tree_flatten(var.as_proxy())[1])
                for i, item in enumerate(flat_items):
                    mangled = f"{name}.{i}"
                    if item.is_python_constant():
                        constant_args[mangled] = item.as_python_constant()
                    else:
                        tensor_args[variables.ConstantVariable.create(mangled)] = item
            else:
                tensor_args[variables.ConstantVariable.create(name)] = var
        if container_specs:
            constant_args["__container_specs"] = container_specs

        # Emit HOP node into FX graph and unflatten output
        output_spec = infer_output_spec(
            self._kernel,
            tuple(
                _unwrap_arg(param_vars[name]) if name in param_vars else p.default
                for name, p in sig_params.items()
                if name in param_vars or p.default is not p.empty
            ),
        )

        # Relax stride guards unless the kernel specializes on strides.
        if not self._kernel.settings.static_shapes and not output_spec.get(
            "has_specialized_strides"
        ):
            _relax_stride_guards(tx, tensor_args)

        hop_kwargs = {
            "kernel_idx": self._kernel_idx,
            "constant_args": constant_args,
            "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(),
            "output_spec": output_spec,
        }

        tree_spec_str = cast("str | None", output_spec.get("tree_spec_str"))
        if tree_spec_str is None:
            # Pure mutation kernel: emit HOP for side effects, return None
            tx.output.create_proxy(
                "call_function",
                helion_kernel_wrapper_mutation,
                (),
                hop_kwargs,
            )
            return variables.ConstantVariable.create(None)

        tree_spec = pytree.treespec_loads(tree_spec_str)
        leaf_specs = cast("list[dict[str, object]]", output_spec["leaf_specs"])
        masks = [s["type"] == "scalar" for s in leaf_specs]
        ret_spec = _HopOutputSpec(
            treespec=tree_spec,
            masks_to_filter_const_values=masks if any(masks) else None,
            const_values=[s.get("scalar_value") for s in leaf_specs]
            if any(masks)
            else None,
        )
        result = _call_function_and_unflatten_output(
            tx,
            helion_kernel_wrapper_mutation,
            (),
            hop_kwargs,
            None,
            ret_spec,
            None,
        )
        return _replace_direct_aliases(result, output_spec, param_vars)


def register_dynamo_variable() -> None:
    """Register HelionKernelVariable with Dynamo's VariableBuilder."""

    def wrap_helion_kernel(self: VariableBuilder, value: Kernel) -> VariableTracker:
        if os.environ.get("_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION", "0") == "1":
            if not requires_torch_version("2.11"):
                raise RuntimeError(
                    "Helion kernel torch.compile fusion requires "
                    "PyTorch >= 2.11. Please upgrade PyTorch or unset "
                    "_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION environment variable."
                )
            # Import template_buffer to register the HOP's Inductor lowering
            from helion._compiler._inductor import template_buffer  # noqa: F401

            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        return self.wrap_user_defined(value)

    VariableBuilder._type_dispatch()[Kernel] = wrap_helion_kernel
