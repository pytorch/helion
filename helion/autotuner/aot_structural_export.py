"""Standalone CuTe linking with one exact module namespace per variant."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from pathlib import Path
import tempfile
import textwrap
import types
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._dynamo.source import AttrSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from .._compiler.runtime_tensor_guards import _CanonicalLayoutClassifier
from ..runtime.structural_policy import StructuralPolicyError
from .aot_structural_policy import MODEL_MANIFEST
from .aot_structural_policy import model_configs
from .aot_structural_policy import policy_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ..runtime.kernel import BoundKernel
    from ..runtime.structural_policy import StructuralPolicy

_GuardProjection = tuple[int, tuple[int | str | tuple[str, int], ...]]
_RuntimeGuardBase = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[_GuardProjection, ...],
]
_CanonicalRuntimeGuard = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[_GuardProjection, ...],
    tuple[bool, tuple[str, ...], int],
]
RuntimeGuard = _RuntimeGuardBase | _CanonicalRuntimeGuard
GuardedCallKey = tuple[tuple[RuntimeGuard, ...], tuple[object, ...], tuple[object, ...]]


def export_runtime_guards(
    bound: BoundKernel, args: tuple[object, ...]
) -> tuple[tuple[RuntimeGuard, ...], tuple[object, ...]]:
    """Snapshot relocatable storage and canonical metadata classifiers.

    The same named classifier executes during exported dispatch. Its source
    module identity is checked when the artifact loads; no tensor contents or
    pointer address is baked into a key. Opaque/content-dependent classifiers
    cannot be relocated by this exporter and retain runtime AOT support.
    """
    from .aot_compile import _standalone_value_key

    indices = {
        name: index for index, name in enumerate(bound.kernel.signature.parameters)
    }

    def projection(source: object) -> _GuardProjection:
        if (
            isinstance(source, TensorPropertySource)
            and source.prop in (TensorProperty.SIZE, TensorProperty.STRIDE)
            and source.idx is not None
        ):
            index, path = projection(source.base)
            property_name = "size" if source.prop == TensorProperty.SIZE else "stride"
            return index, (*path, (property_name, source.idx))
        if isinstance(source, LocalSource) and source.local_name in indices:
            return indices[source.local_name], ()
        if isinstance(source, AttrSource):
            index, path = projection(source.base)
            return index, (*path, source.member)
        if (
            isinstance(source, GetItemSource)
            and isinstance(source.index, (int, str))
            and not source.index_is_slice
        ):
            index, path = projection(source.base)
            return index, (*path, source.index)
        raise StructuralPolicyError(
            "Standalone AOT has an unsupported guard projection"
        )

    guards: list[RuntimeGuard] = []
    for key, specialization in sorted(bound.env.runtime_input_specializations.items()):
        classifier = specialization.classifier
        canonical = type(classifier) is _CanonicalLayoutClassifier
        if not canonical and not specialization.reusable_tensor_properties:
            raise StructuralPolicyError(
                "Standalone AOT cannot relocate tensor-content specialization "
                "classifiers; retain runtime AOT for this kernel"
            )
        if not canonical and (
            not isinstance(classifier, types.FunctionType)
            or "<locals>" in classifier.__qualname__
        ):
            raise StructuralPolicyError(
                "Standalone AOT needs an importable storage classifier"
            )
        implementation = (
            _CanonicalLayoutClassifier
            if canonical
            else cast("types.FunctionType", classifier)
        )
        source_file = inspect.getsourcefile(implementation)
        if source_file is None:
            raise StructuralPolicyError(
                "Standalone AOT cannot identify storage classifier source"
            )
        base_guard: _RuntimeGuardBase = (
            key,
            implementation.__module__,
            implementation.__qualname__,
            hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
            repr(specialization.classifier_identity),
            tuple(sorted(specialization.reusable_tensor_properties)),
            tuple(projection(source) for source in specialization.sources),
        )
        if canonical:
            classifier = cast("_CanonicalLayoutClassifier", classifier)
            guards.append(
                (
                    *base_guard,
                    (
                        classifier.require_canonical_input,
                        classifier.expressions,
                        classifier.parameter_count,
                    ),
                )
            )
        else:
            guards.append(base_guard)
    plans = tuple(guards)
    classifiers = _export_classifiers(plans)
    for guard, imported in zip(plans, classifiers, strict=True):
        declared = bound.env.runtime_input_specializations[guard[0]].classifier
        if type(declared) is _CanonicalLayoutClassifier:
            if type(imported) is not _CanonicalLayoutClassifier or declared != imported:
                raise StructuralPolicyError(
                    "Standalone canonical metadata classifier changed"
                )
            continue
        assert isinstance(declared, types.FunctionType)
        assert isinstance(imported, types.FunctionType)
        # Tracing can clone a function while retaining its exact code and
        # globals dictionary. Identity of the function object is not required;
        # changed code, globals, defaults or closure state is not relocatable.
        if declared is not imported and not (
            declared.__code__ == imported.__code__
            and declared.__globals__ is imported.__globals__
            and declared.__defaults__ is imported.__defaults__
            and declared.__kwdefaults__ is imported.__kwdefaults__
            and declared.__closure__ is None
            and imported.__closure__ is None
        ):
            raise StructuralPolicyError(
                "Standalone storage classifier is not its declared global"
            )
    observed = _export_guard_key(args, plans, classifiers)
    snapshots = dict(bound.env.bound_runtime_input_specialization_results)
    for guard in plans:
        if (
            len(guard) == 8
            and guard[0] in bound.env.bound_metadata_specialization_results
        ):
            snapshots[guard[0]] = bound.env.bound_metadata_specialization_results[
                guard[0]
            ]
    if any(guard[0] not in snapshots for guard in plans):
        raise StructuralPolicyError(
            "Standalone storage guard requires a new binding snapshot"
        )
    expected = tuple(
        (guard[0], _standalone_value_key(snapshots[guard[0]])) for guard in plans
    )
    if observed != expected:
        raise StructuralPolicyError(
            "Standalone inputs changed their bound storage class; rebind before exporting"
        )
    return plans, observed


def _export_classifiers(
    guards: tuple[RuntimeGuard, ...],
) -> tuple[Callable[..., object], ...]:
    """Resolve exact imported classifier implementations before dispatch."""
    result: list[Callable[..., object]] = []
    for guard in guards:
        module = importlib.import_module(guard[1])
        classifier = module
        for attribute in guard[2].split("."):
            classifier = getattr(classifier, attribute)
        if not (
            len(guard) == 8
            and classifier is _CanonicalLayoutClassifier
            or len(guard) == 7
            and isinstance(classifier, types.FunctionType)
        ):
            raise ValueError("Standalone storage classifier implementation changed")
        source_file = inspect.getsourcefile(classifier)
        if (
            source_file is None
            or hashlib.sha256(Path(source_file).read_bytes()).hexdigest() != guard[3]
        ):
            raise ValueError("Standalone storage classifier source identity changed")
        if len(guard) == 8:
            definition = cast("_CanonicalRuntimeGuard", guard)[7]
            result.append(_CanonicalLayoutClassifier(*definition))
        else:
            assert isinstance(classifier, types.FunctionType)
            result.append(classifier)
    return tuple(result)


def _export_guard_key(
    args: tuple[object, ...],
    guards: tuple[RuntimeGuard, ...],
    classifiers: tuple[Callable[..., object], ...],
) -> tuple[object, ...]:
    from .aot_compile import _standalone_value_key

    def project(path: _GuardProjection) -> object:
        index, steps = path
        value = args[index]
        for step in steps:
            if isinstance(step, tuple):
                if not isinstance(value, torch.Tensor):
                    return None
                property_name, dimension = step
                if not -value.ndim <= dimension < value.ndim:
                    return None
                if property_name == "size":
                    value = value.size(dimension)
                elif property_name == "stride":
                    value = value.stride(dimension)
                else:
                    return None
            elif (
                isinstance(step, int)
                and isinstance(value, (list, tuple))
                and 0 <= step < len(value)
                or isinstance(value, dict)
                and step in value
            ):
                value = value[step]
            elif isinstance(step, str) and value is not None:
                value = getattr(value, step, None)
            else:
                return None
        return value

    return tuple(
        (
            guard[0],
            _standalone_value_key(
                classifier(tuple(project(path) for path in guard[6]))
            ),
        )
        for guard, classifier in zip(guards, classifiers, strict=True)
    )


def _export_call_key(
    args: tuple[object, ...], *, dynamic: bool = False
) -> tuple[object, ...]:
    """Literal metadata and argument-identity guards, never pointer addresses.

    Dynamic tensors retain arbitrary sizes (including zero/one) and strides.
    The index-width bucket matches automatic Int32/Int64 binding. Keeping that
    bucket for explicitly fixed index types is conservative. Compile-time values
    and extra bound facts use the exact form instead.
    """
    from .aot_compile import _standalone_value_key

    tensors: list[torch.Tensor] = []

    def key(value: object) -> tuple[object, ...]:
        if isinstance(value, torch.Tensor):
            identity = next(
                (index for index, previous in enumerate(tensors) if previous is value),
                len(tensors),
            )
            if identity == len(tensors):
                tensors.append(value)
            metadata = (
                ("tensor", str(value.dtype), value.ndim, value.numel() > 2**31 - 1)
                if dynamic
                else _standalone_value_key(value)
            )
            return (*metadata, value.device.type, identity)
        if dynamic and type(value) in (bool, int, float):
            return (type(value).__name__,)
        if type(value) is list or type(value) is tuple:
            return (type(value).__name__, tuple(key(item) for item in value))
        if type(value) is dict:
            return (
                "dict",
                tuple(
                    (_standalone_value_key(name), key(item))
                    for name, item in sorted(
                        value.items(),
                        key=lambda pair: repr(_standalone_value_key(pair[0])),
                    )
                ),
            )
        return _standalone_value_key(value)

    return key(args)


def _select_guarded_variant(
    args: tuple[object, ...],
    plans: tuple[tuple[RuntimeGuard, ...], ...],
    classifiers: tuple[tuple[Callable[..., object], ...], ...],
    variants: dict[tuple[object, ...], int | list[int]],
    dynamic: bool,
) -> int | list[int]:
    """Prefer exact calls, then dynamic calls with more checked conditions."""
    observed = [
        (plan, _export_guard_key(args, plan, functions))
        for plan, functions in zip(plans, classifiers, strict=True)
    ]
    exact_call_key = _export_call_key(args)
    for dynamic_key in (False, True) if dynamic else (False,):
        call_key = (
            _export_call_key(args, dynamic=True) if dynamic_key else exact_call_key
        )
        for plan, guard_key in observed:
            key: tuple[object, ...] = (plan, call_key, guard_key)
            if dynamic:
                key = ("dynamic" if dynamic_key else "exact", key)
            if key in variants:
                return variants[key]
    raise ValueError(
        f"No standalone variant for call signature {exact_call_key!r}; "
        f"observed per-plan guards {observed!r}"
    )


def _model_manifest(code: str, name: str, policy: StructuralPolicy) -> dict[str, Any]:
    tree = ast.parse(code)
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == MODEL_MANIFEST
            for target in node.targets
        )
    ]
    if len(values) != 1:
        raise StructuralPolicyError("Standalone AOT needs one literal model manifest")
    try:
        payload = ast.literal_eval(values[0])
    except (ValueError, TypeError) as error:
        raise StructuralPolicyError(
            "Standalone AOT needs a literal model manifest"
        ) from error
    module = types.ModuleType("model_manifest")
    vars(module)[MODEL_MANIFEST] = payload
    model_configs(module, name, policy)
    assert isinstance(payload, dict)
    return payload


def _user_key_source(fn: Callable[..., object]) -> tuple[str, str]:
    """Snapshot a plain source-defined key and its immutable/module globals.

    Opaque callables, decorators and mutable closure state cannot be faithfully
    relocated. They remain supported by runtime AOT and static export, and fail
    before replacing a dynamic standalone artifact.
    """
    if not isinstance(fn, types.FunctionType):
        raise StructuralPolicyError(
            "Dynamic standalone needs a source-defined user key"
        )
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as error:
        raise StructuralPolicyError(
            "Cannot read dynamic standalone user key source"
        ) from error
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise StructuralPolicyError(
            "Dynamic standalone user key must be a named function"
        )
    node = tree.body[0]
    if node.decorator_list or fn.__defaults__ or fn.__kwdefaults__:
        raise StructuralPolicyError(
            "Dynamic standalone user key cannot carry decorators/defaults"
        )
    closure = inspect.getclosurevars(fn)
    lines = ["from __future__ import annotations", "import torch"]
    for name, value in {**closure.globals, **closure.nonlocals}.items():
        if isinstance(value, types.ModuleType):
            lines.append(f"import {value.__name__} as {name}")
        elif isinstance(value, torch.dtype):
            lines.append(f"{name} = {value}")
        else:

            def immutable(item: object) -> bool:
                return type(item) in (type(None), bool, int, float, str, bytes) or (
                    type(item) is tuple and all(immutable(part) for part in item)
                )

            if not immutable(value):
                raise StructuralPolicyError(
                    "Dynamic standalone user key has unsupported closure state"
                )
            try:
                ast.literal_eval(repr(value))
            except (SyntaxError, ValueError) as error:
                raise StructuralPolicyError(
                    "Dynamic standalone user key has a nonliteral capture"
                ) from error
            lines.append(f"{name} = {value!r}")
    lines.append(source)
    return "\n".join(lines), node.name


def generate_policy_standalone(
    kernel_name: str,
    codes: list[str],
    heuristic_code: str,
    output_dir: Path,
    kernel_source_file: str | None,
    dispatch_keys: list[tuple[object, ...]] | None,
    policy: StructuralPolicy,
    user_key: Callable[..., object] | None,
    dynamic_groups: list[tuple[tuple[object, ...], list[int]]] | None,
    runtime_guards: tuple[RuntimeGuard, ...],
    runtime_guard_plans: tuple[tuple[RuntimeGuard, ...], ...] | None = None,
) -> Path:
    """Link exact generated modules, including embedded multi-launch children.

    CuTe already requires its Helion launcher/helpers at runtime. PyCodeCache
    retains real source files for CuTe's AST inspection and isolates *all*
    globals, imports, decorators and host-plan attachments. Kernel source and
    embedded child source bytes are not renamed, stripped or reserialized.
    """
    from .aot_compile import _standalone_call_key
    from .aot_compile import _standalone_value_key
    from .aot_kernel import _flatten_key_value

    if runtime_guard_plans is None:
        # Keep the legacy tuple formats at the API boundary, then link every
        # variant through the same plan-aware dispatcher.
        runtime_guard_plans = (runtime_guards,)
        if dispatch_keys is not None:
            dispatch_keys = [(runtime_guards, *key) for key in dispatch_keys]
        if dynamic_groups is not None:
            dynamic_groups = [
                (
                    (key[0], (runtime_guards, *cast("tuple[object, ...]", key[1]))),
                    indices,
                )
                for key, indices in dynamic_groups
            ]
    elif runtime_guards:
        raise StructuralPolicyError(
            "runtime_guards and runtime_guard_plans cannot both be specified"
        )
    runtime_guard_plans = tuple(
        sorted(set(runtime_guard_plans), key=lambda plan: (-len(plan), repr(plan)))
    )
    if dispatch_keys is not None and any(
        key[0] not in runtime_guard_plans for key in dispatch_keys
    ):
        raise StructuralPolicyError(
            "Static variant plan is absent from runtime_guard_plans"
        )
    if dynamic_groups is not None and any(
        cast("GuardedCallKey", key[1])[0] not in runtime_guard_plans
        for key, _indices in dynamic_groups
    ):
        raise StructuralPolicyError(
            "Dynamic variant plan is absent from runtime_guard_plans"
        )
    payload = _model_manifest(heuristic_code, kernel_name, policy)
    if not codes:
        raise StructuralPolicyError("Standalone AOT requires at least one variant")
    if dispatch_keys is not None:
        if len(dispatch_keys) != len(codes) or len(set(dispatch_keys)) != len(
            dispatch_keys
        ):
            raise StructuralPolicyError(
                "Static standalone call keys must be unique and match variants"
            )
        ordered = sorted(
            zip(dispatch_keys, codes, strict=True), key=lambda item: repr(item[0])
        )
        dispatch_keys, codes = map(list, zip(*ordered, strict=True))
    elif dynamic_groups is None:
        assert isinstance(payload, dict)
        if len(payload["configs"][kernel_name]) != len(codes):
            raise StructuralPolicyError(
                "Standalone variants must retain model config index order"
            )
    else:
        assert isinstance(payload, dict)
        count = len(payload["configs"][kernel_name])
        if any(len(indices) != count for _key, indices in dynamic_groups) or sorted(
            index for _key, indices in dynamic_groups for index in indices
        ) != list(range(len(codes))):
            raise StructuralPolicyError(
                "Dynamic standalone groups must preserve all config indices"
            )
    for code in codes:
        tree = ast.parse(code)
        hosts = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == kernel_name
        ]
        if len(hosts) != 1:
            raise StructuralPolicyError(
                "Standalone variant must contain its allocating host wrapper"
            )

    # No generated kernel/module shares this outer namespace. Even unusual
    # public kernel names cannot shadow the linker's imports or dispatcher state.
    prefix = "_helion_aot"
    while kernel_name.startswith(prefix):
        prefix += "_"
    linked_helpers = []
    for helper in (
        _export_call_key,
        _export_classifiers,
        _export_guard_key,
        _select_guarded_variant,
    ):
        function = ast.parse(textwrap.dedent(inspect.getsource(helper))).body[0]
        assert isinstance(function, ast.FunctionDef)
        # These helpers are linked alongside the exact key helper.
        function.body = [
            node for node in function.body if not isinstance(node, ast.ImportFrom)
        ]
        linked_helpers.append(ast.unparse(function))
    utility_code = "\n".join(
        [
            "from __future__ import annotations",
            "import torch",
            "import hashlib",
            "import importlib",
            "import inspect",
            "import types",
            "from pathlib import Path",
            "from typing import cast",
            "from helion._compiler.runtime_tensor_guards import _CanonicalLayoutClassifier",
            textwrap.dedent(inspect.getsource(_standalone_value_key)),
            textwrap.dedent(inspect.getsource(_standalone_call_key)),
            textwrap.dedent(inspect.getsource(_flatten_key_value)),
            *linked_helpers,
        ]
    )
    lines = [
        f"# Standalone CuTe module for {kernel_name!r}; exact per-variant source closures.",
        "from __future__ import annotations",
        f"import inspect as {prefix}_inspect",
        f"from torch._inductor.codecache import PyCodeCache as {prefix}_cache",
        f"{prefix}_policy = {policy.to_dict()!r}",
        f"{prefix}_manifest = {payload!r}",
        f"{prefix}_source_hashes = {tuple(hashlib.sha256(code.encode()).hexdigest() for code in codes)!r}",
        f"{prefix}_sources = {tuple(codes)!r}",
        f"{prefix}_modules = tuple({prefix}_cache.load(code, extra={policy.identity()!r}) for code in {prefix}_sources)",
        f"{prefix}_functions = tuple(module.{kernel_name} for module in {prefix}_modules)",
        f"{prefix}_utilities = {prefix}_cache.load({utility_code!r}, extra={policy.identity()!r})",
        f"{prefix}_signature = {prefix}_inspect.signature({prefix}_functions[0])",
        f"{prefix}_parameters = tuple(name for name, parameter in {prefix}_signature.parameters.items() if parameter.kind in ({prefix}_inspect.Parameter.POSITIONAL_ONLY, {prefix}_inspect.Parameter.POSITIONAL_OR_KEYWORD))",
    ]
    lines += [
        f"{prefix}_guard_plans = {runtime_guard_plans!r}",
        f"{prefix}_classifiers = tuple({prefix}_utilities._export_classifiers(plan) for plan in {prefix}_guard_plans)",
    ]
    if dispatch_keys is None:
        if dynamic_groups is not None:
            lines.append(f"{prefix}_groups = {dict(dynamic_groups)!r}")
        lines.append(
            f"{prefix}_model = {prefix}_cache.load({heuristic_code!r}, extra={policy.identity()!r})"
        )
        if user_key is not None:
            key_code, key_name = _user_key_source(user_key)
            lines.append(
                f"{prefix}_user_key = {prefix}_cache.load({key_code!r}, extra={policy.identity()!r}).{key_name}"
            )
    else:
        lines.append(
            f"{prefix}_variants = {dict(zip(dispatch_keys, range(len(codes)), strict=True))!r}"
        )
    lines += [
        f"def {kernel_name}(*args, **kwargs):",
        f"    bound = {prefix}_signature.bind(*args, **kwargs)",
        "    bound.apply_defaults()",
        f"    normalized = tuple(bound.arguments[name] for name in {prefix}_parameters)",
    ]
    if dispatch_keys is not None:
        lines.append(
            f"    index = {prefix}_utilities._select_guarded_variant(normalized, {prefix}_guard_plans, {prefix}_classifiers, {prefix}_variants, False)"
        )
    else:
        if dynamic_groups is not None:
            lines.append(
                f"    indices = {prefix}_utilities._select_guarded_variant(normalized, {prefix}_guard_plans, {prefix}_classifiers, {prefix}_groups, True)"
            )
        if user_key is not None:
            lines.append(
                f"    normalized = {prefix}_utilities._flatten_key_value({prefix}_user_key(*normalized))"
            )
        lines.append(f"    index = {prefix}_model.key_{kernel_name}(*normalized)")
        lines += [
            f"    if type(index) is not int or not 0 <= index < {len(payload['configs'][kernel_name])}:",
            "        raise ValueError('AOT key returned an invalid config index')",
        ]
        if dynamic_groups is not None:
            lines.append("    index = indices[index]")
    lines.append(f"    return {prefix}_functions[index](*args, **kwargs)")
    content = "\n".join(lines) + "\n"
    compile(content, "<standalone-aot>", "exec")
    if kernel_source_file is None:
        path = output_dir / f"{kernel_name}_standalone.py"
    else:
        source = Path(kernel_source_file)
        path = source.parent / f"{source.stem}_{kernel_name}_standalone.py"
    path = policy_path(path, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path
