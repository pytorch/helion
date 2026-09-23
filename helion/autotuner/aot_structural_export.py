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

import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource

from ..runtime.cute_structural_config import StructuralPolicyError
from .aot_structural_policy import MODEL_MANIFEST
from .aot_structural_policy import model_configs
from .aot_structural_policy import policy_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ..runtime.cute_structural_policy import CuteStructuralPolicy
    from ..runtime.kernel import BoundKernel

RuntimeGuard = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[tuple[int, tuple[int | str, ...]], ...],
]


def export_runtime_guards(
    bound: BoundKernel, args: tuple[object, ...]
) -> tuple[tuple[RuntimeGuard, ...], tuple[object, ...]]:
    """Snapshot declared storage-only classifiers and their input projections.

    The same named classifier executes during exported dispatch. Its source
    module identity is checked when the artifact loads; no tensor contents or
    pointer address is baked into a key. Opaque/content-dependent classifiers
    cannot be relocated by this exporter and retain runtime AOT support.
    """
    from .aot_compile import _standalone_value_key

    indices = {
        name: index for index, name in enumerate(bound.kernel.signature.parameters)
    }

    def projection(source: object) -> tuple[int, tuple[int | str, ...]]:
        if isinstance(source, LocalSource) and source.local_name in indices:
            return indices[source.local_name], ()
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
        if not specialization.reusable_tensor_properties:
            raise StructuralPolicyError(
                "Standalone AOT cannot relocate tensor-content specialization "
                "classifiers; retain runtime AOT for this kernel"
            )
        if (
            not isinstance(classifier, types.FunctionType)
            or "<locals>" in classifier.__qualname__
        ):
            raise StructuralPolicyError(
                "Standalone AOT needs an importable storage classifier"
            )
        source_file = inspect.getsourcefile(classifier)
        if source_file is None:
            raise StructuralPolicyError(
                "Standalone AOT cannot identify storage classifier source"
            )
        guards.append(
            (
                key,
                classifier.__module__,
                classifier.__qualname__,
                hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
                repr(specialization.classifier_identity),
                tuple(sorted(specialization.reusable_tensor_properties)),
                tuple(projection(source) for source in specialization.sources),
            )
        )
    plans = tuple(guards)
    classifiers = _export_classifiers(plans)
    for guard, imported in zip(plans, classifiers, strict=True):
        declared = bound.env.runtime_input_specializations[guard[0]].classifier
        assert isinstance(declared, types.FunctionType)
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
    snapshots = bound.env.bound_runtime_input_specialization_results
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
) -> tuple[types.FunctionType, ...]:
    """Resolve exact imported classifier implementations before dispatch."""
    result = []
    for (
        _key,
        module_name,
        qualified_name,
        digest,
        _identity,
        _properties,
        _paths,
    ) in guards:
        module = importlib.import_module(module_name)
        classifier = module
        for attribute in qualified_name.split("."):
            classifier = getattr(classifier, attribute)
        if not isinstance(classifier, types.FunctionType):
            raise ValueError("Standalone storage classifier implementation changed")
        source_file = inspect.getsourcefile(classifier)
        if (
            source_file is None
            or hashlib.sha256(Path(source_file).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Standalone storage classifier source identity changed")
        result.append(classifier)
    return tuple(result)


def _export_guard_key(
    args: tuple[object, ...],
    guards: tuple[RuntimeGuard, ...],
    classifiers: tuple[Callable[..., object], ...],
) -> tuple[object, ...]:
    from .aot_compile import _standalone_value_key

    def project(path: tuple[int, tuple[int | str, ...]]) -> object:
        index, steps = path
        value = args[index]
        for step in steps:
            if (
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
                classifier(tuple(project(path) for path in guard[-1]))
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


def _model_manifest(
    code: str, name: str, policy: CuteStructuralPolicy
) -> dict[str, Any]:
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
    policy: CuteStructuralPolicy,
    user_key: Callable[..., object] | None,
    dynamic_groups: list[tuple[tuple[object, ...], list[int]]] | None,
    runtime_guards: tuple[RuntimeGuard, ...],
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
    for helper in (_export_call_key, _export_classifiers, _export_guard_key):
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
        f"{prefix}_runtime_guards = {runtime_guards!r}",
        f"{prefix}_classifiers = {prefix}_utilities._export_classifiers({prefix}_runtime_guards)",
        f"{prefix}_signature = {prefix}_inspect.signature({prefix}_functions[0])",
        f"{prefix}_parameters = tuple(name for name, parameter in {prefix}_signature.parameters.items() if parameter.kind in ({prefix}_inspect.Parameter.POSITIONAL_ONLY, {prefix}_inspect.Parameter.POSITIONAL_OR_KEYWORD))",
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
        f"    guards = {prefix}_utilities._export_guard_key(normalized, {prefix}_runtime_guards, {prefix}_classifiers)",
    ]
    if dispatch_keys is not None:
        lines += [
            f"    key = ({prefix}_utilities._export_call_key(normalized), guards)",
            "    try:",
            f"        index = {prefix}_variants[key]",
            "    except KeyError:",
            "        raise ValueError(f'No standalone variant for call signature {key!r}') from None",
        ]
    else:
        if dynamic_groups is not None:
            lines += [
                f"    exact_key = ('exact', ({prefix}_utilities._export_call_key(normalized), guards))",
                f"    dynamic_key = ('dynamic', ({prefix}_utilities._export_call_key(normalized, dynamic=True), guards))",
                f"    indices = {prefix}_groups.get(exact_key, {prefix}_groups.get(dynamic_key))",
                "    if indices is None:",
                "        raise ValueError('No standalone variant for call signature')",
            ]
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
