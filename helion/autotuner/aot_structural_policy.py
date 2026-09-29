"""Versioned policy partitions for AOT data and generated models.

The policy selects a config schema before binding. It is never a model feature
or a search dimension. The unversioned path keeps its existing file formats.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Sequence
import copy
import hashlib
import importlib.util
import json
from typing import TYPE_CHECKING

from ..runtime.cute_structural_config import CuteStructuralConfig
from ..runtime.cute_structural_config import StructuralPolicyError
from ..runtime.cute_structural_config import config_artifact_json
from ..runtime.cute_structural_config import decode_structural_policy
from ..runtime.cute_structural_config import require_same_structural_policy

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from ..runtime.config import Config
    from ..runtime.cute_structural_policy import CuteStructuralPolicy

MODEL_MANIFEST = "_HELION_AOT_STRUCTURAL_MANIFEST"
_MODEL_SCHEMA = "helion.aot.structural_model"
_CONFIG_READER = "_helion_aot_structural_config"


def load_policy_module(path: Path, source: bytes) -> ModuleType:
    """Execute the exact bytes used for the model's identity, bypassing .pyc.

    A same-size edit within a filesystem timestamp tick must not turn a new
    source digest into an old bytecode module. Legacy loading is unchanged.
    """
    spec = importlib.util.spec_from_file_location("heuristic", path)
    if spec is None or spec.loader is None:
        raise StructuralPolicyError("Cannot import policy-bearing AOT model")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), vars(module))
    return module


def policy_suffix(policy: CuteStructuralPolicy | None) -> str:
    return "" if policy is None else f"__policy_{policy.identity()}"


def policy_path(path: Path, policy: CuteStructuralPolicy | None) -> Path:
    return path.with_name(f"{path.stem}{policy_suffix(policy)}{path.suffix}")


def measurement_files(data_dir: Path, hardware_id: str) -> list[Path]:
    """Enumerate legacy first, then explicit partitions deterministically."""
    legacy = data_dir / f"measurements_{hardware_id}.csv"
    return ([legacy] if legacy.exists() else []) + sorted(
        data_dir.glob(f"measurements_{hardware_id}__policy_*.csv")
    )


def measurement_metadata(policy: CuteStructuralPolicy | None) -> dict[str, str]:
    if policy is None:
        return {}
    return {
        "aot_policy_version": "1",
        "cute_structural_policy": policy.to_json(),
        "cute_structural_policy_id": policy.identity(),
    }


def measurement_policy(row: Mapping[str, str]) -> CuteStructuralPolicy | None:
    fields = {
        "aot_policy_version",
        "cute_structural_policy",
        "cute_structural_policy_id",
    }
    if not fields.intersection(row):
        return None
    if not fields.issubset(row) or row["aot_policy_version"] != "1":
        raise StructuralPolicyError("Unsupported AOT measurement policy version")
    try:
        policy = decode_structural_policy(json.loads(row["cute_structural_policy"]))
    except (TypeError, ValueError) as error:
        raise StructuralPolicyError("Invalid AOT measurement policy") from error
    if row["cute_structural_policy_id"] != policy.identity():
        raise StructuralPolicyError("AOT measurement policy identity mismatch")
    return policy


def measurement_config_hash(config: Config, policy: CuteStructuralPolicy | None) -> str:
    return hashlib.sha256(config_artifact_json(config, policy).encode()).hexdigest()[
        :16
    ]


def model_configs(
    module: ModuleType,
    kernel_name: str,
    policy: CuteStructuralPolicy | None,
    *,
    required: bool = True,
) -> tuple[CuteStructuralConfig, ...] | None:
    """Check a model before running its key/config selector or reading knobs.

    ``required=False`` retains the earlier manually authored envelope-returning
    config callback API. Such a callback is only usable after explicit binding;
    pre-binding key selection and standalone enumeration require this manifest.
    """
    payload = vars(module).get(MODEL_MANIFEST)
    if payload is None:
        if required and policy is not None:
            raise StructuralPolicyError("Policy-bearing AOT model needs a manifest")
        return None
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "version",
        "policy",
        "policy_id",
        "configs",
    }:
        raise StructuralPolicyError("Expected an AOT structural model manifest")
    if (
        payload["schema"] != _MODEL_SCHEMA
        or type(payload["version"]) is not int
        or payload["version"] != 1
    ):
        raise StructuralPolicyError("Unsupported AOT structural model version")
    recorded = decode_structural_policy(payload["policy"])
    require_same_structural_policy(recorded, policy, context="AOT model")
    if payload["policy_id"] != recorded.identity():
        raise StructuralPolicyError("AOT model policy identity mismatch")
    entries = payload["configs"]
    if not isinstance(entries, Mapping) or kernel_name not in entries:
        raise StructuralPolicyError("AOT model does not contain the requested kernel")
    result: dict[str, tuple[CuteStructuralConfig, ...]] = {}
    for name, records in entries.items():
        if not isinstance(name, str) or not isinstance(records, list) or not records:
            raise StructuralPolicyError("Expected ordered AOT config envelopes")
        configs: list[CuteStructuralConfig] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise StructuralPolicyError("Expected an AOT config envelope")
            config = CuteStructuralConfig.from_dict(record)
            require_same_structural_policy(
                config.policy, recorded, context="AOT config"
            )
            configs.append(config)
        result[name] = tuple(configs)
    return result[kernel_name]


def attach_model_policy(
    code: str,
    configs: Mapping[str, Sequence[Config]],
    policy: CuteStructuralPolicy | None,
    *,
    feature_names: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Wrap config returns after model assembly, leaving numeric selectors exact."""
    if policy is None:
        return code
    tree = ast.parse(code)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    if {MODEL_MANIFEST, _CONFIG_READER}.intersection(names):
        raise StructuralPolicyError("AOT model uses a reserved policy symbol")
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    lines = code.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for name, selected in configs.items():
        if not selected or f"key_{name}" not in functions:
            raise StructuralPolicyError("AOT backend must emit an indexed key selector")
        fn = functions.get(f"autotune_{name}")
        if fn is None or fn.end_lineno is None or fn.decorator_list:
            raise StructuralPolicyError(
                "AOT backend must emit an undecorated config selector"
            )
        replacements.append(
            (
                fn.lineno - 1,
                fn.end_lineno,
                (
                    f"def autotune_{name}(*args):\n"
                    f"    return {_CONFIG_READER}.from_dict(\n"
                    f"        {MODEL_MANIFEST}['configs'][{name!r}][key_{name}(*args)]\n"
                    "    )\n"
                ),
            )
        )
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    payload = {
        "schema": _MODEL_SCHEMA,
        "version": 1,
        "policy": policy.to_dict(),
        "policy_id": policy.identity(),
        "configs": {
            name: [
                CuteStructuralConfig(config, policy).to_dict() for config in selected
            ]
            for name, selected in configs.items()
        },
    }
    result = (
        "".join(lines)
        + f"\nfrom helion.runtime.cute_structural_config import CuteStructuralConfig as {_CONFIG_READER}\n"
        + f"\n{MODEL_MANIFEST} = {payload!r}\n"
    )
    if feature_names is not None:
        for name, features in feature_names.items():
            result += _feature_evaluator(functions[f"key_{name}"], name, features)
    return result


def _feature_evaluator(
    key: ast.FunctionDef, name: str, feature_names: Sequence[str]
) -> str:
    """Reuse a backend's exact selector on already measured feature values.

    Both installed backends emit direct feature assignments with this common
    generator. Substitute only those exact assignments; retain every decision,
    nearest-neighbor comparison and config index. Runtime key code is untouched.
    """
    from .heuristic_generator import generate_feature_extraction_code

    expected: dict[str, tuple[str, str]] = {}
    for feature in feature_names:
        source = "def extract(*args):\n" + generate_feature_extraction_code([feature])
        function = ast.parse(source).body[0]
        assert isinstance(function, ast.FunctionDef)
        for statement in function.body:
            if isinstance(statement, ast.Assign) and isinstance(
                statement.targets[0], ast.Name
            ):
                expected[statement.targets[0].id] = (feature, ast.dump(statement))
    function = copy.deepcopy(key)
    function.name = f"select_config_{name}"
    argument = "_helion_measured_features"
    while any(
        isinstance(node, ast.Name) and node.id == argument
        for node in ast.walk(function)
    ):
        argument += "_"
    function.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=argument)],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    function.returns = None
    for index, statement in enumerate(function.body):
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in expected
        ):
            feature, original = expected[statement.targets[0].id]
            if ast.dump(statement) != original:
                raise StructuralPolicyError(
                    "AOT backend has an unsupported feature extraction"
                )
            function.body[index] = ast.Assign(
                targets=statement.targets,
                value=ast.Subscript(
                    value=ast.Name(id=argument, ctx=ast.Load()),
                    slice=ast.Constant(value=feature),
                    ctx=ast.Load(),
                ),
            )
    if any(
        isinstance(node, ast.Name) and node.id == "args" for node in ast.walk(function)
    ):
        raise StructuralPolicyError(
            "AOT backend uses arguments outside feature extraction"
        )

    class Returns(ast.NodeTransformer):
        def visit_Return(self, node: ast.Return) -> ast.Return:
            assert node.value is not None
            return ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=_CONFIG_READER, ctx=ast.Load()),
                        attr="from_dict",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Subscript(
                            value=ast.parse(
                                f"{MODEL_MANIFEST}['configs'][{name!r}]", mode="eval"
                            ).body,
                            slice=node.value,
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            # Nested dtype extraction helpers return feature values, not indices.
            return node

    function.body = [Returns().visit(statement) for statement in function.body]
    ast.fix_missing_locations(function)
    return "\n" + ast.unparse(function) + "\n"
