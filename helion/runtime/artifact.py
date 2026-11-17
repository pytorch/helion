"""Explicitly selected, immutable Triton programs exported from native bindings.

Artifacts contain trusted executable Python. Their digest detects corruption,
not an untrusted author's intent. Loading never consults the original kernel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import tempfile
import threading
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codecache import PyCodeCache
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .._compat import device_num_sm
from .._compat import get_device_name
from .._compat import target_device_capability
from .._compiler.output_code_utils import close_artifact_source
from .._utils import indexing_uses_tensor_descriptor
from ..autotuner.aot_compile import _standalone_call_key
from ..autotuner.aot_compile import generate_standalone_file
from .kernel import Kernel
from .kernel import OutputCodeOptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    import os

    from .kernel import ConfigLike

_FORMAT = "helion.explicit-triton-program"
_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ArtifactCase:
    """One explicit input signature and configuration to compile into an artifact."""

    args: tuple[object, ...]
    config: ConfigLike
    kwargs: dict[str, object] = dataclasses.field(default_factory=dict)


def _check_execution_context() -> None:
    if (
        torch._C._is_torch_function_mode_enabled()
        or _get_current_dispatch_mode_stack()
        or torch.is_autocast_enabled("cpu")
        or torch.is_autocast_enabled("cuda")
    ):
        raise ValueError(
            "Artifacts require ordinary execution without Torch modes or autocast"
        )


def _runtime_identity() -> dict[str, object]:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    return {
        "python": list(sys.version_info[:3]),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "system": platform.system(),
        "machine": platform.machine(),
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
    }


def _target_identity(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        return {"device": str(device), "name": platform.machine()}
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Artifact target requires an available CUDA/ROCm device")
    capability = target_device_capability(device)
    return {
        "device": str(device),
        "name": get_device_name(device),
        "capability": None if capability is None else list(capability),
        "multiprocessors": device_num_sm(device),
    }


def _value_spec(
    value: object,
    tensors: list[torch.Tensor],
    *,
    allow_tensor: bool = True,
    seen: frozenset[int] = frozenset(),
) -> object:
    if type(value) in (tuple, list, dict):
        if id(value) in seen:
            raise ValueError("Artifact inputs cannot contain cyclic containers")
        seen = seen | {id(value)}
    if isinstance(value, torch.Tensor):
        if not allow_tensor or type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise ValueError("Artifacts require ordinary Tensor or Parameter inputs")
        tensor = cast("torch.Tensor", value)
        if (
            tensor.layout != torch.strided
            or tensor.is_quantized
            or tensor.requires_grad
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            raise ValueError(
                "Artifacts require strided inference tensors without conjugate or negative views"
            )
        storage = tensor.untyped_storage()
        object_alias = next(
            (i for i, other in enumerate(tensors) if other is tensor), len(tensors)
        )
        storage_alias = next(
            (
                i
                for i, other in enumerate(tensors)
                if other.untyped_storage()._cdata == storage._cdata
            ),
            len(tensors),
        )
        tensors.append(tensor)
        return {
            "type": "tensor",
            "tensor_type": "parameter"
            if type(tensor) is torch.nn.Parameter
            else "tensor",
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(),
            "storage_bytes": storage.nbytes(),
            "object_alias": object_alias,
            "storage_alias": storage_alias,
        }
    if value is None:
        return {"type": "none"}
    if type(value) in (bool, int, str):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, (torch.dtype, torch.device)):
        return {
            "type": "dtype" if isinstance(value, torch.dtype) else "device",
            "value": str(value),
        }
    if type(value) in (tuple, list):
        return {
            "type": type(value).__name__,
            "items": [
                _value_spec(item, tensors, allow_tensor=allow_tensor, seen=seen)
                for item in cast("Sequence[object]", value)
            ],
        }
    if type(value) is dict:
        return {
            "type": "dict",
            "items": [
                [
                    _value_spec(key, [], allow_tensor=False, seen=seen),
                    _value_spec(item, tensors, allow_tensor=allow_tensor, seen=seen),
                ]
                for key, item in value.items()
            ],
        }
    raise ValueError(f"Unsupported artifact input type {type(value).__name__}")


def _literal_from_spec(spec: object) -> object:
    if not isinstance(spec, dict):
        raise ValueError("Invalid artifact default value")
    kind, value = spec.get("type"), spec.get("value")
    if kind == "none":
        return None
    if kind in ("bool", "int", "str") and type(value).__name__ == kind:
        return value
    if kind == "float" and isinstance(value, str):
        return float.fromhex(value)
    if kind == "bytes" and isinstance(value, str):
        return bytes.fromhex(value)
    if kind == "device" and isinstance(value, str):
        return torch.device(value)
    if kind == "dtype" and isinstance(value, str) and value.startswith("torch."):
        dtype = vars(torch).get(value.removeprefix("torch."))
        if isinstance(dtype, torch.dtype):
            return dtype
    if kind in ("tuple", "list") and isinstance(spec.get("items"), list):
        items = [_literal_from_spec(item) for item in spec["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict" and isinstance(spec.get("items"), list):
        return {
            _literal_from_spec(key): _literal_from_spec(item)
            for key, item in spec["items"]
        }
    raise ValueError("Invalid artifact default value")


def _signature_spec(signature: inspect.Signature) -> list[dict[str, object]]:
    result = []
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError("Artifacts require a fixed positional parameter signature")
        item: dict[str, object] = {"name": parameter.name, "kind": parameter.kind.name}
        if parameter.default is not inspect.Parameter.empty:
            item["default"] = _value_spec(parameter.default, [], allow_tensor=False)
        result.append(item)
    return result


def _signature_from_spec(spec: object) -> inspect.Signature:
    if not isinstance(spec, list):
        raise ValueError("Invalid artifact signature")
    kinds = {
        "POSITIONAL_ONLY": inspect.Parameter.POSITIONAL_ONLY,
        "POSITIONAL_OR_KEYWORD": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    parameters = []
    for item in spec:
        if not isinstance(item, dict) or item.get("kind") not in kinds:
            raise ValueError("Invalid artifact parameter")
        parameters.append(
            inspect.Parameter(
                item["name"],
                kinds[item["kind"]],
                default=_literal_from_spec(item["default"])
                if "default" in item
                else inspect.Parameter.empty,
            )
        )
    return inspect.Signature(parameters)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def export_artifact(
    kernel: Kernel,
    cases: Sequence[ArtifactCase],
    path: str | os.PathLike[str],
) -> Path:
    """Compile explicit static cases and atomically write one trusted program.

    CPU examples support code generation and contract inspection. Triton
    artifacts require accelerator tensors for execution. Autotuning is never run.
    """
    _check_execution_context()
    if not isinstance(kernel, Kernel) or kernel.settings.backend != "triton":
        raise ValueError("Explicit artifacts currently support Triton kernels")
    if kernel.settings.distributed or kernel._declares_process_group:
        raise ValueError("Distributed kernels cannot be exported as artifacts")
    if not cases:
        raise ValueError("Artifact export requires at least one explicit case")
    signature = _signature_spec(kernel.signature)
    codes: list[str] = []
    dispatch_keys: list[tuple[object, ...]] = []
    contracts: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, ArtifactCase) or case.config is None:
            raise ValueError("Every artifact case requires an explicit configuration")
        try:
            args = kernel.normalize_args(*case.args, **case.kwargs)
        except TypeError as error:
            raise ValueError(str(error)) from error
        tensors: list[torch.Tensor] = []
        inputs = _value_spec(args, tensors)
        if not tensors or len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("Each artifact case requires tensors on one device")
        target = _target_identity(tensors[0].device)
        dispatch_key = _standalone_call_key(args)
        if dispatch_key in dispatch_keys:
            raise ValueError("Artifact cases have ambiguous static dispatch signatures")
        native = Kernel(
            kernel.fn,
            configs=[case.config],
            settings=kernel.settings.copy(static_shapes=True, force_autotune=False),
        )
        bound = native.bind(args)
        config = bound._normalized_config_copy(case.config)
        if bound.env._is_distributed or bound.env.runtime_input_specializations:
            raise ValueError(
                "Artifact input semantics require unsupported runtime specialization"
            )
        if any(
            indexing_uses_tensor_descriptor(value)
            for value in (config.indexing, config.atomic_indexing)
        ):
            raise ValueError("Tensor-descriptor indexing is not supported by artifacts")
        source = bound.to_code(
            config, options=OutputCodeOptions(allow_helion_deps=False)
        )
        if bound.env.runtime_input_specializations:
            raise ValueError(
                "Code generation introduced unsupported runtime specialization"
            )
        if _value_spec(args, []) != inputs:
            raise ValueError("Artifact inputs changed during export")
        codes.append(close_artifact_source(source))
        dispatch_keys.append(dispatch_key)
        contracts.append(
            {"inputs": inputs, "target": target, "config": config.to_json()}
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".helion-artifact-", dir=destination.parent
    ) as temporary:
        standalone = generate_standalone_file(
            kernel.name,
            codes,
            "",
            Path(temporary),
            dispatch_keys=dispatch_keys,
        )
        source = close_artifact_source(standalone.read_text())
    payload = {
        "format": _FORMAT,
        "version": _VERSION,
        "entrypoint": kernel.name,
        "source": source,
        "signature": signature,
        "runtime": _runtime_identity(),
        "cases": contracts,
    }
    envelope = {
        "payload": payload,
        "sha256": hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(_canonical_json(envelope))
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


class FrozenKernel:
    """An explicitly loaded program with an exact, inference-only input contract."""

    def __init__(
        self, payload: dict[str, object], signature: inspect.Signature
    ) -> None:
        self._payload = payload
        self.__signature__ = signature
        self._run: Callable[..., object] | None = None
        self._load_lock = threading.Lock()

    def __call__(self, *args: object, **kwargs: object) -> object:
        _check_execution_context()
        runtime = cast("dict[str, object]", self._payload["runtime"])
        if runtime["default_dtype"] != str(torch.get_default_dtype()) or runtime[
            "default_device"
        ] != str(torch.get_default_device()):
            raise ValueError("Artifact tensor-factory defaults have changed")
        try:
            bound = self.__signature__.bind(*args, **kwargs)
        except TypeError as error:
            raise ValueError(str(error)) from error
        bound.apply_defaults()
        normalized = tuple(bound.arguments.values())
        tensors: list[torch.Tensor] = []
        inputs = _value_spec(normalized, tensors)
        cases = cast("list[dict[str, object]]", self._payload["cases"])
        matched = next((case for case in cases if case["inputs"] == inputs), None)
        if matched is None:
            raise ValueError(
                "No artifact case matches the input metadata, values and alias relationships"
            )
        if not tensors or tensors[0].device.type == "cpu":
            raise ValueError(
                "Triton artifacts require accelerator tensors for execution"
            )
        run = self._run
        if run is None:
            with self._load_lock:
                if self._run is None:
                    module = PyCodeCache.load(cast("str", self._payload["source"]))
                    self._run = getattr(
                        module, cast("str", self._payload["entrypoint"])
                    )
                run = self._run
        return run(*normalized)


def load_artifact(path: str | os.PathLike[str]) -> FrozenKernel:
    """Load a trusted artifact without importing or recompiling its original kernel."""
    _check_execution_context()
    try:
        envelope = json.loads(Path(path).read_text())
        payload = envelope["payload"]
        if (
            envelope["sha256"]
            != hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        ):
            raise ValueError("Artifact integrity check failed")
        if (
            payload["format"] != _FORMAT
            or type(payload["version"]) is not int
            or payload["version"] != _VERSION
        ):
            raise ValueError("Unsupported artifact format or version")
        if payload["runtime"] != _runtime_identity():
            raise ValueError("Artifact runtime identity is incompatible")
        if (
            not isinstance(payload["source"], str)
            or not isinstance(payload["entrypoint"], str)
            or not payload["entrypoint"].isidentifier()
        ):
            raise ValueError("Invalid artifact program")
        signature = _signature_from_spec(payload["signature"])
        cases = payload["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("Invalid artifact input cases")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("inputs"), dict):
                raise ValueError("Invalid artifact input contract")
            target = case["target"]
            if target != _target_identity(torch.device(target["device"])):
                raise ValueError("Artifact target device is incompatible")
        if close_artifact_source(payload["source"]) != payload["source"]:
            raise ValueError("Artifact source is not closed and canonical")
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid artifact file") from error
    return FrozenKernel(payload, signature)
