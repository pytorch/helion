"""Resolve opt-in 16/32-byte cache-policy requests with whole-kernel alias proof.

Candidates come from the ordinary packet emitter's existing alignment/mask
proof. This pass changes only their load opcode, never its address, dtype,
mask or order. Unsafe or unsupported candidates remain ordinary loads.
"""

from __future__ import annotations

import ast
from collections import Counter

from ..tile_strategy import _is_proven_relocatable_call
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cache_policy_loads import _CUTE_NC_L2_FIRST_16B_HELPER
from .cache_policy_loads import _CUTE_NC_L2_FIRST_32B_HELPER
from .cache_policy_loads import _CUTE_NC_L2_FIRST_CANDIDATE
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .fuse_two_pass_loads import _tensor_roots_may_alias

_STORE_HELPERS = frozenset(
    {
        "_cute_store_u8_vec",
        "_cute_store_u16_vec",
        "_cute_store_u32_vec",
        "_cute_store_fp8e4m3fn_vec",
        "_cute_store_fp8e5m2_vec",
        "_cute_store_u16x8_l2_evict_last",
        "_cute_store_u32x4_l2_evict_last",
    }
)
_ATOMIC_MEMBERS = frozenset(
    {
        "atomic_add",
        "atomic_max",
        "atomic_min",
        "atomic_cas",
        "atomic_xchg",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
    }
)
_PURE_CONSTRUCTORS = frozenset(
    {"range", "cutlass.range_constexpr", "cute.make_rmem_tensor"}
)
_WARP_REDUCTIONS = frozenset(
    {
        "cute.arch.warp_reduction",
        "cute.arch.warp_reduction_sum",
        "cute.arch.warp_reduction_max",
    }
)
_PURE_REDUCTION_OPS = frozenset(
    {"_cute_nan_max", "_cute_nan_min", "operator.add", "operator.mul"}
)
_RESERVED = (
    frozenset(
        {
            "cute",
            "cutlass",
            "ir",
            "math",
            "operator",
            "range",
            "_cute_nan_max",
            "_cute_nan_min",
        }
    )
    | _STORE_HELPERS
    | _CUTE_CACHE_LOAD_HELPER_NAMES
)
_ELEMENT_BITS = {
    **{
        f"cutlass.{name}": 8 for name in ("Int8", "Uint8", "Float8E4M3FN", "Float8E5M2")
    },
    **{f"cutlass.{name}": 16 for name in ("Int16", "Uint16", "BFloat16", "Float16")},
    **{f"cutlass.{name}": 32 for name in ("Int32", "Uint32", "Float32")},
    **{f"cutlass.{name}": 64 for name in ("Int64", "Uint64", "Float64")},
}


def readonly_packet_load_supported(
    target_device_capability: tuple[int, int] | None, cuda_version: str | None
) -> bool:
    """PTX v8.b32 global loads require Blackwell and CUDA13 or newer."""
    major = cuda_version.split(".", 1)[0] if cuda_version else ""
    return (
        target_device_capability is not None
        and target_device_capability >= (10, 0)
        and major.isdigit()
        and int(major) >= 13
    )


def _root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bindings_are_intact(
    module: ast.Module, tensor_names: set[str], argument_names: set[str]
) -> bool:
    protected = _RESERVED | tensor_names
    if argument_names & _RESERVED:
        return False
    for node in ast.walk(module):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            return False
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in protected
        ):
            return False
        if isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            return False
        if isinstance(node, ast.Assign) and _root(node.value) in _RESERVED:
            return False
    return True


def _private_containers(module: ast.Module, argument_names: set[str]) -> set[str]:
    writes = Counter(
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    )
    private = set()
    for node in ast.walk(module):
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue
        name, value = node.targets[0].id, node.value
        if name in argument_names or writes[name] != 1:
            continue
        if isinstance(value, ast.List) and not value.elts:
            private.add(name)
        if (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "cute.make_rmem_tensor"
        ):
            private.add(name)
    return private


def _known_write(call: ast.Call, private: set[str]) -> bool:
    name = ast.unparse(call.func)
    if name in _STORE_HELPERS or name == "cute.arch.store":
        return True
    if name.startswith("cute.arch.") and name.rsplit(".", 1)[-1] in _ATOMIC_MEMBERS:
        return True
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"store", "__setitem__"}
        and (
            (
                call.func.attr == "__setitem__"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in private
            )
            or any(
                isinstance(node, ast.Attribute) and node.attr == "iterator"
                for node in ast.walk(call.func.value)
            )
        )
    )


def _write_roots(
    module: ast.Module, tensor_names: set[str], private: set[str]
) -> list[frozenset[str] | None]:
    result = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and _is_store_call(node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "__setitem__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in private
            ):
                continue
            result.append(
                _store_tensor_roots(node, tensor_names)
                if _known_write(node, private)
                else None
            )
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            root = _root(node.value)
            if isinstance(node.value, ast.Name) and node.value.id in private:
                continue
            result.append(
                frozenset({root}) if root is not None and root in tensor_names else None
            )
    return result


def _effects_are_known(
    module: ast.Module, tensor_names: set[str], private: set[str]
) -> bool:
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if _is_store_call(node):
            if not _known_write(node, private):
                return False
            continue
        if (
            name in _PURE_CONSTRUCTORS
            or name in _CUTE_CACHE_LOAD_HELPER_NAMES
            or name == "cute.arch.load"
        ):
            continue
        if name in _WARP_REDUCTIONS:
            operators = node.args[1:] if name == "cute.arch.warp_reduction" else []
            operators.extend(
                keyword.value for keyword in node.keywords if keyword.arg == "op"
            )
            if any(
                not isinstance(operator, ast.Lambda)
                and ast.unparse(operator) not in _PURE_REDUCTION_OPS
                for operator in operators
            ):
                return False
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "__getitem__"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in private
        ):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            if _tensor_arg_roots(node.func.value, tensor_names) is None and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id in private
            ):
                return False
            continue
        if not _is_proven_relocatable_call(
            node, allow_load=False, allow_reduction=False
        ):
            return False
    return True


def _packet_bytes(call: ast.Call) -> int | None:
    if len(call.args) != 2 or call.keywords:
        return None
    vector = call.args[1]
    if (
        not isinstance(vector, ast.Call)
        or ast.unparse(vector.func) != "ir.VectorType.get"
        or len(vector.args) != 2
        or vector.keywords
    ):
        return None
    shape, dtype = vector.args
    if (
        not isinstance(shape, ast.List)
        or len(shape.elts) != 1
        or not isinstance(shape.elts[0], ast.Constant)
        or type(shape.elts[0].value) is not int
        or not isinstance(dtype, ast.Attribute)
        or dtype.attr != "mlir_type"
    ):
        return None
    bits = shape.elts[0].value * _ELEMENT_BITS.get(ast.unparse(dtype.value), 0)
    return bits // 8 if bits in (128, 256) else None


def lower_readonly_packet_loads(
    body: list[ast.stmt],
    *,
    tensor_names: set[str],
    argument_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    target_device_capability: tuple[int, int] | None,
    cuda_version: str | None,
) -> list[ast.stmt]:
    """Resolve every request to a proven NC load or an ordinary load.

    A source is read-only only if all stores/atomics anywhere in the kernel
    are resolved and proven disjoint from its storage. Unknown effects,
    rebinding or incomplete alias facts decline the noncoherent opcode.
    Alignment and the exact packet mask come from the existing emitter.
    """
    module = ast.Module(body=body, type_ignores=[])
    candidates = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == _CUTE_NC_L2_FIRST_CANDIDATE
    ]
    if not candidates:
        return body
    private = _private_containers(module, argument_names)
    eligible = (
        readonly_packet_load_supported(target_device_capability, cuda_version)
        and _bindings_are_intact(module, tensor_names, argument_names)
        and _effects_are_known(module, tensor_names, private)
    )
    writes = _write_roots(module, tensor_names, private) if eligible else []
    for call in candidates:
        source = _tensor_arg_roots(call.args[0], tensor_names) if call.args else None
        packet_bytes = _packet_bytes(call)
        readonly = (
            eligible
            and packet_bytes is not None
            and source is not None
            and len(source) == 1
            and not any(
                _tensor_roots_may_alias(
                    source, destination, proven_disjoint_tensor_pairs
                )
                for destination in writes
            )
        )
        call.func = (
            ast.Name(
                id=_CUTE_NC_L2_FIRST_32B_HELPER
                if packet_bytes == 32
                else _CUTE_NC_L2_FIRST_16B_HELPER,
                ctx=ast.Load(),
            )
            if readonly
            else ast.parse("cute.arch.load", mode="eval").body
        )
    return body
