from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_fp8_vector_store import _config
from test.test_cute_fp8_vector_store import _quantize_rows

import helion
from helion import exc
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from helion._compiler.cute.masked_packet_load_names import (
    MASKED_PACKET_LOAD_HELPER_NAMES,
)
from helion._compiler.cute.packet_store_policy import _STORE_HELPERS
from helion._compiler.cute.packet_store_policy import lower_packet_store_policy
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_fragment import EnumFragment


def test_store_policy_is_independent_of_async_recurrence_config() -> None:
    config = helion.Config(cute_store_policy="l2_evict_last")
    assert config.config == {"cute_store_policy": "l2_evict_last"}
    assert config.cute_store_policy == "l2_evict_last"
    assert config.cute_async_store_policy == "default"
    assert (
        helion.Config.from_json(config.to_json()).cute_store_policy == "l2_evict_last"
    )
    assert helion.Config().cute_store_policy == "default"
    with patch("helion.autotuner.config_spec.torch.version.cuda", "13.0"):
        spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 3))
        spec.normalize(config)
        field = cast("EnumFragment", spec._flat_fields()["cute_store_policy"])
    assert config.cute_store_policy == "l2_evict_last"
    assert field.choices == ("default", "l2_evict_last")


@pytest.mark.parametrize(
    ("capability", "cuda_version", "supported"),
    [
        ((10, 3), "13.0", True),
        ((10, 3), "13.1", True),
        ((10, 0), "13.0", False),
        ((12, 0), "13.0", False),
        ((10, 3), "12.8", False),
        ((10, 3), "14.0", False),
        ((10, 3), None, False),
        (None, "13.0", False),
    ],
)
def test_store_policy_target_gate_controls_normalization_and_search(
    capability: tuple[int, int] | None, cuda_version: str | None, supported: bool
) -> None:
    with patch("helion.autotuner.config_spec.torch.version.cuda", cuda_version):
        spec = ConfigSpec(backend=CuteBackend(), target_device_capability=capability)
        config: dict[str, object] = {"cute_store_policy": "l2_evict_last"}
        spec.normalize(config)
        field = cast("EnumFragment", spec._flat_fields()["cute_store_policy"])
        prior = CuteBackend().config_value_priors(spec)["cute_store_policy"]
    assert config["cute_store_policy"] == ("l2_evict_last" if supported else "default")
    assert field.choices == (
        ("default", "l2_evict_last") if supported else ("default",)
    )
    if not supported:
        assert prior(field, 0) == "default"


@pytest.mark.parametrize("value", ["streaming", "last", True, 1, None])
def test_store_policy_rejects_invalid_values(value: object) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    with pytest.raises(exc.InvalidConfig, match="cute_store_policy"):
        spec.normalize({"cute_store_policy": value})
    repaired = {"cute_store_policy": value}
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_store_policy"] == "default"


def test_store_policy_is_cute_only() -> None:
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        ConfigSpec(backend=TritonBackend()).normalize(
            {"cute_store_policy": "l2_evict_last"}
        )


def test_packet_policy_helper_names_are_registered_without_cuda() -> None:
    imports = CuteBackend().library_imports
    assert set(_STORE_HELPERS.values()) <= imports.keys()
    # Cache-hinted loads plus the predicated packet loads of the sunk V-loop.
    assert {name for name in imports if name.startswith("_cute_load_")} == (
        _CUTE_CACHE_LOAD_HELPER_NAMES | MASKED_PACKET_LOAD_HELPER_NAMES
    )


def _rewrite(
    source: str,
    *,
    policy: str = "l2_evict_last",
    capability: tuple[int, int] | None = (10, 3),
    cuda_version: str | None = "13.0",
) -> ast.Module:
    module = ast.parse(source)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = lower_packet_store_policy(
        function.body,
        policy=policy,
        global_tensor_names=frozenset({"out"}),
        target_device_capability=capability,
        cuda_version=cuda_version,
    )
    return ast.fix_missing_locations(module)


def _packet_source(helper: str, width: int) -> str:
    return f"""
def kernel(data, out, valid):
    values = []
    for lane in cutlass.range_constexpr({width}):
        values.append(data[lane])
    if valid:
        {helper}(out.iterator, values)
"""


@pytest.mark.parametrize(
    ("helper", "hinted", "width"),
    [
        ("_cute_store_u16_vec", "_cute_store_u16x8_l2_evict_last", 8),
        ("_cute_store_u32_vec", "_cute_store_u32x4_l2_evict_last", 4),
        (
            "_cute_store_fp8e4m3fn_vec",
            "_cute_store_fp8e4m3fn_vec_l2_evict_last",
            16,
        ),
        ("_cute_store_fp8e5m2_vec", "_cute_store_fp8e5m2_vec_l2_evict_last", 16),
    ],
)
@pytest.mark.parametrize("scale", [0.5, 1, 2])
def test_only_full_sixteen_byte_packets_select_hint(
    helper: str, hinted: str, width: int, scale: float
) -> None:
    original = _packet_source(helper, int(width * scale))
    rewritten = ast.unparse(_rewrite(original))
    assert (hinted in rewritten) is (scale == 1)
    assert "if valid:" in rewritten
    assert "out.iterator, values" in rewritten


@pytest.mark.parametrize(
    "mutation",
    [
        "values.clear()",
        "consume(values)",
        "values = alternate",
        "alias = values",
    ],
)
def test_unknown_collection_mutation_or_alias_keeps_default_store(
    mutation: str,
) -> None:
    source = _packet_source("_cute_store_fp8e4m3fn_vec", 16).replace(
        "    if valid:", f"    {mutation}\n    if valid:"
    )
    assert "_l2_evict_last" not in ast.unparse(_rewrite(source))


def test_unproven_pointer_or_packet_keeps_default_store() -> None:
    source = _packet_source("_cute_store_fp8e4m3fn_vec", 16)
    for unproven in (
        # A register or shared pointer is not a global packet.
        source.replace("out.iterator", "scratch.iterator"),
        # A rebound tensor parameter no longer names the global argument.
        source.replace("    values = []", "    out = scratch\n    values = []"),
        source.replace(
            "values.append(data[lane])",
            "if lane == 3:\n            break\n        values.append(data[lane])",
        ),
        # A list from an enclosing unrolled loop grows on every iteration.
        source.replace(
            "    for lane in cutlass.range_constexpr(16):\n"
            "        values.append(data[lane])\n"
            "    if valid:\n"
            "        _cute_store_fp8e4m3fn_vec(out.iterator, values)",
            "    for outer in cutlass.range_constexpr(2):\n"
            "        for lane in cutlass.range_constexpr(16):\n"
            "            values.append(data[lane])\n"
            "        if valid:\n"
            "            _cute_store_fp8e4m3fn_vec(out.iterator, values)",
        ),
    ):
        assert unproven != source
        assert "_l2_evict_last" not in ast.unparse(_rewrite(unproven))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "default"},
        {"capability": (10, 0)},
        {"cuda_version": "12.8"},
    ],
)
def test_default_and_unsupported_targets_preserve_ast(kwargs: dict[str, Any]) -> None:
    source = _packet_source("_cute_store_fp8e4m3fn_vec", 16)
    assert ast.dump(_rewrite(source, **kwargs)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("limit", [0, 15, 16, 17, 31, 32])
def test_hint_preserves_full_packet_guard_and_scalar_tail_writes(limit: int) -> None:
    source = """
def kernel(data, out, limit):
    for base in range(0, 32, 16):
        values = []
        for lane in cutlass.range_constexpr(16):
            values.append(data[base + lane])
        if base + 15 < limit:
            _cute_store_fp8e4m3fn_vec(out.iterator + base, values)
        else:
            for lane in range(16):
                if base + lane < limit:
                    (out.iterator + base + lane).store(values[lane])
"""

    class Pointer:
        def __init__(self, values, offset=0):
            self.values = values
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.values, self.offset + offset)

        def store(self, value):
            self.values[self.offset] = value

    def store(pointer, values):
        pointer.values[pointer.offset : pointer.offset + len(values)] = values

    data = [(index * 19 + 128) & 255 for index in range(32)]
    outputs = []
    hinted_offsets = []

    def hinted(pointer, values):
        hinted_offsets.append(pointer.offset)
        store(pointer, values)

    for module in (ast.parse(source), _rewrite(source)):
        output = [0xA5] * 32
        namespace: dict[str, Any] = {
            "cutlass": SimpleNamespace(range_constexpr=range),
            "_cute_store_fp8e4m3fn_vec": store,
            "_cute_store_fp8e4m3fn_vec_l2_evict_last": hinted,
        }
        exec(compile(module, "<packet-store-policy>", "exec"), namespace)
        namespace["kernel"](data, SimpleNamespace(iterator=Pointer(output)), limit)
        outputs.append(output)
    assert outputs[0] == outputs[1] == data[:limit] + [0xA5] * (32 - limit)
    assert hinted_offsets == list(range(0, limit // 16 * 16, 16))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_fp8_codegen_selects_policy_after_packing(
    dtype: torch.dtype, vec_width: int
) -> None:
    args = (torch.empty((37, 32), dtype=torch.bfloat16), dtype, hl.constexpr(32))
    with (
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch("helion.autotuner.config_spec.torch.version.cuda", "13.0"),
    ):
        bound = _cpu_bind(_quantize_rows, args)
        bound.config_spec.target_device_capability = (10, 3)
        config = _config(bound, vec_width)
        config.config["cute_store_policy"] = "l2_evict_last"
        code = bound.to_code(config)
    suffix = "fp8e4m3fn" if dtype is torch.float8_e4m3fn else "fp8e5m2"
    assert (f"_cute_store_{suffix}_vec_l2_evict_last" in code) is (vec_width == 16)
    assert "_helion_persistent_branch_vec_store" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_into(x: torch.Tensor, out: torch.Tensor) -> None:
    rows, width = x.shape
    for row in hl.tile(rows):
        columns = hl.arange(width)
        values = x[row, columns].float()
        maximum = torch.clamp(values.abs().amax(-1), min=1)
        out[row, columns] = (values / maximum[:, None]).to(out.dtype)


def test_misaligned_fp8_output_keeps_scalar_fallback() -> None:
    x = torch.empty((37, 32), dtype=torch.bfloat16)
    storage = torch.empty((37, 33), dtype=torch.float8_e4m3fn)
    out = storage[:, 1:]
    with (
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch("helion.autotuner.config_spec.torch.version.cuda", "13.0"),
    ):
        bound = _cpu_bind(_convert_into, (x, out))
        bound.config_spec.target_device_capability = (10, 3)
        config = _config(bound, 16)
        config.config["cute_store_policy"] = "l2_evict_last"
        code = bound.to_code(config)
    assert "_cute_store_fp8e4m3fn_vec_l2_evict_last" not in code
    assert ".store(" in code


def _constant_bytes(value: Any) -> bytes:
    from helion._compiler.cute._mlir_compat import ir

    operation = value.owner.operation
    if operation.name in ("vector.bitcast", "arith.bitcast"):
        return _constant_bytes(operation.operands[0])
    if operation.name == "vector.extract":
        (index,) = operation.attributes["static_position"]
        width = value.type.width // 8
        return _constant_bytes(operation.operands[0])[
            index * width : (index + 1) * width
        ]
    assert operation.name == "arith.constant", str(operation)
    attribute = operation.attributes["value"]
    if isinstance(value.type, ir.VectorType):
        width = value.type.element_type.width
        elements = list(attribute)
    else:
        width = value.type.width
        elements = [attribute.value]
    mask = (1 << width) - 1
    return b"".join(
        (element & mask).to_bytes(width // 8, "little") for element in elements
    )


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@pytest.mark.parametrize("carrier", ["u8", "u16", "u32", "u64"])
def test_l2_store_preserves_every_bit_of_packed_carriers(carrier: str) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import func
    from cutlass._mlir.dialects import vector

    from helion._compiler.cute import packet_store_helpers
    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir
    from helion._compiler.cute.l2_policy import store_u16x8_l2_evict_last
    from helion._compiler.cute.l2_policy import store_u32x4_l2_evict_last

    class Pointer:
        def toint(self):
            return cutlass.Uint64(0x10000)

    # Distinct lane bytes expose any permutation; with the complement, every
    # bit of every lane is stored as both 0 and 1.
    distinct = bytes((17 * lane + 0x5A) & 255 for lane in range(16))
    for payload in (distinct, bytes(255 - value for value in distinct)):
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("store", ([], []))
                block = function.add_entry_block()
            with (
                ir.InsertionPoint(block),
                patch("helion._compiler.cute.l2_policy.torch.version.cuda", "13.0"),
            ):
                pointer = Pointer()
                if carrier == "u8":
                    vec_utils.store_u8_vec(
                        pointer,
                        [cutlass.Uint8(value) for value in payload],
                        packet_store=packet_store_helpers.store_v16b_l2_evict_last,
                    )
                elif carrier == "u16":
                    store_u16x8_l2_evict_last(
                        pointer,
                        [
                            cutlass.Uint16(
                                int.from_bytes(payload[index : index + 2], "little")
                            )
                            for index in range(0, 16, 2)
                        ],
                    )
                elif carrier == "u32":
                    store_u32x4_l2_evict_last(
                        pointer,
                        [
                            cutlass.Uint32(
                                int.from_bytes(payload[index : index + 4], "little")
                            )
                            for index in range(0, 16, 4)
                        ],
                    )
                else:
                    packed = vector.from_elements(
                        ir.VectorType.get([2], cutlass.Uint64.mlir_type),
                        [
                            cutlass.Uint64(
                                int.from_bytes(payload[index : index + 8], "little")
                            ).ir_value()
                            for index in (0, 8)
                        ],
                    )
                    packet_store_helpers.store_v16b_l2_evict_last(pointer, packed)
                func.ReturnOp([])
            manager: Any = passmanager
            manager.PassManager.parse("builtin.module(canonicalize)").run(
                module.operation
            )
            assembly = next(
                operation
                for operation in block.operations
                if operation.operation.name == "llvm.inline_asm"
            )
            assert "st.global.L2::cache_hint.v4.u32" in str(assembly)
            assert (
                b"".join(_constant_bytes(operand) for operand in assembly.operands[1:5])
                == payload
            )
            assert (
                assembly.operands[5].owner.operation.attributes["value"].value
                == 0x14F0000000000000
            )


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@pytest.mark.parametrize("dtype_name", ["Float8E4M3FN", "Float8E5M2"])
@pytest.mark.parametrize("source_name", ["Float32", "Float16", "BFloat16"])
@pytest.mark.parametrize("runtime", [False, True])
def test_fp8_policy_preserves_original_conversion_packet(
    dtype_name: str, source_name: str, runtime: bool
) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import func
    from cutlass._mlir.dialects import vector

    from helion._compiler.cute import packet_store_helpers
    from helion._compiler.cute import vec_utils
    from helion._compiler.cute._mlir_compat import ir

    source_dtype = getattr(cutlass, source_name)
    destination = getattr(cutlass, dtype_name)
    captured_default = []
    captured_policy = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        packet_type = ir.VectorType.get([4], cutlass.Uint32.mlir_type)
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "convert",
                ([source_dtype.mlir_type] * 16 if runtime else [], [packet_type] * 2),
            )
            block = function.add_entry_block()
        with (
            ir.InsertionPoint(block),
            patch("helion._compiler.cute.l2_policy.torch.version.cuda", "13.0"),
            patch.object(
                vec_utils.cute.arch,
                "store",
                side_effect=lambda pointer, value: captured_default.append(value),
            ),
            patch.object(
                packet_store_helpers,
                "store_u32x4_l2_evict_last",
                side_effect=lambda pointer, values: captured_policy.append(values),
            ),
        ):
            if runtime:
                values = [source_dtype(value) for value in block.arguments]
            else:
                # Delayed literals must retain their scalar rounding, including
                # the first value just beyond an E4M3 midpoint before FP32 rounding.
                literals = [
                    1.0625 + 2**-26,
                    -1.0625 - 2**-26,
                    0.0,
                    -0.0,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                    -float("nan"),
                ] * 2
                values = [source_dtype(value) for value in literals]
            vec_utils._store_fp8_vec(None, values, destination)
            hinted = getattr(
                packet_store_helpers,
                "store_fp8e4m3fn_vec_l2_evict_last"
                if dtype_name == "Float8E4M3FN"
                else "store_fp8e5m2_vec_l2_evict_last",
            )
            hinted(None, values)
            assert len(captured_default) == len(captured_policy) == 1
            original_packet = vector.bitcast(packet_type, captured_default[0])
            retained_packet = vector.from_elements(
                packet_type, [value.ir_value() for value in captured_policy[0]]
            )
            func.ReturnOp([original_packet, retained_packet])
        manager: Any = passmanager
        manager.PassManager.parse("builtin.module(canonicalize,cse)").run(
            module.operation
        )
        original, retained = block.operations[-1].operands
        assert ir.Value.__eq__(original, retained), str(module)
