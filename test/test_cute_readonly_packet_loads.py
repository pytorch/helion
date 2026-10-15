"""CPU tests for opted-in read-only packet hints and their alias/ISA proof."""

from __future__ import annotations

import ast
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_16B_HELPER
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_32B_HELPER
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_CANDIDATE
from helion._compiler.cute.readonly_packet_loads import lower_readonly_packet_loads
import helion.language as hl


def _candidate(bits: int = 16, lanes: int = 16, tensor: str = "source") -> str:
    carrier = {8: "Uint8", 16: "Uint16", 32: "Uint32", 64: "Uint64"}[bits]
    return f"packet = {_CUTE_NC_L2_FIRST_CANDIDATE}({tensor}.iterator + cutlass.Int32(0), ir.VectorType.get([{lanes}], cutlass.{carrier}.mlir_type))\n"


def _rewrite(source: str, **kwargs: Any) -> str:
    options = {
        "tensor_names": {"source", "output", "other"},
        "argument_names": {"source", "output", "other"},
        "proven_disjoint_tensor_pairs": {frozenset({"source", "output"})},
        "target_device_capability": (10, 3),
        "cuda_version": "13.0",
    }
    options.update(kwargs)
    body = lower_readonly_packet_loads(ast.parse(source).body, **options)
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )


@pytest.mark.parametrize(
    "bits,lanes,helper",
    [
        (16, 8, _CUTE_NC_L2_FIRST_16B_HELPER),
        (32, 4, _CUTE_NC_L2_FIRST_16B_HELPER),
        (64, 2, _CUTE_NC_L2_FIRST_16B_HELPER),
        (16, 16, _CUTE_NC_L2_FIRST_32B_HELPER),
        (32, 8, _CUTE_NC_L2_FIRST_32B_HELPER),
        (64, 4, _CUTE_NC_L2_FIRST_32B_HELPER),
    ],
)
def test_supported_full_packets_use_readonly_hint(
    bits: int, lanes: int, helper: str
) -> None:
    code = _rewrite(_candidate(bits, lanes))
    assert helper in code
    assert _CUTE_NC_L2_FIRST_CANDIDATE not in code


def test_stores_to_proven_disjoint_outputs_do_not_disable_readonly_source() -> None:
    code = _rewrite(_candidate() + "_cute_store_u32_vec(output.iterator, values)\n")
    assert _CUTE_NC_L2_FIRST_32B_HELPER in code


def test_each_source_requires_its_own_disjointness_proof() -> None:
    source = (
        _candidate()
        + _candidate(tensor="other")
        + "_cute_store_u32_vec(output.iterator, values)\n"
    )
    code = _rewrite(source)
    assert code.count(_CUTE_NC_L2_FIRST_32B_HELPER) == 1
    assert code.count("cute.arch.load") == 1


@pytest.mark.parametrize(
    "write",
    [
        "_cute_store_u32_vec(source.iterator, values)",
        "(source.iterator + index).store(value)",
        "source[index] = value",
        "cute.arch.atomic_add(source.iterator, value)",
        "_cute_atomic_unknown(output.iterator, source)",
        "_cute_store_u32_vec(unknown_pointer, values)",
        "unknown_write(source)",
        "_cute_store_unknown(output.iterator, source)",
    ],
)
def test_writes_aliases_or_unknown_effects_keep_ordinary_load(write: str) -> None:
    code = _rewrite(_candidate() + write + "\n")
    assert _CUTE_NC_L2_FIRST_32B_HELPER not in code
    assert "cute.arch.load" in code


def test_unknown_external_aliases_are_not_assumed_disjoint() -> None:
    code = _rewrite(
        _candidate() + "_cute_store_u32_vec(output.iterator, values)\n",
        proven_disjoint_tensor_pairs=set(),
    )
    assert "cute.arch.load" in code


@pytest.mark.parametrize(
    "source",
    [
        "alias = source\n(alias.iterator + index).store(value)\n",
        "pointer = source.iterator\n(pointer + index).store(value)\n",
        "if condition:\n    _cute_store_u32_vec(source.iterator, values)\n",
        "for step in range(2):\n    _cute_store_u32_vec(source.iterator, values)\n",
        "if unknown_effect(source):\n    value = 1\n",
        "while unknown_effect(source):\n    value = 1\n",
    ],
)
def test_nested_or_aliased_writes_are_checked_across_the_whole_kernel(
    source: str,
) -> None:
    code = _rewrite(_candidate() + source)
    assert "cute.arch.load" in code


def test_register_caches_and_lists_do_not_count_as_global_writes() -> None:
    source = (
        _candidate()
        + """
cache = cute.make_rmem_tensor(16, cutlass.Float32)
values = []
for lane in cutlass.range_constexpr(16):
    value = cutlass.Uint16(packet[lane]).bitcast(cutlass.BFloat16)
    cache[lane] = cutlass.Float32(value)
    values.append(value)
maximum = cute.arch.warp_reduction_max(value, op=_cute_nan_max, threads_in_group=2)
_cute_store_fp8e4m3fn_vec(output.iterator, values)
"""
    )
    code = _rewrite(source)
    assert _CUTE_NC_L2_FIRST_32B_HELPER in code


@pytest.mark.parametrize(
    "write",
    [
        "values.append(source.iterator)\nvalues[0].store(cutlass.BFloat16(0))",
        "values.append(source)\nvalues[0][index] = value",
        "values.append(source)\n(values[0].iterator + index).store(value)",
    ],
)
def test_private_list_referents_may_still_alias_global_source(write: str) -> None:
    code = _rewrite(_candidate() + "values = []\n" + write + "\n")
    assert "cute.arch.load" in code
    assert _CUTE_NC_L2_FIRST_32B_HELPER not in code


def test_direct_private_list_assignment_does_not_write_its_global_referent() -> None:
    code = _rewrite(
        _candidate() + "values = []\nvalues.append(source)\nvalues[0] = output\n"
    )
    assert _CUTE_NC_L2_FIRST_32B_HELPER in code


@pytest.mark.parametrize(
    "declaration",
    [
        "source = output",
        "cute.arch.load = mystery",
        "namespace = cute\nnamespace.arch.load = mystery",
        "import mystery as cutlass",
        "def _cute_nan_max(a,b):\n    return a",
    ],
)
def test_namespace_or_tensor_rebinding_declines_noncoherent_hint(
    declaration: str,
) -> None:
    code = _rewrite(declaration + "\n" + _candidate())
    assert "cute.arch.load" in code


def test_unknown_reduction_callable_cannot_hide_global_writes() -> None:
    source = (
        _candidate()
        + "maximum = cute.arch.warp_reduction_max(value, op=unknown_op, threads_in_group=2)\n"
    )
    assert "cute.arch.load" in _rewrite(source)


@pytest.mark.parametrize(
    "capability,cuda",
    [
        ((9, 0), "13.0"),
        ((8, 0), "13.0"),
        ((10, 3), "12.9"),
        ((10, 3), None),
        (None, "13.0"),
    ],
)
def test_unsupported_isa_or_toolchain_keeps_ordinary_load(
    capability: tuple[int, int] | None, cuda: str | None
) -> None:
    code = _rewrite(
        _candidate(), target_device_capability=capability, cuda_version=cuda
    )
    assert "cute.arch.load" in code


@pytest.mark.parametrize("lanes", [1, 2, 4, 32])
def test_unsupported_widths_do_not_emit_a_native_packet(lanes: int) -> None:
    code = _rewrite(_candidate(16, lanes))
    assert "cute.arch.load" in code


@pytest.mark.parametrize(
    "packet_bytes,bits", [(16, 16), (16, 32), (16, 64), (32, 16), (32, 32), (32, 64)]
)
def test_runtime_helper_ir_reinterprets_every_loaded_bit(
    packet_bytes: int, bits: int
) -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir.dialects import func

    from helion._compiler.cute import readonly_packet_load_helpers as helpers
    from helion._compiler.cute._mlir_compat import ir

    class Pointer:
        def __init__(self, value: Any) -> None:
            self.value = value

        def toint(self, *, loc: Any = None, ip: Any = None) -> Any:
            return cutlass.Uint64(self.value)

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        vector_type = ir.VectorType.get(
            [packet_bytes * 8 // bits], ir.IntegerType.get_signless(bits)
        )
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "packet", ([cutlass.Uint64.mlir_type], [vector_type])
            )
            block = function.add_entry_block()
        with ir.InsertionPoint(block):
            helper = (
                helpers.load_nc_l2_evict_first_32b
                if packet_bytes == 32
                else helpers.load_nc_l2_evict_first_16b
            )
            result = helper(Pointer(block.arguments[0]), vector_type)
            func.ReturnOp([result])
        assert module.operation.verify()
        code = str(module)
    assert "ld.global.nc" in code
    assert "v8.b32" in code if packet_bytes == 32 else "v4.b32" in code
    assert ("createpolicy" in code) is (packet_bytes == 16)
    assert "arith.truncf" not in code and "arith.extf" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_into(x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        maximum = data.abs().amax(-1)
        output[row, column] = (data / (maximum[:, None] + 1)).to(output.dtype)
    return output


def _generate(x: torch.Tensor, output: torch.Tensor, width: int) -> str:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_rows_into, (x, output))
        config = helion.Config(
            block_sizes=[64],
            num_threads=[32, 32 // width],
            cute_vector_widths=[width, 1],
            cute_lane_layouts=["blocked", "blocked"],
            cute_reduction_reloads=["register"],
            load_eviction_policies="nc_l2_first",
        )
        return bound.to_code(config)


@pytest.mark.parametrize(
    "width,helper",
    [(8, _CUTE_NC_L2_FIRST_16B_HELPER), (16, _CUTE_NC_L2_FIRST_32B_HELPER)],
)
def test_actual_codegen_uses_no_write_alias_proof_and_packet_width(
    width: int, helper: str
) -> None:
    code = _generate(
        torch.empty((128, 32), dtype=torch.bfloat16),
        torch.empty((128, 32), dtype=torch.float8_e4m3fn),
        width,
    )
    assert helper in code, code
    assert _CUTE_NC_L2_FIRST_CANDIDATE not in code


def test_actual_codegen_declines_overlapping_external_storage() -> None:
    x = torch.empty((128, 32), dtype=torch.bfloat16)
    output = x.view(torch.float8_e4m3fn)[:, :32]
    with pytest.raises(exc.BackendUnsupported, match="potentially aliasing write"):
        _generate(x, output, 16)


def test_actual_codegen_declines_32b_misalignment_from_input_view() -> None:
    storage = torch.empty(128 * 32 + 8, dtype=torch.bfloat16)
    x = storage[8:].view(128, 32)
    code = _generate(x, torch.empty((128, 32), dtype=torch.float8_e4m3fn), 16)
    assert _CUTE_NC_L2_FIRST_32B_HELPER not in code, code
