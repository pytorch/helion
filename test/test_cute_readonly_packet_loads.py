"""CPU codegen for opted-in read-only packet loads and their alias/ISA gate.

``load_eviction_policies="nc_l2_first"`` emits the noncoherent L2 evict-first
helper only for a full 16/32-byte packet of a source that no store or atomic
in the kernel can alias, on SM100+ with CUDA 13+; every other site keeps its
ordinary load.  The GPU numerics are at the end of the module.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import helion
from helion._compiler.cute.cache_policy_loads import _CUTE_NC_L2_FIRST_HELPERS
from helion._compiler.cute.cutedsl_compat import (
    fixed_l2_evict_last_store_policy_supported,
)
from helion._compiler.cute.cutedsl_compat import readonly_packet_load_supported
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_NC_16B = _CUTE_NC_L2_FIRST_HELPERS[16]
_NC_32B = _CUTE_NC_L2_FIRST_HELPERS[32]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_into(x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        maximum = data.abs().amax(-1)
        output[row, column] = (data / (maximum[:, None] + 1)).to(output.dtype)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_in_place(x: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        x[row, column] = (data / (data.abs().amax(-1)[:, None] + 1)).to(x.dtype)
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_through_asm(
    x: torch.Tensor, output: torch.Tensor, is_pure: hl.constexpr
) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        scaled = data / (data.abs().amax(-1)[:, None] + 1)
        copied = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;",
            "=r,r",
            [scaled.view(torch.int32)],
            dtype=torch.int32,
            is_pure=is_pure,
            pack=1,
        )
        output[row, column] = copied.view(torch.float32).to(output.dtype)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_rows(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        column = hl.arange(x.size(1))
        data = x[row, column].to(torch.float32)
        maximum = data.abs().amax(-1)
        out[row, column] = (data * scale[row][:, None] / (maximum[:, None] + 1)).to(
            x.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _double(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2
    return out


def _row_config(width: int, **options: Any) -> helion.Config:
    return helion.Config(
        block_sizes=[64],
        num_threads=[32, 32 // width],
        cute_vector_widths=[width, 1],
        cute_lane_layouts=["blocked", "blocked"],
        cute_reduction_reloads=["register"],
        load_eviction_policies="nc_l2_first",
        **options,
    )


def _generate(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config,
    capability: tuple[int, int] | None = None,
) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, args)
        if capability is not None:
            bound.config_spec.target_device_capability = capability
        return bound.to_code(config)


@pytest.mark.parametrize("width,helper", [(8, _NC_16B), (16, _NC_32B)])
def test_readonly_source_uses_the_packet_width_helper(width: int, helper: str) -> None:
    x = torch.empty((128, 32), dtype=torch.bfloat16)
    output = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    code = _generate(_rows_into, (x, output), _row_config(width))
    assert f"{helper}(x.iterator" in code, code


def test_in_place_source_keeps_the_ordinary_load() -> None:
    # Keep ``x`` alive: the lane split proves the in-place write from it.
    x = torch.empty((128, 32), dtype=torch.bfloat16)
    code = _generate(_rows_in_place, (x,), _row_config(16))
    assert _NC_32B not in code, code
    assert "cute.arch.load(x.iterator" in code, code


@pytest.mark.parametrize("is_pure", [True, False])
def test_only_scalar_pure_inline_asm_keeps_the_readonly_load(is_pure: bool) -> None:
    # Assembly the scalar-arithmetic proof cannot read might store anywhere.
    x = torch.empty((128, 32), dtype=torch.bfloat16)
    output = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    code = _generate(_rows_through_asm, (x, output, is_pure), _row_config(16))
    assert (f"{_NC_32B}(x.iterator" in code) is is_pure, code


def test_misaligned_input_view_keeps_the_ordinary_32_byte_load() -> None:
    storage = torch.empty(128 * 32 + 8, dtype=torch.bfloat16)
    x = storage[8:].view(128, 32)
    output = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    code = _generate(_rows_into, (x, output), _row_config(16))
    assert _NC_32B not in code, code


def test_scalar_word_load_drops_the_readonly_hint() -> None:
    # The policy applies to every load site; only packets have the helper.
    x = torch.empty((37, 64), dtype=torch.bfloat16)
    scale = torch.empty((37,), dtype=torch.float32)
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_scaled_rows, (x, scale))
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        spec = bound.config_spec
        code = bound.to_code(
            helion.Config(
                block_sizes=[16],
                num_threads=[
                    4 if block == reduction else 8
                    for block in spec.num_threads.valid_block_ids()
                ],
                cute_vector_widths=[
                    16 if block == reduction else 1
                    for block in spec.cute_vector_widths.valid_block_ids()
                ],
                cute_lane_layouts=["blocked", "blocked"],
                cute_reduction_reloads=["register"],
                load_eviction_policies="nc_l2_first",
            )
        )
    assert f"{_NC_32B}(x.iterator" in code, code
    assert "__nc_l2_first__" not in code, code


@pytest.mark.parametrize("width,helper", [(8, _NC_16B), (16, _NC_32B)])
def test_pointwise_tile_packets_use_the_readonly_helper(
    width: int, helper: str
) -> None:
    x = torch.empty((64, 1024), dtype=torch.bfloat16)
    config = helion.Config(
        block_sizes=[4, 128],
        num_threads=[4, 128 // width],
        cute_vector_widths=[1, width],
        load_eviction_policies="nc_l2_first",
    )
    code = _generate(_double, (x,), config)
    assert f"{helper}(x.iterator" in code, code
    assert "__nc_l2_first__" not in code, code


def test_unsupported_target_keeps_the_ordinary_load() -> None:
    x = torch.empty((128, 32), dtype=torch.bfloat16)
    output = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    code = _generate(_rows_into, (x, output), _row_config(16), capability=(9, 0))
    assert _NC_32B not in code and _NC_16B not in code, code


@pytest.mark.parametrize(
    "capability,cuda,supported",
    [
        ((10, 3), "13.0", True),
        ((10, 0), "13.1", True),
        ((12, 0), "14.0", True),
        ((9, 0), "13.0", False),
        ((8, 0), "13.0", False),
        ((10, 3), "12.9", False),
        ((10, 3), None, False),
        (None, "13.0", False),
    ],
)
def test_readonly_packets_need_sm100_and_cuda13(
    capability: tuple[int, int] | None, cuda: str | None, supported: bool
) -> None:
    assert readonly_packet_load_supported(capability, cuda) is supported


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


# GPU numerics for read-only noncoherent packet loads and L2 evict-last
# packet stores (test_cute_store_policy.py has the store policy's CPU codegen).
#
# A cache policy changes only the opcode of a proven packet: every hinted
# render must match the reference bit for bit.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        out[tile] = (x[tile] * 2).to(out.dtype)
    return out


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0), reason="ld.global.nc.v8 needs SM100+")
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.bfloat16, 8),
        (torch.bfloat16, 16),
        (torch.float32, 4),
        (torch.float32, 8),
    ],
)
def test_readonly_packets_load_every_bit(
    dtype: torch.dtype, width: int, pdl: bool
) -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device=DEVICE).to(dtype)
    out = torch.empty_like(x)
    # Isolated bindings keep each pinned configuration independent.
    bound = _scale_into._bind_isolated((x, out))
    config = helion.Config(
        block_sizes=[4, 256],
        num_threads=[1, 256 // width],
        cute_vector_widths=[1, width],
        load_eviction_policies=["nc_l2_first"],
        cute_pdl=pdl,
    )
    helper = _CUTE_NC_L2_FIRST_HELPERS[width * dtype.itemsize]
    assert f"{helper}(x.iterator" in bound.to_code(config)
    fn = bound.compile_config(config)
    for _ in range(2):
        # A producer rewrites every source element ahead of each launch: the
        # read-only packets must observe it after the ``cute_pdl`` wait.
        x.add_(1)
        fn(x, out)
    torch.testing.assert_close(out, x * 2, atol=0, rtol=0)


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "out_dtype,width",
    [(torch.bfloat16, 8), (torch.float32, 4), (torch.float8_e4m3fn, 16)],
)
def test_l2_evict_last_packets_store_every_bit(
    out_dtype: torch.dtype, width: int
) -> None:
    if not fixed_l2_evict_last_store_policy_supported(
        torch.cuda.get_device_capability(), torch.version.cuda
    ):
        pytest.skip("the fixed L2 store descriptor is validated for SM103/CUDA13")
    torch.manual_seed(0)
    x = torch.randn(128, 32, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty(128, 32, device=DEVICE, dtype=out_dtype)
    bound = _rows_into._bind_isolated((x, out))
    config = helion.Config(
        block_sizes=[64],
        num_threads=[32, 32 // width],
        cute_vector_widths=[width, 1],
        cute_lane_layouts=["blocked", "blocked"],
        cute_reduction_reloads=["register"],
        load_eviction_policies="nc_l2_first",
        cute_store_policy="l2_evict_last",
    )
    assert "_l2_evict_last(output.iterator" in bound.to_code(config)
    bound.compile_config(config)(x, out)
    data = x.float()
    reference = (data / (data.abs().amax(-1, keepdim=True) + 1)).to(out_dtype)
    assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
