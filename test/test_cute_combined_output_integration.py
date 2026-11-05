"""Integration of reviewed bool-aux leaves with mixed-width store ownership."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test.test_cute_bool_aux_native import _args
from test.test_cute_bool_aux_native import _chains
from test.test_cute_bool_aux_native import _external_bool_epilogue

import helion
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute import cute_mma
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target


@pytest.mark.parametrize("narrow_index", (0, 3))
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16))
def test_readonly_bool_and_mixed_output_proofs_compose_without_half_rounding(
    narrow_index, input_dtype
):
    with cpu_codegen_target(), forbid_cuda_work():
        arguments = list(_args())
        arguments[0] = arguments[0].to(input_dtype)
        arguments[1] = arguments[1].to(input_dtype)
        arguments[7 + narrow_index] = torch.empty((128, 128), dtype=input_dtype)
        bound = cpu_bind(_external_bool_epilogue, tuple(arguments))
        assert bound.config_spec.cute_tcgen05_search_enabled
        graphs, anchor, chains = _chains(bound)
        assert chains is not None and len(chains) == 4
        with bound.env, bound.host_function:
            candidate = cute_mma.analyze_cute_mma_node(
                anchor, device_ir=bound.host_function.device_ir
            )
            mixed = cute_mma.tcgen05_mixed_precision_store_plan(
                graphs, anchor, candidate.operands
            )
            assert mixed is not None
            assert mixed.dtypes[narrow_index] == input_dtype
            assert sum(dtype is torch.float32 for dtype in mixed.dtypes) == 3
        for _, chain in chains:
            assert (
                sum(
                    isinstance(leaf, cute_epilogue._ReadonlyBoolAuxLoadExpr)
                    for leaf in chain.auxiliary_tensor_loads
                )
                == 1
            )
        values = []
        original = CuteDeviceFunctionState.register_tcgen05_store_value

        def record(state, name, value):
            values.append(value)
            return original(state, name, value)

        with patch.object(
            CuteDeviceFunctionState, "register_tcgen05_store_value", new=record
        ):
            source = bound.to_code(
                helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
            )
        assert len(values) == 1
        assert values[0].epi_elem_dtype_str == "cutlass.Float32"
        assert values[0].mixed_precision_stores is not None
        assert values[0].mixed_precision_stores.dtypes == mixed.dtypes
        assert "tcgen05_ab_tma" in source
        assert "tcgen05_fp32_tmem_desc" in source
        assert "!= cutlass.Uint8(0)" in source
        assert (
            bound._normalized_config_copy(
                helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
            ).get("cute_mma_dag", "off")
            == "off"
        )
