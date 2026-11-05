from __future__ import annotations

import pytest
import sympy
import torch

from test.test_cute_quotient_aux import _args
from test.test_cute_quotient_aux import _quotient_cols

import helion
from helion._compiler.cute.aux_tensor import (
    host_function_tcgen05_exact_shape_aux_tensors,
)
from helion._compiler.cute.quotient_aux import prove_quotient_aux_load_map
from helion._compiler.cute.quotient_aux import quotient_aux_load_map
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_tile_quotient_cols(
    a, b, aux, first, second, c: hl.constexpr, fixed: hl.constexpr
):
    m, k = a.shape
    n = b.size(1)
    for y, x in hl.tile([m, n], block_size=fixed):
        acc = hl.zeros([y, x], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[y, r], b[r, x])
        first[y, x] = acc
        v = aux[y.index.reshape([y, 1]), x.index.reshape([1, x]) // c]
        second[y, x] = (acc * v.to(torch.float32)).to(torch.float32)


@pytest.mark.parametrize("fixed", ([64, 32], [64, None], [None, 32]))
def test_fixed_tile_quotient_bind_keeps_original_fallback(fixed):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_fixed_tile_quotient_cols, (*_args(), fixed))
        assert not bound.config_spec.cute_tcgen05_search_enabled
        assert (
            bound.config_spec._cute_tcgen05_config._modular_aux_protocol_subject is None
        )
        with bound.env, bound.host_function:
            load, options = _typed_load(bound)
            assert quotient_aux_load_map(load, **options) is not None
        source = bound.to_code(bound.config_spec.default_config())
        assert "tcgen05_ab_tma" not in source
        assert "num_bits_per_copy=32" not in source


def _typed_load(bound):
    for gi in bound.host_function.device_ir.graphs:
        for node in gi.graph.nodes:
            if node.target is helion.language.load and node.args[0].meta[
                "val"
            ].shape == (128, 4):
                store = next(
                    n for n in node.graph.nodes if n.target is helion.language.store
                )
                return node, {
                    "carrier_tile_shape": tuple(node.meta["val"].shape),
                    "carrier_tile_index_nodes": tuple(store.args[1]),
                    "carrier_global_shape": (128, 64),
                }
    raise AssertionError("missing actual typed quotient load")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_projection_atom_is_exactly_one_original_element(dtype):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args(dtype=dtype))
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(helion.Config(block_sizes=[64, 32, 64], pid_type="flat"))
        assert f"assumed_align={dtype.itemsize}" in source
        assert f"num_bits_per_copy={dtype.itemsize * 8}" in source
        with bound.env, bound.host_function:
            assert not host_function_tcgen05_exact_shape_aux_tensors(
                bound.host_function
            )
            assert bound.env.cute_fission_plan is None


@pytest.mark.parametrize(
    "bad_dtype", (torch.bool, torch.int32, torch.int64, torch.float64)
)
def test_other_quotient_storage_dtypes_keep_fallback(bad_dtype):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args(dtype=bad_dtype))
        assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize(
    "fault",
    (
        "float_index",
        "wrong_axis",
        "unowned",
        "overflow_expression",
        "kwargs",
        "extra_mask",
        "wrong_result_dtype",
    ),
)
def test_typed_indices_and_views_not_shape_only(fault):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        with bound.env, bound.host_function:
            load, options = _typed_load(bound)
            assert quotient_aux_load_map(load, **options) is not None
            quotient = load.args[1][1]
            if fault == "float_index":
                quotient.meta["val"] = quotient.meta["val"].to(torch.float32)
            elif fault == "wrong_axis":
                options["carrier_tile_index_nodes"] = tuple(
                    reversed(options["carrier_tile_index_nodes"])
                )
            elif fault == "unowned":
                options["carrier_tile_index_nodes"] = (object(), object())
            elif fault == "overflow_expression":
                before = quotient.args[0]
                with quotient.graph.inserting_before(quotient):
                    overflow = quotient.graph.call_function(
                        torch.ops.aten.mul.Tensor, (before, 2**30)
                    )
                overflow.meta["val"] = before.meta["val"]
                quotient.args = (overflow, 16)
            elif fault == "kwargs":
                load.kwargs = {"foreign": True}
            elif fault == "extra_mask":
                load.args = (*load.args[:2], True, None)
            else:
                load.meta["val"] = load.meta["val"].to(torch.bfloat16)
            assert quotient_aux_load_map(load, **options) is None


@pytest.mark.parametrize(
    "fault", ("undersize", "negative_stride", "span", "offset", "overflow")
)
def test_source_bounds_offset_and_original_integer_span(fault):
    y, x = sympy.symbols("y x", integer=True, nonnegative=True)
    shape, strides, offset, span, extents = (128, 4), (4, 1), 0, 512, (128, 64)
    if fault == "undersize":
        shape = (128, 3)
    elif fault == "negative_stride":
        strides = (-4, 1)
    elif fault == "span":
        span = 511
    elif fault == "offset":
        offset = 1
    else:
        strides = (2**30, 1)
    assert (
        prove_quotient_aux_load_map(
            (y, sympy.floor(x / 16)),
            (y, x),
            extents,
            shape,
            strides,
            storage_offset=offset,
            storage_elements=span,
        )
        is None
    )


def test_padded_offset_naturally_aligned_scalar_view_is_legal():
    a, b, _aux, first, second, c = _args()
    storage = torch.empty(128 * 5 + 1)
    aux = torch.as_strided(storage, (128, 4), (5, 1), storage_offset=1)
    assert aux.data_ptr() % 4 == 0
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, (a, b, aux, first, second, c))
        source = bound.to_code(helion.Config(block_sizes=[64, 32, 64], pid_type="flat"))
        assert "num_bits_per_copy=32" in source
        assert "assumed_align=4" in source


def test_misaligned_scalar_base_is_not_a_vector_alignment_shortcut():
    a, b, _aux, first, second, c = _args()
    storage = bytearray(128 * 4 * 4 + 1)
    aux = torch.frombuffer(
        storage, dtype=torch.float32, count=128 * 4, offset=1
    ).reshape(128, 4)
    assert aux.data_ptr() % 4 != 0
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, (a, b, aux, first, second, c))
        with pytest.raises(helion.exc.InvalidConfig):
            bound.to_code(helion.Config(block_sizes=[64, 32, 64], pid_type="flat"))


def test_new_native_admission_preserves_saved_collective_tile_sizes():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        config = helion.Config(
            block_sizes=[32, 32, 64],
            num_threads=[4, 32, 1],
            cute_collective_mma=True,
            cute_collective_compute="warp",
            cute_collective_tmem_a=True,
        )
        normalized = bound._normalized_config_copy(config)
        assert normalized.block_sizes == [32, 32, 64]
        source = bound.to_code(config)
        assert "_BLOCK_SIZE_0 = 32" in source
        assert "MmaF16BF16Op" in source
        assert "num_bits_per_copy=32" not in source


def test_initial_population_uses_existing_scalar_protocol_constraints():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args())
        subject = bound.config_spec._cute_tcgen05_config._modular_aux_protocol_subject
        assert subject is not None
        generation = ConfigGeneration(bound.config_spec).initial_population_view()
        with bound.env, bound.host_function:
            indices = (
                bound.config_spec._cute_tcgen05_config._modular_aux_block_indices()
            )
            assert indices is not None
            for config in generation.random_population(16, user_seed_configs=()):
                restored = generation.unflatten(generation.flatten(config))
                assert restored == config
                assert subject.refusal(config.config, indices) is None
                assert config.config.get("tcgen05_cluster_m", 1) == 1
                assert config.config.get("tcgen05_epi_warp_groups", 1) == 1
                assert config.config.get("tcgen05_epi_subtile_n") is None


def test_divisor_one_preserves_the_actual_index_axis():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, _args(c=1))
        with bound.env, bound.host_function:
            load = next(
                node
                for gi in bound.host_function.device_ir.graphs
                for node in gi.graph.nodes
                if node.target is helion.language.load
                and node.args[0].meta["val"].dtype is torch.float32
            )
            store = next(
                node
                for node in load.graph.nodes
                if node.target is helion.language.store
            )
            mapping = quotient_aux_load_map(
                load,
                carrier_tile_shape=tuple(load.meta["val"].shape),
                carrier_tile_index_nodes=tuple(store.args[1]),
                carrier_global_shape=(128, 64),
            )
            assert mapping is not None
            assert mapping.divisor == 1 and mapping.quotient_axis == 1


def test_permuted_physical_source_axes_preserve_scalar_offsets():
    y, x = sympy.symbols("y x", integer=True, nonnegative=True)
    mapping = prove_quotient_aux_load_map(
        (sympy.floor(x / 16), y),
        (y, x),
        (128, 64),
        (4, 128),
        (128, 1),
        storage_offset=0,
        storage_elements=512,
    )
    assert mapping is not None and mapping.source_axes == (1, 0)
    for row in range(128):
        for col in range(64):
            assert mapping.offset(row, col) == row + (col // 16) * 128


@pytest.mark.parametrize("fault", ("stride", "offset", "alias"))
def test_codegen_declines_projection_with_live_view_contradicting_bound_facts(fault):
    a, b, _aux, first, second, c = _args()
    storage = torch.empty(128 * 8 + 8)
    aux = torch.as_strided(storage, (128, 4), (8, 1))
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_quotient_cols, (a, b, aux, first, second, c))
        config = helion.Config(block_sizes=[64, 32, 64], pid_type="flat")
        assert "num_bits_per_copy=32" in bound.to_code(config)
        if fault == "stride":
            aux.as_strided_((128, 4), (4, 1))
        elif fault == "offset":
            aux.as_strided_((128, 4), (8, 1), storage_offset=1)
        else:
            aux.set_(first.untyped_storage(), 0, (128, 4), (64, 1))
        try:
            after = bound.to_code(config)
        except (helion.exc.InvalidConfig, helion.exc.BackendUnsupported):
            # A stored native plan contradicting the live facts must refuse.
            return
        # Re-analysis can instead select the existing original fallback. It
        # must not retain the newly proved quotient transport or native plan.
        assert "num_bits_per_copy=32" not in after
        assert "tcgen05_ab_tma" not in after
