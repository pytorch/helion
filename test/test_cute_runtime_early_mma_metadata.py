from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_epilogue_dag import _args
from test.test_cute_epilogue_dag import _biased_gelu

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute import cute_mma
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target

KEY = "tensor_storage_disjoint_matrix_v1"

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_codegen_target(), forbid_cuda_work():
        cute_mma.get_cute_mma_support().tcgen05_tf32 = True
        yield


def _store_chains(bound):
    graphs = bound.host_function.device_ir.graphs
    mma = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    return analyze_tcgen05_matmul_store_chains(graphs, mma)


def test_early_mma_admission_reads_registered_live_facts_without_publishing_snapshots():
    original = cute_epilogue._analyze_tcgen05_pointwise_dag
    observed = []

    def analyze(*args, **kwargs):
        env = CompileEnvironment.current()
        before_facts = dict(env.bound_runtime_input_specialization_results)
        before_classifiers = dict(env.runtime_input_specializations)
        result = original(*args, **kwargs)
        if env.early_mma_metadata_scope_active and result is not None:
            observed.append((before_facts, before_classifiers))
            assert env.bound_runtime_input_specialization_results == before_facts
            assert env.runtime_input_specializations == before_classifiers
        return result

    with patch.object(cute_epilogue, "_analyze_tcgen05_pointwise_dag", analyze):
        bound = cpu_bind(_biased_gelu, _args())
    assert bound.config_spec.cute_tcgen05_search_enabled
    assert observed and all(
        not facts and KEY in classifiers for facts, classifiers in observed
    )
    assert KEY in bound.env.bound_runtime_input_specialization_results
    assert not bound.env.early_mma_metadata_scope_active
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source and "tcgen05_dag_step" in source


@pytest.mark.parametrize("kind", ["missing", "fake", "aliased"])
def test_early_mma_unknown_fake_or_aliased_inputs_fail_closed_without_new_facts(kind):
    args = _args()
    bound = cpu_bind(_biased_gelu, args)
    values = dict(zip(bound.kernel.signature.parameters, args, strict=True))
    if kind == "missing":
        values = {}
    elif kind == "fake":
        values = dict(bound.host_function.params.arguments)
    else:
        values["bias"] = args[3][0]
    with bound.env, bound.host_function:
        saved = bound.env.bound_runtime_input_specialization_results
        bound.env.bound_runtime_input_specialization_results = {}
        before = dict(bound.env.runtime_input_specializations)
        try:
            with bound.env.use_runtime_arg_values(values, capture_metadata=True):
                assert _store_chains(bound) is None
                assert not bound.env.bound_runtime_input_specialization_results
                assert bound.env.runtime_input_specializations == before
        finally:
            bound.env.bound_runtime_input_specialization_results = saved


def test_live_admission_requires_a_registered_guard_and_never_registers_it():
    args = _args()
    bound = cpu_bind(_biased_gelu, args)
    values = dict(zip(bound.kernel.signature.parameters, args, strict=True))
    with bound.env, bound.host_function:
        saved_facts = bound.env.bound_runtime_input_specialization_results
        classifier = bound.env.runtime_input_specializations.pop(KEY)
        bound.env.bound_runtime_input_specialization_results = {}
        try:
            with bound.env.use_runtime_arg_values(values, capture_metadata=True):
                assert _store_chains(bound) is None
            assert KEY not in bound.env.runtime_input_specializations
            assert not bound.env.bound_runtime_input_specialization_results
        finally:
            bound.env.runtime_input_specializations[KEY] = classifier
            bound.env.bound_runtime_input_specialization_results = saved_facts


def test_construction_scope_does_not_allow_unbound_codegen_or_refresh_bound_facts():
    args = _args()
    bound = cpu_bind(_biased_gelu, args)
    values = dict(zip(bound.kernel.signature.parameters, args, strict=True))
    with bound.env, bound.host_function:
        saved = bound.env.bound_runtime_input_specialization_results
        bound.env.bound_runtime_input_specialization_results = {}
        try:
            with bound.env.use_runtime_arg_values(values, capture_metadata=True):
                assert _store_chains(bound) is not None
                spec = bound.env.runtime_input_specializations[KEY]
                bound.env.register_runtime_input_specialization(KEY, spec)
                assert not bound.env.bound_runtime_input_specialization_results
            assert not bound.env.early_mma_metadata_scope_active
            with bound.env.use_runtime_arg_values(values):
                assert _store_chains(bound) is None
                assert not bound.env.bound_runtime_input_specialization_results
        finally:
            bound.env.bound_runtime_input_specialization_results = saved


def test_early_live_facts_cannot_override_a_contradictory_bound_snapshot():
    args = _args()
    bound = cpu_bind(_biased_gelu, args)
    values = dict(zip(bound.kernel.signature.parameters, args, strict=True))
    values["bias"] = args[3][0]
    with bound.env, bound.host_function:
        saved = dict(bound.env.bound_runtime_input_specialization_results)
        with bound.env.use_runtime_arg_values(values, capture_metadata=True):
            assert _store_chains(bound) is None
        assert bound.env.bound_runtime_input_specialization_results == saved


def test_non_mma_readonly_discovery_never_borrows_mma_scope_or_mutates_facts():
    from test.cute_population_contracts import _target
    from test.test_atomic_ops import _fragment_discovery_composed_scans

    from helion._compiler.autotuner_heuristics.cute_fragment_common import (
        computed_fragment_discovery_supported,
    )
    from helion._compiler.cute.computed_fragment import computed_fragment_supported

    x = torch.ones((3, 35), dtype=torch.int32, device="cpu")
    limits = torch.tensor([0, 1, 3], dtype=torch.int32, device="cpu")
    out = torch.empty(3, dtype=torch.int32, device="cpu")
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_fragment_discovery_composed_scans, (x, limits, out))
        env = bound.env
        values = {"x": x, "limits": limits, "out": out}
        with env, bound.host_function:
            saved = env.bound_runtime_input_specialization_results
            env.bound_runtime_input_specialization_results = {}
            classifiers = dict(env.runtime_input_specializations)
            try:
                with env.use_runtime_arg_values(values, capture_metadata=True):
                    graphs = bound.host_function.device_ir.graphs
                    assert computed_fragment_discovery_supported(env, graphs)
                    assert not computed_fragment_supported(env, graphs)
                    assert env.runtime_input_specializations == classifiers
                    assert not env.bound_runtime_input_specialization_results
            finally:
                env.bound_runtime_input_specialization_results = saved


def test_dead_publication_stays_rejected_and_valid_publication_stays_live():
    from test.cute_population_contracts import _target
    from test.test_atomic_ops import _fragment_discovery_composed_scans
    from test.test_cute_published_scalars import _code

    x = torch.ones((3, 35), dtype=torch.int32, device="cpu")
    limits = torch.tensor([0, 1, 3], dtype=torch.int32, device="cpu")
    out = torch.empty(3, dtype=torch.int32, device="cpu")
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_fragment_discovery_composed_scans, (x, limits, out))
        assert not bound.config_spec.cute_fragment_published_scalar_root_ids
        requested = bound.config_spec.default_config()
        requested.config["cute_fragment_published_scalars"] = True
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.create_config_generation().strict_config_pair(requested)
    valid_bound, source = _code(x, True)
    assert "fragment_published_scalar" in source
    assert not valid_bound.env.early_mma_metadata_scope_active


def test_dispatch_and_prepared_guard_do_not_reuse_early_disjoint_admission_for_aliases():
    disjoint_args = _args()
    alias_args = list(_args())
    alias_args[2] = alias_args[3][0]
    disjoint = _biased_gelu.bind(disjoint_args)
    alias = _biased_gelu.bind(tuple(alias_args))
    assert disjoint is not alias
    assert disjoint.config_spec.cute_tcgen05_search_enabled
    assert not alias.config_spec.cute_tcgen05_search_enabled
    assert (
        disjoint.env.bound_runtime_input_specialization_results[KEY]
        != alias.env.bound_runtime_input_specialization_results[KEY]
    )
    disjoint_keys = [
        key for key, value in _biased_gelu._bound_kernels.items() if value is disjoint
    ]
    alias_keys = [
        key for key, value in _biased_gelu._bound_kernels.items() if value is alias
    ]
    assert disjoint_keys and alias_keys
    assert disjoint_keys[0].extra_results != alias_keys[0].extra_results
    with disjoint.env, disjoint.host_function:
        specialization = disjoint.env.runtime_input_specializations[KEY]
        fact = disjoint.env.bound_runtime_input_specialization_results[KEY]
        extractors = tuple(
            lambda args, index=index: args[index]
            for index in range(len(specialization.sources))
        )
        guard = specialization.prepared_guard_factory(extractors, fact)
        assert guard(disjoint_args)
        assert not guard(tuple(alias_args))


def _genuinely_aligned_cpu_tensor(
    count: int, dtype: torch.dtype, alignment_bytes: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Own a real aligned storage slice, with zero-based typed view offsets."""
    itemsize = dtype.itemsize
    assert count > 0 and alignment_bytes > 0
    assert alignment_bytes % itemsize == 0
    byte_count = count * itemsize
    raw = torch.empty(byte_count + alignment_bytes, dtype=torch.uint8, device="cpu")
    byte_offset = (-raw.data_ptr()) % alignment_bytes
    assert 0 <= byte_offset < alignment_bytes
    assert byte_offset + byte_count <= raw.numel()
    # A real UntypedStorage slice retains its parent allocation. Rebase the
    # storage view itself: packed-region offsets must start at zero, not at
    # the alignment padding within raw. No pointer/metadata is fabricated.
    storage = raw.untyped_storage()[byte_offset : byte_offset + byte_count]
    root = torch.empty(0, dtype=dtype, device="cpu").set_(storage, 0, (count,), (1,))
    aligned = root.view(count)
    assert aligned.data_ptr() == raw.data_ptr() + byte_offset
    assert aligned.data_ptr() % alignment_bytes == 0
    assert aligned.storage_offset() == 0
    assert aligned.untyped_storage().nbytes() == byte_count
    return aligned, raw


def _genuinely_aligned_recurrence_args() -> tuple[object, ...]:
    from test.test_cute_chunk_recurrence import _dv4_runtime_args

    original = _dv4_runtime_args(
        torch.device("cpu"), value=0.125, decay=0.5, identity_aq=True
    )
    factors = original[:5]
    assert all(isinstance(value, torch.Tensor) for value in factors)
    workspace_bytes = sum(value.numel() * value.element_size() for value in factors)
    storage, _owner = _genuinely_aligned_cpu_tensor(workspace_bytes, torch.uint8)
    cursor = 0
    original_base_offset = factors[0].storage_offset() * factors[0].element_size()
    packed = []
    for value in factors:
        size = value.numel() * value.element_size()
        # Preserve exact region offsets relative to the original physical base;
        # its raw allocation can have leading alignment padding.
        assert (
            value.storage_offset() * value.element_size() - original_base_offset
            == cursor
        )
        assert value.data_ptr() - factors[0].data_ptr() == cursor
        assert cursor % 256 == 0
        tensor = storage[cursor : cursor + size].view(value.dtype).view(value.shape)
        tensor.copy_(value)
        assert tensor.shape == value.shape and tensor.stride() == value.stride()
        assert tensor.dtype == value.dtype and tensor.device == value.device
        assert tensor.data_ptr() == storage.data_ptr() + cursor
        assert tensor.data_ptr() % 256 == 0
        assert tensor.storage_offset() * tensor.element_size() == cursor
        assert cursor + size <= storage.untyped_storage().nbytes()
        packed.append(tensor)
        cursor += size
    assert cursor == workspace_bytes
    return (*packed, *original[5:])


@pytest.mark.parametrize("dv_partitions", [2, 4])
def test_managed_recurrence_bind_records_complete_late_packed_workspace_snapshot(
    dv_partitions,
):
    from test.test_cute_chunk_recurrence import _CONFIG
    from test.test_cute_chunk_recurrence import _bt16_resident_chain

    from helion._compiler.cute.chunk_prepare import _PACKED_WORKSPACE_SPECIALIZATION_KEY
    from helion._compiler.cute.chunk_prepare import _packed_workspace_runtime_signature

    # Preserve the authentic packed carrier and classifier, but remove the
    # CPU allocator-alignment lottery with a genuinely owned aligned view.
    args = _genuinely_aligned_recurrence_args()
    assert _packed_workspace_runtime_signature(args[:5])
    assert args[0].data_ptr() % 256 == 0
    bound = _bt16_resident_chain.bind(args)
    env = bound.env
    registered = set(env.runtime_input_specializations)
    assert _PACKED_WORKSPACE_SPECIALIZATION_KEY in registered
    assert KEY in registered and "cute_persistent_vec_alignment_matrix_v3" in registered
    assert set(env.bound_runtime_input_specialization_results) == registered
    assert (
        env.bound_runtime_input_specialization_results[
            _PACKED_WORKSPACE_SPECIALIZATION_KEY
        ]
        is True
    )
    assert not env.early_mma_metadata_scope_active
    config = helion.Config.from_dict(
        dict(_CONFIG.config) | {"cute_chunk_recurrence_dv_partitions": dv_partitions}
    )
    source = bound.to_code(config)
    expected = (
        "chunk_recurrence_sm100" if dv_partitions == 2 else "chunk_recurrence_warp_dv4"
    )
    assert expected in source
    assert "dot_acc = dot_acc +" not in source


@pytest.mark.parametrize(
    "dtype", (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
)
@pytest.mark.parametrize("count", (1, 17))
def test_genuine_aligned_cpu_view_has_exact_pointer_offsets_bounds_and_owner(
    dtype, count
):
    import gc
    import weakref

    tensor, raw = _genuinely_aligned_cpu_tensor(count, dtype)
    pointer = tensor.data_ptr()
    byte_offset = pointer - raw.data_ptr()
    byte_count = count * dtype.itemsize
    assert 0 <= byte_offset < 256
    assert byte_offset + byte_count <= raw.numel()
    assert tensor.shape == (count,) and tensor.stride() == (1,)
    assert tensor.dtype == dtype and tensor.device.type == "cpu"
    assert tensor.storage_offset() == 0 and tensor.data_ptr() % 256 == 0
    assert tensor._base is not None
    raw.fill_(0x5A)
    fill = 7 if dtype is torch.uint8 else 0.125
    tensor.fill_(fill)
    assert torch.equal(
        raw[byte_offset : byte_offset + byte_count], tensor.view(torch.uint8)
    )
    assert torch.all(raw[:byte_offset] == 0x5A)
    assert torch.all(raw[byte_offset + byte_count :] == 0x5A)
    # Physical aliasing is real in both directions, not just equal metadata.
    raw[byte_offset : byte_offset + byte_count].zero_()
    assert torch.equal(tensor, torch.zeros(count, dtype=dtype))
    tensor.fill_(fill)
    owner = weakref.ref(raw)
    del raw
    gc.collect()
    assert owner() is None
    # The tensor/storage view, not the raw Python object, owns the allocation.
    assert tensor.data_ptr() == pointer
    assert torch.equal(tensor, torch.full((count,), fill, dtype=dtype))
    tensor[-1] = fill


def test_aligned_recurrence_fixture_keeps_real_packed_aliases_data_and_guard_rejection():
    from test.test_cute_chunk_recurrence import _dv4_runtime_args

    from helion._compiler.cute.chunk_prepare import _packed_workspace_runtime_signature

    args = _genuinely_aligned_recurrence_args()
    original = _dv4_runtime_args(
        torch.device("cpu"), value=0.125, decay=0.5, identity_aq=True
    )
    factors = args[:5]
    shared = factors[0].untyped_storage()
    cursor = 0
    for actual, wanted in zip(factors, original[:5], strict=True):
        assert actual.untyped_storage() is shared
        assert actual.storage_offset() * actual.element_size() == cursor
        assert actual.data_ptr() == factors[0].data_ptr() + cursor
        assert actual.data_ptr() % 256 == 0
        assert actual.shape == wanted.shape and actual.stride() == wanted.stride()
        assert actual.dtype == wanted.dtype and torch.equal(actual, wanted)
        cursor += actual.numel() * actual.element_size()
    assert cursor == shared.nbytes()
    assert args[5] is args[6]
    assert _packed_workspace_runtime_signature(factors)
    unaligned = list(factors)
    unaligned[0] = factors[0].as_strided(factors[0].shape, factors[0].stride(), 1)
    assert unaligned[0].data_ptr() % 256 != 0
    assert not _packed_workspace_runtime_signature(unaligned)
    independent = list(factors)
    independent[1] = factors[1].clone()
    assert not _packed_workspace_runtime_signature(independent)
