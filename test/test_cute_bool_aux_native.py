"""Readonly bool bytes may enter native epilogues only through explicit FP32 casts."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test.test_cute_epilogue_dag import _evaluate
from test.test_cute_epilogue_dag import _render

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import cute_mma
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_bool_epilogue(
    a: torch.Tensor,
    b: torch.Tensor,
    residual0: torch.Tensor,
    residual1: torch.Tensor,
    boolean: torch.Tensor,
    scale: torch.Tensor,
    residual2: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
    mode: hl.constexpr = "explicit",
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            right = b[inner, col].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, left, right)
        value = (residual0[row, col] + acc) + residual1[row, col]
        if mode == "extra_mask":
            mask = hl.load(boolean, [row, col], extra_mask=row.index[:, None] >= 0)
        elif mode == "offset_index":
            mask = hl.load(boolean, [row, col.index + 1])
        elif mode == "transpose":
            mask = boolean[col, row].T
        else:
            mask = boolean[row, col]
        if mode == "implicit":
            factor = mask * hl.full([], 1.1111111111111112, dtype=torch.float32)
        elif mode == "bool_half":
            factor = mask.to(torch.float16).to(torch.float32)
        elif mode == "float_bool":
            factor = residual0[row, col].to(torch.bool).to(torch.float32)
        elif mode == "sin":
            factor = torch.sin(mask.to(torch.float32))
        else:
            factor = mask.to(torch.float32) * hl.full(
                [], 1.1111111111111112, dtype=torch.float32
            )
        masked = value * factor
        out0[row, col] = masked
        scaled = masked * scale[col]
        out1[row, col] = scaled
        out2[row, col] = scaled * hl.full([], 768.0, dtype=torch.float32)
        out3[row, col] = masked * residual2[row, col]


def _args(*, offset=0, stride=1, alias=None):
    shape = (128, 128)
    a = torch.empty(shape, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    res = [torch.empty(shape, dtype=torch.float32) for _ in range(3)]
    storage = torch.empty(shape[0] * shape[1] * stride + offset, dtype=torch.bool)
    boolean = storage[offset : offset + shape[0] * shape[1] * stride : stride].view(
        shape
    )
    outputs = [torch.empty(shape, dtype=torch.float32) for _ in range(4)]
    if alias == "boolean_output":
        backing = torch.empty(shape[0] * shape[1] * 4, dtype=torch.uint8)
        boolean = backing[: shape[0] * shape[1]].view(torch.bool).view(shape)
        outputs[0] = backing.view(torch.float32).view(shape)
    elif alias == "outputs":
        outputs[1] = outputs[0]
    elif alias == "residual":
        res[0] = outputs[0]
    return (
        a,
        b,
        res[0],
        res[1],
        boolean,
        torch.empty((shape[1],), dtype=torch.float32),
        res[2],
        *outputs,
    )


def _chains(bound):
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        anchor = next(
            node
            for graph in graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        chains = analyze_tcgen05_matmul_store_chains(graphs, anchor)
    return graphs, anchor, chains


@pytest.mark.parametrize("offset,stride", ((0, 1), (7, 1), (0, 2)))
def test_bool_native_keeps_all_four_external_outputs_and_byte_storage(offset, stride):
    args = _args(offset=offset, stride=stride)
    bound = cpu_bind(_external_bool_epilogue, args)
    assert bound.config_spec.cute_tcgen05_search_enabled
    graphs, anchor, chains = _chains(bound)
    assert chains is not None and len(chains) == 4
    for _, chain in chains:
        booleans = [
            leaf
            for leaf in chain.auxiliary_tensor_loads
            if isinstance(leaf, epi._ReadonlyBoolAuxLoadExpr)
        ]
        assert len(booleans) == 1
        assert booleans[0].load_node.meta["val"].dtype == torch.bool
        assert (
            booleans[0].template == "(({aux}) != cutlass.Uint8(0)).to(cutlass.Float32)"
        )
    with bound.env, bound.host_function:
        candidate = cute_mma.analyze_cute_mma_node(
            anchor, device_ir=bound.host_function.device_ir
        )
        assert candidate.operands.lhs.mma_dtype == torch.bfloat16
        assert candidate.operands.rhs.mma_dtype == torch.bfloat16
        assert cute_mma.tcgen05_half_source_store_eligible(
            graphs, anchor, candidate.operands
        )
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    ast.parse(source)
    assert "tcgen05_ab_tma" in source
    assert "cutlass.Uint8(0)" in source
    assert "cutlass.Uint8)" in source
    assert source.count("'kind': 'tcgen05_d_tma'") == 4
    assert (
        args[4].stride() == (128 * stride, stride)
        and args[4].storage_offset() == offset
    )
    assert tuple(args[index].dtype for index in range(7, 11)) == (torch.float32,) * 4


@pytest.mark.parametrize(
    "mode",
    (
        "implicit",
        "bool_half",
        "float_bool",
        "extra_mask",
        "offset_index",
        "transpose",
        "sin",
    ),
)
def test_unproved_bool_access_or_cast_stays_out_of_native_namespace(mode):
    bound = cpu_bind(_external_bool_epilogue, (*_args(), mode))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert _chains(bound)[2] is None
    assert "cute_collective_mma" in bound.config_spec._flat_fields()


@pytest.mark.parametrize("alias", ("boolean_output", "outputs", "residual"))
def test_bool_readonly_and_all_output_disjoint_proofs_fail_closed(alias):
    bound = cpu_bind(_external_bool_epilogue, _args(alias=alias))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert _chains(bound)[2] is None


def test_bool_fake_only_or_cleared_alias_facts_are_not_a_readonly_proof():
    args = _args()
    with FakeTensorMode() as mode:
        fake = tuple(mode.from_tensor(value) for value in args)
    assert not cpu_bind(
        _external_bool_epilogue, fake
    ).config_spec.cute_tcgen05_search_enabled
    bound = cpu_bind(_external_bool_epilogue, args)
    graphs, anchor, _ = _chains(bound)
    with bound.env, bound.host_function:
        key = "tensor_storage_disjoint_matrix_v1"
        saved = bound.env.bound_runtime_input_specialization_results.pop(key)
        try:
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
        finally:
            bound.env.bound_runtime_input_specialization_results[key] = saved


def test_bool_alias_rebinding_and_prepared_guard_cannot_reuse_readonly_native_proof():
    disjoint_args = _args()
    overlap_args = _args(alias="boolean_output")
    disjoint = _external_bool_epilogue.bind(disjoint_args)
    overlap = _external_bool_epilogue.bind(overlap_args)
    rebound = _external_bool_epilogue.bind(_args())
    assert disjoint is not overlap
    assert disjoint.config_spec.cute_tcgen05_search_enabled
    assert not overlap.config_spec.cute_tcgen05_search_enabled
    assert rebound.config_spec.cute_tcgen05_search_enabled
    key = "tensor_storage_disjoint_matrix_v1"
    with disjoint.env, disjoint.host_function:
        spec = disjoint.env.runtime_input_specializations[key]
        result = disjoint.env.bound_runtime_input_specialization_results[key]
        extractors = tuple(
            (lambda values, i=i: values[i]) for i in range(len(spec.sources))
        )
        guard = spec.prepared_guard_factory(extractors, result)
        assert guard(disjoint_args)
        assert not guard(overlap_args)


def test_rendered_noncanonical_bool_bytes_and_ieee_arithmetic_are_exact():
    bound = cpu_bind(_external_bool_epilogue, _args())
    graphs, anchor, chains = _chains(bound)
    assert chains is not None
    values = (
        torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan"), 1.0e-8]
        )
        .repeat(8)
        .view(4, 16)
    )
    raw = (
        torch.tensor([0, 1, 2, 128, 255, 0, 255, 2], dtype=torch.uint8)
        .repeat(8)
        .view(4, 16)
    )
    r0 = torch.zeros_like(values)
    r1 = torch.zeros_like(values)
    r2 = torch.full_like(values, 0.3)
    scale = torch.full((16,), -1.0, dtype=torch.float32)
    mask_value = raw.ne(0).to(torch.float32)
    prefix = ((r0 + values) + r1) * (
        mask_value * torch.full((), 1.1111111111111112, dtype=torch.float32)
    )
    expected = (
        prefix,
        prefix * scale,
        (prefix * scale) * torch.full((), 768.0, dtype=torch.float32),
        prefix * r2,
    )
    for (_store, chain), want in zip(chains, expected, strict=True):
        bindings = {}
        for leaf in chain.auxiliary_tensor_loads:
            tensor = leaf.load_node.args[0].args[0]
            bindings[leaf] = {
                "boolean": "raw",
                "residual0": "r0",
                "residual1": "r1",
                "residual2": "r2",
                "scale": "scale",
            }[tensor]
        source, expression = _render((chain, anchor), aux=bindings)
        got = _evaluate(
            source,
            expression,
            values,
            raw=raw,
            r0=r0,
            r1=r1,
            r2=r2,
            scale=scale,
            cutlass=SimpleNamespace(
                Float32=torch.float32,
                Float16=torch.float16,
                BFloat16=torch.bfloat16,
                Uint8=torch.uint8,
            ),
        )
        torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)
        finite = ~torch.isnan(want)
        assert torch.equal(torch.signbit(got[finite]), torch.signbit(want[finite]))
        assert source.count("raw") == 1


def test_bool_is_not_admitted_as_a_general_floating_cast_or_mma_operand():
    bound = cpu_bind(_external_bool_epilogue, _args())
    graphs, anchor, chains = _chains(bound)
    leaf = next(
        leaf
        for leaf in chains[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._ReadonlyBoolAuxLoadExpr)
    )
    conversion = next(
        node
        for node in leaf.load_node.users
        if node.target is torch.ops.prims.convert_element_type.default
    )
    assert epi._auxiliary_cast_operand(conversion) is None
    assert epi._floating_cast_step(conversion) is None
    with bound.env, bound.host_function:
        original = cute_mma.analyze_cute_mma_node(
            anchor, device_ir=bound.host_function.device_ir
        )
        assert original.operands.lhs.source_fake.dtype is torch.bfloat16
        assert original.operands.rhs.source_fake.dtype is torch.bfloat16


def test_compact_multi_output_seeds_fit_and_native_store_lowering_agrees():
    bound = cpu_bind(_external_bool_epilogue, _args())
    seeds = bound.config_spec._cute_tcgen05_config._multi_output_seed_configs()
    assert seeds
    with bound.env, bound.host_function:
        from helion.autotuner.config_generation import ConfigGeneration

        generation = ConfigGeneration(bound.config_spec)
        transferred = generation.seed_flat_config_pairs()
        assert transferred
    from helion._compiler.cute.device_state import CuteDeviceFunctionState

    original_claim = CuteDeviceFunctionState.claim_tcgen05_store_site
    claims = []

    def track_claim(state, value, store):
        result = original_claim(state, value, store)
        claims.append((store, result))
        return result

    for _, seed in transferred:
        claims.clear()
        with patch.object(
            CuteDeviceFunctionState, "claim_tcgen05_store_site", new=track_claim
        ):
            source = bound.to_code(seed)
        assert "tcgen05_ab_tma" in source
        assert len(claims) == 4 and len({site for site, _ in claims}) == 4
        assert [result for _, result in claims] == [
            (False, False),
            (True, False),
            (True, False),
            (True, True),
        ]
    for seed in seeds:
        normalized = bound._normalized_config_copy(seed)
        assert normalized.get("tcgen05_flat_role_coordinates", False) is False
        source = bound.to_code(seed)
        assert "tcgen05_ab_tma" in source
        assert source.count("'kind': 'tcgen05_d_tma'") == 4
        assert "cutlass.Uint8(0)" in source


@pytest.mark.parametrize("budget", (32768, 65536))
def test_too_small_multi_output_budget_preserves_bind_default_base_and_seed_transfer(
    budget,
):
    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
    from helion.autotuner.config_generation import ConfigGeneration

    with (
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=budget
        ),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_budget_bytes", return_value=budget
        ),
        patch.object(
            CuteTcgen05Config, "per_cta_ab_smem_budget_bytes", return_value=budget
        ),
    ):
        # Actual managed binding is also checked: discovering an infeasible
        # seed must not interrupt normal snapshot/reference construction.
        kernel = helion.kernel(
            _external_bool_epilogue.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        bound = kernel.bind(_args())
        assert bound.config_spec.cute_tcgen05_search_enabled
        spec = bound.config_spec
        defaults = (spec.default_config(), spec._base_default_config())
        with bound.env, bound.host_function:
            generation = ConfigGeneration(spec)
            transferred = generation.seed_flat_config_pairs()
            assert transferred
            assert len(transferred) < len(spec.compiler_seed_configs)
            for default in defaults:
                flat, reference = generation.canonicalize_flat(
                    generation.flatten(default)
                )
                flat2, reference2 = generation.canonicalize_flat(flat)
                assert flat2 == flat and reference2 == reference
        # The original default/base and all transferred reference candidates
        # are fully rendered; only an explicitly impossible request refuses.
        for config in (*defaults, *(config for _, config in transferred)):
            source = bound.to_code(config)
            assert "tcgen05_ab_tma" in source
            assert "cutlass.Uint8(0)" in source
        impossible = helion.Config(
            block_sizes=[128, 64, 64],
            pid_type="flat",
            tcgen05_cluster_m=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
        )
        with pytest.raises(helion.exc.InvalidConfig, match="SMEM|multi-output"):
            bound._normalized_config_copy(impossible)
        with bound.env:
            requested = dict(impossible.config)
            if budget == 32768:
                with pytest.raises(helion.exc.InvalidConfig, match="SMEM|multi-output"):
                    spec.normalize(requested, _fix_invalid=True)
            else:
                spec.normalize(requested, _fix_invalid=True)
                assert (
                    requested["tcgen05_ab_stages"]
                    < impossible.config["tcgen05_ab_stages"]
                )
        if budget == 65536:
            # Normal repair selects the genuinely fitting shallower arena;
            # strict normalization and complete lowering must agree with it.
            projected = helion.Config.from_dict(requested)
            source = bound.to_code(projected)
            assert "tcgen05_ab_tma" in source and "cutlass.Uint8(0)" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _single_bool_epilogue(a: torch.Tensor, b: torch.Tensor, boolean: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        out[row, col] = acc * boolean[row, col].to(torch.float32)
    return out


def test_single_output_bool_simt_aux_producer_uses_uint8_and_tma_aux_stays_excluded():
    args = (
        torch.empty((128, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bool),
    )
    bound = cpu_bind(_single_bool_epilogue, args)
    assert bound.config_spec.cute_tcgen05_search_enabled
    config = helion.Config(
        block_sizes=[128, 64, 64],
        pid_type="persistent_blocked",
        tcgen05_ab_stages=2,
        tcgen05_strategy="role_local_with_scheduler",
        tcgen05_warp_spec_scheduler_warps=1,
        tcgen05_warp_spec_c_input_warps=1,
    )
    source = bound.to_code(config)
    assert "tcgen05_ab_tma" in source
    assert "cutlass.Uint8(0)" in source
    assert "cutlass.Uint8" in source
    with pytest.raises(helion.exc.BackendUnsupported, match="dtype|match|aux"):
        bound.to_code(
            helion.Config.from_dict({**config.config, "tcgen05_aux_load_mode": "tma"})
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bool_mma_operand(a: torch.Tensor, b: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner].to(torch.bfloat16), b[inner, col])
        out[row, col] = acc
    return out


def test_bool_underlying_mma_operand_remains_ineligible():
    bound = cpu_bind(
        _bool_mma_operand,
        (
            torch.empty((128, 128), dtype=torch.bool),
            torch.empty((128, 128), dtype=torch.bfloat16),
        ),
    )
    assert not bound.config_spec.cute_tcgen05_search_enabled
    with bound.env, bound.host_function:
        for graph in bound.host_function.device_ir.graphs:
            for node in graph.graph.nodes:
                if node.target is torch.ops.aten.addmm.default:
                    assert (
                        cute_mma.analyze_cute_mma_node(
                            node, device_ir=bound.host_function.device_ir
                        )
                        is None
                    )


def test_bool_dag_requires_single_anchor_and_keeps_existing_depth_limit():
    bound = cpu_bind(_external_bool_epilogue, _args())
    graphs, anchor, chains = _chains(bound)
    with bound.env, bound.host_function:
        graph = anchor.graph
        with graph.inserting_after(anchor):
            other = graph.call_function(anchor.target, anchor.args)
            other.meta = dict(anchor.meta)
        try:
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
        finally:
            graph.erase_node(other)
        store = chains[0][0]
        graph = store.graph
        original = store.args
        nodes = []
        with graph.inserting_before(store):
            value = original[2]
            for _ in range(36):
                node = graph.call_function(torch.ops.aten.neg.default, (value,))
                node.meta = dict(value.meta)
                nodes.append(node)
                value = node
        store.args = (*original[:2], value, *original[3:])
        try:
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
        finally:
            store.args = original
            for node in reversed(nodes):
                graph.erase_node(node)


def test_bool_cast_metadata_and_depth_whitelist_cannot_be_bypassed():
    bound = cpu_bind(_external_bool_epilogue, _args())
    graphs, anchor, chains = _chains(bound)
    leaf = next(
        leaf
        for leaf in chains[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._ReadonlyBoolAuxLoadExpr)
    )
    conversion = next(
        node
        for node in leaf.load_node.users
        if node.target is torch.ops.prims.convert_element_type.default
    )
    with bound.env, bound.host_function:
        val = conversion.meta["val"]
        altered = torch.empty_strided(
            val.shape,
            tuple(reversed(val.stride())),
            dtype=torch.float32,
            device=val.device,
        )
        with patch.dict(conversion.meta, {"val": altered}):
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
