"""The result-fission schedule: native producer GEMM + SIMT consumer, CPU proofs."""

from __future__ import annotations

import ast
import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute import result_fission as rf
from helion._compiler.cute import result_fission_codegen as rfc
from helion._compiler.cute import result_fission_config as cfgkeys
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl

M, N, K = 256, 256, 128
SETTINGS = {
    "backend": "cute",
    "static_shapes": True,
    "autotune_effort": "full",
    "fast_math": False,
}


@helion.kernel(**SETTINGS)
def four_outputs(a, b, u, v, o0, o1, o2, o3):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc.to(torch.float32)
        o0[row, col] = t0.to(torch.float32)
        t1 = torch.neg(t0)
        t2 = v[row, col].to(torch.float32)
        t3 = t1 * t2
        o1[row, col] = t3.to(torch.float32)
        t4 = u[row].to(torch.float32)
        t5 = hl.full([], 1e-06, dtype=torch.float32)
        t6 = t4 + t5
        t7 = t0 / t6.reshape([row, 1])
        t8 = t7 / t6.reshape([row, 1])
        o2[row, col] = t8.to(torch.float32)
        t9 = t0 * t2
        o3[row, col] = t9.to(torch.float32)


@helion.kernel(**SETTINGS)
def identity_middle(a, b, v, o0, o1, o2):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc.to(torch.float32)
        t1 = torch.neg(t0)
        o0[row, col] = t1.to(torch.float32)
        o1[row, col] = t0.to(torch.float32)
        t2 = v[row, col].to(torch.float32)
        t3 = t0 * t2
        o2[row, col] = t3.to(torch.float32)


@helion.kernel(**SETTINGS)
def identity_last(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        t1 = torch.neg(t0)
        o0[row, col] = t1
        o1[row, col] = t0


@helion.kernel(**SETTINGS)
def no_identity(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        t1 = torch.neg(t0)
        t2 = torch.abs(t0)
        o0[row, col] = t1
        o1[row, col] = t2


@helion.kernel(**SETTINGS)
def multiply_add(a, b, u, v, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = t0 * v[row, col]
        t2 = t1 + u[row].reshape([row, 1])
        o1[row, col] = t2


@helion.kernel(**SETTINGS)
def negated_multiply_add(a, b, u, v, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = t0 * v[row, col]
        t2 = torch.neg(t1)
        t3 = t2 + u[row].reshape([row, 1])
        o1[row, col] = t3


@helion.kernel(**SETTINGS)
def nonzero_init(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.full([row, col], 1.0, dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = torch.neg(t0)
        o1[row, col] = t1


@helion.kernel(**SETTINGS)
def scaled_operand(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, torch.neg(a[row, inner]), b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = torch.neg(t0)
        o1[row, col] = t1


@helion.kernel(**SETTINGS)
def reads_output(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = t0 * o1[row, col]
        o1[row, col] = t1


@helion.kernel(**SETTINGS)
def host_local(a, b, o0, o1):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        t0 = acc
        o0[row, col] = t0
        t1 = t0 * k
        o1[row, col] = t1


@helion.kernel(**SETTINGS)
def single_output(a, b, o0):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        o0[row, col] = acc


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _a(dtype=torch.bfloat16):
    return torch.empty((M, K), dtype=dtype)


def _b(dtype=torch.bfloat16):
    return torch.empty((K, N), dtype=dtype)


def _out(dtype=torch.float32):
    return torch.empty((M, N), dtype=dtype)


def bind(kernel, values):
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        return cpu_bind(kernel, values)


def four_args():
    return (_a(), _b(), torch.empty(M), _out(), _out(), _out(), _out(), _out())


def proof_of(bound):
    with bound.env, bound.host_function:
        return rf.analyze_result_fission(
            bound.env, bound.host_function.device_ir, allow_unbound=False
        )


def fission_seeds(bound):
    seeds = bound.config_spec.compiler_seed_configs
    flags = [bool(seed.config.get(cfgkeys.RESULT_FISSION_KEY)) for seed in seeds]
    first = flags.index(True) if True in flags else len(flags)
    # Appended only: every earlier seed is an ordinary fused seed.
    assert all(flags[first:]) and not any(flags[:first])
    return [seed for seed in seeds if seed.config.get(cfgkeys.RESULT_FISSION_KEY)]


def embedded_modules(source: str) -> dict[str, str]:
    module = ast.parse(source)
    return {
        node.targets[0].id: node.value.args[0].value
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "PyCodeCache.load"
    }


def test_four_output_identity_proof_and_consumer_statements():
    bound = bind(four_outputs, four_args())
    proof = proof_of(bound)
    assert proof is not None
    assert proof.v_output == "o0" and proof.outputs == ("o0", "o1", "o2", "o3")
    assert (proof.m, proof.n, proof.k) == (M, N, K)
    assert proof.consumer_params == ("u", "v", "o0", "o1", "o2", "o3")
    assert proof.consumer_statements[0] == "t0 = o0[row, col]"
    # Every other statement is the original statement, in original order,
    # including both sequential divisions and every cast.
    assert proof.consumer_statements[1:] == (
        "t1 = torch.neg(t0)",
        "t2 = v[row, col].to(torch.float32)",
        "t3 = t1 * t2",
        "o1[row, col] = t3.to(torch.float32)",
        "t4 = u[row].to(torch.float32)",
        "t5 = hl.full([], 1e-06, dtype=torch.float32)",
        "t6 = t4 + t5",
        "t7 = t0 / t6.reshape([row, 1])",
        "t8 = t7 / t6.reshape([row, 1])",
        "o2[row, col] = t8.to(torch.float32)",
        "t9 = t0 * t2",
        "o3[row, col] = t9.to(torch.float32)",
    )
    assert bound.config_spec.cute_result_fission_available
    assert bound.config_spec.cute_result_fission_shape == (M, N, K)


@pytest.mark.parametrize(
    "kernel,values,position",
    [
        (identity_middle, lambda: (_a(), _b(), _out(), _out(), _out(), _out()), "o1"),
        (identity_last, lambda: (_a(), _b(), _out(), _out()), "o1"),
    ],
)
def test_identity_positions(kernel, values, position):
    proof = proof_of(bind(kernel, values()))
    assert proof is not None and proof.v_output == position
    assert "acc" not in proof.consumer_source


@pytest.mark.parametrize(
    "kernel,values",
    [
        (no_identity, lambda: (_a(), _b(), _out(), _out())),
        (multiply_add, lambda: (_a(), _b(), torch.empty(M), _out(), _out(), _out())),
        (
            negated_multiply_add,
            lambda: (_a(), _b(), torch.empty(M), _out(), _out(), _out()),
        ),
        (nonzero_init, lambda: (_a(), _b(), _out(), _out())),
        (
            identity_last,
            lambda: (
                _a(),
                _b(),
                torch.empty(M, N + 16)[:, :N],
                torch.empty(M, N + 16)[:, :N],
            ),
        ),
        (
            identity_last,
            lambda: (_a(), _b(), torch.empty(N, M).t(), torch.empty(N, M).t()),
        ),
        (scaled_operand, lambda: (_a(), _b(), _out(), _out())),
        (reads_output, lambda: (_a(), _b(), _out(), _out())),
        (host_local, lambda: (_a(), _b(), _out(), _out())),
        (single_output, lambda: (_a(), _b(), _out())),
        (
            identity_last,
            lambda: (_a(), _b(), _out(torch.bfloat16), _out(torch.bfloat16)),
        ),
        (identity_last, lambda: (_a(torch.float32), _b(torch.float32), _out(), _out())),
    ],
    ids=[
        "no_identity",
        "multiply_feeds_add",
        "multiply_feeds_add_through_neg",
        "nonzero_accumulator_init",
        "padded_identity_output",
        "column_major_identity_output",
        "transformed_operand",
        "reads_output",
        "host_local_in_consumer",
        "single_output",
        "rounded_bf16_identity",
        "fp32_operands",
    ],
)
def test_refusals_disable_the_schedule(kernel, values):
    bound = bind(kernel, values())
    assert proof_of(bound) is None
    assert not bound.config_spec.cute_result_fission_available
    assert not fission_seeds(bound)
    with pytest.raises(exc.InvalidConfig):
        bound.config_spec.normalize(
            helion.Config.from_dict(
                {
                    **bound.config_spec.default_config().config,
                    cfgkeys.RESULT_FISSION_KEY: True,
                }
            )
        )


def test_output_alias_and_input_alias_refuse():
    shared = _out()
    values = (_a(), _b(), torch.empty(M), _out(), shared, shared, _out(), _out())
    assert proof_of(bind(four_outputs, values)) is None
    out2 = _out()
    values = (_a(), _b(), torch.empty(M), out2, _out(), _out(), out2, _out())
    assert proof_of(bind(four_outputs, values)) is None


def test_default_and_saved_configs_are_unchanged():
    bound = bind(four_outputs, four_args())
    default = bound.config_spec.default_config()
    assert not any(key.startswith("cute_result_fission") for key in default.config)
    normalized = bound._normalized_config_copy(default)
    assert not any(key.startswith("cute_result_fission") for key in normalized)
    source = bound.to_code(default)
    assert "_helion_cute_result_fission" not in source


def _seed(bound):
    seeds = fission_seeds(bound)
    assert seeds
    return seeds[0]


def test_bundle_two_launches_protocols_and_no_allocation():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    source = bound.to_code(seed)
    modules = embedded_modules(source)
    assert len(modules) == 2
    producer, consumer = modules.values()
    assert "tcgen05" in producer and producer.count("PipelineTmaStore.create(") == 1
    assert "tcgen05" not in consumer and "PipelineTmaStore" not in consumer
    stores = re.findall(r"\((o\d)\.iterator[^\n]*\)\.store\(", consumer)
    assert sorted(stores) == ["o1", "o2", "o3"]
    for text in (producer, consumer, source):
        assert "griddepcontrol" not in text
    assert "1.0 /" not in consumer and "exact_div" not in consumer
    host = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "four_outputs"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(host) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") for c in calls)
    launches = [
        c
        for c in calls
        if c.endswith(("_result_fission_producer", "_helion_result_fission_consumer"))
    ]
    assert launches[0].endswith("_result_fission_producer")
    assert launches[1].endswith("_helion_result_fission_consumer")
    assert "_helion_cute_kernels" in source and "_helion_cute_source_hash" in source


def test_vector_consumer_layout_renders_coalesced_vectors():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    config = helion.Config.from_dict(
        {
            **seed.config,
            cfgkeys.CONSUMER_BLOCK_SIZES_KEY: [8, 256],
            cfgkeys.CONSUMER_NUM_THREADS_KEY: [8, 64],
            cfgkeys.CONSUMER_VECTOR_WIDTH_KEY: 4,
        }
    )
    bound.config_spec.normalize(config)
    consumer = list(embedded_modules(bound.to_code(config)).values())[1]
    assert "VectorType.get([4]" in consumer
    assert "thread_idx()[0]) * 4" in consumer
    assert "1.0 /" not in consumer


def test_lost_final_proof_fails_loudly():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    with (
        patch.object(rfc, "analyze_result_fission", return_value=None),
        pytest.raises(exc.BackendUnsupported, match="final bound proof was lost"),
    ):
        bound.to_code(seed)


def test_producer_config_is_strict():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    config = helion.Config.from_dict(
        {
            **seed.config,
            cfgkeys.PRODUCER_BLOCK_SIZES_KEY: [256, 256, 128],
            cfgkeys.PRODUCER_AB_STAGES_KEY: 16,
        }
    )
    # Priced at normalization: never reaches codegen.
    with pytest.raises(exc.InvalidConfig, match="not admitted by the producer"):
        bound.to_code(config)
    # A 16-row tile is not a template tile: refused at normalization.
    tiny = helion.Config.from_dict(
        {
            **seed.config,
            cfgkeys.PRODUCER_BLOCK_SIZES_KEY: [16, 16, 16],
            cfgkeys.PRODUCER_AB_STAGES_KEY: 2,
        }
    )
    with pytest.raises(exc.InvalidConfig, match="producer_block_sizes"):
        bound.config_spec.normalize(tiny)
    # The producer's own normalizer stays the final authority at codegen: a
    # schedule that slipped past normalization is refused, never adjusted.
    with (
        patch.object(cfgkeys, "producer_schedule_problem", return_value=None),
        pytest.raises(exc.BackendUnsupported, match="producer"),
    ):
        bound.to_code(tiny)


@pytest.mark.parametrize(
    "update",
    [
        {cfgkeys.CONSUMER_NUM_THREADS_KEY: [4, 100]},
        {cfgkeys.CONSUMER_VECTOR_WIDTH_KEY: 3},
        {cfgkeys.PRODUCER_PID_TYPE_KEY: "persistent_bogus"},
        {"cute_pdl": True},
        {cfgkeys.RESULT_FISSION_KEY: "yes"},
    ],
)
def test_normalization_rejects_invalid_schedule(update):
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    with pytest.raises(exc.InvalidConfig):
        bound.config_spec.normalize(helion.Config.from_dict({**seed.config, **update}))


def test_stage_keys_require_the_schedule():
    bound = bind(four_outputs, four_args())
    values = dict(bound.config_spec.default_config().config)
    values[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] = 4
    with pytest.raises(exc.InvalidConfig):
        bound.config_spec.normalize(helion.Config.from_dict(values))
    fixed = helion.Config.from_dict(values)
    bound.config_spec.normalize(fixed, _fix_invalid=True)
    assert cfgkeys.CONSUMER_VECTOR_WIDTH_KEY not in fixed.config


def test_seed_producer_schedules_come_from_native_seeds():
    bound = bind(four_outputs, four_args())
    native = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if not seed.config.get(cfgkeys.RESULT_FISSION_KEY)
    ]
    native_tiles = {
        tuple(seed.config["block_sizes"])
        for seed in native
        if "tcgen05_ab_stages" in seed.config
    }
    for seed in fission_seeds(bound):
        assert tuple(seed.config[cfgkeys.PRODUCER_BLOCK_SIZES_KEY]) in native_tiles
        assert seed.config.get("cute_pdl", False) is False


def test_vector_consumer_keeps_unaligned_views_scalar():
    misaligned = torch.empty(M * N + 1)[1:].view(M, N)
    values = (_a(), _b(), torch.empty(M), _out(), _out(), misaligned, _out(), _out())
    bound = bind(four_outputs, values)
    seed = _seed(bound)
    assert seed.config[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] == 1
    assert len(embedded_modules(bound.to_code(seed))) == 2
    vector = helion.Config.from_dict(
        {
            **seed.config,
            cfgkeys.CONSUMER_BLOCK_SIZES_KEY: [8, 256],
            cfgkeys.CONSUMER_NUM_THREADS_KEY: [8, 64],
            cfgkeys.CONSUMER_VECTOR_WIDTH_KEY: 4,
        }
    )
    consumer = list(embedded_modules(bound.to_code(vector)).values())[1]
    # Per-view cache-key facts: the unaligned output keeps scalar stores,
    # every aligned view keeps 16-byte vectors.
    assert re.findall(r"_cute_store_u32_vec\((o\d)\.iterator", consumer) == ["o2", "o3"]
    assert re.findall(r"\((o\d)\.iterator[^\n]*\)\.store\(", consumer) == ["o1"]
    assert sorted(re.findall(r"cute\.arch\.load\((\w+)\.iterator", consumer)) == [
        "o0",
        "v",
    ]


def test_every_producer_seed_has_a_vector_consumer_twin():
    # Each admitted producer is seeded with the coalesced scalar consumer and
    # its N-vectorized twin, adjacent and otherwise identical; the producer
    # list itself (the native-seed transfer) is unchanged.
    bound = bind(four_outputs, four_args())
    seeds = fission_seeds(bound)
    assert len(seeds) % 2 == 0
    producer = [tuple(repr(s.config[k]) for k in cfgkeys.PRODUCER_KEYS) for s in seeds]
    for scalar, vector in zip(seeds[::2], seeds[1::2], strict=True):
        assert scalar.config[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] == 1
        assert vector.config[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] == 4
        assert vector.config[cfgkeys.CONSUMER_BLOCK_SIZES_KEY] == [8, 256]
        differing = {
            k
            for k in set(scalar.config) | set(vector.config)
            if scalar.config.get(k) != vector.config.get(k)
        }
        assert differing == {
            cfgkeys.CONSUMER_BLOCK_SIZES_KEY,
            cfgkeys.CONSUMER_NUM_THREADS_KEY,
            cfgkeys.CONSUMER_VECTOR_WIDTH_KEY,
        }
    assert producer[::2] == producer[1::2]
    assert len(set(producer[::2])) == len(producer[::2])
    assert len(seeds) <= 2 * cfgkeys_max_producer_seeds()


def cfgkeys_max_producer_seeds():
    from helion._compiler.autotuner_heuristics.cute_result_fission import (
        MAX_PRODUCER_SEEDS,
    )

    return MAX_PRODUCER_SEEDS


def test_vector_twins_compile_native_and_coalesced_for_every_producer():
    bound = bind(four_outputs, four_args())
    for seed in fission_seeds(bound)[1::2]:
        producer, consumer = embedded_modules(bound.to_code(seed)).values()
        assert producer.count("PipelineTmaStore.create(") == 1
        assert "griddepcontrol" not in producer + consumer
        assert "cute.arch.load(" in consumer and "_cute_store_u32_vec(" in consumer


def test_no_vector_twin_when_shape_admits_no_whole_warp_vectors():
    # n=128 (not a multiple of 256): scalar consumers only.
    values = (
        torch.empty(M, K, dtype=torch.bfloat16),
        torch.empty(K, 128, dtype=torch.bfloat16),
        torch.empty(M),
        torch.empty(M, 128),
        torch.empty(M, 128),
        torch.empty(M, 128),
        torch.empty(M, 128),
        torch.empty(M, 128),
    )
    bound = bind(four_outputs, values)
    seeds = fission_seeds(bound)
    assert seeds and all(
        s.config[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] == 1 for s in seeds
    )


def test_unaligned_identity_output_refuses_the_tma_producer():
    misaligned = torch.empty(M * N + 1)[1:].view(M, N)
    values = (_a(), _b(), torch.empty(M), _out(), misaligned, _out(), _out(), _out())
    bound = bind(four_outputs, values)
    with pytest.raises(exc.BackendUnsupported, match="TensorMap alignment"):
        bound.to_code(_seed(bound))


def test_stage_bodies_and_launch_flags_are_pdl_free():
    bound = bind(four_outputs, four_args())
    for seed in fission_seeds(bound):
        source = bound.to_code(seed)
        assert "use_pdl" not in source and "_helion_cute_use_pdl" not in source
        for text in embedded_modules(source).values():
            assert "griddepcontrol" not in text and "_helion_cute_use_pdl" not in text


def test_pdl_emitting_stage_is_refused_not_stripped():
    with pytest.raises(exc.BackendUnsupported, match="would launch with PDL"):
        rfc._require_pdl_free("producer", "cute.arch.griddepcontrol_wait()\n", [])
    with pytest.raises(exc.BackendUnsupported, match="would launch with PDL"):
        rfc._require_pdl_free("consumer", "k._helion_cute_use_pdl = True\n", [])
    with pytest.raises(exc.BackendUnsupported, match="would launch with PDL"):
        rfc._require_pdl_free("producer", "", [{"use_pdl": True}])
    with pytest.raises(exc.BackendUnsupported, match="would launch with PDL"):
        rfc._require_pdl_free(
            "producer", "k._helion_cute_wrapper_plans = [{'use_pdl': True}]\n", []
        )
    rfc._require_pdl_free("producer", "plain\n", [{"use_pdl": False}])


def test_paired_producer_builds_without_pdl():
    # The reviewed paired PDL-off repair: a CTA-pair producer with
    # cute_pdl=False emits no griddepcontrol and no PDL launch flag.
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    paired = helion.Config.from_dict(
        {
            **seed.config,
            cfgkeys.PRODUCER_BLOCK_SIZES_KEY: [256, 128, 64],
            cfgkeys.PRODUCER_AB_STAGES_KEY: 2,
            cfgkeys.PRODUCER_CLUSTER_M_KEY: 2,
            cfgkeys.PRODUCER_CTA_GROUP_KEY: "two",
        }
    )
    bound.config_spec.normalize(paired)
    source = bound.to_code(paired)
    assert "use_pdl" not in source
    producer, consumer = embedded_modules(source).values()
    for text in (producer, consumer):
        assert "griddepcontrol" not in text and "_helion_cute_use_pdl" not in text
    assert producer.count("PipelineTmaStore.create(") == 1
    assert "cluster" in producer.lower()


def test_seed_transfer_failure_never_fails_the_bind():
    from helion._compiler.autotuner_heuristics import cute_result_fission as heuristic

    def boom(*args, **kwargs):
        raise RuntimeError("producer template compile exploded")

    # A fresh kernel object: the module-level one caches earlier binds.
    fresh = helion.kernel(four_outputs.fn, settings=four_outputs.settings)
    with patch.object(
        heuristic.CuteResultFissionHeuristic, "transferred_seed_configs", boom
    ):
        bound = bind(fresh, four_args())
    assert bound.config_spec.cute_result_fission_available
    assert not fission_seeds(bound)
    assert bound.config_spec.compiler_seed_configs


# ---- stage-key search: fields, inactive canonicalization, real variation ----


def _generation(bound):
    from helion.autotuner.config_generation import ConfigGeneration

    return ConfigGeneration(bound.config_spec)


def _index(generation, key):
    indices, is_sequence = generation._key_to_flat_indices[key]
    assert not is_sequence and len(indices) == 1
    return indices[0]


STAGE_SEARCH_KEYS = (
    cfgkeys.PRODUCER_BLOCK_SIZES_KEY,
    cfgkeys.PRODUCER_PID_TYPE_KEY,
    cfgkeys.PRODUCER_AB_STAGES_KEY,
    cfgkeys.PRODUCER_ACC_STAGES_KEY,
    cfgkeys.PRODUCER_C_STAGES_KEY,
    cfgkeys.PRODUCER_CLUSTER_M_KEY,
    cfgkeys.PRODUCER_CTA_GROUP_KEY,
    cfgkeys.CONSUMER_BLOCK_SIZES_KEY,
    cfgkeys.CONSUMER_VECTOR_WIDTH_KEY,
)


def test_stage_search_fields_exist_only_with_the_proof():
    bound = bind(four_outputs, four_args())
    fields = bound.config_spec._flat_fields()
    for key in (cfgkeys.RESULT_FISSION_KEY, *STAGE_SEARCH_KEYS):
        assert key in fields, key
    # The thread count is derived from the tile and vector width.
    assert cfgkeys.CONSUMER_NUM_THREADS_KEY not in fields
    refused = bind(no_identity, (_a(), _b(), _out(), _out()))
    assert not any(
        key.startswith("cute_result_fission")
        for key in refused.config_spec._flat_fields()
    )


def test_random_samples_keep_the_schedule_off_and_stage_keys_inactive():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    # The population carries the compiler seeds; every random sample keeps
    # the schedule off and carries no stage keys.
    seeds = fission_seeds(bound)
    configs = generation.random_population(64)
    assert configs
    for config in configs:
        if config.config.get(cfgkeys.RESULT_FISSION_KEY):
            assert config in seeds
            continue
        assert not any(k.startswith("cute_result_fission") for k in config.config)
    for _ in range(32):
        flat = generation.random_flat()
        assert flat[_index(generation, cfgkeys.RESULT_FISSION_KEY)] is False


def test_inactive_stage_coordinates_canonicalize_away():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    base = generation.default_flat()
    varied = list(base)
    varied[_index(generation, cfgkeys.PRODUCER_AB_STAGES_KEY)] = 7
    varied[_index(generation, cfgkeys.CONSUMER_VECTOR_WIDTH_KEY)] = 4
    a = generation.unflatten(list(base))
    b = generation.unflatten(varied)
    assert a == b
    flat_b, _ = generation.canonicalize_flat(varied)
    assert (
        flat_b[_index(generation, cfgkeys.PRODUCER_AB_STAGES_KEY)]
        == (base[_index(generation, cfgkeys.PRODUCER_AB_STAGES_KEY)])
    )


def test_seed_round_trips_through_flat_space():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    for seed in fission_seeds(bound):
        flat = generation.flatten(seed)
        assert generation.unflatten(list(flat)) == seed


def test_inert_fused_coordinates_do_not_create_distinct_candidates():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    seed = _seed(bound)
    flat = generation.flatten(seed)
    indices, _ = generation._key_to_flat_indices["block_sizes"]
    mutated = list(flat)
    mutated[indices[0]] = mutated[indices[0]] * 2 if mutated[indices[0]] < 256 else 16
    assert generation.unflatten(mutated) == generation.unflatten(list(flat))


@pytest.mark.parametrize(
    "key,value,stage",
    [
        (cfgkeys.PRODUCER_BLOCK_SIZES_KEY, [128, 64, 64], 0),
        (cfgkeys.PRODUCER_AB_STAGES_KEY, 3, 0),
        (cfgkeys.CONSUMER_BLOCK_SIZES_KEY, [8, 128], 1),
        (cfgkeys.CONSUMER_VECTOR_WIDTH_KEY, 2, 1),
    ],
)
def test_stage_mutation_changes_the_generated_stage(key, value, stage):
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    seed = _seed(bound)
    flat = generation.flatten(seed)
    mutated = list(flat)
    mutated[_index(generation, key)] = value
    config = generation.unflatten(mutated)
    assert config.config[key] == value
    assert config.config[cfgkeys.RESULT_FISSION_KEY] is True
    before = list(embedded_modules(bound.to_code(seed)).values())
    after = list(embedded_modules(bound.to_code(config)).values())
    assert after[stage] != before[stage]
    assert after[1 - stage] == before[1 - stage]
    for text in after:
        assert "griddepcontrol" not in text


def test_consumer_threads_are_derived_from_tile_and_vector():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    flat = generation.flatten(_seed(bound))
    flat[_index(generation, cfgkeys.CONSUMER_BLOCK_SIZES_KEY)] = [8, 256]
    flat[_index(generation, cfgkeys.CONSUMER_VECTOR_WIDTH_KEY)] = 4
    config = generation.unflatten(flat)
    assert config.config[cfgkeys.CONSUMER_NUM_THREADS_KEY] == [8, 64]
    # An impossible layout (fewer than a warp of N threads) repairs to the
    # coalesced scalar default rather than reaching codegen.
    flat[_index(generation, cfgkeys.CONSUMER_BLOCK_SIZES_KEY)] = [1, 32]
    config = generation.unflatten(flat)
    assert config.config[cfgkeys.CONSUMER_VECTOR_WIDTH_KEY] == 1
    assert config.config[cfgkeys.CONSUMER_BLOCK_SIZES_KEY] == [4, 256]


def test_producer_ab_ring_is_priced():
    bound = bind(four_outputs, four_args())
    spec = bound.config_spec
    admission = spec.cute_result_fission_producer_facts
    assert admission is not None
    seed = _seed(bound)
    deep = {
        **seed.config,
        cfgkeys.PRODUCER_BLOCK_SIZES_KEY: [128, 256, 128],
        cfgkeys.PRODUCER_AB_STAGES_KEY: 16,
    }
    limit = admission.max_ab_stages[
        (128, 256, 128, 1, "auto", 2, 2, "persistent_interleaved")
    ]
    assert 1 <= limit < 16
    with pytest.raises(exc.InvalidConfig, match="not admitted by the producer"):
        spec.normalize(helion.Config.from_dict(dict(deep)))
    repaired = helion.Config.from_dict(dict(deep))
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired.config[cfgkeys.PRODUCER_AB_STAGES_KEY] == limit
    # The repaired depth is what the producer's own normalizer and codegen
    # accept: a real priced configuration, not only a key value.
    producer = next(iter(embedded_modules(bound.to_code(repaired)).values()))
    assert producer.count("PipelineTmaStore.create(") == 1


def test_cta_pair_requires_a_cluster():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    values = {
        **seed.config,
        cfgkeys.PRODUCER_CTA_GROUP_KEY: "two",
        cfgkeys.PRODUCER_CLUSTER_M_KEY: 1,
    }
    with pytest.raises(exc.InvalidConfig, match="cluster_m"):
        bound.config_spec.normalize(helion.Config.from_dict(dict(values)))
    fixed = helion.Config.from_dict(dict(values))
    bound.config_spec.normalize(fixed, _fix_invalid=True)
    assert fixed.config[cfgkeys.PRODUCER_CTA_GROUP_KEY] == "auto"


def test_neighbors_of_a_fission_seed_explore_stage_keys():
    bound = bind(four_outputs, four_args())
    generation = _generation(bound)
    flat = generation.flatten(_seed(bound))
    explored = set()
    for key in STAGE_SEARCH_KEYS:
        field = generation.flat_spec[_index(generation, key)]
        for neighbor in field.pattern_neighbors(flat[_index(generation, key)]):
            candidate = list(flat)
            candidate[_index(generation, key)] = neighbor
            config = generation.unflatten(candidate)
            if config.config.get(cfgkeys.RESULT_FISSION_KEY):
                explored.add((key, repr(config.config.get(key))))
    assert {key for key, _ in explored} >= set(STAGE_SEARCH_KEYS) - {
        cfgkeys.PRODUCER_CTA_GROUP_KEY
    }


# ---- v4: admission == the producer's own normalizer; CTA plan; fallback ----


def _producer_child_admits(bound):
    """keys -> whether the producer child's strict normalizer admits them."""
    from contextlib import ExitStack

    env = bound.env
    stack = ExitStack()
    stack.enter_context(env)
    stack.enter_context(bound.host_function)
    proof = rf.analyze_result_fission(
        env, bound.host_function.device_ir, allow_unbound=True
    )
    child = rfc._child(env)
    stack.enter_context(env.suspend())
    stack.enter_context(child)
    host, _ = rfc._compile_child(
        env,
        child,
        rfc._result_fission_producer,
        rfc._producer_metadata(proof),
        tma_inputs=True,
    )
    stack.enter_context(host)
    assert rfc._register_producer_facts(
        child, host, rfc.CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
    )

    def admits(keys):
        requested = rfc.producer_config(helion.Config.from_dict(dict(keys)))
        try:
            normalized = child.config_spec.normalized_config(requested)
            rfc._strict(requested, normalized, "producer")
        except (exc.InvalidConfig, exc.BackendUnsupported):
            return False
        return rfc._lowers_to_tcgen05(proof, normalized, env.device)

    return admits, stack


def test_producer_admission_matches_the_producer_normalizer():
    import itertools

    bound = bind(four_outputs, four_args())
    admits, stack = _producer_child_admits(bound)
    with stack:
        mismatches = []
        for bm, bn, bk, cluster, group in itertools.product(
            (16, 32, 64, 128, 256),
            (16, 32, 64, 128, 256),
            (16, 32, 64, 128),
            (1, 2),
            ("auto", "two"),
        ):
            keys = {
                cfgkeys.PRODUCER_BLOCK_SIZES_KEY: [bm, bn, bk],
                cfgkeys.PRODUCER_PID_TYPE_KEY: "persistent_interleaved",
                cfgkeys.PRODUCER_AB_STAGES_KEY: 2,
                cfgkeys.PRODUCER_ACC_STAGES_KEY: 2,
                cfgkeys.PRODUCER_C_STAGES_KEY: 2,
                cfgkeys.PRODUCER_CLUSTER_M_KEY: cluster,
                cfgkeys.PRODUCER_CTA_GROUP_KEY: group,
            }
            static = cfgkeys.producer_schedule_problem(keys, None) is None
            child = admits(keys)
            # Statically refused schedules: the producer never admits them,
            # except the clustered single-CTA family, which it admits but
            # lowers without the TMA store ring (refused on protocol).  The
            # statically legal ones are priced by the admission table.
            clustered_one = cluster == 2 and not cfgkeys.producer_runs_cta_pair(
                [bm, bn, bk], cluster_m=cluster, cta_group=group
            )
            if not static and child and not clustered_one:
                mismatches.append((bm, bn, bk, cluster, group))
    assert mismatches == []


def test_clustered_single_cta_family_is_refused_on_protocol():
    bound = bind(four_outputs, four_args())
    seed = _seed(bound)
    clustered = helion.Config.from_dict(
        {
            **seed.config,
            **cfgkeys.producer_schedule_keys(
                (128, 128, 64, 2, "auto", 2, 2, "persistent_interleaved"), 2
            ),
        }
    )
    with pytest.raises(exc.InvalidConfig, match="clustered single-CTA"):
        bound.config_spec.normalize(clustered)
    # Even past normalization, the emitted body is checked: no TMA store ring
    # means no producer.
    with (
        patch.object(cfgkeys, "producer_schedule_problem", return_value=None),
        pytest.raises(exc.BackendUnsupported, match="native TCGEN05"),
    ):
        bound.to_code(clustered)


def test_search_tile_axes_are_template_axes():
    bound = bind(four_outputs, four_args())
    field = bound.config_spec._flat_fields()[cfgkeys.PRODUCER_BLOCK_SIZES_KEY]
    assert [axis.choices for axis in field.axes] == [
        cfgkeys.PRODUCER_BLOCK_M,
        cfgkeys.PRODUCER_BLOCK_N,
        cfgkeys.PRODUCER_BLOCK_K,
    ]
    assert 16 not in cfgkeys.PRODUCER_BLOCK_M and 32 not in cfgkeys.PRODUCER_BLOCK_M
    neighbors = field.pattern_neighbors([128, 64, 64])
    assert all(n[0] in cfgkeys.PRODUCER_BLOCK_M for n in neighbors)
    assert cfgkeys.PRODUCER_CTA_GROUPS == ("auto", "two")


def test_admission_table_is_the_producer_normalizer():
    # Pricing is the producer child's own strict admission: for every
    # statically legal schedule and every AB depth, our table admits exactly
    # what the producer normalizer admits (so the binary search is exact and
    # the effective CTA plan, C ring and SMEM model are the producer's own).
    bound = bind(four_outputs, four_args())
    admission = bound.config_spec.cute_result_fission_producer_facts
    assert admission is not None
    schedules = list(cfgkeys.producer_schedules())
    assert set(admission.max_ab_stages) == set(schedules)
    admits, stack = _producer_child_admits(bound)
    mismatches = []
    with stack:
        for schedule in schedules:
            for stages in range(1, cfgkeys.MAX_PRODUCER_AB_STAGES + 1):
                keys = cfgkeys.producer_schedule_keys(schedule, stages)
                ours = cfgkeys.producer_schedule_problem(keys, admission) is None
                if ours != admits(keys):
                    mismatches.append((schedule, stages, ours))
    assert mismatches == []
    # The clustered bm=128 'auto' schedule is the single-CTA family (no TMA
    # store ring): not a producer plan; the explicit pair on the same tile is.
    assert (
        128,
        128,
        64,
        2,
        "auto",
        2,
        2,
        "persistent_interleaved",
    ) not in admission.max_ab_stages
    assert (
        admission.max_ab_stages[
            (128, 128, 64, 2, "two", 2, 2, "persistent_interleaved")
        ]
        >= 1
    )
    assert (
        admission.max_ab_stages[
            (256, 128, 64, 2, "auto", 2, 2, "persistent_interleaved")
        ]
        >= 1
    )


def _fission_values(stages, blocks=(128, 256, 128)):
    return {
        cfgkeys.RESULT_FISSION_KEY: True,
        "cute_pdl": False,
        cfgkeys.PRODUCER_BLOCK_SIZES_KEY: list(blocks),
        cfgkeys.PRODUCER_PID_TYPE_KEY: "persistent_interleaved",
        cfgkeys.PRODUCER_AB_STAGES_KEY: stages,
        cfgkeys.PRODUCER_ACC_STAGES_KEY: 2,
        cfgkeys.PRODUCER_C_STAGES_KEY: 2,
        cfgkeys.PRODUCER_CLUSTER_M_KEY: 1,
        cfgkeys.PRODUCER_CTA_GROUP_KEY: "auto",
    }


def _admission(entries):
    table = dict.fromkeys(cfgkeys.producer_schedules(), 0)
    table.update(entries)
    return cfgkeys.ProducerAdmission(table)


def test_producer_fallback_is_revalidated():
    shape = (M, N, K)
    requested = (128, 256, 128, 1, "auto", 2, 2, "persistent_interleaved")
    default = (128, 128, 64, 1, "auto", 2, 2, "persistent_interleaved")
    # Requested schedule admits no AB ring, the default admits two: the
    # default schedule is adopted only after its own admission check.
    config = _fission_values(8)
    admission = _admission({default: 2})
    cfgkeys.normalize_result_fission_config(
        config, available=True, fix_invalid=True, shape=shape, producer_facts=admission
    )
    assert config[cfgkeys.PRODUCER_BLOCK_SIZES_KEY] == [128, 128, 64]
    assert config[cfgkeys.PRODUCER_AB_STAGES_KEY] == 2
    assert cfgkeys.producer_schedule_problem(config, admission) is None
    # A requested schedule admitted at a shallower ring keeps its tile.
    config = _fission_values(8)
    cfgkeys.normalize_result_fission_config(
        config,
        available=True,
        fix_invalid=True,
        shape=shape,
        producer_facts=_admission({requested: 1, default: 4}),
    )
    assert config[cfgkeys.PRODUCER_BLOCK_SIZES_KEY] == [128, 256, 128]
    assert config[cfgkeys.PRODUCER_AB_STAGES_KEY] == 1
    # Nothing admitted, not even the default: the schedule is off, never an
    # unpriced default.
    none = _admission({})
    config = _fission_values(8)
    cfgkeys.normalize_result_fission_config(
        config, available=True, fix_invalid=True, shape=shape, producer_facts=none
    )
    assert not any(key.startswith("cute_result_fission") for key in config)
    with pytest.raises(exc.InvalidConfig, match="not admitted by the producer"):
        cfgkeys.normalize_result_fission_config(
            _fission_values(8),
            available=True,
            fix_invalid=False,
            shape=shape,
            producer_facts=none,
        )


def test_fused_random_samples_are_unchanged_by_the_schedule():
    from contextlib import ExitStack
    import random

    def samples(available):
        fresh = helion.kernel(four_outputs.fn, settings=four_outputs.settings)
        with ExitStack() as stack:
            if not available:
                stack.enter_context(
                    patch.object(rf, "analyze_result_fission", return_value=None)
                )
            bound = bind(fresh, four_args())
            generation = _generation(bound)
            assert bound.config_spec.cute_result_fission_available is available
            random.seed(1234)
            uniform = [
                generation.unflatten(generation.random_flat()) for _ in range(16)
            ]
            biased = [
                generation.unflatten(generation.biased_random_flat()) for _ in range(16)
            ]
            tail = random.random()
            keys = set(bound.config_spec._flat_fields())
        return uniform, biased, tail, keys

    on_uniform, on_biased, on_tail, on_keys = samples(True)
    off_uniform, off_biased, off_tail, off_keys = samples(False)
    extra = sorted(on_keys - off_keys)
    assert extra and all(key.startswith("cute_result_fission") for key in extra)
    # The new coordinates consume no random draw: every random sample (and
    # the generator state after them) matches the kernel without the proof.
    assert on_uniform == off_uniform
    assert on_biased == off_biased
    assert on_tail == off_tail


def test_inert_stage_defaults_pass_strict_transfers():
    # Strict coverage transfers normalize flat configs that carry every
    # stage coordinate at its inert default: those must pass, and the fused
    # coverage groups (e.g. the matmul min-blocks pair) must register exactly
    # as for a kernel without the proof.
    from contextlib import ExitStack

    def groups(available):
        fresh = helion.kernel(four_outputs.fn, settings=four_outputs.settings)
        with ExitStack() as stack:
            if not available:
                stack.enter_context(
                    patch.object(rf, "analyze_result_fission", return_value=None)
                )
            bound = bind(fresh, four_args())
        spec = bound.config_spec
        generation = _generation(bound)
        _, carrier = generation.canonicalize_flat(generation.default_flat())
        generation.strict_config_pair(carrier)
        fused_fields = [
            key
            for key in spec._flat_fields()
            if not key.startswith("cute_result_fission")
        ]
        return (
            sorted(group.key for group in spec.compiler_coverage_groups),
            fused_fields,
            spec.cute_matmul_min_blocks_search_enabled,
        )

    assert groups(True) == groups(False)
    assert groups(True)[2] is True


def test_tile_fragment_equality_compares_every_axis():
    field = cfgkeys.result_fission_search_fields()[cfgkeys.PRODUCER_BLOCK_SIZES_KEY]
    same = cfgkeys.result_fission_search_fields()[cfgkeys.PRODUCER_BLOCK_SIZES_KEY]
    other = cfgkeys._ProducerTileFragment(
        (
            cfgkeys.EnumFragment(cfgkeys.PRODUCER_BLOCK_M),
            cfgkeys.EnumFragment((128, 64)),
            cfgkeys.EnumFragment(cfgkeys.PRODUCER_BLOCK_K),
        )
    )
    assert field == same and field != other
    assert field.fingerprint() != other.fingerprint()


def test_turned_off_schedule_keeps_the_fused_pdl_choice():
    shape = (M, N, K)
    config = {**_fission_values(8), "cute_pdl": True}
    cfgkeys.normalize_result_fission_config(
        config,
        available=True,
        fix_invalid=True,
        shape=shape,
        producer_facts=_admission({}),
    )
    assert config == {"cute_pdl": True}
    kept = {**_fission_values(2, blocks=(128, 128, 64)), "cute_pdl": True}
    cfgkeys.normalize_result_fission_config(
        kept,
        available=True,
        fix_invalid=True,
        shape=shape,
        producer_facts=_admission(
            {(128, 128, 64, 1, "auto", 2, 2, "persistent_interleaved"): 6}
        ),
    )
    assert kept["cute_pdl"] is False and kept[cfgkeys.RESULT_FISSION_KEY] is True


def test_admitted_depth_emits_native_tcgen05_and_one_deeper_does_not():
    # The SIMT fallback a normalized config can still take (the MMA
    # lowering's own shared-memory check) is priced and gated: at each
    # sampled schedule's deepest admitted ring the emitted producer is native
    # TCGEN05 with one TMA store ring; one stage deeper is refused, never
    # silently lowered to SIMT.
    bound = bind(four_outputs, four_args())
    spec = bound.config_spec
    admission = spec.cute_result_fission_producer_facts
    seed = _seed(bound)
    samples = [
        (128, 256, 128, 1, "auto", 2, 2, "persistent_interleaved"),
        (128, 128, 64, 1, "auto", 2, 2, "persistent_interleaved"),
        (128, 128, 64, 2, "two", 2, 2, "persistent_interleaved"),
        (256, 128, 64, 2, "auto", 2, 2, "persistent_interleaved"),
        (64, 64, 64, 1, "auto", 1, 4, "persistent_interleaved"),
    ]
    for schedule in samples:
        limit = admission.max_ab_stages[schedule]
        assert limit >= 1, schedule
        keys = cfgkeys.producer_schedule_keys(schedule, limit)
        config = helion.Config.from_dict({**seed.config, **keys})
        spec.normalize(config)
        producer = next(iter(embedded_modules(bound.to_code(config)).values()))
        assert "tcgen05" in producer
        assert producer.count("PipelineTmaStore.create(") == 1
        if limit < cfgkeys.MAX_PRODUCER_AB_STAGES:
            deeper = helion.Config.from_dict(
                {**seed.config, **cfgkeys.producer_schedule_keys(schedule, limit + 1)}
            )
            with pytest.raises(exc.InvalidConfig, match="not admitted"):
                spec.normalize(deeper)
            with (
                patch.object(cfgkeys, "producer_schedule_problem", return_value=None),
                pytest.raises(exc.BackendUnsupported, match="producer"),
            ):
                bound.to_code(deeper)


def test_admission_failure_never_fails_the_bind():
    fresh = helion.kernel(four_outputs.fn, settings=four_outputs.settings)
    with patch.object(rfc, "producer_admission", side_effect=RuntimeError("boom")):
        bound = bind(fresh, four_args())
    spec = bound.config_spec
    assert spec.cute_result_fission_available is False
    assert spec.cute_result_fission_producer_facts is None
    assert not any(k.startswith("cute_result_fission") for k in spec._flat_fields())
    assert fission_seeds(bound) == []
    bound.to_code(spec.default_config())


def test_admission_is_keyed_by_pid_type():
    # The CTA pair under a flat pid schedule prices differently from the
    # persistent schedules: the table keys every producer pid type.
    bound = bind(four_outputs, four_args())
    table = bound.config_spec.cute_result_fission_producer_facts.max_ab_stages
    pids = {schedule[-1] for schedule in table}
    assert pids == set(cfgkeys.PRODUCER_PID_TYPES)


def test_registration_analysis_failure_never_fails_the_bind():
    fresh = helion.kernel(four_outputs.fn, settings=four_outputs.settings)
    with patch.object(rf, "analyze_result_fission", side_effect=RuntimeError("boom")):
        bound = bind(fresh, four_args())
    assert bound.config_spec.cute_result_fission_available is False
    assert fission_seeds(bound) == []
    bound.to_code(bound.config_spec.default_config())


def test_clustered_flat_schedule_is_refused_statically():
    # Clustered tcgen05 codegen raises for a flat launch: never admitted.
    bound = bind(four_outputs, four_args())
    table = bound.config_spec.cute_result_fission_producer_facts.max_ab_stages
    assert not any(s[3] == 2 and s[7] == "flat" for s in table)
    keys = cfgkeys.producer_schedule_keys((256, 128, 64, 2, "auto", 2, 2, "flat"), 1)
    assert "persistent" in cfgkeys.producer_schedule_problem(keys, None)
