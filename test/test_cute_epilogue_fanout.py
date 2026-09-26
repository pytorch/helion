from __future__ import annotations

import ast
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from itertools import count
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch._subclasses.fake_tensor import FakeTensor

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.epilogue_fanout import FanoutIteration
from helion._compiler.cute.epilogue_fanout import FanoutStore
from helion._compiler.cute.epilogue_fanout import PairedFanoutPlan
from helion._compiler.cute.epilogue_fanout import _fresh_returned_tensors
from helion._compiler.cute.epilogue_fanout import combine_stores
from helion._compiler.cute.epilogue_fanout import prove_paired_fanout
from helion._compiler.cute.epilogue_fanout import render_chain
from helion._compiler.cute.epilogue_fanout import render_shared_suffix
from helion._compiler.cute.epilogue_fanout import schedule_supported
from helion._compiler.type_info import LiteralType
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
import helion.runtime.cute.launcher as launcher

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from typing import Any

    from torch.fx.node import Argument

    from helion._compiler.cute.cute_epilogue import _AuxiliaryTensorLoadExpr
    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel


def paired(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    gate_out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    product_out = torch.empty_like(gate_out)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        gate = torch.sigmoid(accumulator).to(left.dtype)
        gate_out[row, column] = gate
        product_out[row, column] = residual[row, column] * gate
    return gate_out, product_out


def renamed_relu(
    a: torch.Tensor, b: torch.Tensor, extra: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    height, inner = a.shape
    width = b.size(1)
    first = torch.empty((height, width), dtype=a.dtype, device=a.device)
    second = torch.empty_like(first)
    for i, j in hl.tile((height, width)):
        value = hl.zeros([i, j], dtype=torch.float32)
        for r in hl.tile(inner):
            value = torch.addmm(value, a[i, r], b[r, j])
        intermediate = torch.relu(value + 0.25).to(a.dtype)
        first[i, j] = intermediate
        second[i, j] = intermediate + extra[i, j]
    return first, second


def duplicate(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    first = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    second = torch.empty_like(first)
    for i, j in hl.tile((rows, columns)):
        accumulator = hl.zeros([i, j], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[i, k], right[k, j])
        gate = torch.sigmoid(accumulator)
        first[i, j] = gate
        second[i, j] = gate
    return first, second


def unrounded(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    first = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    second = torch.empty_like(first)
    for i, j in hl.tile((rows, columns)):
        accumulator = hl.zeros([i, j], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[i, k], right[k, j])
        gate = torch.sigmoid(accumulator)
        first[i, j] = gate
        second[i, j] = residual[i, j] * gate
    return first, second


def aliased_outputs(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    first = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    second = first
    for i, j in hl.tile((rows, columns)):
        accumulator = hl.zeros([i, j], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[i, k], right[k, j])
        gate = torch.sigmoid(accumulator).to(left.dtype)
        first[i, j] = gate
        second[i, j] = residual[i, j] * gate
    return first, second


def output_reload(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    first = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    second = torch.empty_like(first)
    for i, j in hl.tile((rows, columns)):
        accumulator = hl.zeros([i, j], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[i, k], right[k, j])
        first[i, j] = torch.sigmoid(accumulator)
        second[i, j] = residual[i, j] * first[i, j]
    return first, second


_mutate_output_storage = False


def _opaque_metadata(first: torch.Tensor, second: torch.Tensor) -> int:
    if _mutate_output_storage:
        # Installed Torch stubs omit the supported Tensor overload of set_.
        first.set_(cast("Any", second))
    return 0


def opaque_scalar(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    gate_out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    product_out = torch.empty_like(gate_out)
    token = _opaque_metadata(gate_out, product_out)
    for row, column in hl.tile((rows, columns + token)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        gate = torch.sigmoid(accumulator).to(left.dtype)
        gate_out[row, column] = gate
        product_out[row, column] = residual[row, column] * gate
    return gate_out, product_out


class _MetadataProvider(ModuleType):
    reads = 0

    @property
    def zero(self) -> int:
        self.reads += 1
        return 0

    @property
    def empty(self) -> Callable[..., torch.Tensor]:
        self.reads += 1
        return torch.empty


# Modules use the ordinary frontend's actual attribute lookup. A subclass can
# expose effectful descriptors even when the sampled result is an integer.
_metadata_provider = _MetadataProvider("_epilogue_metadata")


def opaque_property(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1) + _metadata_provider.zero
    gate_out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    product_out = torch.empty_like(gate_out)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        gate = torch.sigmoid(accumulator).to(left.dtype)
        gate_out[row, column] = gate
        product_out[row, column] = residual[row, column] * gate
    return gate_out, product_out


def opaque_factory(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, reduction = left.shape
    columns = right.size(1)
    gate_out = _metadata_provider.empty(
        (rows, columns), dtype=left.dtype, device=left.device
    )
    product_out = torch.empty_like(gate_out)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        gate = torch.sigmoid(accumulator).to(left.dtype)
        gate_out[row, column] = gate
        product_out[row, column] = residual[row, column] * gate
    return gate_out, product_out


def known_metadata(
    left: torch.Tensor, right: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = (left.shape[0] + 1) - 1
    reduction = left.size(1)
    columns = right.size(1)
    gate_out = torch.empty((rows, columns), dtype=torch.float32, device=left.device)
    product_out = torch.empty_like(gate_out)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        gate = torch.sigmoid(accumulator)
        gate_out[row, column] = gate
        product_out[row, column] = gate
    return gate_out, product_out


@pytest.fixture(autouse=True)
def cuda_trace() -> Generator[None, None, None]:
    """Real static converter, with only the traced device replaced by CUDA."""
    old_init = CompileEnvironment.__init__
    old_convert = CompileEnvironment._to_fake_tensor
    old_threads = torch.get_num_threads()
    initial_cuda_state = torch.cuda.is_initialized()
    torch.set_num_threads(1)

    def initialize(env: Any, device: torch.device, *args: Any, **kwargs: Any) -> None:
        assert device.type in ("cpu", "cuda")
        old_init(env, torch.device("cuda:0"), *args, **kwargs)

    def convert(env: Any, tensor: torch.Tensor, origin: Any) -> FakeTensor:
        result = old_convert(env, tensor, origin)
        assert isinstance(result, FakeTensor) and result.fake_mode is env.fake_mode
        result.fake_device = torch.device("cuda:0")
        return result

    try:
        with ExitStack() as stack:
            stack.enter_context(_target())
            stack.enter_context(
                patch.object(CompileEnvironment, "__init__", initialize)
            )
            stack.enter_context(
                patch.object(CompileEnvironment, "_to_fake_tensor", convert)
            )
            stack.enter_context(_mock_cuda_unavailable())
            stack.enter_context(
                patch(
                    "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
                )
            )
            stack.enter_context(
                patch("torch.cuda.get_device_capability", return_value=(10, 0))
            )
            stack.enter_context(
                patch.object(
                    DeviceProperties,
                    "create",
                    return_value=DeviceProperties(
                        type="cuda",
                        index=0,
                        multi_processor_count=148,
                        cc=100,
                        major=10,
                        regs_per_multiprocessor=65536,
                        max_threads_per_multi_processor=2048,
                        max_threads_per_block=1024,
                        warp_size=32,
                    ),
                )
            )
            yield
    finally:
        torch.set_num_threads(old_threads)
        assert torch.cuda.is_initialized() == initial_cuda_state


def bind(
    fn: Callable[..., object] = paired,
    *,
    dtype: torch.dtype = torch.float16,
    rows: int = 256,
    columns: int = 128,
    **settings: Any,
) -> BoundKernel[Any]:
    args = (
        torch.empty((rows, 64), dtype=dtype),
        torch.empty((64, columns), dtype=dtype),
        torch.empty((rows, columns), dtype=dtype),
    )
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full", **settings
    )
    return kernel._bind_isolated(args)


def config(bound: BoundKernel[Any], mode: str = "shared") -> helion.Config:
    requested = helion.Config(
        block_sizes=[128, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_num_epi_warps=4,
    )
    requested.config[FANOUT_CONFIG_KEY] = mode
    return bound.config_spec.normalized_config(requested)


def plan_for(bound: BoundKernel[Any]) -> PairedFanoutPlan:
    (plan,) = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    return plan


@pytest.mark.parametrize("fn", (paired, renamed_relu, duplicate))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@skipUnlessBackends(["cute"])
def test_typed_public_fanout_and_default_source(
    fn: Callable[..., object], dtype: torch.dtype
) -> None:
    bound = bind(fn, dtype=dtype)
    plan = plan_for(bound)
    assert plan.output_dtype is dtype
    off, shared = config(bound, "off"), config(bound)
    assert FANOUT_CONFIG_KEY not in off.config
    assert bound.to_code(off) != bound.to_code(shared)
    off.config[FANOUT_CONFIG_KEY] = "off"
    assert bound.to_code(off) == bound.to_code(config(bound, "off"))
    assert plan.rounded_suffix is (fn is not duplicate)


@pytest.mark.parametrize("fn", (unrounded, aliased_outputs, output_reload))
@skipUnlessBackends(["cute"])
def test_unproved_rounding_alias_and_reload_decline(fn: Callable[..., object]) -> None:
    bound = bind(fn)
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    with pytest.raises(InvalidConfig, match=FANOUT_CONFIG_KEY):
        config(bound)
    requested = helion.Config.from_dict(
        deepcopy(config(bound, "off").config) | {FANOUT_CONFIG_KEY: "shared"}
    )
    bound.config_spec.normalize(requested, _fix_invalid=True)
    assert FANOUT_CONFIG_KEY not in requested.config


@pytest.mark.parametrize("rows,columns", ((257, 128), (256, 129), (130, 65)))
@skipUnlessBackends(["cute"])
def test_partial_tiles_retain_separate_fallback(rows: int, columns: int) -> None:
    bound = bind(rows=rows, columns=columns)
    assert plan_for(bound)
    with pytest.raises(InvalidConfig, match=FANOUT_CONFIG_KEY):
        config(bound)
    if rows == 130:
        # The inherited C1 implementation also declines this double-edge tile.
        with pytest.raises(BackendUnsupported, match="double-edge"):
            bound.to_code(config(bound, "off"))
    else:
        assert bound.to_code(config(bound, "off"))


@pytest.mark.parametrize("value", (None, True, False, 0, 1, "invalid"))
@skipUnlessBackends(["cute"])
def test_mode_is_typed_and_normalized(value: object) -> None:
    bound = bind()
    raw = deepcopy(config(bound, "off").config)
    raw[FANOUT_CONFIG_KEY] = value
    with pytest.raises(InvalidConfig, match=FANOUT_CONFIG_KEY):
        bound.config_spec.normalize(helion.Config.from_dict(raw))


@pytest.mark.parametrize(
    "key,value",
    (
        ("tcgen05_cluster_n", 2),
        ("tcgen05_c_stages", 3),
        ("tcgen05_num_epi_warps", 2),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_warp_spec_store_warps", 1),
        ("tcgen05_warp_spec_c_input_warps", 1),
        ("tcgen05_c_acquire_placement", "first_in_loop"),
        ("tcgen05_acc_wait_placement", "before_subtile_loop"),
        ("tcgen05_epilogue_layout", "split_first_t2r"),
        ("tcgen05_layout_overrides_epi_tile_n", 32),
    ),
)
def test_unproved_protocol_compositions_decline(key: str, value: object) -> None:
    assert not schedule_supported({key: value})


def original_bound() -> BoundKernel[Any]:
    args = tuple(
        torch.empty(shape, dtype=torch.float16, requires_grad=True)
        for shape in (
            (4096, 1024),
            (1024, 512),
            (512, 1024),
        )
    )
    kernel = helion.kernel(
        squeeze_and_excitation_net_fwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
    )
    return kernel._bind_isolated(args)


@skipUnlessBackends(["cute"])
def test_original_cuda_traced_two_region_binding() -> None:
    bound = original_bound()
    plan = plan_for(bound)
    assert plan.shape == (4096, 1024)
    assert plan.prefix_steps == 1 and plan.rounded_suffix
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) == 2
    selected = helion.Config(
        block_sizes=[128] * 6,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_num_epi_warps=4,
    )
    selected.config[FANOUT_CONFIG_KEY] = "shared"
    assert bound.to_code(selected)


@pytest.mark.parametrize(
    "statement",
    (
        "gate_out.set_(left)",
        "opaque(gate_out)",
        "alias = gate_out.view(-1)",
        "gate_out = left",
        "left = gate_out",
        "gate_out[0] = 1",
    ),
)
@skipUnlessBackends(["cute"])
def test_typed_host_freshness_rejects_effects_and_rebinding(statement: str) -> None:
    bound = bind()
    host = bound.host_function
    assert host is not None
    root = next(i for i, stmt in enumerate(host.body) if isinstance(stmt, ast.For))
    altered = cast(
        "HostFunction",
        SimpleNamespace(
            params=host.params,
            body=[*host.body[:root], *ast.parse(statement).body, *host.body[root:]],
        ),
    )
    with bound.env, host:
        assert len(_fresh_returned_tensors(host)) == 2
        assert not _fresh_returned_tensors(altered)


@pytest.mark.parametrize("fn", (opaque_scalar, opaque_property, opaque_factory))
@skipUnlessBackends(["cute"])
def test_actual_typed_opaque_metadata_declines(fn: Callable[..., object]) -> None:
    reads = _metadata_provider.reads
    bound = bind(fn)
    if fn is not opaque_scalar:
        assert _metadata_provider.reads > reads
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        assert not _fresh_returned_tensors(host)
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert not any(
        group.key == FANOUT_CONFIG_KEY
        for group in bound.config_spec.compiler_coverage_groups
    )
    with pytest.raises(InvalidConfig, match=FANOUT_CONFIG_KEY):
        config(bound)
    # The opaque scalar/properties really propagate through the installed
    # frontend; this is not an artificial untyped AST inserted after binding.
    literals = [
        node
        for statement in host.body
        for node in ast.walk(statement)
        if isinstance(node, (ast.Call, ast.Attribute))
        and isinstance(node, ExtendedAST)
        and isinstance(node._type_info, LiteralType)
        and node._type_info.value == 0
    ]
    if fn is not opaque_factory:
        assert literals


@skipUnlessBackends(["cute"])
def test_opaque_scalar_fallback_preserves_actual_allocating_host_effect() -> None:
    global _mutate_output_storage
    values = (
        torch.empty((256, 64), dtype=torch.float16),
        torch.empty((64, 128), dtype=torch.float16),
        torch.empty((256, 128), dtype=torch.float16),
    )
    kernel = helion.kernel(
        opaque_scalar, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = kernel.bind(values)
    bound.set_config(config(bound, "off"))
    host = bound._run
    assert host is not None
    defaults = host.__kwdefaults__
    assert defaults is not None
    assert defaults["_launcher"] is launcher.default_cute_launcher
    calls = []

    def hold(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    defaults["_launcher"] = hold
    try:
        before = bound(*values)
        assert before[0].untyped_storage()._cdata != before[1].untyped_storage()._cdata
        _mutate_output_storage = True
        assert kernel.bind(values) is bound
        after = bound(*values)
        assert bound._run is host
        assert after[0].untyped_storage()._cdata == after[1].untyped_storage()._cdata
        assert len(calls) == 2
    finally:
        _mutate_output_storage = False
        defaults["_launcher"] = launcher.default_cute_launcher


@skipUnlessBackends(["cute"])
def test_known_metadata_constants_and_fp32_outputs_remain_supported() -> None:
    bound = bind(known_metadata)
    assert plan_for(bound).output_dtype is torch.float32
    assert bound.to_code(config(bound))


@skipUnlessBackends(["cute"])
def test_carrierless_binding_keeps_old_search_registry() -> None:
    for bound in (bind(), bind(rows=257)):
        assert plan_for(bound)
        state = bound.config_spec._cute_tcgen05_config
        assert not state.epilogue_fanout_search_enabled
        assert not any(
            group.key == FANOUT_CONFIG_KEY
            for group in bound.config_spec.compiler_coverage_groups
        )


@pytest.mark.parametrize(
    "change", ("mask", "indices", "third_store", "wrong_graph", "dtype")
)
@skipUnlessBackends(["cute"])
def test_live_typed_store_proof_negatives(change: str) -> None:
    bound = bind()
    plan = plan_for(bound)
    first, second = plan.stores
    original = second.args
    host = bound.host_function
    assert host is not None
    graphs = host.device_ir.graphs
    anchor = next(
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    stores = list(plan.stores)
    target = second.args[0]
    assert isinstance(target, torch.fx.Node)
    target_value = target.meta["val"]
    try:
        if change == "mask":
            second.args = (*second.args[:3], True)
        elif change == "indices":
            second.args = (
                second.args[0],
                list(reversed(cast("list[Argument]", second.args[1]))),
                *second.args[2:],
            )
        elif change == "third_store":
            stores.append(first)
        elif change == "wrong_graph":
            foreign = torch.fx.Graph()
            stores[1] = foreign.placeholder("other")
        elif change == "dtype":
            target.meta["val"] = target_value.to(torch.float32)
        with bound.env, host:
            assert (
                prove_paired_fanout(bound.env, host, graphs, stores, {anchor}) is None
            )
    finally:
        second.args = original
        target.meta["val"] = target_value


@skipUnlessBackends(["cute"])
def test_registered_witnesses_are_independent_and_cache_visible() -> None:
    bound = original_bound()
    spec = bound.config_spec
    (group,) = [g for g in spec.compiler_coverage_groups if g.key == FANOUT_CONFIG_KEY]
    assert len(group.witnesses) == 4
    payload = [w.carrier.config for w in group.witnesses]
    assert [row["tcgen05_c_stages"] for row in payload] == [4, 4, 2, 2]
    identities = [
        {key: value for key, value in row.items() if key != "tcgen05_c_stages"}
        for row in payload
    ]
    assert all(row == identities[0] for row in identities)
    first = group.witnesses[0].carrier
    first.block_sizes[0] = 999
    assert [w.carrier.config for w in group.witnesses] == payload
    assert all(FANOUT_CONFIG_KEY not in c.config for c in spec.compiler_seed_configs)
    assert spec.compiler_default_config is not None
    assert FANOUT_CONFIG_KEY not in spec.compiler_default_config.config
    old_fingerprint = spec.cache_fingerprint_hash()
    old_groups = spec._compiler_coverage_groups
    try:
        spec._compiler_coverage_groups = tuple(
            replace(g, version=2) if g.key == FANOUT_CONFIG_KEY else g
            for g in old_groups
        )
        assert spec.cache_fingerprint_hash() != old_fingerprint
    finally:
        spec._compiler_coverage_groups = old_groups
    with bound.env:
        gen = spec.create_config_generation()
        for witness in group.witnesses:
            requested = witness.carrier
            requested.config[FANOUT_CONFIG_KEY] = witness.value
            flat, effective = gen.strict_config_pair(requested)
            assert effective.config.get(FANOUT_CONFIG_KEY, "off") == witness.value
            assert gen.unflatten(flat) == effective
            assert gen.strict_config_pair(effective) == (flat, effective)


def _rendered_plan(
    dtype: torch.dtype,
) -> tuple[PairedFanoutPlan, dict[_AuxiliaryTensorLoadExpr, str]]:
    bound = bind(dtype=dtype)
    plan = plan_for(bound)
    leaves = plan.chains[1].auxiliary_tensor_loads
    assert len(leaves) == 1
    return plan, {leaves[0]: "aux"}


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@skipUnlessBackends(["cute"])
def test_actual_typed_suffix_ieee_bits_and_required_rounding(
    dtype: torch.dtype,
) -> None:
    """Test arbitrary FP32 prefix results; do not emulate CUDA's approximate exp."""
    plan, auxiliary = _rendered_plan(dtype)
    names = count()
    rendered = render_chain(
        plan.chains[1], "prefix", lambda name: f"{name}_{next(names)}", "", auxiliary
    )
    source, result = render_shared_suffix(plan, rendered, "rounded", "", auxiliary)
    generator = torch.Generator().manual_seed(9013)
    # Exhaust the low-precision bit domain and finite FP32 midpoint perturbations.
    aux = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(dtype).float()
    prefix = torch.randn(1 << 16, generator=generator) * 4
    prefix[:8] = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            2**-24,
            -(2**-24),
            1.0001,
        ]
    )
    rounded = prefix.to(dtype)
    namespace = {
        "rounded": rounded,
        "aux": aux,
        "cutlass": SimpleNamespace(
            Float16=torch.float16, BFloat16=torch.bfloat16, Float32=torch.float32
        ),
    }
    exec(compile(source, "typed-shared-suffix", "exec"), namespace)
    actual = namespace[result].to(dtype)
    expected = (aux * prefix.to(dtype).float()).to(dtype)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    unrounded_value = (aux * prefix).to(dtype)
    finite = torch.isfinite(actual) & torch.isfinite(unrounded_value)
    assert torch.any(
        actual.view(torch.int16)[finite] != unrounded_value.view(torch.int16)[finite]
    )


def _store_records() -> tuple[FanoutStore, FanoutStore]:
    plan, aux = _rendered_plan(torch.float16)
    names = count()
    records = []
    for index, chain in enumerate(plan.chains):
        rendered = render_chain(
            chain,
            f"acc{index}",
            lambda name: f"{name}_{next(names)}",
            "        ",
            {} if index == 0 else aux,
        )
        main = cast("ast.If", ast.parse("if True:\n    pass").body[0])
        iteration = FanoutIteration(
            tmem_read=f"        read{index}()\n",
            auxiliary_loads="",
            arithmetic=rendered.source,
            assignment=f"        value{index} = {rendered.result}\n",
            value_name=f"value{index}",
            target_dtype="cutlass.Float16",
            release="        release()\n" if index else "",
            register_store=f"        store{index}()\n",
            chain=rendered,
            auxiliary_names={} if index == 0 else aux,
        )
        records.append(
            FanoutStore(
                plan=plan,
                site=plan.stores[index],
                main=main,
                setup=(),
                setup_without_acquire=(),
                iteration=iteration,
                subtile_count=f"count{index}",
                acquire="        acquire()\n",
                wait="        wait()\n" if not index else "",
                barrier=f"barrier{index}",
                buffer=f"buffer{index}",
                buffer_expr="tile * 4 + _tcgen05_subtile",
                c_stages=4,
                r2s=f"        r2s{index}()\n",
                tma=("        if warp0:\n" if not index else "")
                + f"            tma{index}()\n",
                commit="            commit()\n",
                advance="advance()\n" if index else "",
            )
        )
    return records[0], records[1]


@pytest.mark.parametrize(
    "fault",
    (
        "reversed",
        "stage",
        "early_release",
        "missing_release",
        "early_advance",
        "missing_advance",
    ),
)
@skipUnlessBackends(["cute"])
def test_pair_lifetime_negatives(fault: str) -> None:
    first, second = _store_records()
    if fault == "reversed":
        first, second = second, first
    elif fault == "stage":
        second = replace(second, c_stages=2)
    elif fault == "early_release":
        first = replace(first, iteration=replace(first.iteration, release="release()"))
    elif fault == "missing_release":
        second = replace(second, iteration=replace(second.iteration, release=""))
    elif fault == "early_advance":
        first = replace(first, advance="advance()")
    elif fault == "missing_advance":
        second = replace(second, advance="")
    with pytest.raises(BackendUnsupported, match="lifetime"):
        combine_stores(first, second)


@skipUnlessBackends(["cute"])
def test_complete_store_registration_and_single_commit_group() -> None:
    first, second = _store_records()
    state = CuteDeviceFunctionState()
    for store in (first, second):
        state.register_tcgen05_per_tile_stmts([store.main])
        state.register_tcgen05_epi_role_stmts([store.main])
    assert not state.register_paired_fanout_store(first, [])
    with pytest.raises(BackendUnsupported, match="incomplete"):
        state.finalize_tcgen05_pure_lifecycle_stores()
    assert state.register_paired_fanout_store(second, [])
    state.finalize_tcgen05_pure_lifecycle_stores()
    emitted = ast.unparse(first.main)
    assert emitted.count("commit()") == 1
    assert emitted.count("barrier0.arrive_and_wait()") == 2
    assert "barrier1.arrive_and_wait()" not in emitted
    assert emitted.index("tma0()") < emitted.index("tma1()") < emitted.index("commit()")
    assert emitted.index("r2s0()") < emitted.index("fence_view_async_shared()")
    assert emitted.index("r2s1()") < emitted.index("fence_view_async_shared()")
    assert state.is_tcgen05_epi_role(first.main)
    assert not state.is_tcgen05_epi_role(second.main)
    assert not state.is_tcgen05_per_tile(second.main)
