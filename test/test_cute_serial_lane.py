from __future__ import annotations

import ast
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
import helion
from helion import exc
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import serial_lane_guard
from helion.runtime.kernel import OutputCodeOptions

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_serial_lane_schedule"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _recurrence(states, coefficient, out):
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _specialized(states, coefficient, out, steps):
    steps = hl.specialize(steps)
    batch, _, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], 0.25, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fresh(states, coefficient):
    batch, steps, features = states.shape
    out = torch.empty_like(states)
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cross_lane(states, coefficient, out):
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                coefficient[bi.begin, qi] * carry
                + states[bi.begin, qi, (fi + 1) % features]
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _negative(states, coefficient, out, kind):
    kind = hl.specialize(kind)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        if kind == 3:
            carry = states[bi.begin, 0, fi].float()
        else:
            carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(steps):
            if kind == 0:
                states[bi.begin, qi, fi] = carry
            if kind == 2:
                out[bi.begin, qi, fi] = (carry + 1).to(out.dtype)
            else:
                out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
            if kind == 1:
                carry = carry.to(torch.float16).float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _derived_view(states, coefficient, out):
    states = states.reshape(2, 7, 64)
    for bi, fi in hl.tile([2, 64], block_size=[1, None]):
        carry = hl.zeros([fi], dtype=torch.float32)
        for qi in hl.grid(7):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


def _args(dtype=torch.bfloat16, *, features=64, stride=1, offset=False):
    def tensor(shape, dtype):
        if offset:
            count = 1
            for size in shape:
                count *= size
            return torch.empty(count + 16 // dtype.itemsize, dtype=dtype)[
                16 // dtype.itemsize :
            ].view(shape)
        backing = torch.empty((*shape[:-1], shape[-1] * stride), dtype=dtype)
        return backing[..., ::stride]

    return (
        tensor((2, 7, features), torch.float32),
        tensor((2, 7), torch.float32),
        tensor((2, 7, features), dtype),
    )


def _config(mode="step_major_vector", width=4):
    values: dict[str, object] = {
        "block_sizes": [64],
        "num_threads": [64 // width],
        "cute_vector_widths": [1, width, 1],
    }
    if mode is not None:
        values[KEY] = mode
    return helion.Config.from_dict(values)


def _source(args=None, mode="step_major_vector", width=4, kernel=_recurrence):
    with _cpu_codegen():
        bound = kernel._bind_isolated(_args() if args is None else args)
        return bound.to_code(_config(mode, width))


def _module(source: str, launches: list) -> ModuleType:
    """Execute real generated host, but never compile or execute its devices."""
    tree = ast.parse(source)
    tree.body = [
        n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ModuleType("serial_cpu_generated")
    module.__file__ = "<serial-cpu-generated>"
    module.__dict__.update(
        torch=torch,
        _default_cute_launcher=lambda fn, grid, *args, **kw: launches.append(
            (fn, grid, args, kw)
        ),
    )
    exec(
        compile(ast.fix_missing_locations(tree), module.__file__, "exec"),
        module.__dict__,
    )
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("mode", ["step_major", "step_major_vector"])
def test_source_original_fallback_math_and_typed_transfers(dtype, width, mode):
    args = _args(dtype)
    old = _source(args, None, width)
    new = _source(args, mode, width)
    old_kernel = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef))
    kernels = [n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)]
    assert ast.dump(kernels[0]) == ast.dump(old_kernel)
    assert "serial_carry_values" in new
    assert new.count("cute.math.exp2(") == 2
    assert "validate_entry" in new and "select_kernel" in new
    assert "sync_threads" not in new
    if mode == "step_major_vector":
        assert f"num_bits_per_copy={width * 32}" in new
        assert f"num_bits_per_copy={width * dtype.itemsize * 8}" in new
        assert ".toint() % " in new and ".layout.stride[2] == 1" in new
    else:
        assert "CopyUniversalOp" not in new


def test_missing_and_explicit_default_whole_source_equal():
    assert _source(mode=None) == _source(mode="lane_major")


@pytest.mark.parametrize("value", [True, False, 2, "other"])
def test_invalid_schedule_rejects(value):
    with pytest.raises(exc.InvalidConfig, match="serial"):
        _source(mode=value)


@pytest.mark.parametrize("kind", ["stride", "offset"])
def test_current_stride_and_offset_contract(kind):
    args = _args(stride=2 if kind == "stride" else 1, offset=kind == "offset")
    launches = []
    module = _module(_source(args), launches)
    with patch.object(serial_lane_guard, "_require_cuda"):
        assert module._recurrence(*args) is args[2]
    assert launches[-1][0].__name__.endswith("_serial_lane")


def test_actual_raw_compile_callable_rechecks_current_aliases():
    from torch._inductor.codecache import PyCodeCache

    args = _args()
    launches = []
    sources = []

    def load(source, **kwargs):
        sources.append(source)
        return _module(source, launches)

    with _cpu_codegen(), patch.object(PyCodeCache, "load", side_effect=load):
        bound = _recurrence._bind_isolated(args)
        raw = bound.compile_config(_config(), allow_print=False)
        assert bound.compile_config(_config(), allow_print=False) is raw
    alias = (
        args[0].view(torch.bfloat16).flatten()[: args[2].numel()].view(args[2].shape)
    )
    with patch.object(serial_lane_guard, "_require_cuda"):
        raw(*args)
        raw(args[0], args[1], alias)
        raw(*_args())
    assert [row[0].__name__.endswith("_serial_lane") for row in launches] == [
        True,
        False,
        True,
    ]
    assert len(sources) == 1


def test_public_specialization_guard_precedes_overwrite():
    args = (*_args(), 7)
    source = _source(args, kernel=_specialized)
    host = ast.parse(source).body[-1]
    text = ast.unparse(host)
    assert text.index("validate_entry") < text.index("_launcher(")
    # The actual generated device signature omits this literal. A late-only
    # launch guard could not recover a changed original steps argument.
    assert "steps" in text[text.index("validate_entry") : text.index("_launcher(")]
    launches = []
    module = _module(source, launches)
    with patch.object(serial_lane_guard, "_require_cuda"):
        with pytest.raises(exc.BackendUnsupported, match="scalar value"):
            module._specialized(*args[:-1], 6)
        assert launches == []
        module._specialized(*args)
    assert len(launches) == 1


@pytest.mark.parametrize("kind", ["shape", "stride", "lazy", "dtype"])
def test_raw_current_metadata_rejects_before_launcher(kind):
    args = _args()
    launches = []
    module = _module(_source(args), launches)
    changed = list(args)
    if kind == "shape":
        changed[0] = args[0].view(2, 64, 7)
    elif kind == "stride":
        changed[0] = torch.empty(2, 64, 7).transpose(1, 2)
    elif kind == "lazy":
        changed[0] = torch._neg_view(args[0])
    else:
        changed[0] = args[0].to(torch.float16)
    with (
        patch.object(serial_lane_guard, "_require_cuda"),
        pytest.raises(exc.BackendUnsupported),
    ):
        module._recurrence(*changed)
    assert launches == []


def test_fresh_current_host_output_and_retention():
    args = _args()[:2]
    launches = []
    module = _module(_source(args, kernel=_fresh), launches)
    with patch.object(serial_lane_guard, "_require_cuda"):
        first = module._fresh(*args)
        second = module._fresh(*args)
    assert first is not second and first.data_ptr() != second.data_ptr()
    assert launches[0][2][0] is first and launches[1][2][0] is second


def test_cross_lane_and_tail_reject():
    with pytest.raises(exc.InvalidConfig, match="serial"):
        _source(kernel=_cross_lane)
    with pytest.raises(exc.BackendUnsupported, match="serial"):
        _source(_args(features=65))


def test_same_object_bind_does_not_erase_public_dependency():
    args = _args(torch.float32)
    with pytest.raises(exc.BackendUnsupported, match="serial"):
        _source((args[0], args[1], args[0]))


def test_dependency_free_export_rejects():
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
    with pytest.raises(NotImplementedError, match="allow_helion_deps=False"):
        bound.to_code(_config(), options=OutputCodeOptions(allow_helion_deps=False))


def test_genuine_cpu_dynamo_rejects_before_any_launcher():
    launches = []
    module = _module(_source(), launches)
    # Device functions are never compiled or executed. Dynamo itself may
    # initialize CUDA while inspecting tensor guards in a CUDA-enabled process.
    fn = torch.compile(module._recurrence, backend="eager", fullgraph=True)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        fn(*_args())
    assert launches == []


def test_runtime_device_check_not_weakened():
    initialized = torch.cuda.is_initialized()
    launches = []
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only")):
        module = _module(_source(), launches)
        with pytest.raises(exc.BackendUnsupported, match="requires CUDA"):
            module._recurrence(*_args())
    assert launches == [] and torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("managed", [False, True])
def test_real_bound_and_prepared_calls_keep_host_guards(managed):
    from torch._inductor.codecache import PyCodeCache

    args = _args()
    launches = []
    kernel = helion.kernel(
        _recurrence.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache,
            "load",
            side_effect=lambda source, **kwargs: _module(source, launches),
        ),
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        bound = kernel.bind(args) if managed else kernel._bind_isolated(args)
        bound.set_config(_config())
        bound(*args)
        bound(*args)
        alias = (
            args[0]
            .view(torch.bfloat16)
            .flatten()[: args[2].numel()]
            .view(args[2].shape)
        )
        bound(args[0], args[1], alias)
        bound(*args)
    assert [v[0].__name__.endswith("_serial_lane") for v in launches] == [
        True,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize("route", ["dynamo_full", "dynamo_break", "export", "jit"])
def test_genuine_tracing_never_captures_pointer_selection(route):
    args = _args()
    launches = []
    host = _module(_source(args), launches)._recurrence

    class Wrapper(torch.nn.Module):
        def forward(self, states, coefficient, out):
            return host(states, coefficient, out)

    with (
        patch.object(serial_lane_guard, "_require_cuda"),
        pytest.raises((torch._dynamo.exc.Unsupported, exc.BackendUnsupported)),
    ):
        if route.startswith("dynamo"):
            compiled = torch.compile(
                Wrapper(), backend="eager", fullgraph=route == "dynamo_full"
            )
            compiled(*args)
        elif route == "export":
            torch.export.export(Wrapper(), args)
        else:
            torch.jit.trace(Wrapper(), args)
    assert launches == []


@pytest.mark.parametrize(
    "kind", ["subclass", "layout", "offset", "read_alias", "output_overlap"]
)
def test_late_guard_ordinary_launcher_contract_and_whole_backing(kind):
    args = list(_args())
    launches = []
    host = _module(_source(args), launches)._recurrence
    if kind == "subclass":
        args[0] = torch.nn.Parameter(args[0])
    elif kind == "layout":
        args[0] = args[0].to_sparse()
    elif kind == "offset":
        args[0] = torch.empty(args[0].numel() + 1)[1:].view(args[0].shape)
    elif kind == "read_alias":
        args[1] = args[0].flatten()[: args[1].numel()].view(args[1].shape)
    else:
        backing = torch.empty(args[0].numel() + args[2].numel())
        args[0] = backing[: args[0].numel()].view(args[0].shape)
        args[2] = (
            backing[args[0].numel() :]
            .view(torch.bfloat16)[: args[2].numel()]
            .view(args[2].shape)
        )
    with patch.object(serial_lane_guard, "_require_cuda"):
        if kind in ("subclass", "layout", "offset"):
            with pytest.raises(exc.BackendUnsupported):
                host(*args)
            assert not launches
        else:
            host(*args)
            assert launches[0][0].__name__.endswith("_serial_lane") == (
                kind == "read_alias"
            )


def test_general_seeds_default_filtered_order_and_actual_initial100():
    from helion._compiler.autotuner_heuristics import compiler_seed_configs
    from helion.autotuner.config_generation import ConfigGeneration

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        spec = bound.config_spec
        assert bound.host_function is not None
        with bound.env:
            pool = spec.compiler_seed_configs
            default = spec.default_config()
            spec.cute_serial_lane_schedule_enabled = False
            old = compiler_seed_configs(bound.env, bound.host_function.device_ir)
            spec.cute_serial_lane_schedule_enabled = True
            assert [c for c in pool if KEY not in c.config] == old
            assert spec.default_config() == default
            generation = ConfigGeneration(spec)
            prefix = [
                generation.unflatten(v) for v in generation.random_population_flat(100)
            ]
        enabled = [c for c in prefix if KEY in c.config]
        assert {c.config[KEY] for c in enabled} == {"step_major", "step_major_vector"}
        for raw in enabled[:4]:
            normalized = bound._normalized_config_copy(raw)
            assert bound.to_code(raw) == bound.to_code(normalized)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "mode,stride",
    [("step_major", 1), ("step_major_vector", 1), ("step_major_vector", 2)],
)
@pytest.mark.parametrize("width", [2, 4])
def test_actual_generated_device_math_coverage_and_scalar_transfer(
    dtype, mode, stride, width
):
    from ._serial_lane_model import run

    args = _args(dtype, stride=stride)
    generator = torch.Generator().manual_seed(319)
    args[0].copy_(torch.randn(args[0].shape, generator=generator))
    args[1].copy_(-torch.rand(args[1].shape, generator=generator))
    source = _source(args, mode, width)
    original, old_counts = run(source, args, reordered=False)
    actual, counts = run(source, args, reordered=True)
    assert torch.equal(actual, original)
    assert counts["state_reads"] == old_counts["state_reads"] == args[0].numel()
    assert counts["coefficient_reads"] * width == old_counts["coefficient_reads"]
    assert (
        bool(counts["vector_load"])
        == bool(counts["vector_store"])
        == (mode == "step_major_vector" and stride == 1)
    )


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_actual_fx_dependency_precision_and_store_order_rejections(kind):
    with _cpu_codegen():
        bound = _negative._bind_isolated((*_args(), kind))
        assert not bound.config_spec.cute_serial_lane_schedule_enabled
        with pytest.raises(exc.InvalidConfig, match="serial"):
            bound.to_code(_config())


@pytest.mark.parametrize(
    "config",
    [
        {"cute_vector_widths": [1, 1, 1]},
        {"cute_vector_widths": [1, 8, 1]},
        {"cute_cluster_n": 2},
        {"cute_lane_layouts": ["blocked", "striped", "blocked"]},
    ],
)
def test_unsupported_late_ownership_rejects(config):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        values = {**_config().config, **config}
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(helion.Config.from_dict(values))


def test_original_entry_metadata_cannot_be_laundered_through_view():
    args = _args()
    source = _source(args, kernel=_derived_view)
    calls = []
    module = _module(source, calls)
    with patch.object(serial_lane_guard, "_require_cuda"):
        module._derived_view(*args)
        changed = args[0].transpose(0, 1).contiguous().transpose(0, 1)
        assert changed.shape == args[0].shape and changed.stride() != args[0].stride()
        with pytest.raises(exc.BackendUnsupported, match="stride"):
            module._derived_view(changed, *args[1:])
    assert len(calls) == 1


@pytest.mark.parametrize("route", ["full", "break", "export"])
@pytest.mark.parametrize("mode", [None, "lane_major", "step_major_vector"])
def test_actual_kernel_hop_boundary_and_default_controls(route, mode):
    from torch._inductor.codecache import PyCodeCache

    args = _args()
    calls = []
    kernel = helion.kernel(
        _recurrence.fn,
        backend="cute",
        static_shapes=True,
        configs=[_config(mode)],
        autotune_effort="none",
    )

    class Wrapper(torch.nn.Module):
        def forward(self, states, coefficient, out):
            return kernel(states, coefficient, out)

    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache,
            "load",
            side_effect=lambda source, **kwargs: _module(source, calls),
        ),
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        bound = kernel.bind(args)
        bound.set_config(_config(mode))

        def invoke():
            if route == "export":
                return torch.export.export(Wrapper(), args, strict=True)
            fn = torch.compile(Wrapper(), backend="eager", fullgraph=route == "full")
            return fn(*args)

        if mode == "step_major_vector":
            with pytest.raises(
                (
                    torch._dynamo.exc.InternalTorchDynamoError,
                    exc.BackendUnsupported,
                    torch._dynamo.exc.Unsupported,
                ),
                match="current-call host contract",
            ):
                invoke()
            assert calls == []
        else:
            invoke()
            assert len(calls) == (route != "export")


@pytest.mark.parametrize(
    "mode", [None, "lane_major", "step_major", "step_major_vector"]
)
def test_actual_inductor_render_boundary_before_generate_ast(mode):

    from helion._compiler._inductor import template_buffer as tb

    class Stop(BaseException):
        pass

    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        renderer = object.__new__(tb.HelionTemplateBuffer)
        renderer._bound_kernel = bound
        renderer._fusion_metadata = tb._FusionMetadata(
            epilogue_idx_by_param={},
            epilogue_keep_store=set(),
            prologue_vars={},
            prologue_fused_params=set(),
            prologue_has_source=set(),
        )
        renderer._extra_params = []
        with patch.object(tb, "generate_ast", side_effect=Stop) as generate:
            if mode in (None, "lane_major"):
                with pytest.raises(Stop):
                    renderer._build_and_unparse(_config(mode))
                assert generate.call_count == 1
            else:
                with pytest.raises(
                    exc.BackendUnsupported, match="current-call host contract"
                ):
                    renderer._build_and_unparse(_config(mode))
                generate.assert_not_called()


def test_genuine_kernel_prepared_reuse_checks_every_current_call():
    from torch._inductor.codecache import PyCodeCache

    args = _args()
    calls = []
    kernel = helion.kernel(
        _recurrence.fn,
        backend="cute",
        static_shapes=True,
        configs=[_config()],
        autotune_effort="none",
    )
    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache,
            "load",
            side_effect=lambda source, **kwargs: _module(source, calls),
        ),
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        kernel(*args)
        assert kernel._prepared_call is not None
        prepared = kernel._prepared_call
        assert prepared.matches(kernel, args)
        kernel(*args)
        assert kernel._prepared_call is prepared
        args[0].set_(
            torch.empty(args[0].numel() * 2).untyped_storage(),
            0,
            args[0].shape,
            (896, 128, 2),
        )
        # The raw callable used by the actual prepared fast path cannot
        # launder new metadata, even if invoked independently of matches().
        with pytest.raises(exc.BackendUnsupported, match="stride"):
            assert prepared.bound._run is not None
            prepared.bound._run(*args)
    assert len(calls) == 2


class _ForeignProvider(ModuleType):
    value: float = 0.0


_foreign_provider = _ForeignProvider("torch")
_torch_alias = torch
_language_alias = hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _offset_initializer(states, coefficient, out):
    initial = hl.specialize(states.storage_offset())
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], float(initial), dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi].float()) * carry
                + states[bi.begin, qi, fi].float()
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _foreign_initializer(states, coefficient, out):
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], float(_foreign_provider.value), dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi].float()) * carry
                + states[bi.begin, qi, fi].float()
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _generator_initializer(states, coefficient, out):
    # Preserve the exact generator regression, not a comprehension or next().
    initial = list(_foreign_provider.value for _i in range(1))[0]  # noqa: C400, RUF015
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], float(initial), dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi].float()) * carry
                + states[bi.begin, qi, fi].float()
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _metadata_positive(states, coefficient, out):
    values = [states.size(0), states.stride(0)]
    pair = (values[0], values[1])
    initial = float((pair[0] + values[1]) % 2)
    batch, steps, features = states.shape
    for bi, fi in _language_alias.tile([batch, features], block_size=[1, None]):
        carry = _language_alias.full([fi], initial, dtype=_torch_alias.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            carry = (
                torch.exp(coefficient[bi.begin, qi].float()) * carry
                + states[bi.begin, qi, fi].float()
            )
    return out


@pytest.mark.parametrize(
    "kernel, reason",
    [
        (_offset_initializer, "host metadata"),
        (_foreign_initializer, "global provider"),
        (_generator_initializer, "nested Python"),
    ],
)
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("route", ["raw", "isolated", "managed", "prepared"])
def test_original_metadata_regressions_reject_before_load(
    kernel, reason, offset, route
):
    from torch._inductor.codecache import PyCodeCache

    args = _args(offset=offset)
    requested = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        configs=[_config()],
        autotune_effort="none",
    )
    with (
        _cpu_codegen(),
        patch.object(PyCodeCache, "load") as load,
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        with pytest.raises(exc.BackendUnsupported, match=reason):
            if route == "prepared":
                requested(*args)
            else:
                bound = (
                    requested.bind(args)
                    if route == "managed"
                    else requested._bind_isolated(args)
                )
                if route == "raw":
                    bound.compile_config(_config(), allow_print=False)
                else:
                    bound.set_config(_config())
                    bound(*args)
        load.assert_not_called()
        assert requested._prepared_call is None


@pytest.mark.parametrize(
    "kernel", [_offset_initializer, _foreign_initializer, _generator_initializer]
)
def test_metadata_rejection_does_not_change_default_source(kernel):
    assert _source(kernel=kernel, mode=None) == _source(
        kernel=kernel, mode="lane_major"
    )


@pytest.mark.parametrize("offset", [False, True])
def test_trusted_aliases_literal_containers_and_shape_metadata(offset):
    args = _args(offset=offset)
    source = _source(args, kernel=_metadata_positive)
    calls = []
    raw = _module(source, calls)._metadata_positive
    with patch.object(serial_lane_guard, "_require_cuda"):
        raw(*args)
    assert len(calls) == 1 and calls[0][0].__name__.endswith("_serial_lane")


def test_raw_and_prepared_unspecialized_offset_rebinding_remains_valid():
    from torch._inductor.codecache import PyCodeCache

    calls = []
    args = _args()
    kernel = helion.kernel(
        _recurrence.fn,
        backend="cute",
        static_shapes=True,
        configs=[_config()],
        autotune_effort="none",
    )
    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache, "load", side_effect=lambda source, **kw: _module(source, calls)
        ),
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        kernel(*args)
        prepared = kernel._prepared_call
        assert prepared is not None and prepared.bound._run is not None
        prepared.bound._run(*_args(offset=True))
        backing = torch.empty(args[0].numel() + 4)
        args[0].set_(backing.untyped_storage(), 4, args[0].shape, args[0].stride())
        prepared.bound._run(*args)
    assert len(calls) == 3 and all(
        row[0].__name__.endswith("_serial_lane") for row in calls
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_trusted_flag(states, coefficient, out):
    initial = float(torch.backends.cudnn.enabled)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_module_scalar(states, coefficient, out):
    # The fixture deliberately installs a noncanonical torch attribute.
    # pyrefly: ignore [missing-attribute]
    initial = float(torch._serial_review_scalar)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_module_container(states, coefficient, out):
    # pyrefly: ignore [missing-attribute]
    initial = float(torch._serial_review_values[0])
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_module_dtype(states, coefficient, out):
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        # pyrefly: ignore [missing-attribute]
        carry = hl.full([fi], initial, dtype=torch._serial_review_dtype)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_module_callable(states, coefficient, out):
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            # pyrefly: ignore [missing-attribute]
            factor = torch._serial_review_exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_local_dtype_alias(states, coefficient, out):
    # pyrefly: ignore [missing-attribute]
    dtype = torch._serial_review_dtype
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=dtype)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_local_callable_alias(states, coefficient, out):
    # pyrefly: ignore [missing-attribute]
    exp = torch._serial_review_exp
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_nested_dtype(states, coefficient, out):
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        # pyrefly: ignore [missing-attribute]
        carry = hl.full([fi], initial, dtype=torch._serial_review_namespace.dtype)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_nested_callable(states, coefficient, out):
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            # pyrefly: ignore [missing-attribute]
            factor = torch._serial_review_namespace.exp(
                coefficient[bi.begin, qi].float()
            )
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_dtype_property(states, coefficient, out):
    initial = float(torch.float32.is_floating_point)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_local_dtype_property(states, coefficient, out):
    dtype = torch.float32
    initial = float(dtype.is_floating_point)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_callable_property(states, coefficient, out):
    initial = float(len(torch.exp.__name__))
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_canonical_alias(states, coefficient, out):
    t = torch
    language = hl
    dtype = t.float32
    exp = t.exp
    initial = 0.0
    batch, steps, features = states.shape
    for bi, fi in language.tile([batch, features], block_size=[1, None]):
        carry = language.full([fi], initial, dtype=dtype)
        for qi in language.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_unbound_metadata(states, coefficient, out):
    initial = float((torch.Tensor.size(states, 0) + torch.Tensor.stride(states, 0)) % 2)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _attribute_shape_containers(states, coefficient, out):
    values = [states.size(0), states.stride(0)]
    pair = (values[0], values[1])
    initial = float((pair[0] + pair[1]) % 2)
    batch, steps, features = states.shape
    for bi, fi in hl.tile([batch, features], block_size=[1, None]):
        carry = hl.full([fi], initial, dtype=torch.float32)
        for qi in hl.grid(steps):
            out[bi.begin, qi, fi] = carry.to(out.dtype)
            factor = torch.exp(coefficient[bi.begin, qi].float())
            carry = factor * carry + states[bi.begin, qi, fi].float()
    return out


_ATTRIBUTE_NEGATIVES = {
    "_attribute_trusted_flag": _attribute_trusted_flag,
    "_attribute_module_scalar": _attribute_module_scalar,
    "_attribute_module_container": _attribute_module_container,
    "_attribute_module_dtype": _attribute_module_dtype,
    "_attribute_module_callable": _attribute_module_callable,
    "_attribute_local_dtype_alias": _attribute_local_dtype_alias,
    "_attribute_local_callable_alias": _attribute_local_callable_alias,
    "_attribute_nested_dtype": _attribute_nested_dtype,
    "_attribute_nested_callable": _attribute_nested_callable,
    "_attribute_dtype_property": _attribute_dtype_property,
    "_attribute_local_dtype_property": _attribute_local_dtype_property,
    "_attribute_callable_property": _attribute_callable_property,
}
_ATTRIBUTE_POSITIVES = {
    "_attribute_canonical_alias": _attribute_canonical_alias,
    "_attribute_unbound_metadata": _attribute_unbound_metadata,
    "_attribute_shape_containers": _attribute_shape_containers,
}


@pytest.fixture
def original_attribute_aliases():
    initialized = torch.cuda.is_initialized()
    namespace = ModuleType("serial_review_namespace")
    namespace.__dict__.update(dtype=torch.float32, exp=torch.exp)
    with (
        torch.backends.cudnn.flags(enabled=True),
        patch.object(torch, "_serial_review_scalar", 0.0, create=True),
        patch.object(torch, "_serial_review_values", [0.0], create=True),
        patch.object(torch, "_serial_review_dtype", torch.float32, create=True),
        patch.object(torch, "_serial_review_exp", torch.exp, create=True),
        patch.object(torch, "_serial_review_namespace", namespace, create=True),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only")),
    ):
        yield
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.usefixtures("original_attribute_aliases")
@pytest.mark.parametrize("kernel", _ATTRIBUTE_NEGATIVES.values())
@pytest.mark.parametrize("offset", [False, True])
def test_original_attribute_negatives_default_control(kernel, offset):
    args = _args(offset=offset)
    with pytest.raises(exc.BackendUnsupported, match="attribute provenance"):
        _source(args, kernel=kernel)
    assert _source(args, kernel=kernel, mode=None) == _source(
        args, kernel=kernel, mode="lane_major"
    )


@pytest.mark.usefixtures("original_attribute_aliases")
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
def test_original_flag_raw_normalized_reject_before_load(flag, canonical):
    from torch._inductor.codecache import PyCodeCache

    with (
        torch.backends.cudnn.flags(enabled=flag),
        _cpu_codegen(),
        patch.object(PyCodeCache, "load") as load,
    ):
        bound = _attribute_trusted_flag._bind_isolated(_args())
        config = _config()
        if canonical:
            config = bound._normalized_config_copy(config)
        with pytest.raises(exc.BackendUnsupported, match="attribute provenance"):
            bound.compile_config(config, allow_print=False)
        load.assert_not_called()


@pytest.mark.usefixtures("original_attribute_aliases")
@pytest.mark.parametrize("route", ["isolated", "managed", "prepared"])
def test_original_flag_call_routes_reject_before_load(route):
    from torch._inductor.codecache import PyCodeCache

    args = _args()
    requested = helion.kernel(
        _attribute_trusted_flag.fn,
        backend="cute",
        static_shapes=True,
        configs=[_config()],
        autotune_effort="none",
    )
    with _cpu_codegen(), patch.object(PyCodeCache, "load") as load:
        with pytest.raises(exc.BackendUnsupported, match="attribute provenance"):
            if route == "prepared":
                requested(*args)
            else:
                bound = (
                    requested.bind(args)
                    if route == "managed"
                    else requested._bind_isolated(args)
                )
                bound.set_config(_config())
                bound(*args)
        load.assert_not_called()
        assert requested._prepared_call is None


@pytest.mark.parametrize("kernel", _ATTRIBUTE_POSITIVES.values())
@pytest.mark.parametrize("offset", [False, True])
def test_original_attribute_aliases_admit_current_offset(kernel, offset):
    args = _args(offset=offset)
    source = _source(args, kernel=kernel)
    calls = []
    module = _module(source, calls)
    module.__dict__["hl"] = hl  # Restore the exact emitted local-provider import.
    with patch.object(serial_lane_guard, "_require_cuda"):
        getattr(module, kernel.fn.__name__)(*args)
    assert len(calls) == 1 and calls[0][0].__name__.endswith("_serial_lane")


def test_original_attribute_name_requires_exact_identity():
    from helion._compiler.cute import serial_lane_recurrence as serial

    with (
        patch.dict(serial._ORIGINAL_TORCH_EXPORTS, {"exp": torch.neg}),
        pytest.raises(exc.BackendUnsupported, match="noncanonical module export"),
    ):
        _source(kernel=_recurrence)


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("target", ["device", "fixture"])
def test_cpu_checks_preserve_existing_cuda_initialization(initialized, target):
    with (
        patch.object(torch.cuda, "is_initialized", return_value=initialized),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only")),
    ):
        if target == "device":
            test_runtime_device_check_not_weakened()
        else:
            fixture = original_attribute_aliases._get_wrapped_function()()
            next(fixture)
            with pytest.raises(StopIteration):
                next(fixture)
        assert torch.cuda.is_initialized() == initialized
