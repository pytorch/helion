from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from .test_cute_serial_coarsen import _recurrence as coarsened
from .test_cute_serial_coarsen import args as coarsen_args
from .test_cute_serial_coarsen import config as coarsen_config
from .test_cute_serial_lane import _args
from .test_cute_serial_lane import _config
from .test_cute_serial_lane import _module
from .test_cute_serial_lane import _recurrence
import helion
from helion import exc
from helion._compiler.cute.host_fastpath import KEY
from helion._compiler.cute.host_fastpath_proof import Proof
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute import host_fastpath as guard
from helion.runtime.cute import serial_lane_guard

pytestmark = skipUnlessBackends(["cute"])


# A basic two-contraction fixture, deliberately independent of the unmerged
# initialized-accumulator/late-RHS/K64 compiler options.
@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pair(a, b, c, d, scale, mode: hl.constexpr):
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n]):
        kk = hl.arange(k)
        left = (a[row, kk].float() * 1.01).to(a.dtype)
        first = hl.dot(left, b[kk, col])
        seed = first * torch.exp(scale[row])[:, None]
        right = (d[kk, col].float() * scale[kk][:, None]).to(a.dtype)
        second = hl.dot(c[row, kk], right)
        out[row, col] = (seed + second).to(a.dtype)
    return out


def chain_args(dtype=torch.bfloat16, n=64):
    values = [
        torch.empty(shape, dtype=dtype)
        for shape in ((128, 128), (128, n), (128, 128), (128, n))
    ]
    return (*values, torch.empty(128, dtype=torch.float32), "plain")


def chain_config(n=64, **extra):
    return helion.Config.from_dict(
        {
            "block_sizes": [128, n],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        }
        | extra
    )


_cpu = _cpu_codegen


def source(family, enabled=True):
    if family == "serial":
        fn, args, config, context = _recurrence, _args(), _config(), _cpu_codegen
    elif family == "coarsen":
        fn, args, config, context = (
            coarsened,
            coarsen_args(),
            coarsen_config(),
            _cpu_codegen,
        )
    else:
        fn, args, config, context = (
            _pair,
            chain_args(),
            chain_config(cute_chained_pointwise_vectorize=True),
            _cpu,
        )
    if enabled is not None:
        config.config[KEY] = enabled
    with context():
        bound = fn._bind_isolated(args)
        canonical = bound._normalized_config_copy(config)
        text = bound.to_code(config)
        assert text == bound.to_code(canonical)
        return text, args, fn.fn.__name__


@pytest.mark.parametrize("family", ["serial", "coarsen", "chained"])
def test_source_default_and_original_device_identity(family):
    old, _, _ = source(family, None)
    disabled, _, _ = source(family, False)
    assert old == disabled
    new, _, _ = source(family)
    old_nodes = {
        n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)
    }
    new_nodes = {
        n.name: n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)
    }
    for name, node in old_nodes.items():
        if any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list):
            assert ast.dump(node) == ast.dump(new_nodes[name])
    fast = next(n for n in new_nodes.values() if n.name.endswith("_host_fast"))
    original = new_nodes[fast.name.removesuffix("_host_fast")]

    def guards(node):
        return sum(
            isinstance(n, ast.If) and ".toint()" in ast.unparse(n.test)
            for n in ast.walk(node)
        )

    assert guards(fast) < guards(original)
    assert "host_fastpath" in new and "validate_trace()" in new


@pytest.mark.parametrize("family", ["serial", "coarsen"])
def test_real_host_current_offset_and_alias_grid(family):
    text, args, name = source(family)
    assert isinstance(args[0], torch.Tensor) and isinstance(args[2], torch.Tensor)
    launches = []
    module = _module(text, launches)
    fn = module.__dict__[name]
    with (
        patch.object(serial_lane_guard, "_require_cuda"),
        patch.object(guard, "_is_cuda", return_value=True),
    ):
        fn(*args)
        assert launches[-1][0].__name__.endswith("_host_fast")
        fast_grid = launches[-1][1]
        alias = (
            args[0]
            .view(torch.bfloat16)
            .flatten()[: args[2].numel()]
            .view(args[2].shape)
        )
        fn(args[0], args[1], alias)
        assert not launches[-1][0].__name__.endswith("_host_fast")
        if family == "coarsen":
            assert launches[-1][1][0] == 2 * fast_grid[0]
        fn(*args)
        assert launches[-1][0].__name__.endswith("_host_fast")


@pytest.mark.parametrize("value", [0, 1, "true", None, 1.0, [], {}])
@pytest.mark.parametrize("repair", [False, True])
def test_bad_config_never_repaired(value, repair):
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(_args())
        config = _config().config | {KEY: value}
        with pytest.raises(exc.InvalidConfig, match="boolean"):
            bound.config_spec.normalize(config, _fix_invalid=repair)


def test_unknown_and_overflow_proof_retains_branches():
    contract = ((128, 64), (64, 1), "torch.float32", 4)
    text = """i = cutlass.Int32(cute.arch.thread_idx()[0])
p = x.iterator + i * 4
if (p.toint() % 16 == 0) & (x.layout.stride[1] == 1):
    cute.copy(a, b, c)
else:
    fallback()
"""
    for multiplier, removed in [(4, 1), (1, 0), (2**30, 0)]:
        body = ast.parse(text.replace("i * 4", f"i * {multiplier}")).body
        proof = Proof({"x": contract}, (1, 1, 1), (128, 1, 1))
        proof.block_body(body, {})
        assert proof.removed == removed


def test_loop_carried_address_cannot_launder_alignment():
    body = ast.parse("""p = x.iterator
for i in range(4):
    if p.toint() % 16 == 0:
        cute.copy(a, b, c)
    else:
        fallback()
    p = p + 1
""").body
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (128, 1, 1))
    proof.block_body(body, {})
    assert proof.removed == 0


@pytest.mark.parametrize(
    "kind",
    [
        "aligned_offset",
        "unaligned_offset",
        "stride",
        "alias",
        "negative",
        "gradient",
        "cpu",
        "foreign_selection",
        "grid",
    ],
)
def test_current_span_selection_and_exact_fallback(kind):
    values = (torch.empty(32), torch.empty(32))
    contracts = tuple(
        (tuple(t.shape), tuple(t.stride()), str(t.dtype), t.element_size())
        for t in values
    )
    original, fast, foreign = object(), object(), object()
    grid, selected = (1,), original
    cuda = kind != "cpu"
    if kind in ("aligned_offset", "unaligned_offset"):
        offset = 4 if kind == "aligned_offset" else 1
        values = (torch.empty(32 + offset)[offset:], values[1])
    if kind == "stride":
        values = (torch.empty(64)[::2], values[1])
    if kind == "alias":
        values = (values[0], values[0])
    if kind == "negative":
        values = (torch._neg_view(values[0]), values[1])
    if kind == "gradient":
        values[0].requires_grad_(True)
    if kind == "foreign_selection":
        selected = foreign
    if kind == "grid":
        grid = (2,)
    with patch.object(guard, "_is_cuda", return_value=cuda):
        result = guard.select_kernel(
            selected,
            original,
            fast,
            values,
            contracts,
            grid,
            (1, 1, 1),
            (32, 1, 1),
            (32, 1, 1),
        )
    assert result is (fast if kind == "aligned_offset" else selected)


def test_genuine_chained_raw_callable_checks_each_current_launch():
    from torch._inductor.codecache import PyCodeCache

    args = chain_args()
    assert isinstance(args[0], torch.Tensor)
    config = chain_config(cute_chained_pointwise_vectorize=True)
    config.config[KEY] = True
    calls = []
    with (
        _cpu(),
        patch.object(
            PyCodeCache, "load", side_effect=lambda text, **kw: _module(text, calls)
        ),
        patch.object(guard, "_is_cuda", return_value=True),
    ):
        bound = _pair._bind_isolated(args)
        fn = bound.compile_config(config, allow_print=False)
        out0 = fn(*args)
        assert calls[-1][0].__name__.endswith("_host_fast")
        # Fresh aligned backing is admitted by the same cached raw callable.
        base = torch.empty(args[0].numel() + 8, dtype=args[0].dtype)
        aligned = base[8:].view(args[0].shape)
        out1 = fn(aligned, *args[1:])
        assert calls[-1][0].__name__.endswith("_host_fast")
        assert out0.untyped_storage().data_ptr() != out1.untyped_storage().data_ptr()
        bad = torch.empty(args[0].numel() + 1, dtype=args[0].dtype)[1:].view(
            args[0].shape
        )
        fn(bad, *args[1:])
        assert not calls[-1][0].__name__.endswith("_host_fast")
        changed = torch.empty((128, 256), dtype=args[0].dtype)[:, ::2]
        fn(changed, *args[1:])
        assert not calls[-1][0].__name__.endswith("_host_fast")
        fn(*args)
        assert calls[-1][0].__name__.endswith("_host_fast")


def test_real_prepared_host_preserves_alias_fallback():
    from torch._inductor.codecache import PyCodeCache

    import helion

    args = _args()
    config = _config()
    config.config[KEY] = True
    kernel = helion.kernel(
        _recurrence.fn,
        backend="cute",
        static_shapes=True,
        configs=[config],
        autotune_effort="none",
    )
    calls = []
    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache, "load", side_effect=lambda text, **kw: _module(text, calls)
        ),
        patch.object(guard, "_is_cuda", return_value=True),
        patch.object(serial_lane_guard, "_require_cuda"),
    ):
        kernel(*args)
        prepared = kernel._prepared_call
        assert prepared is not None and prepared.bound._run is not None
        kernel(*args)
        assert kernel._prepared_call is prepared
        assert calls[-1][0].__name__.endswith("_host_fast")
        alias = (
            args[0]
            .view(torch.bfloat16)
            .flatten()[: args[2].numel()]
            .view(args[2].shape)
        )
        prepared.bound._run(args[0], args[1], alias)
        assert not calls[-1][0].__name__.endswith("_host_fast")
        prepared.bound._run(*args)
        assert calls[-1][0].__name__.endswith("_host_fast")


@pytest.mark.parametrize("route", ["full", "break", "export"])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_kernel_hop_rejection_and_default_control(route, enabled):
    from torch._inductor.codecache import PyCodeCache

    import helion

    args = _args()
    config = _config(None)
    config.config[KEY] = enabled
    kernel = helion.kernel(
        _recurrence.fn,
        backend="cute",
        static_shapes=True,
        configs=[config],
        autotune_effort="none",
    )
    calls = []

    class Wrapper(torch.nn.Module):
        def forward(self, a, b, c):
            return kernel(a, b, c)

    with (
        _cpu_codegen(),
        patch.object(
            PyCodeCache, "load", side_effect=lambda text, **kw: _module(text, calls)
        ),
    ):

        def invoke():
            if route == "export":
                return torch.export.export(Wrapper(), args, strict=True)
            return torch.compile(Wrapper(), backend="eager", fullgraph=route == "full")(
                *args
            )

        if enabled:
            with pytest.raises(
                (
                    exc.BackendUnsupported,
                    torch._dynamo.exc.InternalTorchDynamoError,
                    torch._dynamo.exc.Unsupported,
                ),
                match="host-selected fastpath",
            ):
                invoke()
            assert not calls
        else:
            invoke()
            assert len(calls) == (route != "export")


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_direct_inductor_render_gate(enabled):
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
        config = _config(None)
        config.config[KEY] = enabled
        with patch.object(tb, "generate_ast", side_effect=Stop) as generate:
            with pytest.raises(exc.BackendUnsupported if enabled else Stop):
                renderer._build_and_unparse(config)
            assert generate.call_count == (not enabled)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_stride",
        "pointer_reassign",
        "unknown_branch",
        "import_shadow",
        "negative_math",
    ],
)
def test_proof_negatives(kind):
    contract = ((128,), (1,), "torch.float32", 4)
    prefix = "p = x.iterator\n"
    if kind == "pointer_reassign":
        prefix += "p = unknown()\n"
    if kind == "unknown_branch":
        prefix += "if opaque():\n    p = unknown()\n"
    if kind == "import_shadow":
        prefix += "import foreign as p\n"
    if kind == "negative_math":
        prefix = "p = x.iterator + (-1)\n"
    condition = "p.toint() % 16 == 0"
    if kind == "unknown_stride":
        condition += " and x.layout.stride[0] == 2"
    body = ast.parse(
        prefix + f"if {condition}:\n    cute.copy(a,b,c)\nelse:\n    scalar()\n"
    ).body
    proof = Proof({"x": contract}, (1, 1, 1), (32, 1, 1))
    proof.block_body(body, {})
    assert proof.removed == 0
