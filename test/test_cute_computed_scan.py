from __future__ import annotations

import ast
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _cpu_context() -> Iterator[None]:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        yield


def _computed(
    x: torch.Tensor,
    coefficient: torch.Tensor,
    out: torch.Tensor,
    tile: hl.constexpr,
    reverse: hl.constexpr,
    mode: hl.constexpr,
    channels: hl.constexpr,
) -> torch.Tensor:
    for batch, channel, row in hl.tile(
        [x.size(0), x.size(2), x.size(1)], block_size=[1, channels, tile]
    ):
        value = x[batch, row, channel].float()
        if mode == "clamp":
            value = value.clamp(min=0.0) * coefficient[channel]
        elif mode == "arithmetic":
            value = ((value + 2.0) * coefficient[channel] - 1.0).clamp(max=5.0)
        elif mode == "roundtrip":
            value = (value * coefficient[channel]).to(torch.bfloat16).float()
        else:
            value = value * coefficient[channel]
        out[batch, row, channel] = hl.cumsum(
            value,
            dim=1,
            # pyrefly: ignore [bad-argument-type]
            reverse=reverse,
        )
    return out


def _code(
    *,
    dtype: torch.dtype = torch.bfloat16,
    tile: int = 128,
    length: int = 128,
    reverse: bool = False,
    mode: str = "clamp",
    channels: int = 1,
    alias: str | None = None,
    strided: bool = False,
) -> str:
    with _cpu_context():
        x = torch.empty((2, length, 3), dtype=dtype)
        out = torch.empty(x.shape, dtype=torch.float32)
        coefficient = torch.empty(3)
        if strided:
            x = torch.empty((2, 3, length), dtype=dtype).transpose(1, 2)
            out = torch.empty((2, 3, length)).transpose(1, 2)
        if alias == "x":
            assert dtype == torch.float32
            out = x.view_as(x)
        elif alias == "coefficient":
            coefficient = out.flatten()[:3]
        kernel = helion.kernel(_computed, backend="cute", static_shapes=True)
        args = (x, coefficient, out, tile, reverse, mode, channels)
        bound = kernel.bind(args)
        return bound.to_code(bound.env.config_spec.default_config())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["clamp", "arithmetic", "multiply"])
@pytest.mark.parametrize("reverse", [False, True])
def test_computed_scan_codegen(dtype: torch.dtype, mode: str, reverse: bool) -> None:
    code = _code(dtype=dtype, mode=mode, reverse=reverse)
    assert "scan_warp_prefix" in code and "scan_expr_value" in code
    assert "scan_initialized" not in code
    assert ("shuffle_sync_down" if reverse else "shuffle_sync_up") in code
    if mode == "clamp":
        assert "cute.math.max(" in code
    if mode == "arithmetic":
        assert "cute.math.min(" in code


@pytest.mark.parametrize("alias", ["x", "coefficient"])
def test_computed_scan_all_leaf_alias_rejected(alias: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="ownership or alias"):
        _code(dtype=torch.float32, alias=alias)


@pytest.mark.parametrize("mode", ["roundtrip"])
def test_computed_scan_narrowing_stays_unsupported(mode: str) -> None:
    with pytest.raises(exc.BackendUnsupported):
        _code(mode=mode)


@pytest.mark.parametrize("tile,channels", [(2048, 1), (128, 2)])
def test_computed_scan_unproved_physical_layout_rejected(
    tile: int, channels: int
) -> None:
    with pytest.raises(exc.BackendUnsupported, match="computed add scan"):
        _code(tile=tile, length=2048 if tile == 2048 else 128, channels=channels)


@pytest.mark.parametrize("tile,length", [(32, 53), (64, 53), (128, 191), (256, 511)])
@pytest.mark.parametrize("strided", [False, True])
def test_computed_scan_tail_and_strided_codegen(
    tile: int, length: int, strided: bool
) -> None:
    code = _code(tile=tile, length=length, strided=strided, reverse=True)
    assert "scan_warp_prefix" in code
    assert f"scan_row < cutlass.Int32({length})" in code
    assert "block=(" + str(tile) + ", 1, 1)" in code
    assert "tile_offset_2" in code
    tree = ast.parse(code)
    # Identity masks must enclose the final expression, not just its leaves.
    values = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "scan_value"
            for target in node.targets
        )
    ]
    assert len(values) == (1 if tile == 32 else 2)
    for value in values:
        assert isinstance(value.value, ast.IfExp)
        assert ast.unparse(value.value.orelse) == "cutlass.Float32(0)"


def _two_outputs(
    x: torch.Tensor,
    y: torch.Tensor,
    coefficient: torch.Tensor,
    intermediate: torch.Tensor,
    out: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    for batch, channel, row in hl.tile(
        [x.size(0), x.size(2), x.size(1)], block_size=[1, 1, 128]
    ):
        value = x[batch, row, channel].float().clamp(min=0.0)
        intermediate[batch, row, channel] = value
        if mode == "two":
            value = (value + y[batch, row, channel].float()) * coefficient[channel]
        elif mode == "scalar":
            value = value * coefficient[0] + 1.0
        elif mode == "gather":
            value = value * y[batch, row.index * 2, channel].float()
        else:
            value = value * coefficient[channel]
        out[batch, row, channel] = hl.cumsum(value, dim=1)
    return out


@pytest.mark.parametrize("mode", ["two", "scalar", "uniform"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_computed_scan_real_multi_leaf_and_two_writes(
    mode: str, static_shapes: bool
) -> None:
    with _cpu_context():
        x = torch.empty(2, 191, 3, dtype=torch.bfloat16)
        # Read-only leaves may alias one another.
        args = (
            x,
            x.view_as(x),
            torch.empty(3),
            torch.empty_like(x, dtype=torch.float32),
            torch.empty_like(x, dtype=torch.float32),
            mode,
        )
        kernel = helion.kernel(
            _two_outputs, backend="cute", static_shapes=static_shapes
        )
        bound = kernel.bind(args)
        if not static_shapes and mode != "scalar":
            # Dynamic broadcast inserts an explicit _cute_layout_change.
            # This initial typed-DAG extension deliberately does not erase it.
            with pytest.raises(exc.BackendUnsupported, match="input load"):
                bound.to_code(bound.config_spec.default_config())
            return
        code = bound.to_code(bound.config_spec.default_config())
        assert "scan_warp_prefix" in code
        assert "intermediate" in code


@pytest.mark.parametrize("leaf", ["x", "y", "coefficient"])
@pytest.mark.parametrize("output", ["intermediate", "out"])
def test_computed_scan_every_leaf_every_write_cache_guard(
    leaf: str, output: str
) -> None:
    with _cpu_context():
        args = [
            torch.empty(2, 128, 3),
            torch.empty(2, 128, 3),
            torch.empty(3),
            torch.empty(2, 128, 3),
            torch.empty(2, 128, 3),
            "two",
        ]
        kernel = helion.kernel(_two_outputs, backend="cute", static_shapes=True)
        bound = kernel.bind(tuple(args))
        config = bound.config_spec.default_config()
        assert "scan_warp_prefix" in bound.to_code(config)
        destination = args[3 if output == "intermediate" else 4]
        assert isinstance(destination, torch.Tensor)
        args[{"x": 0, "y": 1, "coefficient": 2}[leaf]] = (
            destination.flatten()[:3]
            if leaf == "coefficient"
            else destination.view_as(destination)
        )
        aliased = kernel.bind(tuple(args))
        assert aliased is not bound
        with pytest.raises(exc.BackendUnsupported, match="ownership or alias"):
            aliased.to_code(config)


def test_computed_scan_gather_rejected() -> None:
    with _cpu_context():
        args = (
            torch.empty(2, 128, 3),
            torch.empty(2, 256, 3),
            torch.empty(3),
            torch.empty(2, 128, 3),
            torch.empty(2, 128, 3),
            "gather",
        )
        bound = helion.kernel(_two_outputs, backend="cute", static_shapes=True).bind(
            args
        )
        with pytest.raises(exc.BackendUnsupported, match="ownership or alias"):
            bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_computed_scan_unsupported_leaf_dtype_rejected(dtype: torch.dtype) -> None:
    with pytest.raises(exc.BackendUnsupported):
        _code(dtype=dtype)


@pytest.mark.parametrize("mode", ["clamp", "arithmetic"])
@pytest.mark.parametrize("coefficient", [2.125, float("inf"), float("nan")])
def test_actual_emitted_value_cast_order_and_tail_identity(
    mode: str, coefficient: float
) -> None:
    code = _code(mode=mode, length=53, tile=64)
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion")
    )
    statements: list[ast.stmt] = []
    for node in device.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "scan_expr_load":
                statements = [node]
                continue
            if statements:
                statements.append(node)
            if node.targets[0].id == "scan_value":
                break
    assert statements
    expression = compile(
        ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
        "<emitted-scan-value>",
        "exec",
    )
    # Execute only the real emitted scalar expression on CPU; no CuTe import,
    # device compilation or simulated warp communication is involved.
    x = torch.arange(2 * 53 * 3).view(2, 53, 3).to(torch.bfloat16) / 7
    coeff = torch.tensor([1.0, -1.0, coefficient])
    namespace = {
        "x": x,
        "coefficient": coeff,
        "indices_0": 1,
        "indices_1": 2,
        "cutlass": SimpleNamespace(
            Int32=int,
            Float32=lambda v: torch.as_tensor(v, dtype=torch.float32),
            BFloat16=lambda v: torch.as_tensor(v, dtype=torch.bfloat16),
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                max=lambda a, b, *, propagate_nan: torch.maximum(a, b),
                min=lambda a, b, *, propagate_nan: torch.minimum(a, b),
            )
        ),
    }
    for row in (0, 31, 52, 53, 63):
        namespace["scan_row"] = row
        exec(expression, namespace)
        result = namespace["scan_value"]
        assert isinstance(result, torch.Tensor)
        if row >= 53:
            assert float(result) == 0.0
        else:
            value = x[1, row, 2].float()
            expected = (
                value.clamp(min=0.0) * coeff[2]
                if mode == "clamp"
                else ((value + 2.0) * coeff[2] - 1.0).clamp(max=5.0)
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)


def _guarded_scan(
    x: torch.Tensor, out: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        if mode == "divergent":
            if channel.begin == 0:
                out[row, channel] = hl.cumsum(x[row, channel] * 2.0, dim=0)
        elif mode == "extra_mask":
            value = hl.load(x, [row, channel], extra_mask=(row.index % 2) == 0)
            out[row, channel] = hl.cumsum(value * 2.0, dim=0)
        elif mode == "nested":
            for step in range(2):
                out[row, channel] = hl.cumsum(x[row, channel] * (step + 1.0), dim=0)
        else:
            out[row, channel] = hl.cumsum(x[row, channel] * 2.0, dim=0)
    return out


@pytest.mark.parametrize("mode", ["divergent", "extra_mask", "nested"])
def test_computed_scan_control_flow_and_explicit_masks_rejected(mode: str) -> None:
    with _cpu_context():
        args = (torch.empty(256, 3), torch.empty(256, 3), mode)
        bound = helion.kernel(_guarded_scan, backend="cute", static_shapes=True).bind(
            args
        )
        with pytest.raises(exc.BackendUnsupported):
            bound.to_code(bound.config_spec.default_config())


def _configurable_scan(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, None]):
        out[row, channel] = hl.cumsum(x[row, channel] * 2.0, dim=0)
    return out


def _multigrid_scan(
    x: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    other = torch.empty_like(x)
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        out[row, channel] = hl.cumsum(x[row, channel] * 2.0, dim=0)
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 256]):
        other[row, channel] = x[row, channel]
    return out, other


def test_computed_scan_multigrid_rejected() -> None:
    with _cpu_context():
        args = (torch.empty(256, 3), torch.empty(256, 3))
        bound = helion.kernel(_multigrid_scan, backend="cute", static_shapes=True).bind(
            args
        )
        with pytest.raises(exc.BackendUnsupported, match="physical layout"):
            bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("threads", [32, 64, 256])
def test_computed_scan_requires_exact_physical_width(threads: int) -> None:
    with _cpu_context():
        args = (torch.empty(256, 3), torch.empty(256, 3))
        bound = helion.kernel(
            _configurable_scan, backend="cute", static_shapes=True
        ).bind(args)
        config = helion.Config(block_sizes=[128], num_threads=[threads])
        error = "block size must be divisible" if threads > 128 else "computed add scan"
        with pytest.raises(exc.BackendUnsupported, match=error):
            bound.to_code(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reverse", [False, True])
def test_computed_scan_runtime(dtype: torch.dtype, reverse: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = DEVICE
    length, tile = 191, 128
    x = torch.empty(2, length, 3, device=device, dtype=dtype)
    coefficient = torch.empty(3, device=device)
    # Generic CuTe launchers require a 16-byte-aligned tensor base. Keep four
    # FP32 guard elements on each side without weakening the strided-view test.
    backing = torch.full((2, 3, length + 8), 73.0, device=device)
    out = backing[:, :, 4:-4].transpose(1, 2)
    args = (x, coefficient, out, tile, reverse, "clamp", 1)
    bound = helion.kernel(_computed, backend="cute", static_shapes=True).bind(args)
    config = bound.config_spec.default_config()
    assert "scan_expr_value" in bound.to_code(config)
    fn = bound.compile_config(config)
    for seed in range(5):
        torch.manual_seed(49200 + seed)
        x.normal_()
        coefficient.normal_()
        saved_x, saved_coefficient = x.clone(), coefficient.clone()
        product = (x.float().clamp(min=0.0) * coefficient).double()
        expected = torch.empty_like(product)
        for start in range(0, length, tile):
            part = product[:, start : start + tile]
            expected[:, start : start + tile] = (
                part.flip(1).cumsum(1).flip(1) if reverse else part.cumsum(1)
            )
        fn(*args)
        first = out.clone()
        fn(*args)
        assert torch.equal(out, first)
        torch.testing.assert_close(out.double(), expected, atol=3e-5, rtol=3e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*args)
        graph.replay()
        assert torch.equal(out, first)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out, first)
        assert torch.equal(x, saved_x) and torch.equal(coefficient, saved_coefficient)
        assert torch.all(backing[:, :, :4] == 73.0)
        assert torch.all(backing[:, :, -4:] == 73.0)


def _owner_domain_scan(
    x: torch.Tensor, coefficient: torch.Tensor, out: torch.Tensor, begin: int
) -> torch.Tensor:
    begin = hl.specialize(begin)
    for channel, row in hl.tile(
        [0, begin], [out.size(1), out.size(0)], block_size=[1, 128]
    ):
        value = x[row, channel] * coefficient[channel] + 1.0
        out[row, channel] = hl.cumsum(value, dim=0)
    return out


@pytest.mark.parametrize("shape,coeff_size", [((191, 3), 3), ((256, 4), 4)])
@pytest.mark.parametrize("begin", [0, 7])
def test_computed_scan_owner_domain_not_leaf_bounds(
    shape: tuple[int, int], coeff_size: int, begin: int
) -> None:
    x = torch.ones(shape)
    coefficient = torch.ones(coeff_size)
    out = torch.empty(191, 3)
    with _cpu_context():
        bound = helion.kernel(
            _owner_domain_scan, backend="cute", static_shapes=True
        ).bind((x, coefficient, out, begin))
        code = bound.to_code(bound.config_spec.default_config())
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion")
    )
    statements: list[ast.stmt] = []
    for node in device.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "scan_expr_load":
                statements = [node]
                continue
            if statements:
                statements.append(node)
            if node.targets[0].id == "scan_value":
                break
    assert statements
    identity = statements[-1]
    assert isinstance(identity, ast.Assign) and isinstance(identity.value, ast.IfExp)
    # The final predicate contains only the owning row interval. Source
    # coefficient/channel/row bounds remain in individual guarded loads.
    assert (
        ast.unparse(identity.value.test)
        == f"cutlass.Int32({begin}) <= scan_row < cutlass.Int32(191)"
    )
    expression = compile(
        ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
        "<owner-domain-value>",
        "exec",
    )
    namespace: dict[str, object] = {
        "x": x,
        "coefficient": coefficient,
        "cutlass": SimpleNamespace(
            Int32=int, Float32=lambda v: torch.as_tensor(v, dtype=torch.float32)
        ),
    }
    for channel in (0, 2):
        namespace["indices_0"] = channel
        for row in (-1, 0, 6, 7, 63, 64, 127, 180, 190, 191, 255):
            namespace["scan_row"] = row
            exec(expression, namespace)
            result = namespace["scan_value"]
            assert isinstance(result, torch.Tensor)
            if begin <= row < 191:
                source = 1.0 if 0 <= row < shape[0] and channel < shape[1] else 0.0
                coeff = 1.0 if channel < coeff_size else 0.0
                expected = source * coeff + 1.0
            else:
                expected = 0.0
            assert float(result) == expected


@pytest.mark.parametrize(
    "shape,coeff_size", [((191, 3), 2), ((191, 2), 3), ((64, 3), 3), ((64, 2), 2)]
)
@pytest.mark.parametrize("begin", [0, 7])
def test_computed_scan_unsafe_original_leaf_domain_rejected(
    shape: tuple[int, int], coeff_size: int, begin: int
) -> None:
    with _cpu_context():
        args = (torch.ones(shape), torch.ones(coeff_size), torch.empty(191, 3), begin)
        bound = helion.kernel(
            _owner_domain_scan, backend="cute", static_shapes=True
        ).bind(args)
        with pytest.raises(exc.BackendUnsupported, match="original load bounds"):
            bound.to_code(bound.config_spec.default_config())


def _offset_scan(
    x: torch.Tensor, coefficient: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for channel, row in hl.tile([x.size(1), x.size(0)], block_size=[1, 128]):
        value = x[row, channel] * coefficient[channel.index + 1] + 1.0
        out[row, channel] = hl.cumsum(value, dim=0)
    return out


def test_computed_scan_unsafe_coefficient_offset_rejected() -> None:
    with _cpu_context():
        args = (torch.ones(128, 3), torch.ones(3), torch.empty(128, 3))
        bound = helion.kernel(_offset_scan, backend="cute", static_shapes=True).bind(
            args
        )
        with pytest.raises(exc.BackendUnsupported):
            bound.to_code(bound.config_spec.default_config())


def test_computed_scan_negative_original_load_interval_rejected() -> None:
    with _cpu_context():
        args = (torch.ones(191, 3), torch.ones(3), torch.empty(191, 3), -7)
        bound = helion.kernel(
            _owner_domain_scan, backend="cute", static_shapes=True
        ).bind(args)
        with pytest.raises(exc.BackendUnsupported, match="original load bounds"):
            bound.to_code(bound.config_spec.default_config())
