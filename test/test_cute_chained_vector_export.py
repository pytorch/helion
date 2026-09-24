from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from test.test_cute_chained_scan_export import _config
from test.test_cute_chained_scan_export import _cpu_codegen

import helion
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _vectors(a, b, delta, coefficient, mode: str):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    terminal = torch.empty((batch,), dtype=torch.float32, device=a.device)
    prefix = torch.empty(
        (reduction, batch)
        if mode == "permuted"
        else (batch, reduction + int(mode == "shape")),
        dtype=torch.bfloat16 if mode == "dtype" else torch.float32,
        device=a.device,
    )
    rawdt = torch.empty((batch, reduction), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, kk]
        product = raw.float().clamp(min=0) * coefficient[bi].float()
        if mode == "dependent":
            product = product + rb * 0.01
        scan = hl.cumsum(product, dim=0)
        left = (a[bi, row, kk].float() * torch.exp(scan)[None, :] * raw[None, :]).to(
            a.dtype
        )
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
        hl.store(terminal, [bi], scan[reduction - 1], extra_mask=(rb == 0) & (cb == 0))
        exported = scan
        if mode == "narrow":
            exported = exported.to(torch.bfloat16).float()
        if mode == "permuted":
            hl.store(prefix, [kk, bi], exported, extra_mask=(rb == 0) & (cb == 0))
        elif mode == "missing":
            hl.store(prefix, [bi, kk], exported, extra_mask=rb == 0)
        elif mode == "offset":
            hl.store(prefix, [bi, kk + 1], exported, extra_mask=(rb == 0) & (cb == 0))
        else:
            hl.store(prefix, [bi, kk], exported, extra_mask=(rb == 0) & (cb == 0))
        raw_export = raw
        if mode == "computed":
            raw_export = raw * 2
        hl.store(rawdt, [bi, kk], raw_export, extra_mask=(rb == 0) & (cb == 0))
    return out, terminal, prefix, rawdt


def _args(
    dtype: torch.dtype = torch.bfloat16, k: int = 128, view: str = "dense"
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(4201)
    a = torch.randn((3, 256, k), dtype=dtype, generator=g) * 0.05
    b = torch.randn((3, k, 192), dtype=dtype, generator=g) * 0.05
    raw = torch.rand((3, k), dtype=torch.float32, generator=g)
    if view == "stride":
        raw = torch.stack((raw, raw), dim=-1)[..., 0]
    elif view == "offset":
        backing = torch.empty(raw.numel() + 4, dtype=torch.float32)
        backing[4:].copy_(raw.flatten())
        raw = backing[4:].view(raw.shape)
    elif view == "short":
        raw = raw[:, :-1].contiguous()
    elif view == "narrow":
        raw = raw.to(dtype)
    return a, b, raw, -torch.rand((3,), generator=g)


def _code(
    mode: str = "normal",
    dtype: torch.dtype = torch.bfloat16,
    k: int = 128,
    view: str = "dense",
    n: int = 64,
) -> str:
    with _cpu_codegen():
        return _vectors._bind_isolated((*_args(dtype, k, view), mode)).to_code(
            _config(n)
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [32, 64, 128, 256])
def test_vector_exports_source(dtype: torch.dtype, k: int) -> None:
    code = _code(dtype=dtype, k=k)
    assert code.count("chain_scan_0_pointer =") == 1
    assert f"chain_scan_0_values[{k - 1}]" in code
    assert "chain_export_1_index < " + str(k) in code
    assert "chain_export_2_index < " + str(k) in code
    assert "chain_scan_0_values[chain_export_1_index]" in code
    assert "chain_origin_1 == 0 and chain_origin_2 == 0" in code
    assert (
        code.index("chain_epi_values")
        < code.index("for chain_export_1_step")
        < code.index("chain_allocator.free")
    )
    ast.parse(code)


@pytest.mark.parametrize("view", ["dense", "stride", "offset"])
@pytest.mark.parametrize("mode", ["normal", "permuted"])
def test_vector_exports_valid_views(mode: str, view: str) -> None:
    _code(mode, view=view)


@pytest.mark.parametrize(
    "mode", ["missing", "offset", "shape", "dtype", "narrow", "computed", "dependent"]
)
def test_vector_exports_reject(mode: str) -> None:
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _code(mode)


@pytest.mark.parametrize("view", ["short", "narrow"])
def test_vector_exports_leaf_reject(view: str) -> None:
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _code(view=view)


def test_vector_exports_padded_scan_reject() -> None:
    # A physical padded scan is not a proven full-vector logical store.
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        _code(k=48)


@pytest.mark.parametrize("mode", ["missing", "narrow", "computed"])
def test_vector_exports_no_ordinary_fallback(mode: str) -> None:
    with _cpu_codegen():
        bound = _vectors._bind_isolated((*_args(), mode))
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _consume(a, b, prefix, rawdt):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        kk = hl.arange(reduction)
        left = (
            a[bi, row, kk].float()
            * torch.exp(prefix[bi, kk])[None, :]
            * rawdt[bi, kk][None, :]
        ).to(a.dtype)
        out[bi, row, col] = hl.dot(left, b[bi, kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alias_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    prefix = torch.empty((batch, reduction), dtype=torch.float32, device=a.device)
    rawdt = prefix
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, kk]
        scan = hl.cumsum(raw, dim=0)
        out[bi, row, col] = hl.dot(
            (a[bi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, kk, col]
        )
        hl.store(prefix, [bi, kk], scan, extra_mask=(rb == 0) & (cb == 0))
        hl.store(rawdt, [bi, kk], raw, extra_mask=(rb == 0) & (cb == 0))
    return out, prefix, rawdt


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_export(a, b, delta):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi: Any = bt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        scan = hl.cumsum(delta[bi, kk], dim=0)
        out[bi, row, col] = hl.dot(
            (a[bi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, kk, col]
        )
        hl.store(delta, [bi, kk], scan, extra_mask=(rb == 0) & (cb == 0))
    return out, delta


@pytest.mark.parametrize("kernel", [_alias_export, _external_export])
def test_vector_exports_storage_alias_reject(kernel: Any) -> None:
    with _cpu_codegen(), pytest.raises((BackendUnsupported, InvalidConfig)):
        kernel._bind_isolated(_args()[:3]).to_code(_config())


def _host(code: str, name: str) -> tuple[Any, list[tuple[Any, ...]]]:
    host = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    launches: list[tuple[Any, ...]] = []

    def launch(*args: Any, **kwargs: Any) -> None:
        launches.append(args)

    namespace: dict[str, Any] = {
        "torch": torch,
        "_default_cute_launcher": launch,
        "_helion_" + name: SimpleNamespace(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[host], type_ignores=[])),
            "<actual-generated-host>",
            "exec",
        ),
        namespace,
    )
    return namespace[name], launches


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_vector_exports_actual_host_fresh_returns_and_consumer(
    dtype: torch.dtype,
) -> None:
    with _cpu_codegen():
        args = _args(dtype)
        producer = _vectors._bind_isolated((*args, "normal"))
        host, launches = _host(producer.to_code(_config()), "_vectors")
        retained: list[torch.Tensor] = []
        for _repeat in range(2):
            current = tuple(value.clone() for value in args)
            outputs = host(*current, "normal")
            assert tuple(value.shape for value in outputs) == (
                (3, 256, 192),
                (3,),
                (3, 128),
                (3, 128),
            )
            assert all(value.dtype is torch.float32 for value in outputs)
            storage = [
                value.untyped_storage().data_ptr()
                for value in (*current, *retained, *outputs)
            ]
            assert len(storage) == len(set(storage))
            assert all(
                launches[-1][i] is value
                for i, value in zip((4, 5, 2, 3), current, strict=True)
            )
            assert all(
                actual is value
                for actual, value in zip(launches[-1][-4:], outputs, strict=True)
            )
            consumer_args = (*current[:2], outputs[2], outputs[3])
            consumer = _consume._bind_isolated(consumer_args)
            code = consumer.to_code(_config())
            assert "chain_scan_" not in code
            consumer_host, consumer_calls = _host(code, "_consume")
            result = consumer_host(*consumer_args)
            assert result.dtype is torch.float32
            assert consumer_calls[-1][-1] is result
            assert all(
                any(value is actual for actual in consumer_calls[-1][2:])
                for value in consumer_args
            )
            retained.extend(outputs)


class _Pointer:
    def __init__(
        self, values: torch.Tensor, writes: list[int], offset: int = 0
    ) -> None:
        self.values = values
        self.writes = writes
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.values, self.writes, self.offset + offset)

    def store(self, value: torch.Tensor) -> None:
        assert 0 <= self.offset < self.values.numel()
        self.writes[self.offset] += 1
        self.values[self.offset] = value


@pytest.mark.parametrize("k", [32, 128, 256])
@pytest.mark.parametrize("mode", ["normal", "permuted"])
def test_vector_exports_actual_store_ast_bijection_and_bits(k: int, mode: str) -> None:
    code = _code(mode, k=k)
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion__vectors"
    )
    stores: list[ast.stmt] = [
        node
        for node in device.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("chain_export_")
    ]
    assert len(stores) == 2
    program = compile(
        ast.fix_missing_locations(ast.Module(body=stores, type_ignores=[])),
        "<actual-export-stores>",
        "exec",
    )
    # Include signed zeros, subnormals, infinities and two distinct NaN payloads.
    bits = torch.tensor(
        [0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2143289351, 1065353216],
        dtype=torch.int32,
    )
    source = bits.repeat((3 * k + 7) // 8)[: 3 * k].view(torch.float32).reshape(3, k)
    shapes = ((k, 3) if mode == "permuted" else (3, k), (3, k))
    outputs = [torch.empty(shape) for shape in shapes]
    counters = [[0] * output.numel() for output in outputs]
    env: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int, Float32=lambda value: value, range_constexpr=range
        )
    }
    for name, output, counter in zip(
        ("input_tensor_6", "input_tensor_7"), outputs, counters, strict=True
    ):
        env[name] = SimpleNamespace(
            iterator=_Pointer(output.view(-1), counter),
            layout=SimpleNamespace(stride=output.stride()),
        )
    for batch in range(3):
        env.update(
            chain_origin_0=batch,
            chain_scan_0_values=source[batch],
            chain_scan_0_input_0=source[batch],
        )
        for row in (0, 128):
            for column in (0, 64, 128):
                for thread in range(128):
                    env.update(
                        chain_origin_1=row, chain_origin_2=column, chain_thread=thread
                    )
                    exec(program, env)
    assert all(count == 1 for counts in counters for count in counts)
    expected = source.T.contiguous() if mode == "permuted" else source
    assert torch.equal(outputs[0].view(torch.int32), expected.view(torch.int32))
    assert torch.equal(outputs[1].view(torch.int32), source.view(torch.int32))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _projected_vectors(a, b, delta, kind: str):
    batch, groups, rows, reduction = a.shape
    columns = b.shape[3]
    out = torch.empty(
        (batch, groups, rows, columns), dtype=torch.float32, device=a.device
    )
    side = torch.empty((groups, reduction, batch), dtype=torch.float32, device=a.device)
    for bt, gt, row, col in hl.tile(
        [batch, groups, rows, columns], block_size=[1, 1, None, None]
    ):
        bi: Any = bt.begin
        gi: Any = gt.begin
        rb: Any = row.begin
        cb: Any = col.begin
        kk = hl.arange(reduction)
        raw = delta[bi, gi, kk]
        scan = hl.cumsum(raw, dim=0)
        out[bi, gi, row, col] = hl.dot(
            (a[bi, gi, row, kk].float() * scan[None, :]).to(a.dtype), b[bi, gi, kk, col]
        )
        value = raw
        if kind == "scan":
            value = scan
        if kind == "computed":
            value = raw * 2
        hl.store(side, [gi, kk, bi], value, extra_mask=(rb == 0) & (cb == 0))
    return out, side


@pytest.mark.parametrize("kind", ["scan", "raw"])
def test_vector_exports_multiple_outer_axes_and_vector_middle(kind: str) -> None:
    args = (
        torch.zeros(2, 3, 256, 128, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128, 192, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128),
        kind,
    )
    with _cpu_codegen():
        code = _projected_vectors._bind_isolated(args).to_code(_config())
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion__projected_vectors"
    )
    loop = next(
        node
        for node in device.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_export_0_step"
    )
    # Extract the actual target pointer arithmetic, not a handwritten owner map.
    store = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    )
    assert isinstance(store.func, ast.Attribute)
    pointer = compile(
        ast.Expression(body=store.func.value), "<actual-projected-pointer>", "eval"
    )
    target = next(
        node.id
        for node in ast.walk(store.func.value)
        if isinstance(node, ast.Name) and node.id.startswith("input_tensor")
    )
    env: dict[str, Any] = {
        "cutlass": SimpleNamespace(Int32=int),
        target: SimpleNamespace(iterator=0, layout=SimpleNamespace(stride=(256, 2, 1))),
    }
    offsets = []
    for bi in range(2):
        for gi in range(3):
            for index in range(128):
                env.update(
                    chain_origin_0=bi, chain_origin_1=gi, chain_export_0_index=index
                )
                offset = eval(pointer, env)
                assert offset == gi * 256 + index * 2 + bi
                offsets.append(offset)
    assert sorted(offsets) == list(range(2 * 3 * 128))
    assert "chain_origin_2 == 0 and chain_origin_3 == 0" in ast.unparse(loop)


def test_vector_exports_raw_only_computed_family_fails_closed() -> None:
    args = (
        torch.zeros(2, 3, 256, 128, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128, 192, dtype=torch.bfloat16),
        torch.zeros(2, 3, 128),
        "computed",
    )
    with _cpu_codegen():
        bound = _projected_vectors._bind_isolated(args)
        with pytest.raises(BackendUnsupported, match="scan export"):
            bound.to_code(helion.Config(block_sizes=[128, 64], num_warps=4))


@pytest.mark.parametrize("direct", [False, True])
def test_vector_exports_m64_transport(direct: bool) -> None:
    with _cpu_codegen():
        config = helion.Config(
            block_sizes=[64, 64],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_direct_output=direct,
        )
        source = _vectors._bind_isolated((*_args(), "normal")).to_code(config)
    assert "Ld16x256bOp" in source
    assert "chain_export_1_index" in source
    assert "chain_export_2_index" in source
