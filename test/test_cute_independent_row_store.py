from __future__ import annotations

import ast
from collections import Counter
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler import tile_strategy as lanes
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _normalize_and_activate(
    q: torch.Tensor,
    k: torch.Tensor,
    raw: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for row in hl.tile(q.size(0)):
        qv = q[row, :].float()
        kv = k[row, :].float()
        qr = torch.rsqrt((qv * qv).sum(-1, keepdim=True) + 1e-6)
        kr = torch.rsqrt((kv * kv).sum(-1, keepdim=True) + 1e-6)
        query[row, :] = (qv * qr).to(q.dtype)
        key[row, :] = (kv * kr).to(k.dtype)
        beta[row] = torch.sigmoid(raw[row].float()).to(raw.dtype)
    return query, key, beta


def _config() -> helion.Config:
    return helion.Config(block_sizes=[4], num_threads=[4, 32])


def _code(rows: int, channels: int, dtype: torch.dtype) -> str:
    args = tuple(
        torch.empty(shape, dtype=dtype)
        for shape in (
            (rows, channels),
            (rows, channels),
            (rows,),
            (rows, channels),
            (rows, channels),
            (rows,),
        )
    )
    with _cpu_codegen():
        return _normalize_and_activate._bind_isolated(args).to_code(_config())


class _Memory:
    def __init__(self, count: int, stride: tuple[int, ...]) -> None:
        self.count = count
        self.layout = SimpleNamespace(stride=stride)
        self.writes: Counter[int] = Counter()
        self.iterator = _Pointer(self)


class _Pointer:
    def __init__(self, memory: _Memory, offset: int = 0) -> None:
        self.memory, self.offset = memory, offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.memory, self.offset + int(offset))

    def load(self) -> float:
        assert 0 <= self.offset < self.memory.count
        return 0.25

    def store(self, value: float) -> None:
        assert 0 <= self.offset < self.memory.count
        self.memory.writes[self.offset] += 1


@pytest.mark.parametrize("rows,channels", [(8, 128), (7, 65)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_generated_normalizer_has_one_writer_for_every_output(
    rows: int, channels: int, dtype: torch.dtype
) -> None:
    source = _code(rows, channels, dtype)
    tree = ast.parse(source)
    device = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    globals_ = [n for n in tree.body if isinstance(n, ast.Assign)]
    module = ast.Module(body=[*globals_, device], type_ignores=[])
    arch = SimpleNamespace(warp_reduction_sum=lambda x, **kwargs: x * 32)
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int, Int64=int, Float32=float, Float16=float, BFloat16=float
        ),
        "cute": SimpleNamespace(
            kernel=lambda f: f,
            arch=arch,
            math=SimpleNamespace(
                rsqrt=lambda x: 1 / math.sqrt(x),
                exp2=lambda x, **kwargs: 2**x,
                rcp=lambda x, **kwargs: 1 / x,
            ),
        ),
    }
    exec(
        compile(ast.fix_missing_locations(module), "<generated-ownership>", "exec"),
        namespace,
    )
    args = [
        _Memory(rows * channels, (channels, 1)),
        _Memory(rows * channels, (channels, 1)),
        _Memory(rows, (1,)),
        _Memory(rows * channels, (channels, 1)),
        _Memory(rows * channels, (channels, 1)),
        _Memory(rows, (1,)),
    ]
    named = dict(zip(("q", "k", "raw", "query", "key", "beta"), args, strict=True))
    # Execute actual emitted branches/index expressions at every physical
    # thread. The reduction stub is only a value placeholder; ownership and
    # bounds are checked against the exact generated stores.
    for block in range((rows + 3) // 4):
        arch.block_idx = lambda block=block: (block, 0, 0)
        for y in range(4):
            for x in range(32):
                arch.thread_idx = lambda x=x, y=y: (x, y, 0)
                arch.lane_idx = lambda x=x: x
                namespace[device.name](*(named[arg.arg] for arg in device.args.args))
    for name, count in (
        ("query", rows * channels),
        ("key", rows * channels),
        ("beta", rows),
    ):
        assert named[name].writes == Counter(dict.fromkeys(range(count), 1))
    assert not any(named[name].writes for name in ("q", "k", "raw"))


def _marker(
    *,
    group_pre: int = 1,
    group_span: int = 0,
    group_lane_expr: str = "",
    group_count: int = 1,
    group_cluster_n: int = 1,
    owner_lane: str | None = "serial_lane",
) -> lanes._LaneReduceMarker:
    return lanes._LaneReduceMarker(
        result_var="reduced",
        input_name="partial",
        reduction_type="sum",
        identity_expr="cutlass.Float32(0)",
        threads_in_group=32,
        wrap_template="__HELION_FINALIZED__",
        owner_lane=owner_lane,
        group_pre=group_pre,
        group_span=group_span,
        group_lane_expr=group_lane_expr,
        group_count=group_count,
        group_cluster_n=group_cluster_n,
    )


def _owners(
    source: str,
    marker: lanes._LaneReduceMarker,
    axes: dict[str, frozenset[int]] | None = None,
) -> list[str]:
    return lanes._lane_reduction_owner_exprs_for_statement(
        ast.parse(source).body[0],
        [(0, marker)],
        [(marker, {"reduced"})],
        axes or {},
        {},
    )


def test_grouped_owner_preserves_lower_and_upper_free_coordinates() -> None:
    expression = "cutlass.Int32(cute.arch.thread_idx()[0]) + 4 * cutlass.Int32(cute.arch.thread_idx()[1]) + 32 * cutlass.Int32(cute.arch.thread_idx()[2])"
    marker = _marker(
        group_pre=4, group_span=32, group_lane_expr=expression, group_count=2
    )
    owners = _owners(
        "(out.iterator + row).store(value)",
        marker,
        {"row": frozenset((0, 2)), "value": frozenset((0, 2))},
    )
    assert len(owners) == 1
    selected = Counter()
    arch = SimpleNamespace()
    for z in range(2):
        for y in range(8):
            for x in range(4):
                arch.thread_idx = lambda x=x, y=y, z=z: (x, y, z)
                if eval(
                    owners[0],
                    {
                        "cute": SimpleNamespace(arch=arch),
                        "cutlass": SimpleNamespace(Int32=int),
                    },
                ):
                    selected[x + 4 * z] += 1
    assert selected == Counter(dict.fromkeys(range(8), 1))


def test_packed_unresolved_reduction_coordinate_rejects() -> None:
    marker = _marker(
        group_pre=4,
        group_span=32,
        group_lane_expr="cutlass.Int32(cute.arch.thread_idx()[0])",
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="no proved reduction axis"):
        _owners("out.store(value)", marker)


def test_independent_cluster_store_requires_a_separate_owner_proof() -> None:
    with pytest.raises(helion.exc.BackendUnsupported, match="no proved cluster owner"):
        _owners("out.store(value)", _marker(group_cluster_n=2))


def test_legacy_unowned_marker_keeps_original_independent_store() -> None:
    assert _owners("out.store(value)", _marker(owner_lane=None)) == []


@pytest.mark.parametrize(
    "source",
    [
        "out.store(value)",
        "if cutlass.Int32(cute.arch.thread_idx()[0]) > 0:\n    out.store(1)",
        "if cutlass.Int32(cute.arch.thread_idx()[0]) < 16:\n    out.store(1)",
        "if predicate:\n    out.store(1)",
    ],
)
def test_varying_value_or_nonunique_thread_predicate_rejects(source: str) -> None:
    with pytest.raises(helion.exc.BackendUnsupported, match="thread-varying store"):
        _owners(
            source, _marker(), {"value": frozenset((0,)), "predicate": frozenset((0,))}
        )


def test_already_unique_and_per_channel_stores_remain_unchanged() -> None:
    for source, axes in [
        (
            "if cutlass.Int32(cute.arch.thread_idx()[0]) == 7:\n    out.store(value)",
            {"value": frozenset((0,))},
        ),
        (
            "(out.iterator + col).store(value)",
            {"col": frozenset((0,)), "value": frozenset((0,))},
        ),
    ]:
        assert _owners(source, _marker(), axes) == []


@pytest.mark.parametrize(
    "source",
    [
        "writer(value)",
        "out.store(mystery())",
        "out.atomic_add(value)",
        "if flag:\n    cute.arch.sync_threads()\n    out.store(value)",
        "if flag:\n    out.store(value)\n    other.store(value)",
    ],
)
def test_unknown_or_compound_effects_cannot_gain_owner(source: str) -> None:
    with pytest.raises(helion.exc.BackendUnsupported):
        _owners(source, _marker())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows,channels", [(8, 128), (7, 65)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_normalizer_independent_export_and_guards(
    rows: int, channels: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(213)
    q, k = [
        torch.randn((rows, channels), generator=generator, device=DEVICE, dtype=dtype)
        for _ in range(2)
    ]
    raw = torch.linspace(-100, 100, rows, device=DEVICE).to(dtype)
    raw[:2] = torch.tensor([-0.0, 0.0], device=DEVICE, dtype=dtype)
    original = [x.clone() for x in (q, k, raw)]
    storage = [
        torch.full((rows + 2, channels), 17, device=DEVICE, dtype=dtype)
        for _ in range(2)
    ]
    beta_storage = torch.full((rows + 2,), 17, device=DEVICE, dtype=dtype)
    outputs = [x[1:-1] for x in storage] + [beta_storage[1:-1]]
    expected = [
        (
            x.double()
            * torch.rsqrt((x.double() * x.double()).sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
        for x in (q, k)
    ]
    expected.append(torch.sigmoid(raw.double()).to(dtype))
    args = (q, k, raw, *outputs)
    bound = _normalize_and_activate._bind_isolated(args)
    source = bound.to_code(_config())
    assert "cute.arch.lane_idx()" in source
    fn = bound.compile_config(_config())
    for _ in range(2):
        for out in outputs:
            out.fill_(float("nan"))
        fn(*args)
        for actual, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.01)
        for backing in [*storage, beta_storage]:
            assert torch.all(backing[0] == 17) and torch.all(backing[-1] == 17)
        for actual, reference in zip((q, k, raw), original, strict=True):
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
