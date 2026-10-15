"""Native correctness for raw consumer-packet caches across a row reduction."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = x.float()
    inverse = torch.rsqrt((values * values).mean(-1) + 1e-6)
    return (values * inverse[:, None] * weight.float()).to(x.dtype)


def _check_motion(code: str, *, width: int, vector: int) -> None:
    tree = ast.parse(code)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )
    loops = [node for node in kernel.body if isinstance(node, ast.For)]
    assert len(loops) == 2
    allocations = [
        node.value
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_consume_prefetch_")
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    ]
    assert len(allocations) == 1
    padded = 1 << (width - 1).bit_length()
    assert ast.literal_eval(allocations[0].args[0]) == padded // 1024

    def weight_packets(loop: ast.For) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and node.args
            and (
                ast.unparse(node.func) == "cute.arch.load"
                or ast.unparse(node.func).startswith("_cute_load_")
            )
            and any(
                isinstance(term, ast.Attribute)
                and isinstance(term.value, ast.Name)
                and term.value.id == "weight"
                and term.attr == "iterator"
                for term in ast.walk(node.args[0])
            )
        ]

    moved = weight_packets(loops[0])
    assert len(moved) == 1
    assert weight_packets(loops[1]) == []
    assert ast.literal_eval(moved[0].args[1].args[0]) == [vector]
    if width != padded:
        assert isinstance(moved[0].args[0], ast.IfExp), (
            "padded packets need their original mask"
        )
    if width == 7168 and vector == 4:
        lane_loop = loops[0].body[0]
        assert isinstance(lane_loop, ast.For)
        assert ast.unparse(lane_loop.iter) == "range(2)"


def _exercise(width: int, dtype: torch.dtype, *, offset_weight: bool = False) -> None:
    columns = torch.arange(width, device=DEVICE)
    x = (((columns % 67).float() - 33) * 0.03125 - 2.3).to(dtype)[None, :]
    initial_weight = (0.5 + (columns % 31).float() / 32).to(dtype)
    weight = (
        torch.empty(width + 1, dtype=dtype, device=DEVICE)[1:]
        if offset_weight
        else torch.empty_like(initial_weight)
    )
    weight.copy_(initial_weight)
    assert weight.storage_offset() == int(offset_weight)
    args = (x, weight, 1e-6)
    kernel = helion.kernel(_rms_norm, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    spec = bound.config_spec
    owner = spec.reduction_loops[0].block_id
    chunk = 1 << (width - 1).bit_length()
    vector = 8 if width == 7168 and dtype != torch.float32 else 4
    requested = helion.Config(
        block_sizes=[1],
        num_threads=[
            1024 if block == owner else 1
            for block in spec.num_threads.valid_block_ids()
        ],
        reduction_loops=[chunk],
        cute_vector_widths=[
            vector if block == owner else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_reduction_reloads=["register_prefetch"],
    )
    with bound.env:
        generation = ConfigGeneration(spec)
        flat, effective = generation.strict_config_pair(requested)
        assert generation.strict_unflatten(flat) == effective
        assert effective.config["reduction_loops"] == [chunk]
        assert effective.config["cute_reduction_reloads"] == ["register_prefetch"]
    code = bound.to_code(effective)
    assert f"_REDUCTION_BLOCK_{owner} = {chunk}" in code
    assert "block=(1024, 1, 1)" in code
    if offset_weight:
        assert "_consume_prefetch_" not in code
        control = helion.Config(
            **dict(effective.config, cute_reduction_reloads=["register"])
        )
        assert bound.to_code(control) == code
    else:
        _check_motion(code, width=width, vector=vector)

    compiled = bound.compile_config(effective)
    originals = (x.clone(), weight.clone())
    eager = compiled(*args)
    torch.testing.assert_close(eager, _reference(x, weight))
    for tensor, original in zip((x, weight), originals, strict=True):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    pointers = tuple(tensor.data_ptr() for tensor in (x, weight, captured))
    assert (
        len({tensor.untyped_storage().data_ptr() for tensor in (x, weight, captured)})
        == 3
    )
    for phase in ("initial", "changed", "restored"):
        if phase == "changed":
            x.mul_(-0.375).add_(0.4375)
            weight.mul_(0.625).add_(0.25)
        elif phase == "restored":
            x.copy_(originals[0])
            weight.copy_(originals[1])
        before = (x.clone(), weight.clone())
        expected = _reference(x, weight)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected)
        for tensor, original in zip((x, weight), before, strict=True):
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
        assert tuple(tensor.data_ptr() for tensor in (x, weight, captured)) == pointers


@skipUnlessCuteAvailable("requires the supported CuTe runtime")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [4096, 7168])
def test_consumer_packet_prefetch_graph(width: int, dtype: torch.dtype) -> None:
    _exercise(width, dtype)


@skipUnlessCuteAvailable("requires the supported CuTe runtime")
def test_consumer_packet_prefetch_offset_weight_falls_back() -> None:
    _exercise(7168, torch.bfloat16, offset_weight=True)
