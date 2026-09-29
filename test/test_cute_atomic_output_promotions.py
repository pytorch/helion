from __future__ import annotations

import ast
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_chunk_seeds import _partitioned_matmul

import helion
from helion._compiler.cute.atomic_output_promotions import fresh_half_atomic_outputs
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "source,accepted",
    [
        ("out = torch.zeros((2, 3))", True),
        ("out = torch.empty_like(x)", True),
        ("out = torch.zeros((2, 3), out=x)", False),
        ("out = torch.zeros((2, 3), **kwargs)", False),
        ("out = x.view(2, 3)", False),
        ("out = make_output()", False),
        ("out = torch.zeros((2, 3))\nout = x", False),
        ("out = torch.zeros((2, 3))\nfor out in values:\n    pass", False),
        ("out = torch.zeros((2, 3))\nout += x", False),
        ("out = alias = torch.zeros((2, 3))", False),
        ("out = torch.zeros((2, 3))\nalias = out", False),
        ("out = torch.zeros((2, 3))\nalias = out.view(2, 3)", False),
        ("out = torch.zeros((2, 3))\nalias = out[:1, :].expand(2, 3)", False),
        ("out = torch.zeros((2, 3))\nalias = (out,)", False),
        ("out = torch.zeros((2, 3))\nconsume(out)", False),
        ("out = torch.zeros((2, 3))\nshape = out.shape\nreturn out", True),
        ("out = torch.zeros((2, 3))\nreturn [out, (out,)]", True),
        ("out = torch.zeros((2, 3))\nmethod = out.size", False),
        ("out = torch.zeros((2, 3))\nreceiver = recover(out.stride)", False),
        ("out = torch.zeros((2, 3))\nmethod = out.numel", False),
        ("out = torch.zeros((2, 3))\nmethod = out.element_size", False),
        ("out = torch.zeros((2, 3))\nn = out.size(1)", True),
        ("out = torch.zeros((2, 3))\nn = out.stride(0)", True),
        ("out = torch.zeros((2, 3))\nn = out.numel()", True),
        ("out = torch.zeros((2, 3))\nn = out.element_size()", True),
        ("out = torch.zeros((2, 3))\ndef recover():\n    return out", False),
        ("out = torch.zeros((2, 3))\nrecover = lambda: out", False),
    ],
)
def test_output_promotion_requires_single_fresh_factory_binding(
    source: str, accepted: bool
) -> None:
    result = fresh_half_atomic_outputs(ast.parse(source).body, {"out": torch.float16})
    assert result == ({"out": torch.float16} if accepted else {})


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _aliased_output_matmul(
    a: torch.Tensor, b: torch.Tensor, broadcast: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    alias = out[:1, :].expand(m, n) if broadcast else out.view(m, n)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        previous = alias[row, column]
        hl.atomic_add(out, [row, column], acc + previous, sem="relaxed")
    return out


def _metadata_receiver(method: Any) -> torch.Tensor:
    return method.__self__


@dataclass
class _HostFactory:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)


_host_factory = _HostFactory()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unhashable_host_factory_output(a: torch.Tensor) -> torch.Tensor:
    out = _host_factory(a)
    for index in hl.tile(a.numel()):
        hl.atomic_add(out, [index], a[index], sem="relaxed")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _metadata_alias_output_matmul(
    a: torch.Tensor, b: torch.Tensor, broadcast: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    receiver = _metadata_receiver(out.size)
    alias = receiver[:1, :].expand(m, n) if broadcast else receiver.view(m, n)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        previous = alias[row, column]
        hl.atomic_add(out, [row, column], acc + previous, sem="relaxed")
    return out


def _inputs(dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty((64, 512), dtype=dtype), torch.empty((512, 128), dtype=dtype)


def _config(*, collective: bool = True, compute: str = "warp") -> helion.Config:
    return helion.Config(
        block_sizes=[
            32 if compute == "warp" else 64,
            64,
            32 if compute == "warp" else 64,
        ],
        num_threads=[2, 64, 1],
        cute_collective_mma=collective,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_stages=4 if compute == "warp" else 1,
        partitions=4,
    )


def _assert_fp32_atomic_output(code: str, input_name: str) -> None:
    assert f"dtype=torch.float32, device={input_name}.device" in code
    assert "val=cutlass.Float32(" in code
    assert "val=cutlass.Float16(" not in code
    assert "return out.to(torch.float16)" in code
    assert "_cute_atomic_add_f16x8" not in code


@skipUnlessBackends(["cute"])
def test_unhashable_host_callable_is_not_a_proven_fresh_factory() -> None:
    bound = _cpu_bind(
        _unhashable_host_factory_output, (torch.empty(32, dtype=torch.float16),)
    )
    assert not bound.env.cute_half_atomic_output_promotions


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("metadata_alias", [False, True])
@skipUnlessBackends(["cute"])
def test_output_storage_aliases_preserve_requested_half_atomics(
    collective: bool, broadcast: bool, metadata_alias: bool
) -> None:
    kernel = _metadata_alias_output_matmul if metadata_alias else _aliased_output_matmul
    bound = _cpu_bind(kernel, (*_inputs(), broadcast))
    assert not bound.env.cute_half_atomic_output_promotions
    config = _config(collective=collective).config.copy()
    config.pop("partitions")
    code = bound.to_code(helion.Config.from_dict(config))
    assert "dtype=a.dtype, device=a.device" in code
    assert "val=cutlass.Float16(" in code
    assert "return out.to(torch.float16)" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_into_argument(out: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    for index in hl.tile(updates.numel()):
        hl.atomic_add(out, [index], updates[index], sem="relaxed")
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@skipUnlessBackends(["cute"])
def test_argument_atomic_destination_keeps_its_requested_dtype(
    dtype: torch.dtype,
) -> None:
    args = (torch.empty(32, dtype=dtype), torch.empty(32, dtype=torch.float32))
    bound = _cpu_bind(_atomic_into_argument, args)
    assert not bound.env.cute_half_atomic_output_promotions
    code = bound.to_code(helion.Config(block_sizes=[32], num_threads=[32]))
    name = "Float16" if dtype == torch.float16 else "Float32"
    assert f"val=cutlass.{name}(" in code
    assert "return out.to(" not in code
    assert "torch.zeros(" not in code


@pytest.mark.parametrize("collective", [False, True])
@skipUnlessBackends(["cute"])
def test_all_accumulation_schedules_keep_fp32_storage_and_the_final_cast(
    collective: bool,
) -> None:
    bound = _cpu_bind(_partitioned_matmul, _inputs())
    assert bound.env.cute_half_atomic_output_promotions == {"out": torch.float16}
    config = _config(collective=collective)
    code = bound.to_code(config)
    _assert_fp32_atomic_output(code, "a")
    assert code == bound.to_code(config)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat = generation.flatten(config)
        restored = generation.unflatten(flat)
        assert generation.flatten(restored) == flat
        assert "cute_native_half_atomics" not in restored


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("value", [False, True, 1, "native"])
@skipUnlessBackends(["cute"])
def test_retired_accumulation_override_is_not_a_valid_config(
    collective: bool, value: object
) -> None:
    bound = _cpu_bind(_partitioned_matmul, _inputs())
    config = _config(collective=collective).config | {"cute_native_half_atomics": value}
    with pytest.raises(
        InvalidConfig, match="Invalid config keys.*cute_native_half_atomics"
    ):
        bound.to_code(helion.Config.from_dict(config))


@pytest.mark.parametrize("collective", [False, True])
@skipUnlessBackends(["cute"])
def test_nonpromoted_output_dtype_remains_unchanged(collective: bool) -> None:
    bound = _cpu_bind(_partitioned_matmul, _inputs(torch.bfloat16))
    assert not bound.env.cute_half_atomic_output_promotions
    code = bound.to_code(_config(collective=collective))
    assert "val=cutlass.BFloat16(" in code
    assert "dtype=a.dtype, device=a.device" in code
    assert "return out.to(" not in code


@pytest.mark.parametrize(
    "collective,compute", [(False, "warp"), (True, "warp"), (True, "tcgen05")]
)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_original_conditional_epilogue_keeps_fp32_output(
    collective: bool, compute: str, with_bias: bool, static_shapes: bool
) -> None:
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    a, b = _inputs()
    bias = torch.empty(b.size(1), dtype=a.dtype)
    inputs = (a, b, lambda acc, tile: acc + bias[tile[1]]) if with_bias else (a, b)
    bound = _cpu_bind(kernel, inputs)
    options = _config(collective=collective, compute=compute).config.copy()
    options["split_k"] = options.pop("partitions")
    _assert_fp32_atomic_output(bound.to_code(helion.Config.from_dict(options)), "x")


@skipUnlessBackends(["cute"])
def test_search_and_chunk_seeds_cannot_restore_half_accumulation() -> None:
    bound = _cpu_bind(_partitioned_matmul, _inputs())
    with bound.env:
        assert "cute_native_half_atomics" not in bound.config_spec._flat_fields()
        generation = ConfigGeneration(bound.config_spec)
        seeds = [config for _flat, config in generation.seed_flat_config_pairs()]
    assert seeds
    assert all("cute_native_half_atomics" not in seed for seed in seeds)
    choices = [
        seed
        for seed in seeds
        if seed.get("cute_collective_mma")
        and seed.get("cute_collective_copy") == "async_cached"
        and seed["partitions"] > 1
    ]
    assert {seed.get("cute_collective_stages", 1) for seed in choices} >= {1, 2, 4}
    seed = next(seed for seed in choices if seed.get("cute_collective_stages") == 4)
    _assert_fp32_atomic_output(bound.to_code(seed), "a")


@pytest.mark.parametrize(
    "collective,compute", [(False, "warp"), (True, "warp"), (True, "tcgen05")]
)
@skipUnlessBackends(["cute"])
def test_generated_host_resets_fp32_destination_and_casts_once(
    collective: bool, compute: str
) -> None:
    inputs = _inputs()
    bound = _cpu_bind(_partitioned_matmul, inputs)
    code = bound.to_code(_config(collective=collective, compute=compute))
    module = ast.parse(code)
    host = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(argument.arg == "_launcher" for argument in node.args.kwonlyargs)
    )
    buffers: list[torch.Tensor] = []

    def launcher(kernel: object, grid: object, *args: object, **kwargs: object) -> None:
        candidates = [
            value
            for value in args
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32
        ]
        assert len(candidates) == 1
        destination = candidates[0]
        assert torch.count_nonzero(destination).item() == 0
        destination.copy_(
            torch.linspace(-2.001, 2.001, destination.numel()).reshape(
                destination.shape
            )
        )
        buffers.append(destination)

    namespace: dict[str, Any] = {
        "torch": torch,
        "helion": helion,
        "_default_cute_launcher": launcher,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node is not host:
            namespace[node.name] = SimpleNamespace()
    host_module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, host],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(host_module), "<host-reset-test>", "exec"),
        namespace,
    )
    run = cast("Callable[..., torch.Tensor]", namespace[host.name])
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")):
        for iteration in range(3):
            result = run(*inputs)
            assert len(buffers) == iteration + 1
            destination = buffers[-1]
            assert destination.dtype == torch.float32
            assert result.dtype == torch.float16
            assert result.data_ptr() != destination.data_ptr()
            assert all(destination is not previous for previous in buffers[:-1])
            torch.testing.assert_close(
                result, destination.to(torch.float16), rtol=0, atol=0
            )
            result.fill_(float("nan"))
            destination.fill_(float("nan"))


_CUDA_CASES = [
    ("warp", 32, 4096, 64, [32, 64, 32], "async_cached", 4, 64),
    ("warp", 64, 16384, 128, [32, 32, 32], "async", 2, 128),
    ("warp", 128, 32768, 256, [128, 64, 64], "async", 1, 128),
    ("warp", 128, 32768, 256, [32, 64, 32], "async_cached", 4, 64),
    ("warp", 65, 530, 71, [32, 64, 32], "async_cached", 4, 4),
    ("warp", 65, 1, 71, [32, 64, 32], "async_cached", 1, 1),
    ("tcgen05", 64, 16384, 128, [64, 64, 64], "async_cached", 1, 64),
    ("tcgen05", 128, 32768, 256, [64, 64, 64], "async_cached", 1, 64),
    ("tcgen05", 65, 530, 71, [64, 64, 64], "async_cached", 1, 4),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("compute,m,k,n,blocks,copy,stages,partitions", _CUDA_CASES)
@skipUnlessBackends(["cute"])
def test_fp32_atomic_original_example_and_poisoned_graph_reset(
    compute: str,
    m: int,
    k: int,
    n: int,
    blocks: list[int],
    copy: str,
    stages: int,
    partitions: int,
    with_bias: bool,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(831)
    a = torch.randn((m, k + 8), device=CUDA_DEVICE, dtype=torch.float16)[:, :k]
    b = torch.randn((k, n + 8), device=CUDA_DEVICE, dtype=torch.float16)[:, :n]
    bias = torch.randn(n, device=CUDA_DEVICE, dtype=torch.float16)
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    inputs = (a, b, lambda acc, tile: acc + bias[tile[1]]) if with_bias else (a, b)
    bound = kernel._bind_isolated(inputs)
    config = helion.Config(
        block_sizes=blocks,
        num_threads=[128 // blocks[1], blocks[1], 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy=copy,
        cute_collective_stages=stages,
        split_k=partitions,
    )
    _assert_fp32_atomic_output(bound.to_code(config), "x")
    bound.set_config(config)
    destinations: list[torch.Tensor] = []
    original_zeros = torch.zeros

    def capture_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_zeros(*args, **kwargs)
        if value.dtype == torch.float32 and value.shape == (m, n):
            destinations.append(value)
        return value

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        count = len(destinations)
        result = bound(*inputs)
        assert len(destinations) == count + 1
        return result, destinations[-1]

    def expected() -> torch.Tensor:
        result = a.float() @ b.float()
        if with_bias:
            result = result + bias.float()
        return result.to(a.dtype)

    with patch("torch.zeros", side_effect=capture_zeros):
        replay = _make_cudagraph_replay(run)
        for _iteration in range(3):
            a.normal_()
            b.normal_()
            bias.normal_()
            reference = expected()
            eager, eager_scratch = run()
            torch.testing.assert_close(eager, reference, atol=1, rtol=1e-2)
            torch.testing.assert_close(eager, eager_scratch.to(a.dtype), atol=0, rtol=0)
            for _replay in range(4):
                result, scratch = replay()
                assert scratch.dtype == torch.float32
                torch.testing.assert_close(result, reference, atol=1, rtol=1e-2)
                torch.testing.assert_close(result, scratch.to(a.dtype), atol=0, rtol=0)
                result.fill_(float("nan"))
                scratch.fill_(float("nan"))
        a.zero_()
        b.zero_()
        bias.zero_()
        result, scratch = replay()
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        torch.testing.assert_close(scratch, torch.zeros_like(scratch), rtol=0, atol=0)
