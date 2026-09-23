from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from test.test_cute_grouped_gemm_split_sizes import _selected_config
from test.test_cute_shared_rhs_grouped import _bind
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.cute_mma import _tcgen05_full_allocation_b_index_domain
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    import helion


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _one_config(k: int = 64, stages: int = 2) -> helion.Config:
    config = _selected_config(k, source_m_tile=32)
    config.config.update(tcgen05_cluster_m=1, tcgen05_ab_stages=stages)
    return config


def _shared_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((257, 128), dtype=dtype),
        torch.empty((128, 256), dtype=dtype),
        torch.tensor([0, 1, 193, 257, 257], dtype=torch.int32),
    )


def _grouped_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((257, 128), dtype=dtype),
        torch.empty((8, 256, 128), dtype=dtype),
        torch.tensor([0, 1, 193, 257, 257, 257, 257, 257, 257], dtype=torch.int32),
    )


def _check_absolute_b(source: str, *, shared_rhs: bool) -> None:
    plans = _plans(source)
    grouped = next(p for p in plans if p["kind"] == "tcgen05_grouped_static_persistent")
    assert grouped["cluster_m"] == 1
    assert grouped["dynamic_d_tensormap"] is True
    assert grouped["dynamic_ab_tensormaps"] is True
    assert grouped["device_split_sizes"] is True
    assert grouped["scheduler_mode"] == "device_group_search"
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    copies = [node for node in calls if ast.unparse(node.func) == "cute.copy"]
    b_copies = [node for node in copies if ast.unparse(node.args[0]) == "tma_atom_b"]
    assert b_copies
    assert all(
        not any(k.arg == "tma_desc_ptr" for k in call.keywords) for call in b_copies
    )
    assert "init_tensormap_from_atom(tma_atom_b" not in source
    assert "tcgen05_grouped_tensormap_real_b =" not in source
    assert "fence_tensormap_update(tcgen05_grouped_tensormap_b_ptr)" not in source
    assert (
        "cute.local_tile(cute.domain_offset((tcgen05_grouped_global_m_start, "
        "cutlass.Int32(0)), tma_tensor_b), (32, "
    ) in source
    assert "tcgen05_grouped_d_tensormap_manager.update_tensormap" in source
    assert "tcgen05_grouped_d_tensormap_manager.fence_tensormap_update" in source
    if shared_rhs:
        assert "init_tensormap_from_atom(tma_atom_a" not in source
    else:
        assert "init_tensormap_from_atom(tma_atom_a" in source
        assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" in source
        assert "fence_tensormap_update(tcgen05_grouped_tensormap_a_ptr)" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("k,stages", [(64, 2), (64, 5), (128, 3)])
@skipUnlessBackends(["cute"])
def test_one_shared_rhs_uses_immutable_packed_b(
    dtype: torch.dtype, column_major: bool, k: int, stages: int
) -> None:
    args = _shared_inputs(dtype)
    if column_major:
        args = (args[0], torch.empty((256, 128), dtype=dtype).T, args[2])
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        source = bound.to_code(_one_config(k, stages))
    _check_absolute_b(source, shared_rhs=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["offsets", "split_sizes"])
@skipUnlessBackends(["cute"])
def test_one_grouped_rhs_preserves_its_dynamic_a_descriptor(
    dtype: torch.dtype, layout: str
) -> None:
    args = _grouped_inputs(dtype)
    kernel = _device_offsets_kernel
    if layout == "split_sizes":
        args = (
            args[0],
            args[1],
            torch.tensor([1, 192, 64, 0, 0, 0, 0, 0], dtype=torch.int32),
        )
        kernel = _device_split_sizes_kernel
    with _target():
        bound = _bind(kernel.fn, args)
        source = bound.to_code(_one_config())
    _check_absolute_b(source, shared_rhs=False)


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@skipUnlessBackends(["cute"])
def test_offset_values_and_stride_do_not_specialize_absolute_b(
    offset_dtype: torch.dtype,
) -> None:
    a, b, offsets = _shared_inputs(torch.bfloat16)
    offsets = offsets.to(offset_dtype)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, (a, b, offsets))
        before = bound.to_code(_one_config())
        offsets.copy_(
            torch.tensor([-(1 << 31), (1 << 31) - 1, 193, 17, 257], dtype=offset_dtype)
        )
        assert bound.to_code(_one_config()) == before
        backing = torch.empty(10, dtype=offset_dtype)
        backing[::2].copy_(offsets)
        replacements = (a.clone(), b.clone(), backing[::2])
        rebound = _bind(grouped_gemm_jagged.fn, replacements)
        assert rebound.to_code(_one_config()) == before
    _check_absolute_b(before, shared_rhs=True)


@pytest.mark.parametrize("bad", ["a_gap", "a_offset", "b_gap", "b_offset"])
@skipUnlessBackends(["cute"])
def test_one_declines_unproved_input_layout_or_alignment(bad: str) -> None:
    a, b, offsets = _shared_inputs(torch.bfloat16)
    if bad == "a_gap":
        a = torch.empty((257, 129), dtype=a.dtype)[:, :128]
    elif bad == "a_offset":
        a = torch.empty(a.numel() + 1, dtype=a.dtype)[1:].view(a.shape)
    elif bad == "b_gap":
        b = torch.empty((128, 257), dtype=b.dtype)[:, :256]
    else:
        b = torch.empty(b.numel() + 1, dtype=b.dtype)[1:].view(b.shape)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, (a, b, offsets))
        with pytest.raises(BackendUnsupported):
            bound.to_code(_one_config())


@skipUnlessBackends(["cute"])
def test_one_grouped_rhs_declines_input_output_alias() -> None:
    source = inspect.getsource(_device_offsets_kernel.fn)
    source = source[source.index("def _device_offsets_kernel(") :]
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    allocation = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "out"
            for target in node.targets
        )
    )
    allocation.value = ast.Name("a_packed", ast.Load())
    module = PyCodeCache.load(
        "import torch\nimport helion.language as hl\n" + ast.unparse(tree)
    )
    a = torch.empty((257, 128), dtype=torch.bfloat16)
    b = torch.empty((8, 128, 128), dtype=a.dtype)
    offsets = _grouped_inputs(a.dtype)[2]
    with _target():
        bound = _bind(module._device_offsets_kernel, (a, b, offsets))
        assert not bound.config_spec.cute_tcgen05_search_enabled
        with pytest.raises(InvalidConfig, match="tcgen05-enabled"):
            bound.to_code(_one_config())


@skipUnlessBackends(["cute"])
def test_one_grouped_rhs_declines_unproved_effect_order() -> None:
    args = _grouped_inputs(torch.bfloat16)
    with _target():
        bound = _bind(_device_offsets_kernel.fn, args)
        with (
            patch(
                "helion._compiler.cute.cute_mma._shared_rhs_region_is_exclusive",
                return_value=False,
            ),
            pytest.raises(BackendUnsupported, match="pure contraction"),
        ):
            bound.to_code(_one_config())


@pytest.mark.parametrize(
    "m,k,tile,expected",
    [
        (1, 64, 32, True),
        (257, 128, 32, True),
        (2560, 512, 32, True),
        (0, 128, 32, False),
        (-1, 128, 32, False),
        (128, 0, 32, False),
        (128, 64, 0, False),
        (((1 << 31) - 1) // 64, 64, 32, True),
        ((1 << 31) // 64, 64, 32, False),
        ((1 << 31) - 32, 1, 32, True),
        ((1 << 31) - 31, 1, 32, False),
    ],
)
def test_absolute_b_signed_index_domain(
    m: int, k: int, tile: int, expected: bool
) -> None:
    assert _tcgen05_full_allocation_b_index_domain(m, k, tile) is expected


@skipUnlessBackends(["cute"])
def test_existing_two_cta_shared_rhs_keeps_dynamic_b() -> None:
    args = _shared_inputs(torch.bfloat16)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        source = bound.to_code(_selected_config(64))
    assert "init_tensormap_from_atom(tma_atom_b" in source
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" in source
    assert "cute.domain_offset((tcgen05_grouped_global_m_start" not in source


@skipUnlessBackends(["cute"])
def test_one_config_roundtrip_and_discoverable_seed() -> None:
    import helion

    config = _one_config()
    assert helion.Config.from_dict(config.config).config == config.config
    args = _shared_inputs(torch.bfloat16)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        seeds = bound.config_spec.compiler_seed_configs
        one = [seed for seed in seeds if seed.config.get("tcgen05_cluster_m") == 1]
        assert one
        for seed in one:
            _check_absolute_b(bound.to_code(seed), shared_rhs=True)
