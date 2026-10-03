from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CPU_DEVICE = "cpu"


if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel


def _ordinary_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HELION_BACKEND", "cute")
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "full")
    with (
        _target(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _mock_cuda_unavailable(),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=_cuda_fake,
        ),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda", index=0, cc=100, warp_size=32, multi_processor_count=148
            ),
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(dtype: torch.dtype, shape: tuple[int, int, int]) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
    )
    kernel = helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True)
    return kernel.bind(args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2048, 4096, 1024), (4096, 2048, 2048)])
def test_ordinary_search_and_validated_ffi_default(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    bound = _bind(dtype, shape)
    with bound.env:
        tc = bound.config_spec._cute_tcgen05_config
        assert tc.full_tile_direct_entry_seed_eligible()
        optional = tc.optional_fragments(for_search=True)
        ffi = optional["tcgen05_tvm_ffi_launch"]
        assert isinstance(ffi, EnumFragment)
        assert ffi.default() is True
        assert ffi.search_values() == [False]
        assert set(ffi.choices) == {False, True}
        assert optional["tcgen05_flat_role_coordinates"].search_values() == [False]
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_d_store_box_n",
        ):
            assert optional[key].search_values() == [None]
        layout = tc.strategy_autotune_fragments()["tcgen05_layout_strategy"]
        assert isinstance(layout, EnumFragment)
        assert layout.default() == "default"
        assert layout.search_values() == ["default"]
        assert set(layout.choices) == {"default", "explicit_epi_tile"}

        generator = ConfigGeneration(bound.config_spec)
        default = generator.unflatten(generator.default_flat())
        seed = tc.full_tile_direct_entry_seed_config()
        assert seed is not None
        seed_flat, transferred = generator.canonicalize_flat(generator.flatten(seed))
        assert generator.unflatten(seed_flat) == transferred
        assert default.config["tcgen05_tvm_ffi_launch"] is True
        assert default.config["tcgen05_flat_role_coordinates"] is True
        assert default.config["tcgen05_layout_strategy"] == "explicit_epi_tile"
        assert default == transferred

        # User validation keeps its original two Boolean controls. The
        # random-search restrictions must not narrow explicit configuration.
        validation = tc.optional_fragments()
        assert isinstance(validation["tcgen05_tvm_ffi_launch"], BooleanFragment)
        assert isinstance(validation["tcgen05_flat_role_coordinates"], BooleanFragment)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.bfloat16, (128, 128, 128)),
        (torch.bfloat16, (4096, 4080, 4096)),
        (torch.float16, (1024, 1024, 1024)),
        (torch.float32, (2048, 4096, 1024)),
    ],
)
def test_ineligible_bind_has_no_ordinary_ffi_search_coordinate(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    bound = _bind(dtype, shape)
    with bound.env:
        tc = bound.config_spec._cute_tcgen05_config
        assert not tc.full_tile_direct_entry_seed_eligible()
        assert tc.full_tile_direct_entry_seed_config() is None
        assert "tcgen05_tvm_ffi_launch" not in tc.optional_fragments(for_search=True)


def test_missing_ab_budget_retains_ffi_decline() -> None:
    with patch.object(
        CuteTcgen05Config, "per_cta_ab_smem_budget_bytes", return_value=0
    ):
        bound = _bind(torch.bfloat16, (2048, 4096, 1024))
        with bound.env:
            assert not bound.config_spec._cute_tcgen05_config.full_tile_direct_entry_seed_eligible()


@pytest.mark.parametrize("grouping", [1, 8, 16, 32])
def test_non_ffi_sibling_survives_flat_transfer(grouping: int) -> None:
    bound = _bind(torch.bfloat16, (2048, 4096, 1024))
    with bound.env:
        config = helion.Config(
            block_sizes=[256, 256, 128],
            loop_orders=[[0, 1]],
            indexing=["tensor_descriptor"] * 3,
            pid_type="persistent_interleaved",
            l2_groupings=[grouping],
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=3,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_tvm_ffi_launch=False,
        )
        normalized = bound.config_spec.normalized_config(config)
        generator = ConfigGeneration(bound.config_spec)
        roundtrip = generator.unflatten(generator.flatten(normalized))
        assert roundtrip == normalized
        assert roundtrip.config["tcgen05_tvm_ffi_launch"] is False
        assert roundtrip.l2_groupings == [grouping]


@pytest.mark.parametrize("pid_type", ["flat", "persistent_interleaved"])
def test_one_cta_n_multicast_repairs_only_search(
    pid_type: Literal["flat", "persistent_interleaved"],
) -> None:
    bound = _bind(torch.bfloat16, (2048, 4096, 1024))
    with bound.env:
        requested = helion.Config(
            block_sizes=[128, 64, 64],
            loop_orders=[[0, 1]],
            indexing=["tensor_descriptor"] * 3,
            pid_type=pid_type,
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=2,
            tcgen05_tvm_ffi_launch=False,
        )
        with pytest.raises(InvalidConfig, match="tcgen05_cluster_n=2 requires"):
            bound.config_spec.normalized_config(requested)
        fixed = helion.Config.from_dict(copy.deepcopy(requested.config))
        bound.config_spec.normalize(fixed, _fix_invalid=True)
        assert fixed.config["tcgen05_cluster_m"] == 1
        assert fixed.config["tcgen05_cluster_n"] == 1
        assert fixed.block_sizes == requested.block_sizes
        bound.config_spec.normalize(fixed)


@pytest.mark.parametrize(
    "invalid",
    [
        {"tcgen05_ab_stages": 6, "tcgen05_c_stages": 4},
        {"tcgen05_cta_group": "two"},
        {"tcgen05_num_epi_warps": 2},
        {"tcgen05_tvm_ffi_launch": 1},
    ],
)
def test_ffi_validation_rejects_existing_illegal_combinations(
    invalid: dict[str, object],
) -> None:
    bound = _bind(torch.bfloat16, (2048, 4096, 1024))
    with bound.env:
        seed = (
            bound.config_spec._cute_tcgen05_config.full_tile_direct_entry_seed_config()
        )
        assert seed is not None
        values = copy.deepcopy(seed.config)
        values.update(invalid)
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(values)


def _cuda_fake(backend: object, tensor: torch.Tensor) -> torch.Tensor:
    assert isinstance(tensor, FakeTensor)
    with torch._C._DisableTorchDispatch():
        meta = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=torch.device("meta"),
            requires_grad=tensor.requires_grad,
        )
    return FakeTensor(tensor.fake_mode, meta, torch.device("cuda:0"))
