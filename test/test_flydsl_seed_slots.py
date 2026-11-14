"""Exercise FlyDSL candidate construction without its ROCm compiler/runtime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from helion._compiler.flydsl.backend import FlyDSLBackend
from helion._testing import cpu_render_target
import helion.autotuner
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "persistent", [(False,), (True,), (True, False), (False, True)]
)
def test_candidates_preserve_legacy_slots_and_registered_vector_widths(
    monkeypatch: pytest.MonkeyPatch, persistent: tuple[bool, ...]
) -> None:
    with cpu_render_target():
        backend = FlyDSLBackend()
        spec = ConfigSpec(backend=backend, target_device_capability=None)
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=2))
        env = SimpleNamespace(config_spec=spec, block_sizes=())
        for index, required in enumerate(persistent):
            block_id = index + 7
            spec.reduction_loops.append(
                ReductionLoopSpec(
                    block_id=block_id,
                    size_hint=8192,
                    persistent_reason="complete ownership" if required else None,
                )
            )
            if not required:
                backend.register_reduction_loop_config_slots(env, block_id, 8192)

        bound = MagicMock(
            config_spec=spec, env=env, settings=Settings(backend="flydsl")
        )
        bound.settings.autotune_baseline_fn = lambda *args: None
        candidates = []

        def capture_search(bound_kernel, args, configs):
            candidates.extend(configs)
            return SimpleNamespace(autotune=lambda: configs[0])

        monkeypatch.setattr(helion.autotuner, "FiniteSearch", capture_search)
        backend.autotune(bound, (torch.empty(2, 8192),))
        bound.to_triton_code.assert_called_once()
        assert candidates
        rolled = []
        for candidate in candidates:
            loops = candidate.config.get("reduction_loops")
            if loops is not None:
                rolled.append(candidate)
                assert len(loops) == len(persistent)
                assert all(
                    loops[index] is None
                    for index, required in enumerate(persistent)
                    if required
                )
            widths = candidate.config.get("cute_vector_widths")
            if widths is not None:
                assert len(widths) == len(spec.cute_vector_widths)
            spec.normalize(candidate)
            assert len(candidate.reduction_loops) == len(persistent)
        assert bool(rolled) == (not all(persistent))


@pytest.mark.parametrize("extents", [(8192, 768), (768, 8192)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_candidates_bound_every_rolled_axis(
    monkeypatch: pytest.MonkeyPatch,
    extents: tuple[int, int],
    dtype: torch.dtype,
) -> None:
    with cpu_render_target():
        backend = FlyDSLBackend()
        spec = ConfigSpec(backend=backend, target_device_capability=None)
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=2))
        # A short mandatory-persistent axis is not subject to the chunk bound.
        spec.reduction_loops.append(
            ReductionLoopSpec(
                block_id=3, size_hint=3, persistent_reason="complete ownership"
            )
        )
        env = SimpleNamespace(config_spec=spec, block_sizes=())
        for block_id, extent in zip((7, 11), extents, strict=True):
            spec.reduction_loops.append(
                ReductionLoopSpec(block_id=block_id, size_hint=extent)
            )
            backend.register_reduction_loop_config_slots(env, block_id, extent)

        bound = MagicMock(
            config_spec=spec, env=env, settings=Settings(backend="flydsl")
        )
        candidates = []

        def capture_search(bound_kernel, args, configs):
            candidates.extend(configs)
            assert bound_kernel.settings.autotune_baseline_fn is not None
            return SimpleNamespace(autotune=lambda: configs[0])

        monkeypatch.setattr(helion.autotuner, "FiniteSearch", capture_search)
        backend.autotune(bound, (torch.empty(2, extents[0], dtype=dtype),))
        bound.to_triton_code.assert_called_once()
        bound.compile_config.assert_called_once()
        assert (
            bound.compile_config.call_args.args[0].config.get("reduction_loops") is None
        )
        assert bound.settings.autotune_baseline_fn is None
        assert candidates
        assert any(
            "reduction_loops" not in candidate.config for candidate in candidates
        )

        chunks_by_rows: dict[int, set[int]] = {1: set(), 2: set()}
        for candidate in candidates:
            explicitly_rolled = "reduction_loops" in candidate.config
            spec.normalize(candidate)
            first, second, third = candidate.reduction_loops
            assert first is None
            if not explicitly_rolled:
                continue
            assert second is not None
            assert second == third
            chunks_by_rows[candidate.block_sizes[0]].add(second)
        # Chunk 512 would pass the 8192-axis check, but its second 4-wide
        # vector pass reaches element 1020 on the 768-element axis. Chunk 256
        # stays valid for both axes, in both orders and for both input dtypes.
        assert chunks_by_rows == {1: {256}, 2: {64, 128, 256}}
