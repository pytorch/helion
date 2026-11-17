"""Shared pieces of the capture-contract replay tests.

A kernel with the capture contract (the proved paired attention kernel under
guarded dependent launch) may read its inputs ahead of a predecessor the
capture guard proved harmless. These helpers observe that decision, capture
back-to-back launches, compare a rescheduled kernel bit for bit, and set up
the grid of more than one wave on which the release moves from each loader's
tail to the entry of every CTA.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import torch

from .cute_paged_attention_schedule import paged_attention_args
from .cute_pdl import kernel_definition
import helion
from helion.runtime.cute import capture_pdl_launch
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion.runtime.cute.capture_pdl import CaptureGuard
    from helion.runtime.cute.capture_pdl import LaunchEffects
    from helion.runtime.cute.capture_pdl_cuda import CaptureDriver
    from helion.runtime.cute.capture_pdl_cuda import CaptureSnapshot
    from helion.runtime.kernel import BoundKernel

_Outputs = TypeVar("_Outputs", bound="Sequence[torch.Tensor | None]")

# One paired CTA per SM: a full wave, then a partial one of this many CTAs.
MULTIWAVE_LAST_WAVE = 24
RELEASE_CALL = "cute.arch.griddepcontrol_launch_dependents()"


@contextmanager
def observed_wait_modes() -> Iterator[list[WaitMode]]:
    """The wait mode of each capture-contract launch in the block, in order."""
    modes: list[WaitMode] = []
    finish = capture_pdl_launch.finish_capture_launch

    def observed(
        driver: CaptureDriver,
        guard: CaptureGuard,
        before: CaptureSnapshot,
        effects: LaunchEffects,
        parameter_sizes: tuple[int, ...],
        **options: int,
    ) -> WaitMode:
        mode = finish(driver, guard, before, effects, parameter_sizes, **options)
        modes.append(mode)
        return mode

    with patch.object(capture_pdl_launch, "finish_capture_launch", observed):
        yield modes


def capture_launches(
    graph: torch.cuda.CUDAGraph,
    compiled: Callable[..., _Outputs],
    args: Sequence[object],
    calls: int,
    *,
    overlap: bool = True,
) -> list[_Outputs]:
    """Capture ``calls`` back-to-back launches of a warmed kernel into ``graph``.

    With ``overlap`` the kernel has the capture contract: the first launch has
    no tracked predecessor and keeps the entry wait, and each later one reads
    its inputs before the launch ahead of it finishes. The caller keeps every
    output alive, so the graph pool cannot reuse their storage.
    """
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        captured = [compiled(*args) for _ in range(calls)]
    if overlap:
        expected = [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * (calls - 1)
        assert modes == expected, modes
    return captured


def fill_nan(captured: Sequence[Sequence[torch.Tensor | None]]) -> None:
    """Poison captured outputs, so a replay that skips a write fails."""
    for output in captured:
        for value in output:
            if value is not None:
                value.fill_(float("nan"))


def assert_identical(
    actual: Sequence[torch.Tensor | None], expected: Sequence[torch.Tensor | None]
) -> None:
    """Bit for bit: a dependent launch only reschedules a call, so every row
    runs the same operations in the same order as in the plain kernel."""
    for value, reference in zip(actual, expected, strict=True):
        if value is None:
            assert reference is None
        else:
            torch.testing.assert_close(value, reference, rtol=0, atol=0, equal_nan=True)


def ramp_attention_operands(args: Sequence[object]) -> None:
    """Deterministic Q, K and V contents with rows that differ."""
    for index, period in enumerate((19, 15, 17)):
        tensor = cast("torch.Tensor", args[index])
        ramp = torch.arange(tensor.numel(), device=tensor.device).reshape(tensor.shape)
        tensor.copy_((ramp % period - period // 2) / (32 if index == 0 else 1))


def multiwave_attention_args(
    device: torch.device | str, *, distinct_pages: bool = False
) -> list[object]:
    """``paged_attention_args`` for a grid of more than one wave of paired CTAs.

    One request is one paired CTA of eight rows (one query position's grouped
    heads), and there are ``MULTIWAVE_LAST_WAVE`` more requests than the
    device has SMs. Q, K and V hold ramps, the scales four exponents, and
    every length is zero. The page table cycles through the thirteen physical
    pages, or with ``distinct_pages`` gives every logical page its own, so a
    test can write to keys no row sees without reaching a visible one.
    """
    requests = (
        torch.cuda.get_device_properties(device).multi_processor_count
        + MULTIWAVE_LAST_WAVE
    )
    args = list(
        paged_attention_args(
            rows=8,
            requests=requests,
            pages=requests * 17 if distinct_pages else 13,
            device=device,
        )
    )
    k, _v, ks, vs, pages, lengths = (cast("torch.Tensor", x) for x in args[1:7])
    ramp_attention_operands(args)
    for scale in (ks, vs):
        scale.copy_(
            (torch.arange(scale.numel(), device=device) % 4 + 125).reshape(scale.shape)
        )
    table = torch.arange(pages.numel(), device=device)
    table = table.flip(0) if distinct_pages else table * 7 % k.size(0)
    pages.copy_(table.reshape(pages.shape))
    lengths.zero_()
    return args


def multiwave_lengths(
    requests: int, shift: int, device: torch.device | str
) -> torch.Tensor:
    """Empty, shorter-than-a-page, page-boundary and full ranges among mixed
    ones, moved to other CTAs (and so to other waves) by ``shift``."""
    lengths = (torch.arange(requests, dtype=torch.int64) * 389 + 97 * shift) % 2177
    for index, value in enumerate((0, 1, 5, 128, 129, 2176)):
        lengths[(shift + index * 7) % requests :: 11] = value
    return lengths.to(dtype=torch.int32, device=device)


def compile_multiwave_defaults(
    bound: BoundKernel,
    args: Sequence[object],
    *,
    last_wave_guard: bool = False,
    **schedule: object,
) -> tuple[
    helion.Config,
    str,
    Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
]:
    """The paired kernel's defaults on the multi-wave grid, beside the joined
    kernel without dependent launch.

    The config names the schedule (query TMA, V lag 2 and ``schedule``) and
    no dependent-launch, occupancy or retirement option. It therefore
    normalizes to the guarded capture overlap with early retirement, and
    past one wave the release moves from each loader's tail to the entry of
    every CTA: one unit per CTA holds its SM, so the dependents launch as
    the last CTA enters. With ``last_wave_guard`` the config spells that
    policy out on the ``last_wave`` release instead: only the CTAs of the
    last wave release, and the dependents wait for the first CTAs in
    particular. Returns the normalized config, its source, that kernel and
    its counterpart with ``cute_pdl=False`` and the final join, both warmed:
    the capture guard tracks only launches of a warmed kernel.
    """
    config = helion.Config.from_dict(
        {
            "block_sizes": [8],
            "cute_mma_dag": "tcgen05",
            "cute_mma_dag_warps": 4,
            "cute_mma_dag_stages": 2,
            "cute_mma_dag_pass_warps": "attention_paired",
            "cute_mma_dag_query_tma": True,
            "cute_mma_dag_v_lag": 2,
            **schedule,
        }
    )
    if last_wave_guard:
        config.config.update(
            cute_pdl=True, cute_pdl_capture_overlap=True, cute_pdl_release="last_wave"
        )
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_pdl") is True
    assert normalized.get("cute_pdl_capture_overlap") is True
    assert normalized.get("cute_pdl_release") == (
        "last_wave" if last_wave_guard else "loader_tail"
    )
    assert normalized.get("cute_mma_dag_early_tmem_retirement") is True
    source = bound.to_code(config)
    # The capture contract's conditional wait, then the release: of every
    # CTA, or of the last wave's under its guard.
    wait, release = kernel_definition(source).body[:2]
    assert ast.unparse(cast("ast.If", wait).test).endswith("capture_pdl_wait_mode == 0")
    assert ("pdl_cta" in source) is last_wave_guard
    if last_wave_guard:
        assert f"if pdl_cta >= {MULTIWAVE_LAST_WAVE}:" in source
    else:
        assert ast.unparse(release) == RELEASE_CALL
    assert "capture_release_tail=True" not in source
    tracked = bound.compile_config(config)
    # The counterpart names no overlap and no release: neither holds without
    # dependent launch.
    schedule_alone = {
        key: value
        for key, value in config.config.items()
        if key not in ("cute_pdl_capture_overlap", "cute_pdl_release")
    }
    plain = bound.compile_config(
        helion.Config.from_dict(
            {
                **schedule_alone,
                "cute_pdl": False,
                "cute_mma_dag_early_tmem_retirement": False,
            }
        )
    )
    tracked(*args)
    plain(*args)
    torch.cuda.synchronize()
    return normalized, source, tracked, plain
