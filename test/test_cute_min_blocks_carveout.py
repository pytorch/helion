"""The shared-memory carveout preference the CuTe launcher passes
(``helion.runtime.cute.launch_descriptor.preferred_smem_carveout``)."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion.runtime.cute import launcher
from helion.runtime.cute.launch_descriptor import DRIVER_DEFAULT_SMEM_CARVEOUT
from helion.runtime.cute.launch_descriptor import min_blocks_per_mp
from helion.runtime.cute.launch_descriptor import preferred_smem_carveout

if TYPE_CHECKING:
    from collections.abc import Callable

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

# The DSL resolves a kernel's annotations in its module's globals.
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cutlass_dsl.cutlass as cutlass_dsl

pytestmark = pytest.mark.cpu_only


def _kernel(**attributes: object) -> SimpleNamespace:
    """A kernel carrying ``_helion_cute_<name>`` launch attributes, as the
    module statements and the standalone emitters (row_resident_codegen,
    row_matrix_transport_codegen, packed_operand_codegen) assign them."""
    return SimpleNamespace(
        **{f"_helion_cute_{name}": value for name, value in attributes.items()}
    )


@pytest.mark.parametrize("min_blocks", [2, 4, 6])
def test_min_blocks_past_one_leave_the_carveout_to_the_driver(min_blocks: int) -> None:
    assert (
        preferred_smem_carveout(_kernel(min_blocks_per_mp=min_blocks))
        == DRIVER_DEFAULT_SMEM_CARVEOUT
    )


@cute.kernel
def _rotate(x: cute.Tensor) -> None:
    """Rotates 64 values through 256 bytes of dynamic shared memory."""
    shared = cutlass.utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(64)
    )
    tid = cute.arch.thread_idx()[0]
    shared[tid] = x[tid]
    cute.arch.sync_threads()
    x[tid] = shared[(tid + 1) % 64]


def test_the_preference_stops_the_dsl_carveout_and_sets_none() -> None:
    """What the launch's preference does in the pinned DSL.  With
    min_blocks_per_mp 4 and no preference the compiled module sets the
    carveout the DSL computes from the launch's dynamic shared memory (here
    256 bytes); with the preference the module sets no carveout, and the
    value only lands in the compiled function's kernel_info, which the
    loader does not read."""

    def compiled(preference: int | None) -> tuple[str, dict[object, object]]:
        @cute.jit
        def launch(x: cute.Tensor) -> None:
            if cutlass.const_expr(preference is None):
                _rotate(x).launch(grid=(1, 1, 1), block=(64, 1, 1), min_blocks_per_mp=4)
            else:
                _rotate(x).launch(
                    grid=(1, 1, 1),
                    block=(64, 1, 1),
                    min_blocks_per_mp=4,
                    preferred_smem_carveout=preference,
                )

        function = cute.compile(launch, from_dlpack(torch.empty(64)))
        (info,) = function.kernel_info.values()
        return str(function.ir_module), dict(info)

    preference = preferred_smem_carveout(_kernel(min_blocks_per_mp=4))
    # The DSL reads its arch (``CUTE_DSL_ARCH``) when it is imported and
    # otherwise probes the device, which a CPU-only process cannot; its
    # calculation reads the SM's shared memory from the device too.
    with (
        patch.object(cutlass_dsl.CuTeDSL._get_dsl().envar, "_arch", "sm_100a"),
        patch.object(
            cutlass_dsl.cuda_helpers, "get_device_attribute", return_value=233472
        ),
    ):
        dsl_module, dsl_info = compiled(None)
        module, info = compiled(preference)
    assert "preferred_shared_memory_carveout" in dsl_module
    assert dsl_info == {}
    assert "preferred_shared_memory_carveout" not in module
    assert list(info.values()) == [DRIVER_DEFAULT_SMEM_CARVEOUT]


@pytest.mark.parametrize("min_blocks", [None, 0, 1])
def test_one_cta_per_sm_keeps_the_dsl_carveout(min_blocks: int | None) -> None:
    kernel = _kernel() if min_blocks is None else _kernel(min_blocks_per_mp=min_blocks)
    assert preferred_smem_carveout(kernel) is None


@pytest.mark.parametrize(
    ("attributes", "carveout", "min_blocks"),
    [
        ({"min_blocks_per_mp": 4}, ["-1"], ["4"]),
        ({"min_blocks_per_mp": 1}, [], ["1"]),
        ({}, [], []),
    ],
    ids=["min_blocks", "one_cta", "none"],
)
def test_the_launch_passes_at_most_one_carveout_beside_its_min_blocks(
    attributes: dict[str, object], carveout: list[str], min_blocks: list[str]
) -> None:
    """The carveout follows the min_blocks_per_mp the launch passes."""
    wrapper = launcher._create_cute_wrapper(_kernel(**attributes), (), (128, 1, 1))
    source = inspect.getsource(cast("Callable[..., object]", wrapper))
    assert re.findall(r"preferred_smem_carveout=(-?\d+)", source) == carveout
    assert re.findall(r"min_blocks_per_mp=(\d+)", source) == min_blocks


def test_the_compile_cache_key_carries_the_launched_carveout() -> None:
    """The key holds the preference the launch derives
    (``preferred_smem_carveout``), not the kernel's raw attribute: the same
    source with min_blocks_per_mp 4, which the launch gives -1 though the
    kernel chose no carveout, gets another key than without it."""
    keys = [
        launcher._cute_disk_cache_key(
            _kernel(source_hash="same generated source", min_blocks_per_mp=4),
            (),
            (128, 1, 1),
            (),
            None,
            None,
        ),
        launcher._cute_disk_cache_key(
            _kernel(source_hash="same generated source"),
            (),
            (128, 1, 1),
            (),
            None,
            None,
        ),
    ]
    assert None not in keys
    assert keys[0] != keys[1]


def test_the_fa4_topology_launches_one_cta_per_sm() -> None:
    """fa4 declares one CTA per SM so ptxas honors its register reallocation,
    which keeps the DSL's carveout; an emitter's min_blocks wins over it, and
    the carveout follows the min_blocks the launch passes."""
    fa4 = [{"topology": "fa4"}]
    assert min_blocks_per_mp(_kernel(), fa4) == 1
    assert preferred_smem_carveout(_kernel(), fa4) is None
    assert min_blocks_per_mp(_kernel(min_blocks_per_mp=4), fa4) == 4
    assert (
        preferred_smem_carveout(_kernel(min_blocks_per_mp=4), fa4)
        == DRIVER_DEFAULT_SMEM_CARVEOUT
    )
    assert min_blocks_per_mp(_kernel(), [{"topology": "ws_overlap"}]) is None
    assert min_blocks_per_mp(_kernel(min_blocks_per_mp=True)) is None
