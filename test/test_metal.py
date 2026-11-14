from __future__ import annotations

import ast
import inspect
import itertools
import math
import re
import sys
import textwrap
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache
import torch.nn.functional as F

import helion
from helion import exc
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.metal import free_arange
from helion._compiler.metal import lane_reduction
from helion._compiler.metal import row_reduction_heuristic
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.metal.lane_fold import LaneJam
from helion._compiler.metal.lane_fold import balanced_combine
from helion._compiler.metal.lane_fold import jam_lanes_into_loops
from helion._compiler.metal.lane_fold import unroll_lane_folds
from helion._compiler.metal.metal_overrides import MetalOverrides
from helion._compiler.metal.mpp_matmul import names_read
from helion._compiler.metal.msl_ast_walker import _ast_expr_to_msl
from helion._compiler.metal.msl_expr import parse_msl_expression
from helion._compiler.metal.msl_reduction import GROUP_PATHS
from helion._compiler.metal.row_reduction_heuristic import MetalRowReductionHeuristic
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._testing import import_example
from helion._testing import skipIfRefEager
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

helion_gdn_fwd_h = import_example("gdn_fwd_h").helion_gdn_fwd_h
jsd_forward = import_example("jsd").jsd_forward
kl_div_forward = import_example("kl_div").kl_div_forward
layer_norm_bwd = import_example("layer_norm").layer_norm_bwd
rms_norm_bwd = import_example("rms_norm").rms_norm_bwd
_examples_rope = import_example("rope")
rope_bwd = _examples_rope.rope_bwd
rope_fwd = _examples_rope.rope_fwd
rope_pytorch = _examples_rope.rope_pytorch
softmax_two_pass = import_example("softmax").softmax_two_pass
welford = import_example("welford").welford

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.platform == "darwin":
    from helion._compiler.metal.metal_jit import _constexpr_decl
    from helion._compiler.metal.metal_jit import _MetalKernel

# Metal runs on macOS only: CI's Linux jobs skip the whole file.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS"
)

_requires_darwin = unittest.skipIf(
    sys.platform != "darwin", "Metal tests require macOS"
)


def _full_threadgroup(test: Callable[..., None]) -> Callable[..., None]:
    """For a test that pins the layout of a full 1024-thread threadgroup:
    configs that leave ``num_warps`` unset take 32 instead of the default 4,
    whose 128-thread budget walks more of each block in lanes."""
    return patch("helion.autotuner.config_spec.DEFAULT_NUM_WARPS", 32)(test)


DEVICE = "mps"

_DEFAULT_CONFIG = [helion.Config(block_sizes=[256], num_warps=4)]


def _run_with_msl(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    config: helion.Config | None = None,
) -> tuple[object, str, str]:
    """Run *kernel* through the normal Helion pipeline; return its result, its
    Python code and its MSL.

    Uses PyCodeCache to load the generated module (same as Helion's runtime)
    and calls the host function, which makes metal_jit generate the MSL.
    """
    code = kernel.bind(args).to_code(config)
    module = PyCodeCache.load(code)
    result = getattr(module, kernel.fn.__name__)(*args)
    (msl,) = [
        obj.msl_source
        for obj in vars(module).values()
        if isinstance(obj, _MetalKernel) and obj.msl_source is not None
    ]
    return result, code, msl


def _get_msl(kernel: helion.Kernel, args: tuple[object, ...]) -> str:
    """The MSL *kernel* generates for *args*."""
    return _run_with_msl(kernel, args)[2]


def _element_loops(msl: str) -> list[str]:
    """The text of each MPP element loop: ``for (auto _it = c.begin(); ...) {``
    through its closing brace."""
    loops = []
    header = r"for \(auto _it = (\w+)\.begin\(\); _it != \1\.end\(\); _it\+\+\) \{"
    for match in re.finditer(header, msl):
        depth, end = 1, match.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(msl[end], 0)
            end += 1
        loops.append(msl[match.start() : end])
    return loops


def _group_reductions(msl: str) -> list[tuple[str, str, str, int, int]]:
    """``(op, path, partials, axis, span)`` of each ``helion_red::group_*`` call.

    ``path`` is the one ``metal_jit`` picked for the launch, and ``partials``
    the cross-SIMD-group scratch buffer.
    """
    paths = dict(re.findall(r"constexpr uint (_red_site\w*) = (\d+);", msl))
    calls = re.findall(
        r"helion_red::group_(\w+)\((_red_site\w*), (\w+), .*?, tid, "
        r"(\d+), (\d+), _TG_SIZE_X",
        msl,
    )
    return [
        (op, GROUP_PATHS[int(paths[site])], partials, int(axis), int(span))
        for op, site, partials, axis, span in calls
    ]


def _static_threadgroup_memory(msl: str) -> int:
    """Threadgroup memory the Metal compiler allocates for *msl*'s kernel.

    Declarations whose every use folds away cost nothing, so this, not the
    MSL text, says whether a kernel touches threadgroup memory.
    """
    name = re.search(r"kernel void (\w+)\(", msl)
    assert name is not None
    library = torch.mps.compile_shader(msl)  # type: ignore[attr-defined]
    return getattr(library, name.group(1)).static_thread_group_memory_length


# ---------------------------------------------------------------------------
# Kernel definitions – copy
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def copy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def copy_into(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def masked_copy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.where(mask[tile], x[tile], 0.0)
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – arithmetic
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] - y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_neg(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = -x[tile]
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – scalar args
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def saxpy(x: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = a * x[tile] + b * y[tile]
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – activations
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.where(x[tile] > 0, x[tile], 0.0)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = (
            0.5
            * x[tile]
            * (1.0 + torch.tanh(0.7978845608 * (x[tile] + 0.044715 * x[tile] ** 3)))
        )
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – math ops
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def exp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.exp(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def log_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.log(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def sqrt_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sqrt(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def abs_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.abs(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def sincos_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sin(x[tile]) + torch.cos(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def special_kernel(x: torch.Tensor) -> torch.Tensor:
    """Ops Inductor emits as ``c10::metal`` special functions."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        out[tile] = torch.erf(v) + torch.lgamma(v.abs() + 1)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def floordiv_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inductor lowers float ``//`` with logical ops and ``where``."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] // y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def atan2_kernel(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Inductor spells atan2 ``::metal::precise::atan2``, from global scope."""
    out = torch.empty_like(y)
    for tile in hl.tile(y.size(0)):
        out[tile] = torch.atan2(y[tile], x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def logical_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a, b = x[tile] > 0, y[tile] > 0
        out[tile] = torch.logical_and(a, b).float() + 2 * torch.logical_or(a, b)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def gelu_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        out[tile] = F.gelu(v) - F.gelu(v, approximate="tanh")
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def gelu_tanh_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = F.gelu(x[tile], approximate="tanh")
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def clamp_kernel(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.clamp(x[tile], lo, hi)
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – multi-dimensional
# ---------------------------------------------------------------------------


@helion.kernel(
    backend="metal", configs=[helion.Config(block_sizes=[64, 64], num_warps=4)]
)
def elementwise_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(
    backend="metal",
    configs=[helion.Config(block_sizes=[16, 16, 16], num_warps=4)],
)
def elementwise_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_0, tile_1, tile_2 in hl.tile([x.size(0), x.size(1), x.size(2)]):
        out[tile_0, tile_1, tile_2] = (
            x[tile_0, tile_1, tile_2] + y[tile_0, tile_1, tile_2]
        )
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – large block (1D, block_size > 1024)
# ---------------------------------------------------------------------------


@helion.kernel(
    backend="metal", configs=[helion.Config(block_sizes=[2048], num_warps=4)]
)
def large_block_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


# ---------------------------------------------------------------------------
# Tests – copy
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalCopy(unittest.TestCase):
    """Copy kernels that test load/store + masking."""

    def test_copy(self) -> None:
        """Aligned size: out[tile] = x[tile] with size 1024."""
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(copy_kernel(x), x)

    def test_copy_non_aligned(self) -> None:
        """Non-aligned size: mask must be active for correctness."""
        x = torch.randn(1000, device=DEVICE)
        torch.testing.assert_close(copy_kernel(x), x)

    def test_masked_copy(self) -> None:
        """torch.where with a boolean mask exercises _mask_to lowering."""
        x = torch.randn(1024, device=DEVICE)
        mask = torch.randint(0, 2, (1024,), device=DEVICE, dtype=torch.bool)
        result = masked_copy(x, mask)
        expected = torch.where(mask, x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)

    def test_masked_copy_non_aligned(self) -> None:
        """Masked copy with non-aligned size: both tile mask and user mask active."""
        x = torch.randn(1000, device=DEVICE)
        mask = torch.randint(0, 2, (1000,), device=DEVICE, dtype=torch.bool)
        result = masked_copy(x, mask)
        expected = torch.where(mask, x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests – bounds masking
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalBoundsMasking(unittest.TestCase):
    """Bounds masking tests using vector_add for both load and store paths."""

    def test_store_no_oob_write(self) -> None:
        """Sentinel buffer detects OOB store writes."""
        n = 999
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        out = buf[:n]
        copy_into(x, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    def test_store_no_oob_write_size_1(self) -> None:
        """Extreme case: N=1 with block_size=256 → 255 OOB threads."""
        n = 1
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        out = buf[:n]
        copy_into(x, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    def test_codegen_has_mask(self) -> None:
        """Generated MSL must contain bounds checks for non-aligned sizes."""
        x = torch.randn(999, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertIn("mask_0", msl, "mask variable not found in generated MSL")
        self.assertIn("if (mask_0)", msl, "store guard not found in generated MSL")
        self.assertIn("?", msl, "load ternary not found in generated MSL")

    def test_copy_codegen(self) -> None:
        """The MSL of an aligned copy: a flattened tile keeps its mask for an
        aligned size too (its strategy is built before ``pre_codegen`` judges
        the launch); no trivial ``* 1`` strides; array subscripts, not pointer
        dereferences; and no MPP headers."""
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertIn("mask_0", msl, "mask variable expected even for aligned size")
        self.assertNotIn("* 1)", msl, "trivial * 1 stride found in generated MSL")
        self.assertNotIn("* 1]", msl, "trivial * 1 stride found in generated MSL")
        self.assertNotIn(
            "*((", msl, "pointer dereference found; expected array subscript"
        )
        self.assertIn("x[", msl, "array subscript load not found in generated MSL")
        self.assertIn("out[", msl, "array subscript store not found in generated MSL")
        self.assertNotIn("<metal_tensor>", msl)
        self.assertNotIn("MetalPerformancePrimitives", msl)
        self.assertNotIn("mpp::tensor_ops", msl)

    def test_vector_add_non_aligned(self) -> None:
        """vector_add with non-aligned size exercises mask on both load and store."""
        x = torch.randn(1000, device=DEVICE)
        y = torch.randn(1000, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_tile_mask_bounds_threads_to_the_tile(self) -> None:
        """A tail's mask must also bound threads to their tile."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32], num_warps=32)],
        )
        def add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                out[tm, tn] = x[tm, tn] + y[tm, tn]
            return out

        x = torch.randn(100, 100, device=DEVICE)
        y = torch.randn(100, 100, device=DEVICE)
        msl = _get_msl(add2d, (x, y))
        self.assertIn("(tid[0] < _BLOCK_SIZE_0)", msl)
        self.assertIn("(tid[1] < _BLOCK_SIZE_1)", msl)

    def test_a_dynamic_tile_thread_extent_is_rejected(self) -> None:
        """A threadgroup is shaped at launch, so its extents must be known then.

        ``thread_block_sizes`` reports only the extents it can resolve, and the
        launcher turns those into the literal ``_block_dims``.  An extent it
        drops leaves the axis one thread wide while the index expression stays
        ``offset + tid``, so each tile is computed in its first row alone and
        the rest of the output keeps whatever ``empty_like`` handed back.

        Only a block size read off a tensor under ``static_shapes=False`` gets
        here; a config-chosen one is an ``int``.  Three ways in, one per
        strategy: flattened, ND, and the ND lane path -- the last being the one
        whose thread mask skips an axis it cannot bound, on the strength of the
        axis never reaching codegen.
        """

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def flat(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm in hl.tile(x.size(0), block_size=x.size(0) // 2):
                out[tm] = x[tm] * 2.0
            return out

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def nd_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(
                [x.size(0), x.size(1)], block_size=[x.size(0) // 2, 16]
            ):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        # ``num_threads`` below the (static) block size gives axis 0 a lane
        # loop, which is what routes this through
        # ``PerThreadNDTileStrategy.codegen_grid`` rather than the base ND path.
        @helion.kernel(
            backend="metal",
            static_shapes=False,
            configs=[helion.Config(block_sizes=[64], num_threads=[16])],
        )
        def nd_lane(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(
                [x.size(0), x.size(1)], block_size=[None, x.size(1) // 2]
            ):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        x = torch.randn(128, 32, device=DEVICE)
        for kernel in (flat, nd_kernel, nd_lane):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, "not known until"),
            ):
                kernel(x[:, 0] if kernel is flat else x)


# ---------------------------------------------------------------------------
# Tests – arithmetic
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalArithmetic(unittest.TestCase):
    """Basic arithmetic elementwise kernels."""

    def test_vector_add(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_vector_sub(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_sub(x, y), x - y)

    def test_vector_mul(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_mul(x, y), x * y)

    def test_vector_div(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE).abs() + 0.1
        torch.testing.assert_close(vector_div(x, y), x / y)

    def test_vector_neg(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_neg(x), -x)


# ---------------------------------------------------------------------------
# Tests – scalar args
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalScalarArgs(unittest.TestCase):
    """Kernels that accept scalar (SymbolArgument) parameters."""

    def test_saxpy(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        a, b = 2.5, -1.0
        torch.testing.assert_close(saxpy(x, y, a, b), a * x + b * y)

    def test_runtime_scalars_bind_by_type(self) -> None:
        """A runtime size binds as ``long``, so integer index math (the 2-D
        program id's floor division) type-checks; a float stays ``float``."""

        @helion.kernel(
            backend="metal",
            static_shapes=False,
            configs=[helion.Config(block_sizes=[16, 16])],
        )
        def add_sized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        for shape in ((64, 64), (100, 72), (33, 129)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE)
                y = torch.randn(*shape, device=DEVICE)
                out, _, msl = _run_with_msl(add_sized, (x, y))
                torch.testing.assert_close(out, x + y)
                self.assertRegex(msl, r"constant long& \w+ \[\[buffer")
        x = torch.randn(1024, device=DEVICE)
        _, _, msl = _run_with_msl(saxpy, (x, x, 2.5, -1.0))
        self.assertIn("constant float& a [[buffer", msl)


# ---------------------------------------------------------------------------
# Tests – activations
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalActivations(unittest.TestCase):
    """Activation function kernels."""

    def test_relu(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(relu(x), torch.relu(x))

    def test_silu(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(
            silu(x), torch.nn.functional.silu(x), atol=1e-5, rtol=1e-5
        )

    def test_gelu_approx(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        expected = torch.nn.functional.gelu(x, approximate="tanh")
        torch.testing.assert_close(gelu_approx(x), expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests – math ops
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMathOps(unittest.TestCase):
    """Math function kernels."""

    def test_exp(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(exp_kernel(x), torch.exp(x), atol=1e-5, rtol=1e-5)

    def test_log(self) -> None:
        x = torch.rand(1024, device=DEVICE) + 0.1
        torch.testing.assert_close(log_kernel(x), torch.log(x), atol=1e-5, rtol=1e-5)

    def test_sqrt(self) -> None:
        x = torch.rand(1024, device=DEVICE) + 0.1
        torch.testing.assert_close(sqrt_kernel(x), torch.sqrt(x))

    def test_abs(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(abs_kernel(x), torch.abs(x))

    def test_sincos(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        expected = torch.sin(x) + torch.cos(x)
        torch.testing.assert_close(sincos_kernel(x), expected, atol=1e-5, rtol=1e-5)

    def test_special_functions(self) -> None:
        """erf and lgamma reach Helion's own special-function overrides."""
        x = torch.randn(1024, device=DEVICE)
        expected = torch.erf(x) + torch.lgamma(x.abs() + 1)
        torch.testing.assert_close(special_kernel(x), expected, atol=1e-5, rtol=1e-5)

    def test_atan2(self) -> None:
        y = torch.randn(1024, device=DEVICE)
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(
            atan2_kernel(y, x), torch.atan2(y, x), atol=1e-5, rtol=1e-5
        )

    def test_float_floor_division(self) -> None:
        """Including the corners Inductor's lowering special-cases: an
        infinite dividend or divisor, a zero divisor, and NaN."""
        x = torch.randn(1024, device=DEVICE) * 10
        y = torch.randn(1024, device=DEVICE)
        inf, nan = float("inf"), float("nan")
        x[:6] = torch.tensor([inf, -inf, 1.0, -1.0, 3.0, nan])
        y[:6] = torch.tensor([2.0, 3.0, inf, inf, 0.0, 1.0])
        torch.testing.assert_close(floordiv_kernel(x, y), x // y, equal_nan=True)

    def test_logical_ops(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        a, b = x > 0, y > 0
        expected = torch.logical_and(a, b).float() + 2 * torch.logical_or(a, b)
        torch.testing.assert_close(logical_kernel(x, y), expected)

    def test_gelu(self) -> None:
        """Both forms, each in float and rounded back once for 16-bit
        inputs, as eager computes them."""
        for dtype, tol in (
            (torch.float32, 1e-5),
            (torch.float16, 1e-3),
            (torch.bfloat16, 1e-2),
        ):
            with self.subTest(dtype=dtype):
                x = torch.randn(1024, device=DEVICE, dtype=dtype) * 3
                torch.testing.assert_close(
                    gelu_tanh_kernel(x),
                    F.gelu(x, approximate="tanh"),
                    atol=tol,
                    rtol=tol,
                )
                expected = F.gelu(x) - F.gelu(x, approximate="tanh")
                torch.testing.assert_close(gelu_kernel(x), expected, atol=tol, rtol=tol)

    def test_clamp(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(
            clamp_kernel(x, -0.5, 0.5), torch.clamp(x, -0.5, 0.5)
        )


# ---------------------------------------------------------------------------
# Tests – dtypes
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalDtypes(unittest.TestCase):
    """Elementwise ops across different dtypes."""

    def test_float16_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_bfloat16_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_int32_add(self) -> None:
        x = torch.randint(-100, 100, (1024,), device=DEVICE, dtype=torch.int32)
        y = torch.randint(-100, 100, (1024,), device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_float16_neg(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_neg(x), -x)

    def test_bfloat16_mul(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_mul(x, y), x * y)


# ---------------------------------------------------------------------------
# Dtype conversions
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile].to(torch.bfloat16)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def to_float16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile].to(torch.float16)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def to_float32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile].to(torch.float32)
    return out


_VIA_FLOAT = "static_cast<bfloat>(static_cast<float>("


def _bits_mismatch(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Where two equal-dtype float tensors differ in bits, NaNs aside."""
    nan = torch.isnan(out) & torch.isnan(ref)
    return (out.view(torch.int16) != ref.view(torch.int16)) & ~nan


@_requires_darwin
class TestMetalBfloatConversion(unittest.TestCase):
    """Every conversion to bfloat goes through float, the way torch CPU and
    CUDA convert, since c10's BFloat16 is built from a float only."""

    def test_integers_round_through_float(self) -> None:
        # The large values are ones where a direct int -> bfloat cast rounds
        # once and lands one ulp away from rounding through float.
        sources = {
            "int32": torch.tensor(
                [33685505, 34471935, -2034237486, 0, 1, -1, 7, -8, 2**24 + 1],
                dtype=torch.int32,
            ),
            "int64": torch.tensor(
                [
                    3810045181170676896,
                    -1391612340377475687,
                    3954160416921393058,
                    0,
                    -8,
                ],
                dtype=torch.int64,
            ),
            "int8": torch.arange(-128, 128, dtype=torch.int8),
        }
        for name, x in sources.items():
            with self.subTest(source=name):
                out, _, msl = _run_with_msl(to_bfloat16, (x.to(DEVICE),))
                self.assertIn(_VIA_FLOAT, msl)
                ref = x.to(torch.bfloat16)
                self.assertFalse(_bits_mismatch(out.cpu(), ref).any())

    def test_floating_sources_keep_their_values(self) -> None:
        gen = torch.Generator().manual_seed(0)
        random_bits = torch.randint(
            -(2**31), 2**31 - 1, (1 << 16,), generator=gen, dtype=torch.int32
        )
        specials = torch.tensor(
            [0.0, -0.0, 1.0, -1.5, 65504.0, 1e30, -3.4e38, math.inf, -math.inf],
            dtype=torch.float32,
        )
        sources = {
            "float16": torch.arange(-(2**15), 2**15, dtype=torch.int16).view(
                torch.float16
            ),
            "float32": torch.cat([specials, random_bits.view(torch.float32)]),
        }
        smallest_normal = torch.finfo(torch.bfloat16).smallest_normal
        for name, x in sources.items():
            with self.subTest(source=name):
                out = to_bfloat16(x.to(DEVICE)).cpu()
                ref = x.to(torch.bfloat16)
                # Metal's float -> bfloat cast flushes bfloat subnormals to
                # zero, with or without the float step.
                subnormal = (ref != 0) & (ref.float().abs() < smallest_normal)
                self.assertFalse((_bits_mismatch(out, ref) & ~subnormal).any())

    def test_other_targets_keep_the_direct_cast(self) -> None:
        x = torch.randint(-1000, 1000, (1000,), device=DEVICE, dtype=torch.int32)
        for kernel, dtype in ((to_float16, "half"), (to_float32, "float")):
            with self.subTest(target=dtype):
                out, _, msl = _run_with_msl(kernel, (x,))
                self.assertRegex(msl, rf"static_cast<{dtype}>\(load")
                self.assertNotIn(f"static_cast<{dtype}>(static_cast<float>(", msl)
                torch.testing.assert_close(out, x.to(out.dtype))


# ---------------------------------------------------------------------------
# Tests – multi-dimensional
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMultiDim(unittest.TestCase):
    """Multi-dimensional elementwise kernels."""

    def test_elementwise_2d(self) -> None:
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_elementwise_2d_non_aligned(self) -> None:
        x = torch.randn(100, 100, device=DEVICE)
        y = torch.randn(100, 100, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_elementwise_3d(self) -> None:
        x = torch.randn(16, 16, 16, device=DEVICE)
        y = torch.randn(16, 16, 16, device=DEVICE)
        torch.testing.assert_close(elementwise_3d(x, y), x + y)


# ---------------------------------------------------------------------------
# Kernel definitions – index arithmetic
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def gather_row_group(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Each run of ``h // g`` output rows reads one row of ``x``."""
    h, n = out.shape
    g = x.size(0)
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin // (h // g), tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_by_negative_quotient(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin, tn] * ((th.begin - 3) // 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gather_negative_divisor(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[(th.begin + 1) // -2 + 4, tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gather_row_mod(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin % 3, tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_by_negative_mod(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin, tn] * ((th.begin - 5) % 3)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_one_device_grid_count(x: torch.Tensor) -> torch.Tensor:
    """The inner grid's count is computed on the device from a block size."""
    m, n = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(16, 16)
    for tm in hl.tile(m):
        for gn in hl.grid((n + bn - 1) // bn):
            out[tm, gn] = x[tm, gn] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_half_block_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    bm = hl.register_block_size(m)
    for tm, tn in hl.tile([m, n], block_size=[bm, None]):
        out[tm, tn] = x[tm, tn] + y[bm // 2]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_block_ratio_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tm, tn in hl.tile([m, n], block_size=[bm, bn]):
        out[tm, tn] = x[tm, tn] + y[bm // bn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_block_mod_negative_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    bm = hl.register_block_size(m)
    for tm, tn in hl.tile([m, n], block_size=[bm, None]):
        out[tm, tn] = x[tm, tn] + y[bm % -3 + 2]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gather_quotient_minus_row(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin - 5 + (th.begin + 11) // 2, tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_by_quotient_minus_row(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[th.begin, tn] * ((th.begin + 1) // 2 - th.begin)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gather_square_third(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    h, n = out.shape
    for th, tn in hl.tile([h, n], block_size=[1, None]):
        out[th.begin, tn] = x[(th.begin * th.begin) // 3, tn]
    return out


# ---------------------------------------------------------------------------
# Tests – index arithmetic
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalIndexArithmetic(unittest.TestCase):
    """Integer ``//`` and ``%`` of index values (tile offsets, block sizes)
    follow Python's floor semantics for any operand sign.  An expression
    taking that path is computed in ``long``; one whose division the printer
    proves nonnegative, or a block size divided by an integer, keeps its
    plain ``//``."""

    def _run(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None = None,
    ) -> tuple[torch.Tensor, str]:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        code = bound.to_code(config)
        return bound.compile_config(config)(*args), code

    def _assert_signed(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        expected: torch.Tensor,
        config: helion.Config | None = None,
    ) -> None:
        out, code = self._run(kernel, args, config)
        torch.testing.assert_close(out, expected)
        self.assertNotIn("triton_helpers", code)
        self.assertIn("static_cast(long, ", code)

    def _scale_rows(self, x: torch.Tensor, factors: list[int]) -> torch.Tensor:
        return x * torch.tensor(factors, device=DEVICE, dtype=x.dtype)[:, None]

    def test_floordiv_tile_offset_by_group(self) -> None:
        """A non-power-of-two group: 9 rows in groups of 3."""
        x = torch.randn(3, 40, device=DEVICE)
        out = torch.zeros(9, 40, device=DEVICE)
        self._assert_signed(gather_row_group, (x, out), x.repeat_interleave(3, dim=0))

    def test_floordiv_negative_operand(self) -> None:
        """``(b - 3) // 2`` is -2 at b = 0 (C division gives -1), and a
        negative divisor floors toward negative infinity too."""
        x = torch.randn(8, 40, device=DEVICE)
        out = torch.zeros(8, 40, device=DEVICE)
        with self.subTest("negative lhs"):
            expected = self._scale_rows(x, [(b - 3) // 2 for b in range(8)])
            self._assert_signed(scale_by_negative_quotient, (x, out), expected)
        with self.subTest("negative divisor"):
            x4 = torch.randn(4, 40, device=DEVICE)
            rows = [(b + 1) // -2 + 4 for b in range(8)]
            self._assert_signed(gather_negative_divisor, (x4, out), x4[rows])

    def test_python_mod_offsets(self) -> None:
        """``%`` takes the divisor's sign: ``(b - 5) % 3`` is 1 at b = 0."""
        out = torch.zeros(8, 40, device=DEVICE)
        with self.subTest("nonnegative lhs"):
            x3 = torch.randn(3, 40, device=DEVICE)
            rows = [b % 3 for b in range(8)]
            self._assert_signed(gather_row_mod, (x3, out), x3[rows])
        with self.subTest("negative lhs"):
            x = torch.randn(8, 40, device=DEVICE)
            expected = self._scale_rows(x, [(b - 5) % 3 for b in range(8)])
            self._assert_signed(scale_by_negative_mod, (x, out), expected)

    def test_floordiv_device_grid_count(self) -> None:
        """``(56 + 15) // 16`` columns: 4."""
        x = torch.randn(8, 56, device=DEVICE)
        expected = torch.zeros_like(x)
        expected[:, :4] = x[:, :4] + 1
        self._assert_signed(add_one_device_grid_count, (x,), expected)

    def test_floordiv_constexpr_lhs_keeps_plain_division(self) -> None:
        x = torch.randn(32, 24, device=DEVICE)
        y = torch.randn(64, device=DEVICE)
        bound = add_half_block_row.bind((x, y))
        block = bound.config_spec.default_config().block_sizes[0]
        out, code = self._run(add_half_block_row, (x, y))
        self.assertIn("_BLOCK_SIZE_0 // 2", code)
        self.assertNotIn("static_cast(long, ", code)
        torch.testing.assert_close(out, x + y[block // 2])

    def test_block_size_division_outside_constexpr_exemption(self) -> None:
        """The plain ``//`` is kept only for a ``FloorDiv`` by an integer."""
        x = torch.randn(32, 24, device=DEVICE)
        y = torch.randn(64, device=DEVICE)
        with self.subTest("block size divisor"):
            config = helion.Config(block_sizes=[32, 8])
            self._assert_signed(add_block_ratio_row, (x, y), x + y[4], config)
        with self.subTest("mod by a negative integer"):
            # 16 % -3 is -2 (C's remainder gives 1).
            config = helion.Config(block_sizes=[16, 32])
            self._assert_signed(add_block_mod_negative_row, (x, y), x + y[0], config)

    def test_floordiv_quotient_mixed_with_negative_terms(self) -> None:
        """The quotient is added to terms that are negative for small b, so
        the whole expression, not just the division, must be signed."""
        out = torch.zeros(8, 40, device=DEVICE)
        with self.subTest("index"):
            x16 = torch.randn(16, 40, device=DEVICE)
            rows = [b - 5 + (b + 11) // 2 for b in range(8)]
            self._assert_signed(gather_quotient_minus_row, (x16, out), x16[rows])
        with self.subTest("value"):
            x = torch.randn(8, 40, device=DEVICE)
            expected = self._scale_rows(x, [(b + 1) // 2 - b for b in range(8)])
            self._assert_signed(scale_by_quotient_minus_row, (x, out), expected)

    def test_floordiv_nonnegative_operands_keep_plain_division(self) -> None:
        x = torch.randn(17, 40, device=DEVICE)
        out = torch.zeros(8, 40, device=DEVICE)
        result, code = self._run(gather_square_third, (x, out))
        self.assertIn("offset_0 * offset_0 // 3", code)
        self.assertNotIn("static_cast(long, ", code)
        torch.testing.assert_close(result, x[[(b * b) // 3 for b in range(8)]])


# ---------------------------------------------------------------------------
# Tests – large block / auto-capping
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalLargeBlock(unittest.TestCase):
    """Tests for threadgroup auto-capping when block_size > 1024."""

    def test_large_block_1d(self) -> None:
        """1D kernel with block_size=2048 auto-caps to 1024 threads."""
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        torch.testing.assert_close(large_block_add(x, y), x + y)

    def test_large_block_1d_non_aligned(self) -> None:
        """Non-aligned size with large block still works correctly."""
        x = torch.randn(3000, device=DEVICE)
        y = torch.randn(3000, device=DEVICE)
        torch.testing.assert_close(large_block_add(x, y), x + y)

    def test_explicit_count_past_the_budget_is_capped(self) -> None:
        """An explicit ``num_threads`` past the threadgroup budget is capped to
        it, as an auto count is; lanes walk the rest of the block."""
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        config = helion.Config(block_sizes=[2048], num_threads=[4096], num_warps=4)
        out, code, _ = _run_with_msl(large_block_add, (x, y), config)
        self.assertIn("_block_dims=(128, 1, 1)", code)
        torch.testing.assert_close(out, x + y)

    def test_codegen_lane_loop(self) -> None:
        """Generated MSL must contain a for loop when block_size > 1024."""
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        msl = _get_msl(large_block_add, (x, y))
        self.assertIn("for (int", msl, "lane loop not found in generated MSL")

    def test_large_block_2d(self) -> None:
        """2D kernel with block_sizes=[64,64] (4096 threads) auto-caps."""
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_large_block_3d(self) -> None:
        """3D kernel with block_sizes=[16,16,16] (4096 threads) auto-caps."""
        x = torch.randn(16, 16, 16, device=DEVICE)
        y = torch.randn(16, 16, 16, device=DEVICE)
        torch.testing.assert_close(elementwise_3d(x, y), x + y)


# ---------------------------------------------------------------------------
# Kernel definitions – matmul
# ---------------------------------------------------------------------------


_DEFAULT_MATMUL_CONFIG = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]


@helion.kernel(backend="metal", configs=_DEFAULT_MATMUL_CONFIG)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_MATMUL_CONFIG)
def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _make_matmul_kernel(
    block_sizes: list[int], num_warps: int = 4
) -> helion.Kernel[torch.Tensor]:
    cfg = [helion.Config(block_sizes=block_sizes, num_warps=num_warps)]

    @helion.kernel(backend="metal", configs=cfg)
    def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _k2, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    return _matmul


# ---------------------------------------------------------------------------
# Tests – matmul
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMatmul(unittest.TestCase):
    """Matmul kernels using MPP matmul2d through the standard pipeline."""

    def test_mpp_rejects_paravirtual_mps_device(self) -> None:
        from helion._compiler.metal.metal_jit import _raise_if_mpp_unsupported_device

        with (
            patch.object(
                torch._C,
                "_mps_get_name",
                return_value="Apple Paravirtual device",
            ),
            self.assertRaisesRegex(exc.BackendUnsupported, "paravirtual MPS"),
        ):
            _raise_if_mpp_unsupported_device()

    def test_matmul_basic(self) -> None:
        """Square matmul: 64x64 @ 64x64."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_non_square(self) -> None:
        """Non-square matmul: 128x64 @ 64x256."""
        kernel = _make_matmul_kernel([32, 64, 32])
        x = torch.randn(128, 64, device=DEVICE)
        y = torch.randn(64, 256, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_k_loop(self) -> None:
        """K > TILE_K forces multiple K-loop iterations: 128x512 @ 512x128."""
        x = torch.randn(128, 512, device=DEVICE)
        y = torch.randn(512, 128, device=DEVICE)
        result = matmul_kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_large_square(self) -> None:
        """Larger square matmul: 256x256 @ 256x256."""
        kernel = _make_matmul_kernel([64, 64, 64])
        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_tall_skinny(self) -> None:
        """Tall-skinny: 512x32 @ 32x64."""
        kernel = _make_matmul_kernel([32, 32, 32])
        x = torch.randn(512, 32, device=DEVICE)
        y = torch.randn(32, 64, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_non_divisible_tiles(self) -> None:
        kernel = _make_matmul_kernel([32, 32, 32])
        for m, k, n in [(70, 64, 64), (64, 70, 64), (64, 64, 70), (71, 65, 73)]:
            with self.subTest(shape=(m, k, n)):
                x = torch.randn(m, k, device=DEVICE)
                y = torch.randn(k, n, device=DEVICE)
                result = kernel(x, y)
                expected = torch.mm(x, y)
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_float16(self) -> None:
        """Float16 in/out with float32 accumulation stores via MPP conversion.

        The cooperative store requires matching dtypes, so this mixed pair
        takes a masked per-element convert-and-store loop instead.
        """
        kernel = _make_matmul_kernel([32, 32, 32])
        for m, k, n in [(64, 64, 64), (128, 64, 256), (70, 70, 73)]:
            with self.subTest(shape=(m, k, n)):
                x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
                y = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
                result = kernel(x, y)
                expected = torch.mm(x, y)
                torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_matmul_float16_transposed_operand(self) -> None:
        """Mixed fp32-acc/fp16-out store with a transposed (TN) operand."""
        kernel = _make_matmul_kernel([32, 32, 32])
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y_base = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
        y = y_base.t()
        assert not y.is_contiguous()
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_matmul_float16_float32_output(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_fp32_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def dot_fp32_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        expected = torch.mm(x.float(), y.float())
        for kernel in (matmul_fp32_out, dot_fp32_out):
            with self.subTest(kernel=kernel.name):
                result = kernel(x, y)
                torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        msl = _get_msl(dot_fp32_out, (x, y))
        self.assertRegex(msl, r"get_destination_cooperative_tensor<.*, float>\(\)")
        self.assertIn("tensor<device half", msl)

    def test_matmul_bfloat16(self) -> None:
        """Bfloat16 in/out with float32 accumulation stores via the scalar loop."""
        kernel = _make_matmul_kernel([32, 32, 32])
        for m, k, n in [(64, 64, 64), (70, 70, 73)]:
            with self.subTest(shape=(m, k, n)):
                x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
                y = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
                result = kernel(x, y)
                expected = torch.mm(x.float(), y.float()).to(torch.bfloat16)
                torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_output_dtypes(self) -> None:
        """The converting store covers integer and bool outputs."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32, 32], num_warps=4)],
        )
        def mm_into(x: torch.Tensor, y: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                o[tile_m, tile_n] = acc.to(o.dtype)
            return o

        x = torch.randn(64, 64, device=DEVICE) * 4
        y = torch.randn(64, 64, device=DEVICE) * 4
        for dtype in (torch.int32, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                o = torch.empty(64, 64, device=DEVICE, dtype=dtype)
                torch.testing.assert_close(mm_into(x, y, o), (x @ y).to(dtype))

    def test_matmul_codegen_has_mpp(self) -> None:
        """Generated MSL must contain MPP matmul2d constructs."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        self.assertIn("matmul2d", _get_msl(matmul_kernel, (x, y)))
        msl = _get_msl(dot_kernel, (x, y))
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL")
        self.assertRegex(msl, r"_mpp_op_\d+\.run\(", "MPP run call not found in MSL")
        # The op runs on every SIMD group of the kernel's own threadgroup.
        self.assertIn(
            "execution_simdgroups<(_TG_SIZE_X * _TG_SIZE_Y * _TG_SIZE_Z) / 32>", msl
        )
        self.assertNotIn("auto acc = float(0.0)", msl)

    def test_matmul_with_relu(self) -> None:
        """Matmul with ReLU epilogue fusion."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE) * 0.05
        y = torch.randn(64, 64, device=DEVICE) * 0.05
        result = matmul_relu(x, y)
        expected = torch.mm(x, y).relu()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_supported_epilogues(self) -> None:
        """Each epilogue runs on the cooperative tensor, for ``torch.addmm``
        and ``hl.dot`` alike."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_epilogue(
            x: torch.Tensor,
            y: torch.Tensor,
            epilogue: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = epilogue(acc)
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def dot_epilogue(
            x: torch.Tensor,
            y: torch.Tensor,
            epilogue: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = epilogue(acc)
            return out

        epilogues: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "sigmoid": torch.sigmoid,
            "exp": torch.exp,
            "neg": torch.neg,
            "add": lambda v: v + 1.25,
            "sub": lambda v: v - 0.5,
            "mul": lambda v: v * 0.25,
            "div": lambda v: v / 2.0,
            "chain": lambda v: torch.sigmoid(v.relu() * 0.5 + 1.0),
        }
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        mm = torch.mm(x, y)
        for kind, epilogue in epilogues.items():
            for kernel in (matmul_epilogue, dot_epilogue):
                with self.subTest(kind=kind, kernel=kernel.name):
                    result = kernel(x, y, epilogue)
                    torch.testing.assert_close(
                        result, epilogue(mm), atol=1e-4, rtol=1e-4
                    )
            msl = _get_msl(dot_epilogue, (x, y, epilogue))
            self.assertTrue(any("out[" in loop for loop in _element_loops(msl)))

    @unittest.skip(
        "Flaky numerical mismatch on the metal-m2 CI runner (fixed-config "
        "matmul+aux epilogue); skip on metal pending investigation."
    )
    def test_matmul_aux_tensor_epilogue(self) -> None:
        """An epilogue that reads another tensor runs in the element loop."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_add_aux(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc + z[tile_m, tile_n]
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def dot_add_aux(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc + z[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE) * 0.05
        y = torch.randn(64, 64, device=DEVICE) * 0.05
        z = torch.randn(64, 64, device=DEVICE)
        expected = torch.mm(x, y) + z
        for kernel in (matmul_add_aux, dot_add_aux):
            with self.subTest(kernel=kernel.name):
                result = kernel(x, y, z)
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

        msl = _get_msl(dot_add_aux, (x, y, z))
        self.assertIn("matmul2d", msl)
        self.assertTrue(any("z[" in loop for loop in _element_loops(msl)))

    def test_matmul_epilogue_codegen(self) -> None:
        """ReLU epilogue must appear inside cooperative_tensor iteration."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def dot_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        msl = _get_msl(dot_relu, (x, y))
        self.assertTrue(
            any("c10::metal::max(" in loop for loop in _element_loops(msl)),
            "ReLU not found in the element loop",
        )
        self.assertNotIn("auto acc = float(0.0)", msl)

    def test_mpp_matmul_followed_by_scalar_outer_work(self) -> None:
        """Scalar work after a K loop in the root graph stays out of the
        matmul's element loop, for ``hl.dot`` and ``torch.addmm`` alike: the
        planner jams the 4-warp config's M/N lane loop around the op."""
        cfg = helion.Config(block_sizes=[32, 32, 32], num_warps=4)

        @helion.kernel(backend="metal", configs=[cfg])
        def dot_then_scalar(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            side = torch.empty([m, n], dtype=z.dtype, device=z.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(x.dtype)
                side[tile_m, tile_n] = z[tile_m, tile_n] + 1.0
            return out, side

        @helion.kernel(backend="metal", static_shapes=True)
        def matmul_then_scalar(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            side = torch.empty([m, n], dtype=z.dtype, device=z.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
                side[tile_m, tile_n] = z[tile_m, tile_n] + 1.0
            return out, side

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        z = torch.randn(64, 64, device=DEVICE)
        code = dot_then_scalar.bind((x, y, z)).to_code()
        self.assertIn("_helion_mpp_acc(", code)

        out, side = dot_then_scalar(x, y, z)
        torch.testing.assert_close(out, torch.mm(x, y), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(side, z + 1.0)

        msl = _get_msl(dot_then_scalar, (x, y, z))
        self.assertIn("matmul2d", msl)
        loops = _element_loops(msl)
        self.assertTrue(any("out[" in loop for loop in loops))
        self.assertFalse(any("side[" in loop for loop in loops))
        self.assertIn("side[", msl)

        bound = matmul_then_scalar.bind((x, y, z))
        out, side = bound.compile_config(bound.config_spec.default_config())(x, y, z)
        torch.testing.assert_close(out, torch.mm(x, y), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(side, z + 1.0)
        out, side = bound.compile_config(cfg)(x, y, z)
        torch.testing.assert_close(out, torch.mm(x, y), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(side, z + 1.0)

    def test_matmul_via_hl_dot(self) -> None:
        """``hl.dot`` reaches the same MPP planner as ``torch.addmm``."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_dot(x, y)
        torch.testing.assert_close(result, torch.mm(x, y), atol=1e-4, rtol=1e-4)
        # A column-strided lhs is not a view MPP can read; it is staged.
        strided = torch.randn(64, 128, device=DEVICE)[:, ::2]
        torch.testing.assert_close(
            matmul_dot(strided, y), torch.mm(strided, y), atol=1e-4, rtol=1e-4
        )

        msl = _get_msl(matmul_dot, (x, y))
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL (hl.dot)")
        self.assertRegex(
            msl, r"_mpp_op_\d+\.run\(", "MPP run call not found in MSL (hl.dot)"
        )

    def test_matmul_via_hl_dot_with_relu(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_dot_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_dot_relu(x, y)
        torch.testing.assert_close(result, torch.mm(x, y).relu(), atol=1e-4, rtol=1e-4)

        msl = _get_msl(matmul_dot_relu, (x, y))
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL (hl.dot)")
        self.assertTrue(
            any("c10::metal::max(" in loop for loop in _element_loops(msl)),
            "ReLU not found in the element loop (hl.dot)",
        )

    def test_matmul_bias_epilogue_is_correct(self) -> None:
        """A bias epilogue over every launch shape the configs give."""

        def matmul_bias(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc + bias[None, tile_n]
            return out

        n = 256
        x = torch.randn(n, n, device=DEVICE)
        y = torch.randn(n, n, device=DEVICE)
        bias = torch.randn(n, device=DEVICE)
        expected = x @ y + bias

        for block_m, warps in [(16, 4), (16, 1), (32, 4), (64, 8), (128, 8), (64, 2)]:
            with self.subTest(block_m=block_m, num_warps=warps):
                kernel = helion.kernel(
                    matmul_bias,
                    backend="metal",
                    configs=[
                        helion.Config(block_sizes=[block_m, 16, 16], num_warps=warps)
                    ],
                )
                torch.testing.assert_close(
                    kernel(x, y, bias), expected, rtol=2e-3, atol=2e-3
                )


# ---------------------------------------------------------------------------
# Kernel definitions - scalar hl.dot (outside an MPP region)
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal")
def dot_full_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        out[tile_m, tile_n] = hl.dot(
            x[tile_m, :], y[:, tile_n], out_dtype=torch.float32
        )
    return out


@helion.kernel(backend="metal")
def matmul_full_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0), y.size(1)])
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        out[tile_m, tile_n] = hl.dot(x[tile_m, :], y[:, tile_n])
    return out


@helion.kernel(backend="metal")
def dot_full_k_shifted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Both operands are nonzero past K, so K's surplus threads must add 0."""
    hl.specialize(x.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        out[tile_m, tile_n] = hl.dot(x[tile_m, :] + 1.0, y[:, tile_n] + 1.0)
    return out


@helion.kernel(backend="metal")
def dot_full_k_bf16_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        out[tile_m, tile_n] = hl.dot(
            x[tile_m, :], y[:, tile_n], out_dtype=torch.bfloat16
        )
    return out


@helion.kernel(backend="metal")
def dot_tile_k_shifted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A K loop over computed operands."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(x.size(1)):
            acc = hl.dot(x[tile_m, tile_k] + 1.0, y[tile_k, tile_n] + 1.0, acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal")
def dot_tile_k_shifted_f16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``dot_tile_k_shifted`` with a float16 acc, which keeps it scalar."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(x.size(1)):
            acc = hl.dot(x[tile_m, tile_k] + 1.0, y[tile_k, tile_n] + 1.0, acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal")
def dot_tile_k_f16_acc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A computed lhs with a float16 acc, which keeps it scalar."""
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(x.size(1)):
            acc = hl.dot(x[tile_m, tile_k] * 1.0, y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal")
def dot_recurrence(
    q: torch.Tensor, qt: torch.Tensor, v: torch.Tensor, chunk: int
) -> torch.Tensor:
    """``s = s / 2 + q_c^T (v_c - q_c s)`` over chunks: one dot over a ``:``
    slice and one over the chunk tile, with the state carried."""
    d = hl.specialize(q.size(1))
    chunk = hl.specialize(chunk)
    out = torch.empty([d, v.size(1)], dtype=torch.float32, device=q.device)
    for tile_n in hl.tile(v.size(1)):
        s = hl.zeros([d, tile_n], dtype=torch.float32)
        for tile_c in hl.tile(q.size(0), block_size=chunk):
            r = v[tile_c, tile_n] - hl.dot(q[tile_c, :], s)
            s = s * 0.5
            s = hl.dot(qt[:, tile_c], r, acc=s)
        out[:, tile_n] = s
    return out


@helion.kernel(backend="metal")
def dot_chained(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``sum_n exp(q k^T / 8) v``: one dot's result is the next dot's lhs."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(kt.size(1)):
            p = torch.exp(hl.dot(q[tile_m, :], kt[:, tile_n]) * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_chained_bf16(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_chained`` with the first product rounded to bfloat16, which keeps
    both dots on the scalar contraction."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(kt.size(1)):
            s = hl.dot(q[tile_m, :], kt[:, tile_n], out_dtype=torch.bfloat16)
            p = torch.exp(s * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_full_n(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(1))
    n = hl.specialize(y.size(1))
    out = torch.empty([x.size(0), n], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = hl.dot(x[tile_m, :], y[:, :])
    return out


@helion.kernel(backend="metal")
def dot_carried_state(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    side: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    """``out[b, t] = ((t + 1) * x[b]) @ y[b]`` from a constant-shape state,
    beside an unrelated ``:`` slice of ``w``'s length."""
    m = hl.specialize(x.size(1))
    k = hl.specialize(x.size(2))
    n = hl.specialize(y.size(2))
    out = torch.empty([x.size(0), steps.size(0), m, n], device=x.device)
    for b in hl.grid(x.size(0)):
        side[b, :] = w[b, :] * 2.0
        s = hl.zeros([m, k], dtype=torch.float32)
        for tile_t in hl.tile(steps.size(0), block_size=1):
            s = s + x[b, :, :]
            out[b, tile_t, :, :] = hl.dot(s, y[b, :, :])[None, :, :]
    return out


@helion.kernel(backend="metal")
def dot_state_square(qt: torch.Tensor, q: torch.Tensor, chunk: int) -> torch.Tensor:
    """``(q^T q).sum(0)``: the dot's M and N are one ``:`` block."""
    d = hl.specialize(q.size(1))
    chunk = hl.specialize(chunk)
    out = torch.empty([d], dtype=torch.float32, device=q.device)
    for _ in hl.tile(1):
        s = hl.zeros([d, d], dtype=torch.float32)
        for tile_c in hl.tile(q.size(0), block_size=chunk):
            s = hl.dot(qt[:, tile_c] * 1.0, q[tile_c, :], acc=s)
        out[:] = s.sum(0)
    return out


@helion.kernel(backend="metal")
def dot_k_slice_by_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """lhs K is a ``:`` slice, rhs K a tile spanning all of K."""
    k = hl.specialize(x.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        for tile_k in hl.tile(k, block_size=k):
            out[tile_m, tile_n] = hl.dot(x[tile_m, :] + 1.0, y[tile_k, tile_n] + 1.0)
    return out


@helion.kernel(backend="metal")
def dot_batched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(2))
    out = torch.empty(
        [x.size(0), x.size(1), y.size(2)], dtype=torch.float32, device=x.device
    )
    for tile_b, tile_m, tile_n in hl.tile([x.size(0), x.size(1), y.size(2)]):
        out[tile_b, tile_m, tile_n] = hl.dot(x[tile_b, tile_m, :], y[tile_b, :, tile_n])
    return out


@helion.kernel(backend="metal")
def dot_acc_bf16_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hl.specialize(x.size(1))
    out = torch.empty([x.size(0), y.size(1)], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        out[tile_m, tile_n] = hl.dot(
            x[tile_m, :], y[:, tile_n], acc=acc, out_dtype=torch.bfloat16
        )
    return out


# ---------------------------------------------------------------------------
# Tests - scalar hl.dot (outside an MPP region)
# ---------------------------------------------------------------------------


def _matmul_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.cpu().double() @ y.cpu().double()).float()


def _default_code(kernel: helion.Kernel, args: tuple[object, ...]) -> str:
    bound = kernel.bind(args)
    return bound.to_code(bound.config_spec.default_config())


@_requires_darwin
class TestMetalScalarDot(unittest.TestCase):
    """``hl.dot`` outside an MPP region forms each product on its own thread
    and sums it with K's reduction strategy: the persistent one for a ``:``
    slice, the block one for a tile.  A kernel whose whole-K ``hl.dot``
    calls the MPP planner admits lowers them there instead."""

    def _run(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None = None,
        lowering: str = "dot_product",
    ) -> tuple[torch.Tensor, str]:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        code = bound.to_code(config)
        self.assertIn(lowering, code)
        return bound.compile_config(config)(*args), code

    def test_full_slice_k(self) -> None:
        """A non-power-of-two K = 100 at the default config."""
        x = torch.randn(32, 100, device=DEVICE)
        y = torch.randn(100, 48, device=DEVICE)
        out, _ = self._run(dot_full_k_shifted, (x, y), lowering="_helion_mpp_run(")
        torch.testing.assert_close(
            out, _matmul_reference(x + 1, y + 1).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_full_slice_k_bf16(self) -> None:
        x = torch.randn(32, 256, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 48, device=DEVICE, dtype=torch.bfloat16)
        out, _ = self._run(
            dot_full_k,
            (x, y),
            helion.Config(block_sizes=[16, 16], num_threads=[2, 2]),
            lowering="_helion_mpp_run(",
        )
        torch.testing.assert_close(
            out, _matmul_reference(x, y).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_full_slice_k_bf16_out_dtype(self) -> None:
        x = torch.randn(32, 48, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(48, 48, device=DEVICE, dtype=torch.bfloat16)
        out, _ = self._run(dot_full_k_bf16_out, (x, y))
        torch.testing.assert_close(
            out, _matmul_reference(x, y).to(DEVICE, torch.bfloat16)
        )

    def test_full_slice_k_in_tile_lane_loops(self) -> None:
        """M and N walk lane loops around K's cross-thread sum; every dim has
        a tail.  A float32 result takes MPP at the same config."""
        x = torch.randn(30, 100, device=DEVICE)
        y = torch.randn(100, 44, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16], num_threads=[2, 2])
        out, code = self._run(dot_full_k_bf16_out, (x, y), config)
        self.assertIn("for lane_0 in range(8)", code)
        self.assertIn("for lane_1 in range(8)", code)
        torch.testing.assert_close(
            out, _matmul_reference(x, y).to(DEVICE, torch.bfloat16)
        )
        out, _ = self._run(dot_full_k, (x, y), config, lowering="_helion_mpp_run(")
        torch.testing.assert_close(
            out, _matmul_reference(x, y).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_full_slice_k_in_slice_lane_loop(self) -> None:
        """N is a ``:`` slice of 300 walked in a lane loop around K's sum."""
        x = torch.randn(20, 32, device=DEVICE)
        y = torch.randn(32, 300, device=DEVICE)
        out, code = self._run(
            dot_full_n, (x, y), helion.Config(block_sizes=[16], num_threads=[2])
        )
        self.assertIn("slice_lane_", code)
        torch.testing.assert_close(
            out, _matmul_reference(x, y).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_k_of_one(self) -> None:
        x = torch.randn(24, 1, device=DEVICE)
        y = torch.randn(1, 40, device=DEVICE)
        bound = dot_full_k.bind((x, y))
        config = helion.Config(block_sizes=[16, 16])
        self.assertNotIn("dot_product", bound.to_code(config))
        torch.testing.assert_close(
            bound.compile_config(config)(x, y), _matmul_reference(x, y).to(DEVICE)
        )

    def test_tile_k_with_acc(self) -> None:
        """K's cross-thread sum inside M and N lane loops, with a K tail; at
        N = 44 all three dims have tails.  A float32 acc takes MPP at the
        same configs."""
        for n, num_threads, lanes in ((40, [8, 8, 16], 2), (44, [4, 4, 16], 4)):
            with self.subTest(n=n, num_threads=num_threads):
                config = helion.Config(
                    block_sizes=[16, 16, 16], num_threads=num_threads
                )
                x = torch.randn(50, 72, device=DEVICE)
                y = torch.randn(72, n, device=DEVICE)
                out, code = self._run(dot_tile_k_shifted_f16, (x, y), config)
                self.assertIn(f"for lane_0 in range({lanes})", code)
                self.assertIn(f"for lane_1 in range({lanes})", code)
                # The acc is rounded to fp16 after every K step.
                torch.testing.assert_close(
                    out.float(),
                    _matmul_reference(x + 1, y + 1).to(DEVICE),
                    rtol=1e-2,
                    atol=1e-2,
                )
                out, _ = self._run(
                    dot_tile_k_shifted, (x, y), config, lowering="_helion_mpp_run("
                )
                torch.testing.assert_close(
                    out,
                    _matmul_reference(x + 1, y + 1).to(DEVICE),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_tile_k_f16_acc(self) -> None:
        x = torch.randn(32, 48, device=DEVICE, dtype=torch.float16) / 4
        y = torch.randn(48, 32, device=DEVICE, dtype=torch.float16) / 4
        out, _ = self._run(
            dot_tile_k_f16_acc,
            (x, y),
            helion.Config(block_sizes=[16, 16, 16], num_threads=[8, 8, 16]),
        )
        self.assertEqual(out.dtype, torch.float16)
        # The acc is rounded to fp16 after every K step.
        torch.testing.assert_close(
            out.float(), _matmul_reference(x, y).to(DEVICE), rtol=2e-3, atol=2e-3
        )

    def test_recurrence(self) -> None:
        """A carried state over chunks with a dot over a ``:`` slice and one
        over the chunk tile; N walks a lane loop."""
        t, d, n, chunk = 64, 16, 20, 8
        q = torch.randn(t, d, device=DEVICE) / d**0.5
        v = torch.randn(t, n, device=DEVICE)
        out, code = self._run(
            dot_recurrence,
            (q, q.T.contiguous(), v, chunk),
            helion.Config(block_sizes=[16], num_threads=[8], num_warps=32),
        )
        self.assertIn("for lane_0 in range(2)", code)
        qd, vd = q.cpu().double(), v.cpu().double()
        s = torch.zeros(d, n, dtype=torch.float64)
        for i in range(0, t, chunk):
            qc = qd[i : i + chunk]
            s = s * 0.5 + qc.T @ (vd[i : i + chunk] - qc @ s)
        torch.testing.assert_close(out, s.float().to(DEVICE), rtol=1e-4, atol=1e-4)

    def test_chained(self) -> None:
        """One dot's computed result feeds the next; M walks a lane loop and
        the second dot's tile K has a tail."""
        # p is rounded to bf16 in the kernel and in the reference; about one
        # draw in 400 puts an element of p on a rounding boundary, where one
        # bf16 ulp exceeds the tolerance, so the inputs must not depend on
        # what the test process ran before.
        torch.manual_seed(0)
        m, n, d = 20, 40, 32
        q = torch.randn(m, d, device=DEVICE, dtype=torch.bfloat16) / 4
        kt = torch.randn(d, n, device=DEVICE, dtype=torch.bfloat16) / 4
        v = torch.randn(n, d, device=DEVICE, dtype=torch.bfloat16)
        out, code = self._run(
            dot_chained_bf16,
            (q, kt, v),
            helion.Config(block_sizes=[16, 16], num_threads=[2, 16]),
        )
        self.assertIn("for lane_0 in range(8)", code)
        s = _matmul_reference(q, kt).to(torch.bfloat16).double()
        p = torch.exp(s * 0.125).to(torch.bfloat16)
        torch.testing.assert_close(
            out, _matmul_reference(p, v).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_unlinked_matmul_operand(self) -> None:
        """gdn_fwd_h with its first product computed transposed: the second
        dot reads a permute of the first's result, which no element loop
        stages, so both dots keep the scalar contraction."""
        args = (*_gdn_args(256, 256, 32, torch.bfloat16), _GDN_CHUNK)
        device_args = tuple(
            a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in args
        )
        out, code = self._run(gdn_fwd_h_transposed, device_args)
        self.assertNotIn("_helion_mpp_run(", code)
        torch.testing.assert_close(
            out.cpu().double(), _gdn_reference(*args[:4]), rtol=2e-2, atol=2e-2
        )

    def test_constant_shape_state_operand(self) -> None:
        """``hl.zeros([4, 16])`` beside an unrelated 16-long slice: K has one
        reduction dimension it can be."""
        x = torch.randn(2, 4, 16, device=DEVICE)
        y = torch.randn(2, 16, 8, device=DEVICE)
        w = torch.randn(2, 16, device=DEVICE)
        steps = torch.zeros(3, device=DEVICE)
        out, _ = self._run(dot_carried_state, (x, y, w, torch.zeros_like(w), steps))
        expected = torch.stack(
            [torch.bmm((i + 1) * x.cpu().double(), y.cpu().double()) for i in range(3)],
            1,
        ).float()
        torch.testing.assert_close(out, expected.to(DEVICE), rtol=1e-4, atol=1e-4)

    def test_padded_constant_shape_rejected(self) -> None:
        """``hl.zeros([4, 12])`` shows K as 16, the length of an unrelated
        slice of ``w``, so w's block looks like K's padding; w's 16 real
        elements rule that out."""
        x = torch.randn(2, 4, 12, device=DEVICE)
        y = torch.randn(2, 12, 8, device=DEVICE)
        w = torch.randn(2, 16, device=DEVICE)
        args = (x, y, w, torch.zeros_like(w), torch.zeros(3, device=DEVICE))
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction dim \d+ pads dim \d+ to 16 elements, but w dim 1 has 16 "
            r"real elements",
        ):
            _default_code(dot_carried_state, args)

    def test_m_and_n_on_one_block_rejected(self) -> None:
        q = torch.randn(64, 16, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "indexes two of M, K, N with one block"
        ):
            dot_state_square.bind((q.T.contiguous(), q, 16)).to_code(
                helion.Config(block_sizes=[1])
            )

    def test_k_slice_against_k_tile_rejected(self) -> None:
        x = torch.randn(8, 16, device=DEVICE)
        y = torch.randn(16, 8, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "contracts lhs and rhs over different blocks"
        ):
            dot_k_slice_by_tile.bind((x, y)).to_code(
                helion.Config(block_sizes=[16, 16], num_threads=[4, 4])
            )

    def test_batched_rejected(self) -> None:
        x = torch.randn(2, 16, 32, device=DEVICE)
        y = torch.randn(2, 32, 16, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "batched"):
            dot_batched.bind((x, y)).to_code(helion.Config(block_sizes=[1, 16, 16]))

    def test_int8_rejected(self) -> None:
        x = torch.randint(-8, 8, (16, 32), device=DEVICE, dtype=torch.int8)
        y = torch.randint(-8, 8, (32, 16), device=DEVICE, dtype=torch.int8)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "only float16, bfloat16 and float32"
        ):
            _default_code(matmul_full_k, (x, y))

    def test_acc_with_other_out_dtype_rejected(self) -> None:
        x = torch.randn(16, 32, device=DEVICE)
        y = torch.randn(32, 16, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "pass an out_dtype equal to the acc's dtype"
        ):
            _default_code(dot_acc_bf16_out, (x, y))


# ---------------------------------------------------------------------------
# Kernel definitions - reductions
# ---------------------------------------------------------------------------

_REDUCTION_CONFIG = [helion.Config(block_sizes=[16])]


@helion.kernel(backend="metal", static_shapes=True)
def tile_inner_reductions(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum/amax/amin/argmax over the inner block of a two-block nested tile."""
    o, a, d = w.shape
    total = torch.empty([o, a], dtype=torch.float32, device=w.device)
    best = torch.empty([o, a], dtype=torch.float32, device=w.device)
    least = torch.empty([o, a], dtype=torch.float32, device=w.device)
    where = torch.empty([o, a], dtype=torch.int64, device=w.device)
    for tile_o in hl.tile(o):
        for tile_a, tile_d in hl.tile([a, d]):
            values = w[tile_o, tile_a, tile_d].to(torch.float32)
            total[tile_o, tile_a] = torch.sum(values, dim=-1)
            best[tile_o, tile_a] = torch.amax(values, dim=-1)
            least[tile_o, tile_a] = torch.amin(values, dim=-1)
            where[tile_o, tile_a] = torch.argmax(values, dim=-1)
    return total, best, least, where


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_amax(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_amin(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_prod(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.prod(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_mean(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.mean(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_argmax(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.argmax(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_argmin(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.argmin(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        values = x[tile_m, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp_v = torch.exp(values - amax)
        out[tile_m, :] = exp_v / torch.sum(exp_v, dim=1, keepdim=True)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        mean_sq = torch.mean(x_tile * x_tile, dim=-1, keepdim=True)
        normalized = x_tile * torch.rsqrt(mean_sq + 1e-5)
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(x.dtype)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def layer_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean = torch.sum(acc, dim=-1, keepdim=True) / n
        centered = acc - mean
        var = torch.sum(centered * centered, dim=-1, keepdim=True) / n
        normalized = centered * torch.rsqrt(var + 1e-5)
        out[tile_m, :] = (normalized * weight[:] + bias[:]).to(x.dtype)
    return out


@helion.kernel(
    backend="metal",
    configs=_REDUCTION_CONFIG,
    ignore_warnings=[exc.TensorOperationInWrapper],
)
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        flat_indices = tile_n.index * v + labels[tile_n]
        correct = hl.load(logits_flat, [flat_indices]).to(torch.float32)
        row = logits[tile_n, :].to(torch.float32)
        row_max = torch.amax(row, dim=-1)
        shifted = row - row_max[:, None]
        logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
        losses[tile_n] = logsumexp + row_max - correct
    return losses


@helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[4, 2])])
def sum_in_device_loop(x: torch.Tensor) -> torch.Tensor:
    """Reduction inside a device loop, so its scratch buffer is reused.

    The inner ``hl.tile(k)`` becomes a serial device loop, and the reduction
    over the trailing dim runs once per iteration against the same
    ``threadgroup`` scratch array.
    """
    m, k, _ = x.shape
    out = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        for tile_k in hl.tile(k):
            out[tile_m, tile_k] = x[tile_m, tile_k, :].to(torch.float32).sum(-1)
    return out


# ---------------------------------------------------------------------------
# Reduction tests
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalReductions(unittest.TestCase):
    """Reductions across all three Metal cross-thread reduction tiers.

    The tier is chosen by the reduction's thread span (see
    ``helion/_compiler/metal/reduction.py``):

    ==============  ==========================================================
    span            ``helion_red::group_*`` reduces with
    ==============  ==========================================================
    ``< 32``        ``seg_*``, a shuffle butterfly over the segment
    ``== 32``       ``simd_*``
    ``> 32``        ``tg_*`` + a ``threadgroup`` scratch buffer
    ==============  ==========================================================

    Spans above one threadgroup (1024) are rolled into a ``for`` loop over the
    reduction dim by ``reduction_loops``.
    """

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        expected: torch.Tensor,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> None:
        result = kernel(*args)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # -- span tiers ---------------------------------------------------------

    def test_sum_sub_simdgroup_span(self) -> None:
        """n=8: several reduction groups share one SIMD group."""
        x = torch.randn(64, 8, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_exact_simdgroup_span(self) -> None:
        x = torch.randn(64, 32, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_multi_simdgroup_span(self) -> None:
        """n=256: 8 SIMD groups per reduction, via threadgroup scratch."""
        x = torch.randn(64, 256, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_full_threadgroup_span(self) -> None:
        x = torch.randn(32, 1024, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_non_power_of_2(self) -> None:
        """The padded lanes must be masked to the identity."""
        x = torch.randn(64, 1000, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_rolled_reduction(self) -> None:
        """n > 1024 rolls into a loop with a per-thread accumulator."""
        x = torch.randn(64, 4096, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_deeply_rolled_reduction(self) -> None:
        x = torch.randn(4, 65536, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-2, atol=1e-2)

    def test_sum_single_row(self) -> None:
        """One row: the whole threadgroup cooperates on a single reduction."""
        x = torch.randn(1, 8192, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    # -- reduction kinds ----------------------------------------------------

    def test_amax(self) -> None:
        x = torch.randn(33, 257, device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    def test_amax_propagates_nan(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        x[3, 17] = float("nan")
        result = row_amax(x)
        self.assertTrue(torch.isnan(result[3]).item())
        torch.testing.assert_close(result[:3], x[:3].amax(-1))

    def test_amin(self) -> None:
        x = torch.randn(33, 100, device=DEVICE)
        self._check(row_amin, (x,), x.amin(-1))

    def test_prod(self) -> None:
        x = torch.rand(16, 16, device=DEVICE) + 0.5
        self._check(row_prod, (x,), x.prod(-1))

    def test_mean(self) -> None:
        x = torch.randn(32, 512, device=DEVICE)
        self._check(row_mean, (x,), x.mean(-1))

    def test_argmax_sub_simdgroup_span(self) -> None:
        x = torch.randn(32, 16, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_exact_simdgroup_span(self) -> None:
        x = torch.randn(32, 32, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_multi_simdgroup_span(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_ties_take_lowest_index(self) -> None:
        x = torch.zeros(4, 64, device=DEVICE)
        x[:, 5] = 1.0
        x[:, 40] = 1.0
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_rolled_argreductions_take_lowest_global_index(self) -> None:
        """The final cross-lane reduction must compare carried indices.

        With a 4096-element row, lanes locally reduce four chunks.  Indices 1
        and 1024 are therefore carried by different lanes in reverse index
        order; selecting the first winning lane incorrectly returns 1024.
        """
        expected = torch.ones(4, dtype=torch.int64, device=DEVICE)
        for kernel, name, extremum in (
            (row_argmax, "argmax", 1.0),
            (row_argmin, "argmin", -1.0),
        ):
            with self.subTest(kernel=name, case="tie"):
                x = torch.zeros(4, 4096, device=DEVICE)
                x[:, 1] = extremum
                x[:, 1024] = extremum
                self._check(kernel, (x,), expected)

            with self.subTest(kernel=name, case="nan"):
                x = torch.zeros(4, 4096, device=DEVICE)
                x[:, 1] = float("nan")
                x[:, 1024] = float("nan")
                self._check(kernel, (x,), expected)

    def test_argmin(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(row_argmin, (x,), x.argmin(-1))

    # -- dtypes -------------------------------------------------------------

    def test_sum_float16(self) -> None:
        # Helion and torch both accumulate in fp32 and round once, so the only
        # slack needed is a rounding boundary flipped by summation order.
        x = torch.randn(32, 128, dtype=torch.float16, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=5e-3, atol=5e-3)

    def test_sum_bfloat16(self) -> None:
        x = torch.randn(32, 128, dtype=torch.bfloat16, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=2e-2, atol=2e-2)

    def test_sum_int32(self) -> None:
        x = torch.randint(-50, 50, (32, 128), dtype=torch.int32, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1).to(torch.int32))

    def test_sum_int64(self) -> None:
        """Metal has no 64-bit SIMD shuffle; c10::metal works around it."""
        x = torch.randint(-50, 50, (32, 128), dtype=torch.int64, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_amax_int64(self) -> None:
        x = torch.randint(-(10**12), 10**12, (32, 128), device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    def test_int64_extrema_are_not_polluted_by_a_zero_fill(self) -> None:
        """An int64 extremum must not be clamped towards zero.

        The cross-SIMD-group tier re-reduces the per-SIMD-group partials, and
        ``c10::metal::threadgroup_*`` does that with only ``span / 32`` lanes
        live.  Metal has no 64-bit SIMD reduction, so c10 emulates ``long``
        with ``simd_shuffle_and_fill_down`` and a fill of 0 -- which is the
        identity for sum and for nothing else, so every one of these returned
        0.  Sign-uniform data is what makes the fill visible; a two-sided
        range hides it, since the true extremum is then almost surely on the
        same side of zero as the fill.
        """
        for n in (64, 100, 128, 256, 512, 1024):
            with self.subTest(n=n):
                neg = torch.randint(-(10**12), -1, (8, n), device=DEVICE)
                self._check(row_amax, (neg,), neg.amax(-1))
                pos = torch.randint(1, 10**12, (8, n), device=DEVICE)
                self._check(row_amin, (pos,), pos.amin(-1))
                ones = torch.ones(8, n, dtype=torch.int64, device=DEVICE)
                self._check(row_prod, (ones,), ones.prod(-1))

    def test_rolled_int64_amax(self) -> None:
        """A rolled int64 reduction emits ``iinfo(int64).min`` as its identity."""
        x = torch.randint(-(10**12), -1, (8, 4096), device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    # -- composite kernels --------------------------------------------------

    def test_softmax(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(softmax_decomposed, (x,), torch.softmax(x, dim=-1))

    def test_softmax_rolled(self) -> None:
        x = torch.randn(8, 4096, device=DEVICE)
        self._check(softmax_decomposed, (x,), torch.softmax(x, dim=-1))

    def test_rms_norm(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        expected = torch.nn.functional.rms_norm(x, (256,), weight, eps=1e-5)
        self._check(rms_norm_fwd, (x, weight), expected, rtol=1e-3, atol=1e-3)

    def test_layer_norm_two_reductions(self) -> None:
        """Mean and variance must not share a threadgroup scratch buffer."""
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        bias = torch.randn(256, device=DEVICE)
        expected = torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-5)
        for _ in range(20):
            # The historical scratch-aliasing race was non-deterministic.
            self._check(
                layer_norm_fwd, (x, weight, bias), expected, rtol=1e-3, atol=1e-3
            )

    def test_cross_entropy(self) -> None:
        logits = torch.randn(64, 512, device=DEVICE)
        labels = torch.randint(0, 512, (64,), device=DEVICE)
        expected = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="none"
        )
        self._check(cross_entropy, (logits, labels), expected, rtol=1e-3, atol=1e-3)

    def test_reduction_in_device_loop(self) -> None:
        """Scratch reuse across device-loop iterations needs a leading barrier."""
        x = torch.randn(16, 6, 128, device=DEVICE)
        for _ in range(20):
            self._check(sum_in_device_loop, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    # -- numeric edge cases across every tier -------------------------------

    _TIERS = (
        (8, "segmented"),
        (32, "simd_group"),
        (256, "threadgroup"),
        (4096, "rolled"),
    )

    def test_nan_and_inf_across_tiers(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                x = torch.randn(16, n, device=DEVICE)
                x[0, n // 2] = float("nan")
                for kernel, ref in (
                    (row_sum, x.sum(-1)),
                    (row_amax, x.amax(-1)),
                    (row_amin, x.amin(-1)),
                    (row_argmax, x.argmax(-1)),
                    (row_argmin, x.argmin(-1)),
                ):
                    result = kernel(x)
                    torch.testing.assert_close(
                        result, ref, rtol=1e-3, atol=1e-3, equal_nan=True
                    )
                # +inf and -inf in the same row must sum to NaN, not cancel.
                z = torch.full((16, n), float("inf"), device=DEVICE)
                z[:, 1] = -float("inf")
                torch.testing.assert_close(row_sum(z), z.sum(-1), equal_nan=True)

    def test_nan_and_all_inf_rows_across_tg_sizes(self) -> None:
        # The cross-SIMD-group tier folds up to 4 partials directly and
        # SIMD-reduces more, padding unused lanes with an identity that must
        # not be the finite extreme (-FLT_MAX would win the max of an
        # all-(-inf) row).  Spans 64..1024 cover 2, 8, 16 and 32 partials.
        for n in (8, 32, 64, 256, 512, 1024):
            with self.subTest(n=n):
                x = torch.randn(16, n, device=DEVICE)
                x[1, n // 2] = float("nan")
                x[2, :] = float("-inf")
                x[3, :] = float("inf")
                torch.testing.assert_close(row_amax(x), x.amax(-1), equal_nan=True)
                torch.testing.assert_close(row_amin(x), x.amin(-1), equal_nan=True)
                torch.testing.assert_close(
                    row_sum(x), x.sum(-1), rtol=1e-4, atol=1e-4, equal_nan=True
                )

    def test_argreductions_over_all_inf_rows_across_tg_sizes(self) -> None:
        # The value pass of a cross-SIMD-group argreduction is a tg_max/tg_min
        # whose padded slots must not beat an all-inf row's value.
        for n in (64, 256, 512, 1024):
            with self.subTest(n=n):
                x = torch.randn(16, n, device=DEVICE)
                x[2, :] = float("-inf")
                x[3, :] = float("inf")
                torch.testing.assert_close(row_argmax(x), x.argmax(-1))
                torch.testing.assert_close(row_argmin(x), x.argmin(-1))

    def test_argmax_edge_positions_across_tiers(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                # All-equal: torch returns the lowest index.
                flat = torch.zeros(16, n, device=DEVICE)
                torch.testing.assert_close(row_argmax(flat), flat.argmax(-1))
                # The extremum in the last lane of the last SIMD group.
                last = torch.zeros(16, n, device=DEVICE)
                last[:, n - 1] = 1.0
                torch.testing.assert_close(row_argmax(last), last.argmax(-1))

    def test_non_power_of_two_extents_at_tier_boundaries(self) -> None:
        for n in (7, 31, 33, 63, 65, 1023, 1025, 1500):
            with self.subTest(n=n):
                x = torch.randn(16, n, device=DEVICE)
                torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(row_amax(x), x.amax(-1))

    def test_int32_sum_wraps_like_torch(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                x = torch.full((16, n), 2**20, dtype=torch.int32, device=DEVICE)
                torch.testing.assert_close(row_sum(x), x.sum(-1).to(torch.int32))

    # -- codegen ------------------------------------------------------------

    @_full_threadgroup
    def test_codegen_uses_c10_metal_helpers(self) -> None:
        """Cross-SIMD-group reductions go through Inductor's reduction_utils.

        Both stages call ``c10::metal::simd_*``; the cross-SIMD-group combine
        around them is Helion's, because c10's runs its second stage with a
        partially populated SIMD group (see ``msl_reduction``).
        """
        x = torch.randn(64, 256, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertIn("#include <c10/metal/reduction_utils.h>", msl)
        self.assertIn("::c10::metal::simd_##NAME", msl)
        self.assertNotIn("::c10::metal::threadgroup_", msl)
        # A span of 256 takes the cross-SIMD-group tier and its scratch.
        [(op, path, partials, _, span)] = _group_reductions(msl)
        self.assertEqual((op, path, span), ("sum", "contiguous", 256))
        self.assertRegex(msl, rf"threadgroup float {partials}\[\d+\];")
        self.assertGreater(_static_threadgroup_memory(msl), 0)
        self.assertNotIn("[[simdgroup_index_in_threadgroup]]", msl)

    @_full_threadgroup
    def test_codegen_simdgroup_span_needs_no_shared_memory(self) -> None:
        x = torch.randn(64, 32, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertEqual(
            [(op, span) for op, _, _, _, span in _group_reductions(msl)],
            [("sum", 32)],
        )
        self.assertEqual(_static_threadgroup_memory(msl), 0)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_codegen_sub_simdgroup_span_uses_segmented_butterfly(self) -> None:
        x = torch.randn(64, 8, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertEqual(
            [(op, span) for op, _, _, _, span in _group_reductions(msl)],
            [("sum", 8)],
        )
        self.assertEqual(_static_threadgroup_memory(msl), 0)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_codegen_two_reductions_get_distinct_scratch(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        bias = torch.randn(256, device=DEVICE)
        msl = _get_msl(layer_norm_fwd, (x, weight, bias))
        partials = [partials for _, _, partials, _, _ in _group_reductions(msl)]
        self.assertEqual(len(partials), 2)
        self.assertEqual(len(set(partials)), 2)

    def test_codegen_elementwise_kernel_has_no_reduction_preamble(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertNotIn("reduction_utils.h", msl)
        self.assertNotIn("helion_red", msl)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_reduction_owns_thread_axis_zero(self) -> None:
        """A reduction strategy owns thread axis 0 at exactly its span.

        That makes its groups runs of consecutive lanes that fill the axis,
        the cheapest group path.
        """
        for block, n in itertools.product([1, 2, 8, 16, 64, 256], [64, 256, 1024]):
            with self.subTest(block=block, n=n):
                kernel = helion.kernel(
                    row_sum.fn,
                    backend="metal",
                    configs=[helion.Config(block_sizes=[block])],
                )
                msl = _get_msl(kernel, (torch.randn(256, n, device=DEVICE),))
                dims = re.search(
                    r"required_threads_per_threadgroup\((\d+), (\d+), (\d+)\)", msl
                )
                self.assertIsNotNone(dims)
                assert dims is not None
                [(_, path, _, axis, span)] = _group_reductions(msl)
                self.assertEqual(axis, 0, "reduction must own thread axis 0")
                self.assertEqual(
                    int(dims.group(1)),
                    span,
                    "thread-block dim 0 must equal the reduction span",
                )
                self.assertEqual(path, "contiguous")

    def test_narrow_integer_and_bool_sums(self) -> None:
        """Sub-32-bit ints and bool must reduce in a promoted accumulator.

        ``c10::metal::threadgroup_sum`` takes ``threadgroup opmath_t<T>*``, and
        c10 promotes char/short/uchar to int; passing the storage type made the
        shader fail to compile above the SIMD width.
        """
        for dtype in (torch.int8, torch.int16, torch.uint8, torch.bool):
            for n in (16, 32, 64, 256, 4096):  # 4096 exercises the rolled path
                with self.subTest(dtype=dtype, n=n):
                    if dtype is torch.bool:
                        x = torch.randint(0, 2, (4, n), device=DEVICE).bool()
                    elif dtype is torch.uint8:
                        x = torch.randint(0, 4, (4, n), dtype=dtype, device=DEVICE)
                    else:
                        x = torch.randint(-3, 3, (4, n), dtype=dtype, device=DEVICE)
                    result = row_sum(x)
                    torch.testing.assert_close(result, x.sum(-1).to(result.dtype))

    def test_narrow_integer_amax(self) -> None:
        for dtype in (torch.int8, torch.int16, torch.uint8):
            for n in (32, 256):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randint(0, 100, (4, n), dtype=dtype, device=DEVICE)
                    torch.testing.assert_close(row_amax(x), x.amax(-1))

    # -- shapes the aliasing and budget guards must not reject ---------------

    def test_reduction_under_dynamic_shapes(self) -> None:
        """Every reduction kernel has a full slice, so the guard sees a SymInt.

        ``_reject_aliased_slice_dims`` used to key a dict on ``tensor.size(dim)``,
        which is unhashable under dynamic shapes -- so any Metal kernel with a
        ``:`` failed to compile at all.

        A rolled reduction steps over the runtime extent, so one compiled
        kernel covers rows longer than the first call's.  A persistent one
        would fix its threads and lanes from the first call and drop a longer
        row's tail, so it is rejected.
        """

        @helion.kernel(
            backend="metal",
            static_shapes=False,
            configs=[helion.Config(block_sizes=[4], reduction_loops=[64])],
        )
        def dyn_row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        for shape in ((48, 96), (64, 128), (16, 1000)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE)
                torch.testing.assert_close(
                    dyn_row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3
                )
        persistent = helion.kernel(
            dyn_row_sum.fn,
            backend="metal",
            static_shapes=False,
            configs=[helion.Config(block_sizes=[4])],
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            persistent(torch.randn(48, 96, device=DEVICE))

    def test_specialized_dim_walks_lanes(self) -> None:
        """A dim ``hl.specialize`` pins under dynamic shapes is as static as a
        literal: wider than num_warps' threads, it walks lanes on them, as it
        does under static shapes."""

        @helion.kernel(backend="metal", static_shapes=False)
        def specialized_row_sum(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            hl.specialize(x.size(1))
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(8, 256, device=DEVICE)
        config = helion.Config(block_sizes=[1], num_warps=4, reduction_loops=[None])
        bound = specialized_row_sum.bind((x,))
        code = bound.to_code(config)
        self.assertIn("_block_dims=(128, 1, 1)", code)
        self.assertRegex(code, r"for slice_lane_\d+ in range\(2\)")
        torch.testing.assert_close(
            bound.compile_config(config)(x), x.sum(-1), rtol=3e-3, atol=3e-3
        )

    def test_dynamic_slice_dim_rejects(self) -> None:
        """A ``:`` slice with no reduction takes the same threads and lanes."""

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def dyn_scale_rows(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = x[tile_m, :] * 2
            return out

        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            dyn_scale_rows(torch.randn(16, 100, device=DEVICE))

    def test_repeated_size_one_slices_are_allowed(self) -> None:
        """Length-1 full slices are indexed by a constant, so they cannot alias.

        They never reach ``allocate_reduction_dimension`` and claim no thread
        axis, but the equal-length check counted them anyway and rejected a
        kernel that compiles correctly.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[16])])
        def scale(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :] = x[tile_m, :, :] * 2.0
            return out

        x = torch.randn(64, 1, 1, device=DEVICE)
        torch.testing.assert_close(scale(x), x * 2.0)

    def test_explicit_block_size_with_a_reduction_walks_lanes(self) -> None:
        """A tile axis with no ``num_threads`` knob must not crash the budget pass.

        ``hl.tile(n, block_size=<int>)`` allocates no ``NumThreadsSpec``, so the
        reduction thread budget has nowhere to write a cap.  The budget pass
        must charge it rather than index the spec map, which would raise a
        bare ``KeyError``.
        16 rows x a 128-wide reduction exceed a threadgroup, so the reduction
        folds the lanes its threads cannot cover: 16 on 8 threads, beside the
        16 explicit rows and the inner tile's 8 threads.
        """
        x = torch.randn(64, 128, device=DEVICE)
        kernel = helion.kernel(
            tile16_rowsum.fn, backend="metal", autotune_effort="none"
        )
        self.assertRegex(
            kernel.bind((x,)).to_code(), r"for slice_lane_\w+ in range\(16\)"
        )
        torch.testing.assert_close(kernel(x), x.sum(-1), rtol=1e-4, atol=1e-4)

    # -- unsupported --------------------------------------------------------

    def test_two_equal_size_reduction_dims_are_rejected(self) -> None:
        """Equal-size reduction dims share a block id, hence a thread index.

        ``allocate_reduction_dimension`` caches by size, so both trailing axes
        of an ``[m, n, n]`` tensor get one index variable.  A tile-level backend
        keeps them apart by broadcasting; Metal would collapse them onto one
        thread and reduce the diagonal, so the kernel must be rejected.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def two_reduce_dims(x: torch.Tensor) -> torch.Tensor:
            m, _, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :, :].sum(-1).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "repeats a block"):
            two_reduce_dims(torch.randn(3, 8, 8, device=DEVICE))

    def test_broadcast_aliased_reduction_is_rejected(self) -> None:
        """Aliasing built by broadcasting two separately-loaded values.

        No single access repeats the block; only the broadcast value does.
        Left unguarded these computed ``dot(a[m], b)`` and broadcast it across
        the whole output row.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def bcast_two_loads(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = (a[tile_m, :, None] * b[None, None, :]).sum(-1)
            return out

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def self_outer(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                out[tile_m] = (row[:, :, None] * row[:, None, :]).sum(-1).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "repeats a block"):
            bcast_two_loads(
                torch.randn(3, 8, device=DEVICE), torch.randn(8, device=DEVICE)
            )
        with self.assertRaisesRegex(exc.BackendUnsupported, "repeats a block"):
            self_outer(torch.randn(5, 8, device=DEVICE))

    def test_two_unequal_reduction_dims_are_correct(self) -> None:
        """Distinct sizes give distinct block ids and disjoint thread axes."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def two_reduce_dims(x: torch.Tensor) -> torch.Tensor:
            m, _, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :, :].sum(-1).sum(-1)
            return out

        x = torch.randn(3, 8, 16, device=DEVICE)
        torch.testing.assert_close(two_reduce_dims(x), x.sum(-1).sum(-1))

    def test_two_slice_dims_without_a_reduction(self) -> None:
        """Two ``:`` slices with no reduction op index disjoint axes."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1, 1])])
        def two_slice_scale(q: torch.Tensor) -> torch.Tensor:
            b, h, s, d = q.shape
            out = torch.empty_like(q)
            for tile_b, tile_s in hl.tile([b, s]):
                out[tile_b, :, tile_s, :] = q[tile_b, :, tile_s, :] * 2
            return out

        q = torch.randn(1, 4, 8, 16, device=DEVICE)
        torch.testing.assert_close(two_slice_scale(q), q * 2)

    def test_rope(self) -> None:
        """``examples/rope.py``: heads and the two halves of the head dim.

        At 32 heads times 64 half-dim positions the threadgroup overflows, so
        the slice-only dims walk their positions in lane loops.
        """
        kernel = helion.kernel(
            rope_fwd.fn,
            backend="metal",
            configs=[helion.Config(block_sizes=[1, 32], num_warps=4)],
        )
        for shape in ((1, 8, 64, 64), (1, 4, 37, 64), (1, 32, 64, 128)):
            with self.subTest(shape=shape):
                q = torch.randn(shape, device=DEVICE, dtype=torch.float16)
                k = torch.randn_like(q)
                cos = torch.randn(shape[2:], device=DEVICE, dtype=torch.float16)
                sin = torch.randn_like(cos)
                if shape[1] == 32:
                    code = kernel.bind((q, k, cos, sin)).to_code()
                    self.assertIn("slice_lane_", code)
                for out, ref in zip(
                    kernel(q, k, cos, sin), rope_pytorch(q, k, cos, sin), strict=True
                ):
                    torch.testing.assert_close(out, ref)

    def test_slice_lane_loop_masks_odd_tail(self) -> None:
        """A lane loop over a non-dividing extent (44) masks its tail per lane."""

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(
                    block_sizes=[4],
                    num_threads=[1],
                    num_warps=32,
                    reduction_loops=[None],
                )
            ],
        )
        def wide_tail(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, :, :] * 2 + 1
            return out

        x = torch.randn(4, 128, 44, device=DEVICE)
        # The rows on one thread leave 128 x 8 threads: 44 positions on 8
        # threads take 6 lanes, not 64 // 8 = 8.
        self.assertRegex(
            wide_tail.bind((x,)).to_code(), r"for slice_lane_\w+ in range\(6\)"
        )
        torch.testing.assert_close(wide_tail(x), x * 2 + 1)

    def test_dynamic_starved_slice_dim_rejects(self) -> None:
        """Lane counts come from the static extent; a symbolic one rejects."""

        @helion.kernel(backend="metal", static_shapes=False, autotune_effort="none")
        def dynamic_wide_tail(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, :, :] * 2 + 1
            return out

        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            dynamic_wide_tail(torch.randn(4, 128, 44, device=DEVICE))

    def test_starved_persistent_reduction_folds_lanes(self) -> None:
        """A real reduction starved of threads folds each thread's lanes
        before combining across threads."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[8])])
        def row_sum_persistent(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(8, 256, device=DEVICE)
        torch.testing.assert_close(
            row_sum_persistent(x), x.sum(-1), rtol=3e-3, atol=3e-3
        )
        with patch.object(
            MetalBackend,
            "adjust_reduction_thread_count",
            lambda self, requested, existing: min(requested, 32),
        ):
            starved = helion.kernel(
                row_sum_persistent.fn,
                backend="metal",
                configs=[helion.Config(block_sizes=[8])],
            )
            self.assertRegex(
                starved.bind((x,)).to_code(), r"for slice_lane_\w+ in range\(8\)"
            )
            torch.testing.assert_close(starved(x), x.sum(-1), rtol=3e-3, atol=3e-3)

    def test_reduced_tile_takes_lowest_thread_axis(self) -> None:
        """A reduction over a nested tile puts that tile on ``tid[0]``.

        Loop nesting alone gives the outer grid ``tid[0]`` (as in
        ``softmax_two_pass``) and leaves the reduced tile 32 lanes apart; the
        reduced-tile axis plan restores contiguous lanes, so the reduction is
        one SIMD-group reduction without scratch.
        """

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32], num_warps=32)],
        )
        def tile_row_sum(w: torch.Tensor) -> torch.Tensor:
            o, d = w.shape
            out = torch.empty([o], dtype=torch.float32, device=w.device)
            for tile_o in hl.tile(o):
                acc = hl.zeros([tile_o], dtype=torch.float32)
                for tile_d in hl.tile(d):
                    acc = acc + torch.sum(w[tile_o, tile_d].to(torch.float32), dim=-1)
                out[tile_o] = acc
            return out

        w = torch.randn(512, 128, device=DEVICE)
        msl = _get_msl(tile_row_sum, (w,))
        self.assertEqual(
            [(path, axis, span) for _, path, _, axis, span in _group_reductions(msl)],
            [("contiguous", 0, 32)],
        )
        self.assertEqual(_static_threadgroup_memory(msl), 0)
        torch.testing.assert_close(tile_row_sum(w), w.sum(-1))

    # Block sizes [o, a, d] for ``tile_inner_reductions``: ``a`` holds
    # ``tid[0]`` inside the reduced strategy, so the ``d`` group is ``block_a``
    # lanes apart; ``block_o > 1`` adds an axis above the group.
    _STRIDED_CASES = (
        [2, 4, 8],  # one SIMD group, no scratch
        [2, 8, 16],
        [2, 2, 32],
        [1, 4, 64],
        [2, 2, 64],
        [1, 128, 8],  # 1024 threads, one row per SIMD lane
        [2, 64, 8],
    )

    def _strided_reductions(
        self, block_sizes: list[int], w: torch.Tensor
    ) -> tuple[str, tuple[torch.Tensor, ...]]:
        kernel = helion.kernel(
            tile_inner_reductions.fn,
            backend="metal",
            static_shapes=True,
            configs=[helion.Config(block_sizes=block_sizes, num_warps=32)],
        )
        return _get_msl(kernel, (w,)), kernel(w)

    def test_strided_tile_reductions_are_correct(self) -> None:
        for block_sizes in self._STRIDED_CASES:
            with self.subTest(block_sizes=block_sizes):
                # 12 rows of ``a`` leave a masked tail tile for block_a = 8.
                w = torch.randn(6, 12, block_sizes[2], device=DEVICE)
                msl, (total, best, least, where) = self._strided_reductions(
                    block_sizes, w
                )
                # The ``d`` group is on thread axis 1, spanning block_d lanes.
                self.assertEqual(
                    {(p, axis, span) for _, p, _, axis, span in _group_reductions(msl)},
                    {("strided", 1, block_sizes[2])},
                )
                torch.testing.assert_close(total, w.sum(-1), rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(best, w.amax(-1))
                torch.testing.assert_close(least, w.amin(-1))
                torch.testing.assert_close(where, w.argmax(-1))

    def test_strided_tile_reductions_propagate_nan_and_inf(self) -> None:
        for block_sizes in self._STRIDED_CASES:
            with self.subTest(block_sizes=block_sizes):
                w = torch.randn(6, 12, block_sizes[2], device=DEVICE)
                w[1, 3, block_sizes[2] // 2] = float("nan")
                w[2, 5, :] = float("-inf")
                w[3, 7, :] = float("inf")
                _, (total, best, least, where) = self._strided_reductions(
                    block_sizes, w
                )
                torch.testing.assert_close(total, w.sum(-1), equal_nan=True)
                torch.testing.assert_close(best, w.amax(-1), equal_nan=True)
                torch.testing.assert_close(least, w.amin(-1), equal_nan=True)
                # MPS's own argmax skips NaN; torch's semantics pick it.
                torch.testing.assert_close(where.cpu(), w.cpu().argmax(-1))

    def test_reduced_tile_with_sibling_loop(self) -> None:
        """A sibling loop shares its nesting level's thread axes.

        ``tile_k`` runs after the reduced ``tile_n`` loop, so both use the
        same threads: the launch is as wide as the wider of the two, not
        their product.  Threads beyond the narrower tile take no part in its
        reduction but still read the row result.
        """

        @helion.kernel(backend="metal", static_shapes=True)
        def max_then_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(y)
            for tile_m in hl.tile(m):
                mx = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                for tile_n in hl.tile(n):
                    mx = torch.maximum(mx, torch.amax(x[tile_m, tile_n], dim=1))
                for tile_k in hl.tile(y.size(1)):
                    out[tile_m, tile_k] = y[tile_m, tile_k] - mx[:, None]
            return out

        x = torch.randn(256, 96, device=DEVICE)
        y = torch.randn(256, 48, device=DEVICE)
        for block_sizes in ([1, 32, 32], [4, 8, 16], [4, 32, 8], [2, 4, 8]):
            with self.subTest(block_sizes=block_sizes):
                bound = max_then_shift.bind((x, y))
                config = helion.Config(block_sizes=block_sizes, num_warps=32)
                code = bound.to_code(config)
                dims = re.search(r"_block_dims=\(([^)]*)\)", code)
                assert dims is not None
                threads = math.prod(int(d) for d in dims.group(1).split(","))
                self.assertEqual(threads, block_sizes[0] * max(block_sizes[1:]))
                torch.testing.assert_close(
                    bound.compile_config(config)(x, y),
                    y - x.amax(1, keepdim=True),
                )

    def test_masked_collective_scratch_budget(self) -> None:
        """Masked collectives need one partial per SIMD group, so five int64
        reductions over 1024 threads fit in threadgroup memory.  Five strided
        collectives there would need 40 KB, so the largest sites fall back to
        the masked path until the scratch fits in 32 KB.  Five shared-memory
        trees over 1023 threads do not fit and are rejected."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[2, 512], num_warps=32)],
        )
        def five_reductions(x: torch.Tensor) -> torch.Tensor:
            o, d = x.shape
            out = torch.empty([5, o], dtype=x.dtype, device=x.device)
            for tile_o, tile_d in hl.tile([o, d]):
                v = x[tile_o, tile_d]
                out[0, tile_o] = v.sum(-1)
                out[1, tile_o] = v.amax(-1)
                out[2, tile_o] = v.amin(-1)
                out[3, tile_o] = (v * 2).sum(-1)
                out[4, tile_o] = (v + 1).amax(-1)
            return out

        x = torch.randint(-100, 100, (4, 512), dtype=torch.int64, device=DEVICE)
        expected = torch.stack(
            (x.sum(-1), x.amax(-1), x.amin(-1), (x * 2).sum(-1), (x + 1).amax(-1))
        )
        torch.testing.assert_close(five_reductions(x), expected)
        many_rows = helion.kernel(
            five_reductions.fn,
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32], num_warps=32)],
        )
        x = torch.randint(-100, 100, (32, 32), dtype=torch.int64, device=DEVICE)
        expected = torch.stack(
            (x.sum(-1), x.amax(-1), x.amin(-1), (x * 2).sum(-1), (x + 1).amax(-1))
        )
        torch.testing.assert_close(many_rows(x), expected)
        self.assertEqual(
            sorted(
                path for _, path, *_ in _group_reductions(_get_msl(many_rows, (x,)))
            ),
            ["masked", "masked", "strided", "strided", "strided"],
        )

        @helion.kernel(
            backend="metal", configs=[helion.Config(block_sizes=[], num_warps=32)]
        )
        def tree_beside_literal_root(
            x: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty([5, x.size(0)], dtype=x.dtype, device=x.device)
            doubled = torch.empty_like(z)
            for tm, tn in hl.tile(x.shape, block_size=[31, 32]):
                v = x[tm, tn]
                out[0, tm] = v.sum(-1)
                out[1, tm] = v.amax(-1)
                out[2, tm] = v.amin(-1)
                out[3, tm] = (v * 2).sum(-1)
                out[4, tm] = (v + 1).amax(-1)
            for tp, tq in hl.tile(z.shape, block_size=[31, 33]):
                doubled[tp, tq] = z[tp, tq] * 2
            return out, doubled

        # The later root widens the group to 1023 threads. Its tree scratch
        # needs 5 * 1023 * 8 bytes, exceeding the 32 KiB limit.
        with self.assertRaisesRegex(exc.BackendUnsupported, "threadgroup memory"):
            tree_beside_literal_root(
                torch.ones(31, 32, dtype=torch.int64, device=DEVICE),
                torch.ones(31, 33, dtype=torch.int64, device=DEVICE),
            )

    # (block sizes [m, n, k], shape of x, shape of y) for the kernels below:
    # the reduced ``tile_n`` is narrower than ``tid[0]``, which a sibling loop
    # (or a second root) launches ``block_k`` wide.
    _NARROW_TILE_CASES = (
        ([1, 4, 32], (64, 4), (64, 32)),  # row within one SIMD group: shuffle
        ([1, 8, 64], (64, 24), (64, 64)),  # 64-wide axis: scratch slot per row
        ([1, 16, 64], (64, 16), (64, 128)),  # likewise, two k tiles
    )

    def test_reduction_narrower_than_its_axis(self) -> None:
        """Threads past the reduced tile still receive their row's result.

        ``tile_k`` shares ``tid[0]`` with ``tile_n``, so threads with
        ``tid[0] >= block_n`` take no part in the reduction (their lanes are
        masked) but read ``s`` and ``mx`` in the sibling loop.
        """

        @helion.kernel(backend="metal", static_shapes=True)
        def sum_max_then_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(y)
            for tile_m in hl.tile(m):
                s = hl.zeros([tile_m], dtype=torch.float32)
                mx = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                for tile_n in hl.tile(n):
                    v = x[tile_m, tile_n]
                    s = s + torch.sum(v, dim=1)
                    mx = torch.maximum(mx, torch.amax(v, dim=1))
                for tile_k in hl.tile(y.size(1)):
                    out[tile_m, tile_k] = y[tile_m, tile_k] * mx[:, None] - s[:, None]
            return out

        for block_sizes, x_shape, y_shape in self._NARROW_TILE_CASES:
            with self.subTest(block_sizes=block_sizes):
                x = torch.randn(x_shape, device=DEVICE)
                y = torch.randn(y_shape, device=DEVICE)
                bound = sum_max_then_shift.bind((x, y))
                config = helion.Config(block_sizes=block_sizes, num_warps=4)
                self.assertIn("helion_red.group_", bound.to_code(config))
                torch.testing.assert_close(
                    bound.compile_config(config)(x, y),
                    y * x.amax(1, keepdim=True) - x.sum(1, keepdim=True),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_argreduction_narrower_than_its_axis(self) -> None:
        @helion.kernel(backend="metal", static_shapes=True)
        def argmax_then_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(y)
            for tile_m in hl.tile(m):
                idx = hl.zeros([tile_m], dtype=torch.int64)
                for tile_n in hl.tile(n):
                    # One tile spans ``n``, so the tile's argmax is the row's.
                    idx = torch.argmax(x[tile_m, tile_n], dim=1)
                for tile_k in hl.tile(y.size(1)):
                    out[tile_m, tile_k] = y[tile_m, tile_k] + idx[:, None].to(
                        torch.float32
                    )
            return out

        for block_n, block_k in ((8, 32), (16, 64)):
            with self.subTest(block_n=block_n, block_k=block_k):
                x = torch.randn(64, block_n, device=DEVICE)
                y = torch.randn(64, block_k, device=DEVICE)
                config = helion.Config(block_sizes=[1, block_n, block_k], num_warps=32)
                torch.testing.assert_close(
                    argmax_then_add.bind((x, y)).compile_config(config)(x, y),
                    y + x.argmax(1, keepdim=True).to(torch.float32),
                )

    def test_reduction_narrower_than_a_second_roots_tile(self) -> None:
        """Roots share one launch, so another root can widen the axis."""

        @helion.kernel(backend="metal", static_shapes=True)
        def max_and_double(
            x: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.size()
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            doubled = torch.empty_like(z)
            for tile_m in hl.tile(m):
                mx = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                for tile_n in hl.tile(n):
                    mx = torch.maximum(mx, torch.amax(x[tile_m, tile_n], dim=1))
                out[tile_m] = mx
            for tile_p in hl.tile(z.size(0)):
                doubled[tile_p] = z[tile_p] * 2
            return out, doubled

        x = torch.randn(256, 64, device=DEVICE)
        z = torch.randn(1024, device=DEVICE)
        # With block_m = 2 the reduced-tile plan moves tile_n to tid[0], which
        # the 32-wide root widens, and the reduction is contiguous, not strided.
        for block_sizes, path in (
            ([1, 16, 64], "contiguous"),
            ([2, 8, 32], "contiguous"),
        ):
            with self.subTest(block_sizes=block_sizes):
                kernel = helion.kernel(
                    max_and_double.fn,
                    backend="metal",
                    static_shapes=True,
                    configs=[helion.Config(block_sizes=block_sizes, num_warps=4)],
                )
                msl = _get_msl(kernel, (x, z))
                self.assertEqual([r[1] for r in _group_reductions(msl)], [path])
                out, doubled = kernel(x, z)
                torch.testing.assert_close(out, x.amax(1))
                torch.testing.assert_close(doubled, z * 2)

    def test_reduction_beside_a_literal_tile(self) -> None:
        """A literal block size can launch an axis at any extent.

        The second root sets the launch: ``tid[0]`` is ``block_p`` wide and
        ``tid[1]`` ``block_q`` wide, while ``tile_n`` reduces ``block_n`` of
        each row of ``tid[0]``. Literal widths use the bounded tree fallback.
        """

        # ``blocks`` only carries the second root's block sizes in its shape.
        @helion.kernel(backend="metal", static_shapes=True)
        def max_and_double(
            x: torch.Tensor, z: torch.Tensor, blocks: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.size()
            block_p = hl.specialize(blocks.size(0))
            block_q = hl.specialize(blocks.size(1))
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            doubled = torch.empty_like(z)
            for tile_m in hl.tile(m):
                mx = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                for tile_n in hl.tile(n):
                    mx = torch.maximum(mx, torch.amax(x[tile_m, tile_n], dim=1))
                out[tile_m] = mx
            for tile_p, tile_q in hl.tile(z.shape, block_size=[block_p, block_q]):
                doubled[tile_p, tile_q] = z[tile_p, tile_q] * 2
            return out, doubled

        def run(block_p: int, block_q: int, block_n: int) -> str:
            x = torch.randn(256, block_n, device=DEVICE)
            z = torch.randn(60, 128, device=DEVICE)
            args = (x, z, torch.empty(block_p, block_q))
            kernel = helion.kernel(
                max_and_double.fn,
                backend="metal",
                static_shapes=True,
                configs=[helion.Config(block_sizes=[1, block_n], num_warps=32)],
            )
            msl = _get_msl(kernel, args)
            out, doubled = kernel(*args)
            torch.testing.assert_close(out, x.amax(1))
            torch.testing.assert_close(doubled, z * 2)
            return msl

        # (block_p, block_q, block_n, group path)
        for block_p, block_q, block_n, path in (
            (6, 64, 2, "shared_tree"),
            (12, 2, 4, "shared_tree"),
            (12, 1, 8, "shared_tree"),
            (8, 4, 8, "contiguous"),
            (12, 2, 8, "shared_tree"),
        ):
            with self.subTest(block_p=block_p, block_q=block_q, block_n=block_n):
                msl = run(block_p, block_q, block_n)
                self.assertIn(
                    f"required_threads_per_threadgroup({block_p}, {block_q}, 1)", msl
                )
                self.assertEqual([r[1] for r in _group_reductions(msl)], [path])

    def test_device_loop_lane_reduction_reduces_every_lane(self) -> None:
        """A reduction over a block on 1024 threads x 2 lanes must not
        under-reduce: each thread folds both its lanes before the combine.
        The store keeps the last k tile's sum, which at k = 3000 is a
        952-element tail."""
        config = helion.Config(block_sizes=[1, 2048], num_threads=[1, 1024])

        @helion.kernel(backend="metal", configs=[config])
        def inner_tile_sum(x: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                for tile_k in hl.tile(k):
                    out[tile_m] = x[tile_m, tile_k].to(torch.float32).sum(dim=1)
            return out

        for shape, last_tile in (((1, 2048), 0), ((2, 3000), 2048)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE)
                code = inner_tile_sum.bind((x,)).to_code(config)
                self.assertNotIn(lane_reduction.MARKER, code)
                self.assertEqual(code.count("for lane_"), 1)
                expected = x.cpu().double()[:, last_tile:].sum(dim=1)
                torch.testing.assert_close(
                    inner_tile_sum(x).cpu(), expected.float(), rtol=1e-4, atol=1e-4
                )

    def test_a_starved_reduction_loop_raises_instead_of_truncating(self) -> None:
        """The thread-budget passes keep the span wide enough; assert it anyway.

        A reduction loop whose span is narrower than its chunk reduces only
        part of each chunk and returns a plausible wrong answer.  The budget
        passes make that unreachable (a chunk wider than its threads steps by
        them), so this forces the condition directly -- the point is that the
        invariant is checked rather than trusted, since it silently broke once
        already.
        """

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(block_sizes=[8], reduction_loops=[64], num_warps=32)
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tm in hl.tile(m):
                out[tm] = x[tm, :].sum(-1)
            return out

        x = torch.randn(64, 256, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3)

        # Starve the reduction: hand it half the threads its chunk needs.
        with (
            patch.object(
                LoopedReductionStrategy, "_reduction_thread_count", lambda self: 32
            ),
            self.assertRaisesRegex(exc.BackendUnsupported, "reduction loop"),
        ):
            helion.kernel(
                row_sum.fn,
                backend="metal",
                configs=[
                    helion.Config(block_sizes=[8], reduction_loops=[64], num_warps=32)
                ],
            )(x)

    def test_block_reduction_over_a_user_tile(self) -> None:
        """Reducing a user tile exercises resolve_group's second loop.

        With a single tile axis on ``tid[0]``, each block is contiguous, so
        the per-tile sum reduces through the ``seg_`` tier and broadcasts
        back over the tile.
        """
        for block in (8, 32):
            with self.subTest(block=block):

                @helion.kernel(
                    backend="metal",
                    configs=[helion.Config(block_sizes=[block])],
                )
                def block_broadcast_sum(x: torch.Tensor) -> torch.Tensor:
                    n = x.shape[0]
                    out = torch.empty([n], dtype=x.dtype, device=x.device)
                    for tile_n in hl.tile(n):
                        out[tile_n] = x[tile_n].sum()
                    return out

                x = torch.randn(64, device=DEVICE)
                self._check(
                    block_broadcast_sum,
                    (x,),
                    x.unflatten(-1, (-1, block)).sum(-1).repeat_interleave(block),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_prod_across_tiers(self) -> None:
        """Well-conditioned values: a wide prod would overflow in fp32."""
        for n in (32, 256, 4096):
            with self.subTest(n=n):
                x = 1 + 0.01 * torch.randn(16, n, device=DEVICE)
                self._check(row_prod, (x,), x.prod(-1), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Reductions over a tile after its loop
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal")
def sum_after_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sums_after_two_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """sum_after_loop over x and over y, in two roots sharing one block."""
    m, n = x.shape
    a = torch.empty([m], dtype=torch.float32, device=x.device)
    b = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(16, 64)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += x[tile_m, tile_n]
        a[tile_m] = acc.sum(-1)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += y[tile_m, tile_n]
        b[tile_m] = acc.sum(-1)
    return a, b


@helion.kernel(backend="metal", static_shapes=True)
def colsum_then_rowsum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x's column sums (x one tile of rows), then y's row sums, both over one
    host block: the roots lay the shared block's loop out differently."""
    m1, n = x.shape
    m2 = y.size(0)
    a = torch.empty([n], dtype=torch.float32, device=x.device)
    b = torch.empty([m2], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(16, 64)
    for tile_m in hl.tile(m1):
        for tile_n in hl.tile(n, block_size=block_n):
            a[tile_n] = x[tile_m, tile_n].sum(0)
    for tile_m in hl.tile(m2):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += y[tile_m, tile_n]
        b[tile_m] = acc.sum(-1)
    return a, b


@helion.kernel(backend="metal", static_shapes=True)
def tile2d_then_rows(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A 2-D tile over two host blocks, then a root that loops over only one
    of them: that loop is the second root's alone."""
    b, m, n = x.shape
    out0 = torch.empty_like(x)
    out1 = torch.empty([b], dtype=torch.float32, device=y.device)
    block_m = hl.register_block_size(4, 32)
    block_n = hl.register_block_size(4, 32)
    for tile_b in hl.tile(b):
        for tile_m, tile_n in hl.tile([m, n], block_size=[block_m, block_n]):
            out0[tile_b, tile_m, tile_n] = x[tile_b, tile_m, tile_n] + 1
    for tile_b in hl.tile(b):
        acc = hl.zeros([tile_b, block_m], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc += y[tile_b, tile_m, 0]
        out1[tile_b] = acc.sum(-1)
    return out0, out1


@helion.kernel(backend="metal", static_shapes=True)
def rolled_sums_in_two_roots(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row sums over one reduction dim in two roots, which share its rolled
    loop's strategy."""
    m, j, _ = x.shape
    a2 = torch.empty([m, j], dtype=torch.float32, device=x.device)
    a = torch.empty([m], dtype=torch.float32, device=x.device)
    b = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        for tile_j in hl.tile(j):
            a2[tile_m, tile_j] = x[tile_m, tile_j, :].sum(-1)
        a[tile_m] = y[tile_m, :].sum(-1)
    for tile_m in hl.tile(m):
        b[tile_m] = z[tile_m, :].sum(-1)
    return a2, a, b


@helion.kernel(backend="metal")
def max_carry_after_loop(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The carried row maxima, reduced by ``amax`` and by ``sum``."""
    m, n = x.shape
    best = torch.empty([m], dtype=torch.float32, device=x.device)
    total = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = torch.maximum(acc, x[tile_m, tile_n])
        best[tile_m] = torch.amax(acc, dim=-1)
        total[tile_m] = acc.sum(-1)
    return best, total


@helion.kernel(backend="metal")
def column_sum_after_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    for tile_n in hl.tile(n):
        acc = hl.zeros([block_m, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc += x[tile_m, tile_n]
        out[tile_n] = acc.sum(0)
    return out


@helion.kernel(backend="metal")
def mean_after_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.mean(-1)
    return out


@helion.kernel(backend="metal")
def same_block_siblings(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        a = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            a += x[tile_m, tile_n]
        b = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            b += y[tile_m, tile_n]
        out[tile_m] = a.sum(-1) + b.amax(-1)
    return out


@helion.kernel(backend="metal")
def two_block_inner_tile(x: torch.Tensor) -> torch.Tensor:
    b, n, k = x.shape
    out = torch.empty([b], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(k)
    for tile_b in hl.tile(b):
        acc = hl.zeros([tile_b, block_n, block_k], dtype=torch.float32)
        for tile_n, tile_k in hl.tile([n, k], block_size=[block_n, block_k]):
            acc += x[tile_b, tile_n, tile_k]
        out[tile_b] = acc.sum(-1).amax(-1)
    return out


@helion.kernel(backend="metal")
def threaded_siblings(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    k = y.shape[1]
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(k)
    for tile_m in hl.tile(m):
        a = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            a += x[tile_m, tile_n]
        b = hl.zeros([tile_m, block_k], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            b += y[tile_m, tile_k]
        out[tile_m] = a.sum(-1) + b.sum(-1)
    return out


@helion.kernel(backend="metal")
def reduced_in_sibling_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    k = y.shape[1]
    out = torch.empty_like(y)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(k)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += x[tile_m, tile_n]
        for tile_k in hl.tile(k, block_size=block_k):
            out[tile_m, tile_k] = y[tile_m, tile_k] + acc.sum(-1)[:, None]
    return out


@helion.kernel(backend="metal")
def beside_a_reduction_dim(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        row = z[tile_m, :].sum(-1)
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1) + row
    return out


def _column_partials(x: torch.Tensor, block: int) -> torch.Tensor:
    """What a ``[rows, block]`` sum carry holds after a loop over ``x``'s columns."""
    pad = -x.shape[-1] % block
    padded = torch.nn.functional.pad(x, (0, pad))
    return padded.unflatten(-1, (-1, block)).sum(-2)


@helion.kernel(backend="metal", static_shapes=True)
def two_slices(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_a = torch.empty_like(a)
    out_b = torch.empty_like(b)
    for tile_m in hl.tile(a.size(0)):
        out_a[tile_m, :] = a[tile_m, :] * 2.0
        out_b[tile_m, :, :] = b[tile_m, :, :] + 1.0
    return out_a, out_b


@helion.kernel(backend="metal", static_shapes=True)
def two_slices_in_a_grid(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a = torch.empty_like(a)
    out_b = torch.empty_like(b)
    for i in hl.grid(a.size(0)):
        out_a[i, :] = a[i, :] * 2.0
        out_b[i, :, :] = b[i, :, :] + 1.0
    return out_a, out_b


@helion.kernel(backend="metal", static_shapes=True)
def two_slices_in_two_loops(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a = torch.empty_like(a)
    out_b = torch.empty_like(b)
    for tile_m in hl.tile(a.size(0)):
        out_a[tile_m, :] = a[tile_m, :] * 2.0
    for tile_m in hl.tile(b.size(0)):
        out_b[tile_m, :, :] = b[tile_m, :, :] + 1.0
    return out_a, out_b


@helion.kernel(backend="metal", static_shapes=True)
def slice_beside_three_positions(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a = torch.empty_like(a)
    out_b = torch.empty_like(b)
    for tile_m in hl.tile(a.size(0)):
        out_a[tile_m, :] = a[tile_m, :] * 2.0
        out_b[tile_m, :, :, :] = b[tile_m, :, :, :] + 1.0
    return out_a, out_b


@helion.kernel(backend="metal", static_shapes=True)
def slice_beside_an_update(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a = torch.empty_like(a)
    for tile_m in hl.tile(a.size(0)):
        out_a[tile_m, :] = a[tile_m, :] * 2.0
        b[tile_m, :, :] = b[tile_m, :, :] * 2.0 + 1.0
    return out_a, b


@helion.kernel(backend="metal", static_shapes=True)
def rope_in_a_grid(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    for i in hl.grid(q.size(0)):
        c = cos[i, None, :]
        s = sin[i, None, :]
        q_lo = q[i, :, :8]
        q_hi = q[i, :, 8:]
        q_out[i, :, :8] = q_lo * c - q_hi * s
        q_out[i, :, 8:] = q_hi * c + q_lo * s
        k_lo = k[i, :, :8]
        k_hi = k[i, :, 8:]
        k_out[i, :, :8] = k_lo * c - k_hi * s
        k_out[i, :, 8:] = k_hi * c + k_lo * s
    return q_out, k_out


_HOSTED_POSITION = re.compile(r"position_lane_\d+_\d+ = tid\[")


@_requires_darwin
class TestMetalRepeatedSliceBlocks(unittest.TestCase):
    """Equal-sized ``:`` slices share one block, so a value can hold it twice.

    ``allocate_reduction_dimension`` caches by size, so ``x[t, :, :h]`` with
    ``h`` rows has the shape ``[t, r, r]``.  The block's lowest position keeps
    its thread index.  Each other position takes the thread axis of another
    ``:`` slice that no value meets it with, or else walks the block in a
    lane loop.
    """

    def _assert_positions(self, code: str, *, hosted: int, loops: int) -> None:
        """*hosted* positions on another slice's thread axis, *loops* in lane
        loops."""
        self.assertEqual(len(_HOSTED_POSITION.findall(code)), hosted)
        self.assertEqual(code.count("for position_lane"), loops)

    def test_equal_size_partial_slice(self) -> None:
        """``examples/rope.py`` reads ``q[tile_b, :, tile_t, :half_dim]``: with
        as many heads as ``half_dim``, one index for both would read only
        ``q[b, i, t, i]``."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def first_half(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, :, : d // 2] * 2
            return out

        # Equal halves, a non-power-of-two one, and unequal extents.
        for shape in ((3, 8, 16), (5, 6, 12), (3, 4, 16)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(first_half(x), x[:, :, : shape[2] // 2] * 2)
        x = torch.randn(3, 8, 16, device=DEVICE)
        self.assertIn("position_lane_", first_half.bind((x,)).to_code())

    def test_rope_with_as_many_heads_as_half_the_head_dim(self) -> None:
        """k's heads and half its head dim share a block.  Its second position
        takes the thread axis of q's heads, which no k statement reads, when
        that axis has at least as many threads: q's 8 heads beside k's 4
        have too few for q's second position, which keeps its lane loop.
        Six heads on 32 threads mask the surplus ones."""
        # The t tile on 4 threads leaves the slices 256 of 1024.
        configs = (
            helion.Config(
                block_sizes=[1, 32],
                num_threads=[1, 4],
                num_warps=32,
                reduction_loops=[None],
            ),
            helion.Config(block_sizes=[1, 16], num_threads=[1, 4], num_warps=32),
        )
        for kernel, q_heads, k_heads, head_dim, dtype, hosted in (
            (rope_fwd, 32, 8, 16, torch.float32, 1),
            (rope_fwd, 8, 4, 16, torch.float32, 0),
            (rope_fwd, 32, 8, 16, torch.float16, 1),
            (rope_fwd, 32, 6, 12, torch.float32, 1),
            (rope_bwd, 32, 8, 16, torch.float32, 1),
        ):
            q = torch.randn(1, q_heads, 64, head_dim, device=DEVICE, dtype=dtype)
            k = torch.randn(1, k_heads, 64, head_dim, device=DEVICE, dtype=dtype)
            # rope_bwd is rope with sin negated given tables with equal halves.
            half = torch.randn(64, head_dim // 2, device=DEVICE)
            angle = torch.cat([half, half], -1)
            cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
            sign = -1 if kernel is rope_bwd else 1
            bound = helion.kernel(kernel.fn, backend="metal", static_shapes=True).bind(
                (q, k, cos, sin)
            )
            for config in configs:
                with self.subTest(
                    kernel=kernel.name,
                    q_heads=q_heads,
                    k_heads=k_heads,
                    dtype=dtype,
                    config=config,
                ):
                    self._assert_positions(
                        bound.to_code(config), hosted=hosted, loops=1 - hosted
                    )
                    for out, ref in zip(
                        bound.compile_config(config)(q, k, cos, sin),
                        rope_pytorch(q, k, cos, sign * sin),
                        strict=True,
                    ):
                        torch.testing.assert_close(out, ref)

    def test_outer_product_of_equal_lengths(self) -> None:
        """Two positions of one block with no reduction, including a masked
        primary (6 on 8 threads) and a starved one (48)."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            (n2,) = b.shape
            out = torch.empty([m, n, n2], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = a[tile_m, :, None] * b[None, None, :]
            return out

        for n, n2 in ((8, 8), (6, 6), (48, 48), (8, 16)):
            with self.subTest(n=n, n2=n2):
                a = torch.randn(3, n, device=DEVICE)
                b = torch.randn(n2, device=DEVICE)
                torch.testing.assert_close(
                    outer_product(a, b), a[:, :, None] * b[None, None, :]
                )

    def test_store_target_repeating_a_block(self) -> None:
        """A store target can repeat a block that its value doesn't."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def broadcast_store(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :] = x[tile_m, :, None]
            return out

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def fill(out: torch.Tensor) -> torch.Tensor:
            for tile_m in hl.tile(out.size(0)):
                out[tile_m, :, :] = 1.0
            return out

        x = torch.randn(3, 8, device=DEVICE)
        for n in (8, 16):
            with self.subTest(n=n):
                torch.testing.assert_close(
                    broadcast_store(x, torch.zeros(3, 8, n, device=DEVICE)),
                    x[:, :, None].expand(3, 8, n),
                )
        torch.testing.assert_close(
            fill(torch.zeros(3, 8, 8, device=DEVICE)),
            torch.ones(3, 8, 8, device=DEVICE),
        )

    def test_three_positions_and_a_strided_slice(self) -> None:
        @helion.kernel(backend="metal", autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :, :] = x[tile_m, :, :, :] + 1.0
            return out

        @helion.kernel(backend="metal", autotune_effort="none")
        def even_columns(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, :, ::2] - 1.0
            return out

        x = torch.randn(2, 12, 12, 12, device=DEVICE)
        torch.testing.assert_close(add_one(x), x + 1)
        x = torch.randn(3, 8, 16, device=DEVICE)
        torch.testing.assert_close(even_columns(x), x[:, :, ::2] - 1)

    def test_start_and_step_on_a_lane_position(self) -> None:
        @helion.kernel(backend="metal", autotune_effort="none")
        def offset_rows(x: torch.Tensor) -> torch.Tensor:
            m, _, n = x.shape
            out = torch.empty([m, n, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, 3 : 3 + n, :] * 2.0
            return out

        @helion.kernel(backend="metal", autotune_effort="none")
        def odd_rows(x: torch.Tensor) -> torch.Tensor:
            m, _, n = x.shape
            out = torch.empty([m, n, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, 1::2, :] + 1.0
            return out

        for n in (8, 6):
            with self.subTest(n=n):
                x = torch.randn(3, n + 5, n, device=DEVICE)
                torch.testing.assert_close(offset_rows(x), x[:, 3 : 3 + n, :] * 2)
                x = torch.randn(3, 2 * n, n, device=DEVICE)
                torch.testing.assert_close(odd_rows(x), x[:, 1::2, :] + 1)

    def test_value_with_concrete_sizes(self) -> None:
        """``hl.zeros`` gives the value concrete sizes, so only the accesses
        say where the block is (positions 2 and 0 here)."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def zeros_between(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, 1, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                z = hl.zeros([tile_m, h, 1, d // 2], dtype=x.dtype)
                out[tile_m, :, :, :] = x[tile_m, :, None, : d // 2] + z
            return out

        x = torch.randn(3, 8, 16, device=DEVICE)
        torch.testing.assert_close(zeros_between(x), x[:, :, None, :8])

    def test_load_repeating_a_block_into_a_reduction_is_rejected(self) -> None:
        """No value repeats the block here, as ``hl.zeros`` makes them
        concrete, but the load does: unchecked, the sum read the diagonal."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def zeros_sum(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                z = hl.zeros([tile_m, h, d // 2], dtype=x.dtype)
                out[tile_m, :] = (x[tile_m, :, : d // 2] + z).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "a load index .* repeats"):
            zeros_sum(torch.randn(3, 8, 16, device=DEVICE))

    def test_ops_the_lanes_dont_follow_are_rejected(self) -> None:
        """Lanes follow a coordinate through right-aligned broadcasts only, so
        a view, a loop carry or a transpose anywhere in the kernel rejects."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def zeros_view(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, 1, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                z = hl.zeros([tile_m, h, d // 2], dtype=x.dtype)
                v = x[tile_m, :, : d // 2] + z
                out[tile_m, :, :, :] = v[:, :, None, :]
            return out

        @helion.kernel(backend="metal", autotune_effort="none")
        def loop_carry(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :, : d // 2] * 0
                for _ in hl.tile(4):
                    acc = acc + x[tile_m, :, : d // 2]
                out[tile_m, :, :] = acc
            return out

        @helion.kernel(backend="metal", autotune_effort="none")
        def transposed(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            out = torch.empty([m, h, d // 2], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = x[tile_m, :, : d // 2].transpose(1, 2)
            return out

        x = torch.randn(3, 8, 16, device=DEVICE)
        for kernel in (zeros_view, loop_carry, transposed):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaisesRegex(exc.BackendUnsupported, "pointwise ops"),
            ):
                kernel(x)

    def test_dynamic_extent_is_rejected(self) -> None:
        """A lane loop's extent is compiled in."""

        @helion.kernel(backend="metal", static_shapes=False, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :, :] = x[tile_m, :, :, :] + 1.0
            return out

        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            add_one(torch.randn(2, 8, 8, 8, device=DEVICE))

    def test_repeated_tile_is_rejected(self) -> None:
        """One 8-wide tile covers the matrix, so ``out[t, t]`` writes every
        element, which only a slice block's lanes could index."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[8])])
        def tile_twice(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, tile_m] = x[tile_m, tile_m] + 1.0
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "not a `:` slice"):
            tile_twice(torch.randn(8, 8, device=DEVICE))

    def test_in_place_updates_with_one_owner_per_element(self) -> None:
        """Writes that each element's one thread and lane iteration owns: equal
        subscripts, or ranges of the thread position's dim that don't meet.
        256 wide, so threads span several SIMD groups and a race would show."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def scale_half(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            for tile_m in hl.tile(m):
                x[tile_m, :, : d // 2] = x[tile_m, :, : d // 2] * 2.0 + 1.0
            return x

        @helion.kernel(backend="metal", autotune_effort="none")
        def swap_halves(x: torch.Tensor) -> torch.Tensor:
            m, h, d = x.shape
            for tile_m in hl.tile(m):
                lo = x[tile_m, :, : d // 2]
                hi = x[tile_m, :, d // 2 :]
                x[tile_m, :, : d // 2] = hi * 2.0
                x[tile_m, :, d // 2 :] = lo + 1.0
            return x

        @helion.kernel(backend="metal", autotune_effort="none")
        def scale_rows(x: torch.Tensor) -> torch.Tensor:
            m, _, n = x.shape
            for tile_m in hl.tile(m):
                x[tile_m, 2 : n + 2, :] = x[tile_m, 2 : n + 2, :] * 4.0
            return x

        @helion.kernel(backend="metal", autotune_effort="none")
        def rope_k_in_place(
            k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ) -> torch.Tensor:
            batch, _, seq_len, head_dim = k.size()
            half = head_dim // 2
            for tile_b, tile_t in hl.tile([batch, seq_len]):
                cos_lo = cos[tile_b, :, tile_t, :half]
                cos_hi = cos[tile_b, :, tile_t, half:]
                sin_lo = sin[tile_b, :, tile_t, :half]
                sin_hi = sin[tile_b, :, tile_t, half:]
                k_lo = k[tile_b, :, tile_t, :half]
                k_hi = k[tile_b, :, tile_t, half:]
                k[tile_b, :, tile_t, :half] = k_lo * cos_lo - k_hi * sin_lo
                k[tile_b, :, tile_t, half:] = k_hi * cos_hi + k_lo * sin_hi
            return k

        n = 256
        x = torch.randn(4, n, 2 * n, device=DEVICE)
        expected = x.clone()
        expected[:, :, :n] = x[:, :, :n] * 2 + 1
        torch.testing.assert_close(scale_half(x.clone()), expected)
        torch.testing.assert_close(
            swap_halves(x.clone()), torch.cat([x[:, :, n:] * 2, x[:, :, :n] + 1], -1)
        )
        x = torch.randn(4, n + 4, n, device=DEVICE)
        expected = x.clone()
        expected[:, 2 : n + 2, :] = x[:, 2 : n + 2, :] * 4
        torch.testing.assert_close(scale_rows(x.clone()), expected)
        q = torch.randn(1, 32, 4, 2 * n, device=DEVICE)
        k = torch.randn(1, n, 4, 2 * n, device=DEVICE)
        cos = torch.randn(4, 2 * n, device=DEVICE)
        sin = torch.randn_like(cos)
        torch.testing.assert_close(
            rope_k_in_place(
                k.clone(), cos.expand(1, 1, 4, 2 * n), sin.expand(1, 1, 4, 2 * n)
            ),
            rope_pytorch(q, k, cos, sin)[1],
        )

    def test_writes_another_thread_or_lane_could_reach_are_rejected(self) -> None:
        """A written element must have one owner: the lanes add no barrier, and
        a lane loop runs iteration ``h``'s store before iteration ``h + 1``'s
        load, where a tile statement reads its whole tile first."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def update_beside(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = a.shape
            out = torch.empty([m, n, n], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = a[tile_m, :, None] * b[None, None, :]
                c[tile_m, :, :] = c[tile_m, :, :] * 2.0 + 1.0
            return out, c

        @helion.kernel(backend="metal", autotune_effort="none")
        def read_column(
            a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = a.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                y[tile_m, :, :] = a[tile_m, :, None] * b[None, None, :]
                out[tile_m, :] = y[tile_m, :, 0] + 1.0
            return out, y

        @helion.kernel(backend="metal", autotune_effort="none")
        def read_column_apart(
            a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = a.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                y[tile_m, 0:n, n : 2 * n] = a[tile_m, :, None] * b[None, None, :]
                out[tile_m, :] = y[tile_m, 0:n, n] + 1.0
            return out, y

        @helion.kernel(backend="metal", autotune_effort="none")
        def overlapping_writes(
            a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, n = a.shape
            for tile_m in hl.tile(m):
                y[tile_m, :, 0:n] = a[tile_m, :, None] * b[None, None, :]
                y[tile_m, :, n // 2 : n // 2 + n] = (
                    a[tile_m, :, None] + b[None, None, :]
                )
            return y

        @helion.kernel(backend="metal", autotune_effort="none")
        def shift_rows(x: torch.Tensor) -> torch.Tensor:
            m, _, n = x.shape
            for tile_m in hl.tile(m):
                x[tile_m, 1 : n + 1, :] = x[tile_m, 0:n, :] * 2.0
            return x

        @helion.kernel(backend="metal", autotune_effort="none")
        def shifted_writes(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            for tile_m in hl.tile(m):
                y[tile_m, 0:n, :] = a[tile_m, None, :] + 1.0
                y[tile_m, 2 : n + 2, :] = a[tile_m, None, :] * 0.0 - 5.0
            return y

        @helion.kernel(backend="metal", autotune_effort="none")
        def shift_other_block(x: torch.Tensor) -> torch.Tensor:
            m, rows, _, _ = x.shape
            for tile_m in hl.tile(m):
                x[tile_m, 1:rows, :, :] = x[tile_m, 0 : rows - 1, :, :] * 2.0
            return x

        @helion.kernel(backend="metal", autotune_effort="none")
        def drop_other_block(
            a: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, _, n, _ = y.shape
            out = torch.empty([m, n, n], dtype=y.dtype, device=y.device)
            for tile_m in hl.tile(m):
                y[tile_m, :, :, :] = a[tile_m, :, :, :] + 1.0
                out[tile_m, :, :] = y[tile_m, 0, :, :] * 3.0
            return out, y

        @helion.kernel(backend="metal", autotune_effort="none")
        def read_through_a_view(
            a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = a.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            y3 = y.view(m, n, n)
            for tile_m in hl.tile(m):
                y3[tile_m, :, :] = a[tile_m, :, None] * b[None, None, :]
                out[tile_m, :] = y[tile_m, ::n] + 1.0
            return out, y

        n = 8
        a = torch.randn(4, n, device=DEVICE)
        b = torch.randn(n, device=DEVICE)
        cases = {
            "thread position missing": lambda: update_beside(
                a, b, torch.zeros(4, n, 1, device=DEVICE)
            ),
            "another tensor dim": lambda: read_column(
                a, b, torch.zeros(4, n, n, device=DEVICE)
            ),
            "another tensor dim, at disjoint offsets": lambda: read_column_apart(
                a, b, torch.zeros(4, n, 2 * n, device=DEVICE)
            ),
            "overlapping ranges": lambda: overlapping_writes(
                a, b, torch.zeros(4, n, n + n // 2, device=DEVICE)
            ),
            "shifted lane position": lambda: shift_rows(
                torch.randn(4, n + 1, n, device=DEVICE)
            ),
            "shifted lane writes": lambda: shifted_writes(
                a, torch.zeros(4, n + 2, n, device=DEVICE)
            ),
            "shifted other block": lambda: shift_other_block(
                torch.randn(4, 7, 4, 4, device=DEVICE)
            ),
            "dropped other block": lambda: drop_other_block(
                torch.randn(4, 6, 4, 4, device=DEVICE),
                torch.zeros(4, 6, 4, 4, device=DEVICE),
            ),
            "two views": lambda: read_through_a_view(
                a, b, torch.zeros(4, n * n, device=DEVICE)
            ),
        }
        for case, call in cases.items():
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "two threads, or two lane iterations"
                ),
            ):
                call()

    @_full_threadgroup
    def test_position_on_an_idle_slice_axis(self) -> None:
        """b's second position takes a's thread axis, which no b statement
        reads; 6 positions on 32 threads mask the surplus, and a's 6
        columns get 8 threads, exactly b's 8 positions, so need no mask."""
        for a_cols, b_rows, masked in ((32, 8, {"8"}), (32, 6, {"6"}), (6, 8, set())):
            a = torch.randn(64, a_cols, device=DEVICE)
            b = torch.randn(64, b_rows, b_rows, device=DEVICE)
            bound = two_slices.bind((a, b))
            for config in (
                bound.config_spec.default_config(),
                helion.Config(block_sizes=[1]),
            ):
                with self.subTest(a_cols=a_cols, b_rows=b_rows, config=config):
                    code = bound.to_code(config)
                    self._assert_positions(code, hosted=1, loops=0)
                    for access in ("tl.load(", "tl.store("):
                        lines = "\n".join(
                            line
                            for line in code.splitlines()
                            if access in line and "position_lane" in line
                        )
                        self.assertEqual(
                            set(re.findall(r"position_lane_\d+_\d+ < (\d+)\)", lines)),
                            masked,
                        )
                    out_a, out_b = bound.compile_config(config)(a, b)
                    torch.testing.assert_close(out_a, a * 2)
                    torch.testing.assert_close(out_b, b + 1)

    def test_hosted_position_beside_a_lane_looped_thread_position(self) -> None:
        """b's first position walks a lane loop of its own: 32 rows on 16
        threads at one row per tile, or starved of threads beside 1000
        columns.  Its second position still takes a's axis."""
        for shape_a, shape_b, config in (
            ((8, 64), (8, 32, 32), helion.Config(block_sizes=[1])),
            ((4, 1000), (4, 8, 8), None),
        ):
            a = torch.randn(shape_a, device=DEVICE)
            b = torch.randn(shape_b, device=DEVICE)
            bound = two_slices.bind((a, b))
            config = config or bound.config_spec.default_config()
            with self.subTest(shape_b=shape_b):
                code = bound.to_code(config)
                self._assert_positions(code, hosted=1, loops=0)
                self.assertIn("slice_lane", code)
                out_a, out_b = bound.compile_config(config)(a, b)
                torch.testing.assert_close(out_a, a * 2)
                torch.testing.assert_close(out_b, b + 1)

    def test_position_on_an_idle_slice_axis_under_a_grid(self) -> None:
        """Under ``hl.grid`` an access may have no mask of its own, so the
        surplus-thread mask stands alone."""
        a = torch.randn(64, 32, device=DEVICE)
        b = torch.randn(64, 8, 8, device=DEVICE)
        bound = two_slices_in_a_grid.bind((a, b))
        for config in (
            bound.config_spec.default_config(),
            helion.Config(num_warps=1),
            helion.Config(num_warps=8),
            helion.Config(num_warps=32),
        ):
            with self.subTest(config=config):
                self._assert_positions(bound.to_code(config), hosted=1, loops=0)
                out_a, out_b = bound.compile_config(config)(a, b)
                torch.testing.assert_close(out_a, a * 2)
                torch.testing.assert_close(out_b, b + 1)
        q = torch.randn(16, 32, 16, device=DEVICE)
        k = torch.randn(16, 8, 16, device=DEVICE)
        cos, sin = torch.randn(2, 16, 8, device=DEVICE).unbind()
        bound = rope_in_a_grid.bind((q, k, cos, sin))
        config = bound.config_spec.default_config()
        self._assert_positions(bound.to_code(config), hosted=1, loops=0)
        cos2, sin2 = (torch.cat([t, t], -1)[:, None, :] for t in (cos, sin))
        for out, x in zip(
            bound.compile_config(config)(q, k, cos, sin), (q, k), strict=True
        ):
            rotated = torch.cat([-x[..., 8:], x[..., :8]], -1)
            torch.testing.assert_close(out, x * cos2 + rotated * sin2)

    @_full_threadgroup
    def test_one_hosted_position_per_axis(self) -> None:
        """b's two extra positions both meet each other, so only one can take
        a's axis; the other keeps its lane loop."""
        a = torch.randn(64, 32, device=DEVICE)
        b = torch.randn(64, 8, 8, 8, device=DEVICE)
        bound = slice_beside_three_positions.bind((a, b))
        config = bound.config_spec.default_config()
        self._assert_positions(bound.to_code(config), hosted=1, loops=1)
        out_a, out_b = bound.compile_config(config)(a, b)
        torch.testing.assert_close(out_a, a * 2)
        torch.testing.assert_close(out_b, b + 1)

    @_full_threadgroup
    def test_in_place_update_through_a_hosted_position(self) -> None:
        """Each element of b has one owner, the thread whose ``tid`` names its
        position; in a lane loop all 64 threads of a's axis would run it.
        37 rows leave a tile tail whose stores the tile mask must
        still drop beside the surplus-thread mask, short of a sentinel."""
        a = torch.randn(37, 64, device=DEVICE)
        b = torch.randn(37, 8, 8, device=DEVICE)
        bound = slice_beside_an_update.bind((a, b))
        for config in (
            bound.config_spec.default_config(),
            helion.Config(block_sizes=[16], num_warps=8),
            helion.Config(block_sizes=[4], num_threads=[8], num_warps=8),
        ):
            self._assert_positions(bound.to_code(config), hosted=1, loops=0)
            kernel = bound.compile_config(config)
            for rep in range(10):
                with self.subTest(config=config, rep=rep):
                    buffer = torch.full((64, 8, 8), 7.0, device=DEVICE)
                    buffer[:37] = b
                    out_a, _ = kernel(a, buffer[:37])
                    torch.testing.assert_close(out_a, a * 2)
                    torch.testing.assert_close(buffer[:37], b * 2 + 1)
                    torch.testing.assert_close(
                        buffer[37:], torch.full((27, 8, 8), 7.0, device=DEVICE)
                    )

    @_full_threadgroup
    def test_repeated_blocks_share_an_axis_unless_a_value_holds_both(self) -> None:
        """y's 1000 columns starve a's and c's blocks of threads, so each
        block's extra position walks a lane loop unless it takes y's axis.
        a's and c's positions both take it when no value holds both; when
        one value does, only the first takes it and the other keeps its
        lane loop.  A position joins only the first axis that takes it, so
        beside y [16, 64] and w [16, 16], c's position still finds w's axis
        when a's took y's."""

        @helion.kernel(backend="metal", static_shapes=True)
        def apart(
            y: torch.Tensor, a: torch.Tensor, c: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out_y = torch.empty_like(y)
            out_a = torch.empty_like(a)
            out_c = torch.empty_like(c)
            for i in hl.grid(y.size(0)):
                out_y[i, :] = y[i, :] * 2.0
                out_a[i, :, :] = a[i, :, :] + 1.0
                out_c[i, :, :] = c[i, :, :] - 1.0
            return out_y, out_a, out_c

        @helion.kernel(backend="metal", static_shapes=True)
        def together(
            y: torch.Tensor, a: torch.Tensor, c: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_y = torch.empty_like(y)
            out = torch.empty(
                [a.size(0), a.size(1), a.size(2), c.size(1), c.size(2)],
                dtype=a.dtype,
                device=a.device,
            )
            for i in hl.grid(y.size(0)):
                out_y[i, :] = y[i, :] * 2.0
                out[i, :, :, :, :] = a[i, :, :, None, None] + c[i, None, None, :, :]
            return out_y, out

        @helion.kernel(backend="metal", static_shapes=True)
        def together_beside_two_slices(
            y: torch.Tensor, w: torch.Tensor, a: torch.Tensor, c: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out_y = torch.empty_like(y)
            out_w = torch.empty_like(w)
            out = torch.empty(
                [a.size(0), a.size(1), a.size(2), c.size(1), c.size(2)],
                dtype=a.dtype,
                device=a.device,
            )
            for i in hl.grid(y.size(0)):
                out_y[i, :] = y[i, :] * 2.0
                out_w[i, :] = w[i, :] * 3.0
                out[i, :, :, :, :] = a[i, :, :, None, None] + c[i, None, None, :, :]
            return out_y, out_w, out

        y = torch.randn(16, 1000, device=DEVICE)
        y64 = torch.randn(16, 64, device=DEVICE)
        w = torch.randn(16, 16, device=DEVICE)
        a = torch.randn(16, 8, 8, device=DEVICE)
        c = torch.randn(16, 4, 4, device=DEVICE)
        a_plus_c = a[:, :, :, None, None] + c[:, None, None]
        for kernel, args, hosted, expected in (
            (apart, (y, a, c), 2, (y * 2, a + 1, c - 1)),
            (together, (y, a, c), 1, (y * 2, a_plus_c)),
            (
                together_beside_two_slices,
                (y64, w, a, c),
                2,
                (y64 * 2, w * 3, a_plus_c),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                config = bound.config_spec.default_config()
                self._assert_positions(
                    bound.to_code(config), hosted=hosted, loops=2 - hosted
                )
                for out, ref in zip(
                    bound.compile_config(config)(*args), expected, strict=True
                ):
                    torch.testing.assert_close(out, ref)

    def test_positions_that_must_keep_their_lane_loops(self) -> None:
        """A position stays in its lane loop when a value meets it with the
        candidate axis's block (a load of both, or a broadcast of one
        against the other), when a store whose value holds it doesn't index
        it, when it is wider than the axis, or when the kernel has two grid
        roots, which number their thread axes independently."""

        @helion.kernel(backend="metal", static_shapes=True)
        def load_of_both(
            x: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_x = torch.empty_like(x)
            out_b = torch.empty_like(b)
            for tile_m in hl.tile(x.size(0)):
                out_x[tile_m, :, :, :] = x[tile_m, :, :, :] + 1.0
                out_b[tile_m, :, :] = b[tile_m, :, :] * 2.0
            return out_x, out_b

        @helion.kernel(backend="metal", static_shapes=True)
        def broadcast_of_both(
            a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            for tile_m in hl.tile(a.size(0)):
                out[tile_m, :, :, :] = a[tile_m, :, None, None] + b[tile_m, None, :, :]
            return out

        a = torch.randn(4, 32, device=DEVICE)
        b = torch.randn(4, 8, 8, device=DEVICE)
        x = torch.randn(4, 32, 8, 8, device=DEVICE)
        wide = torch.randn(4, 64, 64, device=DEVICE)
        eight = torch.randn(4, 8, device=DEVICE)
        nine = torch.randn(4, 9, 9, device=DEVICE)
        # a has 6 columns (8 threads, b's extent): with a wider a, the second
        # grid root would index b past its end, a separate two-root limitation.
        six = torch.randn(4, 6, device=DEVICE)
        cases = {
            "load of both": (
                load_of_both,
                (x, b),
                (x + 1, b * 2),
            ),
            "broadcast of both": (
                broadcast_of_both,
                (a, b, torch.zeros(4, 32, 8, 8, device=DEVICE)),
                (a[:, :, None, None] + b[:, None],),
            ),
            "wider than the axis": (two_slices, (a, wide), (a * 2, wide + 1)),
            "one wider than the axis": (
                two_slices,
                (eight, nine),
                (eight * 2, nine + 1),
            ),
            "two grid roots": (two_slices_in_two_loops, (six, b), (six * 2, b + 1)),
        }
        for case, (kernel, args, expected) in cases.items():
            with self.subTest(case=case):
                bound = kernel.bind(args)
                config = bound.config_spec.default_config()
                self._assert_positions(bound.to_code(config), hosted=0, loops=1)
                out = bound.compile_config(config)(*args)
                for o, e in zip(
                    out if isinstance(out, tuple) else (out,), expected, strict=True
                ):
                    torch.testing.assert_close(o, e)

    def test_store_dropping_the_position_keeps_its_lane_loop(self) -> None:
        """``out[t, :, :] = b[t, :, :]`` into a ``[t, 1, 8]`` target writes b's
        rows to one row.  Helion leaves that undefined; the lane loop writes
        them in order, so the last stays, where threads of a's axis would
        race to write it.  ``hl.zeros`` gives the value concrete sizes, so
        only its input holds the position; a store mask holds it too."""

        @helion.kernel(backend="metal", static_shapes=True)
        def narrowing_store(
            a: torch.Tensor, b: torch.Tensor, out_b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_a = torch.empty_like(a)
            for tile_m in hl.tile(a.size(0)):
                out_a[tile_m, :] = a[tile_m, :] * 2.0
                out_b[tile_m, :, :] = b[tile_m, :, :]
            return out_a, out_b

        @helion.kernel(backend="metal", static_shapes=True)
        def narrowing_store_of_concrete_sizes(
            a: torch.Tensor, b: torch.Tensor, out_b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_a = torch.empty_like(a)
            m, h, d = b.shape
            for tile_m in hl.tile(m):
                out_a[tile_m, :] = a[tile_m, :] * 2.0
                out_b[tile_m, :, :] = b[tile_m, :, :] + hl.zeros(
                    [tile_m, h, d], dtype=b.dtype
                )
            return out_a, out_b

        @helion.kernel(backend="metal", static_shapes=True)
        def narrowing_masked_store(
            a: torch.Tensor, b: torch.Tensor, out_b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out_a = torch.empty_like(a)
            for tile_m in hl.tile(a.size(0)):
                out_a[tile_m, :] = a[tile_m, :] * 2.0
                hl.store(
                    out_b,
                    [tile_m, slice(None), slice(None)],
                    b[tile_m, None, 0, :],
                    extra_mask=b[tile_m, :, :] > 0,
                )
            return out_a, out_b

        a = torch.randn(64, 32, device=DEVICE)
        b = torch.randn(64, 8, 8, device=DEVICE)
        last_row = b[:, 7:8, :]
        masked_first_row = torch.where((b > 0).any(1, keepdim=True), b[:, :1, :], 0.0)
        for kernel, expected in (
            (narrowing_store, last_row),
            (narrowing_store_of_concrete_sizes, last_row),
            (narrowing_masked_store, masked_first_row),
        ):
            with self.subTest(kernel=kernel.name):
                out_b = torch.zeros(64, 1, 8, device=DEVICE)
                bound = kernel.bind((a, b, out_b))
                config = bound.config_spec.default_config()
                self._assert_positions(bound.to_code(config), hosted=0, loops=1)
                bound.compile_config(config)(a, b, out_b)
                torch.testing.assert_close(out_b, expected)


@_requires_darwin
class TestMetalReductionAfterTileLoop(unittest.TestCase):
    """A reduction over a tile block after the block's loop has exited.

    The carried value holds one partial per thread of the loop's axis, so the
    reduction must still combine across that axis.  Only layouts where that
    axis is proven take the combine; the rest are rejected.
    """

    def _run(
        self, kernel: helion.Kernel, config: helion.Config, *args: torch.Tensor
    ) -> object:
        return kernel.bind(args).compile_config(config)(*args)

    def test_sum_after_loop(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        for block_sizes in ([32, 32], [64, 16]):
            with self.subTest(block_sizes=block_sizes):
                config = helion.Config(block_sizes=block_sizes, num_warps=4)
                torch.testing.assert_close(
                    self._run(sum_after_loop, config, x),
                    x.sum(-1),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_sum_after_loop_combines_on_the_loops_axis(self) -> None:
        """The reduced block is planned onto ``tid[0]``, where it combines."""
        x = torch.randn(64, 128, device=DEVICE)
        kernel = helion.kernel(
            sum_after_loop.fn,
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32], num_warps=32)],
        )
        msl = _get_msl(kernel, (x,))
        self.assertEqual(
            [
                (op, path, axis, span)
                for op, path, _, axis, span in _group_reductions(msl)
            ],
            [("sum", "contiguous", 0, 32)],
        )

    def test_tails_and_tiny_shapes(self) -> None:
        for shape, block_sizes in (
            ((37, 100), [32, 32]),
            ((37, 100), [8, 4]),
            ((3, 7), [8, 4]),
        ):
            with self.subTest(shape=shape, block_sizes=block_sizes):
                x = torch.randn(*shape, device=DEVICE)
                config = helion.Config(block_sizes=block_sizes)
                torch.testing.assert_close(
                    self._run(sum_after_loop, config, x),
                    x.sum(-1),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_host_block_in_two_roots(self) -> None:
        """A block registered on the host, which both roots loop over, is
        threaded in each root's loop and combined after it."""
        x = torch.randn(64, 128, device=DEVICE)
        y = torch.randn(64, 128, device=DEVICE)
        config = helion.Config(block_sizes=[16, 32, 32], num_threads=[16, 0, 0])
        a, b = self._run(sums_after_two_roots, config, x, y)
        torch.testing.assert_close(a, x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b, y.sum(-1), rtol=1e-4, atol=1e-4)

    def test_host_block_on_two_axes(self) -> None:
        """Roots that would thread a shared host block's loop on different
        axes are refused: the body indexes that loop with one axis."""
        x = torch.randn(16, 128, device=DEVICE)
        y = torch.randn(64, 128, device=DEVICE)
        config = helion.Config(block_sizes=[16, 32, 32], num_threads=[16, 0, 0])
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "several roots run needs one thread axis"
        ):
            self._run(colsum_then_rowsum, config, x, y)
        a, b = self._run(
            colsum_then_rowsum, helion.Config(block_sizes=[16, 32, 32]), x, y
        )
        torch.testing.assert_close(a, x.sum(0), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b, y.sum(-1), rtol=1e-4, atol=1e-4)

    def test_host_block_loops_by_root(self) -> None:
        """A root gets only the host-block loops in its own graphs: a 1-D
        loop over one block of another root's 2-D tile is the second root's
        alone, and a rolled reduction two roots share stays out of the loop
        assignment."""
        x = torch.randn(4, 32, 32, device=DEVICE)
        y = torch.randn(4, 32, 32, device=DEVICE)
        config = helion.Config(block_sizes=[8, 8, 1, 1], num_threads=[8, 8, 0, 0])
        out0, out1 = self._run(tile2d_then_rows, config, x, y)
        torch.testing.assert_close(out0, x + 1)
        torch.testing.assert_close(out1, y[:, :, 0].sum(-1), rtol=1e-4, atol=1e-4)
        x = torch.randn(8, 4, 64, device=DEVICE)
        y = torch.randn(8, 256, device=DEVICE)
        z = torch.randn(8, 256, device=DEVICE)
        config = helion.Config(
            block_sizes=[8, 4, 8], num_warps=32, reduction_loops=[64]
        )
        a2, a, b = self._run(rolled_sums_in_two_roots, config, x, y, z)
        torch.testing.assert_close(a2, x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(a, y.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b, z.sum(-1), rtol=1e-4, atol=1e-4)

    def test_reduction_over_dim_0(self) -> None:
        x = torch.randn(100, 48, device=DEVICE)
        config = helion.Config(block_sizes=[16, 8])
        torch.testing.assert_close(
            self._run(column_sum_after_loop, config, x),
            x.sum(0),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_max_carry(self) -> None:
        """``amax`` matches the carry's op; ``sum`` of the maxima does not."""
        x = torch.randn(8, 512, device=DEVICE)
        config = helion.Config(block_sizes=[128, 4], num_threads=[128, 4])
        best, total = self._run(max_carry_after_loop, config, x)
        maxima = x.unflatten(-1, (-1, 128)).amax(-2)
        torch.testing.assert_close(best, x.amax(-1))
        torch.testing.assert_close(total, maxima.sum(-1), rtol=1e-4, atol=1e-4)

    def test_mean_divides_by_the_block(self) -> None:
        """The carry has ``block_n`` columns, so its mean is the total over them."""
        x = torch.randn(64, 128, device=DEVICE)
        config = helion.Config(block_sizes=[32, 32])
        torch.testing.assert_close(
            self._run(mean_after_loop, config, x),
            x.sum(-1) / 32,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_dynamic_shapes(self) -> None:
        kernel = helion.kernel(sum_after_loop.fn, backend="metal", static_shapes=False)
        x = torch.randn(37, 100, device=DEVICE)
        config = helion.Config(block_sizes=[32, 32])
        torch.testing.assert_close(
            self._run(kernel, config, x), x.sum(-1), rtol=1e-4, atol=1e-4
        )

    def test_same_block_siblings(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        y = torch.randn(8, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 4])
        torch.testing.assert_close(
            self._run(same_block_siblings, config, x, y),
            x.sum(-1) + _column_partials(y, 16).amax(-1),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_two_block_inner_tile(self) -> None:
        x = torch.randn(4, 32, 32, device=DEVICE)
        config = helion.Config(block_sizes=[8, 4, 4])
        expected = x.unflatten(1, (-1, 8)).sum(1).unflatten(-1, (-1, 4)).sum(-2)
        torch.testing.assert_close(
            self._run(two_block_inner_tile, config, x),
            expected.sum(-1).amax(-1),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_reduced_inside_an_unthreaded_sibling_loop(self) -> None:
        """The exited block is reduced while another loop (and lane loop) runs."""
        x = torch.randn(8, 64, device=DEVICE)
        y = torch.randn(8, 40, device=DEVICE)
        for block_sizes, num_threads in (([32, 4, 4], [32, 1, 0]), ([32, 1, 4], [])):
            with self.subTest(block_sizes=block_sizes, num_threads=num_threads):
                config = helion.Config(block_sizes=block_sizes, num_threads=num_threads)
                torch.testing.assert_close(
                    self._run(reduced_in_sibling_loop, config, x, y),
                    y + x.sum(-1, keepdim=True),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_one_thread_keeps_the_loop_carried_sum(self) -> None:
        """A block lane-looped on one thread already folds its row."""
        x = torch.randn(4, 3000, device=DEVICE)
        config = helion.Config(block_sizes=[4096, 1])
        torch.testing.assert_close(
            self._run(sum_after_loop, config, x), x.sum(-1), rtol=1e-3, atol=1e-3
        )

    def test_threads_by_lanes_is_rejected(self) -> None:
        """Each thread's carried maximum already folded its four lanes."""
        x = torch.randn(8, 512, device=DEVICE)
        config = helion.Config(block_sizes=[128, 4], num_threads=[32, 4])
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "not one element per thread"
        ):
            self._run(max_carry_after_loop, config, x)

    def test_threaded_sibling_loops_are_rejected(self) -> None:
        """The body puts both siblings on one axis; no plan says which."""
        x = torch.randn(8, 64, device=DEVICE)
        y = torch.randn(8, 40, device=DEVICE)
        config = helion.Config(block_sizes=[32, 32, 4], num_threads=[32, 32, 0])
        with self.assertRaisesRegex(exc.BackendUnsupported, "no layout plan"):
            self._run(threaded_siblings, config, x, y)

    def test_a_reduction_dim_alongside_is_rejected(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        z = torch.randn(64, 8, device=DEVICE)
        config = helion.Config(block_sizes=[8, 4])
        with self.assertRaisesRegex(exc.BackendUnsupported, "no layout plan"):
            self._run(beside_a_reduction_dim, config, x, z)


# ---------------------------------------------------------------------------
# Literal dims beside tiles
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def literal_carry_row_sum(x: torch.Tensor) -> torch.Tensor:
    """Column partials of 64-wide tiles in a literal carry, summed after."""
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=64):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_carry_tile_first(x: torch.Tensor) -> torch.Tensor:
    """``literal_carry_row_sum`` with the tile first, stored through ``:``."""
    m, n = x.shape
    out = torch.empty([m, 64], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=64):
            acc = x[tile_m, tile_n] + acc
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_plus_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, :]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_store_into_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = y[tile_m, :] + hl.zeros([tile_m, 64], dtype=x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def load_masked_by_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = hl.load(x, [tile_m, tile_n], extra_mask=y[tile_m, :] > 0)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def store_masked_by_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        hl.store(out, [tile_m, tile_n], x[tile_m, tile_n], extra_mask=y[tile_m, :] > 0)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def load_masked_by_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = hl.load(
            x, [tile_m, tile_n], extra_mask=y[tile_m, tile_n] > 0
        )
    return out


@helion.kernel(backend="metal", static_shapes=True)
def store_masked_by_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        hl.store(
            out, [tile_m, tile_n], x[tile_m, tile_n], extra_mask=y[tile_m, tile_n] > 0
        )
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_holding_other_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A literal dim that holds ``tile_k`` times ``tile_n`` at its position."""
    m, n = x.shape
    k = y.shape[1]
    out = torch.empty([m, n // 32, k // 32], dtype=torch.float32, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k], block_size=[None, 32, 32]):
        t = hl.zeros([tile_m, 32], dtype=torch.float32) + y[tile_m, tile_k]
        out[tile_m, tile_n.id, tile_k.id] = (t * x[tile_m, tile_n]).sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_carry_of_slices(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A literal carry updated with ``:`` slices of its width."""
    m, k = x.shape
    out = torch.empty([m, 64], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + y[tile_m, :] * x[tile_m, tile_k].sum(1, keepdim=True)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_carry_beside_grid_tile(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A literal carry holding a 64-wide grid tile through an inner loop."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_k in hl.tile(w.size(0)):
            acc = acc + x[tile_m, tile_n] * w[tile_k].sum()
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_index_store(x: torch.Tensor) -> torch.Tensor:
    """``:`` and a tile stored beside an ``hl.grid`` index, which keeps no dim."""
    _, n, b = x.shape
    out = torch.empty_like(x)
    for row in hl.grid(b):
        for tile_n in hl.tile(n):
            out[:, tile_n, row] = x[:, tile_n, row] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_index_store_inner(x: torch.Tensor) -> torch.Tensor:
    """``grid_index_store`` with the grid inside the tile loop."""
    _, n, b = x.shape
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        for row in hl.grid(b):
            out[:, tile_n, row] = x[:, tile_n, row] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_into_slice(x: torch.Tensor) -> torch.Tensor:
    """A 64-wide tile stored through ``:``."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, :] = x[tile_m, tile_n] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_into_slice_inner(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=64):
            out[tile_m, :] = x[tile_m, tile_n] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_into_slice_beside_id(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, n // 64, 64], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n.id, :] = x[tile_m, tile_n] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_row_sum(x: torch.Tensor) -> torch.Tensor:
    """A literal dim holding a tile, reduced (by the reduction block of 64)."""
    m, n = x.shape
    out = torch.empty([m, n // 64], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        v = hl.zeros([tile_m, 64], dtype=x.dtype) + x[tile_m, tile_n]
        out[tile_m, tile_n.id] = v.sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def carry_slice_then_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A carry entering as a ``:`` slice and leaving as a 64-wide tile."""
    m, n = x.shape
    out = torch.empty([m, 64], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = y[tile_m, :]
        for tile_n in hl.tile(n, block_size=64):
            acc = x[tile_m, tile_n] * 2
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_unrelated_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``tile_k`` times ``tile_n`` at one position, no literal involved."""
    m, n = x.shape
    k = y.shape[1]
    out = torch.empty([m, n // 32, k // 32], dtype=torch.float32, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k], block_size=[None, 32, 32]):
        prod = y[tile_m, tile_k] * x[tile_m, tile_n]
        out[tile_m, tile_n.id, tile_k.id] = prod.sum(1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_plus_transpose(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two tiles meeting swapped: Helion pairs them by position, as loaded,
    where Metal paired them by block (``x + y.T``)."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, 16]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_n, tile_m]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def torch_zeros_carry(x: torch.Tensor) -> torch.Tensor:
    """``torch.zeros`` is the same along its literal dim, as ``hl.zeros``."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        acc = torch.zeros([tile_m, 64], dtype=x.dtype, device=x.device)
        acc = acc + x[tile_m, tile_n]
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_like_beside_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        t = hl.zeros([tile_m, 64], dtype=x.dtype) + x[tile_m, tile_n]
        out[tile_m, tile_n] = t + (torch.zeros_like(t) + 1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_subscript_store(x: torch.Tensor) -> torch.Tensor:
    """``hl.subscript`` keeps the holder of the literal dim it indexes."""
    m, n = x.shape
    out = torch.empty([m, n, 1], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        t = hl.zeros([tile_m, 64], dtype=x.dtype) + x[tile_m, tile_n]
        out[tile_m, tile_n, :] = t[:, :, None]
    return out


# metal/literal_dims.py's refusal of two unrelated tiles meeting by position.
_PAIRS_TILES = r"pairs, at its dim \d+, dims held by block \d+'s tile, block \d+'s tile"


def _ints(*shape: int) -> torch.Tensor:
    # Integer values keep the sums exact in any order.
    return torch.randint(-4, 5, shape, device=DEVICE).to(torch.float32)


class TestMetalLiteralDims(unittest.TestCase):
    """Literal dims (``hl.zeros([tile_m, 64])``, a ``:`` slice) beside tiles,
    as test_literal_dims.py checks them for Triton and CuTe.

    A Metal thread finds the block of a literal dim by its size, never a
    tile, so where a literal dim holds a tile's elements, or a ``:`` slice
    meets a tile, it read other elements than the positional program: every
    refused kernel here returned wrong values before it was refused.
    """

    def _run(
        self, kernel: helion.Kernel, config: helion.Config, *args: torch.Tensor
    ) -> object:
        return kernel.bind(args).compile_config(config)(*args)

    def test_literal_carry_beside_device_loop_tile(self) -> None:
        x = _ints(96, 640)
        message = r"loop carry .* holds block \d+'s tile .* hl\.register_block_size"
        for block_sizes, num_warps in (([1], 1), ([4], 4), ([16], 4)):
            config = helion.Config(block_sizes=block_sizes, num_warps=num_warps)
            for kernel in (literal_carry_row_sum, literal_carry_tile_first):
                with (
                    self.subTest(kernel=kernel.name, block_sizes=block_sizes),
                    self.assertRaisesRegex(exc.BackendUnsupported, message),
                ):
                    self._run(kernel, config, x)

    def test_slice_beside_tile(self) -> None:
        x, y = _ints(96, 64), _ints(96, 64)
        config = helion.Config(block_sizes=[4], num_warps=4)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            # y[tile_m, :] traces as the literal 64 here.
            r"aten.add.Tensor pairs, at its dim 1, dims held by block \d+'s tile, a "
            r"literal dim .* index the ':' dim it comes from with the tile",
        ):
            self._run(tile_plus_slice, config, x, y)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"a store writes dim 1 .* held by a literal dim .* through block \d+'s "
            r"tile .* index the ':' dim it comes from with the tile",
        ):
            self._run(literal_store_into_tile, config, x, y)

    def test_tile_into_slice(self) -> None:
        # Each row stored one repeated value.
        x = _ints(96, 64)
        for kernel in (
            tile_into_slice,
            tile_into_slice_inner,
            tile_into_slice_beside_id,
        ):
            for block_sizes in ([1], [4], [32]):
                with (
                    self.subTest(kernel=kernel.name, block_sizes=block_sizes),
                    self.assertRaisesRegex(
                        exc.BackendUnsupported,
                        r"a store writes dim 1 .* held by block \d+'s tile, through "
                        r"a ':' slice .* index the ':' dim with the tile",
                    ),
                ):
                    self._run(
                        kernel, helion.Config(block_sizes=block_sizes, num_warps=4), x
                    )

    def test_literal_holding_a_tile_reduced(self) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"aten.sum.dim_IntList reads v .* holds block \d+'s tile.* "
            r"hl\.register_block_size",
        ):
            self._run(literal_row_sum, helion.Config(block_sizes=[4]), _ints(96, 128))

    def test_carry_entering_and_leaving_on_different_blocks(self) -> None:
        x, y = _ints(96, 128), _ints(96, 64)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"control-flow carry .* held by a ':' slice and block \d+'s tile.* "
            r"index the ':' dim with the tile",
        ):
            self._run(carry_slice_then_tile, helion.Config(block_sizes=[4]), x, y)

    def test_unrelated_tiles(self) -> None:
        # Compiled and summed the wrong products at any config.
        x, y = _ints(16, 64), _ints(16, 64)
        for block_sizes in ([1], [4]):
            with (
                self.subTest(block_sizes=block_sizes),
                self.assertRaisesRegex(exc.BackendUnsupported, _PAIRS_TILES),
            ):
                self._run(
                    two_unrelated_tiles,
                    helion.Config(block_sizes=block_sizes, num_warps=4),
                    x,
                    y,
                )
        # Swapped tiles pair by position, as loaded (what ref mode computes),
        # where Metal's pairing by block computed x + y.T.
        x, y = _ints(32, 32), _ints(32, 32)
        with self.assertRaisesRegex(exc.BackendUnsupported, _PAIRS_TILES):
            self._run(tile_plus_transpose, helion.Config(), x, y)

    def test_factory_and_subscript_dims_agree(self) -> None:
        x = _ints(96, 128)
        config = helion.Config(block_sizes=[4], num_warps=4)
        torch.testing.assert_close(self._run(torch_zeros_carry, config, x), x)
        torch.testing.assert_close(
            self._run(literal_like_beside_tile, config, x), x + 1
        )
        torch.testing.assert_close(
            self._run(literal_subscript_store, config, x), x[:, :, None]
        )

    def test_literal_holding_another_tile(self) -> None:
        # Refused at any block size of tile_m: block 1 at [4] compiled and
        # summed the wrong products.
        x, y = _ints(16, 64), _ints(16, 64)
        for block_sizes in ([1], [4]):
            with (
                self.subTest(block_sizes=block_sizes),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    r"aten.mul.Tensor pairs, at its dim 1, dims held by block \d+'s "
                    r"tile, block \d+'s tile",
                ),
            ):
                self._run(
                    literal_holding_other_tile,
                    helion.Config(block_sizes=block_sizes, num_warps=4),
                    x,
                    y,
                )

    def test_agreeing_literal_dims(self) -> None:
        config = helion.Config(block_sizes=[4, 32], num_warps=4)
        x, y = _ints(96, 640), _ints(96, 64)
        torch.testing.assert_close(
            self._run(literal_carry_of_slices, config, x, y),
            y * x.sum(1, keepdim=True),
        )
        w = _ints(48)
        torch.testing.assert_close(
            self._run(literal_carry_beside_grid_tile, config, x, w), x * w.sum()
        )

    def test_grid_index_store(self) -> None:
        x = _ints(4, 64, 6)
        for kernel in (grid_index_store, grid_index_store_inner):
            for block_sizes in ([16], [64]):
                with self.subTest(kernel=kernel.name, block_sizes=block_sizes):
                    config = helion.Config(block_sizes=block_sizes, num_warps=4)
                    torch.testing.assert_close(self._run(kernel, config, x), x * 2)

    def test_masks_beside_tiles(self) -> None:
        # A mask is read at the element its load or store addresses: a ':'
        # mask beside a tile read other rows' columns.
        x, y = _ints(16, 64), _ints(16, 64)
        expected = torch.where(y > 0, x, 0)
        for block_sizes in ([1], [4]):
            config = helion.Config(block_sizes=block_sizes, num_warps=4)
            for kernel, what in (
                (load_masked_by_slice, "a load's mask reads"),
                (store_masked_by_slice, "a store's mask reads"),
            ):
                with (
                    self.subTest(kernel=kernel.name, block_sizes=block_sizes),
                    self.assertRaisesRegex(
                        exc.BackendUnsupported,
                        rf"{what} dim 1 .* held by a ':' slice, through block "
                        r"\d+'s tile .* index the ':' dim with the tile",
                    ),
                ):
                    self._run(kernel, config, x, y)
            for kernel in (load_masked_by_tile, store_masked_by_tile):
                with self.subTest(kernel=kernel.name, block_sizes=block_sizes):
                    torch.testing.assert_close(
                        self._run(kernel, config, x, y), expected
                    )


# ---------------------------------------------------------------------------
# Squeeze
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        at_label = hl.load(logits_flat, [tile_n.index * v + labels[tile_n]])
        rows = logits[tile_n, :]
        max_logits = torch.amax(rows, dim=-1, keepdim=True)
        sum_exp = torch.sum(torch.exp(rows - max_logits), dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        losses[tile_n] = log_sum_exp - at_label
    return losses


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_row_max(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1, keepdim=True).squeeze(-1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def unsqueezed_row_max(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_column_sum(x: torch.Tensor) -> torch.Tensor:
    _, n = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[:, tile_n].sum(0, keepdim=True).squeeze(0)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_middle_max(x: torch.Tensor) -> torch.Tensor:
    b, _, c = x.shape
    out = torch.empty([b, c], dtype=x.dtype, device=x.device)
    for tile_b, tile_c in hl.tile([b, c]):
        out[tile_b, tile_c] = x[tile_b, :, tile_c].amax(1, keepdim=True).squeeze(1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_trailing_column(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].squeeze(-1) * 2.0
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_leading_plane(x: torch.Tensor) -> torch.Tensor:
    _, m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[:, tile_m, tile_n].squeeze(0) + 1.0
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_full_rows(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = x[tile_n, :].squeeze(-1) * 3.0
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_grid_row_sum(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for i in hl.grid(n):
        out[i] = x[i, :].sum(0, keepdim=True).squeeze(0)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_all_row_max(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1, keepdim=True).squeeze()
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def squeezed_dims_row_max(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1, keepdim=True).squeeze((-1,))
    return out


@_requires_darwin
class TestMetalSqueeze(unittest.TestCase):
    """``squeeze`` removes only dims of size 1, which own no thread axis, so
    each thread's value passes through unchanged."""

    def test_squeeze_keepdim_reductions(self) -> None:
        logits = torch.randn(70, 300, device=DEVICE)
        labels = torch.randint(300, (70,), device=DEVICE)
        expected = torch.logsumexp(logits, -1) - logits.gather(
            1, labels[:, None]
        ).squeeze(1)
        for static_shapes in (True, False):
            kernel = helion.kernel(
                squeezed_cross_entropy.fn,
                backend="metal",
                static_shapes=static_shapes,
            )
            bound = kernel.bind((logits, labels))
            default = bound.config_spec.default_config()
            for config in (
                default,
                helion.Config(**{**default, "reduction_loops": [32]}),
                helion.Config(**{**default, "block_sizes": [16], "num_threads": [4]}),
            ):
                with self.subTest(static_shapes=static_shapes, config=config):
                    if not static_shapes and config.reduction_loops == [None]:
                        # A persistent ``:`` of runtime extent rejects.
                        with self.assertRaisesRegex(
                            exc.BackendUnsupported,
                            r"a ':' dimension of runtime extent \w+",
                        ):
                            bound.compile_config(config)
                        continue
                    torch.testing.assert_close(
                        bound.compile_config(config)(logits, labels),
                        expected,
                        rtol=1e-4,
                        atol=1e-4,
                    )

    def test_squeeze_emits_no_code(self) -> None:
        x = torch.randn(70, 300, device=DEVICE)
        squeezed = _get_msl(squeezed_row_max, (x,))
        unsqueezed = _get_msl(unsqueezed_row_max, (x,))
        self.assertEqual(
            re.sub(r"//.*", "", squeezed).replace("squeezed_row_max", "K"),
            re.sub(r"//.*", "", unsqueezed).replace("unsqueezed_row_max", "K"),
        )

    def test_squeeze_leading_and_middle_dims(self) -> None:
        x = torch.randn(70, 300, device=DEVICE)
        torch.testing.assert_close(
            squeezed_column_sum(x), x.sum(0), rtol=1e-4, atol=1e-4
        )
        y = torch.randn(6, 37, 20, device=DEVICE)
        torch.testing.assert_close(squeezed_middle_max(y), y.amax(1))

    def test_squeeze_singleton_input_dims(self) -> None:
        x = torch.randn(70, 1, device=DEVICE)
        torch.testing.assert_close(squeezed_trailing_column(x), x[:, 0] * 2.0)
        y = torch.randn(1, 37, 20, device=DEVICE)
        torch.testing.assert_close(squeezed_leading_plane(y), y[0] + 1.0)

    def test_squeeze_non_singleton_dim_is_identity(self) -> None:
        x = torch.randn(70, 300, device=DEVICE)
        torch.testing.assert_close(squeezed_full_rows(x), x * 3.0)

    def test_squeeze_to_scalar(self) -> None:
        x = torch.randn(70, 300, device=DEVICE)
        torch.testing.assert_close(
            squeezed_grid_row_sum(x), x.sum(-1), rtol=1e-4, atol=1e-4
        )

    def test_squeeze_overloads(self) -> None:
        x = torch.randn(70, 300, device=DEVICE)
        for kernel in (squeezed_all_row_max, squeezed_dims_row_max):
            with self.subTest(kernel=kernel.fn.__name__):
                torch.testing.assert_close(kernel(x), x.amax(-1))


# ---------------------------------------------------------------------------
# aten.permute: a per-thread relabel, and the positional merges it rejects
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_store(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = x[tm, tn].T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_3d_store(x: torch.Tensor) -> torch.Tensor:
    a, b, c = x.shape
    out = torch.empty([c, a, b], dtype=x.dtype, device=x.device)
    for t0, t1, t2 in hl.tile([a, b, c]):
        out[t2, t0, t1] = x[t0, t1, t2].permute(2, 0, 1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_twice(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = (x[tm, tn].T * 3).T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_plus_outer(
    x: torch.Tensor, bm: torch.Tensor, bn: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = x[tm, tn].T + bm[tm] * bn[tn][:, None]
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_lower_rank_operand(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``w[tp, tm].T`` is placed at dims 0 and 2 of the ``[tm, tn, tp]`` sum."""
    m, n, p = v.shape
    out = torch.empty_like(v)
    for tm, tn, tp in hl.tile([m, n, p]):
        out[tm, tn, tp] = v[tm, tn, tp] + w[tp, tm].T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_row_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = (x[tm, :] * 2).T.sum(0)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_column_max(x: torch.Tensor) -> torch.Tensor:
    _, n = x.shape
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        out[tn] = torch.amax(x[:, tn].T, dim=1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_slice_sum(y: torch.Tensor) -> torch.Tensor:
    n, _, m = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=y.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = y[tn, :, tm].permute(2, 0, 1).sum(-1)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def dot_permuted_lhs(x: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A computed, transposed lhs over a tile K (mamba2_chunk_state's form)."""
    k, m = x.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            lhs = (x[tk, tm].T * s[tk][None, :]).to(torch.float32)
            acc = hl.dot(lhs, b[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def dot_permuted_slice_lhs(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """M is a ``:`` slice of a transposed lhs, K a tile (gdn_fwd_h's form)."""
    t, d = k.shape
    _, n = v.shape
    d = hl.specialize(d)
    out = torch.empty([d, n], dtype=torch.float32, device=k.device)
    for tn in hl.tile(n):
        acc = hl.zeros([d, tn], dtype=torch.float32)
        for tc in hl.tile(t, block_size=16):
            acc = hl.dot(k[tc, :].T, v[tc, tn] * 1.0, acc=acc)
        out[:, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def dot_permuted_rhs_load(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n, _ = b.shape
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tn, tk].T, acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_where(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, n = c.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = torch.where(c[tm, tn].T > 0, x[tn, tm], 0.0)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_loop_carry(y: torch.Tensor) -> torch.Tensor:
    k, n, m = y.shape
    out = torch.empty([m, n], dtype=torch.float32, device=y.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k, block_size=1):
            acc = acc + y[tk, tn, tm].sum(0).T
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_slice_store(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[:, tm] = (x[tm, :] + 1.0).T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def permuted_slice_plus_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[:, tm] = x[tm, :].T + y[:, tm]
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def mpp_result_permuted_twice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tm, tn] = acc.T.T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def mpp_result_transposed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([n, m], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tn, tm] = acc.T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def gather_permuted_index_and_vector(
    x: torch.Tensor, ia: torch.Tensor, ic: torch.Tensor
) -> torch.Tensor:
    m, n = ia.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = hl.load(x, [ia[tm, tn].T, ic[tm]])
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def gather_permuted_index_and_wrong_vector(
    x: torch.Tensor, ia: torch.Tensor, ic: torch.Tensor
) -> torch.Tensor:
    m, n = ia.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = hl.load(x, [ia[tm, tn].T, ic[tn]])
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def gather_permuted_index_pair(
    x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor
) -> torch.Tensor:
    m, n = ia.shape
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = hl.load(x, [ia[tm, tn].T, ib[tm, tn]])
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def scatter_permuted_index_pair(
    x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor
) -> torch.Tensor:
    m, n = ia.shape
    out = torch.zeros_like(x)
    for tm, tn in hl.tile([m, n]):
        hl.store(out, [ia[tm, tn].T, ib[tm, tn].T], x[tm, tn].T)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_store(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn].T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = x[tm, tn].T + y[tm, tn]
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_lower_rank_operand(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n, p = v.shape
    out = torch.empty_like(v)
    for tm, tn, tp in hl.tile([m, n, p]):
        out[tm, tn, tp] = v[tm, tn, tp] * w[tm, tn].T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_where(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, n = c.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.where(c[tm, tn] > 0, x[tm, tn].T, 0.0)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_store_mask(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, n = c.shape
    out = torch.zeros_like(x)
    for tm, tn in hl.tile([m, n]):
        hl.store(out, [tm, tn], x[tm, tn], extra_mask=(c[tm, tn] > 0).T)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_load_mask(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, n = c.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = hl.load(x, [tm, tn], extra_mask=(c[tm, tn] > 0).T)
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_accumulator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk] * 1.0, b[tk, tn], acc=acc.T)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_addmm_accumulator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc.T, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        acc = x[tm, tn] * 1.0
        for _ in hl.tile(3, block_size=1):
            acc = acc.T + 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_stepped_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        acc = x[tm, tn] * 1.0
        for _ in hl.grid(0, 6, step=2):
            acc = acc.T + 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_second_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        acc0 = x[tm, tn] * 1.0
        acc1 = x[tm, tn] * 2.0
        for _ in hl.grid(0, 6, step=2):
            acc0 = acc0 + 1.0
            acc1 = acc1.T + 1.0
        out[tm, tn] = acc0 + acc1
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_while_carry(x: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        acc = x[tm, tn] * 1.0
        stop = hl.load(limit, [0])
        steps = torch.zeros([], device=x.device, dtype=torch.int32)
        while steps < stop:
            acc = acc.T + 1.0
            steps = steps + 1
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_branch_carry(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        acc = x[tm, tn] * 1.0
        if hl.load(flag, [0]) > 0:
            acc = acc.T + 1.0
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_slice_plus_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[16, 16]):
        out[tn, tm] = x[tm, :].T + y[tn, tm]
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_tile_plus_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[16, 16]):
        out[tn, tm] = y[tn, tm] + x[tm, :].T
    return out


@helion.kernel(backend="metal", autotune_effort="none")
def rebound_slice_plus_tile_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n], block_size=[16, 16]):
        out[tm] = (x[tm, :].T + y[tn, tm]).sum(0)
    return out


@_requires_darwin
class TestMetalPermute(unittest.TestCase):
    """``permute`` relabels dims without moving an element between threads.

    A Metal thread holds each value by block id, while Helion's tiles are
    positional, so a kernel that permutes is rejected wherever two values
    meet by position on different blocks, or a tile meets a dim without one.
    """

    def _run(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None = None,
    ) -> tuple[torch.Tensor, str]:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        return bound.compile_config(config)(*args), bound.to_code(config)

    def _assert_rejected(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None,
        message: str,
    ) -> None:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        with self.assertRaisesRegex(exc.BackendUnsupported, message):
            bound.to_code(config)

    def test_permute_relabels_tiles(self) -> None:
        x = torch.randn(48, 80, device=DEVICE)
        torch.testing.assert_close(self._run(permuted_store, (x,))[0], x.T)
        torch.testing.assert_close(self._run(permuted_twice, (x,))[0], x * 3)
        x3 = torch.randn(5, 7, 9, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_3d_store, (x3,))[0], x3.permute(2, 0, 1)
        )

    def test_permute_with_broadcast_operands(self) -> None:
        x = torch.randn(48, 80, device=DEVICE)
        bm = torch.randn(48, device=DEVICE)
        bn = torch.randn(80, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_plus_outer, (x, bm, bn))[0],
            x.T + bm[None, :] * bn[:, None],
        )
        v = torch.randn(12, 20, 28, device=DEVICE)
        w = torch.randn(28, 12, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_lower_rank_operand, (v, w))[0], v + w.T[:, None, :]
        )

    def test_permute_then_reduce(self) -> None:
        x = torch.randn(48, 80, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_row_sum, (x,))[0], (x * 2).sum(1), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(self._run(permuted_column_max, (x,))[0], x.amax(0))
        y = torch.randn(20, 16, 36, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_slice_sum, (y,))[0], y.sum(1).T, rtol=1e-4, atol=1e-4
        )

    def test_permute_into_dot(self) -> None:
        tiled = helion.Config(block_sizes=[16, 16, 16], num_threads=[4, 4, 16])
        x = torch.randn(72, 40, device=DEVICE)
        s = torch.randn(72, device=DEVICE)
        b = torch.randn(72, 44, device=DEVICE)
        out, _ = self._run(dot_permuted_lhs, (x, s, b), tiled)
        torch.testing.assert_close(
            out,
            _matmul_reference(x.T * s[None, :], b).to(DEVICE),
            rtol=1e-4,
            atol=1e-4,
        )
        k = torch.randn(64, 32, device=DEVICE)
        v = torch.randn(64, 40, device=DEVICE)
        out, _ = self._run(dot_permuted_slice_lhs, (k, v))
        torch.testing.assert_close(
            out, _matmul_reference(k.T, v).to(DEVICE), rtol=1e-4, atol=1e-4
        )
        a = torch.randn(40, 72, device=DEVICE)
        bt = torch.randn(44, 72, device=DEVICE)
        out, code = self._run(dot_permuted_rhs_load, (a, bt), tiled)
        self.assertIn("_helion_mpp_run(", code)
        torch.testing.assert_close(
            out, _matmul_reference(a, bt.T).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_permute_where_and_carry(self) -> None:
        c = torch.randn(48, 80, device=DEVICE)
        x = torch.randn(80, 48, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_where, (c, x))[0], torch.where(c.T > 0, x, 0.0)
        )
        y = torch.randn(6, 20, 36, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_loop_carry, (y,))[0], y.sum(0).T, rtol=1e-4, atol=1e-4
        )

    def test_permute_beside_slices(self) -> None:
        """A ``:`` slice is an int in traced values and its reduction block in
        an access; both forms meet each other, not a tile."""
        x = torch.randn(40, 12, device=DEVICE)
        torch.testing.assert_close(self._run(permuted_slice_store, (x,))[0], x.T + 1)
        dynamic = helion.kernel(
            permuted_slice_store.fn,
            backend="metal",
            autotune_effort="none",
            static_shapes=False,
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            self._run(dynamic, (x,))
        rolled = helion.Config(block_sizes=[8], reduction_loops=[4])
        for shape in ((40, 12), (24, 30)):
            x = torch.randn(*shape, device=DEVICE)
            torch.testing.assert_close(self._run(dynamic, (x,), rolled)[0], x.T + 1)
        x = torch.randn(48, 16, device=DEVICE)
        y = torch.randn(16, 48, device=DEVICE)
        torch.testing.assert_close(
            self._run(permuted_slice_plus_slice, (x, y))[0], x.T + y
        )

    def test_permute_after_mpp_k_loop(self) -> None:
        config = helion.Config(block_sizes=[16, 16, 16])
        a = torch.randn(64, 64, device=DEVICE)
        b = torch.randn(64, 64, device=DEVICE)
        out, code = self._run(mpp_result_permuted_twice, (a, b), config)
        self.assertNotIn("dot_product", code)
        torch.testing.assert_close(out, a @ b, rtol=1e-2, atol=1e-2)
        a = torch.randn(48, 64, device=DEVICE)
        b = torch.randn(64, 80, device=DEVICE)
        out, code = self._run(mpp_result_transposed, (a, b), config)
        self.assertIn("_helion_mpp_run(", code)
        torch.testing.assert_close(
            out, _matmul_reference(a, b).T.to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_permute_with_index_tensors(self) -> None:
        config = helion.Config(block_sizes=[16, 16])
        x = torch.randn(32, 32, device=DEVICE)
        ia = torch.randint(0, 32, (32, 32), device=DEVICE)
        ic = torch.randint(0, 32, (32,), device=DEVICE)
        out, _ = self._run(gather_permuted_index_and_vector, (x, ia, ic), config)
        torch.testing.assert_close(out, x[ia, ic[:, None]].T)
        # A permutation of the elements, so the scatter has no collisions.
        perm = torch.randperm(32 * 32, device=DEVICE)
        ia, ib = (perm // 32).reshape(32, 32), (perm % 32).reshape(32, 32)
        expected = torch.zeros_like(x)
        expected[ia, ib] = x
        out, _ = self._run(scatter_permuted_index_pair, (x, ia, ib), config)
        torch.testing.assert_close(out, expected)

    def test_rebinding_merges_are_rejected(self) -> None:
        square = helion.Config(block_sizes=[16, 16])
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        cases = [
            (rebound_store, (x,), square, "support: a store "),
            (rebound_add, (x, y), square, "support: an element-wise operand"),
            (rebound_where, (x, y), square, "support: a where operand"),
            (rebound_store_mask, (x, y), square, "support: a store mask"),
            (rebound_load_mask, (x, y), square, "support: a load mask"),
            (
                rebound_lower_rank_operand,
                (torch.randn(32, 32, 16, device=DEVICE), x[:32, :32]),
                helion.Config(block_sizes=[16, 16, 16]),
                "support: a broadcast element-wise operand",
            ),
            (
                rebound_accumulator,
                (x[:, :48], x[:48]),
                helion.Config(block_sizes=[16, 16, 16], num_threads=[4, 4, 16]),
                "support: a matmul accumulator",
            ),
            (
                rebound_addmm_accumulator,
                (x[:, :48], x[:48]),
                helion.Config(block_sizes=[16, 16, 16]),
                "support: a matmul accumulator",
            ),
            # Blocks of extent 1 are not exempt.
            (
                rebound_store,
                (x,),
                helion.Config(block_sizes=[1, 1]),
                "support: a store ",
            ),
        ]
        for kernel, args, config, message in cases:
            with self.subTest(kernel=kernel.fn.__name__, config=config):
                self._assert_rejected(kernel, args, config, message)

    def test_rebinding_carry_is_rejected(self) -> None:
        """A carry through every loop op and through a branch, including the
        second of two carries."""
        config = helion.Config(block_sizes=[16, 16])
        x = torch.randn(64, 64, device=DEVICE)
        limit = torch.tensor([3], device=DEVICE, dtype=torch.int32)
        for kernel, args in (
            (rebound_carry, (x,)),
            (rebound_stepped_carry, (x,)),
            (rebound_second_carry, (x,)),
            (rebound_while_carry, (x, limit)),
            (rebound_branch_carry, (x, limit)),
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                self._assert_rejected(
                    kernel, args, config, "support: a control-flow carry"
                )

    def test_transposed_mpp_result_store(self) -> None:
        """A K loop's result stored through the ``[tn, tm]`` index of its
        transpose: the element loop stores each element at its own index."""
        a = torch.randn(64, 64, device=DEVICE)
        b = torch.randn(64, 64, device=DEVICE)
        for config in (None, helion.Config(block_sizes=[16, 16, 16])):
            with self.subTest(config=config):
                out, code = self._run(mpp_result_transposed, (a, b), config)
                self.assertIn("_helion_mpp_acc(", code)
                torch.testing.assert_close(out, (a @ b).T, rtol=1e-2, atol=1e-2)

    def test_rebinding_index_tensors_are_rejected(self) -> None:
        config = helion.Config(block_sizes=[16, 16])
        x = torch.randn(32, 32, device=DEVICE)
        ia = torch.randint(0, 32, (32, 32), device=DEVICE)
        for kernel, index in (
            (gather_permuted_index_pair, ia.flip(0)),
            (gather_permuted_index_and_wrong_vector, ia[0]),
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                self._assert_rejected(
                    kernel, (x, ia, index), config, "support: an index tensor"
                )

    def test_tile_meeting_slice_is_rejected(self) -> None:
        """A tile meets a ``:`` slice on the operand's side and then the
        store's, on the operand's side only, and on the result's side only."""
        x = torch.randn(48, 16, device=DEVICE)
        y = torch.randn(16, 48, device=DEVICE)
        for kernel in (
            rebound_slice_plus_tile,
            rebound_tile_plus_slice,
            rebound_slice_plus_tile_sum,
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                self._assert_rejected(kernel, (x, y), None, "without a tile")


# ---------------------------------------------------------------------------
# Reductions over a block its own device loop walks in a lane loop
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def lane_chunk_softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n]
            e = torch.exp(v - v.amax(-1)[:, None])
            out[tile_m, tile_n] = e / e.sum(-1)[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_chunk_welford(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    mean_out = torch.empty([m], dtype=torch.float32, device=x.device)
    m2_out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        mean = hl.zeros([tile_m], dtype=torch.float32)
        m2 = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            valid = tile_n.index < n
            chunk = x[tile_m, tile_n]
            count = valid.to(torch.int32).sum()
            chunk_mean = chunk.sum(-1) / count
            centered = torch.where(valid, chunk - chunk_mean[:, None], 0.0)
            chunk_m2 = (centered * centered).sum(-1)
            delta = chunk_mean - mean
            new_cnt = cnt + count
            mean = mean + delta * (count / new_cnt)
            m2 = m2 + chunk_m2 + delta * delta * (cnt * count / new_cnt)
            cnt = new_cnt
        mean_out[tile_m] = mean
        m2_out[tile_m] = m2
    return mean_out, m2_out


@helion.kernel(backend="metal", static_shapes=True)
def lane_center_and_column_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    block_m = hl.register_block_size(m)
    out = torch.empty_like(x)
    sums = torch.empty(
        [(m + block_m - 1) // block_m, n], dtype=torch.float32, device=x.device
    )
    n = hl.specialize(n)
    for rows in hl.tile(m, block_size=block_m):
        acc = hl.zeros([n], dtype=torch.float32)
        for tile_m in hl.tile(rows.begin, rows.end):
            v = x[tile_m, :]
            centered = v - v.mean(-1)[:, None]
            acc = acc + centered.sum(0)
            out[tile_m, :] = centered * 2.0
        sums[rows.id, :] = acc
    return out, sums


@helion.kernel(backend="metal", static_shapes=True)
def lane_chunk_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([3, m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m], dtype=torch.float32)
        best = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        tiles = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            s = x[tile_m, tile_n].sum(-1)
            total = total + s
            best = torch.maximum(best, s)
            tiles = tiles + 1.0
        out[0, tile_m] = total
        out[1, tile_m] = best
        out[2, tile_m] = tiles
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_prod_min_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([3, m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        p = hl.full([tile_m], 1.0, dtype=torch.float32)
        lo = hl.full([tile_m], float("inf"), dtype=torch.float32)
        hi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n]
            p = p * torch.prod(v, -1)
            lo = torch.minimum(lo, v.amin(-1))
            hi = torch.maximum(hi, (-v).amax(-1))
        out[0, tile_m] = p
        out[1, tile_m] = lo
        out[2, tile_m] = hi
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_sum_per_k_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, n = x.size()
    k = y.size(0)
    out = torch.empty([b], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        acc = hl.zeros([tile_b], dtype=torch.float32)
        for _tile_k in hl.tile(k):
            for tile_n in hl.tile(n):
                acc = acc + x[tile_b, tile_n].sum(-1)
        out[tile_b] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_center_2d_inner_tile(x: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out = torch.empty_like(x)
    for tile_b in hl.tile(b):
        for tile_m, tile_n in hl.tile([m, n]):
            v = x[tile_b, tile_m, tile_n]
            out[tile_b, tile_m, tile_n] = v - v.amax(-1, keepdim=True)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_argmax_chunks(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            acc = acc + torch.argmax(x[tile_m, tile_n], -1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_sum_in_branch(x: torch.Tensor, flag: int) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            if flag > 0:
                acc = acc + x[tile_m, tile_n].sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_store_then_load(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            y[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            w = y[tile_m, tile_n]
            out[tile_m, tile_n] = w - w.sum(-1)[:, None]
    return y, out


@helion.kernel(backend="metal", static_shapes=True)
def lane_store_twice(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m] = x[tile_m, tile_n].sum(-1)
            out[tile_m] = x[tile_m, 0]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_store_beside_argument(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            acc = acc + x[tile_m, tile_n].sum(-1)
        sums[tile_m] = acc
    return sums


@helion.kernel(backend="metal", static_shapes=True)
def lane_stores_into_two_arguments(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            vals = (tile_n.index + 1).to(torch.float32)
            b[tile_m] = hl.zeros([tile_m], dtype=torch.float32) + vals.sum()
            a[tile_m] = hl.full([tile_m], 7.0, dtype=torch.float32)
    return a


@helion.kernel(backend="metal", static_shapes=True)
def lane_tile_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
            total = total + acc.sum(-1)
        out[tile_m] = total
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_loop_in_lane_body(x: torch.Tensor) -> torch.Tensor:
    b, k, n = x.size()
    out = torch.empty([b], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(b):
        total = hl.zeros([tile_b], dtype=torch.float32)
        for tile_k in hl.tile(k):
            part = hl.zeros([tile_b, tile_k], dtype=torch.float32)
            for tile_n in hl.tile(n):
                part = part + x[tile_b, tile_k, tile_n].sum(-1)
            total = total + part.sum(-1)
        out[tile_b] = total
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_chunks_of_row_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = x.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    total = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n, :].sum(-1)
            acc = acc + v.sum(-1)
            out[tile_m, tile_n] = v - v.amax(-1)[:, None]
        total[tile_m] = acc
    return out, total


@helion.kernel(backend="metal", static_shapes=True)
def lane_running_extrema(neg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    m, n = neg.size()
    out = torch.empty([2, m], dtype=torch.float32, device=neg.device)
    for tile_m in hl.tile(m):
        hi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        lo = hl.full([tile_m], float("inf"), dtype=torch.float32)
        for tile_n in hl.tile(n):
            hi = torch.maximum(hi, neg[tile_m, tile_n].amax(-1))
            lo = torch.minimum(lo, pos[tile_m, tile_n].amin(-1))
        out[0, tile_m] = hi
        out[1, tile_m] = lo
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_running_prod(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m], 1.0, dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc * x[tile_m, tile_n].prod(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_dot_of_a_computed_operand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k] * 2.0, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_sibling_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    k = y.size(1)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            total = total + x[tile_m, tile_n].sum(-1)
        maxima = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            maxima = maxima + y[tile_m, tile_k].amax(-1)
        out[tile_m] = total + maxima
    return out


def _chunks(x: torch.Tensor, block: int) -> list[torch.Tensor]:
    return list(x.split(block, dim=-1))


_LANE_CONFIG = helion.Config(block_sizes=[16, 16], num_threads=[16, 1])
_THREADS_AND_LANES = helion.Config(block_sizes=[16, 16], num_threads=[16, 4])
_LANE_MESSAGE = "{} reduction over an axis strided by a tile lane loop"


@_requires_darwin
class TestMetalLaneLoopReductions(unittest.TestCase):
    """A reduction over a tile block that its own device loop walks in a lane
    loop splits that lane loop into passes.

    ``num_threads`` of 1 on the inner block makes it pure lane; one between 1
    and the block size puts it on threads x lanes.  Each positive test also
    counts the ``for lane_`` loops of the generated code: one per pass, plus
    any grid lane loop around them.
    """

    def _run(
        self, kernel: helion.Kernel, config: helion.Config, *args: object
    ) -> tuple[object, int]:
        bound = kernel.bind(args)
        code = bound.to_code(config)
        self.assertNotIn(lane_reduction.MARKER, code)
        return bound.compile_config(config)(*args), code.count("for lane_")

    def _assert_rejected(
        self,
        kernel: helion.Kernel,
        config: helion.Config,
        message: str,
        *args: object,
    ) -> None:
        with self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)):
            kernel.bind(args).to_code(config)

    def test_dependent_reductions_and_a_store(self) -> None:
        """max, then a sum of exponentials that needs it, then a store that
        needs both: three passes, over a tail chunk (200 = 12 * 16 + 8)."""
        x = torch.randn(32, 200, device=DEVICE)
        expected = torch.cat([c.softmax(-1) for c in _chunks(x, 16)], -1)
        for static_shapes in (True, False):
            with self.subTest(static_shapes=static_shapes):
                kernel = helion.kernel(
                    lane_chunk_softmax.fn, backend="metal", static_shapes=static_shapes
                )
                out, lane_loops = self._run(kernel, _LANE_CONFIG, x)
                torch.testing.assert_close(out, expected)
                self.assertEqual(lane_loops, 3)

    def test_welford_chunks(self) -> None:
        """A masked integer count beside a sum, then a centred sum of squares
        that needs the chunk mean, merged into carries once per tile."""
        x = torch.randn(24, 100, device=DEVICE)
        config = helion.Config(block_sizes=[8, 16], num_threads=[8, 1])
        (mean, m2), lane_loops = self._run(lane_chunk_welford, config, x)
        torch.testing.assert_close(mean, x.mean(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            m2, (x - x.mean(-1, keepdim=True)).square().sum(-1), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(lane_loops, 2)

    def test_group_collective_in_two_passes(self) -> None:
        """The reduction input needs a mean over the threaded column dim, so
        the fold pass and the store's pass both run that group collective."""
        x = torch.randn(250, 128, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16], num_threads=[0, 1])
        (out, sums), lane_loops = self._run(lane_center_and_column_sums, config, x)
        centered = x - x.mean(-1, keepdim=True)
        torch.testing.assert_close(out, centered * 2.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            sums,
            torch.stack([c.sum(0) for c in centered.split(16)]),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(lane_loops, 2)

    def test_carries_update_once_per_tile(self) -> None:
        """A sum carry, a max carry with a -inf init and a lane-invariant tile
        counter: the counter counts 7 tiles of 16 over 100 columns."""
        x = torch.randn(32, 100, device=DEVICE)
        out, lane_loops = self._run(lane_chunk_sums, _LANE_CONFIG, x)
        chunk_sums = torch.stack([c.sum(-1) for c in _chunks(x, 16)], -1)
        torch.testing.assert_close(out[0], x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out[1], chunk_sums.amax(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out[2], torch.full_like(out[2], 7.0))
        self.assertEqual(lane_loops, 1)

    def test_prod_min_max(self) -> None:
        """The min and max folds propagate NaN, as torch does, and the max
        fold of all-negative chunks starts from -inf, tail chunk included."""
        x = 1.0 + 0.01 * torch.randn(16, 40, device=DEVICE)
        x[3, 21] = float("nan")
        config = helion.Config(block_sizes=[8, 16], num_threads=[8, 1])
        out, lane_loops = self._run(lane_prod_min_max, config, x)
        torch.testing.assert_close(out[0], x.prod(-1), equal_nan=True)
        torch.testing.assert_close(out[1], x.amin(-1), equal_nan=True)
        torch.testing.assert_close(out[2], (-x).amax(-1), equal_nan=True)
        self.assertEqual(lane_loops, 1)

    def test_grid_lanes_around_the_split(self) -> None:
        """A grid lane loop re-runs the carry initialisations with the body,
        so the split may run inside it."""
        x = torch.randn(64, 100, device=DEVICE)
        config = helion.Config(block_sizes=[64, 16], num_threads=[16, 1])
        out, lane_loops = self._run(lane_chunk_sums, config, x)
        torch.testing.assert_close(out[0], x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out[2], torch.full_like(out[2], 7.0))
        self.assertEqual(lane_loops, 2)

    def test_threaded_enclosing_device_loop(self) -> None:
        """An enclosing device loop on threads, not lanes, is allowed."""
        x = torch.randn(8, 40, device=DEVICE)
        y = torch.empty(12, device=DEVICE)
        config = helion.Config(block_sizes=[8, 4, 16], num_threads=[8, 4, 1])
        out, _ = self._run(lane_sum_per_k_tile, config, x, y)
        torch.testing.assert_close(out, x.sum(-1) * 3, rtol=1e-4, atol=1e-4)

    def test_threaded_block_beside_the_lane_block(self) -> None:
        """Only the reduced block must be off threads: m threaded, n in lanes."""
        x = torch.randn(4, 16, 40, device=DEVICE)
        config = helion.Config(block_sizes=[2, 8, 16], num_threads=[2, 8, 1])
        out, lane_loops = self._run(lane_center_2d_inner_tile, config, x)
        expected = torch.cat([c - c.amax(-1, keepdim=True) for c in _chunks(x, 16)], -1)
        torch.testing.assert_close(out, expected)
        self.assertEqual(lane_loops, 2)

    @_full_threadgroup
    def test_persistent_lane_fold_in_the_passes(self) -> None:
        """The persistent k dim, 1500 wide on 1024 threads, folds 2 lanes in
        both passes that recompute v.  The split runs inside k's lane loop,
        which ``unroll_lane_folds`` then jams into the passes; m's grid lane
        loop wraps both, hence 3 ``for lane_`` loops."""
        x = torch.randn(4, 24, 1500, device=DEVICE)
        config = helion.Config(block_sizes=[2, 16], num_threads=[2, 1])
        code = lane_chunks_of_row_sums.bind((x,)).to_code(config)
        self.assertRegex(code, r"for slice_lane_\d+ in range\(2\)")
        self.assertIn("_metal_lane_fold(", code)
        (out, total), lane_loops = self._run(lane_chunks_of_row_sums, config, x)
        v = x.sum(-1)
        expected = torch.cat([c - c.amax(-1, keepdim=True) for c in _chunks(v, 16)], -1)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(total, v.sum(-1), rtol=1e-4, atol=1e-3)
        self.assertEqual(lane_loops, 3)

    def test_extent_below_one_block(self) -> None:
        """5 columns in a block of 16: one tile with 11 masked lanes."""
        x = torch.randn(16, 5, device=DEVICE)
        out, _ = self._run(lane_chunk_softmax, _LANE_CONFIG, x)
        torch.testing.assert_close(out, x.softmax(-1))
        out, _ = self._run(lane_chunk_sums, _LANE_CONFIG, x)
        torch.testing.assert_close(out[0], x.sum(-1), rtol=1e-4, atol=1e-4)

    def test_threads_and_lanes(self) -> None:
        """n on 4 threads x 4 lanes: each fold combines its threads once per
        tile, beside the lane-invariant carries, with and without a tail."""
        for n, tiles in ((64, 4), (70, 5)):
            with self.subTest(n=n):
                x = torch.randn(32, n, device=DEVICE)
                out, lane_loops = self._run(lane_chunk_sums, _THREADS_AND_LANES, x)
                xd = x.cpu().double()
                chunk_sums = torch.stack([c.sum(-1) for c in _chunks(xd, 16)], -1)
                out = out.cpu().double()
                torch.testing.assert_close(out[0], xd.sum(-1), rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(
                    out[1], chunk_sums.amax(-1), rtol=1e-4, atol=1e-4
                )
                torch.testing.assert_close(out[2], torch.full_like(out[2], tiles))
                self.assertEqual(lane_loops, 1)

    def test_threads_and_lanes_two_stages(self) -> None:
        """A max fold, then a sum fold that needs it, then a store.  The
        span-64 collective goes through threadgroup memory."""
        for shape, config in (
            ((32, 70), _THREADS_AND_LANES),
            ((32, 200), helion.Config(block_sizes=[4, 128], num_threads=[4, 64])),
        ):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE)
                block = config.block_sizes[1]
                chunks = _chunks(x.cpu().double(), block)
                expected = torch.cat([c.softmax(-1) for c in chunks], -1)
                out, lane_loops = self._run(lane_chunk_softmax, config, x)
                torch.testing.assert_close(
                    out.cpu().double(), expected, rtol=1e-4, atol=1e-6
                )
                self.assertEqual(lane_loops, 3)

    def test_threads_and_lanes_welford(self) -> None:
        """Three folds in two stages, one of them a count over n alone."""
        x = torch.randn(32, 70, device=DEVICE)
        (mean, m2), lane_loops = self._run(lane_chunk_welford, _THREADS_AND_LANES, x)
        xd = x.cpu().double()
        torch.testing.assert_close(
            mean.cpu().double(), xd.mean(-1), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            m2.cpu().double(),
            (xd - xd.mean(-1, keepdim=True)).square().sum(-1),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(lane_loops, 2)

    def test_threads_and_lanes_extrema(self) -> None:
        """Masked lanes and threads start the max fold of all-negative data
        from -inf and the min fold of all-positive data from inf."""
        neg = -1.0 - torch.rand(16, 100, device=DEVICE)
        pos = 1.0 + torch.rand(16, 100, device=DEVICE)
        config = helion.Config(block_sizes=[8, 32], num_threads=[0, 8])
        out, lane_loops = self._run(lane_running_extrema, config, neg, pos)
        out = out.cpu().double()
        torch.testing.assert_close(out[0], neg.cpu().double().amax(-1))
        torch.testing.assert_close(out[1], pos.cpu().double().amin(-1))
        self.assertEqual(lane_loops, 1)

    def test_threads_and_lanes_prod(self) -> None:
        """Masked lanes, whose loads read 0, and masked threads start the
        prod fold from 1."""
        x = 1.0 + 0.05 * torch.randn(16, 100, device=DEVICE)
        config = helion.Config(block_sizes=[8, 32], num_threads=[0, 8])
        out, lane_loops = self._run(lane_running_prod, config, x)
        torch.testing.assert_close(
            out.cpu().double(), x.cpu().double().prod(-1), rtol=1e-4, atol=1e-4
        )
        self.assertEqual(lane_loops, 1)

    def test_threads_and_lanes_dot(self) -> None:
        """The K sum of a dot, with K on thread axis 2 above m and n, so the
        combine is strided; 100 = 6 * 16 + 4 leaves a tail.  Numerics only,
        so it holds whichever lowering the dot takes."""
        a = torch.randn(32, 100, device=DEVICE, dtype=torch.float16)
        b = torch.randn(100, 32, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[16, 16, 16], num_threads=[0, 0, 4])
        bound = lane_dot_of_a_computed_operand.bind((a, b))
        out = bound.compile_config(config)(a, b)
        expected = (a.cpu().double() * 2.0) @ b.cpu().double()
        torch.testing.assert_close(out.cpu().double(), expected, rtol=1e-4, atol=1e-4)

    def test_threads_and_lanes_narrower_than_launch(self) -> None:
        """The second loop's k runs on 4 threads x 8 lanes of an axis the first
        loop launches 8 wide; the 4 surplus threads add nothing."""
        x = torch.randn(16, 100, device=DEVICE)
        y = -1.0 - torch.rand(16, 70, device=DEVICE)
        config = helion.Config(block_sizes=[8, 32, 32], num_threads=[0, 8, 4])
        out, lane_loops = self._run(lane_sibling_sums, config, x, y)
        yd = y.cpu().double()
        maxima = torch.stack([c.amax(-1) for c in _chunks(yd, 32)], -1).sum(-1)
        torch.testing.assert_close(
            out.cpu().double(),
            x.cpu().double().sum(-1) + maxima,
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(lane_loops, 2)

    def test_argmax_rejected(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        for config in (_LANE_CONFIG, _THREADS_AND_LANES):
            with self.subTest(num_threads=config.num_threads):
                self._assert_rejected(
                    lane_argmax_chunks, config, _LANE_MESSAGE.format("argmax"), x
                )

    def test_reduction_in_a_branch_rejected(self) -> None:
        """Threads x lanes would also leave the collective in a branch that
        some threads may skip."""
        x = torch.randn(32, 64, device=DEVICE)
        for config in (_LANE_CONFIG, _THREADS_AND_LANES):
            with self.subTest(num_threads=config.num_threads):
                self._assert_rejected(
                    lane_sum_in_branch, config, _LANE_MESSAGE.format("sum"), x, 1
                )

    def test_store_then_load_rejected(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        self._assert_rejected(
            lane_store_then_load,
            _LANE_CONFIG,
            "y is stored and loaded in its lane loop",
            x,
        )

    def test_store_twice_rejected(self) -> None:
        """Without the check the stage-0 store would move before the stage-1
        store and lose to it."""
        x = torch.randn(32, 64, device=DEVICE)
        self._assert_rejected(
            lane_store_twice, _LANE_CONFIG, "out stored twice in its lane loop", x
        )

    def test_store_beside_an_argument_rejected(self) -> None:
        """``kernel(x, x)`` stores each tile of x before summing it; the split
        would load x in the fold pass, ahead of the final pass's store."""
        x = torch.randn(32, 40, device=DEVICE)
        self._assert_rejected(
            lane_store_beside_argument,
            _LANE_CONFIG,
            "out is stored in its lane loop and may share memory with x",
            x,
            x,
        )

    def test_stores_into_two_arguments_rejected(self) -> None:
        """``kernel(x, a, a)``: the split stores a, which needs no sum, in the
        first pass and b in the final pass, so b's value would win."""
        x = torch.randn(16, 16, device=DEVICE)
        self._assert_rejected(
            lane_stores_into_two_arguments,
            _LANE_CONFIG,
            "a is stored in its lane loop and may share memory with b",
            x,
            torch.empty(16, device=DEVICE),
            torch.empty(16, device=DEVICE),
        )

    def test_enclosing_device_lane_loop_rejected(self) -> None:
        """k in lanes re-runs the n loop, carry updates included, per k lane."""
        x = torch.randn(8, 40, device=DEVICE)
        y = torch.empty(12, device=DEVICE)
        for num_threads in ([8, 1, 1], [8, 2, 1], [8, 1, 4], [8, 2, 4]):
            with self.subTest(num_threads=num_threads):
                config = helion.Config(block_sizes=[8, 4, 16], num_threads=num_threads)
                self._assert_rejected(
                    lane_sum_per_k_tile, config, _LANE_MESSAGE.format("sum"), x, y
                )

    def test_per_lane_carry_rejected(self) -> None:
        """Block 0 is ``block_n`` (registered first), so n is in lanes."""
        x = torch.randn(32, 64, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16], num_threads=[1, 16])
        self._assert_rejected(
            lane_tile_carry,
            config,
            "per-lane value acc is carried across lanes",
            x,
        )

    def test_two_lane_loops_in_one_device_loop_rejected(self) -> None:
        x = torch.randn(4, 16, 40, device=DEVICE)
        for num_threads in ([2, 1, 1], [2, 2, 4]):
            with self.subTest(num_threads=num_threads):
                config = helion.Config(block_sizes=[2, 8, 16], num_threads=num_threads)
                self._assert_rejected(
                    lane_center_2d_inner_tile, config, _LANE_MESSAGE.format("max"), x
                )

    def test_device_loop_in_lane_body_rejected(self) -> None:
        x = torch.randn(8, 12, 40, device=DEVICE)
        config = helion.Config(block_sizes=[8, 4, 16], num_threads=[8, 1, 16])
        self._assert_rejected(
            lane_loop_in_lane_body,
            config,
            "lane reduction beside a statement the split cannot schedule",
            x,
        )

    def test_schedule_check(self) -> None:
        """The replay accepts the built schedule and rejects reordered ones."""
        x = torch.randn(32, 200, device=DEVICE)
        with patch.object(
            lane_reduction, "check_schedule", wraps=lane_reduction.check_schedule
        ) as check:
            lane_chunk_softmax.bind((x,)).to_code(_LANE_CONFIG)
        body, schedule = check.call_args.args
        lane_reduction.check_schedule(body, schedule)

        passes = [i for i, item in enumerate(schedule) if not isinstance(item, int)]
        first = schedule[passes[0]]
        assert isinstance(first, lane_reduction.LanePass)
        results = {body.reductions[fold].result for fold in first.folds}
        consumer = next(
            item
            for item in schedule[passes[0] + 1 :]
            if isinstance(item, int) and body.reads[item] & results
        )
        hoisted = [
            *schedule[: passes[0]],
            consumer,
            *(item for item in schedule[passes[0] :] if item != consumer),
        ]
        with self.assertRaisesRegex(exc.BackendUnsupported, "split reorders"):
            lane_reduction.check_schedule(body, hoisted)

        # The final pass without a definition an earlier pass also made: its
        # reader would see the earlier pass's last lane.
        final = schedule[-1]
        assert isinstance(final, lane_reduction.LanePass)
        recomputed = next(
            index
            for index in final.statements
            if any(
                index in item.statements
                for item in schedule[:-1]
                if isinstance(item, lane_reduction.LanePass)
            )
        )
        truncated = [
            *schedule[:-1],
            lane_reduction.LanePass(
                [index for index in final.statements if index != recomputed], []
            ),
        ]
        with self.assertRaisesRegex(exc.BackendUnsupported, "split reorders"):
            lane_reduction.check_schedule(body, truncated)

        # A fold input defined only after its pass.
        fold = first.folds[0]
        producer = next(
            index
            for index in first.statements
            if body.reductions[fold].input_name in body.writes[index]
        )
        moved = [
            lane_reduction.LanePass(
                [index for index in item.statements if index != producer], item.folds
            )
            if item is first
            else item
            for item in schedule[:-1]
        ]
        moved.append(lane_reduction.LanePass([*final.statements, producer], []))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "split reorders .* for the fold of"
        ):
            lane_reduction.check_schedule(body, moved)

        # The store left out; nothing reads it.
        store = next(index for index in final.statements if not body.writes[index])
        unstored = [
            *schedule[:-1],
            lane_reduction.LanePass(
                [index for index in final.statements if index != store], []
            ),
        ]
        with self.assertRaisesRegex(exc.BackendUnsupported, "split drops"):
            lane_reduction.check_schedule(body, unstored)


@helion.kernel(backend="metal", static_shapes=True)
def lane_product_m2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            valid = tile_n.index < n
            p = x[tile_m, tile_n].to(torch.float32) * y[tile_m, tile_n].to(
                torch.float32
            )
            mean = p.sum(-1) / valid.to(torch.int32).sum()
            centered = torch.where(valid, p - mean[:, None], 0.0)
            acc = acc + (centered * centered).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_center_scale(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n].to(torch.float32)
            centered = v - (v.sum(-1) / n)[:, None]
            out[tile_m, tile_n] = (centered * w[tile_n].to(torch.float32)).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_online_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """One query row per program attending over every key with an online
    softmax: each key tile's scores are a reduction over the head dim, and
    the narrow key tile is read again after the max."""
    rows, d = q.size()
    keys = k.size(0)
    out = torch.empty_like(q)
    for row in hl.grid(rows):
        qv = q[row, :].to(torch.float32) * scale
        m = hl.full([1], float("-inf"), dtype=torch.float32)
        denom = hl.zeros([1], dtype=torch.float32)
        acc = hl.zeros([d], dtype=torch.float32)
        for tile_s in hl.tile(keys):
            scores = (k[tile_s, :].to(torch.float32) * qv[None, :]).sum(-1)
            m_new = torch.maximum(m, scores.amax(-1, keepdim=True))
            p = torch.exp(scores - m_new)
            alpha = torch.exp(m - m_new)
            denom = denom * alpha + p.sum(-1, keepdim=True)
            acc = acc * alpha + (p[:, None] * v[tile_s, :].to(torch.float32)).sum(0)
            m = m_new
        out[row, :] = (acc / denom).to(out.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_counts(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([3, m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        positive = hl.zeros([tile_m], dtype=torch.int64)
        positive_i32 = hl.zeros([tile_m], dtype=torch.int64)
        negative_i64 = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n]
            positive = positive + (v > 0).sum(-1)
            positive_i32 = positive_i32 + (v > 0).to(torch.int32).sum(-1)
            negative_i64 = negative_i64 + (v < 0).to(torch.int64).sum(-1)
        out[0, tile_m] = positive
        out[1, tile_m] = positive_i32
        out[2, tile_m] = negative_i64
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_integer_sums(small: torch.Tensor, large: torch.Tensor) -> torch.Tensor:
    m, n = small.size()
    out = torch.empty([2, m], dtype=torch.int64, device=small.device)
    for tile_m in hl.tile(m):
        widened = hl.zeros([tile_m], dtype=torch.int64)
        plain = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            widened = widened + small[tile_m, tile_n].to(torch.int32).sum(-1)
            plain = plain + large[tile_m, tile_n].sum(-1)
        out[0, tile_m] = widened
        out[1, tile_m] = plain
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_count_max_and_float_sum(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    maxima = torch.empty([m], dtype=torch.int64, device=x.device)
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        most = hl.zeros([tile_m], dtype=torch.int64)
        total = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            positive = x[tile_m, tile_n] > 0
            most = most + positive.to(torch.int64).amax(-1)
            total = total + positive.sum(-1, dtype=torch.float32)
        maxima[tile_m] = most
        sums[tile_m] = total
    return maxima, sums


def _nan_tail(x: torch.Tensor, pad: int = 64) -> torch.Tensor:
    """``x`` on MPS, sliced from a buffer that holds NaN past its columns."""
    m, n = x.shape
    buffer = torch.full((m, n + pad), float("nan"), dtype=x.dtype)
    buffer[:, :n] = x
    return buffer.to(DEVICE)[:, :n]


def _lane_partials(code: str) -> list[str]:
    """The types of the lane folds' partials, in order."""
    return re.findall(r"^\s*lane_\w+ = (int|long|float)\(", code, re.MULTILINE)


def _welford_reference(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    weight64, bias64, x64 = (t.cpu().double() for t in (weight, bias, x))
    n = x.size(-1)
    return torch.nn.functional.layer_norm(x64, [n], weight64, bias64, 1e-5)


_WELFORD_THREADS_AND_LANES = helion.Config(
    block_sizes=[1, 512, 512], num_threads=[1, 64, 64]
)


@_requires_darwin
class TestMetalLaneArrays(unittest.TestCase):
    """A load narrower than 32 bits that several passes of a lane split
    recompute is loaded once, into a per-thread array the later passes read,
    while the split's arrays fit in 64 bytes per thread."""

    def _arrays(self, code: str) -> list[str]:
        return re.findall(r"_metal_lane_array\(('\w+', \d+)\)", code)

    def test_welford_keeps_its_row_chunk(self) -> None:
        """x, read by the mean pass and the centred-square pass: 8 lanes of
        bfloat on 64 threads and, at the default config, on one thread."""
        weight, bias, x = _welford_args(37, 300, torch.bfloat16)
        expected = _welford_reference(weight, bias, x)
        kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
        bound = kernel.bind((weight, bias, x))
        default = bound.config_spec.default_config()
        for config, lanes in ((_WELFORD_THREADS_AND_LANES, 8), (default, 16)):
            with self.subTest(config=config):
                code = bound.to_code(config)
                self.assertEqual(self._arrays(code), [f"'bfloat', {lanes}"])
                self.assertEqual(code.count("_metal_lane_put("), 1)
                self.assertEqual(code.count("_metal_lane_get("), 1)
                out = bound.compile_config(config)(weight, bias, x)
                torch.testing.assert_close(
                    out.cpu().double(), expected, atol=2e-2, rtol=2e-2
                )

    def test_two_loads_with_a_tail(self) -> None:
        """x and y both feed the second pass; 1000 columns leave a tail of
        lanes whose loads read NaN past the extent unless masked."""
        x, y = (torch.randn(64, 1000, dtype=torch.float16) for _ in range(2))
        config = helion.Config(block_sizes=[4, 1024], num_threads=[1, 128])
        args = (_nan_tail(x), _nan_tail(y))
        bound = lane_product_m2.bind(args)
        self.assertEqual(
            self._arrays(bound.to_code(config)), ["'half', 8", "'half', 8"]
        )
        p = x.double() * y.double()
        expected = (p - p.mean(-1, keepdim=True)).square().sum(-1)
        torch.testing.assert_close(
            bound.compile_config(config)(*args).cpu().double(),
            expected,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_32_bit_loads_reload(self) -> None:
        weight, bias, x = _welford_args(37, 300, torch.float32)
        kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
        bound = kernel.bind((weight, bias, x))
        self.assertEqual(self._arrays(bound.to_code(_WELFORD_THREADS_AND_LANES)), [])
        out = bound.compile_config(_WELFORD_THREADS_AND_LANES)(weight, bias, x)
        torch.testing.assert_close(
            out.cpu().double(),
            _welford_reference(weight, bias, x),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_arrays_over_the_byte_cap_reload(self) -> None:
        """64 lanes of bfloat are 128 bytes, past the cap."""
        weight, bias, x = _welford_args(5, 2048, torch.bfloat16)
        config = helion.Config(block_sizes=[1, 2048, 2048], num_threads=[1, 32, 32])
        kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
        self.assertEqual(
            self._arrays(kernel.bind((weight, bias, x)).to_code(config)), []
        )

    def test_the_cap_holds_per_split(self) -> None:
        """32 lanes of half are 64 bytes each: x fills the cap, y reloads."""
        x, y = (torch.randn(4, 2048, dtype=torch.float16) for _ in range(2))
        config = helion.Config(block_sizes=[2, 2048], num_threads=[1, 64])
        code = lane_product_m2.bind((x, y)).to_code(config)
        self.assertEqual(self._arrays(code), ["'half', 32"])

    def test_a_load_one_pass_reads_reloads(self) -> None:
        """Only the final pass reads w, so only x gets an array."""
        x = torch.randn(37, 300, device=DEVICE, dtype=torch.float16)
        w = torch.randn(300, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[4, 512], num_threads=[1, 64])
        bound = lane_center_scale.bind((x, w))
        code = bound.to_code(config)
        self.assertEqual(self._arrays(code), ["'half', 8"])
        self.assertIn("load_lanes", code)
        x64 = x.cpu().double()
        expected = (x64 - x64.mean(-1, keepdim=True)) * w.cpu().double()
        torch.testing.assert_close(
            bound.compile_config(config)(x, w).cpu().double(),
            expected,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_arrays_one_per_slice_lane(self) -> None:
        """At the default config the 128-wide head dim is 16 threads by 8
        slice lanes, and each key tile's bfloat keys go in an array indexed
        by the key lane.  The jam over the slice lanes gives each its own
        array: one shared array kept the last slice's keys for every slice.
        With 512 threads the head dim needs no slice lanes."""
        q = torch.randn(4, 128, device=DEVICE, dtype=torch.bfloat16)
        k, v = (torch.randn(64, 128, device=DEVICE, dtype=torch.bfloat16) for _ in "kv")
        scale = 128**-0.5
        qd, kd, vd = (t.cpu().double() for t in (q, k, v))
        expected = torch.softmax(qd @ kd.T * scale, -1) @ vd
        for config, slices in (
            (helion.Config(block_sizes=[32], num_warps=4), True),
            (helion.Config(block_sizes=[4], num_warps=16), False),
        ):
            with self.subTest(config=config):
                out, code, msl = _run_with_msl(
                    grid_online_attention, (q, k, v, scale), config
                )
                torch.testing.assert_close(
                    out.cpu().double(), expected, atol=1e-2, rtol=1e-2
                )
                self.assertEqual("slice_lane_" in code, slices)
                if slices:
                    self.assertIn("_metal_lane_put(", code)
                    self.assertRegex(msl, r"bfloat \w+_lanes__lane7\[")


@_requires_darwin
class TestMetalLaneCounts(unittest.TestCase):
    """A lane-split sum of a count, a bool widened to an integer dtype with
    masked elements 0, folds and combines in int32 rather than torch's
    int64, while the block has fewer than 2**31 elements."""

    def test_counts_fold_in_int32(self) -> None:
        """A bool sum and bool casts to int32 and int64, on 32 threads x 8
        lanes and on pure lanes, over a 300-column tail."""
        x = torch.randn(37, 300)
        expected = torch.stack([(x > 0).sum(-1), (x > 0).sum(-1), (x < 0).sum(-1)])
        for num_threads in ([1, 32], [1, 1]):
            with self.subTest(num_threads=num_threads):
                config = helion.Config(block_sizes=[4, 256], num_threads=num_threads)
                args = (_nan_tail(x),)
                bound = lane_counts.bind(args)
                self.assertEqual(_lane_partials(bound.to_code(config)), ["int"] * 3)
                out = bound.compile_config(config)(*args)
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_welford_counts_its_masked_columns_in_int32(self) -> None:
        weight, bias, x = _welford_args(37, 300, torch.bfloat16)
        kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
        code = kernel.bind((weight, bias, x)).to_code(_WELFORD_THREADS_AND_LANES)
        self.assertEqual(_lane_partials(code), ["int", "float", "float"])

    def test_integer_sums_stay_int64(self) -> None:
        """int8 widened to int32 is not a count, and int32 values near 2**31
        would overflow an int32 partial."""
        small = torch.randint(-128, 128, (37, 300), dtype=torch.int8)
        large = torch.randint(2**30, 2**31 - 1, (37, 300), dtype=torch.int32)
        config = helion.Config(block_sizes=[4, 256], num_threads=[1, 32])
        bound = lane_integer_sums.bind((small.to(DEVICE), large.to(DEVICE)))
        self.assertEqual(_lane_partials(bound.to_code(config)), ["long", "long"])
        out = bound.compile_config(config)(small.to(DEVICE), large.to(DEVICE))
        expected = torch.stack([small.long().sum(-1), large.long().sum(-1)])
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_count_max_and_float_sum_keep_their_types(self) -> None:
        x = torch.randn(37, 300, device=DEVICE)
        config = helion.Config(block_sizes=[4, 256], num_threads=[1, 32])
        bound = lane_count_max_and_float_sum.bind((x,))
        self.assertEqual(_lane_partials(bound.to_code(config)), ["long", "float"])
        maxima, sums = bound.compile_config(config)(x)
        positive = (x > 0).cpu()
        torch.testing.assert_close(
            maxima.cpu(),
            torch.stack([c.amax(-1) for c in _chunks(positive.long(), 256)]).sum(0),
        )
        torch.testing.assert_close(sums.cpu(), positive.float().sum(-1))

    def test_blocks_of_2_31_elements_stay_int64(self) -> None:
        for width, partial in ((2**30, "int"), (2**31, "long")):
            with self.subTest(width=width):
                x = torch.empty(1, width, device=torch.device("meta"))
                config = helion.Config(block_sizes=[1, width], num_threads=[1, 1024])
                code = lane_counts.bind((x,)).to_code(config)
                self.assertEqual(_lane_partials(code), [partial] * 3)


@helion.kernel(backend="metal", static_shapes=True)
def scale_cols(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] * w[tn][None, :] + 1.0
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_tile_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two passes over the row with their own blocks, so their own thread
    extents; the first stores each tile minus its sum."""
    m, n = x.size()
    centred = torch.empty_like(x)
    doubled = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            v = x[tm, tn]
            centred[tm, tn] = v - v.sum(-1)[:, None]
        for tn in hl.tile(n):
            doubled[tm, tn] = x[tm, tn] * 2.0
    return centred, doubled


@helion.kernel(backend="metal", static_shapes=True)
def mm_scale_cols(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = (acc * w[tn][None, :]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def grid_mm_scale_cols(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``mm_scale_cols`` under ``hl.grid``."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for _ in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
            out[tm, tn] = (acc * w[tn][None, :]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def center_last(x: torch.Tensor) -> torch.Tensor:
    m, j, _ = x.size()
    out = torch.empty_like(x)
    for tm, tj in hl.tile([m, j]):
        row = x[tm, tj, :]
        out[tm, tj, :] = row - row.mean(-1, keepdim=True)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def flat_add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0)):
        out[t] = x[t] + 1.0
    return out


def _tile_masks(code: str) -> list[str]:
    return sorted(set(re.findall(r"\bmask_\d+\b", code)))


@_requires_darwin
class TestMetalTileMaskElision(unittest.TestCase):
    """A tile whose block divides its extent drops its bounds mask when the
    launch is uniform: every thread axis of every strategy has one extent,
    and no MPP op or free ``hl.arange`` claims threads of its own."""

    def test_welford_drops_every_mask(self) -> None:
        weight, bias, x = _welford_args(64, 1024, torch.bfloat16)
        config = helion.Config(block_sizes=[16, 1024, 128], num_threads=[1, 128, 128])
        bound = helion.kernel(welford.fn, backend="metal", static_shapes=True).bind(
            (weight, bias, x)
        )
        code = bound.to_code(config)
        self.assertEqual(_tile_masks(code), [])
        self.assertNotIn("_mask_to", code)
        out = bound.compile_config(config)(weight, bias, x)
        torch.testing.assert_close(
            out.cpu().double(),
            _welford_reference(weight, bias, x),
            atol=2e-2,
            rtol=2e-2,
        )

    @_full_threadgroup
    def test_2d_tile_on_two_thread_axes(self) -> None:
        x = torch.randn(64, 256, device=DEVICE)
        w = torch.randn(256, device=DEVICE)
        config = helion.Config(block_sizes=[32, 32], num_threads=[32, 32])
        bound = scale_cols.bind((x, w))
        self.assertEqual(_tile_masks(bound.to_code(config)), [])
        torch.testing.assert_close(bound.compile_config(config)(x, w), x * w + 1.0)

    def test_tails_keep_their_masks(self) -> None:
        """37 rows and 300 columns divide no block of the default config."""
        weight, bias, x = _welford_args(37, 300, torch.bfloat16)
        bound = helion.kernel(welford.fn, backend="metal", static_shapes=True).bind(
            (weight, bias, x)
        )
        config = bound.config_spec.default_config()
        self.assertEqual(
            _tile_masks(bound.to_code(config)), ["mask_0", "mask_1", "mask_2"]
        )
        out = bound.compile_config(config)(weight, bias, x)
        torch.testing.assert_close(
            out.cpu().double(),
            _welford_reference(weight, bias, x),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_softmax_rows_on_one_simd_group_drop_their_masks(self) -> None:
        for config in (
            helion.Config(block_sizes=[1, 128], num_threads=[0, 32]),
            helion.Config(block_sizes=[8, 128], num_threads=[1, 32]),
        ):
            with self.subTest(config=config):
                x = torch.randn(64, 384, device=DEVICE, dtype=torch.float16)
                bound = helion.kernel(
                    softmax_two_pass.fn, backend="metal", static_shapes=True
                ).bind((x,))
                code = bound.to_code(config)
                self.assertEqual(_tile_masks(code), [])
                self.assertNotIn("_mask_to", code)
                torch.testing.assert_close(
                    bound.compile_config(config)(x).cpu().double(),
                    torch.softmax(x.cpu().double(), dim=1),
                    atol=1e-3,
                    rtol=1e-3,
                )

    def test_softmax_column_tail_keeps_its_masks(self) -> None:
        """300 columns are no multiple of the 128-wide tile; the row tile is."""
        x = torch.randn(64, 300, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[1, 128], num_threads=[0, 32])
        bound = helion.kernel(
            softmax_two_pass.fn, backend="metal", static_shapes=True
        ).bind((x,))
        code = bound.to_code(config)
        self.assertEqual(_tile_masks(code), ["mask_1", "mask_2"])
        self.assertIn("_mask_to", code)
        torch.testing.assert_close(
            bound.compile_config(config)(x).cpu().double(),
            torch.softmax(x.cpu().double(), dim=1),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_reduction_with_its_own_thread_count_keeps_tile_masks(self) -> None:
        """The 64-thread reduction over the row shares the launch with the
        tile's 4- or 16-thread axes."""
        x = torch.randn(16, 16, 64, device=DEVICE)
        bound = center_last.bind((x,))
        expected = x.cpu().double() - x.cpu().double().mean(-1, keepdim=True)
        for config in (
            helion.Config(block_sizes=[4, 4], num_threads=[4, 4]),
            bound.config_spec.default_config(),
        ):
            with self.subTest(config=config):
                self.assertEqual(
                    _tile_masks(bound.to_code(config)), ["mask_0", "mask_1"]
                )
                torch.testing.assert_close(
                    bound.compile_config(config)(x).cpu().double(), expected
                )

    def test_sibling_passes_with_different_extents_keep_their_masks(self) -> None:
        """The 8-wide pass runs on the 32 threads its sibling launches, though
        the dispatch plans the two passes on separate axes."""
        x = torch.randn(4, 1024, device=DEVICE)
        config = helion.Config(block_sizes=[1, 8, 32], num_threads=[1, 0, 0])
        bound = sibling_tile_sums.bind((x,))
        self.assertEqual(
            _tile_masks(bound.to_code(config)), ["mask_0", "mask_1", "mask_2"]
        )
        centred, doubled = bound.compile_config(config)(x)
        chunks = x.cpu().double().view(4, -1, 8)
        expected = (chunks - chunks.sum(-1, keepdim=True)).view(4, 1024)
        torch.testing.assert_close(centred.cpu(), expected.float())
        torch.testing.assert_close(doubled, x * 2.0)

    def test_mpp_kernels_keep_their_masks(self) -> None:
        """A cooperative accumulator holds the tile in MPP's layout, so the
        Scalar statements its element loop re-runs keep their masks."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        w = torch.randn(64, device=DEVICE)
        expected = (x.cpu().double() @ y.cpu().double()) * w.cpu().double()
        for kernel, config, masks in (
            (
                mm_scale_cols,
                helion.Config(
                    block_sizes=[32, 32, 32], num_threads=[32, 1, 0], num_warps=4
                ),
                ["mask_0", "mask_1", "mask_2"],
            ),
            (
                grid_mm_scale_cols,
                helion.Config(block_sizes=[16, 16, 16], num_threads=[16, 16, 1]),
                ["mask_1", "mask_2", "mask_3"],
            ),
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                bound = kernel.bind((x, y, w))
                self.assertEqual(_tile_masks(bound.to_code(config)), masks)
                torch.testing.assert_close(
                    bound.compile_config(config)(x, y, w).cpu().double(),
                    expected,
                    atol=1e-3,
                    rtol=1e-3,
                )

    def test_flattened_tile_keeps_its_mask_after_a_uniform_config(self) -> None:
        """A flattened strategy reads ``force_tile_mask`` when it is built,
        before ``pre_codegen`` judges the new config's launch."""
        x = torch.randn(1024, device=DEVICE)
        second = helion.Config(block_sizes=[512])
        fresh = helion.kernel(flat_add_one.fn, backend="metal", static_shapes=True)
        expected = fresh.bind((x,)).to_code(second)
        reused = helion.kernel(flat_add_one.fn, backend="metal", static_shapes=True)
        bound = reused.bind((x,))
        bound.to_code(helion.Config(block_sizes=[256]))
        self.assertEqual(bound.to_code(second), expected)
        self.assertEqual(_tile_masks(expected), ["mask_0"])


# ---------------------------------------------------------------------------
# Reductions over a persistent dim walked in a lane loop
# ---------------------------------------------------------------------------
# The inner tile loop keeps the ``:`` dim persistent (the spec offers no
# reduction loop for it), so a dim wider than the threads the tiles leave is
# walked in a lane loop and its reductions fold each thread's lanes.


@helion.kernel(backend="metal")
def nested_rownorm(x: torch.Tensor) -> torch.Tensor:
    m_block = hl.register_block_size(x.size(0))
    out = torch.empty_like(x)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :].to(torch.float32)
            out[tile_m, :] = (xt * xt.sum(-1)[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal")
def nested_softmax(x: torch.Tensor) -> torch.Tensor:
    m_block = hl.register_block_size(x.size(0))
    out = torch.empty_like(x)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :].to(torch.float32)
            e = torch.exp(xt - torch.amax(xt, dim=-1)[:, None])
            out[tile_m, :] = (e / e.sum(-1)[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal")
def nested_rowmin_prod(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    lo = x.new_empty([x.size(0)])
    prod = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            lo[tile_m] = torch.amin(xt, dim=-1)
            prod[tile_m] = torch.prod(1.0 + 0.001 * xt, dim=-1)
    return lo, prod


@helion.kernel(backend="metal")
def nested_argmax(x: torch.Tensor) -> torch.Tensor:
    m_block = hl.register_block_size(x.size(0))
    out = x.new_empty([x.size(0)], dtype=torch.int64)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            out[tile_m] = torch.argmax(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal")
def nested_any(x: torch.Tensor) -> torch.Tensor:
    m_block = hl.register_block_size(x.size(0))
    out = x.new_empty([x.size(0)], dtype=torch.bool)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            out[tile_m] = torch.any(x[tile_m, :] > 2.5, dim=-1)
    return out


@helion.kernel(backend="metal")
def tile16_rowsum(x: torch.Tensor) -> torch.Tensor:
    """An explicit 16-row tile, charged 16 threads of the budget, beside a
    row sum."""
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0), block_size=16):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(backend="metal")
def carry_then_fold(x: torch.Tensor) -> torch.Tensor:
    """A full-width carry across a row loop, folded after the loop."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([(x.size(0) + m_block - 1) // m_block, n])
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n)
        for row in hl.grid(tile_c.begin, tile_c.end):
            acc = acc + x[row, :]
        out[tile_c.id, :] = acc * acc.sum(-1)
    return out


@helion.kernel(backend="metal")
def colsum_threaded(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A row-mean fold, and a column sum over an explicit 16-row tile.

    The explicit tile is charged threads rather than lane-looped, so its
    combine of a lane-dependent value runs once per lane.
    """
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end, block_size=16):
            xt = x[tile_m, :].to(torch.float32)
            y = xt - xt.mean(-1)[:, None]
            acc += y.sum(0)
            out[tile_m, :] = y.to(x.dtype)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def count_beside_fold(x: torch.Tensor) -> torch.Tensor:
    """A lane-independent carry beside a fold."""
    m_block = hl.register_block_size(x.size(0))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            count = hl.zeros([tile_m], dtype=torch.float32)
            for _ in hl.grid(4):
                count = count + 1.0
            out[tile_m] = x[tile_m, :].sum(-1) + count
    return out


@helion.kernel(backend="metal")
def nested_dot_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``nested_rownorm``'s layout with a dot over the ``:`` dim."""
    m_block = hl.register_block_size(x.size(0))
    out = torch.empty([x.size(0), 1], dtype=torch.float32, device=x.device)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            out[tile_m, :] = hl.dot(x[tile_m, :], w[:, :]) * 2.0
    return out


@helion.kernel(backend="metal")
def dot_chained_minus_rowsum(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_chained`` with a row sum over the first dot's K."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        qs = q[tile_m, :].to(torch.float32).sum(-1)
        for tile_n in hl.tile(kt.size(1)):
            p = torch.exp((hl.dot(q[tile_m, :], kt[:, tile_n]) - qs[:, None]) * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_uniform_k(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``dot_chained`` whose first dot's operands are constant along K."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(kt.size(1)):
            p = hl.dot(
                hl.full([tile_m, d], 0.5, dtype=torch.float32),
                hl.full([d, tile_n], 1.0, dtype=torch.float32),
            )
            acc = hl.dot(p, v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_chained_minus_rowsum_bf16(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_chained_minus_rowsum`` with the first product rounded to
    bfloat16."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        qs = q[tile_m, :].to(torch.float32).sum(-1)
        for tile_n in hl.tile(kt.size(1)):
            s = hl.dot(q[tile_m, :], kt[:, tile_n], out_dtype=torch.bfloat16)
            p = torch.exp((s - qs[:, None]) * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_uniform_k_bf16(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_uniform_k`` with the first product rounded to bfloat16."""
    d = hl.specialize(q.size(1))
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(kt.size(1)):
            p = hl.dot(
                hl.full([tile_m, d], 0.5, dtype=torch.float32),
                hl.full([d, tile_n], 1.0, dtype=torch.float32),
                out_dtype=torch.bfloat16,
            )
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@helion.kernel(backend="metal")
def dot_plus_rowsum(c: torch.Tensor, pt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A dot over one ``:`` dim and a row sum over another."""
    hl.specialize(c.size(1))
    hl.specialize(w.size(1))
    out = torch.empty([c.size(0), pt.size(1)], dtype=torch.float32, device=c.device)
    for tile_m, tile_n in hl.tile([c.size(0), pt.size(1)]):
        ws = w[tile_m, :].sum(-1)
        out[tile_m, tile_n] = (
            hl.dot(c[tile_m, :], pt[:, tile_n], out_dtype=torch.float32) + ws[:, None]
        )
    return out


@helion.kernel(backend="metal")
def dot16_plus_rowsum(
    c: torch.Tensor, pt: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``dot_plus_rowsum`` with a float16 dot, which keeps it scalar."""
    hl.specialize(c.size(1))
    hl.specialize(w.size(1))
    out = torch.empty([c.size(0), pt.size(1)], dtype=torch.float32, device=c.device)
    for tile_m, tile_n in hl.tile([c.size(0), pt.size(1)]):
        ws = w[tile_m, :].sum(-1)
        out[tile_m, tile_n] = (
            hl.dot(c[tile_m, :], pt[:, tile_n], out_dtype=torch.float16).to(
                torch.float32
            )
            + ws[:, None]
        )
    return out


@helion.kernel(backend="metal", static_shapes=False)
def dot_chained_dynamic(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``dot_chained`` with a runtime d."""
    d = q.size(1)
    out = torch.empty([q.size(0), d], dtype=torch.float32, device=q.device)
    for tile_m in hl.tile(q.size(0)):
        acc = hl.zeros([tile_m, d], dtype=torch.float32)
        for tile_n in hl.tile(kt.size(1)):
            p = torch.exp(hl.dot(q[tile_m, :], kt[:, tile_n]) * 0.125)
            acc = hl.dot(p.to(v.dtype), v[tile_n, :], acc=acc)
        out[tile_m, :] = acc
    return out


@_requires_darwin
class TestMetalLaneFold(unittest.TestCase):
    """Reductions over a persistent dim wider than its threads.

    Each thread walks its share of the dim in a lane loop, folds its lanes,
    then combines the folds across threads.
    """

    def _run(
        self,
        kernel: helion.Kernel,
        config: helion.Config,
        args: tuple[torch.Tensor, ...],
        lanes: int,
    ) -> tuple[object, str]:
        """Run *kernel* at *config*, checking it folds *lanes* lanes, and
        return its result and MSL."""
        result, code, msl = _run_with_msl(kernel, args, config)
        self.assertIn("_metal_lane_fold(", code)
        self.assertRegex(code, rf"for slice_lane_\d+ in range\({lanes}\)")
        self.assertIn("__lane1", msl)
        self.assertNotIn("slice_lane_", msl)
        if isinstance(result, tuple):
            return tuple(t.cpu() for t in result), msl
        return result.cpu(), msl

    def test_rownorm(self) -> None:
        """2 lanes, 3 lanes with a masked tail, and 12 lanes on one thread."""
        for n, config, lanes in (
            (2048, helion.Config(block_sizes=[32, 32], num_warps=32), 2),
            (3000, helion.Config(block_sizes=[32, 32], num_warps=32), 3),
            (
                12,
                helion.Config(block_sizes=[32, 32], num_threads=[32, 32], num_warps=32),
                12,
            ),
        ):
            with self.subTest(n=n):
                x = torch.randn(64, n, device=DEVICE)
                out, msl = self._run(nested_rownorm, config, (x,), lanes)
                xd = x.cpu().double()
                torch.testing.assert_close(
                    out, (xd * xd.sum(-1, keepdim=True)).float(), rtol=1e-4, atol=1e-4
                )
                if lanes == 12:
                    # A single thread holds the whole dim: nothing to combine.
                    self.assertNotIn("helion_red::", msl)

    def test_dependent_folds(self) -> None:
        """Softmax: a max fold, then an exp-sum fold that depends on it."""
        config = helion.Config(block_sizes=[32, 32], num_warps=32)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(64, 4100, device=DEVICE, dtype=dtype)
                out, _ = self._run(nested_softmax, config, (x,), 5)
                torch.testing.assert_close(
                    out,
                    torch.softmax(x.cpu().double(), -1).to(dtype),
                    rtol=2e-3,
                    atol=1e-5,
                )

    def test_min_and_prod_folds(self) -> None:
        config = helion.Config(block_sizes=[32, 32], num_warps=32)
        x = torch.randn(64, 1500, device=DEVICE)
        (lo, prod), _ = self._run(nested_rowmin_prod, config, (x,), 2)
        xd = x.cpu().double()
        torch.testing.assert_close(lo, xd.amin(-1).float())
        torch.testing.assert_close(
            prod, (1.0 + 0.001 * xd).prod(-1).float(), rtol=1e-4, atol=1e-4
        )

    def test_lane_cap(self) -> None:
        """64 lanes per thread fold; 65 are rejected."""
        config = helion.Config(block_sizes=[16], num_warps=4)
        x = torch.randn(16, 4096, device=DEVICE)
        out, _ = self._run(tile16_rowsum, config, (x,), 64)
        torch.testing.assert_close(
            out, x.cpu().double().sum(-1).float(), rtol=1e-4, atol=1e-4
        )
        x = torch.randn(16, 4097, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"persistent reduction over 4097 elements would need 65 lanes per "
            r"thread on 64 threads \(max 64\)",
        ):
            tile16_rowsum.bind((x,)).to_code(config)

    def test_carry_folded_after_its_loop(self) -> None:
        """A full-width carry becomes one scalar per lane across the loop."""
        config = helion.Config(block_sizes=[16], num_warps=32)
        x = torch.randn(16, 2048, device=DEVICE)
        out, _ = self._run(carry_then_fold, config, (x,), 2)
        acc = x.cpu().double().sum(0, keepdim=True)
        torch.testing.assert_close(out, (acc * acc.sum()).float(), rtol=1e-4, atol=1e-3)

    def test_replicated_tile_combine(self) -> None:
        """A combine over a threaded tile of a lane-dependent value runs once
        per lane, back to back on one scratch buffer."""
        # 64 columns reserve 32 of 512 threads (2 lanes each), which leaves
        # the 16-row tile one thread per row.
        config = helion.Config(block_sizes=[32], num_threads=[1], num_warps=16)
        for (m, n), dtype, lanes in (
            ((64, 64), torch.float32, 2),
            ((64, 64), torch.float16, 2),
            ((50, 64), torch.float32, 2),
        ):
            with self.subTest(m=m, n=n, dtype=dtype):
                x = torch.randn(m, n, device=DEVICE, dtype=dtype)
                (out, cols), msl = self._run(colsum_threaded, config, (x,), lanes)
                xd = x.cpu().double()
                y = xd - xd.mean(-1, keepdim=True)
                tol = 1e-4 if dtype == torch.float32 else 2e-3
                torch.testing.assert_close(out, y.to(dtype), rtol=tol, atol=tol)
                torch.testing.assert_close(
                    cols.sum(0), y.sum(0).float(), rtol=tol, atol=tol
                )
                tile_combines = [
                    partials
                    for op, _, partials, axis, span in _group_reductions(msl)
                    if (op, axis, span) == ("sum", 1, 16)
                ]
                self.assertEqual(len(tile_combines), lanes)
                self.assertEqual(len(set(tile_combines)), 1)

    def test_lane_independent_carry_runs_once(self) -> None:
        config = helion.Config(block_sizes=[32, 32], num_warps=32)
        x = torch.randn(64, 2048, device=DEVICE)
        out, _ = self._run(count_beside_fold, config, (x,), 2)
        torch.testing.assert_close(
            out, (x.cpu().double().sum(-1) + 4).float(), rtol=1e-4, atol=1e-4
        )

    @_full_threadgroup
    def test_dot_k(self) -> None:
        """2 lanes on 1024 threads, for a K the matmul facts name."""
        x = torch.randn(32, 2048, device=DEVICE)
        y = torch.randn(2048, 32, device=DEVICE)
        for block_sizes in ([8, 8], [16, 16]):
            with self.subTest(block_sizes=block_sizes):
                out, _ = self._run(
                    matmul_full_k, helion.Config(block_sizes=block_sizes), (x, y), 2
                )
                torch.testing.assert_close(
                    out, _matmul_reference(x, y), rtol=1e-4, atol=1e-3
                )

    @_full_threadgroup
    def test_dot_k_unnamed_by_matmul_facts(self) -> None:
        """``dot_chained_bf16``'s d is K of one dot and N of the other, so the
        matmul facts name no K: 2 lanes, and 3 with a tail.  The kernel rounds
        ``p`` to bfloat16 from a float32 exp, the reference from a double one,
        so an element can land one bfloat16 step (about 0.008 near 1) apart."""
        config = helion.Config(block_sizes=[16, 16], num_threads=[4, 16])
        for (m, d, n), dtype, lanes in (
            ((32, 32, 64), torch.bfloat16, 2),
            ((48, 40, 72), torch.bfloat16, 3),
        ):
            with self.subTest(d=d, dtype=dtype):
                q = torch.randn(m, d, device=DEVICE, dtype=dtype) / 4
                kt = torch.randn(d, n, device=DEVICE, dtype=dtype) / 4
                v = torch.randn(n, d, device=DEVICE, dtype=dtype)
                out, _ = self._run(dot_chained_bf16, config, (q, kt, v), lanes)
                s = _matmul_reference(q, kt).to(torch.bfloat16).double()
                p = torch.exp(s * 0.125).to(dtype)
                torch.testing.assert_close(
                    out, _matmul_reference(p, v), rtol=1e-4, atol=3e-2
                )

    def test_dot_k_rows(self) -> None:
        """12 lanes on one thread, and 3 lanes with a masked tail."""
        for k, dtype, config, lanes in (
            (
                12,
                torch.float32,
                helion.Config(block_sizes=[32, 32], num_threads=[32, 32], num_warps=32),
                12,
            ),
            (
                3000,
                torch.float16,
                helion.Config(block_sizes=[32, 32], num_warps=32),
                3,
            ),
        ):
            with self.subTest(k=k):
                x = torch.randn(64, k, device=DEVICE, dtype=dtype)
                w = torch.randn(k, 1, device=DEVICE, dtype=dtype)
                out, msl = self._run(nested_dot_rows, config, (x, w), lanes)
                torch.testing.assert_close(
                    out, _matmul_reference(x, w) * 2, rtol=1e-4, atol=1e-3
                )
                if lanes == 12:
                    self.assertNotIn("helion_red::", msl)

    @_full_threadgroup
    def test_dot_k_beside_a_fold(self) -> None:
        """A row sum folds d's lanes from strategy creation, and the dot over
        d asks for the fold again."""
        q = torch.randn(32, 40, device=DEVICE)
        kt = torch.randn(40, 64, device=DEVICE)
        v = torch.randn(64, 40, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16], num_threads=[4, 16])
        out, _ = self._run(dot_chained_minus_rowsum_bf16, config, (q, kt, v), 3)
        qd, ktd, vd = (t.cpu().double() for t in (q, kt, v))
        s = (qd @ ktd).to(torch.bfloat16).double()
        p = torch.exp((s - qd.sum(-1, keepdim=True)) * 0.125)
        torch.testing.assert_close(out, (p @ vd).float(), rtol=1e-4, atol=1e-4)

    @_full_threadgroup
    def test_dot_k_uniform_operands_tail(self) -> None:
        """Operands constant along K still add nothing past it: the last of 3
        lanes has 8 of 16 positions live, and an unmasked tail would sum
        0.5 over 48 positions."""
        q = torch.randn(32, 40, device=DEVICE)
        kt = torch.randn(40, 64, device=DEVICE)
        v = torch.randn(64, 40, device=DEVICE)
        config = helion.Config(block_sizes=[16, 16], num_threads=[4, 16])
        out, _ = self._run(dot_uniform_k_bf16, config, (q, kt, v), 3)
        p = torch.full((32, 64), 0.5 * 40, dtype=torch.float64)
        torch.testing.assert_close(
            out, (p @ v.cpu().double()).float(), rtol=1e-4, atol=1e-4
        )

    def test_chained_dots_through_mpp(self) -> None:
        """The chained dots of the tests above, in float32 and bfloat16,
        take the MPP planner: the second dot's lhs is staged from the first
        op's element loop."""
        cases = (
            (dot_chained, (20, 32, 40), torch.bfloat16, [2, 16], 4),
            (dot_chained, (32, 32, 64), torch.bfloat16, [4, 16], 32),
            (dot_chained, (48, 40, 72), torch.float32, [4, 16], 32),
            (dot_chained_minus_rowsum, (32, 40, 64), torch.float32, [4, 16], 32),
            (dot_uniform_k, (32, 40, 64), torch.float32, [4, 16], 32),
        )
        for kernel, (m, d, n), dtype, threads, warps in cases:
            config = helion.Config(
                block_sizes=[16, 16], num_threads=threads, num_warps=warps
            )
            with self.subTest(kernel=kernel.name, shape=(m, d, n), dtype=dtype):
                q = torch.randn(m, d, device=DEVICE, dtype=dtype) / 4
                kt = torch.randn(d, n, device=DEVICE, dtype=dtype) / 4
                v = torch.randn(n, d, device=DEVICE, dtype=dtype)
                out, code, msl = _run_with_msl(kernel, (q, kt, v), config)
                self.assertIn("_helion_mpp_run(", code)
                self.assertNotIn("dot_product", code)
                self.assertEqual(msl.count("matmul2d_descriptor("), 2)
                qd, ktd, vd = (t.cpu().double() for t in (q, kt, v))
                if kernel is dot_uniform_k:
                    p = torch.full((m, n), 0.5 * d, dtype=torch.float64)
                elif kernel is dot_chained_minus_rowsum:
                    p = torch.exp((qd @ ktd - qd.sum(-1, keepdim=True)) * 0.125)
                else:
                    p = torch.exp(qd @ ktd * 0.125)
                torch.testing.assert_close(
                    out.cpu().double(),
                    p.to(dtype).double() @ vd,
                    rtol=2e-2,
                    atol=2e-2,
                )

    def test_chained_dots_past_the_jam(self) -> None:
        """At four warps, the threads left for d make the lane loop around
        the second op run 80 copies of its Scalar work, more than a jam
        holds; the scalar contraction compiled these."""
        config = helion.Config(block_sizes=[16, 16], num_threads=[4, 16], num_warps=4)
        q = torch.randn(32, 40, device=DEVICE)
        kt = torch.randn(40, 64, device=DEVICE)
        v = torch.randn(64, 40, device=DEVICE)
        for kernel in (dot_chained_minus_rowsum, dot_uniform_k):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    "a lane loop around an MPP op would run 80 copies",
                ),
            ):
                _run_with_msl(kernel, (q, kt, v), config)

    def test_dot_k_beside_another_reduction_dim_rejects(self) -> None:
        c = torch.randn(64, 128, device=DEVICE, dtype=torch.float16)
        pt = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
        w = torch.randn(64, 48, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"in a lane loop beside other reduction dims"
        ):
            dot16_plus_rowsum.bind((c, pt, w)).to_code(
                helion.Config(block_sizes=[16, 16])
            )

    def test_dot_k_beside_another_reduction_dim_through_mpp(self) -> None:
        """The row sum's ``:`` dim holds threads beside the MPP op: the group
        reads only the sum, so those threads are replicas."""
        c = torch.randn(64, 128, device=DEVICE)
        pt = torch.randn(128, 64, device=DEVICE)
        w = torch.randn(64, 48, device=DEVICE)
        expected = _matmul_reference(c, pt) + w.cpu().double().sum(-1, keepdim=True)
        for config in (None, helion.Config(block_sizes=[16, 16])):
            with self.subTest(config=config):
                bound = dot_plus_rowsum.bind((c, pt, w))
                config = config or bound.config_spec.default_config()
                self.assertIn("_helion_mpp_run(", bound.to_code(config))
                torch.testing.assert_close(
                    bound.compile_config(config)(c, pt, w),
                    expected.float().to(DEVICE),
                    rtol=1e-4,
                    atol=1e-4,
                )

    @_full_threadgroup
    def test_runtime_dot_k_rejects(self) -> None:
        """Lane counts and tail masks are compile-time."""
        q = torch.randn(32, 32, device=DEVICE)
        kt = torch.randn(32, 64, device=DEVICE)
        v = torch.randn(64, 32, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"a ':' dimension of runtime extent \w+"
        ):
            dot_chained_dynamic.bind((q, kt, v)).to_code(
                helion.Config(block_sizes=[16, 16], num_threads=[4, 16])
            )

    def test_unfoldable_reduction_types_reject(self) -> None:
        """Rejected at codegen, including on one thread, where no
        cross-thread combine would check the type."""
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "argmax over a persistent dim walked in a lane loop",
        ):
            nested_argmax.bind((torch.randn(64, 2048, device=DEVICE),)).to_code(
                helion.Config(block_sizes=[32, 32], num_warps=4)
            )
        with self.assertRaisesRegex(exc.BackendUnsupported, "reduction 'any'"):
            nested_any.bind((torch.randn(64, 12, device=DEVICE),)).to_code(
                helion.Config(block_sizes=[32, 32], num_threads=[32, 32])
            )


# ---------------------------------------------------------------------------
# A reduction block that is only another block's power-of-two padding
# ---------------------------------------------------------------------------
# ``acc = x.new_zeros(n)`` in a device loop is padded to a power of two, and
# ``acc += xt.sum(0)`` merges that length into the ``x[tile, :]`` slice's dim.
# A row reduction over the dim then gets a second block of the padded length,
# a twin that reduces on the slice block's threads and lanes.


@helion.kernel(backend="metal")
def twin_colcarry_div(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :].to(torch.float32)
            acc += xt.sum(0)
            y = xt - (xt.sum(-1) / n)[:, None]
            out[tile_m, :] = (y * (y * y).sum(-1)[:, None]).to(x.dtype)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_rowmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :].to(torch.float32)
            acc += xt.sum(0)
            out[tile_m] = xt.amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_column_max(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The padded dim leads: x is ``[n, m]``."""
    m_block = hl.register_block_size(x.size(1))
    n = hl.specialize(x.size(0))
    out = x.new_empty([x.size(1)], dtype=torch.float32)
    cols = x.new_empty([(x.size(1) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(1), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[:, tile_m].to(torch.float32)
            acc += xt.sum(1)
            out[tile_m] = xt.amax(0)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_softmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :].to(torch.float32)
            acc += xt.sum(0)
            e = torch.exp(xt - xt.amax(-1)[:, None])
            out[tile_m, :] = (e / e.sum(-1)[:, None]).to(x.dtype)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_min_prod(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.amin(-1) + torch.prod(xt, -1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_argmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.int64)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = torch.argmax(xt, -1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_root_level(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """No inner loop: the padded ``hl.zeros`` meets the slice in the tile loop."""
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    shifted = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        xt = x[tile_m, :]
        shifted[tile_m, :] = xt + hl.zeros([n], dtype=torch.float32)[None, :]
        out[tile_m] = xt.amax(-1)
    return out, shifted


@helion.kernel(backend="metal")
def twin_factory(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``hl.full``'s 64 elements are real, not the twin owner's padding."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = hl.full([tile_m, 64], 1.0, dtype=torch.float32).sum(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_index_table(
    x: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A padded value used as an index would gather at padding positions."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = table[xt.to(torch.int64)].amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_gather(
    x: torch.Tensor, perm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An indexed load's elements past 40 come from the index, not padding."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = x[tile_m, perm].amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_strided(
    x: torch.Tensor, wide: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``wide[:, ::2]`` is 40 long but no full slice of a 40-long dim."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = wide[tile_m, ::2].amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_outer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An outer product carries the padded dim twice."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = (xt[:, :, None] * xt[:, None, :]).amax(-1).amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_if(x: torch.Tensor, flag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The branch takes the padded value as an input the cone can't trace."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_zeros([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            if flag[tile_c.begin] > 0:
                out[tile_m] = xt.amax(-1)
        cols[tile_c.id, :] = acc
    return out, cols


@helion.kernel(backend="metal")
def twin_memory(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y is as wide as x's dim padded."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    y_out = torch.empty_like(y)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.amax(-1)
            y_out[tile_m, :] = y[tile_m, :] * 2.0
        cols[tile_c.id, :] = acc
    return out, y_out, cols


@helion.kernel(backend="metal")
def twin_memory_reduced(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y is as wide as x's dim padded, and reduced over its own dim."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    y_out = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.amax(-1)
            y_out[tile_m] = y[tile_m, :].sum(-1)
        cols[tile_c.id, :] = acc
    return out, y_out, cols


@helion.kernel(backend="metal")
def twin_two_owners(
    z: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z's and x's dims pad to the same length."""
    m_block = hl.register_block_size(x.size(0))
    p = hl.specialize(z.size(1))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)], dtype=torch.float32)
    z_cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, p], dtype=torch.float32)
    x_cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        z_acc = x.new_zeros(p, dtype=torch.float32)
        x_acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            zt = z[tile_m, :]
            z_acc += zt.sum(0)
            out[tile_m] = zt.amax(-1)
            x_acc += (x[tile_m, :] * 2.0 + 1.0).sum(0)
        z_cols[tile_c.id, :] = z_acc
        x_cols[tile_c.id, :] = x_acc
    return out, z_cols, x_cols


@helion.kernel(backend="metal")
def twin_dot(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A dot over the owner's slice, beside a row max over its twin."""
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0), w.size(1)], dtype=torch.float32)
    row_max = x.new_empty([x.size(0)], dtype=torch.float32)
    cols = x.new_empty([(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32)
    for tile_c in hl.tile(x.size(0), block_size=m_block):
        acc = x.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            row_max[tile_m] = xt.amax(-1)
            for tile_n in hl.tile(w.size(1)):
                out[tile_m, tile_n] = hl.dot(xt, w[:, tile_n])
        cols[tile_c.id, :] = acc
    return out, row_max, cols


def _row_mean(t: torch.Tensor) -> torch.Tensor:
    return t.sum(-1) / t.size(-1)


def _first_row_mean(pair: tuple[torch.Tensor, int]) -> torch.Tensor:
    return pair[0].sum(-1) / pair[0].size(-1)


def log_softmax(t: torch.Tensor) -> torch.Tensor:
    """Shares the name of an extent-free torch function, but reads the extent."""
    return t.sum(-1) / t.size(-1)


# Each but ``twin_comprehension`` reads the padded length of the twin's dim
# and divides by it: 64 where the row has 40 elements.


@helion.kernel(backend="metal")
def twin_helper(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = _row_mean(xt)
    return out


@helion.kernel(backend="metal")
def twin_helper_keyword(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = _row_mean(t=xt)
    return out


@helion.kernel(backend="metal")
def twin_helper_starred(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = _row_mean(*[xt])
    return out


@helion.kernel(backend="metal")
def twin_helper_tuple(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = _first_row_mean((xt, 1))
    return out


@helion.kernel(backend="metal")
def twin_helper_torch_name(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = log_softmax(xt)
    return out


@helion.kernel(backend="metal")
def twin_size(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.sum(-1) / xt.size(-1)
    return out


@helion.kernel(backend="metal")
def twin_numel(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.sum(-1) / torch.numel(acc)
    return out


@helion.kernel(backend="metal")
def twin_numel_keyword(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.sum(-1) / torch.numel(input=acc)
    return out


@helion.kernel(backend="metal")
def twin_shape(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.sum(-1) / xt.shape[-1]
    return out


@helion.kernel(backend="metal")
def twin_nbytes(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = xt.sum(-1) * 4 / acc.nbytes
    return out


@helion.kernel(backend="metal")
def twin_comprehension(x: torch.Tensor) -> torch.Tensor:
    n = hl.specialize(x.size(1))
    out = x.new_empty([x.size(0)])
    for tile_c in hl.tile(x.size(0)):
        acc = x.new_zeros(n)
        for tile_m in hl.tile(tile_c.begin, tile_c.end):
            xt = x[tile_m, :]
            acc += xt.sum(0)
            out[tile_m] = sum([v.sum(-1) for v in (xt, xt)])
    return out


def _layer_norm_bwd_args(
    m: int, n: int
) -> tuple[tuple[object, ...], tuple[torch.Tensor, ...]]:
    """``layer_norm_bwd``'s MPS arguments and float64 ``(dx, dw, db)``."""
    x = torch.randn(m, n)
    w = torch.rand(n)
    dy = 0.1 * torch.randn(m, n)
    mean = x.double().mean(-1)
    rstd = torch.rsqrt(x.double().var(-1, unbiased=False) + 1e-5)
    xhat = (x.double() - mean[:, None]) * rstd[:, None]
    wdy = w.double() * dy.double()
    c1 = (xhat * wdy).mean(-1, keepdim=True)
    c2 = wdy.mean(-1, keepdim=True)
    dx = (wdy - (xhat * c1 + c2)) * rstd[:, None]
    args = (*(t.to(DEVICE) for t in (dy, x, mean.float(), rstd.float(), w)), True)
    return args, (dx, (dy.double() * xhat).sum(0), dy.double().sum(0))


def _cols_reference(x: torch.Tensor) -> torch.Tensor:
    return x.double().sum(0)


@_requires_darwin
class TestMetalPaddedTwin(unittest.TestCase):
    """A twin owns no threads, lanes or mask: its reductions run on its
    owner's layout, masked by the owner's extent."""

    _ONE_ROW = helion.Config(block_sizes=[1, 1])

    def _run(
        self, kernel: helion.Kernel, config: helion.Config, args: tuple[object, ...]
    ) -> tuple[object, str, str]:
        """Run *kernel* at *config*; return its result, code and MSL."""
        result, code, msl = _run_with_msl(kernel, args, config)
        if isinstance(result, tuple):
            result = tuple(t.cpu() for t in result if t is not None)
        return result, code, msl

    def _layer_norm_bwd(self) -> helion.Kernel:
        return helion.kernel(
            layer_norm_bwd.fn,
            backend="metal",
            ignore_warnings=[exc.TensorOperationInWrapper],
        )

    def _assert_close(self, actual: object, expected: tuple[torch.Tensor, ...]) -> None:
        assert isinstance(actual, tuple)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.double(), e, rtol=1e-4, atol=1e-4)

    def test_twin_reduces_on_owner_threads(self) -> None:
        """Every row sum combines over the owner's axis and span."""
        for n, span in ((24, 32), (40, 64)):
            with self.subTest(n=n):
                x = torch.randn(64, n, device=DEVICE)
                (out, cols), _, msl = self._run(twin_colcarry_div, self._ONE_ROW, (x,))
                xd = x.cpu().double()
                y = xd - xd.mean(-1, keepdim=True)
                self._assert_close(
                    (out, cols.sum(0)),
                    (y * (y * y).sum(-1, keepdim=True), _cols_reference(xd)),
                )
                self.assertEqual(
                    {(op, axis, s) for op, _, _, axis, s in _group_reductions(msl)},
                    {("sum", 0, span)},
                )
        torch.manual_seed(0)
        args, expected = _layer_norm_bwd_args(64, 24)
        kernel = self._layer_norm_bwd()
        default = kernel.bind(args).config_spec.default_config()
        result, _, _ = self._run(kernel, default, args)
        self._assert_close(result, expected)

    @_full_threadgroup
    def test_twin_folds_owner_lanes(self) -> None:
        """2560 elements on 1024 threads: the twin's reductions fold the
        owner's 3 lanes."""
        torch.manual_seed(0)
        x = torch.randn(16, 2560, device=DEVICE)
        (out, cols), code, _ = self._run(twin_colcarry_div, self._ONE_ROW, (x,))
        self.assertIn("_metal_lane_fold(", code)
        self.assertRegex(code, r"for slice_lane_\d+ in range\(3\)")
        xd = x.cpu().double()
        y = xd - xd.mean(-1, keepdim=True)
        self._assert_close(
            (out, cols.sum(0)),
            (y * (y * y).sum(-1, keepdim=True), _cols_reference(xd)),
        )
        args, expected = _layer_norm_bwd_args(16, 2560)
        kernel = self._layer_norm_bwd()
        default = kernel.bind(args).config_spec.default_config()
        for config in (self._ONE_ROW, default):
            with self.subTest(config=config):
                result, code, _ = self._run(kernel, config, args)
                self.assertIn("_metal_lane_fold(", code)
                self._assert_close(result, expected)

    def test_owner_mask_drops_padding(self) -> None:
        """All-negative rows: a padding element would win the max."""
        for n in (24, 40, 2560):
            with self.subTest(n=n):
                x = -torch.randn(64, n, device=DEVICE).abs() - 1.0
                result, _, _ = self._run(twin_rowmax, self._ONE_ROW, (x,))
                xd = x.cpu().double()
                self._assert_close(
                    (result[0], result[1].sum(0)), (xd.amax(-1), _cols_reference(xd))
                )
        for n in (40, 2560):
            with self.subTest(n=n, leading=True):
                x = -torch.randn(n, 64, device=DEVICE).abs() - 1.0
                result, _, _ = self._run(twin_column_max, self._ONE_ROW, (x,))
                xd = x.cpu().double()
                self._assert_close(
                    (result[0], result[1].sum(0)), (xd.amax(0), xd.sum(1))
                )

    def test_dependent_reductions(self) -> None:
        for n in (40, 2560):
            for dtype in (torch.float32, torch.float16):
                with self.subTest(n=n, dtype=dtype):
                    x = torch.randn(64, n, device=DEVICE, dtype=dtype)
                    (out, cols), _, _ = self._run(twin_softmax, self._ONE_ROW, (x,))
                    xd = x.cpu().double()
                    tol = 1e-4 if dtype == torch.float32 else 2e-3
                    torch.testing.assert_close(
                        out, torch.softmax(xd, -1).to(dtype), rtol=tol, atol=tol
                    )
                    torch.testing.assert_close(
                        cols.sum(0).double(), _cols_reference(xd), rtol=tol, atol=tol
                    )

    def test_min_prod_and_argmax(self) -> None:
        x = 1.0 + 0.01 * torch.randn(64, 40, device=DEVICE)
        result, _, _ = self._run(twin_min_prod, self._ONE_ROW, (x,))
        xd = x.cpu().double()
        self._assert_close(
            (result[0], result[1].sum(0)),
            (xd.amin(-1) + xd.prod(-1), _cols_reference(xd)),
        )
        x = torch.randn(64, 40, device=DEVICE)
        (indices, _), _, _ = self._run(twin_argmax, self._ONE_ROW, (x,))
        torch.testing.assert_close(indices, x.cpu().argmax(-1))

    def test_argmax_over_owner_lanes_rejects(self) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "argmax over a persistent dim walked in a lane loop"
        ):
            twin_argmax.bind((torch.randn(64, 2560, device=DEVICE),)).to_code(
                self._ONE_ROW
            )

    def test_root_level_twin(self) -> None:
        x = torch.randn(64, 40, device=DEVICE)
        bound = twin_root_level.bind((x,))
        result, _, _ = self._run(
            twin_root_level, bound.config_spec.default_config(), (x,)
        )
        xd = x.cpu().double()
        self._assert_close(result, (xd.amax(-1), xd))

    def test_twin_in_reduction_loop_rejects(self) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction dim \d+, the power-of-two padding of dim \d+, in a "
            r"reduction loop",
        ):
            twin_root_level.bind((torch.randn(64, 40, device=DEVICE),)).to_code(
                helion.Config(block_sizes=[4], reduction_loops=[16])
            )

    def test_counting_reduction_rejects(self) -> None:
        """``mean`` divides by the padded length."""
        x = torch.randn(64, 40, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"line \d+ calls \.mean\(\) on a value whose dim \d+ is dim \d+ "
            r"\(40 elements\) padded to 64",
        ):
            colsum_threaded.bind((x,)).to_code(helion.Config(block_sizes=[32]))

    def test_factory_under_twin_reduction_rejects(self) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"sum over dim \d+, the power-of-two padding of dim \d+: \w+ creates "
            r"the padded dim",
        ):
            twin_factory.bind((torch.randn(64, 40, device=DEVICE),)).to_code(
                self._ONE_ROW
            )

    def test_padded_length_in_memory_rejects(self) -> None:
        x = torch.randn(64, 40, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction dim \d+ pads dim \d+ to 64 elements, but \w+ dim 1 has 64 "
            r"real elements",
        ):
            twin_memory.bind((x, y)).to_code(self._ONE_ROW)

    def test_two_owners_reject(self) -> None:
        z = torch.randn(64, 48, device=DEVICE)
        x = torch.randn(64, 40, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction dim \d+ of 64 elements pads dims \[\d+, \d+\]; one block "
            r"cannot carry both extents' masks",
        ):
            twin_two_owners.bind((z, x)).to_code(self._ONE_ROW)

    def test_fold_beside_another_reduction_dim_rejects(self) -> None:
        """y's 64-long dim is a reduction dim of its own, not a twin."""
        x = torch.randn(64, 2560, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction over persistent dim 2 in a lane loop beside other "
            r"reduction dims \[3\]",
        ):
            twin_memory_reduced.bind((x, y)).to_code(self._ONE_ROW)

    def test_dot_over_padded_dim_rejects(self) -> None:
        x = torch.randn(64, 40, device=DEVICE)
        w = torch.randn(40, 8, device=DEVICE)
        bound = twin_dot.bind((x, w))
        for config in (
            helion.Config(block_sizes=[1, 1, 8]),
            bound.config_spec.default_config(),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    r"line \d+ passes a value whose dim \d+ is dim \d+ \(40 "
                    r"elements\) padded to 64 to hl\.dot\(\)",
                ),
            ):
                bound.to_code(config)

    def test_extent_reads_reject(self) -> None:
        passes = r"line \d+ passes a value whose dim \d+ is dim \d+ \(\d+ elements\) padded to \d+ to "
        uses = r" on a value whose dim \d+ is dim \d+ \(\d+ elements\) padded to \d+"
        for kernel, message, sizes in (
            (twin_helper, passes + r"_row_mean\(\)", (40, 2560)),
            (twin_helper_keyword, passes + r"_row_mean\(\)", (40,)),
            (twin_helper_starred, passes + r"_row_mean\(\)", (40,)),
            (twin_helper_tuple, passes + r"_first_row_mean\(\)", (40,)),
            (twin_helper_torch_name, passes + r"log_softmax\(\)", (40,)),
            (twin_numel, passes + r"torch\.numel\(\)", (40,)),
            (twin_numel_keyword, passes + r"torch\.numel\(\)", (40,)),
            (twin_size, r"calls \.size\(\)" + uses, (40, 2560)),
            (twin_shape, r"reads \.shape" + uses, (40,)),
            (twin_nbytes, r"reads \.nbytes" + uses, (40,)),
            (twin_comprehension, r"uses it in a comprehension" + uses, (40,)),
        ):
            for n in sizes:
                with (
                    self.subTest(kernel=kernel.fn.__name__, n=n),
                    self.assertRaisesRegex(exc.BackendUnsupported, message),
                ):
                    kernel.bind((torch.randn(64, n, device=DEVICE),)).to_code(
                        self._ONE_ROW
                    )

    def test_unsound_twin_reductions_reject(self) -> None:
        """A twin reduction may only read the owner's real elements: anything
        that could put a real value past 40 rejects."""
        x = torch.randn(64, 40, device=DEVICE)
        cone = r"max over dim \d+, the power-of-two padding of dim \d+: \w+ "
        for kernel, args, message in (
            (
                twin_index_table,
                (x, torch.randn(100, device=DEVICE)),
                (
                    r"line \d+ indexes with it on a value whose dim \d+ is dim "
                    r"\d+ \(40 elements\) padded to 64"
                ),
            ),
            (
                twin_gather,
                (x, torch.randperm(40, device=DEVICE)),
                cone + "is an indexed load",
            ),
            (
                twin_strided,
                (x, torch.randn(64, 80, device=DEVICE)),
                cone + "loads no full slice of 40 elements",
            ),
            (twin_outer, (x,), cone + "has the padded dim twice"),
            (
                twin_if,
                (x, torch.ones(64, device=DEVICE)),
                cone + "enters from outside its loop or branch",
            ),
        ):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, message),
            ):
                kernel.bind(args).to_code(self._ONE_ROW)


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------

#: Autotuning compiles and benchmarks each candidate inline, so keep the tests
#: to a small budget and small tensors.
_AUTOTUNE_KWARGS = {"autotune_effort": "quick", "autotune_budget_seconds": 5}


@_requires_darwin
class TestMetalAutotune(unittest.TestCase):
    """Autotuning is enabled for Metal; these pin the plumbing it needs."""

    def test_benchmark_hooks_are_batched(self) -> None:
        from helion._compiler.metal.autotune import do_bench_metal
        from helion._compiler.metal.autotune import interleaved_bench_metal

        backend = MetalBackend()
        self.assertIs(backend.get_do_bench(), do_bench_metal)
        self.assertIs(backend.get_interleaved_bench(), interleaved_bench_metal)
        self.assertFalse(backend.supports_precompile())

    def test_interleaved_bench_batches_each_candidate_separately(self) -> None:
        """A cheap candidate must not be measured at an expensive one's batch.

        ``_time_batch_ms`` already returns a per-call time, so a shared batch
        equalizes nothing -- it just drags every candidate down to the slowest
        one's batch size, which is where per-sample overhead is least
        amortized.  One expensive candidate in the list used to collapse the
        batch to 1 for all of them, i.e. batching switched off.
        """
        from helion._compiler.metal.autotune import do_bench_metal
        from helion._compiler.metal.autotune import interleaved_bench_metal

        small = torch.randn(512, 512, device=DEVICE)
        big = torch.randn(2048, 2048, device=DEVICE)
        cheap = lambda: small + 1.0  # noqa: E731
        pricey = lambda: torch.mm(big, big)  # noqa: E731

        truth = float(do_bench_metal(pricey)) / float(do_bench_metal(cheap))
        t_cheap, t_pricey = interleaved_bench_metal([cheap, pricey], repeat=5)
        measured = t_pricey / t_cheap

        self.assertGreater(measured, 1.0)
        # Generous, because the point is the order of magnitude: a shared batch
        # reported ~1.2x where the true gap was ~7x.
        self.assertGreater(measured, truth / 3.0)

    def test_identical_candidates_measure_the_same(self) -> None:
        """The self-test: N copies of one kernel must rank as a tie.

        This needs no reference timing -- the true ratio is exactly 1 by
        construction -- which makes it the one calibration check here that
        cannot be undermined by a drifting baseline.  It is also a direct
        probe of the bug this function had: sharing one batch across
        candidates handed them different amounts of per-sample overhead, and
        any per-candidate asymmetry (batch, position in the round) shows up
        here as spread.

        The operand size is load-bearing.  At 1024x1024 a call costs ~0.03ms, so
        the batch saturates at ``_MAX_BATCH`` and ``repeat`` stops controlling
        anything: every candidate is summarized from the ``_MIN_ROUNDS`` floor of
        three samples, and the assertion becomes a coin flip on a busy machine.
        At 4096x4096 a call is ~0.9ms, the batch lands around 22, and ``repeat``
        buys ~18 rounds -- enough for the ``min`` to mean something.

        Best of three attempts even so, because this is wall-clock on a shared
        GPU.  The defect it exists to catch is not subtle: sharing one batch
        across candidates mismeasured a 7.4x difference as 1.2x and collapsed one
        candidate's batch to a single call, which fails all three attempts.
        """
        from helion._compiler.metal.autotune import interleaved_bench_metal

        x = torch.randn(4096, 4096, device=DEVICE)
        y = torch.randn(4096, 4096, device=DEVICE)
        same = lambda: x + y  # noqa: E731

        for count in (2, 4):
            with self.subTest(candidates=count):
                spreads = []
                for _ in range(3):
                    times = interleaved_bench_metal([same] * count, repeat=400)
                    self.assertEqual(len(times), count)
                    spreads.append(max(times) / min(times))
                self.assertLess(min(spreads), 1.25, f"spreads={spreads}")

    def test_interleaved_repeat_is_a_call_budget_not_a_round_count(self) -> None:
        """``repeat`` counts calls in the shared benchmarks, not rounds.

        Callers derive it as ``target_ms / per_call_ms`` and the shared
        implementations invoke each candidate once per round.  A round here is
        a whole batch, so spending ``repeat`` rounds overshoots the requested
        budget by the batch size -- which is what made final-pick verification
        take longer than the search it was verifying.
        """
        from helion._compiler.metal import autotune as metal_autotune

        batches: list[int] = []

        def spy(fn: object, batch: int) -> float:
            batches.append(batch)
            return 1.0

        def run(repeat: int) -> int:
            """Timed rounds for ``repeat``, with warmup and probing stubbed out."""
            batches.clear()
            with (
                patch.object(metal_autotune, "_warm_up", lambda fn: None),
                patch.object(metal_autotune, "_estimate_per_call_ms", lambda fn: 1.0),
                patch.object(metal_autotune, "_time_batch_ms", spy),
            ):
                metal_autotune.interleaved_bench_metal([lambda: None], repeat=repeat)
            return len(batches)

        # 1ms per call against a 20ms sample target -> batch of 20.
        batch = int(metal_autotune._SAMPLE_TARGET_MS)
        self.assertEqual(run(1000), 1000 // batch)
        self.assertEqual(set(batches), {batch})
        # The floor applies when the budget is smaller than a single batch.
        self.assertEqual(run(1), metal_autotune._MIN_ROUNDS)

    def test_interleaved_honors_max_total_ms(self) -> None:
        """The rebenchmark path always passes ``max_total_ms``.

        ``BaseSearch.rebenchmark`` binds it via ``functools.partial``
        before calling the backend hook, so a hook that does not accept
        it fails every final verification with a TypeError.
        """
        import functools

        from helion._compiler.metal import autotune as metal_autotune

        batches: list[int] = []

        def spy(fn: object, batch: int) -> float:
            batches.append(batch)
            return 1.0

        def run(bench_fn: object, *args: object, **kwargs: object) -> object:
            batches.clear()
            with (
                patch.object(metal_autotune, "_warm_up", lambda fn: None),
                patch.object(metal_autotune, "_estimate_per_call_ms", lambda fn: 1.0),
                patch.object(metal_autotune, "_time_batch_ms", spy),
            ):
                return bench_fn(*args, **kwargs)  # type: ignore[operator]

        # 1ms per call -> batch of 20 -> 20ms per round; a 100ms budget
        # buys 5 rounds out of the 50 that ``repeat`` alone would allow,
        # invoked the way the caller invokes it.
        bench = functools.partial(
            metal_autotune.interleaved_bench_metal, max_total_ms=100.0
        )
        self.assertEqual(run(bench, [lambda: None], repeat=1000), [1.0])
        self.assertEqual(len(batches), 5)
        # No budget: the repeat-derived count stands.
        run(metal_autotune.interleaved_bench_metal, [lambda: None], repeat=1000)
        self.assertEqual(len(batches), 50)

    def test_warmup_budget_is_spent_in_whole_batches(self) -> None:
        """``warmup`` is a millisecond budget, like ``rep``.

        It used to run ``warmup / sample_ms`` *single* calls, so it actually
        covered ``warmup / batch`` ms -- a 25ms request bought under 3ms of
        warmup, and the 1000ms the rebenchmark path asks for bought ~10ms.
        """
        from helion._compiler.metal import autotune as metal_autotune

        calls = 0

        def counted() -> None:
            nonlocal calls
            calls += 1

        seen: list[int] = []
        real_time_batch_ms = metal_autotune._time_batch_ms

        def spy(fn: object, batch: int) -> float:
            seen.append(batch)
            return real_time_batch_ms(fn, batch)

        with patch.object(metal_autotune, "_time_batch_ms", spy):
            metal_autotune.do_bench_metal(counted, warmup=25, rep=100)

        # Every timed loop -- estimate, warmup and rep -- goes through
        # _time_batch_ms, so no call is left un-batched.
        self.assertTrue(seen)
        self.assertEqual(calls, sum(seen) + 1)  # +1 for the initial correctness call
        self.assertGreater(max(seen), 1)

    def test_rejects_an_unknown_return_mode(self) -> None:
        """Summarizing is delegated to the shared fallback, so only the
        argument check is Metal's to make."""
        from helion._compiler.metal.autotune import do_bench_metal

        with self.assertRaises(AssertionError):
            do_bench_metal(lambda: None, return_mode="bogus")

    def test_unsupported_config_is_a_search_miss_not_a_crash(self) -> None:
        backend = MetalBackend()
        self.assertEqual(
            backend.classify_autotune_exception(exc.BackendUnsupported("metal", "x")),
            "debug",
        )
        self.assertEqual(
            backend.classify_autotune_exception(SyntaxError("bad MSL")), "warn"
        )
        # Anything else is still one bad candidate, not a reason to stop.  With
        # no catch-all the shared fallback is classify_triton_exception, which
        # answers "raise" for messages it does not recognize -- and it
        # recognizes no MPS message, so an unfamiliar driver error would kill
        # the search.
        self.assertEqual(
            backend.classify_autotune_exception(RuntimeError("some MPS error")), "warn"
        )
        # ...but an interrupt is not a search miss.
        self.assertIsNone(backend.classify_autotune_exception(KeyboardInterrupt()))

    def test_num_threads_is_clamped_to_a_divisor_of_the_block_size(self) -> None:
        """The shared loop strategy needs block_size % num_threads == 0.

        num_threads is drawn from the tensor extent, not the chosen block size,
        so the search space contains combinations that cannot compile. They are
        clamped rather than skipped, which keeps the whole space usable.
        """

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[64], num_threads=[1024])],
        )
        def scaled_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
            return out

        x = torch.randn(4096, device=DEVICE)
        torch.testing.assert_close(scaled_copy(x), x * 2.0)
        # Pin the value, not just "it compiled": the clamp picks the largest
        # divisor of the block size that is <= the requested count, and since
        # both are powers of two an over-large request always lands on exactly
        # the block size.  Asserting correctness alone would pass for any legal
        # divisor, including 1.
        self.assertIn("_block_dims=(64, 1, 1)", scaled_copy.bind((x,)).to_code())

    def test_autotuned_elementwise_1d_is_correct(self) -> None:
        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(8192, device=DEVICE)
        y = torch.randn(8192, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_autotuned_elementwise_2d_is_correct(self) -> None:
        """2D tiling is where autotuning actually pays: the default
        ``block_sizes=[32, 32]`` is rarely the best tile shape."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                out[tm, tn] = x[tm, tn] + y[tm, tn]
            return out

        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        torch.testing.assert_close(add2d(x, y), x + y)

    def test_autotuned_reduction_is_correct(self) -> None:
        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        x = torch.randn(256, 512, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_autotuned_softmax_is_correct(self) -> None:
        """Two reductions over one dimension, each with its own scratch."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp_v = torch.exp(values - amax)
                out[tile_n, :] = exp_v / torch.sum(exp_v, dim=1, keepdim=True)
            return out

        x = torch.randn(128, 512, device=DEVICE)
        torch.testing.assert_close(
            softmax(x), torch.softmax(x, dim=-1), rtol=1e-3, atol=1e-3
        )

    def test_reduction_loops_is_searchable(self) -> None:
        """The reduction span is the knob worth tuning, so it must be in the space."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        spec = row_sum.bind((torch.randn(256, 1024, device=DEVICE),)).config_spec
        self.assertTrue(spec.reduction_loops.valid_block_ids())
        self.assertIn("reduction_loops", spec.default_config().config)

    def test_explicit_tile_threads_do_not_starve_a_reduction(self) -> None:
        """Autotuning sets num_threads explicitly, unlike the default config.

        A persistent reduction needs one live thread per element, and Metal has
        no lane-loop fallback to cover a shortfall the way CuTe does, so a
        tile that claims the whole threadgroup used to make the config
        unbuildable.  The tile's thread count is capped instead.
        """

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(
                    block_sizes=[64], num_threads=[1024], reduction_loops=[None]
                )
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        x = torch.randn(256, 1024, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_autotuned_matmul_is_correct(self) -> None:
        """Matmul is where tuning matters most: the default 16x16x16 tile
        badly underuses MPP's cooperative matmul."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(matmul(x, y), x @ y, rtol=1e-3, atol=1e-3)

    def test_oversized_threadgroup_is_a_search_miss(self) -> None:
        """An over-budget threadgroup is routine, not noteworthy.

        Metal only rejects one when it builds the pipeline state, so it arrives
        as a RuntimeError from the launcher.  It is the same category as
        BackendUnsupported -- the config does not fit -- and it is common, so it
        is classified alongside the other expected misses rather than warned
        about.
        """
        from helion._compiler.metal.backend import _is_threadgroup_too_large

        backend = MetalBackend()
        err = RuntimeError(
            "Failed to created pipeline state object, error: Specified total "
            "max threads per threadgroup (8192) exceeds the maximum total "
            "threads per threadgroup supported (1024)"
        )
        self.assertTrue(_is_threadgroup_too_large(err))
        self.assertFalse(_is_threadgroup_too_large(RuntimeError("unrelated")))

        # The recognizer has to change the outcome, or it is dead code behind
        # the catch-all: recognized -> "debug", unrecognized -> "warn".
        self.assertEqual(backend.classify_autotune_exception(err), "debug")
        self.assertEqual(
            backend.classify_autotune_exception(RuntimeError("unrelated")), "warn"
        )

    def _mpp_matmul(self, bm: int, bn: int, bk: int, warps: int) -> helion.Kernel:
        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[bm, bn, bk], num_warps=warps)],
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        return matmul

    def test_oversized_mpp_tiles_are_rejected(self) -> None:
        """``matmul2d`` is silently wrong for large M/N tile extents.

        Not a rounding problem: on a 512x512 fp32 product with values of order
        1e2, the errors reach 1.8e29 and NaN, with no compile or runtime
        diagnostic.  Reproduces in plain MSL with no Helion involved.

        The predicate is the *extent*, not the aspect ratio: 128x16 (8:1) is
        correct everywhere while 256x256 (1:1) is wrong at N=1 and N=2.  Nor
        is it accumulator capacity -- 256x16 at 128 elements per thread fails
        while 128x128 at 512 does not.  Up to 256, four SIMD groups are
        enough; past 256 there is often no N that works at all: 512x512 is
        wrong at every N up to the 32 simdgroups a threadgroup can hold.
        """
        x = torch.randn(512, 512, device=DEVICE)
        y = torch.randn(512, 512, device=DEVICE)

        for bm, bn, warps, message in [
            (256, 32, 1, "an MPP matmul tile 256x32 on 1 SIMD groups"),
            (32, 256, 1, "an MPP matmul tile 32x256 on 1 SIMD groups"),
            (256, 256, 1, "an MPP matmul tile 256x256 on 1 SIMD groups"),
            (512, 512, 8, "an MPP matmul tile 512x512x64: one op covers"),
        ]:
            with (
                self.subTest(tile=(bm, bn), num_warps=warps),
                self.assertRaisesRegex(exc.BackendUnsupported, message),
            ):
                self._mpp_matmul(bm, bn, 64, warps)(x, y)

        # Both extents <= 128 is the region that measured clean: 576 of 576
        # across TILE_K in {16..512} and N in {1..32}.  One op takes K <= 256.
        for bm, bn, bk, warps in [
            (128, 128, 64, 1),
            (64, 128, 256, 8),
            (128, 64, 32, 2),
            (256, 32, 64, 4),
            (32, 256, 32, 8),
        ]:
            with self.subTest(tile=(bm, bn, bk), num_warps=warps):
                out = self._mpp_matmul(bm, bn, bk, warps)(x, y)
                torch.testing.assert_close(out, x @ y, rtol=1e-3, atol=1e-3)

    def test_strided_matmul_operands_are_correct(self) -> None:
        """Transposed and row-padded operands lower through MPP correctly.

        Every MPP operand is built as a ``tensor_inline`` from a pointer and
        two extents, so the backend derives the handle geometry from the
        operand's strides instead of assuming packed row-major: a transposed
        operand swaps the handle extents and sets the descriptor transpose
        flag, while a row-padded operand widens the handle's leading extent.
        The launcher binds an offset view at its first logical element.
        ``mm(x, w.t())`` used to return exactly ``x @ w``, off by 24 on a
        32x32 fp32 product, with no error.
        """
        x = torch.randn(32, 32, device=DEVICE)
        w = torch.randn(32, 32, device=DEVICE)
        big = torch.randn(64, 64, device=DEVICE)

        cases = [
            ("transposed rhs", x, w.t()),
            ("transposed lhs", x.t(), w),
            ("transposed lhs and rhs", x.t(), w.t()),
            ("padded lhs", big[:32, :32], w),
            ("padded rhs", x, big[:32, :32]),
            # Packed strides with a nonzero storage offset.
            ("offset view lhs", big[4:36, 8:40], w),
        ]
        for tag, a, b in cases:
            with self.subTest(operand=tag):
                torch.testing.assert_close(
                    self._mpp_matmul(16, 16, 16, 4)(a, b),
                    a @ b,
                    rtol=1e-4,
                    atol=1e-4,
                )

        # Contiguous stays exact, including a contiguous copy of the transpose.
        for a, b in [(x, w), (x, w.t().contiguous())]:
            torch.testing.assert_close(
                self._mpp_matmul(16, 16, 16, 4)(a, b), a @ b, rtol=1e-4, atol=1e-4
            )

    def test_strided_matmul_nonsquare_transpose_is_correct(self) -> None:
        """Transposed operands with non-square shapes and tail tiles."""
        x = torch.randn(32, 64, device=DEVICE)
        w = torch.randn(16, 64, device=DEVICE)
        with self.subTest(operand="non-square transposed rhs"):
            torch.testing.assert_close(
                self._mpp_matmul(16, 16, 16, 4)(x, w.t()),
                x @ w.t(),
                rtol=1e-4,
                atol=1e-4,
            )

        xt = torch.randn(56, 40, device=DEVICE)
        wt = torch.randn(24, 40, device=DEVICE)
        with self.subTest(operand="transposed rhs with M/N/K tails"):
            torch.testing.assert_close(
                self._mpp_matmul(16, 16, 16, 4)(xt, wt.t()),
                xt @ wt.t(),
                rtol=1e-4,
                atol=1e-4,
            )

        wa = torch.randn(40, 56, device=DEVICE)
        wb = torch.randn(24, 40, device=DEVICE)
        with self.subTest(operand="transposed lhs and rhs with M/N/K tails"):
            torch.testing.assert_close(
                self._mpp_matmul(16, 16, 16, 4)(wa.t(), wb.t()),
                wa.t() @ wb.t(),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_row_padded_mpp_tails_are_correct(self) -> None:
        """MPP masks against storage extents, not row-padded logical extents,
        so a row-padded operand with a tail is staged rather than read past
        its logical extent."""
        big = torch.randn(64, 64, device=DEVICE)
        cases = [
            ("lhs K tail", big[:32, :40], torch.randn(40, 32, device=DEVICE)),
            ("rhs N tail", torch.randn(32, 40, device=DEVICE), big[:40, :24]),
        ]
        for tag, a, b in cases:
            with self.subTest(operand=tag):
                out = self._mpp_matmul(16, 16, 16, 4)(a, b)
                torch.testing.assert_close(out, a @ b, rtol=1e-4, atol=1e-4)

    def test_mpp_rejects_an_extra_grid_axis(self) -> None:
        """An unrelated same-sized grid axis that holds threads around an MPP
        matmul is rejected, rather than standing in for M."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 16, 8, 16], num_warps=4)],
        )
        def extra_grid_axis(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = a.size()
            _k, n = b.size()
            side = torch.empty([m], dtype=a.dtype, device=a.device)
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for extra, tm, tn in hl.tile([m, m, n]):
                side[extra] = a[extra, 0]
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
                out[tm, tn] = acc
            return out, side

        a = torch.randn(32, 32, device=DEVICE)
        b = torch.randn(32, 24, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "only its M, N and K blocks may hold threads"
        ):
            extra_grid_axis(a, b)

    def test_matmul_in_a_later_root_grid_is_correct(self) -> None:
        """A later grid uses its rebased tile offsets, not the raw launch ID."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def two_grids(
            a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = a.size()
            _k, n = b.size()
            side = torch.empty_like(z)
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tz in hl.tile(z.size(0)):
                side[tz] = z[tz] + 1.0
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
                out[tm, tn] = acc
            return out, side

        a = torch.randn(128, 128, device=DEVICE)
        b = torch.randn(128, 128, device=DEVICE)
        z = torch.randn(64, device=DEVICE)
        out, side = two_grids(a, b, z)
        torch.testing.assert_close(out, a @ b, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(side, z + 1.0)

        self.assertIn("_helion_mpp_acc(", two_grids.bind((a, b, z)).to_code())

    def test_mpp_tile_guard_uses_resolved_axis(self) -> None:
        """The cap validates M/N even with an unrelated leading grid axis."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def three_axis(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.size()
            _k, n = b.size()
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for _tg, tm, tn in hl.tile([1, m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
                out[tm, tn] = acc
            return out

        a = torch.randn(512, 512, device=DEVICE)
        b = torch.randn(512, 512, device=DEVICE)
        cfg = [1, 128, 256, 32]
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "an MPP matmul tile 128x256 on 2 SIMD groups"
        ):
            helion.kernel(
                three_axis.fn,
                backend="metal",
                configs=[helion.Config(block_sizes=cfg, num_warps=2)],
            )(a, b)

        ok = helion.kernel(
            three_axis.fn,
            backend="metal",
            configs=[helion.Config(block_sizes=[1, 128, 128, 32], num_warps=2)],
        )
        torch.testing.assert_close(ok(a, b), a @ b, rtol=1e-3, atol=1e-3)

    def test_accuracy_check_is_a_second_line_not_the_only_one(self) -> None:
        """Autotuning also validates candidates, but cannot be relied on alone.

        It does reject these configs in practice, because they are wrong by
        many orders of magnitude rather than subtly.  It is not sufficient on
        its own for three reasons, which is why the codegen guard above
        exists: it does nothing for a hand-written config, it can be disabled
        with ``HELION_AUTOTUNE_ACCURACY_CHECK=0``, and it compares against the
        default config run on the caller's own arguments -- so autotuning on
        degenerate input (a zero-filled warm-up batch, say) makes the
        comparison vacuous and it accepts them.
        """
        from helion.runtime.settings import Settings

        self.assertTrue(Settings().autotune_accuracy_check)

        # The guard holds even where the accuracy check would not: zeros make
        # every candidate agree with the baseline.
        zeros = torch.zeros(512, 512, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "an MPP matmul tile 32x256 on 1 SIMD groups"
        ):
            self._mpp_matmul(32, 256, 64, 1)(zeros, zeros)

    def test_effort_none_uses_the_default_config(self) -> None:
        @helion.kernel(backend="metal", autotune_effort="none")
        def scaled(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 3.0
            return out

        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(scaled(x), x * 3.0)
        bound = scaled.bind((x,))
        self.assertEqual(bound._config, bound.config_spec.default_config())

    def test_explicit_threads_with_a_rolled_reduction_are_correct(self) -> None:
        """Regression: this config silently returned a wrong sum.

        Clamping ``num_threads`` to a divisor of the block size let configs
        compile that the loop strategy had previously rejected.  For a *rolled*
        reduction that was not a win: the tile then claimed enough of the
        threadgroup that ``adjust_reduction_thread_count`` shrank the reduction
        span below the chunk size, so each iteration reduced only the first
        ``span`` of its 64 elements.  No error -- just a plausible wrong
        answer.  Only the autotuner produced these, because they need an
        explicit tile ``num_threads`` and an explicit ``reduction_loops``
        together, which no hand-written test config combined.
        """

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(block_sizes=[32], num_threads=[256], reduction_loops=[64])
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tm in hl.tile(m):
                out[tm] = x[tm, :].sum(-1)
            return out

        x = torch.randn(64, 256, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3)

    def test_autotune_budget_bounded_restored_and_bypassed(self) -> None:
        """The search budget defaults, passes through, and is restored.

        ``autotune()`` mutates shared settings, so it must leave them as
        found; the 300 s default must apply unless the caller sets a budget
        or asks for full effort.  The spy stands in for the real search, so
        this tests the settings handling without running one.
        """
        from typing import TYPE_CHECKING
        from typing import Any

        from helion._compiler import backend as compiler_backend
        from helion._compiler.metal.backend import _DEFAULT_AUTOTUNE_BUDGET_SECONDS

        if TYPE_CHECKING:
            from helion.runtime.config import Config
            from helion.runtime.kernel import BoundKernel

        seen: list[object] = []

        def spy(
            backend: MetalBackend,
            bound_kernel: BoundKernel[Any],
            *args: object,
            **kwargs: object,
        ) -> Config:
            seen.append(bound_kernel.settings.autotune_budget_seconds)
            return bound_kernel.config_spec.default_config()

        x = torch.randn(1024, device=DEVICE)
        for effort, budget, expected in [
            ("quick", None, _DEFAULT_AUTOTUNE_BUDGET_SECONDS),
            ("full", None, None),
            ("quick", 7, 7),
        ]:
            with self.subTest(effort=effort, budget=budget):

                @helion.kernel(
                    backend="metal",
                    autotune_effort=effort,
                    autotune_budget_seconds=budget,
                )
                def scaled(x: torch.Tensor) -> torch.Tensor:
                    out = torch.empty_like(x)
                    for tile in hl.tile(x.size(0)):
                        out[tile] = x[tile] * 3.0
                    return out

                bound = scaled.bind((x,))
                seen.clear()
                with patch.object(compiler_backend.Backend, "autotune", spy):
                    MetalBackend().autotune(bound, (x,))
                self.assertEqual(seen, [expected])
                self.assertEqual(bound.settings.autotune_budget_seconds, budget)

    def test_oversized_explicit_threadgroup_is_capped(self) -> None:
        """Explicit counts past the threadgroup budget are capped to it:
        64 * 32 explicit threads launch as 32 * num_warps."""

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(
                    block_sizes=[64, 64, 32], num_threads=[32, 64], num_warps=4
                )
            ],
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        out, code, _ = _run_with_msl(matmul, (x, y))
        self.assertEqual(math.prod(_launch(code)), 128)
        torch.testing.assert_close(out, x @ y, rtol=1e-4, atol=1e-3)


class TestMetalCodegen(unittest.TestCase):
    """Renders that need no Metal device."""

    @skipIfRefEager("renders a pinned config; ref mode runs the kernel eagerly")
    @unittest.skipIf(
        row_sum.settings.debug_dtype_asserts,
        "Metal refuses debug_dtype_asserts (it emits Triton checks)",
    )
    def test_rolled_reduction_has_no_cuda_barrier(self) -> None:
        # The lane-loop wrapper Metal's rolled reductions share with the CuTe
        # backend runs CuTe's cross-thread barrier pass only for CuTe: a
        # ``cute.arch.sync_threads()`` is an undeclared identifier in MSL.
        code = row_sum.bind((torch.randn(64, 4096),)).to_code()
        self.assertIn("helion_red.group_sum(", code)
        self.assertNotIn("sync_threads", code)


class TestMetalMslExpr(unittest.TestCase):
    """Inductor's MetalOverrides write MSL; ``msl_expr`` parses it into the
    device code's Python AST, which ``msl_ast_walker`` prints back."""

    @staticmethod
    def _round_trip(text: str) -> str:
        # Device code is printed as Python source and read back before the
        # walker sees it.
        source = ast.unparse(parse_msl_expression(text))
        return _ast_expr_to_msl(ast.parse(source, mode="eval").body)

    def test_round_trip(self) -> None:
        cases = {
            "a && b || !c": "((a && b) || (!c))",
            "a ? static_cast<decltype(b+c)>(b) : static_cast<decltype(b+c)>(c)": (
                "(a ? static_cast<decltype((b + c))>(b) "
                ": static_cast<decltype((b + c))>(c))"
            ),
            "::metal::precise::atan2(y, x)": "::metal::precise::atan2(y, x)",
            "c10::metal::hypot(metal::fabs(a), metal::fabs(b))": (
                "c10::metal::hypot(metal::fabs(a), metal::fabs(b))"
            ),
            "as_type<uint>(static_cast<float>(a))": "as_type<uint>(static_cast<float>(a))",
            "static_cast<decltype(a)>((a > 0) - (a < 0))": (
                "static_cast<decltype(a)>(((a > 0) - (a < 0)))"
            ),
            # A cast binds to its operand, not to a sum around it.
            "a + static_cast<float>(b) * c": "(a + (static_cast<float>(b) * c))",
            # ``<`` after a name that isn't a template function compares.
            "a < b": "(a < b)",
            "a < b < c": "((a < b) < c)",
            "-HUGE_VALF": "(-HUGE_VALF)",
            "x ? true : false": "(x ? true : false)",
            "a ? b : c ? d : e": "(a ? b : (c ? d : e))",
        }
        for text, msl in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self._round_trip(text), msl)

    def test_outside_the_grammar_is_refused(self) -> None:
        for text in (
            "a = b",
            "[&] { return a; }()",
            "a ? b",
            "static_cast<float>",
            "a.if",
            "a::1",
            "lambda",
            "--a",
            "a++",
            "p->x",
        ):
            with (
                self.subTest(text=text),
                self.assertRaises(helion.exc.BackendUnsupported),
            ):
                parse_msl_expression(text)

    def test_every_override_parses(self) -> None:
        """Each op of Helion's MetalOverrides that renders from operand names
        alone (no kernel in scope) gives MSL the parser reads.  Ops built from
        other ops (``ops.*``) render only inside a kernel and are covered by
        the kernels that use them."""

        class Operand(str):
            __slots__ = ()
            dtype = torch.float32

        overrides = MetalOverrides()
        parsed = set()
        for name in sorted(dir(overrides)):
            if name.startswith("_"):
                continue
            fn = getattr(overrides, name)
            try:
                params = [
                    p
                    for p in inspect.signature(fn).parameters.values()
                    if p.default is p.empty
                ]
            except (TypeError, ValueError):
                continue
            args = [
                torch.float32 if "dtype" in p.name else Operand(f"v_{p.name}")
                for p in params
            ]
            try:
                text = fn(*args)
            except Exception:
                continue
            if not isinstance(text, str):
                continue
            with self.subTest(op=name, text=text):
                self._round_trip(text)
            parsed.add(name)
        # The ops whose MSL isn't Python syntax, and the special functions
        # Helion renders itself.
        self.assertLessEqual(
            {"logical_and", "logical_or", "where", "atan2", "erf", "lgamma", "sign"},
            parsed,
        )
        self.assertGreater(len(parsed), 90)


class TestMetalLaneFoldRewrite(unittest.TestCase):
    """``unroll_lane_folds`` on hand-written device bodies."""

    def _unroll(self, source: str) -> str:
        body = ast.parse(textwrap.dedent(source)).body
        return ast.unparse(
            ast.Module(body=unroll_lane_folds(body, names_read), type_ignores=[])
        )

    def test_lanes_jam_inside_loops(self) -> None:
        """Lane-dependent statements repeat per lane, renamed; a fold combines
        its lane copies; the rest, including the inner loop, appears once."""
        got = self._unroll(
            """
            for slice_lane_2 in range(3):
                indices_2 = tid[0] + slice_lane_2 * 4
                carry = zeros()
                for offset_1 in range(8):
                    v = load(x, indices_2, offset_1)
                    carry = carry + v
                    lane_fold = _metal_lane_fold('sum', v, slice_lane_2, 'float')
                    count = count + 1
                    store(out, indices_2, v * group_sum(lane_fold))
                lane_fold_1 = _metal_lane_fold('max', carry, slice_lane_2, 'half')
                store(best, lane_fold_1)
            """
        )
        expected = """
            indices_2__lane0 = tid[0] + 0 * 4
            indices_2__lane1 = tid[0] + 1 * 4
            indices_2__lane2 = tid[0] + 2 * 4
            carry__lane0 = zeros()
            carry__lane1 = zeros()
            carry__lane2 = zeros()
            for offset_1 in range(8):
                v__lane0 = load(x, indices_2__lane0, offset_1)
                v__lane1 = load(x, indices_2__lane1, offset_1)
                v__lane2 = load(x, indices_2__lane2, offset_1)
                carry__lane0 = carry__lane0 + v__lane0
                carry__lane1 = carry__lane1 + v__lane1
                carry__lane2 = carry__lane2 + v__lane2
                lane_fold = (
                    (static_cast(float, v__lane0) + static_cast(float, v__lane1))
                    + static_cast(float, v__lane2)
                )
                count = count + 1
                store(out, indices_2__lane0, v__lane0 * group_sum(lane_fold))
                store(out, indices_2__lane1, v__lane1 * group_sum(lane_fold))
                store(out, indices_2__lane2, v__lane2 * group_sum(lane_fold))
            lane_fold_1 = c10.metal.max(
                c10.metal.max(
                    static_cast(half, carry__lane0), static_cast(half, carry__lane1)
                ),
                static_cast(half, carry__lane2),
            )
            store(best, lane_fold_1)
            """
        self.assertEqual(got, ast.unparse(ast.parse(textwrap.dedent(expected))))

    def test_lane_arrays_one_per_lane(self) -> None:
        """An array stored at an index other than the lane, from a
        lane-dependent value, is one array per lane; one indexed by the lane
        itself stays shared."""
        got = self._unroll(
            """
            for slice_lane_1 in range(2):
                indices_1 = tid[0] + slice_lane_1 * 16
                keys = _metal_lane_array('bfloat', 4)
                by_lane = _metal_lane_array('bfloat', 2)
                for lane_2 in range(4):
                    k = load(x, indices_1, lane_2)
                    _metal_lane_put(keys, lane_2, k)
                    _metal_lane_put(by_lane, slice_lane_1, k)
                    lane_fold = _metal_lane_fold('sum', k, slice_lane_1, 'float')
                for lane_2 in range(4):
                    store(out, indices_1, _metal_lane_get(keys, lane_2))
            """
        )
        expected = """
            indices_1__lane0 = tid[0] + 0 * 16
            indices_1__lane1 = tid[0] + 1 * 16
            keys__lane0 = _metal_lane_array('bfloat', 4)
            keys__lane1 = _metal_lane_array('bfloat', 4)
            by_lane = _metal_lane_array('bfloat', 2)
            for lane_2 in range(4):
                k__lane0 = load(x, indices_1__lane0, lane_2)
                k__lane1 = load(x, indices_1__lane1, lane_2)
                _metal_lane_put(keys__lane0, lane_2, k__lane0)
                _metal_lane_put(keys__lane1, lane_2, k__lane1)
                _metal_lane_put(by_lane, 0, k__lane0)
                _metal_lane_put(by_lane, 1, k__lane1)
                lane_fold = static_cast(float, k__lane0) + static_cast(float, k__lane1)
            for lane_2 in range(4):
                store(out, indices_1__lane0, _metal_lane_get(keys__lane0, lane_2))
                store(out, indices_1__lane1, _metal_lane_get(keys__lane1, lane_2))
            """
        self.assertEqual(got, ast.unparse(ast.parse(textwrap.dedent(expected))))

    def test_combine_is_balanced(self) -> None:
        """Five lanes nest three deep, the odd lane carried up a level."""
        self.assertEqual(
            balanced_combine("sum", ["a", "b", "c", "d", "e"]),
            "(((a + b) + (c + d)) + e)",
        )
        self.assertEqual(balanced_combine("min", ["a"]), "a")

    def test_body_without_fold_is_unchanged(self) -> None:
        body = ast.parse("for slice_lane_0 in range(2):\n    store(out, x)").body
        self.assertIs(unroll_lane_folds(body, names_read), body)

    def test_run_once_statements(self) -> None:
        """A statement the caller runs once stays whole between the lane
        copies, unrenamed, and is reported once it reads a lane-dependent
        name."""
        loop = ast.parse(
            textwrap.dedent(
                """
                for lane_0 in range(2):
                    v = load(x, lane_0)
                    r = _coop(t, v)
                    store(out, lane_0, v * r)
                """
            )
        ).body[0]
        assert isinstance(loop, ast.For)

        def run_once(stmt: ast.stmt) -> set[str] | None:
            if "_coop" in ast.unparse(stmt):
                return {"t", "v"}
            return None

        jam = LaneJam(loop, run_once)
        expected = """
            v__lane0 = load(x, 0)
            v__lane1 = load(x, 1)
            r = _coop(t, v)
            store(out, 0, v__lane0 * r)
            store(out, 1, v__lane1 * r)
            """
        self.assertEqual(
            ast.unparse(ast.Module(body=jam.jam(), type_ignores=[])),
            ast.unparse(ast.parse(textwrap.dedent(expected))),
        )
        self.assertEqual(jam.lane_reading_units(), [loop.body[1]])

    def test_carry_through_one_branch(self) -> None:
        """A name only one branch of an ``if`` assigns is undeclared after it
        in MSL, so a read there carries it from the previous lane."""
        loop = ast.parse(
            textwrap.dedent(
                """
                for lane_0 in range(2):
                    v = load(x, lane_0)
                    if flag:
                        w = v + 1
                    store(out, lane_0, w)
                """
            )
        ).body[0]
        assert isinstance(loop, ast.For)
        self.assertEqual(LaneJam(loop).carried(), {"w"})

    def test_structural_invariants(self) -> None:
        fold = "    lane_fold = _metal_lane_fold('sum', v, lane, 'float')\n"
        for name, source, message in (
            (
                "dynamic extent",
                "for lane in range(n):\n    v = load(x, lane)\n" + fold,
                "constant extent",
            ),
            (
                "lane-dependent control flow",
                "for lane in range(2):\n    v = load(x, lane)\n"
                "    if v:\n        store(out, v)\n" + fold,
                "control flow on a lane-dependent value",
            ),
            (
                "unsupported statement",
                "for lane in range(2):\n    v = load(x, lane)\n    acc += v\n" + fold,
                r"acc \+= v",
            ),
            (
                "value carried across lanes",
                "for lane in range(2):\n    w = prev + 1\n    v = load(x, lane)\n"
                "    prev = v\n" + fold,
                r"read before assignment: \{'prev'\}",
            ),
            (
                "read after the lane loop",
                "for lane in range(2):\n    v = load(x, lane)\n"
                + fold
                + "store(out, v)\n",
                r"used outside their lane loop: \{'v'\}",
            ),
            (
                "fold outside its lane loop",
                "v = load(x, lane)\n" + fold.strip(),
                "a lane fold outside its lane loop",
            ),
        ):
            with (
                self.subTest(name),
                self.assertRaisesRegex(AssertionError, message),
            ):
                unroll_lane_folds(ast.parse(source).body, names_read)

    def test_lane_loop_jams_into_loop(self) -> None:
        """A lane loop around a loop with a lane-independent load moves
        inside it: the load, and what is computed from it, runs once per
        iteration, and each lane keeps its own carry."""
        body = ast.parse(
            textwrap.dedent(
                """
                for lane_0 in range(2):
                    indices_0 = offset_0 + lane_0
                    acc = 0.0
                    for offset_1 in range(0, 64, 8):
                        w = tl.load(wq, indices_0, offset_1)
                        x = tl.load(xs, offset_1)
                        xe = exp(x)
                        acc = acc + w * xe
                    store(out, indices_0, acc)
                """
            )
        ).body
        expected = """
            indices_0__lane0 = offset_0 + 0
            indices_0__lane1 = offset_0 + 1
            acc__lane0 = 0.0
            acc__lane1 = 0.0
            for offset_1 in range(0, 64, 8):
                w__lane0 = tl.load(wq, indices_0__lane0, offset_1)
                w__lane1 = tl.load(wq, indices_0__lane1, offset_1)
                x = tl.load(xs, offset_1)
                xe = exp(x)
                acc__lane0 = acc__lane0 + w__lane0 * xe
                acc__lane1 = acc__lane1 + w__lane1 * xe
            store(out, indices_0__lane0, acc__lane0)
            store(out, indices_0__lane1, acc__lane1)
            """
        self.assertEqual(
            ast.unparse(ast.Module(body=jam_lanes_into_loops(body), type_ignores=[])),
            ast.unparse(ast.parse(textwrap.dedent(expected))),
        )

    def test_lane_loop_jams_inside_a_loop(self) -> None:
        """A lane loop inside another loop jams too: what the outer loop's
        next iteration reads is assigned before each lane reads it."""
        body = ast.parse(
            textwrap.dedent(
                """
                for offset_0 in range(0, 8, 2):
                    for lane_0 in range(2):
                        indices_0 = offset_0 + lane_0
                        acc = 0.0
                        for o in range(4):
                            v = tl.load(xs, o)
                            acc = acc + tl.load(wq, indices_0, o) * v
                        store(out, indices_0, acc)
                """
            )
        ).body
        expected = """
            for offset_0 in range(0, 8, 2):
                indices_0__lane0 = offset_0 + 0
                indices_0__lane1 = offset_0 + 1
                acc__lane0 = 0.0
                acc__lane1 = 0.0
                for o in range(4):
                    v = tl.load(xs, o)
                    acc__lane0 = acc__lane0 + tl.load(wq, indices_0__lane0, o) * v
                    acc__lane1 = acc__lane1 + tl.load(wq, indices_0__lane1, o) * v
                store(out, indices_0__lane0, acc__lane0)
                store(out, indices_0__lane1, acc__lane1)
            """
        self.assertEqual(
            ast.unparse(ast.Module(body=jam_lanes_into_loops(body), type_ignores=[])),
            ast.unparse(ast.parse(textwrap.dedent(expected))),
        )

    def test_lane_loop_jams_before_a_sibling(self) -> None:
        """A later lane loop that assigns the lane names again before reading
        them doesn't keep an earlier one from jamming."""
        body = ast.parse(
            textwrap.dedent(
                """
                for lane_0 in range(2):
                    acc = 0.0
                    for k in range(8):
                        v = tl.load(y, k)
                        acc = acc + tl.load(x, lane_0, k) * v
                    tl.store(out, lane_0, acc)
                for lane_0 in range(2):
                    tl.store(out2, lane_0, 1.0)
                """
            )
        ).body
        jammed = jam_lanes_into_loops(body)
        self.assertNotIsInstance(jammed[0], ast.For)
        self.assertEqual(ast.unparse(jammed[-1]), ast.unparse(body[1]))

    def test_lane_loop_kept(self) -> None:
        """A lane loop stays whole when the jam saves no load, would copy a
        statement more than ``MAX_JAMMED_COPIES`` times, or can't keep the
        lanes apart."""
        for name, source in (
            (
                "no lane-independent load",
                """
                for lane_0 in range(2):
                    acc = 0.0
                    for o in range(4):
                        idx = tid[0] * 4 + o
                        acc = acc + tl.load(x, lane_0, idx)
                    store(out, lane_0, acc)
                """,
            ),
            (
                "too many lanes",
                """
                for lane_0 in range(33):
                    acc = 0.0
                    for o in range(4):
                        v = tl.load(y, o)
                        acc = acc + tl.load(x, lane_0, o) * v
                    store(out, lane_0, acc)
                """,
            ),
            (
                "too many copies with an earlier jam's",
                """
                for lane_0 in range(16):
                    acc = 0.0
                    for o in range(4):
                        v = tl.load(y, o)
                        acc = acc + tl.load(x, lane_0, o) * v * s__lane7
                    store(out, lane_0, acc)
                """,
            ),
            (
                "lane-dependent value read after the loop",
                """
                r = 0.0
                for lane_0 in range(2):
                    acc = 0.0
                    for o in range(4):
                        v = tl.load(y, o)
                        acc = acc + tl.load(x, lane_0, o) * v
                    r = acc
                store(out, 0, r)
                """,
            ),
            (
                "lane-independent value carried across lanes",
                """
                for lane_0 in range(8):
                    acc_copy = acc
                    t = 0.0
                    for o in range(4):
                        v = tl.load(z, o)
                        t = t + v
                    acc = acc_copy + t
                """,
            ),
            (
                "lane value read by the enclosing loop's next iteration",
                """
                for o in range(4):
                    use(acc)
                    for lane_0 in range(2):
                        acc = 0.0
                        for k in range(4):
                            v = tl.load(y, k)
                            acc = acc + tl.load(x, lane_0, k) * v
                """,
            ),
            (
                "a loop with no shared load beside one with it",
                """
                for lane_0 in range(2):
                    acc = 0.0
                    for o in range(4):
                        acc = acc + tl.load(x, lane_0, o)
                    for o in range(4):
                        v = tl.load(y, o)
                        store(out, lane_0, o, acc * v)
                """,
            ),
            (
                "shared load only in an inner lane loop",
                """
                for lane_0 in range(2):
                    for lane_1 in range(4):
                        v = tl.load(y, lane_1)
                        store(out, lane_0, lane_1, tl.load(x, lane_0, lane_1) + v)
                """,
            ),
            (
                "lane arrays past the budget once copied per lane",
                """
                for lane_0 in range(8):
                    arr = _metal_lane_array('half', 8)
                    for o in range(4):
                        v = tl.load(y, o)
                        _metal_lane_put(arr, o, tl.load(x, lane_0, o) * v)
                    store(out, lane_0, _metal_lane_get(arr, 0))
                """,
            ),
            (
                "lane-dependent control flow",
                """
                for lane_0 in range(2):
                    acc = 0.0
                    for o in range(4):
                        v = tl.load(y, o)
                        if lane_0:
                            acc = acc + v
                    store(out, lane_0, acc)
                """,
            ),
            (
                "value carried across lanes",
                """
                for lane_0 in range(2):
                    for o in range(4):
                        v = tl.load(y, o)
                        acc = acc + v * lane_0
                    store(out, lane_0, acc)
                """,
            ),
        ):
            with self.subTest(name):
                body = ast.parse(textwrap.dedent(source)).body
                self.assertEqual(
                    ast.unparse(
                        ast.Module(body=jam_lanes_into_loops(body), type_ignores=[])
                    ),
                    ast.unparse(ast.Module(body=body, type_ignores=[])),
                )


@helion.kernel(backend="metal", static_shapes=True)
def rows_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T with x's few rows walked as lanes of each thread."""
    m, k = x.size()
    n = w.size(0)
    bk = hl.register_block_size(8, 64)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n, bk], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=bk):
            xk = x[tile_m, tile_k].to(torch.float32)[:, None, :]
            acc = acc + xk * w[tile_n, tile_k].to(torch.float32)[None, :, :]
        out[tile_m, tile_n] = acc.sum(2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lane_sum_of_row_sums(z: torch.Tensor, n: int) -> torch.Tensor:
    """Each row's sum, added once per element of an n-wide tile: a value
    every lane adds to one carry, which depends on no lane."""
    m, j = z.size()
    out = torch.empty([m], dtype=torch.float32, device=z.device)
    bn = hl.register_block_size(8, 8)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=torch.float32)
        for _tile_n in hl.tile(n, block_size=bn):
            t = hl.zeros([tile_m], dtype=torch.float32)
            for tile_j in hl.tile(j):
                t = t + z[tile_m, tile_j].sum(-1)
            acc = acc + t[:, None]
        out[tile_m] = acc.sum(-1)
    return out


@_requires_darwin
class TestMetalLaneJam(unittest.TestCase):
    """``jam_lanes_into_loops`` in a compiled kernel."""

    def test_rows_share_weight_loads(self) -> None:
        """Four rows on lanes load each weight element once per K step."""
        x = torch.randn(4, 64, device=DEVICE, dtype=torch.float16)
        w = torch.randn(48, 64, device=DEVICE, dtype=torch.float16)
        config = helion.Config(block_sizes=[8, 4, 16], num_threads=[8, 1, 16])
        out, code, msl = _run_with_msl(rows_gemv, (x, w), config)
        torch.testing.assert_close(
            out.cpu().double(),
            x.cpu().double() @ w.cpu().double().T,
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertRegex(code, r"for lane_\d+ in range\(4\)")
        self.assertNotIn("for (int lane_", msl)
        self.assertEqual(msl.count("_helion_masked(w["), 1)
        self.assertEqual(msl.count("_helion_masked(x["), 4)

    def test_lane_independent_carry(self) -> None:
        """A carry every lane adds a lane-independent value to keeps its
        lane loop: jammed, the value would be added once."""
        z = torch.randn(8, 32, device=DEVICE)
        config = helion.Config(block_sizes=[8, 4, 8], num_threads=[1, 4, 8])
        out, _, _ = _run_with_msl(lane_sum_of_row_sums, (z, 16), config)
        torch.testing.assert_close(
            out.cpu(), 16 * z.cpu().sum(-1), rtol=1e-4, atol=1e-4
        )


@helion.kernel(backend="metal", static_shapes=True)
def mul_add(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * w[tile] + r[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mul_round_add(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """The product rounded to float16 before the add."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        p = (x[tile] * w[tile]).to(torch.float16).to(torch.float32)
        out[tile] = p + r[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_into(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def double_in_place(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] * 2
    return x


@helion.kernel(backend="metal", static_shapes=False)
def prefix_or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x where it reaches, y past its end: x's loads are all masked off
    when x is empty."""
    m = x.size(0)
    out = torch.empty_like(y)
    for tile in hl.tile(y.size(0)):
        valid = tile.index < m
        xv = hl.load(x, [tile], extra_mask=valid)
        out[tile] = torch.where(valid, xv, y[tile])
    return out


@_requires_darwin
class TestMetalMaskedLoads(unittest.TestCase):
    """A masked load reads unconditionally, at element 0 when masked off,
    and selects its fill afterwards, unless its tensor is empty, written by
    the kernel or sharing storage with another argument."""

    def test_tail(self) -> None:
        x, y = (torch.randn(1000, device=DEVICE) for _ in range(2))
        out, _, msl = _run_with_msl(
            vector_add, (x, y), helion.Config(block_sizes=[256])
        )
        self.assertIn("inline auto _helion_masked(", msl)
        self.assertIn("_helion_masked(x[(mask_0 ? indices_0 : 0)], mask_0, 0)", msl)
        torch.testing.assert_close(out, x + y, atol=0, rtol=0)

    def test_written_tensor_loads_conditionally(self) -> None:
        """A tensor the kernel also stores to keeps its loads under their
        mask: a masked-off lane reading element 0 would race the thread
        that writes it."""
        x = torch.randn(1000, device=DEVICE)
        expected = x * 2
        _, _, msl = _run_with_msl(
            double_in_place, (x,), helion.Config(block_sizes=[256])
        )
        self.assertNotIn("_helion_masked(", msl)
        self.assertIn("(mask_0 ? x[indices_0] : (0))", msl)
        torch.testing.assert_close(x, expected, atol=0, rtol=0)

    def test_aliased_tensor_loads_conditionally(self) -> None:
        """A tensor passed again as the output keeps its loads under their
        mask: the in-place call races element 0 just as a store would.  The
        launch plan keys on the aliasing, so the module compiled for distinct
        tensors compiles again."""
        x = torch.randn(1000, device=DEVICE)
        y = torch.randn(1000, device=DEVICE)
        out = torch.empty_like(x)
        code = add_into.bind((x, y, out)).to_code(helion.Config(block_sizes=[256]))
        module = PyCodeCache.load(code)
        (jit,) = [o for o in vars(module).values() if isinstance(o, _MetalKernel)]
        for aliased in (False, True):
            with self.subTest(aliased=aliased):
                dst = x.clone() if aliased else out
                src = dst if aliased else x
                expected = src + y
                module.add_into(src, y, dst)
                assert jit.msl_source is not None
                self.assertEqual("_helion_masked(x[" in jit.msl_source, not aliased)
                torch.testing.assert_close(dst, expected, atol=0, rtol=0)

    def test_empty_tensor_loads_conditionally(self) -> None:
        """An empty x has no element 0, so its loads stay under their mask.
        The launch plan keys on emptiness: the module compiled for a 5-long x
        compiles again for an empty one instead of reusing the shader."""
        y = torch.randn(70, device=DEVICE)  # a masked tail tile
        code = prefix_or.bind((torch.randn(5, device=DEVICE), y)).to_code(
            helion.Config(block_sizes=[32])
        )
        module = PyCodeCache.load(code)
        (jit,) = [o for o in vars(module).values() if isinstance(o, _MetalKernel)]
        for m in (5, 0):
            with self.subTest(m=m):
                x = torch.randn(m, device=DEVICE)
                out = module.prefix_or(x, y)
                assert jit.msl_source is not None
                self.assertEqual("_helion_masked(x[" in jit.msl_source, m > 0)
                self.assertIn("_helion_masked(y[", jit.msl_source)
                expected = y.clone()
                expected[:m] = x
                torch.testing.assert_close(out, expected, atol=0, rtol=0)


@helion.kernel(
    backend="metal", static_shapes=True, config=helion.Config(block_sizes=[64])
)
def index_minus_three(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = (tile.index - 3).to(torch.float32)
    return out


@helion.kernel(
    backend="metal", static_shapes=True, config=helion.Config(block_sizes=[128])
)
def previous_row(x: torch.Tensor) -> torch.Tensor:
    """Row t is x's row t - 1, 0 for t = 0: a load whose row goes negative,
    masked off by a condition on ``tile.index``."""
    t_len, n = x.shape
    out = torch.empty_like(x)
    for tile_n, tile_t in hl.tile([n, t_len], block_size=[None, 1]):
        t = tile_t.begin
        out[t, tile_n] = hl.load(
            x, [t - 1, tile_n], extra_mask=tile_n.index * 0 + t - 1 >= 0
        )
    return out


@_requires_darwin
class TestMetalTileIndexIsSigned(unittest.TestCase):
    """``tile.index`` has the signed index dtype although Metal's loop
    indices are ``uint``: arithmetic on it must not wrap."""

    def test_negative_values(self) -> None:
        out = index_minus_three(torch.empty(64, device=DEVICE))
        expected = torch.arange(-3, 61, dtype=torch.float32, device=DEVICE)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_mask_on_negative_row(self) -> None:
        """At one row, x's row dimension has size 1 and drops out of the
        address, so the masked-off load of row -1 lands in bounds on row 0:
        the mask alone keeps it out (it used to wrap to true)."""
        for t_len in (1, 3):
            with self.subTest(t_len=t_len):
                x = torch.randn(t_len, 128, device=DEVICE)
                expected = torch.zeros_like(x)
                expected[1:] = x[:-1]
                out = previous_row(x)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)


@_requires_darwin
class TestMetalFmaFusion(unittest.TestCase):
    def test_float_multiply_add(self) -> None:
        x, w, r = (torch.randn(1000, device=DEVICE) for _ in range(3))
        out, _, msl = _run_with_msl(
            mul_add, (x, w, r), helion.Config(block_sizes=[256])
        )
        self.assertIn("metal::fma(", msl)
        torch.testing.assert_close(out, x * w + r, atol=1e-5, rtol=1e-5)

    def test_rounding_kept(self) -> None:
        """The float16 rounding between the multiply and the add stays:
        bit-exact with eager."""
        x, w, r = (torch.randn(1000, device=DEVICE) for _ in range(3))
        out, _, msl = _run_with_msl(
            mul_round_add, (x, w, r), helion.Config(block_sizes=[256])
        )
        self.assertNotIn("metal::fma(", msl)
        expected = (x * w).to(torch.float16).to(torch.float32) + r
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


@helion.kernel(backend="metal", static_shapes=True)
def high_halves(t: torch.Tensor) -> torch.Tensor:
    """The float16 in each int32's high 16 bits."""
    out = torch.empty(t.shape, dtype=torch.float16, device=t.device)
    for tile in hl.tile(t.size(0)):
        out[tile] = (t[tile] >> 16).to(torch.int16).view(torch.float16)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def int32_as_halves(t: torch.Tensor) -> torch.Tensor:
    """Each int32 as its two float16 halves: a view to a narrower dtype."""
    out = torch.empty([t.size(0), 2], dtype=torch.float16, device=t.device)
    for tile in hl.tile(t.size(0)):
        out[tile, :] = t[tile].view(torch.float16).view(-1, 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_bit_sums(x: torch.Tensor) -> torch.Tensor:
    """The sum of each row's 16-bit patterns, as integers."""
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.int32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].view(torch.int16).to(torch.int32).sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def negated_row_bit_sums(t: torch.Tensor) -> torch.Tensor:
    """row_bit_sums of -t: a masked lane of -t holds -0.0, bits 0x8000."""
    m, _ = t.shape
    out = torch.empty([m], dtype=torch.int32, device=t.device)
    for tile_m in hl.tile(m):
        out[tile_m] = (-t[tile_m, :]).view(torch.int16).to(torch.int32).sum(-1)
    return out


@_requires_darwin
class TestMetalViewDtype(unittest.TestCase):
    def test_equal_width(self) -> None:
        hi = torch.randn(100, dtype=torch.float16)
        lo = torch.randn(100, dtype=torch.float16)
        words = (hi.view(torch.int16).to(torch.int32) << 16) | (
            lo.view(torch.int16).to(torch.int32) & 0xFFFF
        )
        out, _, msl = _run_with_msl(
            high_halves, (words.to(DEVICE),), helion.Config(block_sizes=[64])
        )
        self.assertIn("as_type<half>(static_cast<short>(", msl)
        torch.testing.assert_close(out.cpu(), hi, atol=0, rtol=0)

    def test_bit_patterns_in_lane_copies(self) -> None:
        """Rows folded into a thread's lanes view each element's exact bits:
        a signaling NaN, -0.0, a subnormal and infinity among them (a lane
        copy's bfloat conversion leaves a bfloat operand as it is)."""
        special = [0x7F81, 0x8000, 0x0001, 0x7F80, 0xFF80, 0x0081]
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                bits = torch.randn(6, 256).to(dtype).view(torch.int16)
                bits[0, : len(special)] = torch.tensor(special).to(torch.int16)
                x = bits.view(dtype)
                out, _, msl = _run_with_msl(
                    row_bit_sums,
                    (x.to(DEVICE),),
                    helion.Config(block_sizes=[2], num_warps=1),
                )
                self.assertIn("__lane1", msl)
                expected = bits.to(torch.int32).sum(-1, dtype=torch.int32)
                torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    def test_masked_lanes_of_a_view(self) -> None:
        """A view of a value whose masked lanes hold -0.0 isn't zero there,
        so the integer sum over the 100 columns of a 128-wide block keeps
        its mask."""
        t = torch.rand(8, 100, dtype=torch.float16) + 0.5
        out, _, _ = _run_with_msl(
            negated_row_bit_sums, (t.to(DEVICE),), helion.Config(block_sizes=[4])
        )
        expected = (-t).view(torch.int16).to(torch.int32).sum(-1, dtype=torch.int32)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    def test_width_mismatch(self) -> None:
        words = torch.zeros(100, dtype=torch.int32, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "view.dtype from torch.int32 to torch.float16"
        ):
            int32_as_halves.bind((words,)).to_code(helion.Config(block_sizes=[64]))


# ---------------------------------------------------------------------------
# Kernel definitions - hl.arange over a block size that no loop owns
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def free1d(x: torch.Tensor) -> torch.Tensor:
    """A jagged 1-D gather under a grid, whose root strategy is flattened."""
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        idx = g * bm + hl.arange(bm)
        valid = idx < n
        v = hl.load(x, [idx], extra_mask=valid)
        hl.store(out, [idx], v * 2 + idx.to(torch.float32), extra_mask=valid)
    return out


@helion.kernel(backend="metal", static_shapes=False)
def free1d_dynamic(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        idx = g * bm + hl.arange(bm)
        valid = idx < n
        v = hl.load(x, [idx], extra_mask=valid)
        hl.store(out, [idx], v * 2 + idx.to(torch.float32), extra_mask=valid)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free2d(x: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(32, 32)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            valid = (rows < m)[:, None] & (cols < n)[None, :]
            v = hl.load(x, [rows, cols], extra_mask=valid)
            rows_f = rows.to(torch.float32)[:, None]
            val = v + rows_f * 1000 + cols.to(torch.float32)[None, :]
            hl.store(out, [rows, cols], val, extra_mask=valid)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_inner_tile(x: torch.Tensor) -> torch.Tensor:
    """grouped_gemm's layout without the matmul: free rows, an inner tile."""
    m, k = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        rv = rows < m
        for tk in hl.tile(k):
            v = hl.load(x, [rows, tk.index], extra_mask=rv[:, None])
            val = v * 3 + rows.to(torch.float32)[:, None]
            hl.store(out, [rows, tk.index], val, extra_mask=rv[:, None])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_carry(x: torch.Tensor) -> torch.Tensor:
    """A free accumulator carried through an unthreaded inner grid."""
    m, k = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        rv = rows < m
        acc = hl.zeros([bm], dtype=torch.float32)
        for j in hl.grid(k):
            acc = acc + hl.load(x, [rows, j], extra_mask=rv) * (j + 1)
        hl.store(out, [rows], acc, extra_mask=rv)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_beside_tile_root(x: torch.Tensor, nb: int) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(16, 16)
    for tm in hl.tile(m):
        for gn in hl.grid(nb):
            cols = gn * bn + hl.arange(bn)
            cv = cols < n
            v = hl.load(x, [tm, cols], extra_mask=cv[None, :])
            val = v - cols.to(torch.float32)[None, :]
            hl.store(out, [tm, cols], val, extra_mask=cv[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_int64(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros(n, dtype=torch.int64, device=x.device)
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        idx = g * bm + hl.arange(bm, dtype=torch.int64)
        hl.store(out, [idx], idx * 3, extra_mask=idx < n)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_strided(x: torch.Tensor) -> torch.Tensor:
    """A computed start: scatter to the even positions."""
    (n,) = x.shape
    out = torch.zeros(2 * n, dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        src = g * bm + hl.arange(bm)
        v = hl.load(x, [src], extra_mask=src < n)
        hl.store(out, [2 * src], v, extra_mask=src < n)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_step(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    for g in hl.grid(n // (2 * bm)):
        idx = g * 2 * bm + hl.arange(0, 2 * bm, 2)
        out[idx] = x[idx] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_shifted(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros(n, dtype=torch.int32, device=x.device)
    bm = hl.register_block_size(32, 32)
    for g in hl.grid(n // 32):
        idx = g * bm + hl.arange(bm)
        shifted = idx - 5
        hl.store(out, [idx], torch.where(shifted >= 0, shifted, -1))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_beside_sibling_tiles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two sibling tile loops beside a free axis, each writing its own tensor."""
    m, n = x.shape
    out = torch.zeros_like(x)
    out2 = torch.zeros_like(x)
    bm = hl.register_block_size(8, 8)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        rv = rows < m
        for ta in hl.tile(n):
            v = hl.load(x, [rows, ta.index], extra_mask=rv[:, None])
            hl.store(out, [rows, ta.index], v + 1, extra_mask=rv[:, None])
        for tb in hl.tile(n):
            v = hl.load(x, [rows, tb.index], extra_mask=rv[:, None])
            hl.store(out2, [rows, tb.index], v * 2, extra_mask=rv[:, None])
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def free_axis2(x: torch.Tensor) -> torch.Tensor:
    """One free block above a 2-D threaded root tile."""
    m, n, p = x.shape
    out = torch.zeros_like(x)
    bp = hl.register_block_size(4, 4)
    for tm, tn in hl.tile([m, n]):
        for _ in hl.grid(1):
            r = hl.arange(bp)
            out[tm, tn, r] = x[tm, tn, r] + r.to(torch.float32)[None, None, :]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_beside_second_root(
    x: torch.Tensor, y: torch.Tensor, nb: int
) -> tuple[torch.Tensor, torch.Tensor]:
    (n,) = x.shape
    out = torch.zeros_like(x)
    out2 = torch.zeros_like(y)
    bm = hl.register_block_size(32, 32)
    for g in hl.grid(nb):
        idx = g * bm + hl.arange(bm)
        v = hl.load(x, [idx], extra_mask=idx < n)
        hl.store(out, [idx], v * 2, extra_mask=idx < n)
    for t in hl.tile(y.size(0)):
        out2[t] = y[t] * 3 + 1
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def where_ok(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    for g in hl.grid((n + bm - 1) // bm):
        idx = g * bm + hl.arange(bm)
        valid = idx < n
        v = hl.load(x, [idx], extra_mask=valid)
        val = torch.where(v > 0, v, idx.to(torch.float32))
        hl.store(out, [idx], val, extra_mask=valid)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def where_2d_ok(x: torch.Tensor) -> torch.Tensor:
    """A ``[bn]`` operand right-aligned onto a ``[bm, bn]`` result."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(32, 32)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        v = hl.load(x, [rows, cols])
        hl.store(out, [rows, cols], torch.where(v > 0, cols.to(torch.float32), v))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_colon(x: torch.Tensor) -> torch.Tensor:
    """Free rows beside a ``:`` slice, whose unused rolled copy repeats the
    store."""
    m, n = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    for g in hl.grid(m // 16):
        rows = g * bm + hl.arange(bm)
        hl.store(out, [rows, slice(None)], hl.load(x, [rows, slice(None)]) * 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_3d_ok(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``[tm, bp]`` onto ``[tm, bn, bp]``: Triton places it at dims 0, 2."""
    m, n, q = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = hl.arange(bp)
        v = hl.load(x, [tm, cols, p])
        hl.store(out, [tm, cols, p], v + hl.load(y, [tm, p]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_middle_ok(x: torch.Tensor) -> torch.Tensor:
    """``[bn]`` onto ``[tm, bn, bp]``: Triton places it at dim 1."""
    m, n, q = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = hl.arange(bp)
        v = hl.load(x, [tm, cols, p])
        hl.store(out, [tm, cols, p], v + cols.to(torch.float32) * 100.0)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_static_ok(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``[tm, 8]`` (a ``:`` slice) onto ``[tm, bn, 8]``: dims 0, 2."""
    m, n, q = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        v = hl.load(x, [tm, cols, slice(None)])
        w = hl.load(y, [tm, slice(None)])
        hl.store(out, [tm, cols, slice(None)], v + w)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def idx_bcast_ok(x: torch.Tensor, ridx: torch.Tensor) -> torch.Tensor:
    """``x[r2, 7 - cols]``: index tensors ``[tm, bn]`` and ``[bn]``."""
    m, n = x.shape
    out = torch.zeros([m, 8], dtype=x.dtype, device=x.device)
    bn = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        r2 = hl.load(ridx, [tm, cols])
        hl.store(out, [tm, cols], hl.load(x, [r2, 7 - cols]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def idx_bcast_2d_ok(x: torch.Tensor, ridx: torch.Tensor) -> torch.Tensor:
    """``x[r2, c2]``: two ``[tm, bn]`` index tensors."""
    m, n = x.shape
    out = torch.zeros([m, 8], dtype=x.dtype, device=x.device)
    bn = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        r2 = hl.load(ridx, [tm, cols])
        c2 = 7 - r2 % 8 + 0 * cols[None, :]
        hl.store(out, [tm, cols], hl.load(x, [r2, c2]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def idx_cartesian_host(x: torch.Tensor, cidx: torch.Tensor) -> torch.Tensor:
    """``x[rows, cidx]``: 1-D index tensors form a Cartesian product, and
    ``cidx``'s length is no block, so they reach the index-tensor check."""
    m, n = x.shape
    out = torch.zeros([m, cidx.size(0)], dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(16, 16)
    for g in hl.grid(m // 16):
        rows = g * bm + hl.arange(bm)
        out[rows, :] = x[rows, cidx]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dot_free_host(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """grouped_gemm's layout with ``hl.dot(acc=)``, with grid counts passed
    from the host."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = hl.dot(a_blk, b_blk, acc=acc)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dot16_free_host(
    a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int
) -> torch.Tensor:
    """``dot_free_host`` with a float16 acc, which keeps it scalar."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float16, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float16)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = hl.dot(a_blk, b_blk, acc=acc)
            hl.store(out, [rows, cols], acc, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dot_free_tail(a: torch.Tensor, b: torch.Tensor, nbm: int, nbn: int) -> torch.Tensor:
    """``dot_free_host`` with a bias read after the K loop."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid(nbm):
        for gn in hl.grid(nbn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            rv = rows < m
            cv = cols < n
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                a_blk = hl.load(a, [rows, tk.index], extra_mask=rv[:, None])
                b_blk = hl.load(b, [tk.index, cols], extra_mask=cv[None, :])
                acc = hl.dot(a_blk, b_blk, acc=acc)
            val = acc + cols.to(torch.float32)[None, :]
            hl.store(out, [rows, cols], val, extra_mask=rv[:, None] & cv[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dot_rows_rootlanes(a: torch.Tensor, b: torch.Tensor, nbm: int) -> torch.Tensor:
    """Free rows beside a root tile over N, around the dot's K loop."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 8)
    for tn in hl.tile(n):
        for gm in hl.grid(nbm):
            rows = gm * bm + hl.arange(bm)
            acc = hl.zeros([bm, tn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = hl.dot(hl.load(a, [rows, tk.index]), b[tk, tn], acc=acc)
            hl.store(out, [rows, tn.index], acc)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_free(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """grouped_gemm's layout: free rows and columns, an addmm carry."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        for gn in hl.grid((n + bn - 1) // bn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, a[rows, tk], b[tk, cols])
            out[rows, cols] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_reduce(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    bn = hl.register_block_size(16, 32)
    for g in hl.grid(m):
        cols = hl.arange(bn)
        out[g] = hl.load(x, [g, cols], extra_mask=cols < n).sum(0)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mpp_free(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A canonical root matmul, which MPP takes, beside a free arange."""
    m, k = x.shape
    _, n = y.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    aux = torch.zeros([64], dtype=torch.float32, device=x.device)
    bf = hl.register_block_size(16, 16)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
        r = hl.arange(bf)
        aux[r] = r.to(torch.float32) * 2
    return out, aux


@helion.kernel(backend="metal", static_shapes=True)
def free_store_into_tile(x: torch.Tensor) -> torch.Tensor:
    """A free value stored into a tile dim of the same size."""
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    for t in hl.tile(n, block_size=32):
        out[t] = hl.load(x, [t.begin + 31 - hl.arange(bm)])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_store_into_tile_in_branch(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """``free_store_into_tile``'s store inside an if branch, which no carry
    leaves."""
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    for t in hl.tile(n, block_size=32):
        if hl.load(flag, [0]) > 0:
            out[t] = hl.load(x, [t.begin + 31 - hl.arange(bm)])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_mask_rebind(x: torch.Tensor) -> torch.Tensor:
    """A load masked by another free block's mask of equal size."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    bn = hl.register_block_size(32, 32)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = 31 - hl.arange(bn)
        hl.store(out, [rows], hl.load(x, [rows], extra_mask=cols < 16))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def store_mask_free_dim(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A scalar store masked by a ``[bm]`` mask."""
    (m,) = x.shape
    bm = hl.register_block_size(32, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        hl.store(c, [gm], gm * 1.0, extra_mask=rows < 5)
    return c


@helion.kernel(backend="metal", static_shapes=True)
def idx_bcast_load(x: torch.Tensor, ridx: torch.Tensor) -> torch.Tensor:
    """``x[r2, p]``: ``[bp]`` is right-aligned onto ``[tm, bn]``, meeting bn."""
    m, n = x.shape
    out = torch.zeros([m, 8], dtype=x.dtype, device=x.device)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = 7 - hl.arange(bp)
        r2 = hl.load(ridx, [tm, cols])
        hl.store(out, [tm, cols], hl.load(x, [r2, p]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def idx_bcast_store(x: torch.Tensor, ridx: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = 7 - hl.arange(bp)
        r2 = hl.load(ridx, [tm, cols])
        hl.store(out, [r2, p], hl.load(x, [tm, cols]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def idx_bcast_col(x: torch.Tensor, ridx: torch.Tensor) -> torch.Tensor:
    """``x[r2, p[:, None]]``: ``[bp, 1]`` meets tm's position."""
    m, n = x.shape
    out = torch.zeros([m, 8], dtype=x.dtype, device=x.device)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = 7 - hl.arange(bp)
        r2 = hl.load(ridx, [tm, cols])
        hl.store(out, [tm, cols], hl.load(x, [r2, p[:, None]]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def where_other_free_block(x: torch.Tensor) -> torch.Tensor:
    """``where``'s operand is another free block of equal size."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    bn = hl.register_block_size(32, 32)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = 31 - hl.arange(bn)
        v = hl.load(x, [rows])
        hl.store(out, [rows], torch.where(v > 0, cols.to(torch.float32), 0.0))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def where_lower_rank(x: torch.Tensor) -> torch.Tensor:
    """A ``[bm]`` operand right-aligned onto bn's position of the result."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(16, 16)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        v = hl.load(x, [rows, cols])
        hl.store(out, [rows, cols], torch.where(v > 0, rows.to(torch.float32), -1.0))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_transposed_free(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Operands over two free blocks, the second transposed."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(8, 8)
    bn = hl.register_block_size(8, 8)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        v = hl.load(x, [rows, cols]) + hl.load(y, [cols, rows])
        hl.store(out, [rows, cols], v)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def where_transposed_free(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``where`` over two free blocks, its value transposed."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(8, 8)
    bn = hl.register_block_size(8, 8)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        v = torch.where(hl.load(x, [rows, cols]) > 0, hl.load(y, [cols, rows]), 0.0)
        hl.store(out, [rows, cols], v)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_onto_other_free_dim(x: torch.Tensor) -> torch.Tensor:
    """``[bp]`` onto ``[tm, bn]``: Triton places it on bn's dim."""
    m, n = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = hl.arange(bp)
        v = hl.load(x, [tm, cols])
        hl.store(out, [tm, cols], v + p.to(torch.float32) * 100.0)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_swapped_dims(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``[bn, tm]`` onto ``[tm, bn, bp]``: Triton places it at dims 0, 1."""
    m, n, q = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = hl.arange(bp)
        v = hl.load(x, [tm, cols, p])
        hl.store(out, [tm, cols, p], v + hl.load(y, [cols, tm]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lowrank_permuted_free(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``[bp, bn]`` onto ``[tm, bn, bp]``: matched at dims 2, 1, which the
    expansion takes in order, so bp's dim meets bn."""
    m, n, q = x.shape
    out = torch.zeros_like(x)
    bn = hl.register_block_size(8, 8)
    bp = hl.register_block_size(8, 8)
    for tm in hl.tile(m):
        cols = hl.arange(bn)
        p = hl.arange(bp)
        v = hl.load(x, [tm, cols, p])
        hl.store(out, [tm, cols, p], v + hl.load(y, [p, cols]))
    return out


@helion.kernel(backend="metal", static_shapes=True)
def phi_loop_rebind(x: torch.Tensor) -> torch.Tensor:
    """A carry initialised ``[bn, bm]`` and reassigned ``[bm, bn]``."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(16, 16)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = 15 - hl.arange(bn)
        acc = hl.zeros([bn, bm], dtype=torch.float32)
        for t in hl.grid(2):
            acc = hl.load(x, [rows[:, None], cols[None, :]]) + t
        hl.store(out, [cols[:, None], rows[None, :]], acc)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def carry_read_only_in_loop(x: torch.Tensor) -> torch.Tensor:
    """A rebound carry read only inside its loop."""
    out = torch.zeros([2, 16, 16], device=x.device, dtype=x.dtype)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(16, 16)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        acc = hl.zeros([bn, bm], dtype=torch.float32)
        for it in hl.grid(2):
            hl.store(out, [it, cols, rows], acc)
            acc = hl.load(x, [rows, cols])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def phi_if_rebind(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """An if branch reassigning a ``[bn, bm]`` value with a ``[bm, bn]`` one."""
    out = torch.zeros_like(x)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(16, 16)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = 15 - hl.arange(bn)
        v = hl.zeros([bn, bm], dtype=torch.float32)
        if hl.load(flag, [0]) > 0:
            v = hl.load(x, [rows[:, None], cols[None, :]])
        hl.store(out, [cols[:, None], rows[None, :]], v)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_acc_transposed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """An addmm accumulator ``[bn, bm]`` against a ``[bm, bn]`` product."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(16, 16)
    bn = hl.register_block_size(16, 16)
    for _ in hl.grid(1):
        rows = hl.arange(bm)
        cols = hl.arange(bn)
        acc = hl.zeros([bn, bm], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[rows, tk] * 1.0, b[tk, cols])
        out[cols[:, None], rows[None, :]] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_k_free_acc(a: torch.Tensor, b: torch.Tensor, nk: int) -> torch.Tensor:
    """An accumulating addmm whose K is a free block; the computed lhs keeps
    the MPP region out."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(16, 16)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for kc in hl.grid(nk):
            kr = kc * bk + hl.arange(bk)
            acc = torch.addmm(acc, a[tm, kr] * 1.0, b[kr, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_scaled_bias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        for gn in hl.grid((n + bn - 1) // bn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc * 0.5, a[rows, tk], b[tk, cols])
            out[rows, cols] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_overwrite(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        for gn in hl.grid((n + bn - 1) // bn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for tk in hl.tile(k):
                acc = torch.mm(a[rows, tk], b[tk, cols])
            out[rows, cols] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_swap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Two addmm carries whose accumulators swap every K tile."""
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bm = hl.register_block_size(8, 32)
    bn = hl.register_block_size(8, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        for gn in hl.grid((n + bn - 1) // bn):
            rows = gm * bm + hl.arange(bm)
            cols = gn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            acc2 = hl.zeros([bm, bn], dtype=torch.float32) + 1
            for tk in hl.tile(k):
                acc2, acc = (
                    torch.addmm(acc, a[rows, tk], b[tk, cols]),
                    torch.addmm(acc2, a[rows, tk], b[tk, cols]),
                )
            out[rows, cols] = acc - acc2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_lane_carry(x: torch.Tensor) -> torch.Tensor:
    """A carry through a lane-looped inner tile: ``x[:, 0]`` + #K tiles."""
    m, k = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(32, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        rv = rows < m
        acc = hl.zeros([bm], dtype=torch.float32)
        for _ in hl.tile(k):
            acc = acc + 1
        hl.store(out, [rows], acc + hl.load(x, [rows, 0], extra_mask=rv), extra_mask=rv)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def second_root_in_place(x: torch.Tensor, c: torch.Tensor, nb: int) -> torch.Tensor:
    """A second root updates ``c`` in place on every free-axis thread."""
    (n,) = x.shape
    out = torch.zeros_like(x)
    bm = hl.register_block_size(32, 32)
    for g in hl.grid(nb):
        idx = g * bm + hl.arange(bm)
        v = hl.load(x, [idx], extra_mask=idx < n)
        hl.store(out, [idx], v * 2, extra_mask=idx < n)
    for t in hl.tile(c.size(0)):
        c[t] = c[t] * 2 + 1
    return c


@helion.kernel(backend="metal", static_shapes=True)
def lane_reverse(x: torch.Tensor) -> torch.Tensor:
    """An in-place reverse inside each K tile."""
    m, k = x.shape
    bm = hl.register_block_size(32, 32)
    for gm in hl.grid((m + bm - 1) // bm):
        rows = gm * bm + hl.arange(bm)
        rv = rows < m
        for tk in hl.tile(k):
            rev = tk.begin * 2 + tk.block_size - 1 - tk.index
            v = hl.load(x, [rows, tk.index], extra_mask=rv[:, None])
            hl.store(x, [rows, rev], v, extra_mask=rv[:, None])
    return x


@helion.kernel(backend="metal", static_shapes=True)
def free_axis3(x: torch.Tensor) -> torch.Tensor:
    """Two free blocks above a 2-D threaded root tile: no fourth axis."""
    m, n, p, q = x.shape
    out = torch.zeros_like(x)
    bp = hl.register_block_size(4, 4)
    bq = hl.register_block_size(4, 4)
    for tm, tn in hl.tile([m, n]):
        for _ in hl.grid(1):
            r = hl.arange(bp)
            c = hl.arange(bq)
            val = x[tm, tn, r, c] + r.to(torch.float32)[None, None, :, None]
            out[tm, tn, r, c] = val
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_threadgroup_overflow(x: torch.Tensor, nb: int) -> torch.Tensor:
    """A root tile, a free axis and an inner tile, each budgeted on its own."""
    m, n = x.shape
    out = torch.zeros_like(x)
    out2 = torch.zeros_like(x)
    bn = hl.register_block_size(16, 16)
    for tm in hl.tile(m):
        for gn in hl.grid(nb):
            cols = gn * bn + hl.arange(bn)
            cv = cols < n
            for tk in hl.tile(n):
                hl.store(out, [tm, tk.index], hl.load(x, [tm, tk.index]) + 1)
            v = hl.load(x, [tm, cols], extra_mask=cv[None, :])
            hl.store(out2, [tm, cols], v * 2, extra_mask=cv[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_bound(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros_like(x)
    for t in hl.tile(n):
        idx = t.begin + hl.arange(t.block_size)
        out[idx] = x[idx] + idx.to(torch.float32)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_len(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.zeros_like(x)
    for g in hl.grid(n // 16):
        idx = g * 16 + hl.arange(16)
        out[idx] = x[idx] + 1
    return out


# ---------------------------------------------------------------------------
# Tests - hl.arange over a block size that no loop owns
# ---------------------------------------------------------------------------


def _index_rows(m: int) -> torch.Tensor:
    """A ``[m, 8]`` int32 tensor whose rows are a permutation of ``range(m)``."""
    perm = torch.randperm(m, device=DEVICE).to(torch.int32)
    return perm[:, None].expand(m, 8).contiguous()


@_requires_darwin
class TestMetalFreeArange(unittest.TestCase):
    """``base + hl.arange(BLOCK)`` over an ``hl.register_block_size`` block that
    no loop owns gets a thread axis of its own; layouts where a Metal thread's
    by-block values differ from Helion's positional ones are rejected."""

    def _run(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None = None,
    ) -> tuple[object, str]:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        code = bound.to_code(config)
        return bound.compile_config(config)(*args), code

    def _assert_rejected(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        message: str,
        config: helion.Config | None = None,
    ) -> None:
        bound = kernel.bind(args)
        if config is None:
            config = bound.config_spec.default_config()
        with self.assertRaisesRegex(exc.BackendUnsupported, message):
            bound.to_code(config)

    def test_gather_1d(self) -> None:
        """Non-dividing n = 1000.  The root grid's flattened strategy reads
        ``tid[0]`` at extent 1, so the free axis must take ``tid[1]``."""
        x = torch.randn(1000, device=DEVICE)
        expected = x * 2 + torch.arange(1000, device=DEVICE, dtype=torch.float32)
        for config in (
            None,
            helion.Config(block_sizes=[32]),
            helion.Config(block_sizes=[64]),
        ):
            with self.subTest(config=config):
                out, code = self._run(free1d, (x,), config)
                iota = [line for line in code.splitlines() if "iota =" in line]
                self.assertEqual(len(iota), 1)
                self.assertIn("tid[1]", iota[0])
                torch.testing.assert_close(out, expected)

    def test_dynamic_shapes(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        out, _ = self._run(free1d_dynamic, (x,), helion.Config(block_sizes=[64]))
        expected = x * 2 + torch.arange(1000, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(out, expected)

    @_full_threadgroup
    def test_rows_and_columns(self) -> None:
        x = torch.randn(70, 90, device=DEVICE)
        out, _ = self._run(free2d, (x, 5, 3))
        rows = torch.arange(70, device=DEVICE, dtype=torch.float32)[:, None]
        cols = torch.arange(90, device=DEVICE, dtype=torch.float32)[None, :]
        torch.testing.assert_close(out, x + rows * 1000 + cols)

    @_full_threadgroup
    def test_inner_tile_shares_the_budget(self) -> None:
        """The free rows reserve their threads before the inner tile takes
        its own: at the default config the tile gets 1024 / 32."""
        x = torch.randn(70, 90, device=DEVICE)
        rows = torch.arange(70, device=DEVICE, dtype=torch.float32)[:, None]
        expected = x * 3 + rows
        default = free_inner_tile.bind((x,)).config_spec.default_config()
        self.assertEqual(default.config["block_sizes"][0], 32)
        for config in (
            None,
            helion.Config(block_sizes=[16, 32]),
            helion.Config(block_sizes=[32, 16], num_threads=[0, 4]),
        ):
            with self.subTest(config=config):
                out, code = self._run(free_inner_tile, (x,), config)
                if config is None:
                    self.assertIn("_block_dims=(32, 32, 1)", code)
                torch.testing.assert_close(out, expected)

    def test_carry_through_unthreaded_grid(self) -> None:
        x = torch.randn(70, 90, device=DEVICE)
        weights = torch.arange(1, 91, device=DEVICE, dtype=torch.float32)
        for config in (None, helion.Config(block_sizes=[16])):
            with self.subTest(config=config):
                out, _ = self._run(free_carry, (x,), config)
                torch.testing.assert_close(
                    out, (x * weights).sum(1), rtol=1e-4, atol=1e-4
                )

    def test_beside_a_lane_looped_root_tile(self) -> None:
        x = torch.randn(70, 90, device=DEVICE)
        cols = torch.arange(90, device=DEVICE, dtype=torch.float32)[None, :]
        out, _ = self._run(
            free_beside_tile_root,
            (x, 6),
            helion.Config(block_sizes=[16, 4], num_threads=[0, 2]),
        )
        torch.testing.assert_close(out, x - cols)

    def test_dtype_and_start(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        out, _ = self._run(free_int64, (x,))
        expected = torch.arange(1000, device=DEVICE, dtype=torch.int64) * 3
        torch.testing.assert_close(out, expected)
        out, _ = self._run(free_strided, (x,))
        expected = torch.zeros(2000, device=DEVICE)
        expected[::2] = x
        torch.testing.assert_close(out, expected)

    def test_step(self) -> None:
        x = torch.randn(128, device=DEVICE)
        out, _ = self._run(free_step, (x,))
        expected = torch.zeros_like(x)
        expected[::2] = x[::2] + 1
        torch.testing.assert_close(out, expected)

    def test_negative_values_stay_signed(self) -> None:
        """``tid`` is unsigned in MSL; the iota is cast to its dtype, so
        ``idx - 5 >= 0`` is false for the first five."""
        x = torch.randn(64, device=DEVICE)
        out, _ = self._run(free_shifted, (x,))
        shifted = torch.arange(64, device=DEVICE, dtype=torch.int32) - 5
        torch.testing.assert_close(out, torch.where(shifted >= 0, shifted, -1))

    @_full_threadgroup
    def test_sibling_tiles(self) -> None:
        """Both sibling tiles threaded on ``tid[0]``; the free rows go above
        every axis a strategy may read."""
        x = torch.randn(70, 90, device=DEVICE)
        (out, out2), code = self._run(
            free_beside_sibling_tiles,
            (x,),
            helion.Config(block_sizes=[8, 32, 16], num_threads=[0, 32, 16]),
        )
        self.assertIn("_block_dims=(32, 1, 8)", code)
        torch.testing.assert_close(out, x + 1)
        torch.testing.assert_close(out2, x * 2)

    def test_third_axis(self) -> None:
        x = torch.randn(8, 8, 6, device=DEVICE)
        out, code = self._run(
            free_axis2,
            (x,),
            helion.Config(block_sizes=[4, 4, 4], flatten_loops=[False]),
        )
        self.assertIn("_block_dims=(4, 4, 4)", code)
        expected = torch.zeros_like(x)
        r = torch.arange(4, device=DEVICE, dtype=torch.float32)
        expected[:, :, :4] = x[:, :, :4] + r
        torch.testing.assert_close(out, expected)

    def test_second_root(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        (out, out2), _ = self._run(free_beside_second_root, (x, y, 32))
        torch.testing.assert_close(out, x * 2)
        torch.testing.assert_close(out2, y * 3 + 1)

    @_full_threadgroup
    def test_where_by_block(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        out, _ = self._run(where_ok, (x,))
        idx = torch.arange(1000, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(out, torch.where(x > 0, x, idx))
        x = torch.randn(16, 32, device=DEVICE)
        out, _ = self._run(where_2d_ok, (x,))
        cols = torch.arange(32, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(out, torch.where(x > 0, cols[None, :], x))

    def test_unused_rolled_copy_is_not_scanned(self) -> None:
        x = torch.randn(32, 24, device=DEVICE)
        out, _ = self._run(free_colon, (x,))
        torch.testing.assert_close(out, x * 2)

    def test_lower_rank_operands_at_triton_placement(self) -> None:
        x = torch.randn(8, 8, 8, device=DEVICE)
        y = torch.randn(8, 8, device=DEVICE)
        config = helion.Config(block_sizes=[8, 8, 4])
        out, _ = self._run(lowrank_3d_ok, (x, y), config)
        torch.testing.assert_close(out, x + y[:, None, :])
        out, _ = self._run(lowrank_middle_ok, (x,), config)
        cols = torch.arange(8, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(out, x + cols[None, :, None] * 100.0)
        out, _ = self._run(lowrank_static_ok, (x, y))
        torch.testing.assert_close(out, x + y[:, None, :])

    def test_broadcast_positions(self) -> None:
        """A non-block dim takes the first non-block result dim of its size,
        and an unmatched dim the last unused one."""
        bound = free1d.bind((torch.randn(64, device=DEVICE),))
        with bound.env:
            self.assertEqual(
                free_arange._broadcast_positions([8, 4], [8, 2, 4]), [0, 2]
            )
            self.assertEqual(free_arange._broadcast_positions([3], [8, 2]), [1])

    def test_broadcast_index_tensors_by_block(self) -> None:
        x = torch.randn(16, 8, device=DEVICE)
        ridx = _index_rows(16)
        r = ridx.long()
        cols = 7 - torch.arange(8, device=DEVICE)
        for config in (None, helion.Config(block_sizes=[8, 4])):
            with self.subTest(config=config):
                out, _ = self._run(idx_bcast_ok, (x, ridx), config)
                torch.testing.assert_close(out, x[r, cols[None, :].expand(16, 8)])
                out, _ = self._run(idx_bcast_2d_ok, (x, ridx), config)
                torch.testing.assert_close(out, x[r, 7 - r % 8])

    def test_cartesian_index_tensors_pass(self) -> None:
        """Only codegen: Metal has no lowering for a host tensor used in device
        code (``_host_tensor``)."""
        x = torch.randn(32, 24, device=DEVICE)
        cidx = torch.randperm(24, device=DEVICE)[:8]
        bound = idx_cartesian_host.bind((x, cidx))
        self.assertIn("iota", bound.to_code(bound.config_spec.default_config()))

    def test_dot_accumulates_on_one_thread(self) -> None:
        """grouped_gemm's layout with a float16 ``hl.dot`` acc: each thread
        sums its K lanes in registers, with no cross-thread combine."""
        a = torch.randn(64, 48, device=DEVICE, dtype=torch.float16) / 4
        b = torch.randn(48, 64, device=DEVICE, dtype=torch.float16) / 4
        for args, config in (
            ((a, b, 2, 2), helion.Config(block_sizes=[32, 32, 16], num_warps=32)),
            (
                (a, b, 4, 4),
                helion.Config(
                    block_sizes=[16, 16, 16], num_threads=[0, 0, 1], num_warps=32
                ),
            ),
        ):
            with self.subTest(config=config):
                out, code = self._run(dot16_free_host, args, config)
                self.assertIn("lane_sum", code)
                self.assertNotIn("simd_", code)
                self.assertNotIn("group_", code)
                torch.testing.assert_close(
                    out.float(),
                    _matmul_reference(a, b).to(DEVICE),
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_dot_lowers_through_mpp(self) -> None:
        """grouped_gemm's layout with a float32 ``hl.dot`` acc lowers to an
        MPP cooperative accumulator."""
        a = torch.randn(64, 48, device=DEVICE)
        b = torch.randn(48, 64, device=DEVICE)
        a50 = torch.randn(64, 50, device=DEVICE)
        b50 = torch.randn(50, 64, device=DEVICE)
        k1 = helion.Config(
            block_sizes=[16, 16, 16], num_threads=[0, 0, 1], num_warps=32
        )
        cols = torch.arange(64, dtype=torch.float32)[None, :]
        cases = [
            (
                dot_free_host,
                (a, b, 2, 2),
                helion.Config(block_sizes=[32, 32, 16], num_warps=32),
            ),
            (dot_free_host, (a, b, 4, 4), k1),
            (dot_free_tail, (a50, b50, 4, 4), k1),
            (
                dot_rows_rootlanes,
                (a50, b50, 4),
                helion.Config(
                    block_sizes=[16, 8, 16], num_threads=[0, 0, 1], num_warps=32
                ),
            ),
        ]
        for kernel, args, config in cases:
            with self.subTest(kernel=kernel.name, config=config):
                out, code = self._run(kernel, args, config)
                self.assertIn("_helion_mpp_run(", code)
                expected = _matmul_reference(args[0], args[1])
                if kernel is dot_free_tail:
                    expected = expected + cols
                torch.testing.assert_close(
                    out, expected.to(DEVICE), rtol=1e-4, atol=1e-4
                )

    def test_addmm_lowers_through_mpp(self) -> None:
        """grouped_gemm's addmm carry passes every guard and lowers to an MPP
        cooperative accumulator (``test_metal_mpp_matmul.py``)."""
        a = torch.randn(64, 48, device=DEVICE)
        b = torch.randn(48, 64, device=DEVICE)
        out, code = self._run(mm_free, (a, b), helion.Config(block_sizes=[32, 32, 16]))
        self.assertIn("_helion_mpp_acc(", code)
        torch.testing.assert_close(
            out, _matmul_reference(a, b).to(DEVICE), rtol=1e-4, atol=1e-4
        )

    def test_reduction_rejects(self) -> None:
        self._assert_rejected(
            free_reduce, (torch.randn(64, 64, device=DEVICE),), "has a reduction"
        )

    def test_mpp_region_rejects(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        self._assert_rejected(
            mpp_free,
            (x, x),
            "free block 0 is spread over threads around a cooperative MPP matmul",
        )

    def test_store_and_mask_rebinds_reject(self) -> None:
        x = torch.randn(64, device=DEVICE)
        # A free block's value stored through a tile: metal/literal_dims.py.
        store = r"a store writes dim 0 of .* held by block \d+'s tile, through block"
        self._assert_rejected(free_store_into_tile, (x,), store)
        self._assert_rejected(
            free_store_into_tile_in_branch, (x, torch.ones(1, device=DEVICE)), store
        )
        self._assert_rejected(
            free_mask_rebind, (x,), r"a load mask binds a dim of block \d+ to"
        )
        self._assert_rejected(
            store_mask_free_dim,
            (torch.randn(100, device=DEVICE), torch.zeros(4, device=DEVICE)),
            "a store mask has more dims than the value it meets",
        )

    def test_index_tensor_rebinds_reject(self) -> None:
        x = torch.randn(16, 8, device=DEVICE)
        ridx = _index_rows(16)
        config = helion.Config(block_sizes=[8, 8, 8])
        for kernel, what in (
            (idx_bcast_load, "load"),
            (idx_bcast_store, "store"),
            (idx_bcast_col, "load"),
        ):
            with self.subTest(kernel=kernel.name):
                self._assert_rejected(
                    kernel,
                    (x, ridx),
                    rf"a {what}'s index tensor binds a dim of block \d+ to",
                    config,
                )

    def test_where_rebinds_reject(self) -> None:
        self._assert_rejected(
            where_other_free_block,
            (torch.randn(64, device=DEVICE),),
            _PAIRS_TILES,
        )
        self._assert_rejected(
            where_lower_rank, (torch.randn(16, 16, device=DEVICE),), _PAIRS_TILES
        )

    def test_transposed_operands_reject(self) -> None:
        # Operands over the same free blocks, one swapped: paired by position.
        args = (torch.randn(8, 8, device=DEVICE), torch.randn(8, 8, device=DEVICE))
        for kernel in (add_transposed_free, where_transposed_free):
            with self.subTest(kernel=kernel.name):
                self._assert_rejected(kernel, args, _PAIRS_TILES)

    def test_lower_rank_rebinds_reject(self) -> None:
        x = torch.randn(8, 8, 8, device=DEVICE)
        y = torch.randn(8, 8, device=DEVICE)
        # Each meets another free block's dim at a position.
        self._assert_rejected(
            lowrank_onto_other_free_dim,
            (torch.randn(8, 8, device=DEVICE),),
            _PAIRS_TILES,
            helion.Config(block_sizes=[8, 8, 4]),
        )
        config = helion.Config(block_sizes=[8, 8, 8])
        self._assert_rejected(lowrank_swapped_dims, (x, y), _PAIRS_TILES, config)
        self._assert_rejected(lowrank_permuted_free, (x, y), _PAIRS_TILES, config)

    def test_carry_rebinds_reject(self) -> None:
        x = torch.randn(16, 16, device=DEVICE)
        message = r"the carry \w+ binds a dim of block"
        self._assert_rejected(phi_loop_rebind, (x,), message)
        self._assert_rejected(carry_read_only_in_loop, (x,), message)
        self._assert_rejected(phi_if_rebind, (x, torch.ones(1, device=DEVICE)), message)
        self._assert_rejected(
            mm_acc_transposed,
            (torch.randn(16, 32, device=DEVICE), torch.randn(32, 16, device=DEVICE)),
            message,
            helion.Config(block_sizes=[16, 16, 16], num_threads=[0, 0, 1]),
        )

    def test_matmul_k_on_another_block_rejects(self) -> None:
        a = torch.randn(64, 48, device=DEVICE)
        b = torch.randn(48, 64, device=DEVICE)
        self._assert_rejected(
            mm_k_free_acc, (a, b, 3), "is not the block of the device loop"
        )

    @_full_threadgroup
    def test_matmul_k_on_threads_rejects(self) -> None:
        a = torch.randn(64, 48, device=DEVICE)
        b = torch.randn(48, 64, device=DEVICE)
        for config in (
            helion.Config(block_sizes=[16, 16, 16]),
            helion.Config(block_sizes=[16, 16, 4], num_threads=[0, 0, 4]),
        ):
            with self.subTest(config=config):
                self._assert_rejected(
                    mm_free, (a, b), r"K dim \(block \d+\) spans \d+ threads", config
                )

    def test_non_accumulating_matmuls_reject(self) -> None:
        a = torch.randn(64, 48, device=DEVICE)
        b = torch.randn(48, 64, device=DEVICE)
        config = helion.Config(block_sizes=[32, 32, 16])
        for kernel in (mm_scaled_bias, mm_overwrite, mm_swap):
            with self.subTest(kernel=kernel.name):
                self._assert_rejected(
                    kernel, (a, b), "does not accumulate a carry of its K loop", config
                )

    def test_lane_folded_carry_rejects(self) -> None:
        self._assert_rejected(
            free_lane_carry,
            (torch.randn(70, 90, device=DEVICE),),
            "through a lane loop",
            helion.Config(block_sizes=[32, 16], num_threads=[0, 2]),
        )

    def test_stored_tensor_with_other_accesses_rejects(self) -> None:
        message = "a stored tensor has other accesses"
        self._assert_rejected(
            second_root_in_place,
            (torch.randn(1024, device=DEVICE), torch.zeros(4096, device=DEVICE), 32),
            message,
        )
        self._assert_rejected(
            lane_reverse,
            (torch.randn(40, 48, device=DEVICE),),
            message,
            helion.Config(block_sizes=[32, 16], num_threads=[0, 2]),
        )

    @_full_threadgroup
    def test_fourth_axis_rejects(self) -> None:
        self._assert_rejected(
            free_axis3,
            (torch.randn(8, 8, 4, 4, device=DEVICE),),
            "no thread axis left for block",
        )

    def test_oversized_threadgroup_rejects(self) -> None:
        """Each loop is budgeted on its own, so explicit counts that fit one
        by one can still overflow together."""
        self._assert_rejected(
            free_threadgroup_overflow,
            (torch.randn(70, 90, device=DEVICE), 6),
            r"a threadgroup of \[\d+, \d+, \d+\] threads exceeds 1024",
            helion.Config(
                block_sizes=[16, 64, 64], num_threads=[0, 64, 64], num_warps=32
            ),
        )

    def test_free_axis_beside_a_root_and_an_inner_tile_fits_the_budget(self) -> None:
        """At the default the root tile shrinks beside the free axis, which
        the inner tile shares, to 8 x 16 threads."""
        x = torch.randn(70, 90, device=DEVICE)
        out, code = self._run(free_threadgroup_overflow, (x, 6))
        self.assertIn("_block_dims=(8, 16, 1)", code)
        torch.testing.assert_close(out, x + 1)

    def test_other_lengths_reject(self) -> None:
        message = "Metal lowers only a range over a block size that no loop owns"
        self._assert_rejected(tile_bound, (torch.randn(1000, device=DEVICE),), message)
        self._assert_rejected(literal_len, (torch.randn(64, device=DEVICE),), message)


@helion.kernel(backend="metal", static_shapes=True)
def sibling_mean_sub(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_registered_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    b1 = hl.register_block_size(512, 1024)
    b2 = hl.register_block_size(512, 1024)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=b1):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n, block_size=b2):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_registered(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    b1 = hl.register_block_size(16, 256)
    b2 = hl.register_block_size(16, 256)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=b1):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n, block_size=b2):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_registered_unequal_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    b1 = hl.register_block_size(32, 256)
    b2 = hl.register_block_size(32, 1024)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=b1):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n, block_size=b2):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_fixed_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=64):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n, block_size=64):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_grid_root(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for i in hl.grid(m):
        acc = hl.zeros([], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[i, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n):
            out[i, tn] = (x[i, tn].to(torch.float32) - mean).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_root_reduction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        total = y[tm, :].to(torch.float32).sum(-1)
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        for tn in hl.tile(n):
            shift = total[:, None] - acc[:, None]
            out[tm, tn] = (x[tm, tn].to(torch.float32) + shift).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_matmul_passes(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    n = w1.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        a1 = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            a1 = torch.addmm(a1, x[tm, tk], w1[tk, :])
        a2 = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            a2 = torch.addmm(a2, x[tm, tk], w2[tk, :])
        out[tm, :] = a1 * a2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_single_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            out[tm, tn] = x[tm, tn] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_unequal_passes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    k = y.size(1)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + y[tm, tk].to(torch.float32).sum(-1)
        for tn in hl.tile(n):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - acc[:, None]).to(x.dtype)
    return out


def _welford_args(
    m: int, n: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(n, device=DEVICE, dtype=dtype),
        torch.randn(n, device=DEVICE, dtype=dtype),
        torch.randn(m, n, device=DEVICE, dtype=dtype),
    )


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_logsumexp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            v = x[tm, tn].to(torch.float32)
            mn = torch.maximum(mi, torch.amax(v, dim=1))
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)
            mi = mn
        out[tm] = mi + torch.log(di)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_shared_mean_sub(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    bn = hl.register_block_size(n)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_max_abs_scale(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        amax = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            v = x[tm, tn].abs().to(torch.float32)
            amax = torch.maximum(amax, torch.amax(v, dim=1))
        ss = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            v = x[tm, tn].to(torch.float32) / amax[:, None]
            ss = ss + (v * v).sum(dim=1)
        for tn in hl.tile(n):
            scale = amax[:, None] * torch.sqrt(ss[:, None] / n)
            out[tm, tn] = (x[tm, tn].to(torch.float32) / scale).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_int_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.int64)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].to(torch.int64).sum(-1)
        out[tm] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_narrow_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    bn = hl.register_block_size(16, 64)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - acc[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_sq_err_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, bn], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            d = x[tm, tn].to(torch.float32) - y[tm, tn].to(torch.float32)
            acc += d * d
        out[tm] = acc.sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_positive_count(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.int32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, bn], dtype=torch.int32)
        for tn in hl.tile(n, block_size=bn):
            acc += (x[tm, tn] > 0).to(torch.int32)
        out[tm] = acc.sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_centered_sq_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, bn], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            acc += x[tm, tn]
        mean = acc.sum(-1) / n
        acc2 = hl.zeros([tm, bn], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            # Masked tail lanes load 0, so they add 0 to x * (x - mean).
            acc2 += x[tm, tn] * (x[tm, tn] - mean[:, None])
        out[tm] = acc2.sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_inner_dim_sum(x: torch.Tensor) -> torch.Tensor:
    m, n, _ = x.size()
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, bn], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            acc += x[tm, tn, :].sum(-1)
        out[tm] = acc.sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_inner_dim_sum_in_loop(x: torch.Tensor) -> torch.Tensor:
    m, n, _ = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn, :].sum(-1).sum(-1)
        out[tm] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_sum_beside_wide_inner_dim(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    inner = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
            inner[tm, tn] = y[tm, tn, :].sum(-1)
        out[tm] = acc
    return out, inner


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_distinct_pass_blocks(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    bn1 = hl.register_block_size(n)
    bn2 = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, bn1], dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn1):
            acc += x[tm, tn]
        mean = acc.sum(-1) / n
        for tn in hl.tile(n, block_size=bn2):
            out[tm, tn] = x[tm, tn] - mean[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_pass_mixed_width_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        sx = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            sx = sx + x[tm, tn].to(torch.float32).sum(-1)
        sy = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            sy = sy + y[tm, tn].sum(-1)
        out[tm] = sx * sy
    return out


@helion.kernel(backend="metal", static_shapes=True)
def wide_row_block_mean_sub(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    bm = hl.register_block_size(16, 64)
    for tm in hl.tile(m, block_size=bm):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        mean = acc / n
        for tn in hl.tile(n):
            out[tm, tn] = (x[tm, tn].to(torch.float32) - mean[:, None]).to(x.dtype)
    return out


row_pass_softmax = helion.kernel(
    softmax_two_pass.fn, backend="metal", static_shapes=True, fast_math=True
)
row_pass_kl_div = helion.kernel(kl_div_forward.fn, backend="metal", static_shapes=True)
row_pass_jsd = helion.kernel(jsd_forward.fn, backend="metal", static_shapes=True)


def _row_pass_input(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(m, n, generator=gen).to(dtype)


def _shapes(seeds: list[helion.Config]) -> list[tuple[list[int], list[int]]]:
    """Each seed's ``(block_sizes, num_threads)``."""
    return [(s.config["block_sizes"], s.config["num_threads"]) for s in seeds]


def _heuristic_seeds(bound: helion.runtime.kernel.BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        return MetalRowReductionHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def _searched_seeds(
    fn: Callable[..., object], args: tuple[torch.Tensor, ...], **settings: object
) -> tuple[helion.runtime.kernel.BoundKernel, list[helion.Config]]:
    """Autotune ``fn`` at ``args`` and return the compiler seeds the search
    started from."""
    searched: list[list[helion.Config]] = []

    class RecordSeeds:
        def __init__(self, bound: helion.runtime.kernel.BoundKernel) -> None:
            self.bound = bound

        def autotune(self, *, skip_cache: bool = False) -> helion.Config:
            spec = self.bound.config_spec
            searched.append(list(spec.compiler_seed_configs))
            return spec.default_config()

    kernel = helion.kernel(
        fn,
        backend="metal",
        static_shapes=True,
        autotune_effort="quick",
        autotuner_fn=lambda bound, args, **kwargs: RecordSeeds(bound),
        **settings,
    )
    bound = kernel.bind(args)
    bound.autotune(args)
    (seeds,) = searched
    return bound, seeds


@_requires_darwin
class TestMetalRowReductionHeuristic(unittest.TestCase):
    """The autotuner heuristic that seeds a row grid's passes from its
    structure."""

    def _bound_seeds(
        self, kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
    ) -> tuple[helion.runtime.kernel.BoundKernel, list[helion.Config]]:
        """Bind ``kernel`` and return the heuristic's seeds, checking that the
        registry planted exactly those at bind."""
        bound = kernel.bind(args)
        seeds = _heuristic_seeds(bound)
        self.assertLessEqual(len(seeds), 2)
        spec = bound.config_spec
        self.assertEqual(spec.compiler_seed_configs, seeds)
        self.assertEqual(
            spec.autotuner_heuristics,
            [MetalRowReductionHeuristic.name] if seeds else [],
        )
        return bound, seeds

    def _seeds(
        self,
        kernel: helion.Kernel,
        *shapes: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[list[int], list[int]]]:
        args = tuple(torch.empty(*s, device=DEVICE, dtype=dtype) for s in shapes)
        _, seeds = self._bound_seeds(kernel, args)
        return _shapes(seeds)

    def test_bind_plants_the_seeds_through_the_registry(self) -> None:
        kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
        bound, seeds = self._bound_seeds(
            kernel, _welford_args(40, 1024, torch.bfloat16)
        )
        self.assertEqual(
            _shapes(seeds),
            [([4, 256, 256], [0, 32, 32]), ([1, 1024, 1024], [0, 128, 128])],
        )
        self.assertEqual(
            bound.config_spec.autotuner_heuristics, ["metal_row_reduction"]
        )
        # Regenerating the seeds for the same bound kernel gives the same list.
        assert bound.host_function is not None
        with bound.env:
            again = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        self.assertEqual(again, seeds)

    def test_autotune_starts_from_the_seeds(self) -> None:
        for fn, args in (
            (softmax_two_pass.fn, (torch.empty(37, 300, device=DEVICE),)),
            (welford.fn, _welford_args(40, 1024, torch.bfloat16)),
        ):
            with self.subTest(kernel=fn.__name__):
                bound, searched = _searched_seeds(fn, args)
                seeds = _heuristic_seeds(bound)
                self.assertEqual(len(seeds), 2)
                self.assertEqual(searched, seeds)

    def test_lanes_are_one_16_byte_load(self) -> None:
        self.assertEqual(
            self._seeds(row_pass_softmax, (37, 300)),
            [([4, 128], [0, 32]), ([1, 512], [0, 128])],
        )
        self.assertEqual(
            self._seeds(row_pass_softmax, (33, 1000), dtype=torch.float16),
            [([4, 256], [0, 32]), ([1, 1024], [0, 128])],
        )
        self.assertEqual(
            self._seeds(row_pass_int_sum, (37, 300), dtype=torch.int64),
            [([4, 64], [0, 32]), ([1, 256], [0, 128])],
        )
        self.assertEqual(
            self._seeds(row_pass_int_sum, (37, 3000), dtype=torch.int8),
            [([4, 512], [0, 32]), ([1, 2048], [0, 128])],
        )
        # Passes reducing bf16 and fp32 rows take the fp32 width, so no load
        # exceeds 16 bytes.
        _, mixed = self._bound_seeds(
            row_pass_mixed_width_sums,
            (
                torch.empty(37, 300, device=DEVICE, dtype=torch.bfloat16),
                torch.empty(37, 300, device=DEVICE),
            ),
        )
        self.assertEqual(
            _shapes(mixed),
            [([4, 128, 128], [0, 32, 32]), ([1, 512, 512], [0, 128, 128])],
        )
        # The fp32 load reduced over the inner dimension is not a pass's.
        _, beside = self._bound_seeds(
            row_pass_sum_beside_wide_inner_dim,
            (
                torch.empty(37, 300, device=DEVICE, dtype=torch.float16),
                torch.empty(37, 300, 8, device=DEVICE),
            ),
        )
        self.assertEqual(
            _shapes(beside),
            [([4, 256], [0, 32]), ([1, 512], [0, 128])],
        )
        # With no reduction to read a load width from, a pass loads 4 bytes.
        self.assertEqual(
            self._seeds(sibling_single_pass, (37, 300), dtype=torch.float16),
            [([4, 128], [0, 32]), ([1, 512], [0, 128])],
        )

    def test_seeds_follow_block_id_order(self) -> None:
        # The registered pass block gets its id before the row's tile.
        self.assertEqual(
            self._seeds(row_pass_shared_mean_sub, (37, 300)),
            [([128, 4], [32, 0]), ([512, 1], [128, 0])],
        )

    def test_grid_floor_bounds_the_row_block(self) -> None:
        self.assertEqual(
            self._seeds(row_pass_softmax, (65573, 300), dtype=torch.float16),
            [([8, 256], [0, 32]), ([8, 512], [0, 128])],
        )

    def test_one_simd_group_seed_fills_four_simd_groups(self) -> None:
        self.assertEqual(
            self._seeds(row_pass_softmax, (37, 300), dtype=torch.float16),
            [([4, 256], [0, 32]), ([1, 512], [0, 128])],
        )
        # The row block's maximum is the next power of two of the rows.
        self.assertEqual(
            self._seeds(row_pass_softmax, (2, 300), dtype=torch.float16),
            [([2, 256], [0, 32]), ([1, 512], [0, 128])],
        )
        self.assertEqual(
            self._seeds(row_pass_softmax, (3, 300), dtype=torch.float16),
            [([4, 256], [0, 32]), ([1, 512], [0, 128])],
        )
        # A reduction dimension of 16 leaves 64 threads, two SIMD groups.
        self.assertEqual(
            self._seeds(row_pass_inner_dim_sum_in_loop, (37, 300, 16)),
            [([2, 128], [0, 32])],
        )
        self.assertEqual(
            self._seeds(row_pass_inner_dim_sum_in_loop, (37, 300, 32)),
            [([1, 128], [0, 32])],
        )

    def test_rows_beyond_the_budget_run_serially(self) -> None:
        # 16 rows of 128 threads overflow the threadgroup, so the
        # four-SIMD-group seed walks its rows on one thread.
        self.assertEqual(
            self._seeds(wide_row_block_mean_sub, (70, 1024), dtype=torch.bfloat16),
            [([16, 256, 256], [0, 32, 32]), ([16, 1024, 1024], [1, 128, 128])],
        )

    def test_short_passes_clamp_the_width(self) -> None:
        # The pass fragment's maximum is the row's next power of two.  At 128
        # the four-SIMD-group seed would leave each thread one lane.
        self.assertEqual(
            self._seeds(row_pass_softmax, (37, 100)), [([4, 128], [0, 32])]
        )
        self.assertEqual(self._seeds(row_pass_softmax, (37, 40)), [([4, 64], [0, 32])])
        self.assertEqual(
            self._seeds(row_pass_softmax, (1, 129), dtype=torch.float16),
            [([1, 256], [0, 32]), ([1, 256], [0, 128])],
        )
        self.assertEqual(
            self._seeds(row_pass_narrow_pass, (37, 1024)), [([64, 4], [32, 0])]
        )
        self.assertEqual(
            self._seeds(sibling_registered, (37, 1024)),
            [([128, 128, 4], [32, 32, 0]), ([256, 256, 1], [128, 128, 0])],
        )
        # Both passes take the narrower maximum.
        self.assertEqual(
            self._seeds(sibling_registered_unequal_max, (37, 1024)),
            [([128, 128, 4], [32, 32, 0]), ([256, 256, 1], [128, 128, 0])],
        )

    def test_seeds_name_their_launch(self) -> None:
        """Each seed's num_warps gives its own threadgroup, so a seed's layout
        doesn't depend on the default num_warps."""
        for fn, inputs, shape, launches in (
            (softmax_two_pass.fn, 1, (37, 300), [(32, 4, 1), (128, 1, 1)]),
            (kl_div_forward.fn, 2, (37, 300), [(512, 1, 1)]),
            (kl_div_forward.fn, 2, (8, 4096), [(1024, 1, 1)]),
        ):
            args = tuple(torch.randn(*shape, device=DEVICE) for _ in range(inputs))
            kernel = helion.kernel(fn, backend="metal", static_shapes=True)
            bound, seeds = self._bound_seeds(kernel, args)
            with self.subTest(kernel=fn.__name__, shape=shape):
                self.assertEqual(
                    [_launch(bound.to_code(seed)) for seed in seeds], launches
                )
                self.assertEqual(
                    [seed.config["num_warps"] for seed in seeds],
                    [math.prod(launch) // 32 for launch in launches],
                )

    def test_long_rows_keep_one_vector_per_thread(self) -> None:
        self.assertEqual(
            self._seeds(sibling_mean_sub, (4, 32768)),
            [([4, 128, 128], [0, 32, 32]), ([1, 512, 512], [0, 128, 128])],
        )

    def test_reduction_after_its_loop_gets_one_element_per_thread_seed(
        self,
    ) -> None:
        self.assertEqual(
            self._seeds(row_pass_kl_div, (33, 1000), (33, 1000)),
            [([1024, 1], [1024, 0])],
        )
        # The pass fragment's maximum is the row's next power of two, 512.
        self.assertEqual(
            self._seeds(row_pass_jsd, (37, 300), (37, 300)), [([512, 1], [512, 0])]
        )
        # The grid floor puts 8 rows in a threadgroup, 128 threads each.
        self.assertEqual(
            self._seeds(row_pass_jsd, (65573, 300), (65573, 300)),
            [([128, 8], [128, 0])],
        )
        # The lane seeds fail codegen.
        bound = row_pass_kl_div.bind(
            tuple(torch.empty(33, 1000, device=DEVICE) for _ in range(2))
        )
        with patch.object(
            row_reduction_heuristic, "_reduces_pass_after_loop", return_value=False
        ):
            seeds = _heuristic_seeds(bound)
        self.assertTrue(seeds)
        for seed in seeds:
            with (
                self.subTest(config=seed),
                self.assertRaisesRegex(exc.BackendUnsupported, "after its loop"),
            ):
                bound.to_code(seed)

    def _assert_seeds_match(
        self,
        kernel: helion.Kernel,
        inputs: tuple[torch.Tensor, ...],
        expected: torch.Tensor | tuple[torch.Tensor, ...],
        tol: float,
    ) -> None:
        args = tuple(t.to(DEVICE) for t in inputs)
        bound, seeds = self._bound_seeds(kernel, args)
        self.assertTrue(seeds)
        x = inputs[-1]
        for seed in seeds:
            with self.subTest(
                kernel=kernel.name, shape=tuple(x.shape), dtype=x.dtype, config=seed
            ):
                torch.testing.assert_close(
                    bound.compile_config(seed)(*args),
                    expected,
                    atol=tol,
                    rtol=tol,
                    equal_nan=True,
                    check_device=False,
                )

    def test_softmax_seeds_match_torch(self) -> None:
        # Every shape has a tail in its last pass tile.
        for m, n, dtype, tol in (
            (37, 300, torch.float32, 1e-4),
            (33, 1000, torch.float16, 2e-3),
            (5, 600, torch.bfloat16, 2e-2),
            (37, 40, torch.float32, 1e-4),
            (2, 300, torch.float16, 2e-3),
            (3, 300, torch.float16, 2e-3),
            (5, 1001, torch.float16, 2e-3),
        ):
            x = _row_pass_input(m, n, dtype)
            # A row of -inf is NaN, as in torch.
            x[1] = float("-inf")
            expected = torch.softmax(x.double(), dim=1).to(dtype)
            self._assert_seeds_match(row_pass_softmax, (x,), expected, tol)

    def test_reduction_seeds_match_torch(self) -> None:
        x = _row_pass_input(37, 300, torch.float32)
        x64 = x.double()
        amax = x64.abs().amax(-1, keepdim=True)
        rms = ((x64 / amax) ** 2).mean(-1, keepdim=True).sqrt()
        for kernel, expected in (
            (row_pass_logsumexp, torch.logsumexp(x64, dim=1)),
            (row_pass_max_abs_scale, x64 / (amax * rms)),
            (row_pass_shared_mean_sub, x64 - x64.mean(-1, keepdim=True)),
            (sibling_registered_min, x64 - x64.mean(-1, keepdim=True)),
        ):
            self._assert_seeds_match(kernel, (x,), expected.float(), 1e-4)
        # 65573 = 8 * 8196 + 5 rows: the row block of 8 has a tail.
        for kernel, (m, n) in (
            (sibling_mean_sub, (37, 300)),
            (sibling_mean_sub, (65573, 64)),
            (sibling_mean_sub, (4, 32768)),
            (sibling_registered, (37, 1024)),
            (sibling_registered_unequal_max, (37, 1024)),
        ):
            y = _row_pass_input(m, n, torch.float32)
            y64 = y.double()
            expected = (y64 - y64.mean(-1, keepdim=True)).float()
            self._assert_seeds_match(kernel, (y,), expected, 1e-4)
        z = _row_pass_input(37, 1024, torch.float32)
        z64 = z.double()
        expected = (z64 - z64.sum(-1, keepdim=True)).float()
        self._assert_seeds_match(row_pass_narrow_pass, (z,), expected, 1e-4)
        self._assert_seeds_match(sibling_single_pass, (z,), z * 2, 0)
        xb = _row_pass_input(37, 300, torch.bfloat16)
        sums = xb.double().sum(-1) * x64.sum(-1)
        self._assert_seeds_match(row_pass_mixed_width_sums, (xb, x), sums.float(), 1e-4)
        w = _row_pass_input(70, 1024, torch.bfloat16)
        w64 = w.double()
        expected = (w64 - w64.mean(-1, keepdim=True)).bfloat16()
        self._assert_seeds_match(wide_row_block_mean_sub, (w,), expected, 2e-2)
        gen = torch.Generator().manual_seed(0)
        for k in (8, 16):
            v = torch.randn(37, 301, k, generator=gen)
            expected = v.double().sum((1, 2)).float()
            self._assert_seeds_match(
                row_pass_inner_dim_sum_in_loop, (v,), expected, 1e-4
            )
        u = torch.randn(37, 300, 8, generator=gen)
        self._assert_seeds_match(
            row_pass_sum_beside_wide_inner_dim,
            (xb.half(), u),
            (xb.half().double().sum(-1).float(), u.double().sum(-1).float()),
            1e-4,
        )

    def test_int_sum_seeds_are_exact(self) -> None:
        gen = torch.Generator().manual_seed(0)
        x = torch.randint(-1000, 1000, (37, 3000), generator=gen, dtype=torch.int32)
        self._assert_seeds_match(row_pass_int_sum, (x,), x.long().sum(-1), 0)
        y = torch.randint(-(2**40), 2**40, (37, 300), generator=gen)
        self._assert_seeds_match(row_pass_int_sum, (y,), y.sum(-1), 0)
        z = torch.randn(37, 1000, generator=gen)
        counted = (z > 0).sum(-1, dtype=torch.int32)
        self._assert_seeds_match(row_pass_positive_count, (z,), counted, 0)

    def test_welford_seeds_match_torch(self) -> None:
        for m, n, dtype, tol in (
            (40, 1024, torch.bfloat16, 2e-2),
            (33, 300, torch.float32, 1e-4),
        ):
            weight, bias, x = _welford_args(m, n, dtype)
            weight64, bias64, x64 = (t.cpu().double() for t in (weight, bias, x))
            expected = torch.nn.functional.layer_norm(x64, [n], weight64, bias64, 1e-5)
            kernel = helion.kernel(welford.fn, backend="metal", static_shapes=True)
            self._assert_seeds_match(kernel, (weight, bias, x), expected.to(dtype), tol)

    def test_after_loop_seeds_match_torch(self) -> None:
        gen = torch.Generator().manual_seed(0)
        lq, lp = torch.randn(2, 37, 300, generator=gen).double().log_softmax(-1)
        q, p = lq.exp(), lp.exp()
        log_m = (0.5 * (p + q)).log()
        kl_q_m = 0.5 * q * (lq - log_m)
        jsd_rows = (0.5 * p * (lp - log_m) + kl_q_m).sum(-1) / 37
        target = torch.randn(33, 1000, generator=gen).double().softmax(-1)
        pred = torch.randn(33, 1000, generator=gen).double().log_softmax(-1)
        kl_div = (target * (target.log() - pred)).sum() / 33
        x, y = torch.randn(2, 37, 300, generator=gen).bfloat16()
        sq_err = (x.double() - y.double()).pow(2).sum(-1)
        z = torch.randn(37, 300, generator=gen)
        centered = (z.double() - z.double().mean(-1, keepdim=True)).pow(2).sum(-1)
        # A size-1 reduction dimension reserves one thread.
        inner = torch.randn(9, 300, 1, generator=gen)
        for kernel, inputs, expected in (
            (
                row_pass_jsd,
                (lq.float(), lp.float()),
                (jsd_rows.sum().float(), (kl_q_m.sum(-1) / 37).float()),
            ),
            (row_pass_kl_div, (pred.float(), target.float()), kl_div.float()),
            (row_pass_sq_err_sum, (x, y), sq_err.float()),
            (row_pass_centered_sq_sum, (z,), centered.float()),
            (row_pass_inner_dim_sum, (inner,), inner.double().sum((1, 2)).float()),
        ):
            self._assert_seeds_match(kernel, inputs, expected, 1e-4)

    def test_matmul_passes_get_no_seeds(self) -> None:
        self.assertEqual(
            self._seeds(sibling_matmul_passes, (32, 64), (64, 16), (64, 16)), []
        )
        self.assertEqual(
            self._seeds(
                sibling_matmul_passes,
                (32, 1024),
                (1024, 2),
                (1024, 2),
                dtype=torch.float16,
            ),
            [],
        )

    def test_rolled_reduction_gets_no_seeds(self) -> None:
        self.assertEqual(self._seeds(sibling_root_reduction, (40, 300), (40, 2048)), [])
        self.assertEqual(self._seeds(sibling_root_reduction, (40, 300), (40, 8)), [])

    def test_non_row_loops_get_no_seeds(self) -> None:
        self.assertEqual(self._seeds(sibling_unequal_passes, (37, 300), (37, 200)), [])

    def test_blocks_without_a_spec_get_no_seeds(self) -> None:
        self.assertEqual(self._seeds(sibling_fixed_pass, (37, 300)), [])
        self.assertEqual(self._seeds(sibling_grid_root, (37, 300)), [])

    def test_pass_floor_bounds_the_seeds(self) -> None:
        # One SIMD group's 128-wide pass is below the registered floor of 512.
        self.assertEqual(
            self._seeds(sibling_registered_min, (37, 300)),
            [([512, 512, 1], [128, 128, 0])],
        )

    def test_reserved_reduction_threads_bound_the_seeds(self) -> None:
        # The persistent row reserves the whole threadgroup, so a pass would
        # be left one thread.
        kernel = helion.kernel(rms_norm_bwd.fn, backend="metal", static_shapes=True)
        self.assertEqual(
            self._seeds(kernel, (2048, 1024), (2048, 1024), (1024,), (2048, 1)), []
        )

    def test_reduction_dimension_beside_an_after_loop_reduction_gets_no_seed(
        self,
    ) -> None:
        # The row is short enough that the seed's 128 threads fit beside the
        # 8 the reduction dimension reserves.
        self.assertEqual(self._seeds(row_pass_inner_dim_sum, (9, 100, 8)), [])
        # With the reservation seen as one thread, the seed fails codegen.
        bound = row_pass_inner_dim_sum.bind((torch.empty(9, 100, 8, device=DEVICE),))
        with patch.object(MetalBackend, "_reserved_reduction_threads", return_value=1):
            seeds = _heuristic_seeds(bound)
        self.assertTrue(seeds)
        for seed in seeds:
            # At a full threadgroup the seed's 128 row threads stay threads
            # beside the reduction dimension's 8, a layout codegen rejects.
            full = helion.Config(**{**seed.config, "num_warps": 32})
            with (
                self.subTest(config=seed),
                self.assertRaisesRegex(
                    exc.BackendUnsupported, "no layout plan fixes the thread axis"
                ),
            ):
                bound.to_code(full)

    def test_reduction_after_its_loop_beside_a_distinct_pass_block_gets_no_seed(
        self,
    ) -> None:
        self.assertEqual(self._seeds(row_pass_distinct_pass_blocks, (37, 300)), [])

    def test_disabled_heuristics_plant_nothing(self) -> None:
        bound, searched = _searched_seeds(
            welford.fn,
            _welford_args(40, 1024, torch.bfloat16),
            disable_autotuner_heuristics=True,
        )
        self.assertEqual(searched, [])
        self.assertEqual(bound.config_spec.autotuner_heuristics, [])

    def test_autotune_off_default_is_unchanged(self) -> None:
        for fn, args in (
            (softmax_two_pass.fn, (torch.empty(37, 300, device=DEVICE),)),
            (welford.fn, _welford_args(40, 1024, torch.bfloat16)),
            (
                jsd_forward.fn,
                tuple(torch.empty(37, 300, device=DEVICE) for _ in range(2)),
            ),
        ):
            with self.subTest(kernel=fn.__name__):
                seeded = helion.kernel(fn, backend="metal", static_shapes=True)
                unseeded = helion.kernel(
                    fn,
                    backend="metal",
                    static_shapes=True,
                    disable_autotuner_heuristics=True,
                )
                bound, plain = seeded.bind(args), unseeded.bind(args)
                self.assertTrue(bound.config_spec.compiler_seed_configs)
                default = bound.config_spec.default_config()
                plain_default = plain.config_spec.default_config()
                self.assertEqual(default, plain_default)
                self.assertEqual(bound.to_code(default), plain.to_code(plain_default))


@helion.kernel(backend="metal", fast_math=True, configs=_DEFAULT_CONFIG)
def fast_exp_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.exp(x[tile]) / y[tile]
    return out


precise_exp_div = helion.kernel(
    fast_exp_div.fn, backend="metal", fast_math=False, configs=_DEFAULT_CONFIG
)


@helion.kernel(backend="metal", fast_math=True, configs=_DEFAULT_CONFIG)
def fast_int_true_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


@helion.kernel(backend="metal", fast_math=True, configs=_DEFAULT_CONFIG)
def fast_trunc_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="trunc")
    return out


def _exp_div_args(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``exp(x) / y`` operands with ``-inf``, NaN and 0 in ``x`` and inf in ``y``."""
    gen = torch.Generator().manual_seed(0)
    # fp16 overflows above exp(11.09), and the kernel rounds exp(x) to the
    # operand dtype before dividing, as torch does.
    x = (4 * torch.randn(300, generator=gen)).clamp(-10, 10)
    x[:3] = torch.tensor([float("-inf"), float("nan"), 0.0])
    y = torch.rand(300, generator=gen) + 0.5
    y[3] = float("inf")
    return x.to(dtype), y.to(dtype)


@_requires_darwin
@skipIfRefEager("asserts on the generated code; ref mode runs the kernel eagerly")
class TestMetalFastMath(unittest.TestCase):
    """``fast_math=True`` lowers ``exp`` and plain true divisions approximately."""

    def test_exp_and_divide_take_fast_forms(self) -> None:
        for dtype, tol in (
            (torch.float32, 1e-5),
            (torch.float16, 1e-3),
            (torch.bfloat16, 1e-2),
        ):
            with self.subTest(dtype=dtype):
                x, y = _exp_div_args(dtype)
                args = (x.to(DEVICE), y.to(DEVICE))
                code = fast_exp_div.bind(args).to_code()
                self.assertIn("metal.fast.exp(", code)
                self.assertIn("metal.fast.divide(", code)
                expected = torch.exp(x.double()) / y.double()
                torch.testing.assert_close(
                    fast_exp_div(*args).cpu(),
                    expected.to(dtype),
                    atol=tol,
                    rtol=tol,
                    equal_nan=True,
                )

    def test_int_true_division_takes_fast_divide(self) -> None:
        x = torch.randint(-1000, 1000, (300,), dtype=torch.int32)
        y = torch.randint(1, 50, (300,), dtype=torch.int32)
        args = (x.to(DEVICE), y.to(DEVICE))
        self.assertIn("metal.fast.divide(", fast_int_true_div.bind(args).to_code())
        torch.testing.assert_close(
            fast_int_true_div(*args).cpu(),
            (x.double() / y.double()).float(),
        )

    def test_softmax_two_pass_takes_fast_forms(self) -> None:
        # Columns 256-299 are a tail of the 64-wide block.
        x = torch.randn(33, 300, dtype=torch.float16)
        x[7] = float("-inf")
        args = (x.to(DEVICE),)
        kernel = helion.kernel(
            softmax_two_pass.fn,
            backend="metal",
            fast_math=True,
            configs=[helion.Config(block_sizes=[4, 64])],
        )
        code = kernel.bind(args).to_code()
        self.assertEqual(code.count("metal.fast.exp("), 3)
        self.assertEqual(code.count("metal.fast.divide("), 1)
        self.assertNotIn("metal.precise.exp(", code)
        torch.testing.assert_close(
            kernel(*args).cpu(),
            torch.softmax(x.double(), dim=1).half(),
            atol=1e-5,
            rtol=1e-2,
            equal_nan=True,
        )

    def test_without_fast_math_keeps_precise_forms(self) -> None:
        x, y = _exp_div_args(torch.float32)
        args = (x.to(DEVICE), y.to(DEVICE))
        code = precise_exp_div.bind(args).to_code()
        self.assertIn("metal.precise.exp(", code)
        self.assertNotIn("metal.fast.", code)
        torch.testing.assert_close(
            precise_exp_div(*args).cpu(), torch.exp(x) / y, equal_nan=True
        )

    def test_rounding_mode_division_stays_exact(self) -> None:
        # Every quotient is an integer, where an approximate quotient a hair
        # low truncates to the integer below.
        x = 3 * torch.arange(-500, 500, dtype=torch.float32)
        y = torch.full_like(x, 3.0)
        args = (x.to(DEVICE), y.to(DEVICE))
        self.assertNotIn("metal.fast.divide", fast_trunc_div.bind(args).to_code())
        torch.testing.assert_close(
            fast_trunc_div(*args).cpu(),
            torch.div(x, y, rounding_mode="trunc"),
            atol=0,
            rtol=0,
        )


def _axpb_body(x: torch.Tensor, s: float, t: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * s + t
    return out


_AXPB_CONFIG = helion.Config(block_sizes=[64])
# One kernel object per test, so each starts without a prepared call or a
# launch plan.
warm_axpb = helion.kernel(
    _axpb_body, backend="metal", static_shapes=True, config=_AXPB_CONFIG
)
guarded_axpb = helion.kernel(
    _axpb_body, backend="metal", static_shapes=True, config=_AXPB_CONFIG
)
by_value_axpb = helion.kernel(
    _axpb_body, backend="metal", static_shapes=True, config=_AXPB_CONFIG
)
int_scalar_axpb = helion.kernel(
    _axpb_body, backend="metal", static_shapes=True, config=_AXPB_CONFIG
)


@helion.kernel(backend="metal", static_shapes=True, config=_AXPB_CONFIG)
def warm_scale_spec(x: torch.Tensor, c: int) -> torch.Tensor:
    c = hl.specialize(c)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * c
    return out


@helion.kernel(backend="metal", static_shapes=False, config=_AXPB_CONFIG)
def dynamic_shift(x: torch.Tensor, k: int, f: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + k * f
    return out


@helion.kernel(
    backend="metal",
    static_shapes=False,
    config=helion.Config(block_sizes=[8], reduction_loops=[32]),
)
def dynamic_rowsum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].sum(-1)
    return out


def _metal_kernel(bound: helion.runtime.kernel.BoundKernel) -> _MetalKernel:
    """The ``@metal_jit`` kernel in a compiled ``bound``'s generated module."""
    assert bound._run is not None
    (kernel,) = [
        obj for obj in bound._run.__globals__.values() if isinstance(obj, _MetalKernel)
    ]
    return kernel


def _powers_of_two(n: int) -> torch.Tensor:
    """Signed powers of two, so ``x * s`` is exact and ``x * s + t`` rounds
    once whether or not the shader contracts it into an FMA."""
    exponents = torch.arange(n) % 7 - 3
    signs = 1 - 2 * (torch.arange(n) % 2)
    return (signs * 2.0**exponents).to(torch.float32)


@_requires_darwin
@skipIfRefEager("checks the compiled call path; ref mode runs the kernel eagerly")
class TestMetalWarmCall(unittest.TestCase):
    """A warm call reuses the prepared call and the launch plan, and binds
    runtime scalars by value; any other change still takes the full path."""

    def _assert_bitwise(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(
            actual.cpu(), expected, atol=0, rtol=0, equal_nan=True
        )

    def test_warm_call_reuses_prepared_call(self) -> None:
        x = _powers_of_two(100)
        warm_axpb(x.to(DEVICE), 2.0, 1.0)
        self.assertIsNotNone(warm_axpb._prepared_call)
        with (
            patch.object(
                helion.Kernel,
                "_fast_dispatch_key_and_guards",
                side_effect=AssertionError,
            ),
            patch.object(helion.Kernel, "bind", side_effect=AssertionError),
        ):
            for s, t in ((-0.5, 3.0), (math.nan, 1.0)):
                with self.subTest(s=s, t=t):
                    self._assert_bitwise(warm_axpb(x.to(DEVICE), s, t), x * s + t)

    def test_bound_call_reuses_direct_prepared_call(self) -> None:
        x = torch.arange(100, dtype=torch.float32)
        bound = warm_scale_spec.bind((x.to(DEVICE), 3))
        bound(x.to(DEVICE), 3)
        bound(x.to(DEVICE), 3)
        self.assertIsNotNone(bound._direct_prepared_call)
        with patch.object(helion.Kernel, "bind", side_effect=AssertionError):
            self._assert_bitwise(bound(x.to(DEVICE), 3), x * 3)

    def test_runtime_scalars_bind_by_value(self) -> None:
        x = _powers_of_two(100)
        args = (x.to(DEVICE), 2.0, 1.0)
        code = by_value_axpb.bind(args).to_code()
        self.assertIn("_by_value=(2, 3)", code)
        self.assertNotIn("scalar_tensor", code)
        cases = (
            (2.0, 1.0),
            (-0.0, 3.0),
            (math.nan, 1.0),
            (math.inf, -math.inf),
            (16777217.0, 0.1),
        )
        by_value_axpb(*args)
        kernel = _metal_kernel(by_value_axpb.bind(args))
        self.assertIn("constant float& s [[buffer(2)]]", kernel.msl_source)
        plans = len(kernel._plans)
        for s, t in cases:
            with self.subTest(s=s, t=t):
                expected = x * torch.tensor(s) + torch.tensor(t)
                self._assert_bitwise(by_value_axpb(x.to(DEVICE), s, t), expected)
        self.assertEqual(len(kernel._plans), plans)

    def test_runtime_scalars_follow_baked_arguments(self) -> None:
        x = torch.zeros(200)[::2]
        args = (x.to(DEVICE), 16777217, 1.0)
        code = dynamic_shift.bind(args).to_code()
        self.assertIn("x.stride(0), k, f", code)
        self.assertIn("_by_value=(4, 5)", code)
        # k binds exactly; ``k * f`` is float arithmetic, so 16777217 rounds
        # to float32 as torch rounds it.
        self._assert_bitwise(dynamic_shift(*args), torch.full_like(x, 16777216.0))
        self._assert_bitwise(
            dynamic_shift(x.to(DEVICE), 3, -0.5), torch.full_like(x, -1.5)
        )
        # The baked size and strides take no buffer slot, so the scalars
        # follow x and out.
        msl = _metal_kernel(dynamic_shift.bind(args)).msl_source
        self.assertIn("constant long& k [[buffer(2)]]", msl)
        self.assertIn("constant float& f [[buffer(3)]]", msl)

    def test_launch_plan_misses_on_changed_signature(self) -> None:
        shapes = ((8, 100), (8, 65), (8, 100))
        xs = [torch.randn(shape) for shape in shapes]
        for x in xs:
            torch.testing.assert_close(
                dynamic_rowsum(x.to(DEVICE)).cpu(), x.sum(-1), atol=1e-5, rtol=1e-5
            )
        kernel = _metal_kernel(dynamic_rowsum.bind((xs[0].to(DEVICE),)))
        self.assertEqual(len(kernel._plans), 2)

    def test_launch_plan_key(self) -> None:
        key = _MetalKernel._plan_key
        dims = (64, 1, 1)
        x = torch.randn(16, device=DEVICE)
        base = key(dims, (x, 1024, 2.0), (2,))
        self.assertEqual(base, key(dims, (torch.zeros_like(x), 1024, 2.0), (2,)))
        self.assertEqual(base, key(dims, (x, 1024, -7.5), (2,)))
        for changed in (
            key(dims, (x.half(), 1024, 2.0), (2,)),
            key(dims, (x[None], 1024, 2.0), (2,)),
            key(dims, (x, 512, 2.0), (2,)),
            key(dims, (x, 1024, 2), (2,)),
            key((128, 1, 1), (x, 1024, 2.0), (2,)),
            key(dims, (x, 1024, 2.0), ()),
        ):
            self.assertNotEqual(base, changed)

    def test_changed_arguments_miss_prepared_call(self) -> None:
        guarded_axpb(_powers_of_two(100).to(DEVICE), 2.0, 1.0)
        cases = {
            "tail": _powers_of_two(130).to(DEVICE),
            "fp16": _powers_of_two(100).half().to(DEVICE),
            "strided": _powers_of_two(200).to(DEVICE)[::2],
        }
        for name, x in cases.items():
            with (
                self.subTest(name),
                patch.object(
                    helion.Kernel,
                    "_fast_dispatch_key_and_guards",
                    autospec=True,
                    side_effect=helion.Kernel._fast_dispatch_key_and_guards,
                ) as slow_path,
            ):
                self._assert_bitwise(guarded_axpb(x, 2.0, 1.0), x.cpu() * 2.0 + 1.0)
                slow_path.assert_called()

    def test_by_value_scalar_must_be_int_or_float(self) -> None:
        x = torch.ones(100, device=DEVICE)
        bound = int_scalar_axpb.bind((x, 2.0, 1.0))
        bound(x, 2.0, 1.0)
        kernel = _metal_kernel(bound)
        out = torch.empty_like(x)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "binds runtime scalars as int or float"
        ):
            kernel.launch((2,), (64, 1, 1), (x, out, True, 1.0), (2, 3))


# ---------------------------------------------------------------------------
# Kernels - dims of length c * block held as per-thread parts
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy4(a: torch.Tensor) -> torch.Tensor:
    m, k4 = a.shape
    kq = k4 // 4
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kq)
    for tm in hl.tile(m):
        for tk in hl.tile(kq, block_size=bk):
            s = 4 * tk.begin
            out[tm, s : s + 4 * bk] = a[tm, s : s + 4 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy_rows(a: torch.Tensor) -> torch.Tensor:
    k2, n = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tn in hl.tile(n):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[s : s + 2 * bk, tn] = a[s : s + 2 * bk, tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy_from1(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(1, kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dequant_int4(b: torch.Tensor) -> torch.Tensor:
    kh, n = b.shape
    out = torch.zeros([2 * kh, n], dtype=torch.bfloat16, device=b.device)
    bk = hl.register_block_size(kh)
    for tn in hl.tile(n):
        for tk in hl.tile(kh, block_size=bk):
            bt = b[tk, tn]
            lo_raw = (bt & 0xF).to(torch.int32)
            lo = torch.where(lo_raw >= 8, lo_raw - 16, lo_raw)
            hi = (bt >> 4).to(torch.int32)
            st = torch.stack([lo.to(torch.bfloat16), hi.to(torch.bfloat16)], dim=1)
            s = 2 * tk.begin
            out[s : s + 2 * bk, tn] = st.reshape(tk.block_size * 2, tn.block_size)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dequant_int4_cols(b: torch.Tensor) -> torch.Tensor:
    m, kh = b.shape
    out = torch.zeros([m, 2 * kh], dtype=torch.float32, device=b.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            bt = b[tm, tk]
            lo_raw = (bt & 0xF).to(torch.int32)
            lo = torch.where(lo_raw >= 8, lo_raw - 16, lo_raw)
            hi = (bt >> 4).to(torch.int32)
            st = torch.stack([lo.to(torch.float32), hi.to(torch.float32)], dim=-1)
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = st.reshape(tm.block_size, tk.block_size * 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def interleave3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 3 * kh], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tm, tk], y[tm, tk], z[tm, tk]], dim=2)
            s = 3 * tk.begin
            out[tm, s : s + 3 * bk] = st.reshape(tm.block_size, tk.block_size * 3)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_three_same(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 3 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for _tm in hl.tile(m):
        for tj, tk in hl.tile([m, kh], block_size=[None, bk]):
            st = torch.stack([x[tj, tk], x[tj, tk] * 2.0, y[tj, tk]], dim=2)
            s = 3 * tk.begin
            out[tj, s : s + 3 * bk] = st.reshape(tj.block_size, 3 * bk)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def broadcast_store(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k2 = out.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[0:1, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def colon_rows_copy(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[:, s : s + 2 * bk] = a[:, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def multi_dim_device_loop(a: torch.Tensor) -> torch.Tensor:
    b, m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bm = hl.register_block_size(m)
    bk = hl.register_block_size(kh)
    for tb in hl.tile(b):
        for tm, tk in hl.tile([m, kh], block_size=[bm, bk]):
            s = 2 * tk.begin
            out[tb, tm, s : s + 2 * bk] = a[tb, tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def none_before(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k2 = out.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = x[None, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def int_before(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k2 = out.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[1, tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def ellipsis_widened(x: torch.Tensor, kh: hl.constexpr) -> torch.Tensor:
    out = torch.zeros_like(x)
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[..., s : s + 2 * bk] = x[..., s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def sibling_loops_shared_bs(
    a1: torch.Tensor, a2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = a1.size(0)
    o1 = torch.zeros_like(a1)
    o2 = torch.zeros_like(a2)
    bk = hl.register_block_size(a1.size(1) // 2)
    for tm in hl.tile(m):
        for tk in hl.tile(a1.size(1) // 2, block_size=bk):
            s = 2 * tk.begin
            o1[tm, s : s + 2 * bk] = a1[tm, s : s + 2 * bk]
        for tk in hl.tile(a2.size(1) // 2, block_size=bk):
            s = 2 * tk.begin
            o2[tm, s : s + 2 * bk] = a2[tm, s : s + 2 * bk]
    return o1, o2


@helion.kernel(backend="metal", static_shapes=True)
def copy_carry(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    count = torch.zeros([m], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        c = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
            c = c + 1.0
        count[tm] = c
    return out, count


def _widened_copy_intarg(a: torch.Tensor, kh: int) -> torch.Tensor:
    m = a.size(0)
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


def _widened_copy_intarg_specialized(a: torch.Tensor, kh: int) -> torch.Tensor:
    kh = hl.specialize(kh)
    m = a.size(0)
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_into_mm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    n = w.size(1)
    kh = k2 // 2
    out = torch.zeros([m, n], dtype=torch.float32, device=a.device)
    bk = hl.register_block_size(kh)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            acc = acc + torch.mm(a[tm, s : s + 2 * bk], w[s : s + 2 * bk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_rows_two_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([m, kh], block_size=[None, bk]):
            st = torch.stack([x[tj, tk], y[tm, tk]], dim=2)
            s = 2 * tk.begin
            out[tj, s : s + 2 * bk] = st.reshape(tj.block_size, 2 * bk)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_rows_two_tiles_swapped(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([m, kh], block_size=[None, bk]):
            st = torch.stack([y[tm, tk], x[tj, tk]], dim=2)
            s = 2 * tk.begin
            out[tj, s : s + 2 * bk] = st.reshape(tj.block_size, 2 * bk)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_three_mixed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 3 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([m, kh], block_size=[None, bk]):
            st = torch.stack([x[tj, tk], x[tj, tk] * 2.0, y[tm, tk]], dim=2)
            s = 3 * tk.begin
            out[tj, s : s + 3 * bk] = st.reshape(tj.block_size, 3 * bk)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_lit_two_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m, block_size=8):
        for tj, tk in hl.tile([m, kh], block_size=[8, bk]):
            st = torch.stack([x[tj, tk], y[tm, tk]], dim=2)
            s = 2 * tk.begin
            out[tj, s : s + 2 * bk] = st.reshape(8, 2 * bk)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_transposed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A tile stacked beside its transpose: two tiles at each position."""
    m, n = x.shape
    out = torch.zeros([m, 2 * n], dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(8, 8)
    bn = hl.register_block_size(8, 8)
    for tm, tn in hl.tile([m, n], block_size=[bm, bn]):
        st = torch.stack([x[tm, tn], y[tn, tm]], dim=2)
        s = 2 * tn.begin
        out[tm, s : s + 2 * bn] = st.reshape(bm, 2 * bn)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def misaligned_cols(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    kh = out.size(1) // 2
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[:, s : s + 2 * bk] = a[s : s + 2 * bk, 0:1]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def misaligned_cols_stack(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    kh = out.size(1) // 2
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tk, 0:1], y[tk, 0:1]], dim=1)
            s = 2 * tk.begin
            out[:, s : s + 2 * bk] = st.reshape(tk.block_size * 2, 1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def colon_value_under_tile(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[:, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def other_tile_value(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([m, kh], block_size=[None, bk]):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tj, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def lit_other_tile_value(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m, block_size=8):
        for tj, tk in hl.tile([m, kh], block_size=[8, bk]):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tj, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_rows_lit_other(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m, block_size=8):
        for tj, tk in hl.tile([m, kh], block_size=[8, bk]):
            st = torch.stack([x[tj, tk], y[tj, tk]], dim=2)
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = st.reshape(8, tk.block_size * 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_mid_swapped_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh, n = x.shape
    out = torch.zeros([m, 2 * kh, n], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tj, tn in hl.tile([m, n]):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tn, tk, tj], y[tn, tk, tj]], dim=2)
            s = 2 * tk.begin
            out[tj, s : s + 2 * bk, tn] = st.reshape(
                tn.block_size, 2 * bk, tj.block_size
            )
    return out


@helion.kernel(backend="metal", static_shapes=True)
def same_block_rows(a: torch.Tensor) -> torch.Tensor:
    kh = a.size(0)
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tk, s : s + 2 * bk] = a[tk, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def same_block_store_only(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    kh = out.size(0)
    bk = hl.register_block_size(kh)
    for _ in hl.grid(1):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tk, s : s + 2 * bk] = a[0:1, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_blocks_one_loop(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    b1 = hl.register_block_size(kh)
    b2 = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([kh, kh], block_size=[b1, b2]):
            out[tm, 2 * tk.begin : 2 * tk.begin + 2 * b2] = a[
                tm, 2 * tj.begin : 2 * tj.begin + 2 * b1
            ]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def factor_mismatch(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            out[tm, 4 * tk.begin : 4 * tk.begin + 4 * bk] = a[
                tm, 2 * tk.begin : 2 * tk.begin + 2 * bk
            ]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy_into(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_copy_odd(a: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    kh = k // 2
    out = torch.zeros([m, 2 * kh], dtype=a.dtype, device=a.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_plus_one(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk] + 1.0
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stacked_bf16_into_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=torch.float32, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tm, tk], y[tm, tk]], dim=2)
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = st.reshape(tm.block_size, tk.block_size * 2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_iota_as_value(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            idx = torch.ops.prims.iota(
                2 * bk,
                start=2 * tk.begin,
                step=1,
                dtype=torch.int32,
                device=a.device,
                requires_grad=False,
            )
            out[tm, idx] = a[tm, idx] + idx.to(torch.float32)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_step_2(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            idx = torch.ops.prims.iota(
                2 * bk,
                start=2 * tk.begin,
                step=2,
                dtype=torch.int32,
                device=a.device,
                requires_grad=False,
            )
            out[tm, idx] = a[tm, idx]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_start_other_tile(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    kj, m, k2 = out.shape
    kh = k2 // 2
    bk = hl.register_block_size(kh)
    bj = hl.register_block_size(kj)
    for tm in hl.tile(m):
        for tj, tk in hl.tile([kj, kh], block_size=[bj, bk]):
            s = 2 * tk.begin
            t = 2 * tj.begin
            out[tj, tm, s : s + 2 * bk] = a[0:1, tm, t : t + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_start_other_factor(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m = a.size(0)
    kh = out.size(1) // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            t = 3 * tk.begin
            out[tm, s : s + 2 * bk] = a[tm, t : t + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_start_tile_id(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m = a.size(0)
    kh = out.size(1) // 2
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            t = 2 * tk.id
            out[tm, s : s + 2 * bk] = a[tm, t : t + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def constant_at_widened_index(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = 1.0
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_at_colon_index(a: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, :] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_gather_load(a: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = a[rows[tm], s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_gather_store(a: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    m, k2 = a.shape
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            s = 2 * tk.begin
            out[rows[tm], s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_root_tile(a: torch.Tensor) -> torch.Tensor:
    k2 = a.size(1)
    kh = k2 // 2
    out = torch.zeros_like(a)
    bk = hl.register_block_size(kh)
    for tk in hl.tile(kh, block_size=bk):
        s = 2 * tk.begin
        out[:, s : s + 2 * bk] = a[:, s : s + 2 * bk]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def widened_in_inner_loop(
    a: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k2 = a.shape
    kh = k2 // 2
    n = x.size(1)
    out = torch.zeros_like(a)
    out_x = torch.zeros_like(x)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            for tn in hl.tile(n):
                out_x[tm, tn] = x[tm, tn]
                s = 2 * tk.begin
                out[tm, s : s + 2 * bk] = a[tm, s : s + 2 * bk]
    return out, out_x


@helion.kernel(backend="metal", static_shapes=True)
def flattening_view(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([m * n], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m, block_size=4):
        out[tm.begin * n : tm.begin * n + 4 * n] = x[tm, :].reshape(4 * n)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_merged_past_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros([m, 2 * kh], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tm, tk], y[tm, tk]], dim=1)
            s = 2 * tk.begin
            out[tm, s : s + 2 * bk] = st.reshape(tm.block_size, 2 * tk.block_size)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_dim_0(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    kh, n = x.shape
    out = torch.zeros([2 * kh, n], dtype=x.dtype, device=x.device)
    bk = hl.register_block_size(kh)
    for tn in hl.tile(n):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tk, tn], y[tk, tn]], dim=0)
            s = 2 * tk.begin
            out[s : s + 2 * bk, tn] = st.reshape(2 * tk.block_size, tn.block_size)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_of_one(x: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros_like(x)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            st = torch.stack([x[tm, tk]], dim=2)
            out[tm, tk] = st.reshape(tm.block_size, tk.block_size)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def stack_summed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, kh = x.shape
    out = torch.zeros_like(x)
    bk = hl.register_block_size(kh)
    for tm in hl.tile(m):
        for tk in hl.tile(kh, block_size=bk):
            out[tm, tk] = torch.stack([x[tm, tk], y[tm, tk]], dim=2).sum(2)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mpp_beside_widened(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = y.size(1)
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    copy = torch.zeros_like(a)
    kh = a.size(1) // 2
    bk = hl.register_block_size(kh)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    for tr in hl.tile(a.size(0)):
        for tq in hl.tile(kh, block_size=bk):
            s = 2 * tq.begin
            copy[tr, s : s + 2 * bk] = a[tr, s : s + 2 * bk]
    return out, copy


def _unpack_int4(b: torch.Tensor) -> torch.Tensor:
    lo = b.long() & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = b.long() >> 4
    return torch.stack([lo, hi], dim=1).reshape(2 * b.size(0), b.size(1))


# ---------------------------------------------------------------------------
# Tests - dims of length c * block held as per-thread parts
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalFactoredDims(unittest.TestCase):
    """A dim of length ``c * block`` that is the row-major merge of
    ``(block, c)``, from a widened slice ``x[..., c*t.begin : c*t.begin +
    c*bk]`` or a stack merged by a view, is held as ``c`` parts per thread
    and stored at a widened index over the same block and factor, or fed to
    a matmul the MPP planner lowers (test_metal_mpp_matmul.py).  Every other
    layout or consumer is rejected."""

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        expected: object,
        configs: tuple[helion.Config | None, ...] = (None,),
    ) -> None:
        bound = kernel.bind(args)
        for config in configs:
            with self.subTest(kernel=kernel.name, config=config):
                config = config or bound.config_spec.default_config()
                fresh = tuple(
                    arg.clone() if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                )
                result = bound.compile_config(config)(*fresh)
                torch.testing.assert_close(result, expected)

    def _assert_rejected(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        message: str,
        config: helion.Config | None = None,
    ) -> None:
        with (
            self.subTest(kernel=kernel.name, config=config),
            self.assertRaisesRegex(exc.BackendUnsupported, message),
        ):
            bound = kernel.bind(args)
            bound.to_code(config or bound.config_spec.default_config())

    def test_widened_copy(self) -> None:
        """kh = 37, so no block size divides it; ``num_threads=[4, 0]``
        lane-loops the widened tile."""
        a = torch.randn(5, 74, device=DEVICE)
        self._check(
            widened_copy,
            (a,),
            a,
            (
                None,
                helion.Config(block_sizes=[16, 32], num_threads=[4, 0]),
                helion.Config(block_sizes=[64, 4]),
            ),
        )

    def test_widened_copy_variants(self) -> None:
        a = torch.randn(3, 4 * 13, device=DEVICE)
        self._check(widened_copy4, (a,), a, (None, helion.Config(block_sizes=[4, 2])))
        a = torch.randn(2 * 13, 11, device=DEVICE, dtype=torch.float16)
        self._check(
            widened_copy_rows, (a,), a, (None, helion.Config(block_sizes=[4, 8]))
        )
        a = torch.randn(4, 2, device=DEVICE)
        self._check(widened_copy, (a,), a)
        a = torch.randn(5, 2 * 9, device=DEVICE)
        expected = a.clone()
        expected[:, :2] = 0
        self._check(
            widened_copy_from1,
            (a,),
            expected,
            (None, helion.Config(block_sizes=[2, 2])),
        )

    def test_merged_stacks(self) -> None:
        b = torch.randint(-128, 128, (19, 7), dtype=torch.int8, device=DEVICE)
        self._check(
            dequant_int4,
            (b,),
            _unpack_int4(b).to(torch.bfloat16),
            (None, helion.Config(block_sizes=[8, 4])),
        )
        b = torch.randint(-128, 128, (6, 19), dtype=torch.int8, device=DEVICE)
        self._check(
            dequant_int4_cols,
            (b,),
            _unpack_int4(b.T.contiguous()).T.float(),
            (None, helion.Config(block_sizes=[32, 2])),
        )
        x = torch.randn(5, 11, device=DEVICE, dtype=torch.float16)
        y = torch.randn(5, 11, device=DEVICE)
        z = torch.randint(-50, 50, (5, 11), dtype=torch.int32, device=DEVICE)
        expected = torch.stack([x.float(), y, z.float()], dim=2).reshape(5, 33)
        self._check(
            interleave3, (x, y, z), expected, (None, helion.Config(block_sizes=[4, 2]))
        )

    def test_stack_inputs_of_one_layout(self) -> None:
        """Three inputs over one tile, under an unrelated tile's loop."""
        x = torch.randn(8, 8, device=DEVICE)
        y = torch.randn(8, 8, device=DEVICE)
        self._check(
            stack_three_same,
            (x, y),
            torch.stack([x, 2 * x, y], 2).reshape(8, 24),
            (
                helion.Config(block_sizes=[8, 4, 4]),
                helion.Config(block_sizes=[4, 8, 2], num_threads=[0, 2, 0]),
            ),
        )

    def test_value_dims_broadcast_or_match_the_store(self) -> None:
        a = torch.randn(1, 2 * 9, device=DEVICE)
        out = torch.zeros(6, 2 * 9, device=DEVICE)
        self._check(
            broadcast_store,
            (a, out),
            a.expand(6, -1),
            (None, helion.Config(block_sizes=[4, 2])),
        )
        a = torch.randn(6, 2 * 9, device=DEVICE)
        self._check(
            colon_rows_copy,
            (a,),
            a,
            (None, helion.Config(block_sizes=[4], num_threads=[2])),
        )
        a = torch.randn(2, 7, 2 * 11, device=DEVICE)
        self._check(
            multi_dim_device_loop,
            (a,),
            a,
            (None, helion.Config(block_sizes=[8, 2, 1], num_threads=[0, 2, 2])),
        )

    def test_extent_counts_tensor_dims(self) -> None:
        """A ``None`` indexes no tensor dim and an int indexes one."""
        configs = (None, helion.Config(block_sizes=[16, 16], num_threads=[4, 0]))
        x = torch.randn(2 * 9, device=DEVICE)
        out = torch.zeros(5, 2 * 9, device=DEVICE)
        self._check(none_before, (x, out), x[None].expand(5, -1), configs)
        a = torch.randn(2, 5, 2 * 9, device=DEVICE)
        self._check(int_before, (a, out), a[1], configs)
        x = torch.randn(3, 5, 2 * 37, device=DEVICE)
        self._check(ellipsis_widened, (x, 37), x)

    def test_sibling_loops_over_one_block_size(self) -> None:
        """Each widened slice's extent is checked against its own loop's end
        (37 and 13), not the first loop over the block."""
        a1 = torch.randn(5, 2 * 37, device=DEVICE)
        a2 = torch.randn(5, 2 * 13, device=DEVICE)
        bound = sibling_loops_shared_bs.bind((a1, a2))
        for config in (
            bound.config_spec.default_config(),
            helion.Config(block_sizes=[4, 8]),
            helion.Config(block_sizes=[16, 2]),
        ):
            with self.subTest(config=config):
                o1, o2 = bound.compile_config(config)(a1, a2)
                torch.testing.assert_close(o1, a1)
                torch.testing.assert_close(o2, a2)

    def test_copy_beside_a_carry(self) -> None:
        a = torch.randn(5, 2 * 16, device=DEVICE)
        bound = copy_carry.bind((a,))
        config = bound.config_spec.default_config()
        tiles = math.ceil(16 / config.config["block_sizes"][0])
        out, count = bound.compile_config(config)(a)
        torch.testing.assert_close(out, a)
        torch.testing.assert_close(count, torch.full_like(count, tiles))

    def test_specialized_int_extent(self) -> None:
        kernel = helion.kernel(
            _widened_copy_intarg_specialized, backend="metal", static_shapes=True
        )
        a = torch.randn(5, 18, device=DEVICE)
        self._check(kernel, (a, 9), a)

    def test_parts_in_generated_code(self) -> None:
        code = widened_copy.bind((torch.randn(5, 74),)).to_code(
            helion.Config(block_sizes=[16, 32])
        )
        self.assertIn("iota_part_1", code)
        self.assertIn("load_part_1", code)

    def test_matmul_over_parts_rejects(self) -> None:
        """A matmul over parts that the MPP planner doesn't lower (a
        torch.mm; test_metal_mpp_matmul.py covers the ones it does)."""
        message = (
            r"a matmul over a dim held as per-thread parts \(a widened hl\.arange\(\) "
            r"slice or a merged torch\.stack\) that the cooperative MPP lowering "
            r"doesn't admit; "
        )
        self._assert_rejected(
            widened_into_mm, (torch.randn(5, 16), torch.randn(16, 6)), message
        )

    def test_stack_inputs_over_other_blocks_reject(self) -> None:
        # x[tj, tk] beside y[tm, tk]: metal/literal_dims.py refuses tj meeting
        # tm before the stack's lowering sees it (in stack_lit_two_tiles, tj's
        # block size became the literal 8 in x[tj, tk]).
        args = (torch.randn(8, 8), torch.randn(8, 8))
        for config in (
            helion.Config(block_sizes=[8, 8, 8]),
            helion.Config(block_sizes=[8, 4, 4]),
        ):
            self._assert_rejected(stack_rows_two_tiles, args, _PAIRS_TILES, config)
        config = helion.Config(block_sizes=[8, 8, 8])
        self._assert_rejected(stack_rows_two_tiles_swapped, args, _PAIRS_TILES, config)
        self._assert_rejected(stack_three_mixed, args, _PAIRS_TILES, config)
        self._assert_rejected(
            stack_lit_two_tiles, args, _PAIRS_TILES, helion.Config(block_sizes=[4])
        )
        self._assert_rejected(
            stack_transposed, args, _PAIRS_TILES, helion.Config(block_sizes=[8, 8])
        )

    def test_store_dims_over_other_blocks_reject(self) -> None:
        message = (
            r"Metal needs each value dim to be 1 or over the same block as the "
            r"store's, and the widened dims to face each other"
        )
        c8 = helion.Config(block_sizes=[8])
        self._assert_rejected(
            misaligned_cols, (torch.randn(16, 3), torch.zeros(16, 16)), message, c8
        )
        self._assert_rejected(
            misaligned_cols_stack,
            (torch.randn(8, 3), torch.randn(8, 3), torch.zeros(16, 16)),
            message,
            c8,
        )
        a = torch.randn(6, 2 * 8)
        self._assert_rejected(
            colon_value_under_tile, (a,), message, helion.Config(block_sizes=[8, 8])
        )
        self._assert_rejected(
            other_tile_value, (a,), message, helion.Config(block_sizes=[8, 8, 8])
        )
        a = torch.randn(8, 16)
        x, y = torch.randn(8, 8), torch.randn(8, 8)
        for config in (helion.Config(block_sizes=[4]), c8):
            self._assert_rejected(lit_other_tile_value, (a,), message, config)
            self._assert_rejected(stack_rows_lit_other, (x, y), message, config)
        x, y = torch.randn(8, 6, 8), torch.randn(8, 6, 8)
        self._assert_rejected(
            stack_mid_swapped_tiles,
            (x, y),
            message,
            helion.Config(block_sizes=[8, 8, 4]),
        )

    def test_widened_tile_indexed_twice_rejects(self) -> None:
        message = "that tile 0 also indexes in another dim"
        c8 = helion.Config(block_sizes=[8])
        self._assert_rejected(
            same_block_rows, (torch.randn(8, 16),), rf"a load at .*{message}", c8
        )
        self._assert_rejected(
            same_block_store_only,
            (torch.randn(1, 16), torch.zeros(8, 16)),
            rf"a store at .*{message}",
            c8,
        )

    def test_store_of_other_widening_rejects(self) -> None:
        message = "Metal pairs their parts only over the same tile and factor"
        self._assert_rejected(
            factor_mismatch,
            (torch.randn(4, 16), torch.zeros(4, 32)),
            rf"widening tile 0 by 2 at an hl.arange\(\) widening tile 0 by 4; {message}",
        )
        self._assert_rejected(
            two_blocks_one_loop,
            (torch.randn(4, 16),),
            message,
            helion.Config(block_sizes=[8, 8, 4]),
        )

    def test_extent_other_than_the_parts_rejects(self) -> None:
        exact = r"Metal needs the dim to be exactly 2 x the loop's end"
        self._assert_rejected(
            widened_copy_into,
            (torch.randn(4, 16), torch.zeros(4, 15)),
            rf"a store of dim 1 \(size 15\) .*{exact}(?!, and)",
        )
        self._assert_rejected(
            widened_copy_odd,
            (torch.randn(4, 17),),
            rf"a load of dim 1 .*{exact}(?!, and)",
        )
        # Dim 1 is 2 x 37, so a count that took the `...` as one dim would pass.
        self._assert_rejected(
            ellipsis_widened,
            (torch.randn(3, 2 * 37, 80), 37),
            rf"a load of dim 2 \(size 80\) .*{exact}",
        )
        hint = (
            rf"{exact}, and can't prove it of symbolic sizes; pass "
            r"static_shapes=True and hl.specialize\(\) any int argument"
        )
        a = torch.randn(5, 18)
        self._assert_rejected(
            helion.kernel(widened_copy.fn, backend="metal", static_shapes=False),
            (a,),
            hint,
        )
        self._assert_rejected(
            helion.kernel(_widened_copy_intarg, backend="metal", static_shapes=True),
            (a, 9),
            hint,
        )

    def test_other_consumers_reject(self) -> None:
        a = torch.randn(4, 16)
        self._assert_rejected(
            widened_plus_one,
            (a,),
            r"the load \w+ of shape .*, which Metal holds as per-thread parts, "
            r"feeds \w+; only a store at a widened hl.arange\(\) index",
        )
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        self._assert_rejected(
            stacked_bf16_into_f32,
            (x, x.clone()),
            r"the view \w+ .* feeds convert_element_type\w*",
        )
        self._assert_rejected(
            widened_iota_as_value,
            (a,),
            r"hl.arange\(\) \w+ widens tile 0 by 2 and feeds convert_element_type\w*; "
            r"Metal holds it as per-thread parts, which only a load's or a store's "
            r"index can consume",
        )

    def test_other_indices_reject(self) -> None:
        a = torch.randn(4, 16)
        rows = torch.arange(4)
        self._assert_rejected(
            constant_at_widened_index,
            (a,),
            "a store at a widened hl.arange\\(\\) index of a value Metal doesn't "
            "hold as per-thread parts",
        )
        self._assert_rejected(
            widened_at_colon_index,
            (a,),
            "a store of a value Metal holds as per-thread parts at an index "
            "without a widened hl.arange",
        )
        other = r"indexed by a widened hl.arange\(\) and another tensor"
        self._assert_rejected(widened_gather_load, (a, rows), f"a load {other}")
        self._assert_rejected(widened_gather_store, (a, rows), f"a store {other}")

    def test_widened_outside_own_loop_body_rejects(self) -> None:
        message = (
            "widens tile 0 by 2, but not in the body of that tile's own device loop"
        )
        self._assert_rejected(widened_root_tile, (torch.randn(6, 18),), message)
        self._assert_rejected(
            widened_in_inner_loop, (torch.randn(4, 16), torch.randn(4, 4)), message
        )

    @_full_threadgroup
    def test_other_views_and_stacks_reject(self) -> None:
        x, y = torch.randn(4, 8), torch.randn(4, 8)
        self._assert_rejected(
            flattening_view,
            (torch.randn(8, 4),),
            "Metal lowers a view only where it merges a stacked dim",
        )
        merge = "Metal only merges the stacked dim into a block dim right before it"
        self._assert_rejected(stack_merged_past_block, (x, y), merge)
        self._assert_rejected(
            stack_dim_0, (torch.randn(8, 4), torch.randn(8, 4)), merge
        )
        self._assert_rejected(stack_of_one, (x,), "aten.stack of a single tensor")
        self._assert_rejected(
            stack_summed, (x, y), r"aten.stack \w+ whose result isn't merged by a view"
        )

    def test_other_widenings_keep_the_arange_rejection(self) -> None:
        """A step other than 1, or a start other than ``c * begin`` of the
        tile whose block the length widens, isn't a widened tile."""
        message = (
            "hl.arange\\(\\) of length 2\\*u0: Metal lowers only a range over a "
            "block size that no loop owns"
        )
        self._assert_rejected(widened_step_2, (torch.randn(4, 16),), message)
        self._assert_rejected(
            widened_start_other_tile,
            (torch.randn(1, 6, 32), torch.zeros(9, 6, 32)),
            message,
            helion.Config(block_sizes=[8, 8, 2]),
        )
        self._assert_rejected(
            widened_start_other_factor,
            (torch.randn(6, 56), torch.zeros(6, 32)),
            message,
            helion.Config(block_sizes=[4, 2]),
        )
        self._assert_rejected(
            widened_start_tile_id,
            (torch.randn(6, 32), torch.zeros(6, 32)),
            message,
            helion.Config(block_sizes=[4, 4]),
        )

    def test_lane_looped_carry_rejects(self) -> None:
        self._assert_rejected(
            copy_carry,
            (torch.randn(5, 32),),
            r"a store at a widened hl.arange\(\) index: the loop over block\(s\) "
            r"\[0\] carries \w+ through a lane loop",
            helion.Config(block_sizes=[16, 16], num_threads=[4, 0]),
        )

    def test_widened_store_beside_an_mpp_matmul(self) -> None:
        """A store at a widened index in a kernel whose K loop is an MPP op."""
        x = torch.randn(32, 32, device=DEVICE, dtype=torch.float16)
        a = torch.randn(4, 16, device=DEVICE)
        bound = mpp_beside_widened.bind((x, x.clone(), a))
        code = bound.to_code(bound.config_spec.default_config())
        self.assertIn("_helion_mpp_acc(", code)
        self.assertIn("load_part_1", code)
        out, copy = bound.compile_config(bound.config_spec.default_config())(
            x, x.clone(), a
        )
        torch.testing.assert_close(out, x.float() @ x.float(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(copy, a)


# ---------------------------------------------------------------------------
# Threadgroup budget: num_warps sizes the launch
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", static_shapes=True)
def budget_add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + y[tm, tn]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_row_sum(x: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    for tm in hl.tile(x.size(0)):
        out[tm] = x[tm, :].sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_row_argmax(x: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)], dtype=torch.int64)
    for tm in hl.tile(x.size(0)):
        out[tm] = x[tm, :].argmax(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_nested_pair(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        for tn in hl.tile(x.size(1)):
            out[tm, tn] = x[tm, tn] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_dot_k_tiles(
    c: torch.Tensor, p: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    m = c.size(0)
    n = p.size(0)
    out = torch.empty([m, n], dtype=torch.float32, device=c.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = hl.dot(c[tm, :], p[tn, :].T, acc=acc)
        for tj in hl.tile(x.size(1)):
            acc = acc + x[tm, tj].sum(-1, keepdim=True)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_dot16_k_tiles(
    c: torch.Tensor, p: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """``budget_dot_k_tiles`` with a float16 dot, which keeps it scalar."""
    m = c.size(0)
    n = p.size(0)
    out = torch.empty([m, n], dtype=torch.float32, device=c.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.dot(c[tm, :], p[tn, :].T, out_dtype=torch.float16).to(torch.float32)
        for tj in hl.tile(x.size(1)):
            acc = acc + x[tm, tj].sum(-1, keepdim=True)
        out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_dot_beside_row_sum(
    c: torch.Tensor, p: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    m = c.size(0)
    n = p.size(0)
    out = torch.empty([m, n], dtype=torch.float32, device=c.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        acc = hl.dot(c[tm, :], p[tn, :].T, acc=acc)
        out[tm, tn] = acc + r[tm, :].sum(-1, keepdim=True)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def budget_bf16_round_trip_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tm in hl.tile(x.size(0)):
        v = x[tm, :].to(torch.float32) * 3.0
        out[tm] = v.to(torch.bfloat16).to(torch.float32).sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_tile_softmax_sum(x: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    for tm in hl.tile(x.size(0), block_size=8):
        v = x[tm, :]
        out[tm] = (v - v.amax(-1, keepdim=True)).exp().sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_tile_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0), block_size=8):
        out[tm] = x[tm] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_tile_from_begin(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm in hl.tile(8, x.size(0), block_size=8):
        out[tm] = x[tm] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_tile_id(x: torch.Tensor) -> torch.Tensor:
    out = x.new_zeros([(x.size(0) + 7) // 8])
    for tm in hl.tile(x.size(0), block_size=8):
        out[tm.id] = x[tm].sum()
    return out


@helion.kernel(backend="metal", static_shapes=True)
def literal_tile_index(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0), block_size=8):
        out[tm] = x[tm] + tm.index
    return out


@helion.kernel(backend="metal", static_shapes=True)
def registered_tile_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    block = hl.register_block_size(x.size(0))
    for tm in hl.tile(x.size(0), block_size=block):
        out[tm] = x[tm] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def registered_tile_row_sum(x: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    block = hl.register_block_size(x.size(0))
    for tm in hl.tile(x.size(0), block_size=block):
        out[tm] = x[tm, :].sum(-1)
    return out


gdn_fwd_h_metal = helion.kernel(
    helion_gdn_fwd_h.fn, backend="metal", static_shapes=True
)


@helion.kernel(backend="metal", static_shapes=True)
def gdn_fwd_h_transposed(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """gdn_fwd_h with its first product computed transposed, so the second
    dot reads a permute of it, and both dots keep the scalar contraction."""
    batch, seqlen, nheads, dhead = k.shape
    dhead = hl.specialize(dhead)
    chunk_size = hl.specialize(chunk_size)
    dstate = u.shape[-1]
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    h = torch.empty(
        batch, nchunks, nheads, dhead, dstate, dtype=k.dtype, device=k.device
    )
    block_v = hl.register_block_size(dstate)
    for tile_b, tile_h, tile_v in hl.tile(
        [batch, nheads, dstate], block_size=[1, 1, block_v]
    ):
        i_b = tile_b.id
        i_h = tile_h.id
        b_h = hl.zeros([dhead, tile_v], dtype=torch.float32)
        for t_i in hl.tile(seqlen, block_size=chunk_size):
            h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(k.dtype)
            c_h = b_h.to(k.dtype)
            b_vt = hl.dot(c_h.T, w[i_b, t_i, i_h, :].T, out_dtype=torch.float32)
            b_v = u[i_b, t_i, i_h, tile_v].to(torch.float32) - b_vt.T
            t_i_last = min(t_i.begin + chunk_size, seqlen) - 1
            b_g_last = g[i_b, t_i_last, i_h].to(torch.float32)
            b_g = g[i_b, t_i, i_h].to(torch.float32)
            b_v *= torch.exp(b_g_last - b_g)[:, None]
            b_h *= torch.exp(b_g_last)
            b_h = hl.dot(k[i_b, t_i, i_h, :].T, b_v.to(k.dtype), acc=b_h)
    return h


_GDN_CHUNK = 64


def _gdn_args(
    t: int, k: int, v: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """gdn_fwd_h's inputs for one batch and head, with ``w`` orthonormal per
    chunk (as the benchmark builds it) so the carried state stays bounded."""
    gen = torch.Generator().manual_seed(0)
    nt = t // _GDN_CHUNK
    raw = torch.randn(1, nt, 1, _GDN_CHUNK, k, generator=gen)
    wu, _, wv = torch.linalg.svd(raw, full_matrices=False)
    w = (wu @ wv).permute(0, 1, 3, 2, 4).reshape(1, t, 1, k)
    inc = 0.5 * math.log(1 / 256) * torch.rand(1, t, 1, generator=gen)
    return (
        torch.randn(1, t, 1, k, generator=gen).to(dtype),
        w.to(dtype),
        torch.randn(1, t, 1, v, generator=gen).to(dtype),
        torch.cumsum(inc, dim=1),
    )


def _gdn_reference(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """gdn_fwd_h in float64, rounding to ``k``'s dtype where the kernel does."""
    t, dk, dv = k.size(1), k.size(-1), u.size(-1)
    nt = t // _GDN_CHUNK

    def rounded(x: torch.Tensor) -> torch.Tensor:
        return x.to(k.dtype).double()

    def chunk(x: torch.Tensor, i: int) -> torch.Tensor:
        return x[0, i * _GDN_CHUNK : (i + 1) * _GDN_CHUNK, 0].double()

    out = torch.empty(1, nt, 1, dk, dv, dtype=torch.float64)
    state = torch.zeros(dk, dv, dtype=torch.float64)
    for i in range(nt):
        out[0, i, 0] = rounded(state)
        gc = chunk(g, i)
        new_v = (chunk(u, i) - chunk(w, i) @ rounded(state)) * torch.exp(gc[-1] - gc)[
            :, None
        ]
        state = state * torch.exp(gc[-1]) + chunk(k, i).T @ rounded(new_v)
    return out


def _launch(code: str) -> tuple[int, ...]:
    """The launch's threadgroup shape, from the host code's ``_block_dims``."""
    dims = re.search(r"_block_dims=\((\d+), (\d+), (\d+)\)", code)
    assert dims is not None, code
    return tuple(int(d) for d in dims.groups())


@_requires_darwin
class TestMetalThreadgroupBudget(unittest.TestCase):
    """A config's ``num_warps`` sizes the threadgroup: every block that takes
    a thread axis gets its threads within ``32 * num_warps``, and lanes walk
    the rest of its extent."""

    def _run(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | dict[str, object],
    ) -> tuple[object, str, str]:
        """Run *kernel* at *config*; a dict overrides its default config."""
        if isinstance(config, dict):
            default = kernel.bind(args).config_spec.default_config().config
            config = helion.Config(**{**default, **config})
        return _run_with_msl(kernel, args, config)

    def test_num_warps_sizes_the_launch(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        cases = (
            (budget_add_2d, (x, x * 2), {}, x.cpu() * 3),
            (
                budget_row_sum,
                (rows := torch.randn(16, 1000, device=DEVICE),),
                {},
                rows.cpu().double().sum(-1).float(),
            ),
            (
                budget_row_sum,
                (wide := torch.randn(16, 3000, device=DEVICE),),
                {"reduction_loops": [1024]},
                wide.cpu().double().sum(-1).float(),
            ),
            (budget_nested_pair, (x,), {}, x.cpu() * 2),
        )
        for kernel, args, config, expected in cases:
            for num_warps in (1, 4, 8):
                with self.subTest(
                    kernel=kernel.name, config=config, num_warps=num_warps
                ):
                    out, code, _ = self._run(
                        kernel, args, {**config, "num_warps": num_warps}
                    )
                    self.assertEqual(math.prod(_launch(code)), 32 * num_warps)
                    torch.testing.assert_close(
                        out.cpu(), expected, rtol=1e-4, atol=1e-3
                    )

    def test_rolled_chunk_steps_by_its_threads(self) -> None:
        """A rolled chunk wider than the threads it gets steps by them: each
        thread visits every span-th element of the chunk."""
        for kernel, reference in (
            (budget_row_sum, lambda x: x.double().sum(-1).float()),
            (budget_row_argmax, lambda x: x.argmax(-1)),
        ):
            for n in (3000, 2049):
                with self.subTest(kernel=kernel.name, n=n):
                    x = torch.randn(8, n, device=DEVICE)
                    out, code, _ = self._run(
                        kernel, (x,), {"reduction_loops": [1024], "num_warps": 4}
                    )
                    self.assertEqual(_launch(code), (128, 1, 1))
                    self.assertIn("_REDUCTION_BLOCK_1 = 128", code)
                    torch.testing.assert_close(
                        out.cpu(), reference(x.cpu()), rtol=1e-4, atol=1e-3
                    )

    def test_rolled_chunk_keeps_two_threads(self) -> None:
        """A looped reduction needs two threads, so a chunk of 4 at
        num_warps=1 steps by 2 beside a 16-thread tile."""
        config = {"block_sizes": [32], "reduction_loops": [4], "num_warps": 1}
        for n in (3000, 2049, 64):
            with self.subTest(n=n):
                x = torch.randn(40, n, device=DEVICE)
                out, code, _ = self._run(budget_row_sum, (x,), config)
                self.assertEqual(_launch(code), (2, 16, 1))
                self.assertIn("_REDUCTION_BLOCK_1 = 2", code)
                torch.testing.assert_close(
                    out.cpu(), x.cpu().double().sum(-1).float(), rtol=1e-4, atol=1e-3
                )

    def test_combined_dims_reserve_their_share(self) -> None:
        """A dimension Scalar code combines reserves its share of a full
        threadgroup scaled to the budget: a 128-wide scalar-dot K takes 16 of
        128 threads and leaves the tiles theirs, and all 128 of 1024."""
        c = torch.randn(64, 128, device=DEVICE, dtype=torch.float16)
        p = torch.randn(64, 128, device=DEVICE, dtype=torch.float16)
        x = torch.randn(64, 32, device=DEVICE)
        rowsum = x.cpu().double().sum(-1, keepdim=True)
        mm = c.cpu().double() @ p.cpu().double().T
        for num_warps, launch, span in ((4, (16, 8, 1), 16), (32, (128, 8, 1), 128)):
            with self.subTest(num_warps=num_warps):
                out, code, msl = self._run(
                    budget_dot16_k_tiles, (c, p, x), {"num_warps": num_warps}
                )
                self.assertEqual(_launch(code), launch)
                self.assertIn(f"tid, 0, {span}, _TG_SIZE_X", msl)
                torch.testing.assert_close(
                    out.cpu(),
                    (mm.half().double() + rowsum).float(),
                    rtol=1e-3,
                    atol=2e-2,
                )
            with self.subTest(num_warps=num_warps, dot="float32"):
                out, code, _ = self._run(
                    budget_dot_k_tiles,
                    (c.float(), p.float(), x),
                    {"num_warps": num_warps},
                )
                self.assertIn("_helion_mpp_run(", code)
                torch.testing.assert_close(
                    out.cpu(), (mm + rowsum).float(), rtol=1e-4, atol=1e-3
                )
        # A 1024-wide row reserves the whole 128-thread budget, as at 1024.
        x = torch.randn(4, 1024, device=DEVICE)
        out, code, _ = self._run(
            budget_row_sum, (x,), helion.Config(block_sizes=[1], num_warps=4)
        )
        self.assertEqual(_launch(code), (128, 1, 1))
        torch.testing.assert_close(
            out.cpu(), x.cpu().double().sum(-1).float(), rtol=1e-4, atol=1e-3
        )

    def test_nested_literal_tile_shrinks_to_the_budget(self) -> None:
        """gdn_fwd_h's literal 64-row chunk tile, nested in the V tile, takes
        only what K's reservation leaves, so K folds at most 64 lanes."""
        args = (*_gdn_args(128, 256, 64, torch.float32), _GDN_CHUNK)
        expected = _gdn_reference(*args[:4])
        for config in (
            helion.Config(block_sizes=[16], num_warps=4),
            helion.Config(block_sizes=[32], num_threads=[1], num_warps=4),
        ):
            with self.subTest(config=config):
                device_args = tuple(
                    a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in args
                )
                out, code, msl = _run_with_msl(gdn_fwd_h_metal, device_args, config)
                self.assertEqual(_launch(code), (32, 4, 1))
                torch.testing.assert_close(
                    out.cpu().double(), expected, rtol=1e-4, atol=1e-4
                )

    def test_explicit_nested_count_starving_the_fold_stays_rejected(self) -> None:
        """An explicit count on a nested tile is charged in full, so at
        num_warps=1 the scalar dot's 128-wide K is left one thread, past the
        64-lane fold cap.  MPP takes the float32 dot's whole K in one op."""
        config = helion.Config(
            block_sizes=[16, 16, 32], num_threads=[0, 0, 32], num_warps=1
        )
        args = (
            torch.randn(64, 128, dtype=torch.float16),
            torch.randn(64, 128, dtype=torch.float16),
            torch.randn(64, 32),
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"persistent reduction over 128 elements would need 128 lanes per "
            r"thread on 1 thread \(max 64\)",
        ):
            budget_dot16_k_tiles.bind(args).to_code(config)
        c, p, x = (
            torch.randn(64, 128, device=DEVICE),
            torch.randn(64, 128, device=DEVICE),
            torch.randn(64, 32, device=DEVICE),
        )
        out, code, _ = self._run(budget_dot_k_tiles, (c, p, x), config)
        self.assertIn("_helion_mpp_run(", code)
        expected = c.cpu().double() @ p.cpu().double().T
        expected += x.cpu().double().sum(-1, keepdim=True)
        torch.testing.assert_close(out.cpu(), expected.float(), rtol=1e-4, atol=1e-3)

    def test_after_loop_reduction_tile_takes_one_thread(self) -> None:
        """kl_div's pass, reduced after its loop, goes to one thread when the
        tiles overflow the budget: it needs one element per thread or one
        thread."""
        y_pred = torch.randn(16, 512).log_softmax(-1)
        y_true = torch.randn(16, 512).softmax(-1)
        kernel = helion.kernel(kl_div_forward.fn, backend="metal", static_shapes=True)
        out, code, _ = self._run(
            kernel, (y_pred.to(DEVICE), y_true.to(DEVICE)), {"num_warps": 4}
        )
        self.assertEqual(_launch(code), (16, 1, 1))
        expected = torch.nn.functional.kl_div(
            y_pred.double(), y_true.double(), reduction="batchmean"
        )
        torch.testing.assert_close(out.cpu().double(), expected, rtol=1e-4, atol=1e-5)

    def test_tile_reduced_after_its_loop_shrinks_to_one_thread(self) -> None:
        """With no reduction dim to reserve for, the shrink to the budget
        takes a tile reduced after its loop from one thread per element
        straight to one thread; an explicit partial count stays rejected."""
        x = torch.randn(8, 3000, device=DEVICE)
        for num_warps in (1, 4, 32):
            with self.subTest(num_warps=num_warps):
                config = helion.Config(block_sizes=[4096, 1], num_warps=num_warps)
                out, code, _ = _run_with_msl(sum_after_loop, (x,), config)
                self.assertEqual(_launch(code), (1, 1, 1))
                torch.testing.assert_close(
                    out.cpu(), x.cpu().double().sum(-1).float(), rtol=1e-4, atol=1e-3
                )
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"its block of 128 is spread over 16 threads, not one element per thread",
        ):
            sum_after_loop.bind((x.cpu(),)).to_code(
                helion.Config(block_sizes=[128, 1], num_threads=[16, 1], num_warps=4)
            )

    def test_lane_floor(self) -> None:
        """The launch grows past 32 * num_warps to the fewest threads at
        which each reduction's lane form exists: layer_norm_bwd's persistent
        9216-wide reduction folds at most 64 lanes, so 256 threads."""
        x = torch.randn(4, 9216)
        args = (
            torch.randn(4, 9216),
            x,
            x.mean(-1),
            torch.rsqrt(x.var(-1, unbiased=False) + 1e-5),
            torch.randn(9216),
        )
        kernel = helion.kernel(layer_norm_bwd.fn, backend="metal", static_shapes=True)
        device_args = tuple(a.to(DEVICE) for a in args)
        (grad_x, grad_w, grad_b), code, _ = self._run(kernel, device_args, {})
        self.assertEqual(math.prod(_launch(code)), 256)
        dy, xd, _, _, w = (a.double() for a in args)
        xd.requires_grad_(True)
        w.requires_grad_(True)
        bias = torch.zeros_like(w, requires_grad=True)
        torch.nn.functional.layer_norm(xd, (9216,), w, bias, eps=1e-5).backward(dy)
        for out, expected in ((grad_x, xd.grad), (grad_w, w.grad), (grad_b, bias.grad)):
            torch.testing.assert_close(
                out.cpu().double(), expected, rtol=1e-3, atol=1e-3
            )

    def test_argmax_keeps_a_thread_per_element(self) -> None:
        """An argmax over a persistent row has no lane form, so its row keeps
        a thread per element at num_warps=4: 1024 for a 1000-wide row, and
        the whole 128-thread budget for a 100-wide one."""
        for n, launch in ((1000, (1024, 1, 1)), (100, (128, 1, 1))):
            with self.subTest(n=n):
                x = torch.randn(16, n, device=DEVICE)
                out, code, _ = self._run(budget_row_argmax, (x,), {"num_warps": 4})
                self.assertEqual(_launch(code), launch)
                torch.testing.assert_close(out.cpu(), x.cpu().argmax(-1))

    def test_rejections_name_the_budget(self) -> None:
        c = torch.randn(64, 128)
        p = torch.randn(48, 128)
        r = torch.randn(64, 512)
        for config, note in (
            (
                helion.Config(block_sizes=[16, 16], reduction_loops=[8], num_warps=1),
                (
                    r"(?m)reduction loop over chunks of 8 elements needs 8 threads but "
                    r"only 1 are available; .* \(threadgroup budget 32 threads "
                    r"from num_warps=1\)$"
                ),
            ),
            (
                helion.Config(block_sizes=[16, 16], num_warps=4),
                (
                    r"(?m)in a lane loop beside other reduction dims \[\d+\] "
                    r"\(threadgroup budget 128 threads from num_warps=4\)$"
                ),
            ),
            (
                helion.Config(block_sizes=[16, 16], num_warps=32),
                r"(?m)in a lane loop beside other reduction dims \[\d+\]$",
            ),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(exc.BackendUnsupported, note),
            ):
                budget_dot_beside_row_sum.bind((c, p, r)).to_code(config)

    def test_lane_copied_bfloat_conversions_round_in_integers(self) -> None:
        """Metal lowers a float -> bfloat cast that a lane copy repeats as a
        float -> half cast: gdn_fwd_h's scalar contraction at num_warps=1
        folds K over 32 lanes, and its bfloat state came back with half's bit
        patterns.  Lane copies round in integer arithmetic instead."""
        args = (*_gdn_args(256, 256, 32, torch.bfloat16), _GDN_CHUNK)
        device_args = tuple(
            a.to(DEVICE) if isinstance(a, torch.Tensor) else a for a in args
        )
        out, code, msl = _run_with_msl(
            gdn_fwd_h_transposed,
            device_args,
            helion.Config(block_sizes=[32], num_warps=1),
        )
        self.assertEqual(_launch(code), (8, 4, 1))
        self.assertRegex(msl, r"helion_bf16_cast\([^;]*__lane\d+")
        self.assertNotRegex(msl, r"static_cast<bfloat>\([^;]*__lane\d+")
        torch.testing.assert_close(
            out.cpu().double(), _gdn_reference(*args[:4]), rtol=2e-2, atol=2e-2
        )

    def test_lane_copied_bfloat_conversion_values(self) -> None:
        """The integer rounding is torch's: a tie goes to even, a NaN stays a
        NaN.  (A subnormal can't be observed here: the GPU returned 0 for
        one, as float arithmetic around the cast may flush it.)"""
        values = torch.tensor(
            [1 + 2.0**-7, -(1 + 3 * 2.0**-7), float("nan"), 0.75, 70000.0]
        ).to(torch.bfloat16)
        x = torch.zeros(len(values), 256, dtype=torch.bfloat16)
        for row, value in enumerate(values):
            x[row, 17 * row] = value
        out, _, msl = self._run(
            budget_bf16_round_trip_sum, (x.to(DEVICE),), {"num_warps": 1}
        )
        self.assertRegex(msl, r"helion_bf16_cast\([^;]*__lane\d+")
        expected = (values.float() * 3.0).to(torch.bfloat16).float()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)

    def test_root_tile_ignores_its_block_size_argument(self) -> None:
        """A 1-D root ``hl.tile`` with a literal or registered block size is
        indexed as a tunable tile is, not ``offsets * block_size``."""
        rows = torch.randn(40, 1536, device=DEVICE)
        for config in (
            {"reduction_loops": [None], "num_warps": 4},
            {"reduction_loops": [None], "num_warps": 32},
            {"reduction_loops": [64], "num_warps": 4},
            {"reduction_loops": [1024], "num_warps": 4},
        ):
            for m in (40, 37):
                with self.subTest(kernel="softmax_sum", config=config, m=m):
                    x = rows[:m].contiguous()
                    out, code, _ = self._run(literal_tile_softmax_sum, (x,), config)
                    self.assertIn("indices_0 = offsets_0", code)
                    xd = x.cpu().double()
                    expected = (xd - xd.amax(-1, keepdim=True)).exp().sum(-1)
                    torch.testing.assert_close(
                        out.cpu().double(), expected, rtol=1e-4, atol=1e-4
                    )
        values = torch.randn(43, device=DEVICE)
        for m in (40, 37, 8, 1):
            with self.subTest(kernel="literal_tile_scale", m=m):
                x = values[:m].contiguous()
                out, code, _ = self._run(literal_tile_scale, (x,), {})
                self.assertNotIn("offsets_0 * 8", code)
                torch.testing.assert_close(out.cpu(), x.cpu() * 2)
        with self.subTest(kernel="from_begin"):
            out, code, _ = self._run(literal_tile_from_begin, (values,), {})
            self.assertIn("indices_0 = 8 + offsets_0", code)
            expected = values.cpu().clone()
            expected[:8] = 0
            expected[8:] *= 2
            torch.testing.assert_close(out.cpu(), expected)
        x = values[:40].contiguous()
        with self.subTest(kernel="tile_id"):
            out, _, _ = self._run(literal_tile_id, (x,), {})
            torch.testing.assert_close(
                out.cpu(), x.cpu().view(5, 8).sum(-1), rtol=1e-5, atol=1e-5
            )
        with self.subTest(kernel="tile_index"):
            out, _, _ = self._run(literal_tile_index, (x,), {})
            torch.testing.assert_close(out.cpu(), x.cpu() + torch.arange(40))
        table = torch.randn(40, 64, device=DEVICE)
        for m in (40, 37):
            for block in (8, 32):
                for num_warps in (1, 4):
                    config = helion.Config(block_sizes=[block], num_warps=num_warps)
                    with self.subTest(kernel="registered_scale", m=m, config=config):
                        x = values[:m].contiguous()
                        out, code, _ = self._run(registered_tile_scale, (x,), config)
                        self.assertNotIn("offsets_0 * _BLOCK_SIZE_0", code)
                        torch.testing.assert_close(out.cpu(), x.cpu() * 2)
                    with self.subTest(kernel="registered_row_sum", m=m, config=config):
                        x = table[:m].contiguous()
                        out, code, _ = self._run(registered_tile_row_sum, (x,), config)
                        self.assertNotIn("offsets_0 * _BLOCK_SIZE_0", code)
                        torch.testing.assert_close(
                            out.cpu(),
                            x.cpu().double().sum(-1).float(),
                            rtol=1e-4,
                            atol=1e-4,
                        )


@helion.kernel(backend="metal", static_shapes=True)
def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Worker w of ``get_num_sm`` workers computes output tiles w, w + W, ..."""
    num_workers = helion.runtime.get_num_sm(a.device)
    block_m = hl.register_block_size(32, 128)
    block_n = hl.register_block_size(32, 128)
    m, k = a.shape
    _, n = b.shape
    out = torch.empty(m, n, dtype=a.dtype, device=a.device)
    for worker in hl.grid(num_workers):
        m_tiles = (m + block_m - 1) // block_m
        tiles = m_tiles * ((n + block_n - 1) // block_n)
        for step in hl.grid((tiles + num_workers - 1) // num_workers):
            t = step * num_workers + worker
            if t < tiles:
                rows = (t % m_tiles) * block_m + hl.arange(block_m)
                cols = (t // m_tiles) * block_n + hl.arange(block_n)
                acc = hl.zeros([block_m, block_n], dtype=torch.float32)
                for tk in hl.tile(k):
                    x = hl.load(a, [rows, tk.index], extra_mask=(rows < m)[:, None])
                    y = hl.load(b, [tk.index, cols], extra_mask=(cols < n)[None, :])
                    acc = torch.addmm(acc, x, y)
                hl.store(
                    out,
                    [rows, cols],
                    acc.to(out.dtype),
                    extra_mask=(rows < m)[:, None] & (cols < n)[None, :],
                )
    return out


@_requires_darwin
class TestMetalPersistentGrid(unittest.TestCase):
    """On MPS, ``get_num_sm`` is the GPU core count."""

    def test_grid_is_the_core_count(self) -> None:
        """``reserved_sms`` reserves cores; 0 or less reserves none, and at
        least one is left."""
        device = torch.device(DEVICE)
        cores = torch.backends.mps.get_core_count()
        with self.subTest(reserved_sms="default"):
            self.assertEqual(helion.runtime.get_num_sm(device), cores)
        for reserved, workers in (
            (-1, cores),
            (0, cores),
            (1, cores - 1),
            (2, cores - 2),
            (cores, 1),
        ):
            with self.subTest(reserved_sms=reserved):
                self.assertEqual(
                    helion.runtime.get_num_sm(device, reserved_sms=reserved), workers
                )

    def test_persistent_kernel_launches_the_grid(self) -> None:
        """Workers that loop over several tiles, non-dividing tails, and more
        workers than tiles."""
        workers = torch.backends.mps.get_core_count()
        config = helion.Config(block_sizes=[32, 32, 32], num_warps=4)
        for m, k, n in ((300, 64, 300), (200, 96, 136), (48, 40, 40)):
            with self.subTest(m=m, k=k, n=n):
                a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
                b = torch.randn(k, n, device=DEVICE, dtype=torch.float16)
                out, code, _ = _run_with_msl(persistent_matmul, (a, b), config)
                self.assertIn(
                    f"_launcher(_helion_persistent_matmul, ({workers},)", code
                )
                torch.testing.assert_close(
                    out.cpu().float(),
                    a.cpu().float() @ b.cpu().float(),
                    rtol=1e-2,
                    atol=1e-2,
                )


@helion.kernel(backend="metal", static_shapes=True)
def tile_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(0), dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty(m, n, dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_after_barrier(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = out[tile] * 2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_to_tensor_extent(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(n, block_size=32):
        out[tile] = x[tile] + 1
    return out


@_requires_darwin
class TestMetalPersistentPid(unittest.TestCase):
    """``pid_type`` persistent_blocked and persistent_interleaved launch
    ``_NUM_SM * num_sm_multiplier`` threadgroups that loop over a plain
    ``hl.tile`` loop's tiles, where ``_NUM_SM`` is the GPU core count and the
    multiplier defaults to 4."""

    def test_workers_loop_over_tiles(self) -> None:
        """More tiles than workers, so each worker takes several, and fewer, so
        some take none; both shapes have tails."""
        cores = torch.backends.mps.get_core_count()
        workers = helion.runtime.get_num_sm(torch.device(DEVICE))
        rows = 16 * (3 * workers // 13 + 1) + 5
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            for multiplier in (1, 3):
                config = helion.Config(
                    block_sizes=[16, 16],
                    pid_type=pid_type,
                    num_sm_multiplier=multiplier,
                )
                grid = "_NUM_SM" if multiplier == 1 else f"_NUM_SM * {multiplier}"
                for m, n in ((rows, 200), (40, 50)):
                    with self.subTest(pid_type=pid_type, multiplier=multiplier, m=m):
                        x = torch.randn(m, n, device=DEVICE)
                        out, code, msl = _run_with_msl(tile_scale, (x,), config)
                        self.assertIn(f"_launcher(_helion_tile_scale, ({grid},)", code)
                        self.assertIn(f"constexpr int _NUM_SM = {cores};", msl)
                        self.assertIn("for (int virtual_pid = ", msl)
                        torch.testing.assert_close(out, x * 2)

    def test_default_multiplier_is_four_per_core(self) -> None:
        """A persistent config naming no multiplier launches 4 threadgroups per
        core, and the search's persistent neighbours of the default config
        start there."""
        cores = torch.backends.mps.get_core_count()
        x = torch.randn(70, 50, device=DEVICE)
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            config = helion.Config(block_sizes=[16, 16], pid_type=pid_type)
            with self.subTest(pid_type=pid_type):
                out, code, msl = _run_with_msl(tile_scale, (x,), config)
                self.assertIn("_launcher(_helion_tile_scale, (_NUM_SM * 4,)", code)
                self.assertIn(f"constexpr int _NUM_SM = {cores};", msl)
                torch.testing.assert_close(out, x * 2)
        spec = tile_scale.bind((x,)).config_spec
        generation = ConfigGeneration(spec)
        base = generation.flatten(spec.default_config())
        multipliers = set()
        for index, fragment in enumerate(generation.flat_spec):
            for value in fragment.pattern_neighbors(base[index]):
                neighbor = [*base]
                neighbor[index] = value
                config = generation.unflatten(neighbor).config
                if config.get("pid_type", "flat").startswith("persistent"):
                    multipliers.add(config["num_sm_multiplier"])
        with self.subTest("persistent neighbours of the default config"):
            self.assertEqual(multipliers, {4})

    def test_threadgroup_memory_is_reused_across_tiles(self) -> None:
        """A row reduction and an MPP matmul, each worker taking at least four
        tiles, repeated so a missing barrier between tiles would show."""
        workers = helion.runtime.get_num_sm(torch.device(DEVICE))
        x = torch.randn(4 * workers + 3, 256, device=DEVICE)
        a = torch.randn(32 * workers - 5, 40, device=DEVICE, dtype=torch.float16)
        b = torch.randn(40, 125, device=DEVICE, dtype=torch.float16)
        cases = (
            (
                tile_row_sum,
                (x,),
                [1],
                "helion_red::group_sum",
                x.cpu().double().sum(-1),
            ),
            (
                tile_matmul,
                (a, b),
                [32, 32, 32],
                "matmul2d",
                a.cpu().float() @ b.cpu().float(),
            ),
        )
        for kernel, args, block_sizes, feature, expected in cases:
            for pid_type in ("persistent_blocked", "persistent_interleaved"):
                config = helion.Config(
                    block_sizes=block_sizes, pid_type=pid_type, num_sm_multiplier=1
                )
                with self.subTest(kernel=kernel.name, pid_type=pid_type):
                    _, _, msl = _run_with_msl(kernel, args, config)
                    self.assertIn("for (int virtual_pid = ", msl)
                    self.assertIn(feature, msl)
                    compiled = kernel.bind(args).compile_config(config)
                    for _ in range(50):
                        torch.testing.assert_close(
                            compiled(*args).cpu().to(expected.dtype),
                            expected,
                            rtol=1e-2,
                            atol=1e-2,
                        )

    def test_default_config_stays_flat(self) -> None:
        """The default launches one threadgroup per tile, and normalizing it
        drops the multiplier and the ``range_*`` keys Metal doesn't support."""
        x = torch.randn(70, 50, device=DEVICE)
        bound = tile_scale.bind((x,))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config.get("pid_type"), "flat")
        self.assertEqual(
            [
                k
                for k in config.config
                if k.startswith("range_") or k == "num_sm_multiplier"
            ],
            [],
        )
        self.assertNotIn("virtual_pid", bound.to_code(config))

    def test_barrier_stays_rejected(self) -> None:
        """``hl.barrier`` forces persistent_blocked, and Metal has no grid-wide
        barrier to lower it to."""
        x = torch.randn(100, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendImplementationMissing, "codegen for API function barrier"
        ):
            bound = tile_after_barrier.bind((x,))
            bound.to_code(bound.config_spec.default_config())

    def test_extent_from_a_tensor_stays_rejected(self) -> None:
        """A root tile over a tensor's value takes a persistent pid_type, whose
        loop would read the extent with a scalar load Metal lacks."""
        x = torch.randn(100, device=DEVICE)
        n = torch.tensor(77, device=DEVICE)
        bound = tile_to_tensor_extent.bind((x, n))
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            config = helion.Config(
                **{**bound.config_spec.default_config().config, "pid_type": pid_type}
            )
            with (
                self.subTest(pid_type=pid_type),
                self.assertRaisesRegex(exc.BackendUnsupported, "scalar load"),
            ):
                bound.to_code(config)


@helion.kernel(backend="metal", static_shapes=True)
def tile_count_grid(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    bm = hl.register_block_size(8, 32)
    nt = (m + bm - 1) // bm
    out = torch.zeros_like(x)
    for i, j in hl.grid([nt, k]):
        rows = i * bm + hl.arange(bm)
        v = hl.load(x, [rows, j], extra_mask=rows < m)
        hl.store(out, [rows, j], v * 2, extra_mask=rows < m)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_count_device_loop(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    bk = hl.register_block_size(8, 16)
    nk = (k + bk - 1) // bk
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for j in hl.grid(nk):
            cols = j * bk + hl.arange(bk)
            v = hl.load(x, [tm.index, cols], extra_mask=(cols < k)[None, :])
            hl.store(out, [tm.index, cols], v * 3, extra_mask=(cols < k)[None, :])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_count_grid_plus_int(x: torch.Tensor, off: int) -> torch.Tensor:
    m, k = x.size()
    bm = hl.register_block_size(8, 32)
    nt = (m + bm - 1) // bm
    out = torch.zeros_like(x)
    for i, j in hl.grid([nt, k]):
        rows = i * bm + hl.arange(bm)
        v = hl.load(x, [rows, j], extra_mask=rows < m)
        hl.store(out, [rows, j], v + off, extra_mask=rows < m)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scale_by_float(x: torch.Tensor, s: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * s
    return out


@_requires_darwin
class TestMetalConfigFixedScalars(unittest.TestCase):
    """A runtime scalar whose free symbols are all block sizes, such as a tile
    count under static shapes, is fixed by the config, so it is baked into the
    shader as a ``constexpr int`` like a block size.  Other scalars stay bound
    by value as ``float``."""

    def test_tile_count_as_grid_extent(self) -> None:
        """Tail tiles at every block size: 50 rows divide by none of them."""
        x = torch.randn(50, 40, device=DEVICE)
        for bm in (8, 16, 32):
            with self.subTest(bm=bm):
                out, code, msl = _run_with_msl(
                    tile_count_grid, (x,), helion.Config(block_sizes=[bm])
                )
                self.assertIn(f"constexpr int nt = {(50 + bm - 1) // bm};", msl)
                self.assertNotIn("_by_value", code)
                torch.testing.assert_close(out, x * 2)

    def test_tile_count_as_device_loop_extent(self) -> None:
        x = torch.randn(50, 40, device=DEVICE)
        for bk in (8, 16):
            with self.subTest(bk=bk):
                out, _, msl = _run_with_msl(
                    tile_count_device_loop, (x,), helion.Config(block_sizes=[bk, 16])
                )
                self.assertIn(f"constexpr int nk = {(40 + bk - 1) // bk};", msl)
                torch.testing.assert_close(out, x * 3)

    def test_user_scalars_stay_by_value(self) -> None:
        """An int argument beside a tile count, and a float argument."""
        x = torch.randn(50, 40, device=DEVICE)
        for kernel, args, config, expected, by_value in (
            (
                tile_count_grid_plus_int,
                (x, 7),
                helion.Config(block_sizes=[16]),
                x + 7,
                ("_by_value=(3,)", "constant long& off [[buffer"),
            ),
            (
                scale_by_float,
                (x, 1.5),
                helion.Config(block_sizes=[16, 16]),
                x * 1.5,
                ("_by_value=(2,)", "constant float& s [[buffer"),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                out, code, msl = _run_with_msl(kernel, args, config)
                self.assertIn(by_value[0], code)
                self.assertIn(by_value[1], msl)
                torch.testing.assert_close(out, expected)

    def test_dynamic_shape_tile_count_stays_by_value(self) -> None:
        """Without static shapes the tile count also depends on ``m``, which
        the config doesn't fix."""
        kernel = helion.kernel(tile_count_grid.fn, backend="metal", static_shapes=False)
        x = torch.randn(50, 40, device=DEVICE)
        out, code, msl = _run_with_msl(kernel, (x,), helion.Config(block_sizes=[16]))
        self.assertRegex(code, r"_launcher\(.*\bnt\b.*_by_value=\(\d+, \d+\)")
        self.assertIn("constant long& nt [[buffer", msl)
        torch.testing.assert_close(out, x * 2)

    def test_baked_int_must_fit_int32(self) -> None:
        for value in (2**31 - 1, -(2**31)):
            with self.subTest(value=value):
                self.assertEqual(
                    _constexpr_decl("n", value), f"constexpr int n = {value};"
                )
        for value in (2**31, -(2**31) - 1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(exc.BackendUnsupported, "does not fit an int"),
            ):
                _constexpr_decl("n", value)


@helion.kernel(backend="metal", static_shapes=True)
def tile_count_truediv(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    bm = hl.register_block_size(8, 32)
    nt = (m + bm - 1) // bm
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tk in hl.tile(k):
            out[tm, tk] = x[tm, tk] * (nt / 3)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_count_grid_truediv(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    bm = hl.register_block_size(8, 32)
    nt = (m + bm - 1) // bm
    out = torch.zeros_like(x)
    for i, j in hl.grid([nt, k]):
        rows = i * bm + hl.arange(bm)
        v = hl.load(x, [rows, j], extra_mask=rows < m)
        hl.store(out, [rows, j], v * (nt / 3), extra_mask=rows < m)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def block_size_truediv(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    bm = hl.register_block_size(8, 32)
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tk in hl.tile(k):
            out[tm, tk] = x[tm, tk] * (bm / 3)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def int_arg_truediv(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * (n / 3)
    return out


@_requires_darwin
class TestMetalIntTrueDiv(unittest.TestCase):
    """Python's ``/`` on integers divides in floating point; MSL's ``/`` on two
    ints truncates, so the operands are cast first."""

    def test_tile_count_truediv(self) -> None:
        """The tile count inlined as a ``floor_divide`` of block sizes, beside
        a tile loop and beside a grid over the same tile count."""
        x = torch.randn(50, 40, device=DEVICE)
        for kernel in (tile_count_truediv, tile_count_grid_truediv):
            for bm in (8, 16, 32):
                config = helion.Config(
                    block_sizes=[bm, 16] if kernel is tile_count_truediv else [bm]
                )
                with self.subTest(kernel=kernel.name, bm=bm):
                    out, _, msl = _run_with_msl(kernel, (x,), config)
                    self.assertIn("static_cast<float>(c10::metal::floor_divide(", msl)
                    torch.testing.assert_close(out, x * ((50 + bm - 1) // bm / 3))

    def test_block_size_truediv(self) -> None:
        x = torch.randn(50, 40, device=DEVICE)
        for bm in (8, 16):
            with self.subTest(bm=bm):
                out, _, msl = _run_with_msl(
                    block_size_truediv, (x,), helion.Config(block_sizes=[bm, 16])
                )
                self.assertIn("static_cast<float>(_BLOCK_SIZE_0) / 3", msl)
                torch.testing.assert_close(out, x * (bm / 3))

    def test_int_argument_truediv(self) -> None:
        """The argument is bound as a float already; the cast keeps it one."""
        x = torch.randn(50, 40, device=DEVICE)
        out, _, msl = _run_with_msl(
            int_arg_truediv, (x, 4), helion.Config(block_sizes=[16, 16])
        )
        self.assertIn("static_cast<float>(n) / 3", msl)
        torch.testing.assert_close(out, x * (4 / 3))

    def test_lane_invariant_cast_runs_once(self) -> None:
        """A cast is a call in device code; lane-loop placement still moves
        the lane-invariant quotient ahead of the lane loop."""
        x = torch.randn(50, 40, device=DEVICE)
        code = int_arg_truediv.bind((x, 4)).to_code(helion.Config(block_sizes=[16, 16]))
        quotient = code.index("truediv = static_cast(float, n) / 3")
        self.assertLess(quotient, code.index("for lane_0 in range("))


if __name__ == "__main__":
    unittest.main()
