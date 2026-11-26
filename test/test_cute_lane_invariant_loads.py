"""Lane-invariant integer loads leave the synthetic lane loops.

A persistent row reduction runs its row body once per synthetic lane, and the
lane split re-runs in its accumulate sweep everything the reduced value
depends on.  A dropout seed read from a tensor (``seeds[54]``) was loaded in
every lane, and LLVM kept the Philox key schedule derived from it in the
loop too.  ``hoist_lane_invariant_loads`` moves such loads in front of the
loop.  The kernels render on CPU, the pass's conditions are checked on
hand-written lane loops, and the ``cuda`` test runs the kernel against its
form with the pass off, bit for bit.
"""

from __future__ import annotations

import ast
import functools
import textwrap
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute import hoist_lane_invariant_loads as pass_module
from helion._compiler.cute._ast_pass_utils import loop_extent
from helion._compiler.cute.hoist_lane_invariant_loads import hoist_lane_invariant_loads
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", cute_rng_stream="word0"
)
def _seeded_dropout_rows(
    x: torch.Tensor,
    seeds: torch.Tensor,
    scales: torch.Tensor,
    seed_index: hl.constexpr,
    seed_index_1: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        # Two seed words folded into the draw's seed: one Philox chain keeps
        # the GPU test's two compiles short.
        seed = seeds[seed_index] ^ seeds[seed_index_1]
        scale = scales[0]
        keep = hl.rand([row, n], seed=seed) > 0.1
        dropped = keep.to(torch.float32) * x[row, :].to(torch.float32) * scale
        keep_out[row, :] = keep
        out[row, :] = dropped - torch.mean(dropped, dim=-1, keepdim=True)
    return keep_out, out


def _args(device: str | torch.device = "cpu") -> tuple[object, ...]:
    # 96 columns over 32 threads: a masked row walked in rolled lanes.
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((19, 96), generator=generator).to(device, torch.bfloat16)
    seeds = torch.randint(0, 2**31, (8,), generator=generator).to(device)
    scales = torch.tensor([1.25], device=device)
    return x, seeds, scales, 3, 5


def _config() -> helion.Config:
    return helion.Config(block_sizes=[8], num_threads=[8, 32])


def _render() -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(_seeded_dropout_rows, _args()).to_code(_config())


def _kernel_def(code: str) -> ast.FunctionDef:
    (kernel,) = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    ]
    return kernel


def _tensor_loads(node: ast.AST, tensor: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr == "iterator"
        and isinstance(child.value, ast.Name)
        and child.value.id == tensor
    )


def _inside_lane_loops(kernel: ast.FunctionDef, tensor: str) -> int:
    return sum(
        _tensor_loads(loop, tensor)
        for loop in kernel.body
        if isinstance(loop, ast.For)
        and ast.unparse(loop.target).startswith("synthetic_lane_")
    )


def test_seed_loads_leave_the_lane_loop() -> None:
    kernel = _kernel_def(_render())
    lane_loops = [
        node
        for node in kernel.body
        if isinstance(node, ast.For)
        and ast.unparse(node.target).startswith("synthetic_lane_")
    ]
    assert len(lane_loops) == 2, ast.unparse(kernel)
    assert all(loop_extent(loop, kinds=("range",), minimum=2) for loop in lane_loops)
    # Both seeds are loaded once, ahead of the accumulate sweep that draws.
    assert _tensor_loads(kernel, "seeds") == 2, ast.unparse(kernel)
    assert _inside_lane_loops(kernel, "seeds") == 0, ast.unparse(kernel)
    # The float scale stays where the row reads it: moving it could let LLVM
    # hoist a product with it and contract the multiply-adds differently.
    assert _inside_lane_loops(kernel, "scales") == 1, ast.unparse(kernel)


_TENSOR_DTYPES = {
    "seeds": "cutlass.Int64",
    "table": "cutlass.Int32",
    "scales": "cutlass.Float32",
    "out": "cutlass.Float32",
}


def _hoist(
    loop_body: str,
    *,
    lanes: int = 4,
    lane_loop: bool = True,
    disjoint: tuple[frozenset[str], ...] = (),
    renames: dict[str, str] | None = None,
    before: str = "",
    integer_names: tuple[str, ...] = (),
) -> str:
    source = (
        "cache = cute.make_rmem_tensor(4, cutlass.Float32)\n"
        "idx = cute.make_rmem_tensor(4, cutlass.Int32)\n"
        + before
        + f"for lane in range({lanes}):\n"
        + textwrap.indent(textwrap.dedent(loop_body), "    ")
    )
    body = ast.parse(source).body
    loop = body[-1]
    if lane_loop:
        setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane")
    result = hoist_lane_invariant_loads(
        body,
        tensor_dtypes=_TENSOR_DTYPES,
        proven_disjoint_tensor_pairs=disjoint,
        rename_groups=renames,
        integer_names=integer_names,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


_SEED = "seed = (seeds.iterator + cutlass.Int32(54) * seeds.layout.stride[0]).load()\n"
_USE = "cache[lane] = cutlass.Float32(seed & 4294967295)\n"


def _moved(code: str, name: str) -> bool:
    """Whether ``name``'s definition sits ahead of the lane loop."""
    statements = ast.parse(code).body
    loop = next(i for i, s in enumerate(statements) if isinstance(s, ast.For))
    return any(
        isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == name
        for s in statements[:loop]
    )


def test_an_integer_load_and_the_loads_it_addresses_move() -> None:
    code = _hoist(
        "row = (table.iterator + 7).load()\n"
        "seed = (seeds.iterator + row).load()\n"
        "cache[lane] = cutlass.Float32(seed + cutlass.Int64(lane))\n"
    )
    assert _moved(code, "row"), code
    assert _moved(code, "seed"), code
    # In their order, and the per-lane statement stays.
    assert code.index("row =") < code.index("seed =") < code.index("for lane"), code
    assert "cache[lane] =" in code.split("for lane", 1)[1], code


@pytest.mark.parametrize(
    ("loop_body", "kwargs"),
    [
        pytest.param(_SEED + _USE, {}, id="seed"),
        pytest.param(
            _SEED + _USE + "(out.iterator + cutlass.Int32(lane)).store(0.0)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="store-to-a-proven-disjoint-tensor",
        ),
        pytest.param(
            _SEED
            + "s = (seeds.iterator + 1).load() if seed > 0 else cutlass.Int64(0)\n"
            + _USE,
            {},
            id="masked-by-a-moved-value",
        ),
        pytest.param(
            _SEED
            + "keep = (seed & 1) == 0\n"
            + "cache[lane] = cutlass.Float32(lane) if keep else 0.0\n",
            {},
            id="integer-test-made-invariant",
        ),
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32((table.iterator + seed * "
            + "cutlass.Int32(offset // width) + cutlass.Int32(lane)).load())\n",
            {},
            id="address-made-invariant",
        ),
        pytest.param(
            "seed = cute.arch.load(seeds.iterator + 54, cutlass.Int64)\n" + _USE,
            {},
            id="scalar-arch-load",
        ),
        pytest.param(
            "seed = (seeds.iterator + operator.mul(cutlass.Int32(54), "
            "seeds.layout.stride[0])).load()\n" + _USE,
            {},
            id="integer-operator-call",
        ),
        pytest.param(
            _SEED
            + "(out.iterator + cutlass.Int32(lane)).store(cutlass.Float32(seed))\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="invariant-value-stored-where-nothing-loads-it",
        ),
        pytest.param(
            _SEED
            + "key = seed & 255\n"
            + "key = key ^ 3\n"
            + "cache[lane] = cutlass.Float32(key + cutlass.Int64(lane))\n",
            {},
            id="integer-name-written-twice",
        ),
        pytest.param(
            "seed = (seeds.iterator + 54).load() ^ salt\n" + _USE,
            {"integer_names": ("salt",)},
            id="integer-parameter-operand",
        ),
        # What a register or a tensor gives back is what the loop stored
        # there: a lane's value keeps its product in the loop.
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(lane)\n"
            + "cache[lane] = cache[lane] * f + cutlass.Float32(lane)\n"
            + "idx[lane] = cutlass.Int32(seed & 255)\n",
            {},
            id="register-round-trip-of-a-lane-value",
        ),
        pytest.param(
            _SEED
            + "(out.iterator + cutlass.Int32(lane)).store(cutlass.Float32(lane))\n"
            + "cache[lane] = (out.iterator + cutlass.Int32(lane)).load() * f + "
            + "cutlass.Float32(seed & 255)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="tensor-round-trip-of-a-lane-value",
        ),
        # The element read is invariant without the seed: the store that
        # reads the seed is not invariant.
        pytest.param(
            _SEED
            + "cache[0] = f\n"
            + "cache[lane] = cutlass.Float32(seed + cutlass.Int64(lane))\n"
            + "idx[lane] = cutlass.Int32(cache[1] * f + cutlass.Float32(lane))\n",
            {},
            id="register-read-invariant-without-the-moved-value",
        ),
    ],
)
def test_moves(loop_body: str, kwargs: dict[str, object]) -> None:
    code = _hoist(loop_body, **kwargs)  # pyrefly: ignore[bad-argument-type]
    assert _moved(code, "seed"), code


@pytest.mark.parametrize(
    ("loop_body", "kwargs"),
    [
        pytest.param(
            "seed = (scales.iterator + 0).load()\n" + _USE, {}, id="float-tensor"
        ),
        pytest.param(
            "seed = (seeds.iterator + 54).load() ^ salt\n" + _USE,
            {
                "integer_names": ("salt",),
                "before": "salt, other = (cutlass.Float32(0.5), 1)\n",
            },
            id="integer-parameter-rebound-by-a-tuple-target",
        ),
        pytest.param(
            "seed = cutlass.Float32((seeds.iterator + 0).load())\n" + _USE,
            {},
            id="float-arithmetic",
        ),
        pytest.param(
            "seed = (seeds.iterator + cutlass.Int32(lane)).load()\n" + _USE,
            {},
            id="lane-dependent",
        ),
        pytest.param(
            _SEED + _USE + "(seeds.iterator + cutlass.Int32(lane)).store(seed)\n",
            {},
            id="stored-in-the-loop",
        ),
        pytest.param(
            _SEED + _USE + "(out.iterator + cutlass.Int32(lane)).store(0.0)\n",
            {},
            id="store-that-may-alias",
        ),
        pytest.param(_SEED + _USE + "cute.arch.sync_threads()\n", {}, id="barrier"),
        pytest.param(
            "if lane == 3:\n    break\n" + _SEED + _USE, {}, id="lane-cut-short"
        ),
        pytest.param(
            "cache[lane] = cutlass.Float32(seed)\n" + _SEED, {}, id="read-before"
        ),
        pytest.param(_SEED + "seed = seed + 1\n" + _USE, {}, id="written-twice"),
        pytest.param(
            _SEED + "other = cutlass.Int64(lane)\n" + _USE,
            {"renames": {"other": "seed"}},
            id="rename-group-written",
        ),
        pytest.param(
            _SEED + _USE + "table[lane] = 0\n", {}, id="non-register-subscript-store"
        ),
        pytest.param(
            _SEED
            + "scale = cutlass.Float32(seed) * 0.5\n"
            + "cache[lane] = cutlass.Float32(lane) + scale\n",
            {},
            id="float-product-made-invariant",
        ),
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(lane) + cutlass.Float32(seed) * 0.5\n",
            {},
            id="float-subexpression-made-invariant",
        ),
        pytest.param(_SEED + _USE, {"lane_loop": False}, id="not-a-lane-loop"),
        pytest.param(_SEED + _USE, {"lanes": 1}, id="one-lane"),
        pytest.param(
            "idx[lane] = cutlass.Int32(lane) * 3\n"
            "seed = (seeds.iterator + idx[0]).load()\n" + _USE,
            {},
            id="address-from-a-register-the-loop-writes",
        ),
        pytest.param(
            "idx[lane] = cutlass.Int32(lane)\n"
            "seed = (seeds.iterator + 0).load() if idx[0] > 0 else "
            "cutlass.Int64(0)\n" + _USE,
            {},
            id="predicate-from-a-register-the-loop-writes",
        ),
        pytest.param(
            "seed = (seeds.iterator + idx[0]).load()\n"
            "idx[0] = cutlass.Int32(seed)\n" + _USE,
            {},
            id="register-carried-across-lanes",
        ),
        pytest.param(
            "seed = (seeds.iterator + 0).load() + _cute_load_v8b_if(out.iterator"
            " + 0, True, ir.VectorType.get([8], cutlass.Uint8.mlir_type))[0]\n"
            + _USE
            + "(out.iterator + cutlass.Int32(lane)).store(0.0)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="masked-packet-load-of-a-stored-tensor",
        ),
        pytest.param(
            "seed = (seeds.iterator + 0).load() + cutlass.Int64("
            "_cute_load_l1_l2_evict_first(scales.iterator + 0, "
            "ir.VectorType.get([4], cutlass.Float32.mlir_type))[0])\n" + _USE,
            {},
            id="hinted-float-packet-load",
        ),
        pytest.param(
            "seed = cutlass.Int32((seeds.iterator + 0).load())"
            ".bitcast(cutlass.Float32)\n"
            "cache[lane] = seed * 3 + cutlass.Float32(lane)\n",
            {},
            id="float-bitcast",
        ),
        pytest.param(
            "seed = (seeds.iterator + 0).load() * s\n"
            "cache[lane] = seed + cutlass.Float32(lane)\n",
            {},
            id="float-operand-from-outside-the-loop",
        ),
        pytest.param(
            "seed = (seeds.iterator + cutlass.Int32(p + h)).load()\n" + _USE,
            {},
            id="float-names-in-an-integer-conversion",
        ),
        pytest.param(
            _SEED
            + "scale = operator.mul(cutlass.Float32(seed), f)\n"
            + "cache[lane] = cutlass.Float32(lane) + scale\n",
            {},
            id="float-product-call-made-invariant",
        ),
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(lane) + "
            + "cute.math.log(cutlass.Float32(seed))\n",
            {},
            id="math-call-made-invariant",
        ),
        # LLVM forwards the stored element to the load that follows it, so
        # the product becomes loop-invariant.
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(seed)\n"
            + "cache[lane] = cache[lane] * f + cutlass.Float32(lane)\n",
            {},
            id="product-of-a-register-round-trip",
        ),
        pytest.param(
            _SEED
            + "(out.iterator + cutlass.Int32(lane)).store(cutlass.Float32(seed))\n"
            + "cache[lane] = (out.iterator + cutlass.Int32(lane)).load() * f + "
            + "cutlass.Float32(lane)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="product-of-a-tensor-round-trip",
        ),
        # The address a store goes through is not the value it stores.
        pytest.param(
            _SEED
            + "cute.arch.store(out.iterator + cutlass.Int32(lane), "
            + "cutlass.Float32(seed))\n"
            + "cache[lane] = (out.iterator + cutlass.Int32(lane)).load() * f + "
            + "cutlass.Float32(lane)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="product-of-an-arch-store-round-trip",
        ),
        pytest.param(
            _SEED
            + "cache[lane], idx[lane] = (cutlass.Float32(seed), cutlass.Int32(lane))\n"
            + "cache[lane] = cache[lane] * f + cutlass.Float32(lane)\n",
            {},
            id="product-of-a-tuple-target-round-trip",
        ),
        # The float guard over-approximates invariance: a name written more
        # than once, by a tuple target or by an augmented assignment counts
        # as invariant when all of its writes could be.
        pytest.param(
            _SEED
            + "t = cutlass.Float32(seed)\n"
            + "u = t * f\n"
            + "cache[lane] = u + cutlass.Float32(lane)\n",
            {"renames": {"u": "t"}},
            id="float-product-in-a-rename-group",
        ),
        pytest.param(
            _SEED
            + "t = cutlass.Float32(seed)\n"
            + "t = t * f\n"
            + "cache[lane] = t + cutlass.Float32(lane)\n",
            {},
            id="float-product-reassigned",
        ),
        pytest.param(
            _SEED
            + "t = cutlass.Float32(seed)\n"
            + "t *= f\n"
            + "cache[lane] = t + cutlass.Float32(lane)\n",
            {},
            id="float-product-augmented",
        ),
        pytest.param(
            _SEED
            + "if flag:\n"
            + "    t = cutlass.Float32(seed)\n"
            + "else:\n"
            + "    t = cutlass.Float32(seed + 1)\n"
            + "cache[lane] = t * f + cutlass.Float32(lane)\n",
            {},
            id="float-product-of-a-branch-merge",
        ),
        pytest.param(
            _SEED
            + "t, w = (cutlass.Float32(seed), cutlass.Float32(lane))\n"
            + "cache[lane] = t * f + w\n",
            {},
            id="float-product-of-a-tuple-target",
        ),
        pytest.param(
            _SEED
            + "t = cutlass.Float32(seed)\n"
            + "product = t * f\n"
            + "t = cutlass.Float32(lane)\n"
            + "cache[lane] = product + t\n",
            {},
            id="float-product-of-a-name-written-again",
        ),
        pytest.param(
            _SEED
            + "t = cutlass.Float32(seed)\n"
            + "product = t * f\n"
            + "t_1 = cutlass.Float32(lane)\n"
            + "cache[lane] = product + t_1\n",
            {"renames": {"t_1": "t"}},
            id="float-product-of-a-rename-group-written-again",
        ),
        pytest.param(
            _SEED
            + "if seed > 0:\n"
            + "    t = a\n"
            + "else:\n"
            + "    t = b\n"
            + "cache[lane] = t * f + cutlass.Float32(lane)\n",
            {},
            id="float-product-of-a-branch-the-seed-decides",
        ),
        pytest.param(
            "(seeds.iterator + cutlass.Int32(lane)).store(cutlass.Int64(lane))\n"
            "seed = (table.iterator + cutlass.Int32(seeds[2])).load()\n" + _USE,
            {"disjoint": (frozenset({"seeds", "table"}),)},
            id="subscript-of-a-stored-tensor",
        ),
        pytest.param(
            "cache[lane] = cutlass.Float32(lane)\n"
            "seed = (seeds.iterator + cutlass.Int32(alias[0])).load()\n"
            "idx[lane] = cutlass.Int32(seed)\n",
            {"before": "alias = cache\n"},
            id="subscript-of-a-register-under-another-name",
        ),
        pytest.param(
            _SEED
            + "(out.iterator + 0).store(cutlass.Float32(seed))\n"
            + "cache[lane] = out[0] * f + cutlass.Float32(lane)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="invariant-value-read-back-through-a-tensor-subscript",
        ),
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(seed)\n"
            + "(out.iterator + cutlass.Int32(lane)).store("
            + "alias[0] * f + cutlass.Float32(lane))\n",
            {"before": "alias = cache\n", "disjoint": (frozenset({"seeds", "out"}),)},
            id="invariant-value-read-back-through-a-register-alias",
        ),
        # A load gives back what the loop stored at its address, whatever the
        # address: a value invariant already, and one a branch on the seed
        # picks.
        pytest.param(
            _SEED
            + "tmp[0] = f\n"
            + "cache[lane] = tmp[0] * cutlass.Float32(seed) + cutlass.Float32(lane)\n",
            {"before": "tmp = cute.make_rmem_tensor(1, cutlass.Float32)\n"},
            id="invariant-value-forwarded-through-a-register",
        ),
        pytest.param(
            _SEED
            + "(out.iterator + lane).store(f)\n"
            + "cache[lane] = (out.iterator + lane).load() * cutlass.Float32(seed)"
            + " + cutlass.Float32(lane)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="invariant-value-forwarded-through-a-lane-address",
        ),
        pytest.param(
            _SEED
            + "if seed > 0:\n"
            + "    cache[lane] = a\n"
            + "else:\n"
            + "    cache[lane] = b\n"
            + "cache[lane] = cache[lane] * f + cutlass.Float32(lane)\n",
            {},
            id="register-value-a-branch-on-the-seed-picks",
        ),
        # An element the stores do not reach holds what it held before the
        # loop.
        pytest.param(
            _SEED
            + "(out.iterator + cutlass.Int32(lane)).store(cutlass.Float32(lane))\n"
            + "cache[lane] = (out.iterator + 7).load() * cutlass.Float32(seed) + "
            + "cutlass.Float32(lane)\n",
            {"disjoint": (frozenset({"seeds", "out"}),)},
            id="product-of-a-tensor-element-no-lane-stores",
        ),
        pytest.param(
            _SEED
            + "cache[lane] = cutlass.Float32(lane)\n"
            + "idx[lane] = cutlass.Int32(cache[3] * cutlass.Float32(seed) + "
            + "cutlass.Float32(lane))\n",
            {"lanes": 2},
            id="product-of-a-register-element-no-lane-stores",
        ),
        # A pointer held in a name reaches memory that the iterators in the
        # address do not name.
        pytest.param(
            "(out.iterator + cutlass.Int32(lane)).store(cutlass.Float32(lane))\n"
            "seed = cutlass.Int64((pointer + (seeds.iterator + 0).load()).load())\n",
            {
                "before": "pointer = out.iterator + 0\n",
                "disjoint": (frozenset({"seeds", "out"}),),
            },
            id="load-through-a-pointer-held-in-a-name",
        ),
        pytest.param(
            "(pointer + (table.iterator + cutlass.Int32(lane)).load()).store("
            "cutlass.Int64(lane))\n" + _SEED + _USE,
            {
                "before": "pointer = seeds.iterator + 0\n",
                "disjoint": (frozenset({"seeds", "table"}),),
            },
            id="store-through-a-pointer-held-in-a-name",
        ),
        pytest.param(
            "(seeds.iterator + cutlass.Int32(lane)).store(cutlass.Int64(lane))\n"
            "seed = (table.iterator + cutlass.Int32(operator.getitem(seeds, 2)))"
            ".load()\n" + _USE,
            {"disjoint": (frozenset({"seeds", "table"}),)},
            id="operator-getitem-of-a-stored-tensor",
        ),
    ],
)
def test_stays(loop_body: str, kwargs: dict[str, object]) -> None:
    code = _hoist(loop_body, **kwargs)  # pyrefly: ignore[bad-argument-type]
    assert not _moved(code, "seed"), code


def test_arithmetic_inside_an_integer_conversion_needs_integer_operands() -> None:
    """A load addressed through ``cutlass.Int32(row // _BLOCK_SIZE_0)`` moves
    once ``row`` (assigned an integer) and the constexpr ``_BLOCK_SIZE_0`` are
    known integers; without the constexpr's type, the division could be a
    float one, and the load stays."""
    source = textwrap.dedent(
        """\
        cache = cute.make_rmem_tensor(4, cutlass.Float32)
        row = cutlass.Int32(cute.arch.block_idx()[0])
        for lane in range(4):
            seed = (seeds.iterator + cutlass.Int32(row // _BLOCK_SIZE_0)).load()
            cache[lane] = cutlass.Float32(seed & 4294967295)
        """
    )
    for integer_names, moves in (({"_BLOCK_SIZE_0"}, True), ((), False)):
        body = ast.parse(source).body
        setattr(body[2], HELION_LANE_LOOP_VAR_ATTR, "lane")
        result = hoist_lane_invariant_loads(
            body, tensor_dtypes=_TENSOR_DTYPES, integer_names=integer_names
        )
        code = ast.unparse(ast.Module(body=result, type_ignores=[]))
        assert _moved(code, "seed") is moves, code


def test_a_load_leaves_nested_lane_loops() -> None:
    body = ast.parse(
        textwrap.dedent(
            """\
            cache = cute.make_rmem_tensor(4, cutlass.Float32)
            for outer in range(2):
                for lane in range(4):
                    seed = (seeds.iterator + 0).load()
                    cache[lane] = cutlass.Float32(seed + cutlass.Int64(outer))
            """
        )
    ).body
    outer = body[1]
    assert isinstance(outer, ast.For)
    setattr(outer, HELION_LANE_LOOP_VAR_ATTR, "outer")
    setattr(outer.body[0], HELION_LANE_LOOP_VAR_ATTR, "lane")
    result = hoist_lane_invariant_loads(body, tensor_dtypes=_TENSOR_DTYPES)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert code.index("seed =") < code.index("for outer"), code


def _tabulate(
    inner_body: str,
    *,
    inner: str = "cutlass.range_constexpr(4)",
    outer_lead: str = "",
    outer_extra: str = "",
    after: str = "",
    base: str = "cutlass.Int32(cute.arch.thread_idx()[0]) * 2",
    disjoint: tuple[frozenset[str], ...] = (),
    rename_groups: dict[str, str] | None = None,
) -> str:
    """The pass over a thread's two rows (an outer constexpr loop) whose inner
    lane loop runs ``inner_body``, followed by ``after``."""
    source = (
        "cache = cute.make_rmem_tensor(4, cutlass.Float32)\n"
        "for row in cutlass.range_constexpr(2):\n"
        + textwrap.indent(outer_lead, "    ")
        + f"    base = {base}\n"
        f"    for lane in {inner}:\n"
        + textwrap.indent(inner_body, "        ")
        + textwrap.indent(outer_extra, "    ")
        + after
    )
    result = hoist_lane_invariant_loads(
        ast.parse(source).body,
        tensor_dtypes=_TENSOR_DTYPES,
        proven_disjoint_tensor_pairs=disjoint,
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


_KEY_LOADS = (
    "key = base + cutlass.Int32(lane)\n"
    "doc = _cute_scalar_policy_load(seeds.iterator + key, cutlass.Int64, 'last')\n"
    "cache[lane] = cutlass.Float32(doc + cutlass.Int64(row))\n"
)


def test_a_key_load_both_rows_repeat_moves_to_a_table() -> None:
    """A thread's two rows read the same per-key value: it is read once per
    key, before the row loop, and each row indexes the table."""
    code = _tabulate(_KEY_LOADS)
    before, loop = code.split("for row in", 1)
    assert (
        "doc_lanes = []" in before
        and "doc_by_lane = _cute_scalar_policy_load(seeds.iterator + key_by_lane"
        in before
    ), code
    # The table reads the row loop's leading definition of its base.
    assert before.index("base =") < before.index("for lane_table in"), code
    assert "doc = doc_lanes[lane]" in loop and "_cute_scalar_policy_load" not in loop, (
        code
    )
    # The per-row use stays.
    assert "cache[lane] = cutlass.Float32(doc + cutlass.Int64(row))" in loop, code


def test_a_table_reads_every_spelling_of_a_chain_name() -> None:
    """``key_1`` and ``key`` spell one variable (a rename group): the table
    loop's load reads the key it computed for its lane, not the value the
    group held before the row loop.  The table loop walks the lanes with a
    variable of its own."""
    code = _tabulate(
        _KEY_LOADS.replace("key = base", "key_1 = base"),
        rename_groups={"key_1": "key"},
    )
    before, _ = code.split("for row in", 1)
    assert "for lane_table in cutlass.range_constexpr(4)" in before, code
    assert "key_1_by_lane = base + cutlass.Int32(lane_table)" in before, code
    assert "seeds.iterator + key_1_by_lane" in before, code


def test_a_chain_name_read_after_the_row_loop_keeps_its_binding() -> None:
    """``last_key = key`` after the row loop reads the key of the last lane
    of the last row: ``key`` keeps a table, and the rows still bind it."""
    code = _tabulate(_KEY_LOADS, after="last_key = key\n")
    _, loop = code.split("for row in", 1)
    assert "key = key_lanes[lane]" in loop, code


def test_an_address_temporary_of_the_table_loads_gets_no_table() -> None:
    """``key`` addresses the key load and nothing else in the rows: read from
    a table, it would be dead, so it gets none, leaves the row loop and
    takes no value of the budget (32 lanes of doc alone fit)."""
    code = _tabulate(_KEY_LOADS, inner="cutlass.range_constexpr(32)")
    before, loop = code.split("for row in", 1)
    assert "doc_lanes = []" in before and "key_lanes" not in code, code
    assert "key_by_lane = base + cutlass.Int32(lane_table)" in before, code
    assert "key = " not in loop and "doc = doc_lanes[lane]" in loop, code


@pytest.mark.parametrize(
    ("inner_body", "kwargs"),
    [
        pytest.param(
            _KEY_LOADS.replace(
                "base + cutlass.Int32(lane)", "base + cutlass.Int32(lane + row)"
            ),
            {},
            id="row-dependent",
        ),
        pytest.param(_KEY_LOADS, {"inner": "range(4)"}, id="inner-loop-not-unrolled"),
        pytest.param(
            _KEY_LOADS,
            {"outer_extra": "(seeds.iterator + base).store(cutlass.Int64(0))\n"},
            id="row-loop-stores-the-loaded-tensor",
        ),
        pytest.param(
            _KEY_LOADS.replace(
                "cutlass.Int64, 'last'", "cutlass.Float32, 'last'"
            ).replace("seeds.iterator", "scales.iterator"),
            {},
            id="float-load",
        ),
        pytest.param(
            _KEY_LOADS, {"outer_extra": "cute.arch.sync_threads()\n"}, id="barrier"
        ),
        # The row loop reads base before its leading definition: a copy of
        # the definition in front of the loop would change what it reads.
        pytest.param(
            _KEY_LOADS,
            {"outer_lead": "first = base + cutlass.Int32(row)\n"},
            id="leading-definition-read-before",
        ),
        # The same with an inner loop that gets no table in between: the
        # outer body's read history outlives the inner loop's own.
        pytest.param(
            _KEY_LOADS,
            {
                "outer_lead": (
                    "first = base + cutlass.Int32(row)\n"
                    "for j in cutlass.range_constexpr(2):\n"
                    "    cache[j] = cutlass.Float32(first)\n"
                )
            },
            id="leading-definition-read-before-an-inner-loop",
        ),
        # The leading definition reads table, which out (the row loop's
        # store) may alias: copied in front of the loop, every row would read
        # the value the first row read.
        pytest.param(
            _KEY_LOADS,
            {
                "base": "cutlass.Int32(table[0]) * 2",
                "outer_extra": "(out.iterator + base).store(cutlass.Float32(0.0))\n",
                "disjoint": (frozenset({"seeds", "out"}),),
            },
            id="leading-definition-reads-what-the-row-loop-may-store",
        ),
        # A float product of the loaded value: from a table, both rows would
        # compute it from one value (the existing move's integer-only rule).
        pytest.param(
            _KEY_LOADS.replace(
                "cache[lane] = cutlass.Float32(doc + cutlass.Int64(row))",
                "cache[lane] = cutlass.Float32(doc) * cutlass.Float32(0.7) + cache[lane]",
            ),
            {},
            id="float-arithmetic-of-the-loaded-value",
        ),
        # 64 lanes of doc: 64 values live across the row loop.
        pytest.param(
            _KEY_LOADS,
            {"inner": "cutlass.range_constexpr(64)"},
            id="past-the-register-cap",
        ),
    ],
)
def test_a_key_load_stays(inner_body: str, kwargs: dict[str, object]) -> None:
    code = _tabulate(inner_body, **kwargs)  # pyrefly: ignore[bad-argument-type]
    assert "_lanes" not in code, code


def test_a_loop_around_tables_still_gives_up_its_invariants() -> None:
    """The tables' appends write no memory: the lane loop around a row loop
    whose key loads moved to a table still moves its own invariant load out."""
    body = ast.parse(
        textwrap.dedent(
            """\
            cache = cute.make_rmem_tensor(4, cutlass.Float32)
            for v in range(2):
                t = (seeds.iterator + 3).load()
                for row in cutlass.range_constexpr(2):
                    base = cutlass.Int32(cute.arch.thread_idx()[0]) * 2
                    for lane in cutlass.range_constexpr(4):
                        key = base + cutlass.Int32(lane)
                        doc = (seeds.iterator + key + t).load()
                        cache[lane] = cutlass.Float32(doc + cutlass.Int64(row + v))
            """
        )
    ).body
    setattr(body[1], HELION_LANE_LOOP_VAR_ATTR, "v")
    result = hoist_lane_invariant_loads(body, tensor_dtypes=_TENSOR_DTYPES)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "doc_lanes" in code, code
    assert code.index("t = ") < code.index("for v"), code


@pytest.mark.parametrize(
    ("row_store", "after_rows"),
    [
        # A later sibling loop reads the trace lists before the row loop's
        # tables reach the body: the lists still include the tables.
        (
            "cache[lane] = cutlass.Float32(doc + cutlass.Int64(row + v))",
            (
                "for j in cutlass.range_constexpr(2):\n"
                "    cache[j] = cache[j] + cutlass.Float32(v)\n"
            ),
        ),
        # The rows read doc from its table: still an integer.
        (
            (
                "cache[lane] = cutlass.Float32(doc + cutlass.Int64(row))"
                " + cutlass.Float32(v)"
            ),
            "",
        ),
    ],
    ids=["a-later-sibling-loop", "integer-table-reads"],
)
def test_a_loop_around_tables_moves_its_invariant_load(
    row_store: str, after_rows: str
) -> None:
    body = ast.parse(
        "cache = cute.make_rmem_tensor(4, cutlass.Float32)\n"
        "for v in range(2):\n"
        "    t = (seeds.iterator + 3).load()\n"
        "    for row in cutlass.range_constexpr(2):\n"
        "        base = cutlass.Int32(cute.arch.thread_idx()[0]) * 2\n"
        "        for lane in cutlass.range_constexpr(4):\n"
        "            key = base + cutlass.Int32(lane)\n"
        "            doc = (seeds.iterator + key + t).load()\n"
        f"            {row_store}\n" + textwrap.indent(after_rows, "    ")
    ).body
    setattr(body[1], HELION_LANE_LOOP_VAR_ATTR, "v")
    result = hoist_lane_invariant_loads(body, tensor_dtypes=_TENSOR_DTYPES)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "doc_lanes" in code, code
    assert code.index("t = ") < code.index("for v"), code


def test_a_float_of_table_values_keeps_the_feeding_load_in_its_loop() -> None:
    """The rows scale doc by the row as floats: the tables are made (the
    product differs between rows), but within ``for v`` the product would
    not change, so ``t``, which feeds the tables (an append writes its
    list), stays in ``for v``."""
    body = ast.parse(
        textwrap.dedent(
            """\
            cache = cute.make_rmem_tensor(4, cutlass.Float32)
            for v in range(2):
                t = (seeds.iterator + 3).load()
                for row in cutlass.range_constexpr(2):
                    base = cutlass.Int32(cute.arch.thread_idx()[0]) * 2
                    for lane in cutlass.range_constexpr(4):
                        key = base + cutlass.Int32(lane)
                        doc = (seeds.iterator + key + t).load()
                        cache[lane] = cutlass.Float32(doc) * cutlass.Float32(row) + cache[lane]
            """
        )
    ).body
    setattr(body[1], HELION_LANE_LOOP_VAR_ATTR, "v")
    result = hoist_lane_invariant_loads(body, tensor_dtypes=_TENSOR_DTYPES)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "doc_lanes" in code, code
    assert code.index("for v") < code.index("t = "), code


def test_a_float_the_tables_would_make_invariant_keeps_the_loads() -> None:
    """The row loop scales the key loads' values as floats: read from a
    table, ``cutlass.Float32(doc) * 0.5`` would be one product for both rows,
    which LLVM may round apart from the add it contracts with, so the loads
    stay in the rows, and so does the lane loop's own load ``t`` that feeds
    them (the existing move's integer-only rule)."""
    body = ast.parse(
        textwrap.dedent(
            """\
            cache = cute.make_rmem_tensor(4, cutlass.Float32)
            for v in range(2):
                t = (seeds.iterator + 3).load()
                for row in cutlass.range_constexpr(2):
                    base = cutlass.Int32(cute.arch.thread_idx()[0]) * 2
                    for lane in cutlass.range_constexpr(4):
                        key = base + cutlass.Int32(lane)
                        doc = (seeds.iterator + key + t).load()
                        cache[lane] = cutlass.Float32(doc) * cutlass.Float32(0.5) + cache[lane]
            """
        )
    ).body
    setattr(body[1], HELION_LANE_LOOP_VAR_ATTR, "v")
    result = hoist_lane_invariant_loads(body, tensor_dtypes=_TENSOR_DTYPES)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "_lanes" not in code, code
    assert code.index("for v") < code.index("t = "), code


@functools.cache
def _compiled(hoisted: bool) -> tuple[str, tuple[torch.Tensor, torch.Tensor]]:
    args = _args(DEVICE)
    bound = _seeded_dropout_rows._bind_isolated(args)
    if hoisted:
        return bound.to_code(_config()), bound.compile_config(_config())(*args)
    with patch.object(
        pass_module, "hoist_lane_invariant_loads", lambda body, **kwargs: body
    ):
        return bound.to_code(_config()), bound.compile_config(_config())(*args)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_hoisted_seed_loads_match_the_loads_per_lane() -> None:
    hoisted_code, (keep, out) = _compiled(True)
    per_lane_code, (per_lane_keep, per_lane_out) = _compiled(False)
    assert _inside_lane_loops(_kernel_def(hoisted_code), "seeds") == 0
    assert _inside_lane_loops(_kernel_def(per_lane_code), "seeds") == 2
    assert torch.equal(keep.view(torch.uint8), per_lane_keep.view(torch.uint8))
    assert torch.equal(out.view(torch.int32), per_lane_out.view(torch.int32))
    assert 0.8 < keep.float().mean().item() < 0.95
