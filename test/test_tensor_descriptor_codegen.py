"""A reduction dim's block size and extent in generated Triton code, rendered on CPU."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import _has_fbtriton
from helion._testing import cpu_render_target
from helion._testing import skipIfFbTriton
from helion._testing import skipIfRefEagerMark
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cuda_metadata_trace
from helion._testing.cute_codegen import load_generated_module
from helion._testing.row_iota_kernels import half_row_fill
from helion._testing.row_iota_kernels import onehot_row_update
from helion._testing.triton_codegen import interpret_triton_kernel
from helion._utils import triton_is_available
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = [
    pytest.mark.cpu_only,
    skipIfRefEagerMark("renders generated Triton code"),
    skipUnlessBackends(["triton"]),
    pytest.mark.skipif(
        not triton_is_available(), reason="Triton is required for source generation"
    ),
]

# fbtriton's frontend passes store_reduce to create_descriptor_store, which
# its interpreter's InterpreterBuilder does not take (a TypeError).
_FBTRITON_INTERPRETED_DESCRIPTOR_STORE = (
    "fbtriton's interpreter cannot store through a descriptor"
)


@pytest.fixture
def sm90_descriptors() -> Iterator[None]:
    """Render for an sm90 target, which offers tensor descriptors, in a
    process that must not initialize CUDA."""
    from triton.backends.compiler import GPUTarget

    with (
        cpu_render_target(capability=(9, 0), num_sm=132),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        # The TileIR fork consults this driver even in its CPU interpreter.
        # These kernels explicitly target Triton's CUDA descriptor semantics.
        patch("triton.language.semantic.driver", create=True) as semantic_driver,
    ):
        semantic_driver.active.get_current_target.return_value = GPUTarget(
            "cuda", 90, 32
        )
        yield


@pytest.mark.parametrize(
    ("cols", "reduction_loop", "static_shapes", "block", "count"),
    [
        (200, 64, True, "_REDUCTION_BLOCK_1", 2),
        (256, 64, True, "_REDUCTION_BLOCK_1", 2),
        (256, None, True, "_RDIM_SIZE_1", 2),
        # With dynamic shapes the iota lowers to the size's name; out, whose
        # row stride is the dynamic size, keeps pointer indexing.
        (200, 64, False, "_REDUCTION_BLOCK_1", 1),
    ],
)
def test_descriptor_block_of_a_row_beside_its_iota(
    sm90_descriptors: None,
    cols: int,
    reduction_loop: int | None,
    static_shapes: bool,
    block: str,
    count: int,
) -> None:
    # The iota's arithmetic, carrying the row's block size symbol, lowered
    # to the row's extent, which used to stand for that symbol in the
    # descriptors' block shapes: [_BLOCK_SIZE_0, 200] fails Triton's
    # power-of-two check, [_BLOCK_SIZE_0, 256] a looped reduction's chunk of
    # 64, and [_BLOCK_SIZE_0, cols] (a runtime size) Triton's constexpr block
    # shape.  A row stride of cols + 8 elements keeps x's descriptor aligned.
    x = torch.rand(64, cols + 8, dtype=torch.bfloat16)[:, :cols]
    label = torch.randint(0, cols, (64,))
    kernel = helion.kernel(
        onehot_row_update,
        backend="triton",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((x, label))
    default = bound.config_spec.default_config()
    config = helion.Config(
        **(
            default.config
            | {
                "block_sizes": [16],
                "reduction_loops": [reduction_loop],
                "indexing": ["tensor_descriptor"] * len(default.config["indexing"]),
            }
        )
    )
    code = bound.to_triton_code(config)
    descriptors = [
        line.strip() for line in code.splitlines() if "make_tensor_descriptor" in line
    ]
    assert len(descriptors) == count, descriptors
    for line in descriptors:
        assert line.endswith(f"[_BLOCK_SIZE_0, {block}])"), line


@pytest.mark.parametrize("reduction_loop", [None, 64])
@pytest.mark.parametrize("static_shapes", [True, False])
def test_nested_loop_reads_a_scalar_of_the_rows_extent(
    static_shapes: bool, reduction_loop: int | None
) -> None:
    # The interpreter's skip stays limited to values that are exactly the
    # block size symbol: n = row.size(1) // 2 still reaches the nested loop
    # as a capture of the extent's half, the constant 100 with static shapes
    # and, with dynamic ones, the outer floordiv its record names.
    x = torch.rand(64, 200)
    kernel = helion.kernel(
        half_row_fill,
        backend="triton",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((x,))
    config = helion.Config(
        **(
            bound.config_spec.default_config().config
            | {"block_sizes": [16], "reduction_loops": [reduction_loop]}
        )
    )
    code = bound.to_triton_code(config)
    assert not re.search(r"(_RDIM_SIZE_1|_REDUCTION_BLOCK_1)( // |, )2\b", code), code
    if static_shapes:
        assert "tl.range(0, tl.cast(100, tl.int32))" in code, code
    else:
        assert "tl.cast(floordiv, tl.float32)" in code, code


@pytest.mark.parametrize(
    "indices", ["first", "first_copy", "last", "reused", "shifted", "paired"]
)
@skipIfFbTriton(_FBTRITON_INTERPRETED_DESCRIPTOR_STORE)
def test_scalar_and_vector_indices_share_a_broadcast_axis(
    sm90_descriptors: None, tmp_path: Path, indices: str
) -> None:
    @helion.kernel(backend="triton", static_shapes=True)
    def gather(x, plane, rows, out, indices: hl.constexpr):
        for row in hl.tile(rows.size(0), block_size=4):
            if indices == "first":
                out[row, :] = hl.load(x, [plane, rows[row], slice(None)])
            elif indices == "first_copy":
                p = plane
                q = p
                out[row, :] = hl.load(x, [q, rows[row], slice(None)])
            elif indices == "last":
                out[row, :] = x[rows[row], plane[()], :]
            elif indices == "paired":
                out[row, :] = x[plane[()], row.index + 1, rows[row], :]
            elif indices == "reused":
                s = plane[()]
                out[row, :] = x[s, s, rows[row], :]
            else:
                out[row, :] = x[plane[()], row.index + 1, :]

    rows = torch.tensor([3, 1, 4, 0, 2])
    plane = torch.tensor(2)
    if indices == "paired":
        x = torch.arange(3 * 8 * 8 * 8).reshape(3, 8, 8, 8)
        expected = x[plane, torch.arange(rows.numel()) + 1, rows]
    elif indices == "reused":
        x = torch.arange(3 * 5 * 8 * 8).reshape(3, 5, 8, 8)
        expected = x[plane, plane, rows]
    else:
        x = torch.arange(5 * 8 * 8).reshape(5, 8, 8)
        if indices == "shifted":
            expected = x[plane, 1 : 1 + rows.numel()]
        else:
            expected = x[plane, rows] if indices.startswith("first") else x[rows, plane]
    out = torch.empty_like(expected)
    args = (x, plane, rows, out, indices)
    code = cpu_bind(gather, args).to_code(
        helion.Config(block_sizes=[], indexing="tensor_descriptor")
    )
    path = tmp_path / "gather.py"
    path.write_text(code)
    load_generated_module(path).gather(*args, _launcher=interpret_triton_kernel)
    torch.testing.assert_close(out, expected)


def test_correlated_shifted_indices_do_not_form_a_descriptor(
    sm90_descriptors: None, tmp_path: Path
) -> None:
    @helion.kernel(backend="triton", static_shapes=True)
    def diagonal(x, plane, out):
        for row in hl.tile(out.size(0), block_size=4):
            s = plane[()]
            out[row, :] = x[s, row.index + 1, row.index + 2, :]

    x = torch.arange(3 * 32 * 32 * 8).reshape(3, 32, 32, 8)
    plane = torch.tensor(2)
    out = torch.empty(8, 8, dtype=x.dtype)
    args = (x, plane, out)
    with (
        cuda_metadata_trace(),
        patch("helion._compat._supports_host_tensor_descriptor", return_value=True),
    ):
        source = cpu_bind(diagonal, args, capability=(9, 0)).to_code(
            helion.Config(
                block_sizes=[],
                indexing="tensor_descriptor",
                host_tensor_descriptors=True,
            )
        )
    assert "tl.load(x +" in source
    assert "_helion_tensor_descriptor(out," in source
    if _has_fbtriton():
        pytest.skip(_FBTRITON_INTERPRETED_DESCRIPTOR_STORE)
    path = tmp_path / "diagonal.py"
    path.write_text(source)
    load_generated_module(path).diagonal(*args, _launcher=interpret_triton_kernel)
    rows = torch.arange(out.size(0))
    torch.testing.assert_close(out, x[plane, rows + 1, rows + 2])


def test_batched_descriptor_loop_keeps_matched_axes_unconditional(
    sm90_descriptors: None,
) -> None:
    @helion.kernel(backend="triton", static_shapes=True)
    def matmul(x, y, out):
        for batch in hl.grid(x.size(0)):
            for row in hl.tile(x.size(1), block_size=16):
                acc = hl.zeros((row, 16), dtype=torch.float32)
                for k in hl.tile(x.size(2), block_size=16):
                    acc = torch.addmm(acc, x[batch, row, k], y[batch, k, :])
                out[batch, row, :] = acc

    args = (
        torch.empty((4, 32, 64), dtype=torch.float16),
        torch.empty((4, 64, 16), dtype=torch.float16),
        torch.empty((4, 32, 16)),
    )
    src = cpu_bind(matmul, args).to_code(
        helion.Config(indexing="tensor_descriptor", block_sizes=[], num_stages=3)
    )
    kernel = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    assert not any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(kernel))
    assert src.count("_desc.load(") == 2 and src.count("_desc.store(") == 1


@pytest.mark.parametrize(
    ("dtype", "minimum"), [(torch.float16, True), (torch.bfloat16, False)]
)
def test_shifted_half_atomic_keeps_the_native_descriptor_branch(
    sm90_descriptors: None, dtype: torch.dtype, minimum: bool
) -> None:
    @helion.kernel(backend="triton", static_shapes=False)
    def extrema(x, out, shift, minimum: hl.constexpr):
        for row, col in hl.tile([0, 32], x.size()):
            if minimum:
                hl.atomic_min(out, [row, col.index + shift], x[row, col], sem="relaxed")
            else:
                hl.atomic_max(out, [row, col.index + shift], x[row, col], sem="relaxed")

    x = torch.empty((8, 64), dtype=dtype)
    out = torch.empty_like(x)
    code = cpu_bind(extrema, (x, out, -8, minimum)).to_code(
        helion.Config(
            block_sizes=[8, 32], indexing="pointer", atomic_indexing="tensor_descriptor"
        )
    )
    op = "atomic_min" if minimum else "atomic_max"
    branches = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Expr)
            and ast.unparse(stmt.value).startswith(f"out_desc.{op}(")
            for stmt in node.body
        )
    ]
    assert len(branches) == 1, code
    (branch,) = branches
    assert ">= 0" in ast.unparse(branch.test)
    assert "% 8 == 0" in ast.unparse(branch.test)
    assert "requires a 16-byte-aligned descriptor address" in ast.unparse(
        branch.orelse[0]
    )
    assert f"tl.{op}(" not in code


@pytest.mark.parametrize("operation", ["load", "store", "atomic_add"])
def test_descriptor_fallback_widens_before_multiplying_a_large_stride(
    sm90_descriptors: None, operation: str
) -> None:
    @helion.kernel(backend="triton", static_shapes=True)
    def shift_box(x, out, shift, op: hl.constexpr):
        for row, col in hl.tile([0, 32], x.size()):
            if op == "load":
                out[row, col] = x[row, col.index + shift]
            elif op == "store":
                out[row, col.index + shift] = x[row, col]
            else:
                hl.atomic_add(out, [row, col.index + shift], x[row, col], sem="relaxed")

    # Rows fit descriptor coordinates. Row 7's element offset does not fit
    # Int32. Meta tensors exercise the stride without allocating 14 GiB.
    meta = torch.device("meta")
    wide = torch.empty_strided((8, 64), (2**29, 1), device=meta)
    dense = torch.empty((8, 64), device=meta)
    pair = (wide, dense) if operation == "load" else (dense, wide)
    bound = cpu_bind(shift_box, (*pair, -8, operation))
    config = helion.Config(block_sizes=[8, 32], indexing="pointer")
    with pytest.raises(helion.exc.IndexOffsetOutOfRangeForInt32):
        bound.to_code(config)
    if operation == "atomic_add":
        config = helion.Config(
            block_sizes=[8, 32], indexing="pointer", atomic_indexing="tensor_descriptor"
        )
    else:
        indexing = ["tensor_descriptor", "pointer"]
        if operation == "store":
            indexing.reverse()
        config = helion.Config(block_sizes=[8, 32], indexing=indexing)
    source = bound.to_code(config)
    assert f"_desc.{operation}(" in source
    assert "tl.cast(indices_0[:, None], tl.int64) * 536870912" in source
