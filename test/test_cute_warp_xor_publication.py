from __future__ import annotations

import ast
import importlib
from types import SimpleNamespace

import pytest

from helion._compiler.cute.chained_warp_mma import emit_warp_publication


@pytest.mark.parametrize("shape", [(16, 16), (32, 32), (32, 128), (64, 64)])
def test_plain_layout_publication_is_byte_exact(shape: tuple[int, int]) -> None:
    assert emit_warp_publication("stage", shape, xor_scratch=False) == [
        "stage_sc = stage_thr.partition_C(stage_c)",
        "cute.autovec_copy(stage_acc, stage_sc)",
    ]


@pytest.mark.parametrize("shape", [(1, 32), (32, 1), (32, 24)])
def test_identity_layout_retains_existing_copy(shape: tuple[int, int]) -> None:
    assert emit_warp_publication(
        "stage", shape, xor_scratch=True
    ) == emit_warp_publication("stage", shape, xor_scratch=False)


@pytest.mark.parametrize(
    "shape,threads", [((16, 16), 32), ((32, 32), 64), ((32, 128), 64), ((64, 64), 128)]
)
def test_actual_composed_partition_and_publication(
    shape: tuple[int, int], threads: int
) -> None:
    import cutlass
    import cutlass.cute as cute

    from helion._compiler.cute.chained_scratch_layout import xor_swizzle

    ir = importlib.import_module("cutlass._mlir.ir")
    code = "\n".join(emit_warp_publication("stage", shape, xor_scratch=True))
    tree = ast.parse(code)
    assert len(tree.body) == 2 and isinstance(tree.body[1], ast.For)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            params = xor_swizzle(shape)
            assert params is not None
            layout = cute.make_composed_layout(
                cute.make_swizzle(*params),
                0,
                cute.make_layout(shape, stride=(shape[1], 1)),
            )
            shared = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.smem, assumed_align=128
                ),
                layout,
            )
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
                    )
                ),
                atom_layout_mnk=(1, threads // 32, 1),
            )
            coordinates = set()
            physical = set()
            for thread in range(threads):
                owner = mma.get_slice(thread)
                identity = owner.partition_C(cute.make_identity_tensor(shape))
                for index in range(cute.size(identity)):
                    row, column = identity[index]
                    coord = (int(row), int(column))
                    assert coord not in coordinates
                    coordinates.add(coord)
                    physical.add(int(layout(coord)))
                # Execute the emitted static-index copy for one representative
                # thread in IR construction only; no SDK compiler or device.
                if thread == 0:
                    accumulator = cute.make_rmem_tensor(
                        mma.partition_shape_C(shape), cutlass.Float32
                    )
                    target = owner.partition_C(shared)
                    assert cute.size(accumulator) == cute.size(target)
                    if isinstance(target.layout, cute.ComposedLayout):
                        with pytest.raises(
                            TypeError, match="Expected input of type Layout"
                        ):
                            cute.autovec_copy(accumulator, target)
                    else:
                        cute.autovec_copy(accumulator, target)
                    namespace = {
                        "stage_thr": owner,
                        "stage_c": shared,
                        "stage_acc": accumulator,
                        "cute": cute,
                        "cutlass": SimpleNamespace(range_constexpr=range),
                    }
                    exec(code, namespace)
            assert coordinates == {
                (row, column) for row in range(shape[0]) for column in range(shape[1])
            }
            assert physical == set(range(shape[0] * shape[1]))
            assert "memref.store" in str(module) or "cute.store" in str(module)
