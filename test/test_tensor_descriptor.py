from __future__ import annotations

import re
import unittest

import torch

import helion
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_host_tensor_descriptor
from helion._compat import use_tileir_tunables
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import check_example
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import skipUnlessHostTensorDescriptor
from helion._testing import skipUnlessTensorDescriptor
import helion.language as hl


@onlyBackends(["triton"])
class TestTensorDescriptor(RefEagerTestBase, TestCase):
    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptors(self):
        @helion.kernel(autotune_effort="none", static_shapes=False)
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(), block_size=[16, 16]):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([32, 64], device=DEVICE)
        code, result = code_and_output(
            add_one,
            (x,),
            indexing="tensor_descriptor",
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x + 1)
        self.assertIn("_helion_tensor_descriptor(", code)
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def add_one_transposed(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(), block_size=[16, 16]):
                out[tile] = x[tile] + 1
            return out

        transposed = torch.randn([32, 64], device=DEVICE).T
        transposed_code, transposed_result = code_and_output(
            add_one_transposed,
            (transposed,),
            indexing="tensor_descriptor",
            host_tensor_descriptors=True,
        )
        torch.testing.assert_close(transposed_result, transposed + 1)
        self.assertIn("_helion_tensor_descriptor(", transposed_code)
        self.assertIn("tl.permute", transposed_code)
        self.assertNotIn(get_tensor_descriptor_fn_name(), transposed_code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_packed_fp4_base(self):
        @helion.kernel(autotune_effort="none", static_shapes=False)
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(), block_size=[16, 16]):
                out[tile] = x[tile]
            return out

        raw = torch.randint(0, 256, [32, 64], device=DEVICE, dtype=torch.uint8)
        x = raw.view(torch.float4_e2m1fn_x2)
        code, result = code_and_output(
            copy,
            (x,),
            indexing="tensor_descriptor",
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result.view(torch.uint8), raw)
        self.assertIn(".view(torch.uint8)", code)
        self.assertIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_host_tensor_descriptor_runtime_safety_classes(self):
        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16],
                indexing=["tensor_descriptor", "pointer"],
                host_tensor_descriptors=True,
            ),
        )
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        aligned = torch.randn([32, 64], device=DEVICE)
        code, result = code_and_output(copy, (aligned,))
        torch.testing.assert_close(result, aligned)
        self.assertIn("_helion_tensor_descriptor(", code)
        aligned_bound = copy.bind((aligned,))
        self.assertEqual(len(copy._bound_kernels), 1)

        # Both sizes exceed the largest configured descriptor block, so they
        # share the same dynamic specialization.
        same_class = torch.randn([63, 64], device=DEVICE)
        torch.testing.assert_close(copy(same_class), same_class)
        self.assertIs(aligned_bound, copy.bind((same_class,)))
        self.assertEqual(len(copy._bound_kernels), 1)

        # Crossing the block-size threshold recompiles and safely falls back.
        too_small = torch.randn([8, 64], device=DEVICE)
        small_code, small_result = code_and_output(copy, (too_small,))
        torch.testing.assert_close(small_result, too_small)
        self.assertNotIn("_helion_tensor_descriptor(", small_code)
        self.assertEqual(len(copy._bound_kernels), 2)

        empty = torch.empty([0, 64], device=DEVICE)
        empty_code, empty_result = code_and_output(copy, (empty,))
        torch.testing.assert_close(empty_result, empty)
        self.assertNotIn("_helion_tensor_descriptor(", empty_code)
        self.assertEqual(len(copy._bound_kernels), 3)

        storage = torch.randn(32 * 64 + 1, device=DEVICE)
        unaligned = storage[1:].view(32, 64)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        # Exercise the prepared dispatch path before asking for its code.
        torch.testing.assert_close(copy(aligned), aligned)
        torch.testing.assert_close(copy(unaligned), unaligned)
        unaligned_code, unaligned_result = code_and_output(copy, (unaligned,))
        torch.testing.assert_close(unaligned_result, unaligned)
        self.assertNotIn("_helion_tensor_descriptor(", unaligned_code)
        self.assertEqual(len(copy._bound_kernels), 4)

        @helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16],
                indexing=["tensor_descriptor", "pointer"],
                host_tensor_descriptors=True,
            ),
        )
        def static_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        static_code, static_result = code_and_output(static_copy, (aligned,))
        torch.testing.assert_close(static_result, aligned)
        self.assertIn("_helion_tensor_descriptor(", static_code)
        static_unaligned_code, static_unaligned_result = code_and_output(
            static_copy, (unaligned,)
        )
        torch.testing.assert_close(static_unaligned_result, unaligned)
        self.assertNotIn("_helion_tensor_descriptor(", static_unaligned_code)
        self.assertEqual(len(static_copy._bound_kernels), 2)

        @helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16],
                indexing=["tensor_descriptor", "pointer"],
                host_tensor_descriptors=True,
            ),
        )
        def static_dict_copy(xs: dict[str, torch.Tensor]) -> torch.Tensor:
            x = xs["x"]
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        dict_code, dict_result = code_and_output(static_dict_copy, ({"x": aligned},))
        torch.testing.assert_close(dict_result, aligned)
        self.assertIn("_helion_tensor_descriptor(", dict_code)
        dict_unaligned_code, dict_unaligned_result = code_and_output(
            static_dict_copy, ({"x": unaligned},)
        )
        torch.testing.assert_close(dict_unaligned_result, unaligned)
        self.assertNotIn("_helion_tensor_descriptor(", dict_unaligned_code)
        self.assertEqual(len(static_dict_copy._bound_kernels), 2)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_unfixed_cuda_descriptor_extent_cache_caps_at_tma_limit(self):
        @helion.kernel(
            static_shapes=False,
            disable_autotuner_heuristics=True,
        )
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        inputs = [torch.randn([size, 64], device=DEVICE) for size in (512, 1024, 2048)]
        first = copy.bind((inputs[0],))
        for value in inputs[1:]:
            self.assertIs(first, copy.bind((value,)))

        # An empty explicit config list is still an unknown autotune space, so
        # descriptor guards remain active below the CUDA extent ceiling.
        smaller = torch.randn([128, 64], device=DEVICE)
        self.assertIsNot(first, copy.bind((smaller,)))
        self.assertEqual(len(copy._bound_kernels), 2)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_multiple_pointer_configs_skip_descriptor_guards(self):
        @helion.kernel(
            static_shapes=False,
            disable_autotuner_heuristics=True,
            configs=[
                helion.Config(block_sizes=[16, 16], indexing="pointer"),
                helion.Config(block_sizes=[32, 32], indexing="pointer"),
            ],
        )
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        aligned = torch.randn([512, 64], device=DEVICE)
        storage = torch.randn(512 * 64 + 1, device=DEVICE)
        unaligned = storage[1:].view(512, 64)
        larger = torch.randn([1024, 64], device=DEVICE)
        first = copy.bind((aligned,))
        self.assertIs(first, copy.bind((unaligned,)))
        self.assertIs(first, copy.bind((larger,)))
        self.assertEqual(len(copy._bound_kernels), 1)

        @helion.kernel(
            static_shapes=False,
            disable_autotuner_heuristics=True,
            force_autotune=True,
            configs=[
                helion.Config(block_sizes=[16, 16], indexing="pointer"),
                helion.Config(block_sizes=[32, 32], indexing="pointer"),
            ],
        )
        def forced_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        # force_autotune can search outside the decorator configs, so those
        # pointer-only entries cannot suppress descriptor safety guards.
        forced_aligned = forced_copy.bind((aligned,))
        self.assertIsNot(forced_aligned, forced_copy.bind((unaligned,)))
        self.assertEqual(len(forced_copy._bound_kernels), 2)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_multiple_configs_keep_atomic_descriptor_guards(self):
        @helion.kernel(
            static_shapes=False,
            disable_autotuner_heuristics=True,
            configs=[
                helion.Config(
                    block_sizes=[16, 16],
                    indexing="pointer",
                    atomic_indexing="pointer",
                ),
                helion.Config(
                    block_sizes=[32, 32],
                    indexing="pointer",
                    atomic_indexing="tensor_descriptor",
                ),
            ],
        )
        def atomic_add(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                hl.atomic_add(x, tile, value[tile])
            return x

        aligned = torch.zeros([512, 64], device=DEVICE)
        storage = torch.zeros(512 * 64 + 1, device=DEVICE)
        unaligned = storage[1:].view(512, 64)
        value = torch.ones_like(aligned)

        first = atomic_add.bind((aligned, value))
        self.assertIsNot(first, atomic_add.bind((unaligned, value)))
        self.assertEqual(len(atomic_add._bound_kernels), 2)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_scaled_descriptor_extent_cache_classes(self):
        from helion.runtime.kernel import _tensor_descriptor_extent_class

        # A descriptor width can be a multiple of a raw configured block.
        # Keep every CUDA-legal width distinct in the dynamic cache key.
        self.assertEqual(_tensor_descriptor_extent_class(64, 256), 64)
        self.assertEqual(_tensor_descriptor_extent_class(128, 256), 128)
        self.assertEqual(_tensor_descriptor_extent_class(512, 256), 256)

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16],
                indexing=["tensor_descriptor", "pointer"],
                host_tensor_descriptors=True,
            ),
        )
        def copy_scaled_tiles(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            tile_width = hl.register_block_size(16, 16)
            for row, tile in hl.tile(
                [x.size(0), x.size(1) // 2], block_size=[1, tile_width]
            ):
                index = tile.begin * 2 + hl.arange(tile_width * 2)
                mask = index < x.size(1)
                value = hl.load(x, [row.begin, index], extra_mask=mask)
                hl.store(out, [row.begin, index], value, extra_mask=mask)
            return out

        large = torch.randn([4, 64], device=DEVICE)
        large_code, large_result = code_and_output(copy_scaled_tiles, (large,))
        torch.testing.assert_close(large_result, large)
        self.assertIn("_helion_tensor_descriptor(", large_code)
        large_bound = copy_scaled_tiles.bind((large,))

        # The descriptor box is 2 * block_size = 32, not the raw block size
        # of 16. A width-16 input must therefore enter a distinct cache class
        # and compile the safe pointer fallback instead of replaying the
        # width-64 descriptor kernel.
        small = torch.randn([4, 16], device=DEVICE)
        small_code, small_result = code_and_output(copy_scaled_tiles, (small,))
        torch.testing.assert_close(small_result, small)
        self.assertNotIn("_helion_tensor_descriptor(", small_code)
        self.assertIsNot(large_bound, copy_scaled_tiles.bind((small,)))
        self.assertEqual(len(copy_scaled_tiles._bound_kernels), 2)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_infers_omitted_block_sizes(self):
        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                indexing="tensor_descriptor",
                host_tensor_descriptors=True,
            ),
        )
        def copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(), block_size=[16, 16]):
                out[tile] = x[tile]
            return out

        x = torch.randn([32, 64], device=DEVICE)
        code, result = code_and_output(copy, (x,))
        torch.testing.assert_close(result, x)
        self.assertIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_scalar_tensor_offset(self):
        @helion.kernel(
            autotune_effort="none",
            static_shapes=True,
        )
        def select_plane(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            _, rows, columns = x.shape
            out = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
            for tile_expert in hl.tile(1, block_size=1):
                expert = indices[tile_expert.begin]
                for tile_m, tile_n in hl.tile([rows, columns], block_size=[16, 16]):
                    out[tile_m, tile_n] = x[expert, tile_m, tile_n]
            return out

        x = torch.randn([4, 32, 64], device=DEVICE)
        indices = torch.tensor([2], dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(
            select_plane,
            (x, indices),
            indexing="tensor_descriptor",
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x[2])
        self.assertIn("_helion_tensor_descriptor(", code)
        self.assertIn(".load([tl.cast(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_contiguous_tensor_index(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def copy_scaled_tiles(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            tile_width = hl.register_block_size(16, 16)
            for row, tile in hl.tile([4, 32], block_size=[1, tile_width]):
                index = tile.begin * 2 + hl.arange(tile_width * 2)
                out[row.begin, index] = x[row.begin, index]
            return out

        x = torch.randn([4, 64], device=DEVICE)
        code, result = code_and_output(
            copy_scaled_tiles,
            (x,),
            block_sizes=[16],
            indexing=["tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x)
        self.assertEqual(code.count("_helion_tensor_descriptor("), 1)
        self.assertTrue(
            any(
                guard.has_derived_block_extent
                for guard in copy_scaled_tiles.bind(
                    (x,)
                ).env.tensor_descriptor_layout_guards.values()
            )
        )

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_contiguous_index_with_scalar_tensor_base(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def gather_block(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            out = torch.empty([16], dtype=x.dtype, device=x.device)
            for row in hl.tile(1, block_size=1):
                lane = hl.arange(16)
                index = base[()] + lane
                out[lane] = x[row.begin, index]
            return out

        x = torch.randn([1, 64], device=DEVICE)
        base = torch.tensor(16, dtype=torch.int64, device=DEVICE)
        code, result = code_and_output(
            gather_block,
            (x, base),
            indexing=["pointer", "tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x[0, 16:32])
        self.assertEqual(code.count("_helion_tensor_descriptor("), 1)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_non_power_of_two_index_falls_back(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def copy_scaled_tiles(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row, tile in hl.tile([4, 2], block_size=[1, 1]):
                index = tile.begin * 48 + hl.arange(48)
                out[row.begin, index] = x[row.begin, index]
            return out

        x = torch.randn([4, 96], device=DEVICE)
        code, result = code_and_output(
            copy_scaled_tiles,
            (x,),
            indexing=["tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x)
        self.assertNotIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_oversized_index_falls_back(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def copy_block(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(1, block_size=1):
                index = hl.arange(512)
                out[row.begin, index] = x[row.begin, index]
            return out

        x = torch.randn([1, 512], device=DEVICE)
        code, result = code_and_output(
            copy_block,
            (x,),
            indexing=["tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x)
        self.assertNotIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_multiple_tensor_indices_fall_back(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def gather_diagonal(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([16, 16], dtype=x.dtype, device=x.device)
            for _ in hl.tile(1, block_size=1):
                row_index = hl.arange(16)
                column_index = hl.arange(16)
                out[row_index, column_index] = x[row_index, column_index]
            return out

        x = torch.randn([32, 32], device=DEVICE)
        code, result = code_and_output(
            gather_diagonal,
            (x,),
            indexing=["tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x[:16, :16])
        self.assertNotIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_scaled_subtraction_falls_back(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def gather_shifted(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([16], dtype=x.dtype, device=x.device)
            for row in hl.tile(1, block_size=1):
                lane = hl.arange(16)
                index = torch.sub(lane, -16, alpha=2)
                out[lane] = x[row.begin, index]
            return out

        x = torch.randn([1, 64], device=DEVICE)
        code, result = code_and_output(
            gather_shifted,
            (x,),
            indexing=["tensor_descriptor", "pointer"],
            host_tensor_descriptors=True,
        )

        torch.testing.assert_close(result, x[0, 32:48])
        self.assertNotIn("_helion_tensor_descriptor(", code)

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_host_tensor_descriptor_view_alignment_specialization(self):
        def make_copy(indexing: str):
            @helion.kernel(
                autotune_effort="none",
                static_shapes=True,
                config=helion.Config(
                    block_sizes=[16, 16],
                    indexing=indexing,
                    host_tensor_descriptors=True,
                ),
            )
            def copy_view(x: torch.Tensor) -> torch.Tensor:
                view = x.view(16, 128)
                out = torch.empty_like(view)
                for tile in hl.tile(view.size()):
                    out[tile] = view[tile]
                return out

            return copy_view

        aligned = torch.randn([32, 64], device=DEVICE)
        storage = torch.randn(32 * 64 + 1, device=DEVICE)
        unaligned = storage[1:].view(32, 64)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        offset_storage = torch.randn(32 * 64 + 4, device=DEVICE)
        aligned_offset = offset_storage[4:].view(32, 64)
        self.assertEqual(aligned_offset.data_ptr() % 16, 0)
        self.assertNotEqual(aligned_offset.storage_offset(), 0)

        copy_view = make_copy("tensor_descriptor")
        aligned_code, aligned_result = code_and_output(copy_view, (aligned,))
        torch.testing.assert_close(aligned_result, aligned.view(16, 128))
        self.assertIn("_helion_tensor_descriptor(view", aligned_code)
        aligned_bound = copy_view.bind((aligned,))

        unaligned_code, unaligned_result = code_and_output(copy_view, (unaligned,))
        torch.testing.assert_close(unaligned_result, unaligned.view(16, 128))
        self.assertNotIn("_helion_tensor_descriptor(view", unaligned_code)
        self.assertIsNot(aligned_bound, copy_view.bind((unaligned,)))

        aligned_offset_code, aligned_offset_result = code_and_output(
            copy_view, (aligned_offset,)
        )
        torch.testing.assert_close(aligned_offset_result, aligned_offset.view(16, 128))
        self.assertNotIn("_helion_tensor_descriptor(view", aligned_offset_code)
        self.assertIsNot(aligned_bound, copy_view.bind((aligned_offset,)))

        pointer_copy_view = make_copy("pointer")
        pointer_bound = pointer_copy_view.bind((aligned,))
        torch.testing.assert_close(
            pointer_copy_view(unaligned), unaligned.view(16, 128)
        )
        self.assertIs(pointer_bound, pointer_copy_view.bind((unaligned,)))

    @skipUnlessHostTensorDescriptor("Host tensor descriptor support is required")
    def test_host_tensor_descriptor_dynamic_output_falls_back(self):
        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16],
                indexing="tensor_descriptor",
                host_tensor_descriptors=True,
            ),
        )
        def fill(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = 1.0
            return out

        x = torch.randn([32, 64], device=DEVICE)
        code, result = code_and_output(fill, (x,))
        torch.testing.assert_close(result, torch.ones_like(x))
        self.assertNotIn("_helion_tensor_descriptor(", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_permutation_when_stride_one_not_last(self):
        """Test that permutation is applied when stride==1 is not the last dimension."""

        @helion.kernel(autotune_effort="none")
        def kernel_with_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        # Create tensor where stride==1 is the first dimension (not last)
        # This should trigger permutation logic
        x_base = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # This creates stride=[1, 8]

        # Verify the stride pattern we want
        self.assertEqual(x.stride(), (1, 8))
        self.assertEqual(x.stride(0), 1)  # First dimension has stride 1
        self.assertEqual(x.stride(1), 8)  # Second dimension has stride 8

        code, result = code_and_output(
            kernel_with_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check that the result is correct
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains permutation calls
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        # The tensor descriptor should be created with permuted dimensions
        # (sizes and strides should be reordered so stride==1 dim is last)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_no_permutation_when_stride_one_already_last(self):
        """Test that no permutation is applied when stride==1 is already last."""

        @helion.kernel(autotune_effort="none")
        def kernel_no_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] * 2.0
            return result

        # Create tensor where stride==1 is already the last dimension
        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)

        # Verify the stride pattern (last dimension should have stride 1)
        self.assertEqual(x.stride(), (16, 1))
        self.assertEqual(x.stride(-1), 1)  # Last dimension has stride 1

        code, result = code_and_output(
            kernel_no_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check that the result is correct
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

        # Check that the generated code contains tensor descriptor
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        # Should not contain permute calls since no permutation needed
        self.assertNotIn("tl.permute", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_3d_tensor_permutation(self):
        """Test permutation with 3D tensor where stride==1 is in middle."""

        @helion.kernel(autotune_effort="none")
        def kernel_3d_permutation(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 10.0
            return result

        # Create 3D tensor where stride==1 is the middle dimension
        # We'll use as_strided to create a tensor with stride pattern [64, 1, 4]
        # This gives byte strides [256, 4, 16] where 256%16==0 and 16%16==0
        storage_size = 4 * 8 * 16  # Enough storage for the tensor
        base_tensor = torch.randn(storage_size, device=DEVICE, dtype=torch.float32)
        x = base_tensor.as_strided([4, 8, 4], [64, 1, 4])

        # Block sizes must not exceed tensor dim sizes — tensor descriptors
        # require boxDim[i] <= tensorDim[i] in every dimension.
        code, result = code_and_output(
            kernel_3d_permutation,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[4, 8, 4],
        )

        # Check correctness
        expected = x + 10.0
        torch.testing.assert_close(result, expected)

        # Should contain both tensor descriptor and permute operations
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_matrix_transpose_case(self):
        """Test a common case: transposed matrix operations."""

        @helion.kernel(autotune_effort="none")
        def kernel_transpose_case(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] * x[tile]  # Element-wise square
            return result

        # Create a transposed matrix (common in many GPU kernels)
        x_orig = torch.randn([16, 12], device=DEVICE, dtype=torch.float32)
        x = x_orig.t()  # Transpose: shape=[12, 16], stride=[1, 12]

        # Verify this is the problematic case: stride==1 is first, not last
        self.assertEqual(x.shape, (12, 16))
        self.assertEqual(x.stride(), (1, 12))

        code, result = code_and_output(
            kernel_transpose_case,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        # Check correctness
        expected = x * x
        torch.testing.assert_close(result, expected)

        # Should handle the permutation properly
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_permutation_with_different_block_sizes(self):
        """Test that permutation works correctly with different block sizes."""

        @helion.kernel(autotune_effort="none")
        def kernel_different_blocks(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 5.0
            return result

        # Create tensor where stride==1 is not last
        x_base = torch.randn([12, 24], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # stride=[1, 12]

        self.assertEqual(x.stride(), (1, 12))

        code, result = code_and_output(
            kernel_different_blocks,
            (x,),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        expected = x + 5.0
        torch.testing.assert_close(result, expected)

        # Should contain permutation and tensor descriptor
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

        # The block sizes should also be permuted in the tensor descriptor
        # This is important for correctness

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfTileIR("tileir backend will ignore `range_num_stages` hints")
    def test_multistage_range_tensor_descriptor(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[4, 256],
                indexing="tensor_descriptor",
                num_stages=4,
                num_warps=4,
                pid_type="flat",
                range_flattens=[None, False],
                range_multi_buffers=[None, False],
                range_num_stages=[0, 4],
                range_unroll_factors=[0, 0],
                range_warp_specializes=[],
            ),
            static_shapes=True,
        )
        def jsd_forward_kernel(
            _input: torch.Tensor,
            target: torch.Tensor,
            shift_labels: torch.Tensor | None = None,
            beta: float = 0.5,
            ignore_index: int = -100,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            BT, V = _input.shape
            assert target.shape == _input.shape, (
                f"Shape mismatch: {target.shape} != {_input.shape}"
            )
            block_size_n = hl.register_block_size(V)
            block_size_m = hl.register_block_size(BT)

            loss = torch.zeros([BT], dtype=torch.float32, device=_input.device)
            dX = torch.empty_like(loss)

            one_minus_beta = 1 - beta

            n_non_ignore = float(BT)
            if shift_labels is not None:
                n_non_ignore = float((shift_labels != ignore_index).sum().item())
                if n_non_ignore == 0:
                    return torch.zeros(
                        [], dtype=_input.dtype, device=_input.device
                    ), torch.zeros_like(_input)

            for tile_bt in hl.tile(BT, block_size=block_size_m):
                if shift_labels is not None:
                    if shift_labels[tile_bt] == ignore_index:
                        for tile_X in hl.tile(V):
                            dX[tile_bt, tile_X] = 0.0
                        continue
                intermediate_loss = hl.zeros(
                    [tile_bt, block_size_n], dtype=torch.float32
                )
                intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=_input.dtype)
                for tile_v in hl.tile(V, block_size=block_size_n):
                    X = _input[tile_bt, tile_v]
                    Y = target[tile_bt, tile_v]

                    if beta == 0.0:
                        Y_max = torch.amax(Y, dim=0)
                        Y_shift = Y - Y_max
                        Y_prob = torch.exp(Y_shift) * torch.exp(Y_max)
                        intermediate_loss += Y_prob * (Y - X)
                        intermediate_dX += -Y_prob
                    elif beta == 1.0:
                        X_max = torch.amax(X, dim=0)
                        X_shift = X - X_max
                        X_prob = torch.exp(X_shift) * torch.exp(X_max)
                        intermediate_loss += X_prob * (X - Y)
                        intermediate_dX += intermediate_loss + X_prob
                    else:
                        Q = torch.exp(X)
                        P = torch.exp(Y)

                        beta_P = beta * P
                        one_minus_beta_Q = one_minus_beta * Q
                        M = beta_P + one_minus_beta_Q
                        log_M = torch.log(M)
                        x_minus_log_m = X - log_M
                        kl_q_m = one_minus_beta_Q * x_minus_log_m

                        intermediate_loss += beta_P * (Y - log_M) + kl_q_m
                        intermediate_dX += kl_q_m

                scale = 1.0 / n_non_ignore
                loss[tile_bt] = torch.sum(intermediate_loss * scale, dim=1)
                dX[tile_bt] = torch.sum(intermediate_dX * scale, dim=1)

            final_loss = torch.sum(loss)
            return final_loss, dX

        vocab = 512
        batch = 512
        log_q = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)
        log_p = torch.randn(batch, vocab, device=DEVICE).log_softmax(dim=-1)

        code, (loss, _) = code_and_output(jsd_forward_kernel, (log_q, log_p))
        torch.accelerator.synchronize()

        from examples.jsd import TorchJSDBaseline

        baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100).to(DEVICE)
        baseline_loss = baseline(log_q, log_p)

        torch.testing.assert_close(loss, baseline_loss, rtol=5e-2, atol=5e-3)
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        range_stage_values = [
            int(match)
            for line in code.splitlines()
            if "tl.range" in line
            for match in re.findall(r"num_stages=(\d+)", line)
        ]
        # range_num_stages=4 is clamped to 0, so doesn't show up as num_stages in the tl.range call
        self.assertEqual(len(range_stage_values), 0)

        if not supports_host_tensor_descriptor():
            return

        host_config = dict(jsd_forward_kernel.configs[0].config)
        host_config["host_tensor_descriptors"] = True
        host_code, (host_loss, _) = code_and_output(
            jsd_forward_kernel,
            (log_q, log_p),
            **host_config,
        )
        torch.testing.assert_close(host_loss, baseline_loss, rtol=5e-2, atol=5e-3)
        self.assertIn("_helion_tensor_descriptor(", host_code)
        self.assertNotIn(get_tensor_descriptor_fn_name(), host_code)
        host_range_stage_values = [
            int(match)
            for line in host_code.splitlines()
            if "tl.range" in line
            for match in re.findall(r"num_stages=(\d+)", line)
        ]
        self.assertIn(4, host_range_stage_values)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_tiny_matmul_tile_fallback(self) -> None:
        """Tensor descriptor indexing should be rejected when the tile is too small."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[1, 16, 16],
                indexing="tensor_descriptor",
                l2_groupings=[2],
                loop_orders=[[0, 1]],
                num_stages=4,
                num_warps=1,
                pid_type="persistent_blocked",
                range_flattens=[True, True] if not use_tileir_tunables() else [],
                range_multi_buffers=[False, True] if not use_tileir_tunables() else [],
                range_num_stages=[0, 1] if not use_tileir_tunables() else [],
                range_unroll_factors=[0, 4] if not use_tileir_tunables() else [],
            ),
            static_shapes=True,
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        x = torch.randn((64, 64), device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn((64, 64), device=DEVICE, dtype=HALF_DTYPE)

        code, result = code_and_output(matmul, (x, y))
        torch.accelerator.synchronize()
        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        # Ensure we fall back to pointer indexing for accesses that would use the
        # 1x16 tile - there should be no tensor descriptor for the x or out tensors.
        self.assertNotIn("x_desc = tl.make_tensor_descriptor", code)
        self.assertNotIn("out_desc = tl.make_tensor_descriptor", code)
        # The K dimension still has a valid tile size, so the column operand can
        # keep using tensor descriptors.
        self.assertIn("y_desc = tl.make_tensor_descriptor", code)

        # A larger tile should still be able to use tensor descriptors
        code_large, result_large = code_and_output(
            matmul,
            (x, y),
            block_sizes=[16, 16, 16],
            indexing="tensor_descriptor",
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(result_large, expected, atol=1e-2, rtol=1e-2)
        self.assertIn(get_tensor_descriptor_fn_name(), code_large)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_store_operation_permutation(self):
        """Test that store operations also handle permutation correctly."""

        @helion.kernel(autotune_effort="none")
        def kernel_store_permutation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Both tensors might need permutation
            for tile in hl.tile(x.size()):
                y[tile] = x[tile] * 3.0
            return y

        # Create input and output tensors with stride==1 not last
        x_base = torch.randn([8, 12], device=DEVICE, dtype=torch.float32)
        x = x_base.t().contiguous().t()  # stride=[1, 8]

        y_base = torch.zeros([8, 12], device=DEVICE, dtype=torch.float32)
        y = y_base.t().contiguous().t()  # stride=[1, 8]

        self.assertEqual(x.stride(), (1, 8))
        self.assertEqual(y.stride(), (1, 8))

        code, result = code_and_output(
            kernel_store_permutation,
            (x, y),
            indexing="tensor_descriptor",
            block_sizes=[8, 8],
        )

        expected = x * 3.0
        torch.testing.assert_close(result, expected)

        # Should have permutation for both load and store
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertIn("tl.permute", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_attention_tensor_descriptor(self):
        args = (
            torch.randn(1, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(1, 16, 256, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(1, 16, 256, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            emit_code=False,
            block_sizes=[1, 128, 64],
            indexing="tensor_descriptor",
        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_attention_td_dynamic(self):
        args = (
            torch.randn(1, 16, 256, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 16, 256, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 16, 256, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            fn_name="attention_dynamic",
            emit_code=False,
            block_sizes=[1, 16, 16],
            indexing="tensor_descriptor",
        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_minimum_16_byte_block_size_fallback(self):
        """Test that tensor descriptor falls back when block size is too small."""

        @helion.kernel(autotune_effort="none")
        def kernel_small_block(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        # Create a tensor with proper stride alignment
        x = torch.randn([8, 16], device=DEVICE, dtype=torch.float32)

        # Use small block sizes that would result in < 16 bytes in last dimension
        # block_sizes=[4, 2] means last dimension block size = 2
        # 2 * 4 bytes (float32) = 8 bytes < 16 bytes required
        # With the fix, this should fall back to another indexing strategy
        code, result = code_and_output(
            kernel_small_block,
            (x,),
            indexing="tensor_descriptor",  # Request tensor descriptor
            block_sizes=[4, 2],  # Small block size in last dimension
        )

        # Should fall back to block_ptr or pointer indexing instead of tensor descriptor
        # If our fix works, this should NOT contain tensor descriptor
        self.assertNotIn(get_tensor_descriptor_fn_name(), code)

        # But should still work correctly
        expected = x + 1.0
        torch.testing.assert_close(result, expected)

        # Repeat with a 2-byte dtype to ensure the byte-size check scales with
        # dtype, not just element count. block_size=4 gives 4 * 2 = 8 bytes.
        x_half = torch.randn([8, 16], device=DEVICE, dtype=HALF_DTYPE)
        code_half, result_half = code_and_output(
            kernel_small_block,
            (x_half,),
            indexing="tensor_descriptor",
            block_sizes=[4, 4],
        )
        self.assertNotIn(get_tensor_descriptor_fn_name(), code_half)
        torch.testing.assert_close(result_half, x_half + 1.0)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_dynamic_shape_stride_alignment(self):
        """Test that aligned and unaligned strides produce correct results with dynamic shapes.

        When static_shapes=False, _tensor_key buckets sizes to min(s, 2).
        D=1024 and D=2047 both bucket to (2, 2), but D=1024 bf16 has
        16-byte aligned strides while D=2047 does not.  Tensor descriptors
        require 16-byte aligned strides, so these shapes must not share
        a BoundKernel that unconditionally uses tensor descriptors.
        """

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing="tensor_descriptor",
            ),
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        # D=1024 bf16: stride(0)=1024, byte_stride=2048, 16-byte aligned
        x_aligned = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16)
        code_aligned, result_aligned = code_and_output(add_one, (x_aligned,))
        torch.testing.assert_close(result_aligned, x_aligned + 1.0)
        self.assertIn(f"x_desc = {get_tensor_descriptor_fn_name()}", code_aligned)

        # D=2047 bf16: stride(0)=2047, byte_stride=4094, NOT 16-byte aligned
        x_unaligned = torch.randn(64, 2047, device=DEVICE, dtype=torch.bfloat16)
        code_unaligned, result_unaligned = code_and_output(add_one, (x_unaligned,))
        torch.testing.assert_close(result_unaligned, x_unaligned + 1.0)
        self.assertNotIn(f"x_desc = {get_tensor_descriptor_fn_name()}", code_unaligned)

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing="tensor_descriptor",
            ),
        )
        def add_one_reverse_order(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + 1.0
            return result

        code_unaligned_first, result_unaligned_first = code_and_output(
            add_one_reverse_order, (x_unaligned,)
        )
        torch.testing.assert_close(result_unaligned_first, x_unaligned + 1.0)
        self.assertNotIn(
            f"x_desc = {get_tensor_descriptor_fn_name()}", code_unaligned_first
        )

        code_aligned_second, result_aligned_second = code_and_output(
            add_one_reverse_order, (x_aligned,)
        )
        torch.testing.assert_close(result_aligned_second, x_aligned + 1.0)
        self.assertIn(
            f"x_desc = {get_tensor_descriptor_fn_name()}", code_aligned_second
        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_dynamic_shape_stride_one_dim_guard(self):
        """Changing stride-one dim within a shape bucket should recompile TD code."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[8, 8],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_input(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x_contiguous = torch.randn([16, 8], device=DEVICE, dtype=torch.float32)
        x_transposed = torch.randn([8, 16], device=DEVICE, dtype=torch.float32).t()
        self.assertEqual(x_contiguous.stride(), (8, 1))
        self.assertEqual(x_transposed.stride(), (1, 16))

        code_contiguous, result_contiguous = code_and_output(
            copy_input, (x_contiguous,)
        )
        torch.testing.assert_close(result_contiguous, x_contiguous)
        self.assertIn(f"x_desc = {get_tensor_descriptor_fn_name()}", code_contiguous)
        self.assertNotIn("tl.permute", code_contiguous)

        code_transposed, result_transposed = code_and_output(
            copy_input, (x_transposed,)
        )
        torch.testing.assert_close(result_transposed, x_transposed)
        self.assertIn(f"x_desc = {get_tensor_descriptor_fn_name()}", code_transposed)
        self.assertIn("tl.permute", code_transposed)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_dynamic_tensor_descriptor_reuses_specialization_across_batch_sizes(self):
        """Changing only batch size should not trigger extra dynamic TD specializations."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16, 16],
                indexing="tensor_descriptor",
            ),
        )
        def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            batch, k = x.size()
            k2, n = w.size()
            assert k == k2
            out = torch.empty([batch, n], device=x.device, dtype=x.dtype)
            for tile_b, tile_n in hl.tile([batch, n]):
                acc = hl.zeros([tile_b, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_b, tile_k], w[tile_k, tile_n])
                out[tile_b, tile_n] = acc.to(out.dtype)
            return out

        w = torch.randn([64, 64], device=DEVICE, dtype=HALF_DTYPE)
        xs = [
            torch.randn([32, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([128, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([32, 64], device=DEVICE, dtype=HALF_DTYPE),
        ]

        code, result = code_and_output(linear, (xs[0], w))
        torch.testing.assert_close(result, xs[0] @ w, atol=1e-1, rtol=1e-2)
        self.assert_tensor_descriptor_used_for(code, "x")
        self.assert_tensor_descriptor_used_for(code, "w")
        self.assertEqual(len(linear._bound_kernels), 1)

        for x in xs[1:]:
            result = linear(x, w)
            torch.testing.assert_close(result, x @ w, atol=1e-1, rtol=1e-2)
            self.assertEqual(len(linear._bound_kernels), 1)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_pointer_indexing_ignores_tensor_descriptor_layout_specialization(self):
        """Pointer-only configs should not specialize on TD layout predicates."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing="pointer",
            ),
        )
        def copy_input(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[:, :1024]
        self.assertEqual(x_aligned.stride(), (1024, 1))
        self.assertEqual(x_unaligned.stride(), (1025, 1))

        code_aligned, result_aligned = code_and_output(copy_input, (x_aligned,))
        torch.testing.assert_close(result_aligned, x_aligned)
        self.assert_tensor_descriptor_not_used_for(code_aligned, "x")
        bound_aligned = copy_input.bind((x_aligned,))
        self.assertEqual(len(copy_input._bound_kernels), 1)

        result_unaligned = copy_input(x_unaligned)
        torch.testing.assert_close(result_unaligned, x_unaligned)
        self.assertIs(bound_aligned, copy_input.bind((x_unaligned,)))
        self.assertEqual(len(copy_input._bound_kernels), 1)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_mixed_indexing_only_specializes_tensor_descriptor_operands(self):
        """Mixed configs should only guard tensors used by TD-indexed ops."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing=["pointer", "tensor_descriptor", "pointer"],
            ),
        )
        def add_inputs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile] + y[tile]
            return result

        x_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[:, :1024]
        y_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        y_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[:, :1024]
        self.assertEqual(x_aligned.stride(), (1024, 1))
        self.assertEqual(x_unaligned.stride(), (1025, 1))
        self.assertEqual(y_aligned.stride(), (1024, 1))
        self.assertEqual(y_unaligned.stride(), (1025, 1))

        code_aligned, result_aligned = code_and_output(
            add_inputs, (x_aligned, y_aligned)
        )
        torch.testing.assert_close(result_aligned, x_aligned + y_aligned)
        self.assert_tensor_descriptor_not_used_for(code_aligned, "x")
        self.assert_tensor_descriptor_used_for(code_aligned, "y")
        bound_aligned = add_inputs.bind((x_aligned, y_aligned))
        self.assertEqual(len(add_inputs._bound_kernels), 1)

        result_x_unaligned = add_inputs(x_unaligned, y_aligned)
        torch.testing.assert_close(result_x_unaligned, x_unaligned + y_aligned)
        self.assertIs(bound_aligned, add_inputs.bind((x_unaligned, y_aligned)))
        self.assertEqual(len(add_inputs._bound_kernels), 1)

        code_y_unaligned, result_y_unaligned = code_and_output(
            add_inputs, (x_aligned, y_unaligned)
        )
        torch.testing.assert_close(result_y_unaligned, x_aligned + y_unaligned)
        self.assert_tensor_descriptor_not_used_for(code_y_unaligned, "y")
        self.assertIsNot(bound_aligned, add_inputs.bind((x_aligned, y_unaligned)))
        self.assertEqual(len(add_inputs._bound_kernels), 2)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_dynamic_shape_layout_signature_controls_specialization(self):
        """Dynamic TD should specialize on layout predicates but reuse matching layouts."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_input(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_aligned_2 = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[:, :1024]
        x_aligned_3 = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)

        self.assertEqual(x_aligned.stride(), (1024, 1))
        self.assertEqual(x_aligned_2.stride(), (1024, 1))
        self.assertEqual(x_unaligned.stride(), (1025, 1))
        self.assertEqual(x_aligned_3.stride(), (1024, 1))

        code_aligned, result_aligned = code_and_output(copy_input, (x_aligned,))
        torch.testing.assert_close(result_aligned, x_aligned)
        self.assert_tensor_descriptor_used_for(code_aligned, "x")
        bound_aligned = copy_input.bind((x_aligned,))
        self.assertEqual(len(copy_input._bound_kernels), 1)

        result_aligned_2 = copy_input(x_aligned_2)
        torch.testing.assert_close(result_aligned_2, x_aligned_2)
        self.assertIs(bound_aligned, copy_input.bind((x_aligned_2,)))
        self.assertEqual(len(copy_input._bound_kernels), 1)

        code_unaligned, result_unaligned = code_and_output(copy_input, (x_unaligned,))
        torch.testing.assert_close(result_unaligned, x_unaligned)
        self.assert_tensor_descriptor_not_used_for(code_unaligned, "x")
        self.assertIsNot(bound_aligned, copy_input.bind((x_unaligned,)))
        self.assertEqual(len(copy_input._bound_kernels), 2)

        result_aligned_3 = copy_input(x_aligned_3)
        torch.testing.assert_close(result_aligned_3, x_aligned_3)
        self.assertIs(bound_aligned, copy_input.bind((x_aligned_3,)))
        self.assertEqual(len(copy_input._bound_kernels), 2)

        storage = torch.randn(64 * 1024 + 1, device=DEVICE, dtype=HALF_DTYPE)
        x_offset = storage[1:].view(64, 1024)
        self.assertEqual(x_offset.stride(), x_aligned.stride())
        self.assertNotEqual(x_offset.data_ptr() % 16, 0)
        code_offset, result_offset = code_and_output(copy_input, (x_offset,))
        torch.testing.assert_close(result_offset, x_offset)
        self.assert_tensor_descriptor_not_used_for(code_offset, "x")
        self.assertEqual(len(copy_input._bound_kernels), 3)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_tensor_descriptor_container_inputs_static_and_dynamic(self):
        """List/tuple element source paths should preserve TD eligibility."""

        def td_config() -> helion.Config:
            return helion.Config(
                block_sizes=[32, 32],
                indexing=["tensor_descriptor", "pointer"],
            )

        @helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            config=td_config(),
        )
        def copy_first_static(xs: list[torch.Tensor]) -> torch.Tensor:
            x = xs[0]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=td_config(),
        )
        def copy_first_dynamic(xs: list[torch.Tensor]) -> torch.Tensor:
            x = xs[0]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=td_config(),
        )
        def copy_second_dynamic(xs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            x = xs[1]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        unused = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        cases = [
            ("static_list", copy_first_static, ([x, unused],), x),
            ("dynamic_list", copy_first_dynamic, ([x, unused],), x),
            ("dynamic_tuple", copy_second_dynamic, ((unused, x),), x),
        ]

        for name, kernel, args, expected in cases:
            with self.subTest(case=name):
                code, result = code_and_output(kernel, args)
                torch.testing.assert_close(result, expected)
                self.assertIn(get_tensor_descriptor_fn_name(), code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_dynamic_tensor_descriptor_dict_input_sources(self):
        """Dictionary inputs participate in TD layout specialization."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_dict(xs: dict[str, torch.Tensor]) -> torch.Tensor:
            x = xs["x"]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_int_dict(xs: dict[int, torch.Tensor]) -> torch.Tensor:
            x = xs[0]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        for kernel, args in (
            (copy_dict, ({"x": x},)),
            (copy_int_dict, ({0: x},)),
        ):
            code, result = code_and_output(kernel, args)
            torch.testing.assert_close(result, x)
            self.assertIn(get_tensor_descriptor_fn_name(), code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("Test checks bound kernel specialization cache")
    def test_dynamic_tensor_descriptor_list_element_layout_specialization(self):
        """Only TD-accessed list elements should contribute layout cache keys."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[32, 32],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_first(xs: list[torch.Tensor]) -> torch.Tensor:
            x = xs[0]
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_aligned_2 = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        x_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[:, :1024]
        unused_aligned = torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)
        unused_unaligned = torch.randn([64, 1025], device=DEVICE, dtype=HALF_DTYPE)[
            :, :1024
        ]

        code_aligned, result_aligned = code_and_output(
            copy_first, ([x_aligned, unused_aligned],)
        )
        torch.testing.assert_close(result_aligned, x_aligned)
        self.assertIn(get_tensor_descriptor_fn_name(), code_aligned)
        bound_aligned = copy_first.bind(([x_aligned, unused_aligned],))
        self.assertEqual(len(copy_first._bound_kernels), 1)

        bound_unused_unaligned = copy_first.bind(([x_aligned_2, unused_unaligned],))
        self.assertIs(bound_aligned, bound_unused_unaligned)
        self.assertEqual(len(copy_first._bound_kernels), 1)

        code_unaligned, result_unaligned = code_and_output(
            copy_first, ([x_unaligned, unused_aligned],)
        )
        torch.testing.assert_close(result_unaligned, x_unaligned)
        self.assertNotIn(get_tensor_descriptor_fn_name(), code_unaligned)
        bound_unaligned = copy_first.bind(([x_unaligned, unused_aligned],))
        self.assertIsNot(bound_aligned, bound_unaligned)
        self.assertEqual(len(copy_first._bound_kernels), 2)

    @staticmethod
    def _make_td_matmul(static_shapes: bool):
        @helion.kernel(
            static_shapes=static_shapes,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16, 16],
                indexing="tensor_descriptor",
            ),
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty([m, n], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        return matmul

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_matmul_tensor_descriptor_static_and_dynamic(self):
        """Matmul should use TD loads and tl.dot in static and dynamic modes."""

        y = torch.randn([64, 32], device=DEVICE, dtype=HALF_DTYPE)
        cases = [
            (
                "contiguous",
                torch.randn([32, 64], device=DEVICE, dtype=HALF_DTYPE),
                False,
            ),
            (
                "transposed",
                torch.randn([64, 32], device=DEVICE, dtype=HALF_DTYPE).t(),
                True,
            ),
        ]
        static_shape_modes = (True, False)

        for static_shapes in static_shape_modes:
            matmul = self._make_td_matmul(static_shapes)
            for name, x, expect_permute in cases:
                with self.subTest(static_shapes=static_shapes, layout=name):
                    code, result = code_and_output(matmul, (x, y))
                    torch.testing.assert_close(
                        result,
                        x @ y,
                        atol=1e-1,
                        rtol=1e-2,
                    )
                    self.assert_tensor_descriptor_used_for(code, "x")
                    self.assert_tensor_descriptor_used_for(code, "y")
                    self.assertIn("tl.dot", code)
                    if expect_permute:
                        self.assertIn("tl.permute", code)
                    else:
                        self.assertNotIn("tl.permute", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_matmul_tensor_descriptor_mixed_fallback_static_and_dynamic(self):
        """Matmul should fall back per input when only one stride layout is valid."""

        cases = [
            (
                "aligned",
                torch.randn([32, 64], device=DEVICE, dtype=HALF_DTYPE),
                torch.randn([64, 32], device=DEVICE, dtype=HALF_DTYPE),
                True,
                True,
            ),
            (
                "x_unaligned",
                torch.randn([32, 63], device=DEVICE, dtype=HALF_DTYPE),
                torch.randn([63, 32], device=DEVICE, dtype=HALF_DTYPE),
                False,
                True,
            ),
            (
                "y_unaligned",
                torch.randn([32, 64], device=DEVICE, dtype=HALF_DTYPE),
                torch.randn([64, 31], device=DEVICE, dtype=HALF_DTYPE),
                True,
                False,
            ),
        ]

        for static_shapes in (True, False):
            matmul = self._make_td_matmul(static_shapes)
            for name, x, y, expect_x_desc, expect_y_desc in cases:
                with self.subTest(static_shapes=static_shapes, layout=name):
                    code, result = code_and_output(matmul, (x, y))
                    torch.testing.assert_close(
                        result,
                        x @ y,
                        atol=1e-1,
                        rtol=1e-2,
                    )
                    if expect_x_desc:
                        self.assert_tensor_descriptor_used_for(code, "x")
                    else:
                        self.assert_tensor_descriptor_not_used_for(code, "x")
                    if expect_y_desc:
                        self.assert_tensor_descriptor_used_for(code, "y")
                    else:
                        self.assert_tensor_descriptor_not_used_for(code, "y")
                    self.assertIn("tl.dot", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfXPU("XPU tensor descriptor path has issue with stride-0 input")
    def test_dynamic_shape_stride_zero_input(self):
        """Expanded stride-0 dimensions should be TD-eligible with dynamic shapes."""

        @helion.kernel(
            static_shapes=False,
            autotune_effort="none",
            config=helion.Config(
                block_sizes=[16, 16],
                indexing=["tensor_descriptor", "pointer"],
            ),
        )
        def copy_expanded(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty(x.size(), device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size()):
                result[tile] = x[tile]
            return result

        x = torch.randn([1, 64], device=DEVICE, dtype=HALF_DTYPE).expand(32, 64)
        self.assertEqual(x.stride(), (0, 1))

        code, result = code_and_output(copy_expanded, (x,))
        torch.testing.assert_close(result, x)
        self.assert_tensor_descriptor_used_for(code, "x")

    def assert_uses_tensor_descriptors(self, code: str) -> None:
        self.assertIn(get_tensor_descriptor_fn_name(), code)
        self.assertNotIn("tl.load(", code)
        self.assertNotIn("tl.store(", code)

    def assert_tensor_descriptor_used_for(self, code: str, name: str) -> None:
        self.assertIn(f"{name}_desc = {get_tensor_descriptor_fn_name()}", code)

    def assert_tensor_descriptor_not_used_for(self, code: str, name: str) -> None:
        self.assertNotIn(f"{name}_desc = {get_tensor_descriptor_fn_name()}", code)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfXPU(
        "XPU tensor descriptor path has accuracy issue for scalar SymInt subscripts"
    )
    def test_scalar_symint_subscript_allowlist(self):
        """Known scalar SymInt expressions should still use tensor descriptors."""

        def make_gather_rows(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def gather_rows(x: torch.Tensor, start: int) -> torch.Tensor:
                _, n = x.size()
                out = torch.empty([4, n], device=x.device, dtype=x.dtype)
                for tile_n in hl.tile(n):
                    out[0, tile_n] = x[3, tile_n]
                    out[1, tile_n] = x[start, tile_n]
                    out[2, tile_n] = x[start + 3, tile_n]
                    out[3, tile_n] = x[start - start + 1, tile_n]
                return out

            return gather_rows

        def make_copy_offset_rows(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def copy_offset_rows(x: torch.Tensor, start: int) -> torch.Tensor:
                _, n = x.size()
                rows = 4
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for tile_b in hl.tile(rows, block_size=1):
                    for tile_n in hl.tile(n):
                        out[tile_b.begin, tile_n] = x[start + tile_b.begin, tile_n]
                return out

            return copy_offset_rows

        def make_copy_grid_rows(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def copy_grid_rows(x: torch.Tensor) -> torch.Tensor:
                _, n = x.size()
                rows = 4
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for row in hl.grid(rows):
                    for tile_n in hl.tile(n):
                        out[row, tile_n] = x[row, tile_n]
                return out

            return copy_grid_rows

        def make_scalar_noncontiguous_dims(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64, 64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def scalar_noncontiguous_dims(x: torch.Tensor) -> torch.Tensor:
                _, _, t, d = x.size()
                out = torch.empty([t, d], device=x.device, dtype=x.dtype)
                for tile_t, tile_d in hl.tile([t, d]):
                    out[tile_t, tile_d] = x[0, 0, tile_t, tile_d]
                return out

            return scalar_noncontiguous_dims

        x = torch.randn(8, 128, device=DEVICE, dtype=torch.float32)
        x4d = torch.randn(2, 4, 128, 64, device=DEVICE, dtype=torch.float32)

        for static_shapes in (True, False):
            cases = [
                (
                    "gather_rows",
                    make_gather_rows(static_shapes),
                    (x, 2),
                    torch.stack([x[3], x[2], x[5], x[1]]),
                ),
                (
                    "copy_offset_rows",
                    make_copy_offset_rows(static_shapes),
                    (x, 2),
                    x[2:6],
                ),
                (
                    "copy_grid_rows",
                    make_copy_grid_rows(static_shapes),
                    (x,),
                    x[:4],
                ),
                (
                    "scalar_noncontiguous_dims",
                    make_scalar_noncontiguous_dims(static_shapes),
                    (x4d,),
                    x4d[0, 0],
                ),
            ]
            for name, kernel, args, expected in cases:
                with self.subTest(static_shapes=static_shapes, case=name):
                    code, result = code_and_output(kernel, args)
                    torch.testing.assert_close(result, expected)
                    self.assert_tensor_descriptor_used_for(code, "x")

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_scalar_symint_subscript_blocklist(self):
        """Unsafe scalar SymInt expressions should fall back for that tensor."""

        def make_read_tile_end(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def read_tile_end(x: torch.Tensor) -> torch.Tensor:
                _, n = x.size()
                rows = 4
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for tile_b in hl.tile(rows, block_size=1):
                    for tile_n in hl.tile(n):
                        out[tile_b.begin, tile_n] = x[tile_b.end, tile_n]
                return out

            return read_tile_end

        def make_read_tile_count(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def read_tile_count(x: torch.Tensor) -> torch.Tensor:
                _, n = x.size()
                rows = 4
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for tile_b in hl.tile(rows, block_size=1):
                    for tile_n in hl.tile(n):
                        out[tile_b.begin, tile_n] = x[tile_b.count, tile_n]
                return out

            return read_tile_count

        def make_read_tile_id(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def read_tile_id(x: torch.Tensor) -> torch.Tensor:
                _, n = x.size()
                rows = 4
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for tile_b in hl.tile(rows, block_size=1):
                    for tile_n in hl.tile(n):
                        out[tile_b.begin, tile_n] = x[tile_b.id, tile_n]
                return out

            return read_tile_id

        def make_read_indirect_rows(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64, 64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def read_indirect_rows(
                x: torch.Tensor, indices: torch.Tensor
            ) -> torch.Tensor:
                _, n = x.size()
                rows = indices.size(0)
                out = torch.empty([rows, n], device=x.device, dtype=x.dtype)
                for tile_b, tile_n in hl.tile([rows, n]):
                    idx = indices[tile_b]
                    out[tile_b, tile_n] = x[idx, tile_n]
                return out

            return read_indirect_rows

        def make_scalar_contiguous_dim(static_shapes: bool):
            @helion.kernel(
                config=helion.Config(
                    block_sizes=[64],
                    indexing="tensor_descriptor",
                ),
                static_shapes=static_shapes,
            )
            def scalar_contiguous_dim(g: torch.Tensor) -> torch.Tensor:
                _, t, _ = g.size()
                out = torch.empty([t], device=g.device, dtype=g.dtype)
                for tile_t in hl.tile(t):
                    out[tile_t] = g[0, tile_t, 0]
                return out

            return scalar_contiguous_dim

        x = torch.randn(8, 128, device=DEVICE, dtype=torch.float32)
        g = torch.randn(2, 128, 80, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([3, 1, 4, 0], device=DEVICE, dtype=torch.int64)

        for static_shapes in (True, False):
            cases = [
                (
                    "read_tile_end",
                    make_read_tile_end(static_shapes),
                    (x,),
                    x[1:5],
                    "x",
                ),
                (
                    "read_tile_count",
                    make_read_tile_count(static_shapes),
                    (x,),
                    x[4].expand(4, 128),
                    "x",
                ),
                (
                    "read_tile_id",
                    make_read_tile_id(static_shapes),
                    (x,),
                    x[:4],
                    "x",
                ),
                (
                    "read_indirect_rows",
                    make_read_indirect_rows(static_shapes),
                    (x, indices),
                    x[indices],
                    "x",
                ),
                (
                    "scalar_contiguous_dim",
                    make_scalar_contiguous_dim(static_shapes),
                    (g,),
                    g[0, :, 0],
                    "g",
                ),
            ]
            for name, kernel, args, expected, tensor_name in cases:
                with self.subTest(static_shapes=static_shapes, case=name):
                    code, result = code_and_output(kernel, args)
                    torch.testing.assert_close(result, expected)
                    self.assert_tensor_descriptor_not_used_for(code, tensor_name)


if __name__ == "__main__":
    unittest.main()
