"""
Tests for torch.topk support in Helion device loops.

Limitations:
- Only dim=-1 (last dimension) supported
- k must be compile-time constant (use hl.specialize(k))
- float64/int64 not supported (use float32/int32)
- Max input size: 65536 elements per row

Supported: float32, float16, bfloat16, int32, largest=True/False, sorted=True/False
"""
from __future__ import annotations

import pytest
import torch

import helion
import helion.exc
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


# Use block_sizes larger than tensor dimensions to test masking logic
# Use hl.constexpr for largest parameter to make it a compile-time constant
@helion.kernel(config=helion.Config(block_sizes=[64]))
def topk_2d_kernel(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    n, m = x.size()
    values = torch.empty([n, k], dtype=x.dtype, device=x.device)
    indices = torch.empty([n, k], dtype=torch.int64, device=x.device)

    for tile_n in hl.tile(n):
        v, idx = torch.topk(x[tile_n, :], k, dim=-1, largest=largest)
        values[tile_n, :] = v
        indices[tile_n, :] = idx

    return values, indices


@helion.kernel(config=helion.Config(block_sizes=[64]))
def topk_unsorted_kernel(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    n, m = x.size()
    values = torch.empty([n, k], dtype=x.dtype, device=x.device)
    indices = torch.empty([n, k], dtype=torch.int64, device=x.device)

    for tile_n in hl.tile(n):
        v, idx = torch.topk(x[tile_n, :], k, dim=-1, sorted=False)
        values[tile_n, :] = v
        indices[tile_n, :] = idx

    return values, indices


def verify_topk_result(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: bool,
    dim: int = -1,
) -> bool:
    """Verify that topk result is valid.

    Checks:
    1. Values gathered from x using indices match the returned values
    2. The returned values are the correct top-k values (sorted comparison)
    """
    # Check that gathering x at indices gives us the values
    gathered = x.gather(dim=dim, index=indices)
    if not torch.allclose(gathered, values, rtol=1e-3, atol=1e-3):
        return False

    # Check that the values are the correct top-k (order may differ for ties)
    expected_v, _ = torch.topk(x, k, dim=dim, largest=largest)
    our_sorted = values.sort(dim=dim, descending=largest)[0]
    exp_sorted = expected_v.sort(dim=dim, descending=largest)[0]

    return torch.allclose(our_sorted, exp_sorted, rtol=1e-3, atol=1e-3)


requires_cuda = pytest.mark.skipif(
    str(DEVICE) == "cpu", reason="topk requires CUDA"
)


@requires_cuda
class TestTopK:
    """Tests for torch.topk support in Helion."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("largest", [True, False])
    def test_topk_dtypes_and_largest(self, dtype, largest):
        """Test topk with various dtypes and largest parameter."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE, dtype=dtype)

        values, indices = topk_2d_kernel(x, 10, largest)

        # For float16/bfloat16, ties may have different order due to precision
        if dtype in (torch.float16, torch.bfloat16):
            assert verify_topk_result(x, values, indices, 10, largest)
        else:
            expected_v, expected_i = torch.topk(x, 10, dim=-1, largest=largest)
            torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
            assert torch.equal(indices, expected_i)

    @pytest.mark.parametrize("largest", [True, False])
    def test_topk_int32(self, largest):
        """Test topk with int32 dtype."""
        # Use unique values to avoid tie-breaking issues
        torch.manual_seed(42)
        x = torch.arange(256, device=DEVICE, dtype=torch.int32).view(8, 32)
        for i in range(8):
            x[i] = x[i, torch.randperm(32, device=DEVICE)]

        values, indices = topk_2d_kernel(x, 5, largest)
        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=largest)

        torch.testing.assert_close(values, expected_v)
        assert torch.equal(indices, expected_i)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 7, 8, 10, 15, 16, 32])
    def test_topk_k_values(self, k):
        """Test topk with various k values (power-of-2 and non-power-of-2)."""
        torch.manual_seed(42)
        x = torch.randn([16, 64], device=DEVICE)

        values, indices = topk_2d_kernel(x, k, True)
        expected_v, _ = torch.topk(x, k, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "dtype,largest,base_val,known_values,expected_vals,expected_idxs",
        [
            # float32 largest
            (torch.float32, True, 0.0,
             {(0, 5): 10.0, (0, 10): 9.0, (0, 3): 8.0, (1, 15): 6.0, (1, 0): 5.0, (1, 7): 4.0},
             [[10.0, 9.0, 8.0], [6.0, 5.0, 4.0]], [[5, 10, 3], [15, 0, 7]]),
            # int32 largest
            (torch.int32, True, 0,
             {(0, 5): 100, (0, 10): 90, (0, 3): 80, (1, 15): 60, (1, 0): 50, (1, 7): 40},
             [[100, 90, 80], [60, 50, 40]], [[5, 10, 3], [15, 0, 7]]),
            # float32 smallest
            (torch.float32, False, 10.0,
             {(0, 5): 1.0, (0, 10): 2.0, (0, 3): 3.0},
             [[1.0, 2.0, 3.0]], [[5, 10, 3]]),
        ],
        ids=["float32_largest", "int32_largest", "float32_smallest"],
    )
    def test_topk_known_values(self, dtype, largest, base_val, known_values, expected_vals, expected_idxs):
        """Test topk with known input values for exact verification."""
        x = torch.full([4, 16], base_val, device=DEVICE, dtype=dtype)
        for (row, col), val in known_values.items():
            x[row, col] = val

        values, indices = topk_2d_kernel(x, 3, largest)

        for row_idx, (exp_vals, exp_idxs) in enumerate(zip(expected_vals, expected_idxs)):
            assert values[row_idx].tolist() == exp_vals
            assert indices[row_idx].tolist() == exp_idxs

    def test_topk_generated_code(self):
        """Test that generated code contains expected patterns."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE)

        code, (values, indices) = code_and_output(
            topk_2d_kernel, (x, 10, True), block_size=32
        )

        # Should use tl.topk
        assert "tl.topk" in code
        # Should use tl.sort for sorted output
        assert "tl.sort" in code
        # Verify output is correct
        expected_v, _ = torch.topk(x, 10, dim=-1, largest=True)
        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)

    def test_topk_sorted_false(self):
        """Test that sorted=False skips the sorting step."""
        torch.manual_seed(42)
        x = torch.randn([32, 64], device=DEVICE)

        code, (values, indices) = code_and_output(
            topk_unsorted_kernel, (x, 10), block_size=32
        )

        # Should use tl.topk but NOT tl.sort
        assert "tl.topk" in code
        assert "tl.sort" not in code

        # Values should be correct (just maybe not sorted)
        expected_v, _ = torch.topk(x, 10, dim=-1, largest=True)
        our_sorted = values.sort(dim=-1, descending=True)[0]
        exp_sorted = expected_v.sort(dim=-1, descending=True)[0]
        torch.testing.assert_close(our_sorted, exp_sorted, rtol=1e-4, atol=1e-4)

        # Verify indices point to correct values
        gathered = x.gather(dim=-1, index=indices)
        torch.testing.assert_close(gathered, values, rtol=1e-4, atol=1e-4)

    def test_topk_1d_tensor(self):
        """Test topk with 1D tensor input."""

        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def topk_1d_kernel(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            (n,) = x.size()
            values = torch.empty([k], dtype=x.dtype, device=x.device)
            indices = torch.empty([k], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n], k, dim=-1, largest=True)
                values[:] = v
                indices[:] = idx
            return values, indices

        torch.manual_seed(42)
        x = torch.randn([64], device=DEVICE)

        values, indices = topk_1d_kernel(x, 5)
        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)

        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        assert torch.equal(indices, expected_i)

    @pytest.mark.parametrize(
        "shape,block_sizes,tile_dims",
        [
            ([4, 8, 32], [64, 64], [0, 1]),  # 3D
            ([2, 2, 4, 32], [4, 4, 8], [0, 1, 2]),  # 4D
        ],
        ids=["3d", "4d"],
    )
    def test_topk_nd_tensor(self, shape, block_sizes, tile_dims):
        """Test topk with higher-dimensional tensors."""
        ndim = len(shape)

        if ndim == 3:
            @helion.kernel(config=helion.Config(block_sizes=block_sizes))
            def topk_3d_kernel(
                x: torch.Tensor, k: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
                k = hl.specialize(k)
                b, n, m = x.size()
                values = torch.empty([b, n, k], dtype=x.dtype, device=x.device)
                indices = torch.empty([b, n, k], dtype=torch.int64, device=x.device)
                for tile_b, tile_n in hl.tile([b, n]):
                    v, idx = torch.topk(x[tile_b, tile_n, :], k, dim=-1, largest=True)
                    values[tile_b, tile_n, :] = v
                    indices[tile_b, tile_n, :] = idx
                return values, indices

            kernel = topk_3d_kernel
        else:  # 4D
            @helion.kernel(config=helion.Config(block_sizes=block_sizes))
            def topk_4d_kernel(
                x: torch.Tensor, k: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
                k = hl.specialize(k)
                a, b, n, m = x.size()
                values = torch.empty([a, b, n, k], dtype=x.dtype, device=x.device)
                indices = torch.empty([a, b, n, k], dtype=torch.int64, device=x.device)
                for tile_a, tile_b, tile_n in hl.tile([a, b, n]):
                    v, idx = torch.topk(
                        x[tile_a, tile_b, tile_n, :], k, dim=-1, largest=True
                    )
                    values[tile_a, tile_b, tile_n, :] = v
                    indices[tile_a, tile_b, tile_n, :] = idx
                return values, indices

            kernel = topk_4d_kernel

        torch.manual_seed(42)
        x = torch.randn(shape, device=DEVICE)

        values, indices = kernel(x, 5)
        expected_v, expected_i = torch.topk(x, 5, dim=-1, largest=True)

        torch.testing.assert_close(values, expected_v, rtol=1e-4, atol=1e-4)
        assert torch.equal(indices, expected_i)



@requires_cuda
class TestTopKErrors:
    """Test error handling for unsupported topk configurations."""

    def test_error_dim_not_last(self):
        """Test that dim != -1 raises an error."""

        @helion.kernel(config=helion.Config(block_sizes=[64]))
        def topk_dim0(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k = hl.specialize(k)
            n, m = x.size()
            values = torch.empty([k, m], dtype=x.dtype, device=x.device)
            indices = torch.empty([k, m], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n, :], k, dim=0, largest=True)
                values[:, :] = v
                indices[:, :] = idx
            return values, indices

        x = torch.randn([32, 64], device=DEVICE)
        with pytest.raises(helion.exc.InductorLoweringError) as exc_info:
            topk_dim0(x, 10)
        assert "dim=-1" in str(exc_info.value)

    @pytest.mark.parametrize(
        "dtype,dtype_name",
        [(torch.float64, "float64"), (torch.int64, "int64")],
        ids=["float64", "int64"],
    )
    def test_error_unsupported_dtype(self, dtype, dtype_name):
        """Test that float64 and int64 dtypes raise NotImplementedError."""
        if dtype == torch.float64:
            x = torch.randn([32, 64], device=DEVICE, dtype=dtype)
        else:
            x = torch.randint(0, 100, [32, 64], device=DEVICE, dtype=dtype)

        with pytest.raises(helion.exc.InductorLoweringError) as exc_info:
            topk_2d_kernel(x, 10, True)
        assert dtype_name in str(exc_info.value)

    def test_error_input_size_exceeds_maximum(self):
        """Test that input size > 65536 raises an error."""
        x = torch.randn([4, 100000], device=DEVICE)
        with pytest.raises(Exception) as exc_info:
            topk_2d_kernel(x, 10, True)
        error_msg = str(exc_info.value).lower()
        assert "exceeds" in error_msg or "maximum" in error_msg or "numel" in error_msg

    def test_error_k_not_specialized(self):
        """Test that non-specialized k raises an error."""

        @helion.kernel(config=helion.Config(block_sizes=[64]))
        def topk_no_specialize(
            x: torch.Tensor, k: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            n, m = x.size()
            values = torch.empty([n, k], dtype=x.dtype, device=x.device)
            indices = torch.empty([n, k], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                v, idx = torch.topk(x[tile_n, :], k, dim=-1, largest=True)
                values[tile_n, :] = v
                indices[tile_n, :] = idx
            return values, indices

        x = torch.randn([32, 64], device=DEVICE)
        with pytest.raises(helion.exc.ShapeSpecializingAllocation) as exc_info:
            topk_no_specialize(x, 10)
        assert "specialize" in str(exc_info.value).lower()
