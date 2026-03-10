// Minimal repro: MPP reduce_rows static_assert with execution_simdgroups<2>
//
// Expected: reduce_rows should work with multiple simdgroups, allowing
//           matmul2d with >1 simdgroup to feed into row reductions.
// Actual:   static_assert fires: "reduce_rows requires a single SIMD group"
//           at MPPTensorOpsMatMul2dImpl.h:6920
//
// This forces fused matmul+softmax kernels to use execution_simdgroups<1>
// (32 threads), making the matmul significantly slower than standalone
// matmul which can use execution_simdgroups<N> (N*32 threads).
//
// Filed against: MetalPerformancePrimitives (Metal 4+, macOS 26)
// Workaround: store cooperative_tensor to device memory, do softmax
//             via manual SIMD shuffles, then run second matmul.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Works: execution_simdgroups<1>
kernel void test_sg1(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    uint2 tgid [[thread_position_in_grid]])
{
    auto _t_a = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a, dextents<int32_t, 2>(32, 64));
    auto _t_b = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        b, dextents<int32_t, 2>(64, 32));

    constexpr auto desc = matmul2d_descriptor(
        32, 64, dynamic_length_v<int>,
        false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;

    auto A = _t_a.slice(0, tgid.y * 32);
    auto B = _t_b.slice(0, 0);
    auto coop = op.get_destination_cooperative_tensor<
        decltype(A), decltype(B), float>();
    op.run(A, B, coop);

    // This works with execution_simdgroups<1>
    auto cMax = op.get_row_reduction_destination_cooperative_tensor<
        decltype(A), decltype(B), float>();
    reduce_rows(coop, cMax, reduction_operation::max,
        metal::numeric_limits<float>::lowest());
}

// Fails: execution_simdgroups<2>
// Uncomment to see the static_assert error:
//
// kernel void test_sg2(
//     device float* a [[buffer(0)]],
//     device float* b [[buffer(1)]],
//     uint2 tgid [[thread_position_in_grid]])
// {
//     auto _t_a = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
//         a, dextents<int32_t, 2>(32, 64));
//     auto _t_b = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
//         b, dextents<int32_t, 2>(64, 32));
//
//     constexpr auto desc = matmul2d_descriptor(
//         32, 64, dynamic_length_v<int>,
//         false, false, false,
//         matmul2d_descriptor::mode::multiply);
//     matmul2d<desc, execution_simdgroups<2>> op;  // <-- only change: 2 instead of 1
//
//     auto A = _t_a.slice(0, tgid.y * 32);
//     auto B = _t_b.slice(0, 0);
//     auto coop = op.get_destination_cooperative_tensor<
//         decltype(A), decltype(B), float>();
//     op.run(A, B, coop);
//
//     // static_assert: "reduce_rows requires a single SIMD group"
//     auto cMax = op.get_row_reduction_destination_cooperative_tensor<
//         decltype(A), decltype(B), float>();
//     reduce_rows(coop, cMax, reduction_operation::max,
//         metal::numeric_limits<float>::lowest());
// }
