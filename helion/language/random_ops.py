from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.host_function import HostFunction
from .._compiler.rng_utils import BOX_MULLER_MIN
from .._compiler.rng_utils import HALF_MASK16
from .._compiler.rng_utils import PHILOX_KEY_A
from .._compiler.rng_utils import PHILOX_KEY_B
from .._compiler.rng_utils import PHILOX_ROUND_A
from .._compiler.rng_utils import PHILOX_ROUND_B
from .._compiler.rng_utils import PHILOX_ROUNDS
from .._compiler.rng_utils import TWO_PI
from .._compiler.rng_utils import UINT32_TO_UNIFORM_SCALE
from .._compiler.rng_utils import codegen_rng_seed_expr
from .._compiler.rng_utils import philox_rand_ref
from .._compiler.rng_utils import philox_randint_ref
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from torch._prims_common import DeviceLikeType

    from .._compiler.inductor_lowering import CodegenState
    from .tile_interface import TileInterface

__all__ = ["rand", "randint"]

_ShapeDim = int | torch.SymInt
_ArgDesc = tuple[bool, object]

MASK32 = (1 << 32) - 1
SIGN_BIT32 = 1 << 31
INT32_MAX = (1 << 31) - 1


def _pallas_safe_i32_scalar_like(
    ref: torch.Tensor,
    value: int,
) -> int | torch.Tensor:
    if CompileEnvironment.current().backend.name != "pallas":
        return value
    if value > INT32_MAX:
        value -= 1 << 32
    return torch.scalar_tensor(value, dtype=ref.dtype, device=ref.device)


def _mask_u32(x: torch.Tensor) -> torch.Tensor:
    return x & _pallas_safe_i32_scalar_like(x, MASK32)


def _shape_dim_extent(dim: _ShapeDim) -> _ShapeDim:
    env = CompileEnvironment.current()
    if (block_id := env.get_block_id(dim)) is not None:
        full_extent = env.block_sizes[env.canonical_block_id(block_id)].size
        assert isinstance(full_extent, (int, torch.SymInt))
        return full_extent
    return dim


def _shape_dim_index(
    dim: _ShapeDim,
    *,
    device: torch.device,
) -> torch.Tensor:
    from .tile_ops import tile_index

    env = CompileEnvironment.current()
    if env.get_block_id(dim) is not None:
        assert isinstance(dim, torch.SymInt)
        return _convert_element_type(
            tile_index(cast("TileInterface", dim)), torch.int64
        )  # pyrefly: ignore[bad-argument-type]
    if isinstance(dim, int):
        return torch.arange(dim, device=device, dtype=torch.int64)
    # pyrefly: ignore[no-matching-overload]
    return torch.arange(dim, device=device, dtype=torch.int64)


def _explicit_offset_from_shape(
    shape: list[_ShapeDim],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not shape:
        return torch.arange(1, device=device, dtype=torch.int64).reshape([]) * 0

    extents: list[_ShapeDim] = [_shape_dim_extent(dim) for dim in shape]
    indices = [_shape_dim_index(dim, device=device) for dim in shape]
    strides: list[_ShapeDim] = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * extents[i + 1]

    ndim = len(shape)
    init_shape: list[_ShapeDim] = [1] * ndim
    init_shape[0] = shape[0]
    offset = indices[0].reshape(init_shape) * 0
    for dim, (index, stride) in enumerate(zip(indices, strides, strict=True)):
        if ndim > 1:
            view_shape: list[_ShapeDim] = [1] * ndim
            view_shape[dim] = shape[dim]
            index = index.reshape(view_shape)
        offset = cast("torch.Tensor", offset + index * stride)
    return offset


def _ref_rng_shape_and_offset(
    shape: list[int | RefTile],
    *,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    processed_shape: list[int] = []
    full_extents: list[int] = []
    indices: list[torch.Tensor] = []
    for dim in shape:
        if isinstance(dim, RefTile):
            processed_shape.append(dim.end - dim.begin)
            full_extents.append(dim._extent_end - dim._extent_begin)
            indices.append(
                torch.arange(dim.begin, dim.end, dtype=torch.int64, device=device)
            )
        else:
            size = int(dim)
            processed_shape.append(size)
            full_extents.append(size)
            indices.append(torch.arange(size, dtype=torch.int64, device=device))

    if not processed_shape:
        return [], torch.zeros([], dtype=torch.int64, device=device)

    strides = [1] * len(processed_shape)
    for i in range(len(processed_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * full_extents[i + 1]

    offset = torch.zeros(processed_shape, dtype=torch.int64, device=device)
    ndim = len(processed_shape)
    for dim, (index, stride) in enumerate(zip(indices, strides, strict=True)):
        if ndim > 1:
            view_shape = [1] * ndim
            view_shape[dim] = processed_shape[dim]
            index = index.reshape(view_shape)
        offset = offset + index * stride
    return processed_shape, offset


def _as_int64_scalar(
    value: int | torch.SymInt | torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return _convert_element_type(value, torch.int64)
    if isinstance(value, torch.SymInt):
        return torch.scalar_tensor(cast("int", value), dtype=torch.int64, device=device)
    return torch.scalar_tensor(value, dtype=torch.int64, device=device)


def _uint32_to_signed_int64(x: torch.Tensor) -> torch.Tensor:
    x64 = _convert_element_type(x, torch.int64)
    sign_bit32 = _pallas_safe_i32_scalar_like(x64, SIGN_BIT32)
    return _mask_u32(x64 + sign_bit32) - sign_bit32


def _uint32_to_uniform_float(x: torch.Tensor) -> torch.Tensor:
    signed = _uint32_to_signed_int64(x)
    magnitude = torch.where(signed < 0, -signed - 1, signed)
    return _convert_element_type(magnitude, torch.float32) * UINT32_TO_UNIFORM_SCALE


def _convert_element_type(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.prims.convert_element_type.default(x, dtype)


def _mulhi_lo_u32(
    a: int | torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a64 = _convert_element_type(a, torch.int64) if isinstance(a, torch.Tensor) else a
    b64 = _convert_element_type(b, torch.int64)
    a0 = a64 & HALF_MASK16
    a1 = (a64 >> 16) & HALF_MASK16
    b0 = b64 & HALF_MASK16
    b1 = (b64 >> 16) & HALF_MASK16

    t = a0 * b0
    w0 = t & HALF_MASK16
    k = t >> 16

    t = a1 * b0 + k
    w1 = t & HALF_MASK16
    w2 = t >> 16

    t = a0 * b1 + w1
    lo = _mask_u32(((t & HALF_MASK16) << 16) | w0)
    hi = _mask_u32(a1 * b1 + w2 + (t >> 16))
    return hi, lo


def _philox_uint32x4(
    seed: int | torch.SymInt | torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = offset.device
    offset64 = _convert_element_type(offset, torch.int64)
    seed64 = _as_int64_scalar(seed, device=device)

    c0 = _mask_u32(offset64)
    c1 = _mask_u32(offset64 >> 32)
    c2 = c0 * 0
    c3 = c0 * 0
    k0 = _mask_u32(seed64)
    k1 = _mask_u32(seed64 >> 32)

    for _ in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhi_lo_u32(PHILOX_ROUND_B, c2)
        hi1, lo1 = _mulhi_lo_u32(PHILOX_ROUND_A, c0)
        c0 = _mask_u32(hi0 ^ c1 ^ k0)
        c1 = lo0
        c2 = _mask_u32(hi1 ^ c3 ^ k1)
        c3 = lo1
        k0 = _mask_u32(k0 + _pallas_safe_i32_scalar_like(k0, PHILOX_KEY_A))
        k1 = _mask_u32(k1 + _pallas_safe_i32_scalar_like(k1, PHILOX_KEY_B))

    return c0, c1, c2, c3


def _philox_rand_from_seed_and_offset(
    seed: int | torch.SymInt | torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    c0, _, _, _ = _philox_uint32x4(seed, offset)
    return _uint32_to_uniform_float(c0)


def _philox_randn_from_seed_and_offset(
    seed: int | torch.SymInt | torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    c0, c1, _, _ = _philox_uint32x4(seed, offset)
    u1 = torch.clamp_min(_uint32_to_uniform_float(c0), BOX_MULLER_MIN)
    u2 = _uint32_to_uniform_float(c1)
    minus_two = torch.full([], -2.0, dtype=torch.float32, device=u1.device)
    tau = torch.full([], TWO_PI, dtype=torch.float32, device=u1.device)
    radius = torch.sqrt(torch.log(u1) * minus_two)
    return radius * torch.cos(u2 * tau)


def _philox_randint_from_seed_and_offset(
    seed: int | torch.SymInt | torch.Tensor,
    offset: torch.Tensor,
    *,
    low: int,
    high: int,
) -> torch.Tensor:
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    c0, _, _, _ = _philox_uint32x4(seed, offset)
    signed = _uint32_to_signed_int64(c0)
    magnitude = torch.where(signed < 0, -signed, signed)
    return _convert_element_type(low + (magnitude % (high - low)), torch.int32)


@_decorators.api()
def _rng_seed(index: int) -> torch.Tensor:
    raise AssertionError("this should never be called directly")


@_decorators.register_fake(_rng_seed)
def _(index: int) -> torch.Tensor:
    env = CompileEnvironment.current()
    return torch.empty([], dtype=torch.int64, device=env.device)


@_decorators.codegen(_rng_seed, "common")
def _(state: CodegenState) -> ast.AST:
    seed_index = state.proxy_arg(0)
    assert isinstance(seed_index, int)
    return codegen_rng_seed_expr(state.codegen, seed_index)


def _next_rng_seed_slot() -> int:
    return HostFunction.current().allocate_rng_seed_slot()


def _next_ref_rng_seed_slot() -> int:
    from ..runtime.ref_mode import RefModeContext

    return RefModeContext.current().allocate_rng_seed_slot()


def _ref_rng_seed(index: int) -> torch.Tensor:
    from ..runtime.ref_mode import RefModeContext

    return RefModeContext.current().lookup_rng_seed(index)


def _add_rewrite_desc(
    descriptors: list[_ArgDesc],
    value: object,
) -> _ArgDesc:
    if isinstance(value, Node):
        desc: _ArgDesc = (True, len(descriptors))
        descriptors.append((True, value))
        return desc
    return (False, value)


def _shape_rewrite_desc(
    shape_arg: object,
) -> tuple[list[_ArgDesc], list[_ArgDesc]]:
    assert isinstance(shape_arg, (list, tuple, torch.Size))
    descriptors: list[_ArgDesc] = []
    shape_desc = [_add_rewrite_desc(descriptors, dim) for dim in shape_arg]
    return descriptors, shape_desc


def decompose_rand(
    shape: list[int | torch.SymInt],
    *,
    seed: int | torch.SymInt | torch.Tensor,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    offset = _explicit_offset_from_shape(shape, device=env.device)
    return _philox_rand_from_seed_and_offset(seed, offset)


def decompose_randint(
    shape: list[int | torch.SymInt],
    *,
    low: int,
    high: int,
    seed: int | torch.SymInt | torch.Tensor,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    offset = _explicit_offset_from_shape(shape, device=env.device)
    return _philox_randint_from_seed_and_offset(
        seed,
        offset,
        low=low,
        high=high,
    )


def _canonicalize_rng_device(device: DeviceLikeType) -> torch.device:
    requested = torch.device(device)
    env_device = torch.device(CompileEnvironment.current().device)
    if requested.index is None and requested.type == env_device.type:
        requested = torch.device(requested.type, env_device.index)
    return requested


def _assert_rng_device_matches_env(device: DeviceLikeType | None) -> None:
    if device is None:
        return
    env_device = torch.device(CompileEnvironment.current().device)
    requested = _canonicalize_rng_device(device)
    assert requested == env_device, f"expected {env_device}, got {requested}"


def _normalize_implicit_rng_request(
    shape: list[_ShapeDim],
    *,
    dtype: torch.dtype | None,
    default_dtype: torch.dtype,
    device: DeviceLikeType | None,
    requires_grad: object = False,
) -> tuple[list[_ShapeDim], torch.dtype]:
    _assert_rng_device_matches_env(device)
    assert not requires_grad
    resolved_dtype = default_dtype if dtype is None else dtype
    if not resolved_dtype.is_floating_point:
        raise NotImplementedError(
            f"implicit RNG only supports floating-point dtypes, got {resolved_dtype}"
        )
    return shape, resolved_dtype


def _runtime_seeded_random(
    shape: list[_ShapeDim],
    *,
    dtype: torch.dtype,
    sampler: Callable[
        [int | torch.SymInt | torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> torch.Tensor:
    seed_slot = _next_rng_seed_slot()
    seed = _rng_seed(seed_slot)
    env = CompileEnvironment.current()
    offset = _explicit_offset_from_shape(shape, device=env.device)
    values = sampler(seed, offset)
    if dtype != torch.float32:
        values = _convert_element_type(values, dtype)
    return values


def _implicit_random(
    shape: list[_ShapeDim],
    *,
    dtype: torch.dtype | None,
    default_dtype: torch.dtype,
    device: DeviceLikeType | None,
    requires_grad: object = False,
    sampler: Callable[
        [int | torch.SymInt | torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> torch.Tensor:
    shape, dtype = _normalize_implicit_rng_request(
        shape,
        dtype=dtype,
        default_dtype=default_dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return _runtime_seeded_random(shape, dtype=dtype, sampler=sampler)


def _ref_runtime_seeded_random(
    shape: list[int | RefTile],
    *,
    dtype: torch.dtype,
    rng_device: torch.device,
    sampler: Callable[
        [int | torch.SymInt | torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> torch.Tensor:
    seed_slot = _next_ref_rng_seed_slot()
    seed = _ref_rng_seed(seed_slot)
    processed_shape, offset = _ref_rng_shape_and_offset(shape, device=rng_device)
    values = sampler(seed, offset).reshape(processed_shape)
    if dtype != torch.float32:
        values = values.to(dtype)
    return values.to(device=rng_device)


def ref_implicit_random(
    shape: list[int | RefTile],
    *,
    dtype: torch.dtype | None,
    default_dtype: torch.dtype,
    device: DeviceLikeType | None,
    requires_grad: object = False,
    normal: bool,
) -> torch.Tensor:
    rng_shape, resolved_dtype = _normalize_implicit_rng_request(
        cast("list[_ShapeDim]", shape),
        dtype=dtype,
        default_dtype=default_dtype,
        device=device,
        requires_grad=requires_grad,
    )
    env = CompileEnvironment.current()
    rng_device = (
        torch.device(env.device) if device is None else _canonicalize_rng_device(device)
    )
    sampler = (
        _philox_randn_from_seed_and_offset
        if normal
        else _philox_rand_from_seed_and_offset
    )
    return _ref_runtime_seeded_random(
        cast("list[int | RefTile]", rng_shape),
        dtype=resolved_dtype,
        rng_device=rng_device,
        sampler=sampler,
    )


def _rewrite_runtime_args(
    descriptors: list[_ArgDesc],
) -> tuple[list[Node], list[object]]:
    runtime_args: list[Node] = []
    example_args: list[object] = []
    for is_dynamic, value in descriptors:
        if is_dynamic:
            assert isinstance(value, Node)
            runtime_args.append(value)
            example_args.append(value.meta["val"])
    return runtime_args, example_args


def _copy_rewrite_subgraph(
    graph: torch.fx.Graph,
    helper_graph: torch.fx.Graph,
    *,
    before: Node,
    runtime_args: list[Node],
) -> Node:
    helper_placeholders = list(helper_graph.find_nodes(op="placeholder"))
    helper_getattrs = list(helper_graph.find_nodes(op="get_attr"))
    if helper_getattrs:
        raise NotImplementedError(
            f"unexpected helper constants: {[node.target for node in helper_getattrs]!r}"
        )
    with graph.inserting_before(before):
        copied = graph.graph_copy(
            helper_graph,
            dict(zip(helper_placeholders, runtime_args, strict=True)),
        )
    assert isinstance(copied, Node)
    return copied


def _trace_rewrite_subgraph(
    graph: torch.fx.Graph,
    node: Node,
    helper: Callable[..., torch.Tensor],
    descriptors: list[_ArgDesc],
) -> Node:
    from .._compiler.device_ir import _make_fx

    runtime_args, example_args = _rewrite_runtime_args(descriptors)
    helper_graph = _make_fx(helper, *example_args)
    location = node.meta.get("location")
    if location is not None:
        for helper_node in helper_graph.nodes:
            if helper_node.op == "call_function" and "location" not in helper_node.meta:
                helper_node.meta["location"] = location
    return _copy_rewrite_subgraph(
        graph,
        helper_graph,
        before=node,
        runtime_args=runtime_args,
    )


def _resolve_rewrite_arg(
    flat_args: tuple[object, ...],
    desc: _ArgDesc,
) -> object:
    return flat_args[cast("int", desc[1])] if desc[0] else desc[1]


def _resolve_shape_desc(
    flat_args: tuple[object, ...],
    desc: _ArgDesc,
) -> int | torch.SymInt:
    value = _resolve_rewrite_arg(flat_args, desc)
    assert isinstance(value, (int, torch.SymInt))
    return value


def _resolve_int_desc(
    flat_args: tuple[object, ...],
    desc: _ArgDesc,
) -> int:
    value = _resolve_rewrite_arg(flat_args, desc)
    assert isinstance(value, int)
    return value


def _resolve_seed_desc(
    flat_args: tuple[object, ...],
    desc: _ArgDesc,
) -> int | torch.SymInt | torch.Tensor:
    return cast(
        "int | torch.SymInt | torch.Tensor", _resolve_rewrite_arg(flat_args, desc)
    )


def _random_rewrite_nodes(graph: torch.fx.Graph) -> list[Node]:
    targets = (
        rand,
        randint,
        torch.ops.aten.rand.default,
        torch.ops.aten.randn.default,
        torch.ops.aten.rand_like.default,
        torch.ops.aten.randn_like.default,
    )
    return sorted(
        node
        for target in targets
        for node in graph.find_nodes(
            op="call_function",
            target=target,
            sort=False,
        )
    )


def rewrite_implicit_random_ops(graph: torch.fx.Graph) -> None:
    for node in _random_rewrite_nodes(graph):
        if node.target is rand:
            shape_arg = node.args[0]
            descriptors, shape_desc = _shape_rewrite_desc(shape_arg)
            seed_desc = _add_rewrite_desc(descriptors, node.args[1])

            def helper(
                *flat_args: object,
                shape_desc: tuple[_ArgDesc, ...] = tuple(shape_desc),
                seed_desc: _ArgDesc = seed_desc,
            ) -> torch.Tensor:
                shape = [_resolve_shape_desc(flat_args, desc) for desc in shape_desc]
                seed = _resolve_seed_desc(flat_args, seed_desc)
                return decompose_rand(shape, seed=seed)

            replacement = _trace_rewrite_subgraph(graph, node, helper, descriptors)
        elif node.target is randint:
            shape_arg = node.args[0]
            descriptors, shape_desc = _shape_rewrite_desc(shape_arg)
            low_desc = _add_rewrite_desc(descriptors, node.args[1])
            high_desc = _add_rewrite_desc(descriptors, node.args[2])
            seed_desc = _add_rewrite_desc(descriptors, node.args[3])

            def helper(
                *flat_args: object,
                shape_desc: tuple[_ArgDesc, ...] = tuple(shape_desc),
                low_desc: _ArgDesc = low_desc,
                high_desc: _ArgDesc = high_desc,
                seed_desc: _ArgDesc = seed_desc,
            ) -> torch.Tensor:
                shape = [_resolve_shape_desc(flat_args, desc) for desc in shape_desc]
                low_val = _resolve_int_desc(flat_args, low_desc)
                high_val = _resolve_int_desc(flat_args, high_desc)
                seed = _resolve_seed_desc(flat_args, seed_desc)
                return decompose_randint(
                    shape,
                    low=low_val,
                    high=high_val,
                    seed=seed,
                )

            replacement = _trace_rewrite_subgraph(graph, node, helper, descriptors)
        elif node.target in {torch.ops.aten.rand.default, torch.ops.aten.randn.default}:
            descriptors, shape_desc = _shape_rewrite_desc(node.args[0])
            dtype = node.kwargs.get("dtype", torch.float32)
            assert dtype is None or isinstance(dtype, torch.dtype)
            device = node.kwargs.get("device")
            requires_grad = node.kwargs.get("requires_grad", False)

            if node.target is torch.ops.aten.rand.default:
                sampler = _philox_rand_from_seed_and_offset
            else:
                sampler = _philox_randn_from_seed_and_offset

            def helper(
                *flat_args: object,
                shape_desc: tuple[_ArgDesc, ...] = tuple(shape_desc),
                dtype: torch.dtype | None = dtype,
                device_arg: object | None = device,
                requires_grad: object = requires_grad,
                sampler: Callable[
                    [int | torch.SymInt | torch.Tensor, torch.Tensor],
                    torch.Tensor,
                ] = sampler,
            ) -> torch.Tensor:
                shape = [_resolve_shape_desc(flat_args, desc) for desc in shape_desc]
                return _implicit_random(
                    shape,
                    dtype=dtype,
                    default_dtype=torch.float32,
                    device=cast("DeviceLikeType | None", device_arg),
                    requires_grad=requires_grad,
                    sampler=sampler,
                )

            replacement = _trace_rewrite_subgraph(graph, node, helper, descriptors)
        elif node.target in {
            torch.ops.aten.rand_like.default,
            torch.ops.aten.randn_like.default,
        }:
            tensor = node.args[0]
            assert isinstance(tensor, Node)
            descriptors: list[_ArgDesc] = [(True, tensor)]
            dtype = node.kwargs.get("dtype")
            assert dtype is None or isinstance(dtype, torch.dtype)
            device = node.kwargs.get("device")
            requires_grad = node.kwargs.get("requires_grad", False)

            if node.target is torch.ops.aten.rand_like.default:
                sampler = _philox_rand_from_seed_and_offset
            else:
                sampler = _philox_randn_from_seed_and_offset

            def helper(
                *flat_args: object,
                dtype: torch.dtype | None = dtype,
                device_arg: object | None = device,
                requires_grad: object = requires_grad,
                sampler: Callable[
                    [int | torch.SymInt | torch.Tensor, torch.Tensor],
                    torch.Tensor,
                ] = sampler,
            ) -> torch.Tensor:
                (input_tensor,) = flat_args
                assert isinstance(input_tensor, torch.Tensor)
                return _implicit_random(
                    [*input_tensor.shape],
                    dtype=dtype,
                    default_dtype=input_tensor.dtype,
                    device=cast("DeviceLikeType | None", device_arg),
                    requires_grad=requires_grad,
                    sampler=sampler,
                )

            replacement = _trace_rewrite_subgraph(graph, node, helper, descriptors)
        else:
            continue

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


@_decorators.api(tiles_as_sizes=True)
def rand(
    shape: list[object],
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    hl.rand provides a Philox-based pseudorandom number generator (PRNG) that
    operates independently of PyTorch's global random seed. Instead, it
    requires an explicit seed argument. Offsets are derived from the full
    logical sizes of the tiles specified in the shape argument.

    Args:
        shape: A list of sizes for the output tensor
        seed: A single element int64 tensor or int literal
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of float32 dtype filled with uniform
        random values in [0, 1)

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros_like(x)
                (m,) = x.shape
                for tile_m in hl.tile(m):
                    output[tile_m] = hl.rand([tile_m], seed=42)
                return output
    """
    raise NotInsideKernel


@_decorators.register_fake(rand)
def _rand_fake(
    shape: list[int | torch.SymInt],
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=torch.float32,
        device=env.device if device is None else device,
    )


@_decorators.get_masked_value(rand)
def _(node: torch.fx.Node) -> float:
    return 0


@_decorators.ref(rand)
def _(
    shape: list[int | RefTile],
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    rng_device = env.device if device is None else device
    processed_shape, offset = _ref_rng_shape_and_offset(shape, device=rng_device)
    return philox_rand_ref(seed, offset).reshape(processed_shape).to(device=rng_device)


@_decorators.api(tiles_as_sizes=True)
def randint(
    shape: list[object],
    low: int,
    high: int,
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    hl.randint provides a Philox-based pseudorandom integer generator (PRNG)
    that operates independently of PyTorch's global random seed. Instead, it
    requires an explicit seed argument. Offsets are derived from the full
    logical sizes of the tiles specified in the shape argument.

    Args:
        shape: A list of sizes for the output tensor
        low: Lowest integer to be drawn from the distribution (inclusive)
        high: One above the highest integer to be drawn from the distribution
            (exclusive)
        seed: A single element int64 tensor or int literal
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of int32 dtype filled with random
        integers in [low, high)

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(x: torch.Tensor) -> torch.Tensor:
                output = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
                (m,) = x.shape
                for tile_m in hl.tile(m):
                    output[tile_m] = hl.randint([tile_m], low=0, high=10, seed=42)
                return output
    """
    raise NotInsideKernel


@_decorators.register_fake(randint)
def _randint_fake(
    shape: list[int | torch.SymInt],
    low: int,
    high: int,
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=torch.int32,
        device=env.device if device is None else device,
    )


@_decorators.get_masked_value(randint)
def _(node: torch.fx.Node) -> int:
    return 0


@_decorators.ref(randint)
def _(
    shape: list[int | RefTile],
    low: int,
    high: int,
    seed: int | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    rng_device = env.device if device is None else device
    processed_shape, offset = _ref_rng_shape_and_offset(shape, device=rng_device)
    return (
        philox_randint_ref(seed, offset, low, high)
        .reshape(processed_shape)
        .to(device=rng_device)
    )
