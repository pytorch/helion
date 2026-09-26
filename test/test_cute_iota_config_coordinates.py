from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import itertools
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_iota_symbolic_hints import _copy

import helion
from helion import exc
from helion._compiler.cute.aten_lowering import codegen_iota_cute
from helion._compiler.cute.cute_reshape import _get_dim_local_coord
from helion._compiler.cute.cute_reshape import _resolve_tile_extent
from helion._compiler.cute.cute_reshape import codegen_cute_virtual_clone
from helion._compiler.cute.indexing import CuteShapeChainView
from helion._testing import skipUnlessBackends
import helion.language as hl

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def _cpu_codegen_target() -> Generator[None, None, None]:
    """Keep CPU bindings independent of visible or already-initialized CUDA."""
    initialized = torch.cuda.is_initialized()
    with (
        _mock_cuda_unavailable(),
        patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("CPU only")
        ) as current_device,
        patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU only"),
        ) as capability,
        patch.object(
            torch.cuda,
            "get_device_properties",
            side_effect=AssertionError("CPU only"),
        ) as properties,
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU only")
        ) as initialization,
        patch(
            "cutlass.cute.compile", side_effect=AssertionError("CPU only")
        ) as compilation,
    ):
        yield
        current_device.assert_not_called()
        capability.assert_not_called()
        properties.assert_not_called()
        initialization.assert_not_called()
        compilation.assert_not_called()
        assert torch.cuda.is_initialized() == initialized


@pytest.fixture
def _cpu_codegen() -> Generator[None, None, None]:
    with _cpu_codegen_target():
        yield


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _packed_int4(
    A: torch.Tensor, B_packed: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    m, k = A.shape
    _, n = B_packed.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(k)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block_n):
            acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=block_k):
                begin = tile_k.begin
                packed = B_packed[begin // 2 : begin // 2 + block_k // 2, tile_n]
                shift = hl.full((1,), 4, dtype=torch.int8)
                low = (packed << shift) >> shift
                high = packed >> shift
                stacked = torch.stack((low.to(A.dtype), high.to(A.dtype)), dim=2)
                stacked = stacked.permute(0, 2, 1)
                unpacked = stacked.reshape([block_k, block_n])
                acc = hl.dot(A[tile_m, tile_k], unpacked, acc=acc)
            C[tile_m, tile_n] = acc
    return C


def _inputs(m: int, n: int, k: int, *, strided: bool = False):
    generator = torch.Generator().manual_seed(8123)
    a = torch.randn((m, k), generator=generator, dtype=torch.bfloat16)
    storage = torch.randint(
        -128, 128, (k // 2, n * 2), dtype=torch.int8, generator=generator
    )
    packed = storage[:, ::2] if strided else storage[:, :n].contiguous()
    c_storage = torch.full((m, n * 2), -12345.0)
    c = c_storage[:, ::2] if strided else c_storage[:, :n].contiguous()
    return a, packed, c


class _Pointer:
    def __init__(self, tensor: _Tensor, offset: int) -> None:
        self.tensor = tensor
        self.offset = int(offset)

    def __add__(self, offset):
        return _Pointer(self.tensor, self.offset + int(offset))

    def load(self):
        assert self.offset in self.tensor.owned, ("load", self.offset)
        return self.tensor.storage[self.offset]

    def store(self, value) -> None:
        assert self.offset in self.tensor.owned, ("store", self.offset)
        self.tensor.storage[self.offset] = value
        self.tensor.writes[self.offset] += 1


class _Tensor:
    def __init__(self, value: torch.Tensor) -> None:
        size = value.untyped_storage().nbytes() // value.element_size()
        storage = value.as_strided((size,), (1,), storage_offset=0)
        self.storage = (
            storage.numpy().copy()
            if value.dtype == torch.int8
            else storage.float().numpy().copy()
        )
        self.initial = self.storage.copy()
        self.writes = np.zeros(size, dtype=np.int32)
        self.layout = SimpleNamespace(shape=tuple(value.shape), stride=value.stride())
        self.positions = [
            value.storage_offset()
            + sum(i * s for i, s in zip(index, value.stride(), strict=True))
            for index in itertools.product(*(range(size) for size in value.shape))
        ]
        self.owned = set(self.positions)
        self.iterator = _Pointer(self, value.storage_offset())

    def values(self):
        return self.storage[self.positions].reshape(self.layout.shape)


def _exact_nibble_to_bf16(value):
    # This generated program converts only signed four-bit integers to BF16;
    # all are exactly representable. Input A is already rounded by real Torch.
    assert -8 <= int(value) <= 7 and value == int(value)
    return np.float32(value)


def _interpret_scalar_kernel(
    code: str,
    inputs: tuple[torch.Tensor, ...],
    names: tuple[str, ...] = ("A", "B_packed", "C"),
):
    """Execute the emitted scalar AST, including masks and typed pointer math."""
    module = ast.parse(code)
    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    assert len(kernels) == 1
    kernel = copy.deepcopy(kernels[0])
    kernel.decorator_list = []
    constants = [node for node in module.body if isinstance(node, ast.Assign)]
    arch = SimpleNamespace(thread_idx=lambda: (0, 0, 0), block_idx=lambda: (0, 0, 0))
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=np.int32,
            Int8=np.int8,
            Float32=np.float32,
            BFloat16=_exact_nibble_to_bf16,
        ),
        "cute": SimpleNamespace(arch=arch),
        "operator": operator,
    }
    exec(
        compile(
            ast.Module(body=[*constants, kernel], type_ignores=[]),
            "<scalar CuTe oracle>",
            "exec",
        ),
        namespace,
    )
    tensor_map = dict(zip(names, map(_Tensor, inputs), strict=True))
    launch = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    block_node = next(kw.value for kw in launch.keywords if kw.arg == "block")
    block = ast.literal_eval(block_node)
    assert block[1:] == (1, 1), "unexpected independent coordinate or reduction axis"
    grid = eval(compile(ast.Expression(launch.args[1]), "<grid>", "eval"), namespace)
    function = namespace[kernel.name]
    parameters = [tensor_map[arg.arg] for arg in kernel.args.args]
    for cta in range(grid[0]):
        arch.block_idx = lambda cta=cta: (cta, 0, 0)
        for thread in range(block[0]):
            arch.thread_idx = lambda thread=thread: (thread, 0, 0)
            function(*parameters)
    output = tensor_map[names[-1]]
    assert np.all(output.writes[output.positions] == 1)
    padding = [i for i in range(output.storage.size) if i not in output.owned]
    np.testing.assert_array_equal(output.storage[padding], output.initial[padding])
    return output.values()


def _serial_reference(a: torch.Tensor, packed: torch.Tensor):
    raw = packed.to(torch.int16)
    low = ((raw & 15) ^ 8) - 8
    high = raw >> 4
    b = torch.stack((low, high), dim=1).flatten(0, 1).float().numpy()
    x = a.float().numpy()
    result = np.zeros((a.size(0), packed.size(1)), dtype=np.float32)
    for k in range(a.size(1)):
        result = np.float32(result + np.float32(x[:, k, None] * b[k, None, :]))
    return result


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize(
    "shape,blocks,strided",
    [
        ((32, 32, 32), (16, 16, 16), False),
        ((7, 13, 30), (8, 16, 4), False),
        ((13, 9, 34), (16, 8, 8), True),
    ],
)
@skipUnlessBackends(["cute"])
def test_packed_int4_generated_scalar_ast_matches_serial_product(
    shape, blocks, strided
):
    inputs = _inputs(*shape, strided=strided)
    bound = _packed_int4.bind(inputs)
    symbols = [info.var.node._expr for info in bound.env.block_sizes]
    code = bound.to_code(helion.Config(block_sizes=list(blocks)))
    assert "arange_lane" not in code
    np.testing.assert_array_equal(
        _interpret_scalar_kernel(code, inputs), _serial_reference(*inputs[:2])
    )
    assert all(bound.env.shape_env.replace(symbol) == symbol for symbol in symbols)


@pytest.mark.usefixtures("_cpu_codegen")
@skipUnlessBackends(["cute"])
def test_selected_tile_quotients_have_no_shape_env_guards():
    bound = _packed_int4.bind(_inputs(32, 32, 32))
    with bound.env, bound.host_function:
        n, k, m = [info.var for info in bound.env.block_sizes]
        unknown = bound.env.shape_env.create_unbacked_symint()
        for sizes in ([16, 32, 8], [32, 16, 4], [8, 64, 16]):
            config = helion.Config(block_sizes=sizes)
            assert _resolve_tile_extent(k // 2, bound.env, config) == sizes[1] // 2
            assert _resolve_tile_extent(n * (k // 2), bound.env, config) == sizes[0] * (
                sizes[1] // 2
            )
            assert _resolve_tile_extent(unknown // 2, bound.env, config) is None
            assert _resolve_tile_extent(k + unknown, bound.env, config) is None
            for symbol in (n, k, m, unknown):
                assert (
                    bound.env.shape_env.replace(symbol.node._expr) == symbol.node._expr
                )


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize("sizes", [(32, 128, 64), (128, 32, 64)])
@skipUnlessBackends(["cute"])
def test_block_iota_selected_configs_copy_each_element_once(sizes):
    x = torch.arange(257, dtype=torch.float32)
    bound = _copy.bind((x,))
    for size in sizes:
        code = bound.to_code(helion.Config(block_sizes=[size]))
        out = torch.full_like(x, float("nan"))
        result = _interpret_scalar_kernel(code, (x, out), ("x", "out"))
        np.testing.assert_array_equal(result, x.numpy())


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize(
    "extent,active,expected",
    [(16, True, True), (15, True, False), (0, True, False), (16, False, False)],
)
@skipUnlessBackends(["cute"])
def test_quotient_coordinate_requires_same_active_divisible_axis(
    extent, active, expected
):
    bound = _packed_int4.bind(_inputs(32, 32, 32))
    with bound.env, bound.host_function:
        k = bound.env.block_sizes[1].var
        value = torch.empty((k // 2,))
        strategy = SimpleNamespace()
        cg = SimpleNamespace(
            active_device_loops={1: [SimpleNamespace(strategy=strategy)]}
            if active
            else {},
            current_grid_state=None,
            device_function=SimpleNamespace(
                resolved_block_size=lambda block_id: extent
            ),
            index_var=lambda block_id: f"index_{block_id}",
            offset_var=lambda block_id: f"offset_{block_id}",
        )
        coordinate = _get_dim_local_coord(cg, value, 0)
        assert (coordinate != "cutlass.Int32(0)") is expected
        if expected:
            for start in (0, 16, 64):
                for lane in range(extent):
                    result = eval(
                        coordinate,
                        {
                            "cutlass": SimpleNamespace(Int32=int),
                            "index_1": start + lane,
                            "offset_1": start,
                        },
                    )
                    assert result == lane // 2


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize("k_extent,k_active", [(16, False), (15, True)])
@skipUnlessBackends(["cute"])
def test_derived_iota_cannot_borrow_unrelated_axis_with_equal_extent(
    k_extent, k_active
):
    bound = _packed_int4.bind(_inputs(32, 32, 32))
    with bound.env, bound.host_function:
        k = bound.env.block_sizes[1].var
        graph = torch.fx.Graph()
        length = graph.placeholder("length")
        length.meta["val"] = k // 2
        node = graph.call_function(
            torch.ops.prims.iota.default,
            (length,),
            {"start": 0, "step": 1, "dtype": torch.int32},
        )
        node.meta["val"] = torch.empty((k // 2,), dtype=torch.int32)
        graph.output(node)
        loops = {0: [SimpleNamespace(strategy=SimpleNamespace())]}
        if k_active:
            loops[1] = [SimpleNamespace(strategy=SimpleNamespace())]
        config = helion.Config(block_sizes=[k_extent // 2, k_extent, 16])
        cg = SimpleNamespace(
            active_device_loops=loops,
            current_grid_state=None,
            codegen_graphs=[],
            device_function=SimpleNamespace(
                config=config,
                resolved_block_size=lambda block_id: config.block_sizes[block_id],
            ),
            index_var=lambda block_id: f"index_{block_id}",
            offset_var=lambda block_id: f"offset_{block_id}",
        )
        with (
            patch("helion._compiler.generate_ast.GenerateAST", SimpleNamespace),
            pytest.raises(exc.BackendUnsupported, match="proven active coordinate"),
        ):
            codegen_iota_cute(SimpleNamespace(cg=cg), node)


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize(
    "memory_format", [None, torch.contiguous_format, torch.preserve_format]
)
def test_virtual_clone_preserves_immutable_leaf_and_shape_chain(memory_format):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    clone = graph.call_function(
        torch.ops.aten.clone.default, (source,), {"memory_format": memory_format}
    )
    view = graph.call_function(torch.ops.aten.view.default, (clone, [8, 4]))
    graph.output(view)
    value = CuteShapeChainView(source)
    ctx = SimpleNamespace(env={source: value})
    assert codegen_cute_virtual_clone(ctx, clone) is value
    ordinary = ast.Name(id="already_loaded", ctx=ast.Load())
    assert (
        codegen_cute_virtual_clone(SimpleNamespace(env={source: ordinary}), clone)
        is None
    )


@pytest.mark.usefixtures("_cpu_codegen")
@pytest.mark.parametrize(
    "mutation", ["non_shape_use", "unknown_keyword", "channels_last"]
)
def test_virtual_clone_declines_unproved_uses(mutation):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    kwargs = {"memory_format": torch.contiguous_format}
    if mutation == "unknown_keyword":
        kwargs["unknown"] = True
    elif mutation == "channels_last":
        kwargs["memory_format"] = torch.channels_last
    clone = graph.call_function(torch.ops.aten.clone.default, (source,), kwargs)
    if mutation == "non_shape_use":
        consumer = graph.call_function(torch.ops.aten.add.Tensor, (clone, clone))
    else:
        consumer = graph.call_function(torch.ops.aten.view.default, (clone, [8, 4]))
    graph.output(consumer)
    with pytest.raises(exc.BackendUnsupported, match="shape-only consumers"):
        codegen_cute_virtual_clone(
            SimpleNamespace(env={source: CuteShapeChainView(source)}), clone
        )


@pytest.mark.parametrize("initialized", [False, True])
@skipUnlessBackends(["cute"])
def test_cpu_codegen_fixture_overrides_visible_cuda_metadata(initialized):
    with (
        patch.dict("os.environ", {"HELION_CUTE_FLASH": "1"}),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        patch.object(torch.cuda, "is_initialized", return_value=initialized),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU only")
        ) as initialization,
        patch(
            "cutlass.cute.compile", side_effect=AssertionError("CPU only")
        ) as compilation,
    ):
        assert torch.cuda.is_available()
        assert torch.cuda.current_device() == 0
        assert torch.cuda.get_device_capability() == (10, 0)
        with _cpu_codegen_target():
            assert not torch.cuda.is_available()
            inputs = _inputs(7, 13, 30)
            bound = _packed_int4.bind(inputs)
            code = bound.to_code(helion.Config(block_sizes=[8, 16, 4]))
            np.testing.assert_array_equal(
                _interpret_scalar_kernel(code, inputs), _serial_reference(*inputs[:2])
            )
        assert torch.cuda.is_available()
        assert torch.cuda.current_device() == 0
        assert torch.cuda.get_device_capability() == (10, 0)
        assert torch.cuda.is_initialized() == initialized
        initialization.assert_not_called()
        compilation.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,blocks", [((32, 32, 32), [16, 16, 16]), ((7, 13, 30), [8, 16, 4])]
)
@skipUnlessBackends(["cute"])
def test_packed_int4_signed_values_match_exact_matrix_product(shape, blocks):
    m, n, k = shape
    generator = torch.Generator().manual_seed(913)
    a = torch.randint(-2, 3, (m, k), generator=generator).to(
        dtype=torch.bfloat16, device=CUDA_DEVICE
    )
    packed = torch.randint(
        -128, 128, (k // 2, n), dtype=torch.int8, generator=generator
    ).cuda()
    out = torch.full((m, n), float("nan"), device=CUDA_DEVICE)
    raw = packed.to(torch.int16)
    unpacked = (
        torch.stack((((raw & 15) ^ 8) - 8, raw >> 4), dim=1).flatten(0, 1).float()
    )
    expected = a.float() @ unpacked
    bound = _packed_int4.bind((a, packed, out))
    bound.set_config(helion.Config(block_sizes=blocks))
    result = bound(a, packed, out)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
