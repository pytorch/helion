from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_philox_stream import cpu_only  # noqa: F401

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.philox_stream import stage_explicit_seed
from helion._compiler.host_function import HostFunction
from helion._compiler.variable_origin import NameOrigin
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.runtime.cute.rng import stage_explicit_seed as runtime_stage_seed

pytestmark = pytest.mark.usefixtures("cpu_only", "no_native")


@pytest.fixture
def no_native(monkeypatch):
    monkeypatch.delenv("HELION_CUTE_RNG_STREAM", raising=False)
    initialized_before = torch.cuda.is_initialized()
    with patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")):
        yield
    assert torch.cuda.is_initialized() == initialized_before


def _original_function(filename, method, function):
    # Read the real failed test's function, without running its CUDA fixture or
    # replacing its body. Preserve original line numbers for source inspection.
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text())
    test = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    node = deepcopy(
        next(
            node
            for node in test.body
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
    )
    node.decorator_list = []
    namespace = {"torch": torch, "hl": hl, "__name__": __name__}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[function]


# Every failed test has its actual kernel body represented here. Include both
# tile orders, both reduction configurations and both explicit/implicit orders.
@pytest.mark.parametrize(
    "method,function,shape,blocks,static,reduction",
    (
        ("rand_randint_1d", "rng_kernel_1d", (1024,), [256], False, None),
        ("rand_randint_2d", "rng_kernel_2d", (256, 256), [32, 32], False, None),
        ("rand_3d", "rand_kernel_tiled_3d", (16, 32, 64), [4, 8, 32], False, None),
        (
            "rand_block_size_determinism",
            "rand_kernel_2d",
            (64, 128),
            [8, 8],
            False,
            None,
        ),
        (
            "rand_block_size_determinism",
            "rand_kernel_2d",
            (64, 128),
            [32, 32],
            False,
            None,
        ),
        (
            "rand_non_tiled_dimensions",
            "rand_kernel_partial_tile",
            (64, 64, 8),
            [8, 8],
            False,
            None,
        ),
        (
            "rand_mixed_argument_order",
            "rand_kernel_normal_order",
            (16, 32, 8),
            [4, 8, 8],
            False,
            None,
        ),
        (
            "rand_mixed_argument_order",
            "rand_kernel_mixed_order",
            (16, 32, 8),
            [4, 8, 8],
            False,
            None,
        ),
        (
            "rand_rolled_reductions",
            "rand_kernel_with_reduction",
            (64, 128),
            [32],
            False,
            [None],
        ),
        (
            "rand_rolled_reductions",
            "rand_kernel_with_reduction",
            (64, 128),
            [32],
            False,
            [64],
        ),
        (
            "rand_offsets_independence",
            "rand_two_streams_kernel",
            (128,),
            [32],
            False,
            None,
        ),
        ("rand_randint_static_shapes", "rng_kernel_static", (256,), [256], True, None),
        ("rand_randint_specialize", "fn", (128, 1), [128], True, None),
    ),
)
@skipUnlessBackends(["cute"])
def test_actual_failed_random_kernel_codegen(
    method, function, shape, blocks, static, reduction
):
    function = _original_function("test_random.py", "test_hl_" + method, function)
    args = (torch.ones(shape), 1337)
    if method == "rand_randint_specialize":
        args = (args[0], torch.empty(shape), args[1])
    bound = helion.kernel(
        function, backend="cute", static_shapes=static
    )._bind_isolated(args)
    config = helion.Config(
        block_sizes=blocks,
        **({"reduction_loops": reduction} if reduction is not None else {}),
    )
    source = bound.to_code(config)
    assert "_cute_stage_explicit_seed(" in source
    assert "shr.s64 counter, $2, 2" in source
    assert bound.settings.cute_rng_stream == "auto"


@skipUnlessBackends(["cute"])
def test_actual_failed_mixed_rng_keeps_implicit_seed_slots():
    method = "test_explicit_seeded_rng_does_not_shift_implicit_seed_slots"
    sources = []
    for name in (
        "implicit_only_kernel",
        "explicit_then_implicit_kernel",
        "implicit_then_explicit_kernel",
    ):
        function = _original_function("test_rng.py", method, name)
        args = (
            (torch.ones(9, 11),)
            if name == "implicit_only_kernel"
            else (torch.ones(9, 11), 0x12345678)
        )
        bound = helion.kernel(
            function, backend="cute", static_shapes=False
        )._bind_isolated(args)
        sources.append(bound.to_code(helion.Config(block_sizes=[4, 8])))
    seed_assignments = []
    for source in sources:
        tree = ast.parse(source)
        statements = [
            ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_rng_seed_buffer"
                for target in node.targets
            )
        ]
        assert len(statements) == 1
        seed_assignments.append(statements)
    assert seed_assignments[0] == seed_assignments[1] == seed_assignments[2]
    assert "_cute_stage_explicit_seed(" not in sources[0]
    assert all("_cute_stage_explicit_seed(" in source for source in sources[1:])


def _no_tensor_input(device: torch.device, seed: int) -> torch.Tensor:
    out = torch.empty(129, device=device)
    for tile in hl.tile(129):
        out[tile] = hl.rand([tile], seed=seed)
    return out


def _nested_input(inputs: dict[str, list[torch.Tensor]], seed: int) -> torch.Tensor:
    x = inputs["values"][0]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = hl.rand([tile], seed=seed)
    return out


def _host_replay(source, function_name, args):
    tree = ast.parse(source)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    launches, stages = [], []

    def launch(*args, **kwargs):
        launches.append((args, kwargs))

    def stage(seed, like):
        value = runtime_stage_seed(seed, like)
        stages.append((seed, like, value))
        return value

    namespace = {
        "torch": torch,
        "_default_cute_launcher": launch,
        "_cute_stage_explicit_seed": stage,
        **{
            node.name: object()
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
        },
    }
    exec(
        compile(ast.Module(body=[host], type_ignores=[]), "<CPU-host-replay>", "exec"),
        namespace,
    )
    outputs = [namespace[function_name](*args) for _ in range(2)]
    assert len(launches) == len(stages) == 2
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert stages[0][2].data_ptr() != stages[1][2].data_ptr()
    for output, (seed, like, value) in zip(outputs, stages, strict=True):
        assert like is (args[0]["values"][0] if isinstance(args[0], dict) else output)
        assert value.device == output.device
        assert value.shape == (1,) and value.dtype == torch.int64
        assert value.item() == seed == args[-1]


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("seed", (-(2**63), 2**63 - 1))
@skipUnlessBackends(["cute"])
def test_no_input_and_nested_shape_only_input_host_lifetimes(nested, seed):
    function = _nested_input if nested else _no_tensor_input
    args = (
        ({"values": [torch.empty(129)]}, seed)
        if nested
        else (torch.device("cpu"), seed)
    )
    bound = helion.kernel(function, backend="cute", static_shapes=False)._bind_isolated(
        args
    )
    source = bound.to_code(helion.Config(block_sizes=[128]))
    witness = "x" if nested else "out"
    assert f"_cute_stage_explicit_seed(seed, {witness})" in source
    _host_replay(source, function.__name__, args)


@pytest.mark.parametrize(
    "target,device", ((torch.neg, "cpu"), (_tracing_ops._host_tensor, "meta"))
)
def test_fallback_does_not_choose_device_values_or_unrelated_host_locals(
    target, device
):
    graph = torch.fx.Graph()
    node = graph.call_function(target, ("other",))
    node.meta["val"] = torch.empty(1, device=device)
    rng = graph.call_function(torch.neg, (1,))
    state = SimpleNamespace(device_function=SimpleNamespace(arguments=[]), fx_node=rng)
    host = SimpleNamespace(
        literal_expr=lambda value: str(value),
        tensor_to_origin={torch.empty(1): NameOrigin("unrelated_later_allocation")},
    )
    with (
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(device=torch.device("cpu")),
        ),
        patch.object(HostFunction, "current", return_value=host),
        pytest.raises(helion.exc.BackendUnsupported, match="tensor input device"),
    ):
        stage_explicit_seed(state, 123)


def _no_device_argument(seed: int) -> torch.Tensor:
    out = torch.empty(129)
    for tile in hl.tile(129):
        out[tile] = hl.rand([tile], seed=seed)
    return out


def test_missing_device_argument_remains_rejected():
    with pytest.raises(helion.exc.NoTensorArgs):
        helion.kernel(_no_device_argument, backend="cute")._bind_isolated((123,))


@skipUnlessBackends(["cute"])
def test_cpu_codegen_allows_an_already_initialized_process():
    # Model an xdist worker that previously ran CUDA tests; all new device and
    # native calls remain forbidden by the CPU fixtures.
    with patch("torch.cuda.is_initialized", return_value=True):
        test_actual_failed_random_kernel_codegen(
            "rand_randint_1d", "rng_kernel_1d", (1024,), [256], False, None
        )
