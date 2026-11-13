from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def float_branch(x, runtime: float):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if x[tile].sum() > 0:
            value = runtime + 1.0
        else:
            value = runtime + 2.0
        out[tile] = x[tile] * value
    return out


@onlyBackends(["triton", "cute"])
def test_device_branch_can_merge_host_float_expressions():
    kernel = helion.kernel(float_branch.fn, autotune_effort="none")
    with cpu_render_target():
        x = torch.empty(17)
        bound = kernel.bind((x, 1.25))
        config = bound.config_spec.default_config()
        tree = ast.parse(bound.to_code(config))
        device = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_helion_float_branch"
        )
        branch = next(
            node
            for node in ast.walk(device)
            if isinstance(node, ast.If)
            and len(node.body) == len(node.orelse) == 1
            and isinstance(node.body[0], ast.Assign)
            and isinstance(node.orelse[0], ast.Assign)
        )
        left, right = branch.body[0], branch.orelse[0]
        assert isinstance(left, ast.Assign) and isinstance(right, ast.Assign)
        assert ast.dump(left.targets[0]) == ast.dump(right.targets[0])
        assert isinstance(left.targets[0], ast.Name)
        joined = left.targets[0].id
        assert any(
            isinstance(node, ast.BinOp)
            and any(
                isinstance(child, ast.Name) and child.id == joined
                for child in ast.walk(node)
            )
            for node in ast.walk(device)
        )
        branch_code = compile(
            ast.fix_missing_locations(ast.Module([branch], [])), "<device_join>", "exec"
        )
        assert isinstance(branch.test, ast.Name)
        for runtime in (1.25, -3.5):
            launches = []

            def observe(_kernel, _grid, *args, records=launches, **_kwargs):
                records.append(args)

            assert kernel.bind((x, runtime)) is bound
            load_host_function(bound, config, observe)(x, runtime)
            assert len(launches) == 1
            arguments = dict(
                zip((arg.arg for arg in device.args.args), launches[0], strict=True)
            )
            for predicate, expected in ((True, runtime + 1.0), (False, runtime + 2.0)):
                values = {**arguments, branch.test.id: predicate}
                exec(branch_code, values)
                assert type(values[joined]) is float and values[joined] == expected


@helion.kernel(autotune_effort="none")
def float_live_in(x, runtime: float):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = runtime
        if x[tile].sum() > 0:
            value = runtime + 1.0
        out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def float_loop_carry(x, runtime: float):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = runtime
        for _inner in hl.tile(x.numel(), block_size=8):
            value = value + 1.0
        out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize("function", [float_live_in, float_loop_carry])
def test_host_float_mutation_remains_rejected(function):
    kernel = helion.kernel(function.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.CannotModifyHostVariableOnDevice, match="value"),
    ):
        kernel.bind((torch.empty(17), 1.25))


def test_pallas_buffer_slots_exclude_host_float_arguments():
    kernel = helion.kernel(float_branch.fn, autotune_effort="none", backend="pallas")
    with cpu_render_target():
        bound = kernel.bind((torch.empty(17), 1.25))
        assert bound.config_spec.pallas_load_buffer_count.length == 1


@helion.kernel(backend="pallas", autotune_effort="none", static_shapes=True)
def interleaved_float_loads(scale: float, x, offset: float, y):
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        for columns in hl.tile(x.size(1)):
            out[rows, columns] = x[rows, columns] * scale + y[rows, columns] + offset
    return out


@pytest.mark.parametrize("buffer_counts", [[2, 1], [1, 2]])
@onlyBackends(["pallas"])
def test_pallas_dma_slots_exclude_interleaved_host_floats(buffer_counts):
    kernel = helion.kernel(
        interleaved_float_loads.fn,
        backend="pallas",
        autotune_effort="none",
        static_shapes=True,
    )
    with cpu_render_target():
        x, y = torch.empty(8, 256), torch.empty(8, 256)
        bound = kernel.bind((1.25, x, -2.5, y))
        assert bound.config_spec.pallas_load_buffer_count.length == 2
        config = helion.Config(
            block_sizes=[8, 128],
            pallas_loop_type="fori_loop",
            pallas_load_buffer_count=buffer_counts,
        )
        code = bound.to_code(config)
        assert "jax.lax.fori_loop" in code
        prime = next(
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.FunctionDef) and node.name == "_prime_fori_loads"
        )
        copies = [
            node
            for node in ast.walk(prime)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "pltpu.make_async_copy"
        ]
        assert len(copies) == 1
        selected = "x" if buffer_counts[0] == 2 else "y"
        assert ast.unparse(copies[0].args[0]).startswith(f"{selected}.at[")

        # Execute the emitted host handoff only. Both floats must stay ordinary
        # runtime inputs, independently of the tensor-only DMA slot numbering.
        namespace = {}
        exec(compile(code, "<pallas-host>", "exec"), namespace)
        launches = []

        def observe(_kernel, _grid, *args, **_kwargs):
            launches.append(args)
            return torch.empty_like(x)

        for scale, offset in ((1.25, -2.5), (-3.5, 4.25)):
            assert kernel.bind((scale, x, offset, y)) is bound
            namespace["interleaved_float_loads"](scale, x, offset, y, _launcher=observe)
            args = launches[-1]
            assert args[0] is x and args[1] is y
            assert args[3].item() == scale and args[4].item() == offset
        assert len(launches) == 2
