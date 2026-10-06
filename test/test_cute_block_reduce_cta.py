from __future__ import annotations

import ast
from contextlib import suppress
import operator
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_reduce(x, operation: hl.constexpr):
    rows, columns = x.shape
    br = hl.register_block_size(rows)
    bc = hl.register_block_size(columns)
    out = torch.empty((rows, (columns + bc - 1) // bc), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([rows, columns], block_size=[br, bc]):
        values = x[row, col]
        if operation == "max":
            reduced = values.amax(-1)
        elif operation == "min":
            reduced = values.amin(-1)
        elif operation == "sum":
            reduced = values.sum(-1)
        else:
            reduced = values.prod(-1)
        out[row, col.id] = reduced
    return out


def _input(rows, columns, operation):
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    values = (values % 29) - 14
    if operation == "prod":
        values = torch.where(values < 0, -1.0, 1.0)
    # Different warp winners and different rows expose partial reductions.
    values[:, 0] = -1 if operation == "prod" else 1000 + torch.arange(rows)
    return values


def _code(rows, columns, threads, row_threads, operation):
    x = _input(rows, columns, operation)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_tile_reduce, (x, operation))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[row_threads, 2 * threads],
            num_threads=[row_threads, threads],
            cute_vector_widths=[1, 1],
            cute_lane_layouts=["blocked", "blocked"],
            loop_orders=[[1, 0]],
        )
        code = bound.to_code(config)
    return code, x, config


def _execute(code, x, operation):
    """Execute every generated scalar instruction; rendezvous exact collectives."""
    tree = ast.parse(code)
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    }
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
    )
    launch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_launcher"
    )
    block = ast.literal_eval(next(k.value for k in launch.keywords if k.arg == "block"))
    assert block[2] == 1

    class Collective(ast.NodeTransformer):
        def visit_Call(self, n):
            name = ast.unparse(n.func)
            if name == "_cute_grouped_reduce_shared_two_stage":
                kwargs = {k.arg: ast.literal_eval(k.value) for k in n.keywords}
                assert kwargs["pre"] == 1
                assert (
                    kwargs["group_count"] * kwargs["group_span"] == block[0] * block[1]
                )
                return ast.Yield(
                    ast.Tuple(
                        [
                            n.args[0],
                            ast.Constant(kwargs["group_span"]),
                            ast.Constant(False),
                        ],
                        ast.Load(),
                    )
                )
            if name.startswith("cute.arch.warp_reduction"):
                threads = next(
                    ast.literal_eval(k.value)
                    for k in n.keywords
                    if k.arg == "threads_in_group"
                )
                return ast.Yield(
                    ast.Tuple(
                        [n.args[0], ast.Constant(threads), ast.Constant(True)],
                        ast.Load(),
                    )
                )
            return self.generic_visit(n)

    fn.decorator_list = []
    fn = Collective().visit(fn)
    rows, columns = x.shape
    bc = constants["_BLOCK_SIZE_1"]
    if "_BLOCK_SIZE_0" in constants:
        br = constants["_BLOCK_SIZE_0"]
    else:
        assert rows == 1
        br = 1
    parts = (columns + bc - 1) // bc
    output = np.full((rows, parts), np.nan, dtype=np.float32)
    writes = []
    tid, bid = [0], [0]

    class Pointer:
        def __init__(self, values, offset=0):
            self.values, self.offset = values.reshape(-1), int(offset)

        def __add__(self, value):
            return Pointer(self.values, self.offset + int(value))

        def load(self):
            assert 0 <= self.offset < self.values.size
            return self.values[self.offset]

        def store(self, value):
            assert 0 <= self.offset < self.values.size
            writes.append((bid[0], tid[0], self.offset))
            self.values[self.offset] = value

    scope = dict(
        constants,
        operator=operator,
        cutlass=SimpleNamespace(
            Int32=int, Int64=int, Float32=np.float32, range_constexpr=range
        ),
        cute=SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (bid[0], 0, 0),
                thread_idx=lambda: (tid[0] % block[0], tid[0] // block[0], 0),
                lane_idx=lambda: tid[0] % 32,
            )
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([fn], [])),
            "<block-reduce-model>",
            "exec",
        ),
        scope,
    )
    known = {
        "x": SimpleNamespace(
            iterator=Pointer(x.numpy()), layout=SimpleNamespace(stride=x.stride())
        ),
        "out": SimpleNamespace(
            iterator=Pointer(output), layout=SimpleNamespace(stride=(parts, 1))
        ),
        "out_size_1": parts,
    }
    args = {arg.arg: known[arg.arg] for arg in fn.args.args}
    count = block[0] * block[1]
    for cta in range(parts * ((rows + br - 1) // br)):
        bid[0] = cta
        workers = [scope[fn.name](**args) for _ in range(count)]
        sent = [None] * count
        while True:
            values = [None] * count
            for t in range(count):
                tid[0] = t
                with suppress(StopIteration):
                    values[t] = workers[t].send(sent[t])
            if all(value is None for value in values):
                break
            assert all(value is not None for value in values), "nonuniform collective"
            sent = []
            for t, (_, span, warp) in enumerate(values):
                size = min(span, 32) if warp else span
                start = t // size * size
                group = np.array([values[u][0] for u in range(start, start + size)])
                reduced = {
                    "sum": np.sum,
                    "max": np.max,
                    "min": np.min,
                    "prod": np.prod,
                }[operation](group)
                sent.append(np.float32(reduced))
    return torch.from_numpy(output), writes


@pytest.mark.parametrize(
    "threads,row_threads,rows",
    [(16, 2, 3), (32, 2, 3), (64, 1, 1), (64, 2, 3), (128, 2, 3), (512, 2, 3)],
)
@pytest.mark.parametrize("operation", ["sum", "max", "min", "prod"])
def test_complete_tile_reduction_across_physical_warps(
    threads, row_threads, rows, operation
):
    columns = 2 * threads + 5
    code, x, _ = _code(rows, columns, threads, row_threads, operation)
    calls = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call)
        and ast.unparse(n.func).startswith("cute.arch.warp_reduction")
    ]
    for call in calls:
        assert (
            next(
                ast.literal_eval(k.value)
                for k in call.keywords
                if k.arg == "threads_in_group"
            )
            <= 32
        )
    assert ("_cute_grouped_reduce_shared_two_stage(" in code) == (threads > 32)
    observed, writes = _execute(code, x, operation)
    expected = torch.stack(
        [
            getattr(
                x[:, start : start + 2 * threads],
                {"max": "amax", "min": "amin"}.get(operation, operation),
            )(-1)
            for start in range(0, columns, 2 * threads)
        ],
        dim=1,
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert sorted(offset for _, _, offset in writes) == list(range(expected.numel()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("threads", [32, 64, 512])
def test_complete_tile_reduction_native(threads):
    for operation in ("max", "sum"):
        rows, columns = 3, 2 * threads + 5
        _, x, config = _code(rows, columns, threads, 2, operation)
        x = x.cuda()
        actual = _tile_reduce.bind((x, operation)).compile_config(config)(x, operation)
        expected = torch.stack(
            [
                getattr(
                    x[:, start : start + 2 * threads],
                    "amax" if operation == "max" else "sum",
                )(-1)
                for start in range(0, columns, 2 * threads)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("threads", [64, 128, 512])
def test_multiwarp_finalize_without_group_proof_rejects(threads):
    from helion._compiler import tile_strategy

    marker = tile_strategy._LaneReduceMarker(
        input_name="value",
        wrap_template="__HELION_FINALIZED__",
        reduction_type="max",
        identity_expr="cutlass.Float32(float('-inf'))",
        threads_in_group=threads,
        result_var="result",
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="proven CTA reduction group"
    ):
        tile_strategy._finalize_lane_reduce_marker(marker, "acc")
