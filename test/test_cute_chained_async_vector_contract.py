from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_vector_leaf import _plan
import helion
from helion._compiler.cute import chained_prepared_values as prepared
from helion._compiler.cute.chained_vector_leaf import emit_vector_leaf
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _identity_frontier_sequence(a, b, rhs, weights, initial, valid: int):
    valid = hl.specialize(valid)
    steps, rows, reduction = a.shape
    columns = rhs.shape[-1]
    history = torch.empty((steps, rows, columns), device=a.device)
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([rows, columns], block_size=[32, 128]):
        state = initial[rr, cc]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(32)
            image = hl.load(
                weights,
                [step.id, rr, cc],
                extra_mask=(rr.index < valid)[:, None],
            )
            mid = hl.dot(a[step.id, rr, kk] * 0.5, b[step.id, kk, jj]).to(a.dtype)
            state = hl.dot(mid, rhs[step.id, jj, cc], acc=state) + image
            history[step.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _capture(
    dtype,
    *,
    warps=16,
    consumer_warps=4,
    columns=128,
    steps=3,
    valid=13,
    layout="row_major",
    mutation=None,
):
    args = (
        torch.zeros((steps, 19, 16), dtype=torch.bfloat16),
        torch.zeros((steps, 16, 32), dtype=torch.bfloat16),
        torch.zeros((steps, 32, columns), dtype=torch.bfloat16),
        torch.zeros((steps, 19, columns), dtype=dtype),
        torch.zeros((19, columns)),
        valid,
    )
    config = _config(warps, pipeline=True, consumer_warps=consumer_warps)
    config.config.update(
        cute_chained_pointwise_vectorize=True,
        cute_chained_scratch_layout=layout,
        cute_chained_async_vector_store=True,
    )
    bind, emit = prepared.bind_frame_buffers, prepared.emit_prepared_value
    observations, mutations = [], []

    def view(frame, pointer, scratch, **kwargs):
        lines = bind(frame, pointer, scratch, **kwargs)
        for name, sink in tuple(scratch.vector_sinks.items()):
            if mutation == "shape":
                scratch.vector_sinks[name] = replace(
                    sink, shape=(sink.shape[0] + 1, sink.shape[1])
                )
            elif mutation == "dtype":
                scratch.vector_sinks[name] = replace(sink, dtype="cutlass.Float64")
            elif mutation == "drop":
                scratch.vector_sinks.pop(name)
            if mutation is not None:
                mutations.append(name)
        return lines

    def observe(cg, plan, buffer, *args, **kwargs):
        lines = emit(cg, plan, buffer, *args, **kwargs)
        observations.append((buffer.dtype, "\n".join(lines), kwargs.get("shared_sink")))
        return lines

    with (
        patch.object(prepared, "bind_frame_buffers", view),
        patch.object(prepared, "emit_prepared_value", observe),
    ):
        if mutation is None:
            source = _source(_identity_frontier_sequence, args, config)
        else:
            with pytest.raises(
                helion.exc.BackendUnsupported, match="identity shared sink"
            ):
                _source(_identity_frontier_sequence, args, config)
            assert mutations
            source = ""
    return source, observations


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "warps,consumer_warps,columns,steps,valid",
    [(8, 4, 128, 1, 19), (16, 8, 127, 3, 13), (16, 4, 121, 3, 0)],
)
def test_actual_public_identity_frontier_copy_prefix_and_tails(
    dtype, warps, consumer_warps, columns, steps, valid
):
    source, observations = _capture(
        dtype,
        warps=warps,
        consumer_warps=consumer_warps,
        columns=columns,
        steps=steps,
        valid=valid,
    )
    assert "CopyG2SOp" in source
    selected = [
        text
        for actual, text, sink in observations
        if actual == dtype and "CopyG2SOp" in text
    ]
    assert selected
    role_threads = 32 * (warps - consumer_warps)
    copy_threads = 1 << (role_threads.bit_length() - 1)
    for text in selected:
        ast.parse(text)
        assert text.endswith(
            ("    " if copy_threads != role_threads else "")
            + "cute.arch.cp_async_wait_group(0)\nchain_prep_barrier.arrive_and_wait()"
        )
        if copy_threads != role_threads:
            assert text.startswith(f"if chain_prep_thread < {copy_threads}:\n")
        assert "cp_async_commit_group()" in text
        assert f"< {columns}" in text


@pytest.mark.parametrize("mutation", ["shape", "dtype", "drop"])
def test_actual_view_drift_rejects_requested_but_unused_activation(mutation):
    _capture(torch.float32, mutation=mutation)


def test_unread_fp32_xor_view_does_not_inherit_dense_admission():
    _, rows = _capture(torch.float32, layout="xor")
    fp32 = [(text, sink) for dtype, text, sink in rows if dtype == torch.float32]
    assert fp32 and all(sink is None and "CopyG2SOp" not in text for text, sink in fp32)
    assert any(
        "CopyG2SOp" in text for dtype, text, sink in rows if dtype == torch.bfloat16
    )


class _Pointer:
    def __init__(self, address, itemsize):
        self.address, self.itemsize = address, itemsize

    def __add__(self, count):
        return _Pointer(self.address + count * self.itemsize, self.itemsize)

    def toint(self):
        return self.address


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_actual_emitted_guards_all_tails_alignment_and_outside_rows(dtype):
    plan = _plan(dtype=dtype)
    assert plan is not None
    emitted = emit_vector_leaf(
        plan,
        tensor="tensor",
        prefix="leaf",
        pointer_for_indices=lambda indices: (
            f"tensor.iterator + {indices[0]} * 24 + {indices[1]}"
        ),
        scalar_for_element=lambda element: ([], f"original_scalar({element})"),
        shared_pointer="shared.iterator + row * 24 + base",
    )
    tree = ast.parse("\n".join(emitted.lines))
    outer = next(node for node in tree.body if isinstance(node, ast.If))
    inner = next(node for node in outer.body if isinstance(node, ast.If))
    cutlass = SimpleNamespace(Int64=int)
    accepted = rejected = 0
    # These are small exact integers: fixed-width source expressions are
    # separately preserved/tested below and in the original leaf tests.
    for row in range(-1, 5):
        for base in range(-8, 33):
            for limit in (0, 3, 4):
                for source_shift, sink_shift in ((0, 0), (1, 0), (0, 1)):
                    env = {
                        "cutlass": cutlass,
                        "row": row,
                        "base": base,
                        "limit": limit,
                        "tensor": SimpleNamespace(
                            shape=(4, 24),
                            layout=SimpleNamespace(stride=(24, 1)),
                            iterator=_Pointer(
                                0x100000 + source_shift, plan.element_bytes
                            ),
                        ),
                        "shared": SimpleNamespace(
                            iterator=_Pointer(16384 + sink_shift, plan.element_bytes)
                        ),
                    }
                    exec(
                        compile(
                            ast.Module(tree.body[2:5], []), "original_indices", "exec"
                        ),
                        env,
                    )
                    fast = bool(
                        eval(
                            compile(
                                ast.Expression(outer.test), "original_bounds", "eval"
                            ),
                            env,
                        )
                    )
                    if fast:
                        exec(
                            compile(
                                ast.Module(outer.body[:-1], []),
                                "original_addresses",
                                "exec",
                            ),
                            env,
                        )
                        fast = bool(
                            eval(
                                compile(
                                    ast.Expression(inner.test),
                                    "original_addresses",
                                    "eval",
                                ),
                                env,
                            )
                        )
                    first = (
                        0x100000 + source_shift + (row * 24 + base) * plan.element_bytes
                    )
                    destination = (
                        16384 + sink_shift + (row * 24 + base) * plan.element_bytes
                    )
                    expected = (
                        0 <= row < min(4, limit)
                        and 0 <= base <= 16
                        and first % 16 == destination % 16 == 0
                    )
                    assert fast == expected
                    accepted += fast
                    rejected += not fast
    assert accepted and rejected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fixed_width_source_and_endpoint_guards_are_not_reassociated(dtype):
    plan = _plan(
        "cutlass.Int32(base) + element + cutlass.Int64(shift)",
        row="cutlass.Int32(row) * 1073741824 // 1073741824",
        dtype=dtype,
    )
    assert plan is not None
    arguments = {
        "tensor": "tensor",
        "prefix": "leaf",
        "pointer_for_indices": lambda index: (
            f"tensor.iterator + cutlass.Int32({index[0]}) * cutlass.Int32(24) + cutlass.Int32({index[1]})"
        ),
        "scalar_for_element": lambda element: ([], f"original_scalar({element})"),
    }
    old = ast.parse("\n".join(emit_vector_leaf(plan, **arguments).lines))
    new = ast.parse(
        "\n".join(
            emit_vector_leaf(plan, shared_pointer="shared.iterator", **arguments).lines
        )
    )
    assert [ast.dump(node) for node in old.body[:5]] == [
        ast.dump(node) for node in new.body[:5]
    ]
    old_guard = next(node for node in old.body if isinstance(node, ast.If))
    new_guard = next(node for node in new.body if isinstance(node, ast.If))
    assert ast.dump(old_guard.test) == ast.dump(new_guard.test)
    assert ast.dump(old.body[-1]) == ast.dump(new.body[-1])
