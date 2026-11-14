"""Allocations made after opaque effects remain separate from input storage."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.runtime_tensor_guards import tensors_have_proven_disjoint_storage
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _identity(n):
    return n


def _capture_mutator(left, right):
    def mutate():
        original = left.data
        left.data = right
        left.data = original

    return mutate


def _allocate_around_call(x, y, select: int, kind: hl.constexpr):
    if kind == "after":
        n = _identity(x.size(0))
        fresh = torch.empty((n,), dtype=x.dtype, device=x.device)
    elif kind == "captured":
        mutate = _capture_mutator(x, y)
        fresh = torch.empty_like(x)
        mutate()
    else:
        fresh = torch.empty_like(x)
        n = _identity(x.size(0))
    if select > 0:
        selected = fresh
    else:
        selected = x
    for tile in hl.tile(x.numel()):
        fresh[tile] = x[tile]
    return fresh, selected


@pytest.mark.parametrize("kind", ["after", "before", "captured"])
@forbid_cuda_work()
def test_only_post_call_allocations_have_independent_proof(kind):
    args = (torch.empty(8), torch.empty(8), 1, kind)
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(_allocate_around_call), args)
        host = bound.host_function
        values = {
            name: host.local_types[name].fake_value
            for name in ("x", "y", "fresh", "selected")
        }
        with bound.env, host:
            assert tensors_have_proven_disjoint_storage(
                bound.env, values["fresh"], values["x"]
            ) is (kind == "after")
            assert not tensors_have_proven_disjoint_storage(
                bound.env, values["x"], values["y"]
            )
            assert not tensors_have_proven_disjoint_storage(
                bound.env, values["fresh"], values["selected"]
            )
            assert not tensors_have_proven_disjoint_storage(
                bound.env, values["y"], values["selected"]
            )
            assert bound.env.tensor_storage_provenance(values["x"]) is None
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        for select in (1, 0):
            fresh, selected = entry(args[0], args[1], select, kind)
            assert fresh.untyped_storage() != args[0].untyped_storage()
            assert selected is (fresh if select else args[0])


def _opaque_size_matmul(a, b):
    m, k = a.shape
    n = _identity(b.size(1))
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    alias = out.view(m, n)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[row, reduction], b[reduction, column])
        previous = alias[row, column]
        hl.atomic_add(out, [row, column], acc + previous, sem="relaxed")
    return out


@onlyBackends(["cute"])
@forbid_cuda_work()
def test_opaque_size_before_output_preserves_collective_matmul():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            helion.kernel(
                _opaque_size_matmul,
                static_shapes=True,
                cute_promote_half_atomic_outputs=True,
            ),
            (
                torch.empty((64, 512), dtype=torch.float16),
                torch.empty((512, 128), dtype=torch.float16),
            ),
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[32, 64, 32],
                num_threads=[2, 64, 1],
                cute_collective_mma=True,
                cute_collective_compute="warp",
                cute_collective_copy="async_cached",
                cute_collective_stages=4,
            )
        )
    assert "val=cutlass.Float16(" in source
    assert "return out.to(torch.float16)" not in source
