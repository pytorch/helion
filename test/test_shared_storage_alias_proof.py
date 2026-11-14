from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.merge_reduction_dims import _Merger
from helion._compiler.roll_reduction import ReductionRoller
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl
from helion.language._tracing_ops import _host_tensor
from helion.language.memory_ops import load
from helion.language.memory_ops import store

pytestmark = pytest.mark.cpu_only


def _storage_values(x: torch.Tensor, y: torch.Tensor, choose: int):
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    if choose > 0:
        selected = first
    else:
        selected = second
    for i, j in hl.tile(x.size()):
        first[i, j] = x[i, j] + y[i, j]
    return first, selected


@pytest.mark.parametrize("consumer", ["roll", "merge"])
@pytest.mark.parametrize(
    "relationship", ["fresh", "join", "post_effect_fresh", "invalid_old", "external"]
)
def test_shared_reduction_proofs_use_all_storage_origins(consumer, relationship):
    # Both transforms need fresh/external and post-effect-allocation positives,
    # plus synthetic-join and old-output negatives. All five relationships must
    # be checked for each consumer because their alias policies differ.
    kernel = helion.kernel(_storage_values, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        x, y = torch.empty(4, 4), torch.empty(4, 4)
        bound = cpu_bind(kernel, (x, y, 1))
        env, host = bound.env, bound.host_function
        with env, host, env.use_runtime_arg_values({"x": x, "y": y, "choose": 1}):
            read_tensor = host.local_types["x"].fake_value
            write_tensor = host.local_types["first"].fake_value
            if relationship == "join":
                read_tensor = host.local_types["selected"].fake_value
                assert (
                    read_tensor.untyped_storage() is not write_tensor.untyped_storage()
                )
            elif relationship in ("post_effect_fresh", "invalid_old"):
                with env.track_runtime_layout_mutations((), opaque=True):
                    pass
                assert env.tensor_storage_input_source(read_tensor) is not None
                assert env.tensor_storage_provenance(read_tensor) is None
                if relationship == "post_effect_fresh":
                    with env.fake_mode:
                        write_tensor = torch.empty((4, 4))
                    env.register_tensor_factory_layout(
                        torch.empty, ((4, 4),), {}, write_tensor
                    )
                else:
                    assert env.tensor_storage_provenance(write_tensor) is None
            elif relationship == "external":
                write_tensor = host.local_types["y"].fake_value

            graph = torch.fx.Graph()
            source = graph.call_function(_host_tensor, ("read",))
            source.meta["val"] = read_tensor
            destination = graph.call_function(_host_tensor, ("write",))
            destination.meta["val"] = write_tensor
            read = graph.call_function(load, (source, [slice(None), slice(None)]))
            read.meta["val"] = read_tensor
            write = graph.call_function(store, (destination, [], 0.0))
            disjoint = relationship in ("fresh", "post_effect_fresh", "external")
            if consumer == "roll":
                roller = ReductionRoller(
                    host.device_ir,
                    env.block_sizes[0],
                    {},
                    allow_unbound_alias_facts=True,
                )
                assert roller._effect_may_change_read(write, read) is not disjoint
            else:
                before = dict(env.runtime_input_specializations)
                merger = _Merger(
                    read,
                    [0, 1],
                    {},
                    {},
                    {id(write_tensor.untyped_storage()): write_tensor},
                )
                assert merger._flat_load(read) == ((0, 0) if disjoint else None)
                assert merger.needs_alias_guard is (relationship == "external")
                # Proving a candidate cannot publish a guard before the
                # surrounding transform commits its successful rewrite.
                assert env.runtime_input_specializations == before
