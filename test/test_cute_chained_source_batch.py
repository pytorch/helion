from __future__ import annotations

import ast
from functools import partial
import hashlib
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import NoReturn

import pytest

from test.test_cute_source_batch import _bound

from helion._compiler.device_ir import KernelPhase
from helion.runtime.config import Config
from helion.runtime.cute import source_batch as sb

if TYPE_CHECKING:
    import torch

    from helion.runtime.kernel import BoundKernel


def never_called(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("metadata freezing must not call a classifier")


def chained() -> tuple[BoundKernel, torch.Tensor, Config]:
    bound, tensor = _bound()
    bound.config_spec.cute_chained_matmul_search_enabled = True
    bound.host_function.device_ir.root_ids = [0]
    bound.host_function.device_ir.phases = [
        KernelPhase(
            roots=[0], root_nodes=[ast.parse("for i in range(1):\n    pass\n").body[0]]
        )
    ]
    default = Config(
        block_sizes=[16], num_warps=4, cute_chained_mma_schedule="coalesced"
    )
    bound.config_spec.default_config = lambda: default
    bound.env.runtime_input_specializations["loop"] = SimpleNamespace(
        sources=(),
        classifier_identity=("loop",),
        classifier=partial(never_called, writes=(1,)),
        reusable_tensor_properties=frozenset(("data_ptr", "storage_span")),
    )
    bound.env.bound_runtime_input_specialization_results["loop"] = True
    return bound, tensor, default


def test_partial_snapshot_is_structural_and_does_not_call() -> None:
    bound, tensor, _ = chained()
    before = sb._snapshot(bound, (tensor,))
    fn = bound.env.runtime_input_specializations["loop"].classifier
    fn.keywords["writes"] = (2,)
    assert sb._snapshot(bound, (tensor,)) != before
    fn.keywords["writes"] = (1,)
    assert sb._snapshot(bound, (tensor,)) == before
    bound.env.runtime_input_specializations["loop"].classifier = partial(
        len, writes=(1,)
    )
    with pytest.raises(TypeError):
        sb._snapshot(bound, (tensor,))


def test_partial_nested_cycle_and_subclass() -> None:
    bound, _, _ = chained()
    value = partial(partial(never_called, {"nested": [1, 2]}), 3)
    assert sb._freeze_metadata(value, bound)[0] == "partial"
    value.keywords["cycle"] = value
    with pytest.raises(TypeError, match="Cyclic"):
        sb._freeze_metadata(value, bound)

    class Foreign(partial):
        pass

    with pytest.raises(TypeError, match="Unsupported"):
        sb._freeze_metadata(Foreign(never_called), bound)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork admission")
@pytest.mark.parametrize("field", ["roots", "phases", "wrappers", "rng", "fission"])
def test_only_one_plain_region(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    bound, tensor, _ = chained()
    monkeypatch.setattr(sb, "_guard_targets", list)
    assert sb.SourceBatch.create(bound, [tensor]) is not None
    if field == "roots":
        bound.host_function.device_ir.root_ids.append(1)
    elif field == "phases":
        bound.host_function.device_ir.phases.append(
            KernelPhase(roots=[1], root_nodes=[])
        )
    elif field == "wrappers":
        bound.env.cute_resolved_wrapper_plans.append({"kind": "wrapped"})
    elif field == "rng":
        bound.host_function.rng_seed_slot_count = 1
    else:
        bound.env.cute_fission_plan = object()
    assert sb.SourceBatch.create(bound, [tensor]) is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fork supervisor")
@pytest.mark.parametrize(
    "schedule", ["cp_async", "coalesced_unrolled", "k_major", "k_major_padded"]
)
def test_advanced_batch_preserves_original_indices(
    schedule: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, tensor, default = chained()
    monkeypatch.setattr(sb, "_guard_targets", list)
    batch = sb.SourceBatch.create(bound, [tensor])
    assert batch is not None
    parallel = Config(
        **{
            **default.config,
            "block_sizes": [32],
            "cute_chained_mma_schedule": schedule,
        }
    )
    advanced = Config(
        **{
            **default.config,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_pointwise_vectorize": True,
        }
    )
    prepared = batch.prepare([advanced, parallel], lambda: False)
    assert set(prepared) == {0, 1}
    assert prepared[0].consume(bound, advanced) == bound.to_triton_code(advanced)
    assert prepared[1].consume(bound, parallel) == bound.to_triton_code(parallel)
    assert prepared[1].consume(bound, advanced) is None
    assert not bound._compile_cache and not bound._cache_path_map
    bound.host_function.device_ir.root_ids.append(2)
    assert prepared[1].consume(bound, parallel) is None


def test_real_kernel_phase_mutation_invalidates_prepared_source() -> None:
    bound, tensor, config = chained()
    ir = bound.host_function.device_ir
    phase = ir.phases[0]
    assert type(phase) is KernelPhase
    before = sb._snapshot(bound, (tensor,))
    source = "pass\n"
    prepared = sb.PreparedSource(
        bound=bound,
        config_key=sb._config_key(config),
        source=source,
        source_hash=hashlib.sha256(source.encode()).hexdigest(),
        authority=before,
        tensors=(tensor,),
        storages=(tensor.untyped_storage(),),
        output="",
    )
    assert prepared.consume(bound, config) == source
    # Same phase count, device roots, host body and graphs: only phase mapping changes.
    phase.roots[0] = 7
    assert len(ir.phases) == 1 and ir.root_ids == [0]
    assert sb._snapshot(bound, (tensor,)) != before
    assert prepared.consume(bound, config) is None
    phase.roots[0] = 0
    assert sb._snapshot(bound, (tensor,)) == before
    assert prepared.consume(bound, config) == source
    original_iter = phase.root_nodes[0].iter
    phase.root_nodes[0].iter = ast.parse("range(2)", mode="eval").body
    assert len(ir.phases) == 1 and phase.roots == [0] and ir.root_ids == [0]
    assert sb._snapshot(bound, (tensor,)) != before
    assert prepared.consume(bound, config) is None
    phase.root_nodes[0].iter = original_iter
    assert sb._snapshot(bound, (tensor,)) == before
    assert prepared.consume(bound, config) == source
