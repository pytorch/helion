"""Review O1: the initial M-edge family never emits output-TMA stores."""

from __future__ import annotations

from dataclasses import replace

import pytest

from test.test_cute_shared_a_tma_m_edge import INDUCTOR_NATIVE
from test.test_cute_shared_a_tma_m_edge import args_for
from test.test_cute_shared_a_tma_m_edge import bind
from test.test_cute_shared_a_tma_m_edge import capture_code
from test.test_cute_shared_a_tma_m_edge import cpu_only as cpu_only
from test.test_cute_shared_a_tma_m_edge import load_kernel
from test.test_cute_shared_a_tma_m_edge import native_calls
from test.test_cute_shared_a_tma_m_edge import pairs

from helion import exc
from helion._compiler.cute.cute_mma_dag_tcgen05 import _Tcgen05Emitter


def test_all_actual_canonical_Medge_plans_have_only_predicated_packet_outputs():
    bound = bind(load_kernel(INDUCTOR_NATIVE), args_for())
    configs = pairs(bound)
    assert len(configs) == 4
    for _, config in configs:
        source, plan = capture_code(bound, config)
        assert plan.shared_a_m_edge_axis is not None
        assert plan.store_tma == {}
        assert [store.access.name for store in plan.staged_stores] == ["out0", "out1"]
        assert native_calls(source, "stg_v4_b32")
        assert not native_calls(source, "cp_async_bulk_tensor_global_shared_cta")


def test_injected_output_TMA_path_on_real_proved_Medge_plan_fails_loudly():
    bound = bind(load_kernel(INDUCTOR_NATIVE), args_for())
    _, plan = capture_code(bound, pairs(bound)[0][1])
    assert plan.shared_a_m_edge_axis is not None and plan.store_tma == {}
    store = plan.staged_stores[0]
    # A deliberately invalid output-TMA map is injected into an otherwise
    # genuinely proved plan. This is negative fault injection, not a source
    # of positive proof/metadata or an actual emitted output descriptor.
    invalid = replace(plan, store_tma={store.access.node: plan.tma[0]})
    emitter = _Tcgen05Emitter.__new__(_Tcgen05Emitter)
    emitter.tc = invalid
    with pytest.raises(
        exc.BackendUnsupported, match="predicated non-TMA output stores"
    ):
        emitter.emit_tma_stores([store], "0", "0")
