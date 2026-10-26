"""Code shape of the ``cute_early_release`` knob (release the programmatic
dependents at entry).

Knob on: a plain SIMT kernel on sm_90+ executes
``cute.arch.griddepcontrol_launch_dependents()`` as the first statement of
its body, or directly after ``cute_pdl``'s ``griddepcontrol_wait()`` when
that knob applies too (a release ahead of the wait would let the dependents
start while this kernel's predecessors still write); nothing else changes.
Knob off (the default): the line is absent.  Kernels with a dependent-launch
protocol of their own (the tcgen05 pipelines, the materialized-fission
producer that releases at entry already, the consumer launched with
``use_pdl``) and pre-sm_90 targets generate exactly the knob-off code.  GPU
semantics live in ``test_cute_pdl_cuda.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.se_block import se_block_bwd_grad_to_w
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test.test_cute_materialize_operand import _args
from test.test_cute_materialize_operand import _bind
from test.test_cute_materialize_operand import _computed_rhs
from test.test_cute_materialize_operand import _stage_sources
from test.test_cute_materialized_pdl import KEY as MATERIALIZED_PDL_KEY
from test.test_cute_materialized_pdl import _device
from test.test_cute_materialized_pdl import _serial
from test.test_cute_pdl import _COLUMN_SUM
from test.test_cute_pdl import _kernel_def
from test.test_cute_pdl import cpu_only as cpu_only
from test.test_cute_pdl import scale_rows
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only as _cpu_only
from test.test_cute_tcgen05_fixed_chain import _source
from test.test_cute_tcgen05_pdl import _persistent_config

import helion
from helion import exc
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel

cutlass = pytest.importorskip("cutlass")

WAIT = "cute.arch.griddepcontrol_wait()"
RELEASE = "cute.arch.griddepcontrol_launch_dependents()"
ATTRIBUTE = "._helion_cute_use_pdl = True"


def _render(bound: BoundKernel[Any], config: dict[str, object]) -> str:
    spec = bound.config_spec
    return bound.to_code(spec.normalized_config(helion.Config.from_dict(config)))


def _body(code: str) -> list[str]:
    return [ast.unparse(statement) for statement in _kernel_def(code).body]


def _without(code: str, *tokens: str) -> list[str]:
    return [line for line in code.splitlines() if not any(t in line for t in tokens)]


def _pointwise() -> tuple[BoundKernel[Any], dict[str, object]]:
    m, n = 1024, 1024
    args = (
        torch.empty(m, n, dtype=torch.float16),
        torch.empty(m, n, dtype=torch.float16),
        torch.empty(m, n, dtype=torch.float16),
    )
    bound = _cpu_bind(
        helion.kernel(
            se_block_bwd_grad_to_w.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        ),
        args,
    )
    return bound, {
        "block_sizes": [2, 1024],
        "num_threads": [1, 128],
        "cute_vector_widths": [2, 8],
    }


def test_the_release_is_the_first_statement_of_an_ordinary_launch(
    cpu_only: None,
) -> None:
    bound, base = _pointwise()
    off = _render(bound, base)
    on = _render(bound, base | {"cute_early_release": True})
    assert RELEASE not in off and WAIT not in off and ATTRIBUTE not in off
    assert _body(on)[0] == RELEASE
    assert on.count(RELEASE) == 1
    # An ordinary launch: no wait, no launch attribute, nothing else changes.
    assert WAIT not in on and ATTRIBUTE not in on
    assert _without(on, RELEASE) == off.splitlines()
    # ``False`` is the knob-off code, like an absent key.
    assert _render(bound, base | {"cute_early_release": False}) == off


def test_the_release_follows_the_wait_on_a_dependent_launch(cpu_only: None) -> None:
    # Both knobs: ``wait; release; ...``.  The release never precedes the
    # wait (the trigger-before-wait hazard: the dependents would start while
    # this kernel's predecessors still write).
    bound, base = _pointwise()
    pdl = _render(bound, base | {"cute_pdl": True})
    both = _render(bound, base | {"cute_pdl": True, "cute_early_release": True})
    assert _body(pdl)[0] == WAIT and RELEASE not in pdl
    assert _body(both)[:2] == [WAIT, RELEASE]
    assert both.count(RELEASE) == 1 and both.count(WAIT) == 1
    assert both.count(ATTRIBUTE) == 1
    assert _without(both, RELEASE) == pdl.splitlines()
    # The same two lines on a column reduction and on a two-input pointwise
    # kernel of the generic SIMT branch.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    for kernel, args, config in (
        (col_reduce_sum_static, (x,), _COLUMN_SUM),
        (
            scale_rows,
            (x, torch.empty(1024, dtype=torch.bfloat16)),
            {"block_sizes": [16, 64], "num_threads": [4, 64]},
        ),
    ):
        bound = _cpu_bind(kernel, args)
        off = _render(bound, dict(config))
        on = _render(bound, config | {"cute_early_release": True})
        both = _render(bound, config | {"cute_pdl": True, "cute_early_release": True})
        assert _body(on)[0] == RELEASE and _without(on, RELEASE) == off.splitlines()
        assert _body(both)[:2] == [WAIT, RELEASE]


def test_the_knob_is_a_searchable_boolean_defaulting_off(cpu_only: None) -> None:
    bound, base = _pointwise()
    spec = bound.config_spec
    assert spec.default_config().config.get("cute_early_release", False) is False
    with pytest.raises(exc.InvalidConfig, match="cute_early_release must be a boolean"):
        spec.normalized_config(
            helion.Config.from_dict(base | {"cute_early_release": 1})
        )
    # Searched both ways on an sm_90+ target: the per-kernel timer cannot
    # rank it (a release without dependents is a no-op), so a chain-aware
    # regime or a pinned config decides; the default stays off because the
    # sign depends on the dependent.
    from helion.autotuner.config_generation import ConfigGeneration

    fragment = spec._flat_fields()["cute_early_release"]
    assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
    assert fragment.search_choices == (False, True)  # pyrefly: ignore[missing-attribute]
    assert fragment.default() is False
    generation = ConfigGeneration(spec)
    assert {
        generation.unflatten(generation.flatten(config)).config.get(
            "cute_early_release"
        )
        for config in (
            helion.Config.from_dict(base | {"cute_early_release": True}),
            helion.Config.from_dict(base | {"cute_early_release": False}),
        )
    } == {True, False}


def test_pre_hopper_targets_neither_search_nor_apply_the_knob(
    cpu_only: None,
) -> None:
    bound, base = _pointwise()
    spec = bound.config_spec
    with patch.object(spec, "target_device_capability", (8, 0)):
        fragment = spec._flat_fields()["cute_early_release"]
        assert fragment.search_choices == (False,)  # pyrefly: ignore[missing-attribute]
        off = _render(bound, base)
        on = _render(bound, base | {"cute_early_release": True})
        both = _render(bound, base | {"cute_pdl": True, "cute_early_release": True})
    assert RELEASE not in on and on == off
    assert RELEASE not in both and both == off


def test_the_materialized_fission_bundle_keeps_its_own_protocol() -> None:
    bound = _bind(_computed_rhs, _args())
    serial = _serial(bound)
    # With the protocol on, the producer stage already releases its
    # dependents at entry (``generate_stage``) and the consumer's plan
    # launches with ``use_pdl``: the knob is inert on the whole bundle and
    # the producer keeps exactly one release.
    protocol = bound.to_code(
        helion.Config.from_dict(serial.config | {MATERIALIZED_PDL_KEY: True})
    )
    assert (
        bound.to_code(
            helion.Config.from_dict(
                serial.config | {MATERIALIZED_PDL_KEY: True, "cute_early_release": True}
            )
        )
        == protocol
    )
    producer, consumer = _stage_sources(protocol)
    assert ast.unparse(_device(producer).body[0]) == RELEASE
    assert producer.count(RELEASE) == 1
    # The consumer launches with ``use_pdl`` and places its own release after
    # its last MMA; no entry release is added to it.
    assert "'use_pdl': True" in consumer
    assert ast.unparse(_device(consumer).body[0]) != RELEASE
    # Without the protocol the producer is an ordinary pointwise stage and
    # takes the knob like any SIMT kernel; the native consumer never does.
    off = _stage_sources(bound.to_code(serial))
    on = _stage_sources(
        bound.to_code(
            helion.Config.from_dict(serial.config | {"cute_early_release": True})
        )
    )
    assert ast.unparse(_device(on[0]).body[0]) == RELEASE
    assert on[0].count(RELEASE) == 1 and WAIT not in on[0]
    assert _without(on[0], RELEASE) == off[0].splitlines()
    assert on[1] == off[1]
    # The bundle's search space comes from the tcgen05 branch, which never
    # offers the knob; only an explicit config reaches the producer with it.
    assert "cute_early_release" not in bound.config_spec._flat_fields()


def test_native_tcgen05_kernels_generate_the_knob_off_code(_cpu_only: None) -> None:
    # A tcgen05 GEMM places its own late waits and post-MMA release under
    # ``cute_pdl``; ``cute_early_release`` is not on its search surface and
    # an explicit request changes nothing, with ``cute_pdl`` on or off.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env:
        assert "cute_early_release" not in bound.config_spec._flat_fields()
    for pdl in (False, True):
        base = _persistent_config(loop_orders=[[0, 1]], l2_groupings=[1], cute_pdl=pdl)
        off = _source(bound, base)
        on = _source(
            bound, helion.Config(**(base.config | {"cute_early_release": True}))
        )
        assert on == off
        assert off.count(RELEASE) == (1 if pdl else 0)
