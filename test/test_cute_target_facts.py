"""``TargetFacts.for_device`` reads every probe ``patch_target_facts`` installs.

Backend-independent: it exercises the test harness itself, so
a seam the patcher misses (the finding this pins: ``for_device`` read
``_compat.target_device_capability`` while only the by-name bindings were
patched) shows up here on the CPU before a fixture relies on it.
"""

from __future__ import annotations

import torch

from helion._compiler.cute.target_facts import B200
from helion._compiler.cute.target_facts import TargetFacts
from helion._testing import patch_target_facts


def test_for_device_reads_every_patched_probe() -> None:
    facts = TargetFacts(
        capability=(9, 0),
        num_sm=132,
        matmul_num_sm=132,
        smem_capacity_bytes=232448,
        smem_budget_bytes=200000,
        reduction_smem_budget_bytes=232448,
        hardware_name="NVIDIA H100",
        is_hip=False,
    )
    with patch_target_facts(facts):
        assert TargetFacts.for_device(torch.device("cuda:0")) == facts


def test_for_device_capability_on_a_cpu_device() -> None:
    # Only the capability is modelled; the CPU device answers the rest itself
    # (no SM count, no hardware record, a zero SMEM capacity).
    with patch_target_facts(capability=(9, 0), is_hip=False):
        facts = TargetFacts.for_device(torch.device("cpu"))
    assert facts.capability == (9, 0)
    assert facts.compute_capability == "sm90"
    assert facts.num_sm is None
    assert facts.hardware_name is None
    assert facts.smem_capacity_bytes == 0
    assert facts.is_hip is False


def test_models_derive_the_production_budget() -> None:
    # ``smem_budget_bytes`` is unmodelled on the B200 model: the production
    # derivation (capacity minus the 28 KiB AB reserve) stays in force.
    with patch_target_facts(B200, matmul_num_sm=148, reduction_smem_budget_bytes=0):
        facts = TargetFacts.for_device(torch.device("cuda:0"))
    assert facts.capability == (10, 0)
    assert facts.num_sm == 148
    assert facts.smem_capacity_bytes == 232448
    assert facts.smem_budget_bytes == 232448 - 28 * 1024
    assert facts.hardware_name == "NVIDIA B200"
    assert facts.hardware_info() == B200.hardware_info()
