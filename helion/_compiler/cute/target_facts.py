"""Target facts: the device model behind the CuTe planners' budget probes (analysis).

The facts about the compile target are answered by probes:
``helion._compat.target_device_capability`` (also bound by name in
``helion.runtime.kernel`` and ``helion._compiler.compile_environment``;
``patch_target_facts`` patches all three), ``helion.runtime.get_num_sm``,
``cute.tcgen05.search_plan._cuda_num_sms_or_zero`` (the wave-quantization
gate's raw SM count), the per-CTA SMEM probes of this module
(:func:`per_cta_smem_capacity_bytes` and the two budgets derived from it),
``thread_budget.shared_memory_budget_bytes`` and
``helion._hardware.get_hardware_info``.  :class:`TargetFacts` is the one
record of those answers.  :meth:`TargetFacts.for_device` resolves them lazily
through the same probe seams (so a test patching a probe sees the patched
value here as well), and ``helion._testing.patch_target_facts`` installs one
record at every seam at once; the ``B200`` / ``GB300`` constants are the CPU
test models the suite's fixtures pin.

Production readers call the probes through their modules
(``target_facts.per_cta_smem_capacity_bytes(device)``, never a bound name),
which is what keeps a patch at the seam visible to every reader; migrating
them to a per-compile ``TargetFacts`` is the follow-up once the fixtures are
consolidated.
"""

from __future__ import annotations

import dataclasses

import torch

from ..._hardware import HardwareInfo
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES


def per_cta_smem_capacity_bytes(device: torch.device) -> int:
    """The opt-in dynamic shared memory one CTA of ``device`` may request
    (bytes); 0 off CUDA.  The tcgen05 planners' capacity, and the probe the
    device-model fixtures patch (the two budgets below read it through this
    module, so a patched capacity reprices them as well)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(device)
    optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(props.shared_memory_per_block, optin_shared)


def per_cta_smem_budget_bytes(device: torch.device) -> int:
    """The capacity less the fixed AB-gate reservation (never negative)."""
    device_cap = per_cta_smem_capacity_bytes(device)
    return max(0, device_cap - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES)


def per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
    """The 3-stage AB pipeline's budget: the capacity less the reservation,
    or 0 on a device below the pipeline's minimum opt-in capacity."""
    device_cap = per_cta_smem_capacity_bytes(device)
    if device_cap < TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN:
        return 0
    # Keep a fixed headroom reservation: CuTe's raw opt-in limit does not
    # include every barrier/runtime byte the 3-stage AB pipeline needs.
    return device_cap - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES


@dataclasses.dataclass(frozen=True)
class TargetFacts:
    """The per-device facts the CuTe search surface and emitters read.

    ``None`` means "not modelled": ``patch_target_facts`` leaves that probe
    alone and production derives it the usual way (a modelled
    ``smem_capacity_bytes`` with ``smem_budget_bytes=None`` yields the
    production budget, capacity minus the 28 KiB AB reserve).
    """

    capability: tuple[int, int] | None = None
    num_sm: int | None = None
    # The raw SM count ``_cuda_num_sms_or_zero`` reports to the cluster_m=2
    # wave-quantization gate (0 keeps that search live on any host).
    matmul_num_sm: int | None = None
    smem_capacity_bytes: int | None = None
    smem_budget_bytes: int | None = None
    # ``thread_budget.shared_memory_budget_bytes`` (the current
    # device's opt-in SMEM, read by the strided/grouped reductions).
    reduction_smem_budget_bytes: int | None = None
    hardware_name: str | None = None
    runtime_version: str = "13.0"
    is_hip: bool | None = None

    @property
    def compute_capability(self) -> str | None:
        """``sm100``-style name of :attr:`capability`."""
        if self.capability is None:
            return None
        major, minor = self.capability
        return f"sm{major}{minor}"

    def hardware_info(self) -> HardwareInfo:
        """The ``get_hardware_info`` record of this target."""
        assert self.hardware_name is not None and self.capability is not None
        compute_capability = self.compute_capability
        assert compute_capability is not None
        return HardwareInfo(
            device_kind="cuda",
            hardware_name=self.hardware_name,
            runtime_version=self.runtime_version,
            compute_capability=compute_capability,
        )

    @classmethod
    def for_device(cls, device: torch.device) -> TargetFacts:
        """Resolve the facts of ``device`` through the production probes.

        Every probe is looked up through its module at call time, so the
        patches tests install at those seams are honored.
        """
        from ... import _compat
        from ... import _hardware
        from ... import runtime
        from . import thread_budget
        from .tcgen05 import search_plan

        capability = _compat.target_device_capability(device)
        info = _hardware.get_hardware_info(device) if device.type == "cuda" else None
        return cls(
            capability=capability,
            num_sm=runtime.get_num_sm(device) if device.type == "cuda" else None,
            matmul_num_sm=search_plan._cuda_num_sms_or_zero(device),
            smem_capacity_bytes=per_cta_smem_capacity_bytes(device),
            smem_budget_bytes=per_cta_smem_budget_bytes(device),
            reduction_smem_budget_bytes=(
                thread_budget.shared_memory_budget_bytes()
                if device.type == "cuda"
                else None
            ),
            hardware_name=None if info is None else info.hardware_name,
            runtime_version="13.0" if info is None else info.runtime_version,
            is_hip=_compat.is_hip(),
        )


# The static ``cute.arch.alloc_smem`` budget of the planners that size a
# shared-memory exchange or staging buffer at compile time without the
# opt-in dynamic carve-out (``hl.split`` exchanges, the ``cp.async`` state
# ring, the row-resident and row-matrix-transport stagings): the 48 KiB
# every supported capability grants a launch without
# ``cudaFuncAttributeMaxDynamicSharedMemorySize``.  A larger static
# allocation fails in the NVVM backend rather than at the planner's check.
_STATIC_SMEM_BUDGET_BYTES = 48 * 1024


def static_smem_budget() -> int:
    """The static shared-memory budget (bytes) of a planner: the 48 KiB
    static-allocation limit is the same on every CUDA capability, so the
    budget takes no target; a per-capability value belongs here when one
    exists."""
    return _STATIC_SMEM_BUDGET_BYTES


# The CPU test models.  ``B200`` is what ``test/_cute_binding.py`` /
# ``test/cute_population_contracts.py`` pin: capability (10, 0), 148 SMs, the
# 227 KiB opt-in SMEM as the tcgen05 capacity (the budget derives from it in
# production), no HIP.  ``GB300`` is the sm103 sibling with 152 SMs.
B200 = TargetFacts(
    capability=(10, 0),
    num_sm=148,
    smem_capacity_bytes=232448,
    hardware_name="NVIDIA B200",
    is_hip=False,
)
GB300 = TargetFacts(
    capability=(10, 3),
    num_sm=152,
    smem_capacity_bytes=232448,
    hardware_name="NVIDIA GB300",
    is_hip=False,
)
