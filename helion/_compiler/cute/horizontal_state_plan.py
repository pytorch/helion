"""Selected physical schedule; the ordinary route never constructs this plan."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ... import exc
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from .horizontal_state_config import CONFIG_KEY
from .horizontal_state_config import HORIZONTAL
from .horizontal_state_update import HorizontalGeometry
from .horizontal_state_update import InapplicableStateUpdate
from .horizontal_state_update import StateTaskDomain
from .horizontal_state_update import _freeze
from .horizontal_state_update import bind_state_task_domain
from .horizontal_state_update import discover_pointwise_state_region

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


@dataclasses.dataclass(frozen=True)
class HorizontalStatePlan:
    domain: StateTaskDomain
    geometry: HorizontalGeometry
    config: object
    device: DeviceFunction = dataclasses.field(repr=False)

    def check(self) -> None:
        self.domain.check()
        if self.device is not DeviceFunction.current() or self.config != _freeze(
            self.device.config.config
        ):
            raise InapplicableStateUpdate("horizontal plan attempt changed")
        rows = self.device.resolved_block_size(self.domain.row_block_id)
        if type(rows) is not int or rows != self.geometry.rows:
            raise InapplicableStateUpdate("original row task width changed")
        if self.geometry != HorizontalGeometry(
            rows // 16, 2, 32, self.domain.state_shape[-1], 2
        ):
            raise InapplicableStateUpdate(
                "horizontal geometry no longer matches its domain"
            )
        self.geometry.check(48 * 1024)


def plan_horizontal_state(graphs: Sequence[GraphInfo]) -> HorizontalStatePlan | None:
    df = DeviceFunction.current()
    if df.config.config.get(CONFIG_KEY) != HORIZONTAL:
        return None
    try:
        region = discover_pointwise_state_region(graphs)
        if region is None:
            raise InapplicableStateUpdate(
                "selected horizontal root is not pointwise state plus row sum"
            )
        domain = bind_state_task_domain(region)
        capability = CompileEnvironment.current().config_spec.target_device_capability
        rows = df.resolved_block_size(domain.row_block_id)
        if (
            capability is None
            or capability[0] != 10
            or type(rows) is not int
            or rows not in (16, 32, 64, 128)
            or domain.storage_dtype not in (torch.bfloat16, torch.float16)
            or domain.state_shape[-1] % 32
            or df.config.pid_type != "flat"
            or df.cute_state.simt_cluster_n != 1
            or region.state_load.args[2] is not None
            or region.state_store.args[3] is not None
        ):
            raise InapplicableStateUpdate(
                "horizontal geometry, dtype or state mask unsupported"
            )
        geometry = HorizontalGeometry(rows // 16, 2, 32, domain.state_shape[-1], 2)
        geometry.check(48 * 1024)
        plan = HorizontalStatePlan(domain, geometry, _freeze(df.config.config), df)
        plan.check()
        return plan
    except InapplicableStateUpdate as error:
        raise exc.BackendUnsupported("cute", str(error)) from error
