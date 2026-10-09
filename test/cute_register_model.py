"""Execute ordinary register-plan operations in the CPU warp model.

This checks the serialized plan, register ownership, and generated memory code.
The runtime's TensorSSA lowering is covered separately by SDK and native tests.
"""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


def gather_registers(
    value: np.ndarray, mapping: Mapping[str, Any], lane: int
) -> np.ndarray:
    """Model the emitted static local map, including same-dtype zero padding."""
    group = int(lane) % len(mapping["owners"])
    indices = mapping["rows"][mapping["owners"][group]]
    return np.concatenate((value, np.zeros_like(value)))[indices]


def execute_register_plan_events(
    events: Sequence[Any],
) -> tuple[list[tuple[np.ndarray, ...]], int]:
    source = events[0][0][0]
    assert all(event[0][0] == source for event in events)
    plan = json.loads(source)
    groups = plan["groups"]
    assert len(events) == 32 and 32 % groups == 0
    assert all(
        int(event[0][2]) % groups == lane % groups for lane, event in enumerate(events)
    )
    outputs = []
    operations = {
        "aten.minimum.default": np.minimum,
        "aten.maximum.default": np.maximum,
        "aten.fmin.default": np.fmin,
        "aten.fmax.default": np.fmax,
        "aten.add.Tensor": operator.add,
        "aten.sub.Tensor": operator.sub,
        "aten.mul.Tensor": operator.mul,
        "aten.neg.default": operator.neg,
        "aten.eq.Tensor": operator.eq,
        "aten.ne.Tensor": operator.ne,
        "aten.lt.Tensor": operator.lt,
        "aten.le.Tensor": operator.le,
        "aten.gt.Tensor": operator.gt,
        "aten.ge.Tensor": operator.ge,
        "aten.bitwise_and.Tensor": operator.and_,
        "aten.bitwise_or.Tensor": operator.or_,
        "aten.bitwise_xor.Tensor": operator.xor,
    }
    for begin in range(0, 32, groups):
        values = {
            item["id"]: np.stack(
                [events[lane][0][1][index] for lane in range(begin, begin + groups)]
            )
            for index, item in enumerate(plan["inputs"])
        }
        for item in plan["inputs"]:
            assert values[item["id"]].shape == tuple(item["shape"])
            assert values[item["id"]].dtype == np.dtype(item["dtype"])
        for node in plan["nodes"]:
            args = [values[index] for index in node["inputs"]]
            op = node["op"]
            if op == "constant":
                rows = [
                    np.frombuffer(bytes.fromhex(row), dtype=node["dtype"])
                    for row in node["rows"]
                ]
                result = np.stack([rows[owner] for owner in node["owners"]])
            elif op in (
                "aten.gather.default",
                "aten.index_select.default",
                "aten.slice.Tensor",
            ):
                mapping = plan["maps"][node["map"]]
                indices = np.array(
                    [mapping["rows"][owner] for owner in mapping["owners"]]
                )
                result = np.take_along_axis(args[0], indices, axis=node["axis"])
                if node["axis"] == 0:
                    dead = set(range(node["shape"][1])) - set(node["live_registers"])
                    result[:, list(dead)] = 0
            elif op == "aten.where.self":
                if "static_map" in node:
                    mapping = plan["maps"][node["static_map"]]
                    indices = np.array(
                        [mapping["rows"][owner] for owner in mapping["owners"]]
                    )
                    result = np.take_along_axis(
                        np.concatenate(args, axis=1), indices, axis=1
                    )
                else:
                    result = np.where(*args)
            elif op == "aten.cat.default":
                result = np.concatenate(args, axis=1)
            else:
                result = operations[op](*args)
            assert result.shape == tuple(node["shape"])
            assert result.dtype == np.dtype(node["dtype"])
            values[node["id"]] = result
        outputs.extend(
            tuple(values[index][lane].copy() for index in plan["outputs"])
            for lane in range(groups)
        )
    exchanges = sum(
        sum(
            any(
                mapping["rows"][owner][register] != group
                for group, owner in enumerate(mapping["owners"])
            )
            for register in node["live_registers"]
        )
        for node in plan["nodes"]
        if node.get("axis") == 0
        for mapping in [plan["maps"][node["map"]]]
    )
    return outputs, exchanges
