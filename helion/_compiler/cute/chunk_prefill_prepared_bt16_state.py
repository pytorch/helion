"""Original BT16 state, typed recurrence points and native issue bindings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_fragment_epilogue import bind_register_fragment_expression
from .chunk_prefill_prepared_bt16 import node_arg
from .factor_affine_reductions import pack_fp32_constexpr_loops
from .prepared_state_body import StateTransfer
from .prepared_state_body import state_transfer_instruction

if TYPE_CHECKING:
    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chunk_prefill_prepared_bt16 import BT16Bindings
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateView


def _point(
    cg: GenerateAST,
    owner: BT16Bindings,
    name: str,
    source: Node,
    target: Node,
    coordinates: tuple[str, ...],
    target_coordinates: tuple[str, ...],
    *,
    parameters: str = "value",
    fragment: str = "value",
    inputs: dict[Node, tuple[tuple[str, ...], str]] | None = None,
    paired: bool = False,
) -> tuple[str, str]:
    expression = bind_register_fragment_expression(
        cg,
        owner.graph,
        source,
        target,
        coordinates,
        target_coordinates,
        fragment,
        "cutlass.Float32",
        inputs=inputs,
    )
    expression.check()
    if paired:
        lines = [
            "result = []",
            "for element in cutlass.range_constexpr(2):",
            *("    " + line for line in expression.expression.lines),
            f"    result.append({expression.value})",
            "return result[0], result[1]",
        ]
        env = CompileEnvironment.current()
        body = pack_fp32_constexpr_loops(
            ast.parse("\n".join(lines)).body,
            fast_math=env.settings.fast_math,
            target_device_capability=env.config_spec.target_device_capability,
            float_scalar_names={"scale"},
        )
    else:
        body = ast.parse(
            "\n".join((*expression.expression.lines, f"return {expression.value}"))
        ).body
    text = ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )
    return name, "\n".join(
        (
            "@cute.jit",
            f"def {name}({parameters}):",
            *("    " + line for line in text.splitlines()),
            "",
        )
    )


def bind_bt16_state_publications(
    cg: GenerateAST, owner: BT16Bindings
) -> tuple[tuple[str, str], ...]:
    owner.check()
    gamma = node_arg(node_arg(owner.scaled_state, 1), 0)
    rhs = owner.inverse_product.rhs
    product = node_arg(rhs, 0)
    beta_widen = node_arg(node_arg(product, 0), 0)
    beta_cast = node_arg(beta_widen, 0)
    beta = node_arg(beta_cast, 0)
    difference = node_arg(product, 1)
    difference_cast = node_arg(difference, 0)
    subtraction = node_arg(difference_cast, 0)
    raw = node_arg(subtraction, 0)
    projected_cast = node_arg(node_arg(subtraction, 1), 0)
    assert node_arg(projected_cast, 0) is owner.projection.node
    assert subtraction.target is torch.ops.aten.sub.Tensor
    assert product.target is torch.ops.aten.mul.Tensor
    update = owner.output_update.rhs
    output = node_arg(owner.region.step.output_store, 2)
    output_scaled = node_arg(output, 0)
    output_scale = node_arg(output_scaled, 1)
    return (
        _point(
            cg,
            owner,
            "_helion_bt16_state_cast",
            owner.state_input,
            owner.packed_state_nodes[0],
            ("value", "key"),
            ("key", "value"),
        ),
        _point(
            cg,
            owner,
            "_helion_bt16_state_scale",
            owner.state_input,
            owner.scaled_state,
            ("value", "key"),
            ("value", "key"),
            parameters="values, coefficients",
            fragment="cutlass.Float32(values[element])",
            inputs={gamma: (("key",), "cutlass.Float32(coefficients[element])")},
            paired=True,
        ),
        _point(
            cg,
            owner,
            "_helion_bt16_projection_cast",
            owner.projection.node,
            projected_cast,
            ("token", "value"),
            ("token", "value"),
        ),
        _point(
            cg, owner, "_helion_bt16_beta_cast", beta, beta_cast, ("token",), ("token",)
        ),
        _point(
            cg,
            owner,
            "_helion_bt16_rhs",
            projected_cast,
            rhs,
            ("token", "value"),
            ("token", "value"),
            parameters="value, raw, beta",
            inputs={raw: (("token", "value"), "raw"), beta_cast: (("token",), "beta")},
        ),
        _point(
            cg,
            owner,
            "_helion_bt16_update_cast",
            owner.inverse_product.node,
            update,
            ("token", "value"),
            ("token", "value"),
        ),
        _point(
            cg,
            owner,
            "_helion_bt16_output",
            owner.output_update.node,
            output,
            ("token", "value"),
            ("token", "value"),
            parameters="values, scale",
            fragment="cutlass.Float32(values[element])",
            inputs={output_scale: ((), "scale")},
            paired=True,
        ),
    )


def bind_bt16_issues(owner: BT16Bindings) -> tuple[tuple[object, ...], ...]:
    """Physical descriptor geometry for the five original recurrence products."""
    owner.check()
    specs = (
        owner.projection,
        owner.query,
        owner.inverse_product,
        owner.output_update,
        owner.state_update,
    )
    if any(
        spec.operand_dtypes != (torch.bfloat16, torch.bfloat16)
        or spec.result_dtype is not torch.float32
        for spec in specs
    ):
        raise chain._UnsupportedChain("changed BT16 recurrence precision")
    return (
        (16, 128, 16, 1024, 128, 0, (2, 4, 16, 128)),
        (16, 128, 16, 1024, 128, 0, (2, 4, 16, 128)),
        (16, 16, 16, 256, 32, 0, (2, 1, 16, 32)),
        (16, 16, 16, 256, 32, 0, (2, 1, 16, 32)),
        (64, 16, 2048, 1024, 128, 1, (2, 1, 16, 128)),
    )


@dataclass(frozen=True)
class BT16StateBinding:
    owner: BT16Bindings
    source: Node
    destination: Node
    width: int
    transfer: StateTransfer
    port: int

    @property
    def offset(self) -> int:
        return 0

    @property
    def scope(self) -> tuple[object, ...]:
        return id(self.owner), "bt16-native-state-port", self.port

    def facts(self) -> object:
        self.owner.check()
        return (
            id(self.owner),
            self.source,
            self.destination,
            self.width,
            self.transfer,
            self.port,
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .prepared_state_planner import StateView

        return StateView(
            self.destination if store else self.source,
            self.scope,
            (self.transfer, store),
            0,
            self.width,
        )

    def check_publication(self, request: StatePublication) -> None:
        if (
            request.read is not self
            or request.value is not self
            or request.read_before is not None
            or any(
                cut.owner is not self.owner
                or cut.anchor is not self.destination
                or cut.scope != self.scope
                for cut in (
                    request.transform_before,
                    request.store_before,
                    request.complete_before,
                )
                if cut is not None
            )
        ):
            raise chain._UnsupportedChain("changed BT16 state transfer cut")


def bind_bt16_state(owner: BT16Bindings) -> tuple[tuple[tuple[object, ...], ...], ...]:
    from .prepared_state_planner import StateCut
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import plan_state_transfers

    owner.check()
    ports = (
        (owner.state_input, owner.packed_state_nodes[0], 16, "32x32b", 16, "32x32b"),
        (owner.state_input, owner.scaled_state, 32, "32x32b", 32, "32x32b"),
        (owner.projection.node, owner.inverse_product.rhs, 16, "16x256b", 2, "16x128b"),
        (
            owner.inverse_product.node,
            owner.output_update.rhs,
            16,
            "32x32b",
            16,
            "32x32b",
        ),
        (
            owner.output_update.node,
            node_arg(owner.region.step.output_store, 2),
            16,
            "16x256b",
            2,
            None,
        ),
        (owner.state_input, owner.scaled_state, 16, "32x32b", 16, "32x32b"),
        (owner.state_input, owner.packed_state_nodes[0], 32, "32x32b", 32, "32x32b"),
    )
    result = []
    for port, (source, destination, width, load_shape, count, store_shape) in enumerate(
        ports
    ):
        transfer = StateTransfer(
            "source",
            "values",
            "destination",
            "None",
            True,
            load_shape,
            count,
            store_shape,
        )
        binding = BT16StateBinding(owner, source, destination, width, transfer, port)
        cut = StateCut(owner, destination, binding.scope)
        plan = plan_state_transfers(
            (
                StatePublication(
                    binding,
                    binding,
                    binding.state_view(),
                    binding.state_view(store=True),
                    cut,
                    cut,
                    cut,
                ),
            ),
            (cut,),
        )
        plan.check()
        result.append(
            tuple(
                state_transfer_instruction(effect.kind, transfer)
                for effect in plan.actions
                if effect.kind != "transform"
            )
        )
    return tuple(result)


def bind_bt16_state_abi(
    owner: BT16Bindings, semantic_root: GraphInfo, *, max_bits: int = 256
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    from .chunk_prefill_prepared_state_abi import bind_external_state_abi

    owner.check()
    return bind_external_state_abi(owner.region, semantic_root, max_bits=max_bits)


def bind_bt16_output(owner: BT16Bindings) -> tuple[int, int, int, int]:
    """Original typed global store and its existing two-segment SMEM image."""
    owner.check()
    output = node_arg(owner.region.step.output_store, 2)
    scaled = node_arg(output, 0)
    if (
        output.target is not torch.ops.prims.convert_element_type.default
        or output.args[1] is not torch.bfloat16
        or scaled.target is not torch.ops.aten.mul.Tensor
        or node_arg(scaled, 0) is not owner.output_update.node
        or node_arg(scaled, 1) is not owner.region.step.output_scale
    ):
        raise chain._UnsupportedChain("changed original BT16 output publication")
    rows, columns = owner.region.chunk_size, owner.region.value_width
    return rows, columns, 64, columns // 64
