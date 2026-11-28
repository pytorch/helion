"""Lower a certified complete paged K fold with the ordinary fragment MMA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .analysis.gathered_bf16_mma import prove_gathered_bf16_mma
from .fragment_context import Fragment

if TYPE_CHECKING:
    from torch.fx import Node

    from .computed_fragment import FragmentCompiler


def gathered_bf16_dot(
    compiler: FragmentCompiler, node: Node, values: dict[Node, object]
) -> list[Fragment] | None:
    """Keep the dot in registers across K tiles; preserve its outer consumers."""
    capability = compiler.env.config_spec.target_device_capability
    if capability is None or capability[0] < 8 or compiler.threads % 32:
        return None
    proof = prove_gathered_bf16_mma(node, compiler.graphs, compiler.env)
    if proof is None:
        return None
    compiler.used_gathered_bf16_fold = True
    df = compiler.df

    lhs, bank = compiler.host_tensor(proof.lhs), compiler.host_tensor(proof.rhs)
    # Read the already published table result before recycling fragment
    # storage. Do not move the original table load across a control transfer.
    compiler.flush_pending_barrier()
    page = df.new_var("gathered_mma_page")
    compiler.emit(f"{page} = {compiler.scalar(values[proof.page])}")
    page = compiler.normalize_signed_index(page, compiler.sym(bank.value.size(0)))
    valid = df.new_var("gathered_mma_valid")
    compiler.emit(
        f"{valid} = ({page} >= 0) & ({page} < {compiler.sym(bank.value.size(0))})"
    )
    stride_q, stride_qk = (compiler.host_stride(lhs, dim) for dim in (0, 1))
    stride_rk = compiler.host_stride(bank, 2)
    stride_page, stride_row = (compiler.host_stride(bank, dim) for dim in (0, 1))

    def query(coords: tuple[str, ...]) -> str:
        row, k = coords
        return f"({lhs.name}.iterator + cutlass.Int64({row}) * cutlass.Int64({stride_q}) + cutlass.Int64({k}) * cutlass.Int64({stride_qk})).load()"

    def transposed_key(coords: tuple[str, ...]) -> str:
        k, row = coords
        # Mask the dereference itself. An invalid physical page has zero
        # elements, but still participates in MMA with the original Q.
        word = df.new_var("gathered_mma_key")
        compiler.guarded_assign(
            word,
            lambda: (
                f"({bank.name}.iterator + cutlass.Int64({page}) * cutlass.Int64({stride_page})"
                f" + cutlass.Int64({row}) * cutlass.Int64({stride_row}) + cutlass.Int64({k}) * cutlass.Int64({stride_rk})).load()"
            ),
            "cutlass.BFloat16(0)",
            [valid],
        )
        return word

    return [
        compiler.emit_mma(
            Fragment((proof.rows, proof.depth), torch.bfloat16, query),
            Fragment((proof.depth, proof.columns), torch.bfloat16, transposed_key),
            None,
            (proof.rows, proof.columns),
            proof.depth,
            torch.float32,
            full_k_fold=True,
        )
    ]
