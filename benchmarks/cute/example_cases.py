# ruff: noqa: ANN401
"""Three workload sizes for each non-distributed example family.

Collection invokes each example's existing validation setup while temporarily
recording its public benchmark callables. It does not run autotuning or timing.
"""

from __future__ import annotations

from contextlib import ExitStack
import dataclasses
import importlib
import operator
import os
from typing import Any
from typing import Callable
from unittest.mock import patch

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class Case:
    module: str
    name: str
    args: tuple[Any, ...]
    kernels: dict[str, Callable]
    baselines: dict[str, Callable]
    mode: str = "forward"
    rtol: float = 1e-2
    atol: float = 1e-1
    check: str = "close"
    description: str = ""
    gradient_inputs: tuple[torch.Tensor, ...] | None = None
    gradient_names: tuple[str, ...] = ()
    # Defaults to the reference output's dtype. Overrides name one provider's
    # single-tensor result, not an allowed set shared by every implementation.
    output_dtypes: dict[str, torch.dtype] = dataclasses.field(default_factory=dict)
    reference_note: str = ""


# Three deliberately different workload sizes for every operation. Dimensions
# follow each example's check()/test() signature, rather than one universal
# matrix size that is inappropriate for several of these operations.
CHECKS = {
    "add": ("check", [(256, 1024), (4096, 4096), (16384, 4096)]),
    "exp": ("check", [(262144,), (4194304,), (33554432,)]),
    "sum": ("check", [(256, 1024), (4096, 4096), (8192, 16384)]),
    "long_sum": ("check", [(4, 32768), (16, 131072), (64, 524288)]),
    "softmax": ("check", [(256, 1024), (4096, 4096), (4096, 32768)]),
    "acfs.softmax_acf": ("check", [(256, 1024), (4096, 4096), (4096, 32768)]),
    "batch_softmax": ("check", [(4, 64, 256), (8, 256, 1024), (16, 512, 4096)]),
    "rms_norm": ("check", [(256, 1024), (4096, 4096), (8192, 8192)]),
    "welford": ("check", [(1024, 512), (16384, 1024), (65536, 2048)]),
    "matmul": ("check", [(256, 256, 256), (1024, 1024, 1024), (2048, 4096, 2048)]),
    "bmm": ("check", [(4, 64, 128, 128), (8, 256, 256, 512), (16, 512, 768, 1024)]),
    "broadcast_matmul": (
        "check",
        [(4, 64, 128, 128), (8, 256, 256, 512), (16, 512, 768, 1024)],
    ),
    "matmul_layernorm": (
        "check",
        [(128, 256, 256), (512, 1024, 1024), (2048, 2048, 2048)],
    ),
    "matmul_split_k": ("check", [(32, 4096, 64), (64, 16384, 128), (128, 32768, 256)]),
    "split_k_barrier": ("check", [(16, 4096, 16), (64, 8192, 64), (128, 16384, 128)]),
    "epilogue_subtiling": (
        "check",
        [(256, 256, 256), (1024, 1024, 1024), (2048, 4096, 2048)],
    ),
    "bf16xint16_gemm": (
        "check",
        [(128, 256, 128), (512, 1024, 512), (1024, 2048, 1024)],
    ),
    "int4_gemm": ("check", [(128, 256, 128), (512, 1024, 512), (1024, 2048, 1024)]),
    "fp8_gemm": ("check", [(256, 256, 256), (1024, 1024, 1024), (2048, 4096, 2048)]),
    "fp8_matmul": ("check", [(256, 256, 256), (1024, 1024, 1024), (2048, 4096, 2048)]),
    "fused_linear_jsd": (
        "check",
        [(64, 128, 1024), (128, 512, 4096), (256, 1024, 16384)],
    ),
    "gather_gemv": ("check", [(8, 2048, 2), (8, 4096, 2), (8, 8192, 2)]),
    "moe_matmul_ogs": (
        "check",
        [(256, 128, 128, 4), (1024, 512, 512, 8), (4096, 1024, 1024, 16)],
    ),
    "attention": ("test", [(1, 4, 256, 64), (2, 8, 1024, 64), (2, 16, 2048, 128)]),
    "blackwell_attention": (
        "test",
        [(1, 4, 256, 64), (2, 8, 1024, 64), (2, 16, 2048, 128)],
    ),
    "flex_attention": ("test", [(1, 4, 256, 64), (2, 8, 1024, 64), (2, 16, 2048, 128)]),
    "fp8_attention": ("check", [(1, 4, 256, 64), (2, 8, 1024, 64), (2, 16, 2048, 128)]),
    "jagged_hstu_attn": ("test", [(2, 128, 4, 32), (4, 256, 8, 64), (8, 512, 8, 64)]),
    "geglu": ("check_geglu_kernel", [((256, 1024),), ((4096, 4096),), ((8192, 8192),)]),
    "swiglu": (
        "check_swiglu_kernel",
        [((256, 1024),), ((4096, 4096),), ((8192, 8192),)],
    ),
    "jsd": ("check_jsd_kernel", [(1, 128, 1024), (2, 256, 4096), (4, 512, 16384)]),
    "kl_div": (
        "check_kl_div_kernel",
        [(1, 128, 1024), (2, 256, 4096), (4, 512, 16384)],
    ),
    "se_block": ("check", [(256, 128), (1024, 512), (4096, 1024)]),
    "squeeze_and_excitation_net": (
        "check",
        [(256, 64, 256), (1024, 256, 512), (4096, 512, 1024)],
    ),
    "sparse_attn_indexer": ("check", [(1, 4096), (8, 16384), (512, 4096)]),
    "xsa": ("check", [(1, 4, 256, 64), (2, 8, 1024, 64), (2, 16, 2048, 128)]),
}

CUSTOM = [
    "concatenate",
    "cross_entropy",
    "embedding",
    "grouped_gemm",
    "grpo_loss",
    "jagged_dense_add",
    "jagged_dense_bmm",
    "jagged_layer_norm",
    "jagged_mean",
    "jagged_softmax",
    "jagged_sum",
    "jagged_hstu_attn_2",
    "layer_norm",
    "rope",
    "segment_reduction",
    "low_mem_dropout",
    "mamba2_chunk_scan",
    "mamba2_chunk_state",
    "gdn_fwd_h",
    "nvfp4_gemm",
    "nvfp4_gemv",
    "aot_example",
    "aot_compile_example",
]
LINEAR = [
    "linear.example_vanilla_linear_attn",
    "linear.example_simple_gla",
    "linear.example_retention",
    "linear.example_full_gla",
    "linear.example_delta_rule",
    "linear.example_gated_delta_rule",
    "linear.example_kda",
    "linear.example_mamba2_ssd",
]
MODULES = [*CHECKS, *CUSTOM, *LINEAR]


def rand(
    *shape: int, dtype: torch.dtype = torch.float32, grad: bool = False
) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=grad)


def reference_hstu_strict(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    targets: Any,
    length: int,
) -> torch.Tensor:
    # Match the kernel's tile_q > tile_kv mask. The shipped reference instead
    # includes the diagonal, which compares different attention operations.
    lengths = (offsets[1:] - offsets[:-1]).tolist()
    outputs = []
    for qi, ki, vi in zip(
        q.split(lengths), k.split(lengths), v.split(lengths), strict=True
    ):
        qi, ki, vi = qi.transpose(0, 1), ki.transpose(0, 1), vi.transpose(0, 1)
        scores = (qi @ ki.transpose(-1, -2)) / v.shape[-1] ** 2
        scores = (scores / (1 + torch.exp(-scores))) / length
        outputs.append((scores.tril(diagonal=-1) @ vi).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def reference_jsd_logits(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature
    student_prob = student.softmax(-1)
    teacher_prob = teacher.softmax(-1)
    log_mix = ((1 - beta) * student_prob + beta * teacher_prob).log()
    student_kl = (student_prob * (student.log_softmax(-1) - log_mix)).sum(-1)
    teacher_kl = (teacher_prob * (teacher.log_softmax(-1) - log_mix)).sum(-1)
    return ((1 - beta) * student_kl + beta * teacher_kl).mean()


def collect(module: str, shape_index: int) -> list[Case]:
    import helion._testing as testing

    if module not in MODULES or shape_index not in range(3):
        raise ValueError((module, shape_index))
    from examples.linear.linear_attention_utils import ACC_BWD_TOL
    from examples.linear.linear_attention_utils import ACC_FWD_TOL

    cases: list[Case] = []
    description = ""

    def record(
        kernel_fn: Callable | dict,
        baseline_fn: Callable | dict,
        args: tuple = (),
        kernel_name: str = "helion",
        baseline_name: str = "aten",
        bwd: bool = False,
        rtol: float = 1e-2,
        atol: float = 1e-1,
        output_dtypes: dict[str, torch.dtype] | None = None,
        reference_note: str = "",
        **kwargs: Any,
    ) -> dict:
        kernels = kernel_fn if isinstance(kernel_fn, dict) else {kernel_name: kernel_fn}
        baselines = (
            baseline_fn
            if isinstance(baseline_fn, dict)
            else {baseline_name: baseline_fn}
        )
        mode = "backward" if bwd else "forward"
        name = f"{len(cases):02d}_{mode}"
        cases.append(
            Case(
                module,
                name,
                args,
                kernels,
                baselines,
                mode,
                rtol,
                atol,
                description=description,
                output_dtypes={} if output_dtypes is None else output_dtypes,
                reference_note=reference_note,
            )
        )
        return dict.fromkeys([*kernels, *baselines], 1.0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(testing, "run_example", record))
        torch.manual_seed(20260911 + shape_index)
        m = importlib.import_module(f"examples.{module}")
        i = shape_index
        if module in CHECKS:
            fn, shapes = CHECKS[module]
            shape = shapes[i]
            description = f"{fn}{shape}"
            extra = (
                {"dtype": torch.bfloat16}
                if module
                in {
                    "attention",
                    "blackwell_attention",
                    "flex_attention",
                    "jagged_hstu_attn",
                }
                else {}
            )
            if module == "blackwell_attention":
                # This example has an extra timing call after run_example.
                stack.enter_context(patch.object(m, "do_bench", lambda fn: 1.0))
            getattr(m, fn)(*shape, **extra)
            if module == "jagged_hstu_attn":
                cases[0].baselines["torch"] = reference_hstu_strict
            if module == "softmax":
                cases[0].kernels["helion decomposed"] = m.softmax_decomposed
            if module == "exp":
                record(m.exp, torch.exp, cases[0].args)
            if module == "fused_linear_jsd":
                beta, ignore_index, temperature, sw, tw, sx, tx = cases[0].args
                record(
                    m.fused_linear_jsd_kernel,
                    reference_jsd_logits,
                    (beta, ignore_index, temperature, sx @ sw.T, tx @ tw.T),
                )
            if module == "attention":
                cases[0].kernels["output only"] = m.attention_output
                q, k, v = cases[0].args
                record(
                    {
                        "causal": lambda q, k, v: m.causal_attention(q, k, v)[0],
                        "causal output": m.causal_attention_output,
                    },
                    lambda q, k, v: F.scaled_dot_product_attention(
                        q, k, v, is_causal=True
                    ),
                    (q, k, v),
                )
                record(
                    m.attention_relu_output,
                    lambda q, k, v: F.scaled_dot_product_attention(q, k, v).relu(),
                    (q, k, v),
                )
                record(
                    m.causal_attention_relu_output,
                    lambda q, k, v: F.scaled_dot_product_attention(
                        q, k, v, is_causal=True
                    ).relu(),
                    (q, k, v),
                )
                bias = rand(
                    q.shape[0], q.shape[1], q.shape[2], k.shape[2], dtype=q.dtype
                )
                record(
                    {
                        "biased": lambda q, k, v, b: m.biased_attention(q, k, v, b)[0],
                        "biased output": m.biased_attention_output,
                    },
                    lambda q, k, v, b: F.scaled_dot_product_attention(
                        q, k, v, attn_mask=b
                    ),
                    (q, k, v, bias),
                )
            if module == "blackwell_attention":
                q, k, v = cases[0].args
                q.requires_grad_()
                k.requires_grad_()
                v.requires_grad_()
                record(
                    m.blackwell_attention_fwd_bwd,
                    F.scaled_dot_product_attention,
                    (q, k, v),
                    bwd=True,
                )
            if module == "swiglu":
                cases[0].kernels["natural rank"] = m._swiglu_fwd_pallas
                a, b = cases[0].args
                a.requires_grad_()
                b.requires_grad_()
                record(
                    m.swiglu, next(iter(cases[0].baselines.values())), (a, b), bwd=True
                )
            if module == "geglu":
                cases[0].kernels["natural rank"] = m._geglu_pallas
        elif module in LINEAR:
            shape = [(1, 4, 128, 32, 32), (1, 8, 256, 64, 64), (1, 8, 512, 64, 64)][i]
            description = f"B,H,T,D,Dv={shape}, chunk=64"
            if module == "linear.example_mamba2_ssd":
                from examples.linear.linear_attention_engine import (
                    LinearAttentionVariant,
                )
                from examples.linear.linear_attention_harness import (
                    LinearAttentionExampleHarness,
                )

                harness = LinearAttentionExampleHarness(
                    LinearAttentionVariant.MAMBA2_SSD
                )
            else:
                harness = m.HARNESS
            inputs = harness.make_inputs(*shape, requires_grad=False)
            assert inputs.g is not None
            decay = inputs.g
            for tensor_name in harness.grad_tensors:
                getattr(inputs, tensor_name).requires_grad_()
            from examples.linear.linear_attention_utils import ACC_BWD_TOL
            from examples.linear.linear_attention_utils import ACC_FWD_TOL
            from examples.linear.linear_attention_utils import (
                chunked_linear_attn_reference,
            )

            def hl_fn(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return harness.helion_fwd(
                    dataclasses.replace(inputs, q=q, k=k, v=v), 64
                )

            def ref_fn(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return chunked_linear_attn_reference(
                    q * inputs.scale, k, v, decay, inputs.beta, C=64
                )

            def recurrent_ref(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return harness.reference(dataclasses.replace(inputs, q=q, k=k, v=v))

            record(
                hl_fn,
                {
                    "aten": recurrent_ref,
                    "aten.chunked": ref_fn,
                    "torch.compile": torch.compile(ref_fn, fullgraph=True),
                    "compile.recurrent": torch.compile(recurrent_ref, fullgraph=True),
                },
                (inputs.q, inputs.k, inputs.v),
                rtol=ACC_FWD_TOL,
                atol=0,
                output_dtypes={
                    "cute:helion": inputs.q.dtype,
                    "triton:helion": inputs.q.dtype,
                    "baselines:aten.chunked": inputs.q.dtype,
                    "baselines:torch.compile": inputs.q.dtype,
                },
                reference_note=(
                    "The recurrent reference and compile.recurrent return FP32; "
                    "Helion and chunked baselines return the input dtype."
                ),
            )
            record(
                hl_fn,
                ref_fn,
                (inputs.q, inputs.k, inputs.v),
                bwd=True,
                rtol=ACC_BWD_TOL,
                atol=0,
            )
            cases[-1].gradient_inputs = tuple(
                getattr(inputs, name) for name in harness.grad_tensors
            )
            cases[-1].gradient_names = harness.grad_tensors
            if module == "linear.example_kda":
                for varlen in [False, True]:
                    length = shape[2]
                    prepared = harness.make_inputs(
                        *shape,
                        fused_preamble=True,
                        varlen=varlen,
                        varlen_lengths=[length // 2, length // 4, length // 4],
                    )

                    def prepared_hl(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        prepared: Any = prepared,
                    ) -> torch.Tensor:
                        return harness.helion_fwd(
                            dataclasses.replace(prepared, q=q, k=k, v=v), 64
                        )

                    def prepared_ref(
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        prepared: Any = prepared,
                        varlen: bool = varlen,
                    ) -> torch.Tensor:
                        output = harness.reference(
                            dataclasses.replace(prepared, q=q, k=k, v=v)
                        )
                        return output.transpose(1, 2) if varlen else output

                    record(
                        prepared_hl,
                        prepared_ref,
                        (prepared.q, prepared.k, prepared.v),
                        rtol=0.03,
                        atol=0.05,
                        output_dtypes={
                            "cute:helion": prepared.q.dtype,
                            "triton:helion": prepared.q.dtype,
                        },
                        reference_note=(
                            "The recurrent reference and its compiled baseline "
                            "return FP32; Helion returns the input dtype."
                        ),
                    )
            if module in {"linear.example_kda", "linear.example_vanilla_linear_attn"}:
                from examples.linear.linear_attention_engine import recurrent_step
                from examples.linear.linear_attention_utils import (
                    recurrent_step_reference,
                )

                batch, heads, d, dv = [
                    (1, 4, 32, 32),
                    (8, 8, 64, 64),
                    (32, 32, 128, 128),
                ][i]
                q, k, v = (
                    rand(batch, heads, 1, d),
                    rand(batch, heads, 1, d),
                    rand(batch, heads, 1, dv),
                )
                state = rand(batch, heads, d, dv)
                correction = module == "linear.example_kda"
                alpha = torch.rand(
                    (batch, heads, 1, d) if correction else (batch, heads, 1),
                    device="cuda",
                )
                beta = (
                    torch.rand(batch, heads, 1, device="cuda") if correction else None
                )

                def recurrent(
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    state: torch.Tensor,
                    alpha: torch.Tensor,
                    beta: torch.Tensor | None,
                ) -> Any:
                    return recurrent_step(q, k, v, state.clone(), alpha, beta)

                record(
                    recurrent, recurrent_step_reference, (q, k, v, state, alpha, beta)
                )
        else:
            description = f"workload {i + 1}; see tensor metadata"
            if module == "concatenate":
                rows, a, b = [(256, 128, 256), (2048, 512, 768), (8192, 1024, 2048)][i]
                record(
                    {"simple": m.concat2d_dim1_simple, "masked": m.concat2d_dim1},
                    lambda x, y: torch.cat((x, y), dim=1),
                    (rand(rows, a), rand(rows, b)),
                )
            elif module == "cross_entropy":
                rows, cols = [(256, 1024), (1024, 16384), (2048, 65536)][i]
                x = rand(rows, cols)
                y = torch.randint(cols, (rows,), device="cuda")
                record(m.cross_entropy, F.cross_entropy, (x, y), rtol=1e-4, atol=1e-4)
            elif module == "embedding":
                rows, cols, batch, seq = [
                    (4096, 128, 16, 128),
                    (32768, 512, 32, 256),
                    (65536, 1024, 64, 512),
                ][i]
                x = torch.randint(rows, (batch, seq), device="cuda", dtype=torch.int32)
                record(m.embedding, F.embedding, (x, rand(rows, cols)), rtol=0, atol=0)
            elif module == "grouped_gemm":
                groups, rows, k, n = [
                    (4, 64, 128, 128),
                    (8, 128, 512, 512),
                    (16, 256, 1024, 1024),
                ][i]
                aa = [
                    rand(rows * (1 + j % 4), k, dtype=torch.bfloat16)
                    for j in range(groups)
                ]
                bb = [rand(k, n, dtype=torch.bfloat16) for j in range(groups)]
                packed_a, packed_b, offsets = m._pack_group_inputs(aa, bb)

                # Packing is input preparation; all providers consume the same packed layout.
                def reference(
                    a: torch.Tensor, b: torch.Tensor, o: torch.Tensor
                ) -> torch.Tensor:
                    return a @ b

                record(
                    {
                        "jagged": m.grouped_gemm_jagged,
                        "persistent": m.grouped_gemm_jagged_persistent,
                    },
                    reference,
                    (packed_a, packed_b, offsets),
                    rtol=1e-2,
                    atol=0.05,
                )
            elif module == "grpo_loss":
                batch, seq, vocab = [(1, 64, 1024), (2, 128, 4096), (4, 256, 16384)][i]
                logits = rand(batch, seq + 1, vocab, dtype=torch.bfloat16, grad=True)
                old = rand(batch, seq)
                ref = rand(batch, seq)
                ids = torch.randint(vocab, (batch, seq), device="cuda")
                adv = rand(batch)
                mask = torch.ones(batch, seq, device="cuda")
                args = (logits, old, ref, ids, adv, mask, 0.9, 0.2, 0.2, 0.4)
                # The example's own test compares BF16 kernels with an FP32
                # reference. BF16 log-softmax output rounding changes the loss.

                def reference(logits: torch.Tensor, *rest: Any) -> Any:
                    return m.torch_grpo_loss(logits.float(), *rest)

                reference_note = (
                    "Only reference logits are promoted from BF16 to FP32 before "
                    "log-softmax. Forward outputs are FP32 for both providers; "
                    "gradients return to the original BF16 logits. No output "
                    "dtype mismatch is permitted."
                )
                record(
                    m.helion_grpo_loss,
                    reference,
                    args,
                    rtol=0.03,
                    atol=0.03,
                    reference_note=reference_note,
                )
                record(
                    lambda *a: m.helion_grpo_loss(*a)[0],
                    lambda *a: reference(*a)[0],
                    args,
                    bwd=True,
                    rtol=0.03,
                    atol=0.03,
                    reference_note=reference_note,
                )
            elif module.startswith("jagged_"):
                rows, length, features = [(16, 32, 32), (64, 128, 64), (256, 256, 128)][
                    i
                ]
                if module == "jagged_dense_add":
                    x, o = m.random_jagged_2d(rows, length, device=testing.DEVICE)
                    record(
                        m.jagged_dense_add_2d,
                        m.jagged_dense_add_2d_reference,
                        (x, o, rand(rows, length)),
                    )
                elif module == "jagged_dense_bmm":
                    args = m.random_input(
                        D=features,
                        K=features,
                        batch_size=rows,
                        max_seq_len=length,
                        dtype=torch.float32,
                    )
                    record(m.jagged_dense_bmm, m.jagged_dense_bmm_reference, args)
                else:
                    lengths = torch.randint(1, length + 1, (rows,), device="cuda")
                    offsets = F.pad(lengths.cumsum(0), (1, 0))
                    nnz = int(offsets[-1])
                    x = rand(nnz, features)
                    if module == "jagged_layer_norm":
                        record(
                            m.jagged_layer_norm_kernel,
                            m.reference_jagged_layer_norm_pytorch,
                            (x, offsets, 1e-6),
                        )
                    elif module == "jagged_mean":
                        counts = torch.randint(
                            1, features + 1, (rows,), device="cuda", dtype=torch.int32
                        )
                        record(
                            m.jagged_mean_kernel,
                            m.reference_jagged_mean_kernel_pytorch,
                            (x, offsets, counts, features),
                        )
                    elif module == "jagged_softmax":
                        record(
                            m.jagged_softmax_kernel,
                            m.reference_jagged_softmax_pytorch,
                            (x, offsets),
                        )
                    elif module == "jagged_sum":
                        record(
                            m.jagged_sum_kernel,
                            m.reference_jagged_sum_kernel_pytorch,
                            (x, offsets),
                        )
                    elif module == "jagged_hstu_attn_2":
                        heads, d = 4, features
                        args = (
                            length,
                            1 / d**2,
                            1 / length,
                            rand(nnz, heads, d),
                            rand(nnz, heads, d),
                            rand(nnz, heads, d),
                            offsets.int(),
                        )
                        record(
                            m.jagged_hstu_attention,
                            m.reference_jagged_hstu_attention,
                            args,
                            rtol=1e-2,
                            atol=1e-2,
                        )
            elif module == "layer_norm":
                rows, cols = [(256, 1024), (4096, 4096), (8192, 8192)][i]
                x = rand(rows, cols, dtype=torch.bfloat16, grad=True)
                w = rand(cols, dtype=torch.bfloat16, grad=True)
                b = rand(cols, dtype=torch.bfloat16, grad=True)
                for mode in [False, True]:
                    record(
                        m.layer_norm,
                        F.layer_norm,
                        (x, [cols], w, b, 1e-4),
                        bwd=mode,
                        rtol=1e-2,
                        atol=1e-2,
                    )
            elif module == "rope":
                batch, qh, kh, seq, d = [
                    (1, 4, 2, 128, 64),
                    (2, 16, 4, 512, 128),
                    (4, 32, 8, 2048, 128),
                ][i]
                q = rand(batch, qh, seq, d, dtype=torch.bfloat16, grad=True)
                k = rand(batch, kh, seq, d, dtype=torch.bfloat16, grad=True)
                angles = rand(batch, seq, d, dtype=torch.bfloat16)
                for mode in [False, True]:
                    record(
                        m.rope,
                        m.rope_pytorch,
                        (q, k, angles.cos(), angles.sin()),
                        bwd=mode,
                        rtol=1e-2,
                        atol=1e-2,
                    )
            elif module == "segment_reduction":
                nodes, edges, features = [
                    (100, 2000, 128),
                    (1024, 32768, 128),
                    (4096, 262144, 256),
                ][i]
                indices = torch.randint(nodes, (edges,), device="cuda").sort()[0]
                record(
                    m.segmented_reduction_helion,
                    {
                        "aten": m.segmented_reduction_pytorch,
                        "handwritten-triton": m.segmented_reduction_triton,
                    },
                    (indices, rand(edges, features), nodes),
                )
            elif module == "low_mem_dropout":
                size = [262144, 4194304, 33554432][i]
                args = (0.25, rand(size), 123)
                record(
                    m.low_mem_dropout, lambda p, x, seed: F.dropout(x, p, True), args
                )
                record(
                    m.low_mem_dropout_bwd,
                    lambda p, x, seed: F.dropout(x, p, True),
                    args,
                )
                for case in cases:
                    case.check = "dropout"
            elif module in {"mamba2_chunk_scan", "mamba2_chunk_state"}:
                batch, heads, groups, seq, chunk, d, ds = [
                    (1, 4, 1, 256, 64, 32, 32),
                    (2, 8, 2, 1024, 64, 64, 64),
                    (2, 16, 4, 2048, 128, 64, 128),
                ][i]
                nchunks = seq // chunk
                x = rand(batch, seq, heads, d, dtype=torch.bfloat16)
                dt = torch.rand(
                    batch, heads, nchunks, chunk, device="cuda", dtype=x.dtype
                )
                da = -torch.rand_like(dt).cumsum(-1)
                b = rand(batch, seq, groups, ds, dtype=torch.bfloat16)
                if module == "mamba2_chunk_state":
                    record(
                        m.helion_mamba2_chunk_state_kernel,
                        m.ref_chunk_state,
                        (b, x, dt, da),
                    )
                else:
                    cb = rand(
                        batch, nchunks, groups, chunk, chunk, dtype=torch.bfloat16
                    )
                    prev = rand(batch, nchunks, heads, d, ds, dtype=torch.bfloat16)
                    record(
                        m.helion_mamba2_chunk_scan_kernel,
                        m.ref_chunk_scan,
                        (cb, x, dt, da, b, prev, rand(heads, dtype=torch.bfloat16)),
                    )
            elif module == "gdn_fwd_h":
                shape = [
                    (1, 4, 256, 64, 32, 32),
                    (1, 8, 512, 64, 64, 64),
                    (2, 8, 1024, 128, 64, 128),
                ][i]
                m.test(*shape)
            elif module == "nvfp4_gemm":
                # This example pins backend='cute' at decoration time. Explicitly
                # retarget its bodies for the Triton experiment, so a Triton result
                # can never accidentally be a measurement of a CuTe kernel.
                if os.environ.get("HELION_BACKEND") == "triton":
                    for kernel in [
                        m._nvfp4_matmul_single_pass_kernel,
                        m._nvfp4_w4a4_matmul_kernel,
                    ]:
                        kernel.settings.backend = "triton"
                        kernel.configs = []
                shape = [(128, 256, 128), (512, 1024, 512), (1024, 2048, 1024)][i]
                m.check_w4a16(*shape)
                m.check_w4a4(*shape)
                # nvfp4_scaled_matmul is an ATen wrapper, not a Helion kernel.
                cases[-1].baselines["aten.scaled_mm"] = m.nvfp4_scaled_matmul
            elif module == "nvfp4_gemv":
                rows, kbytes = [(256, 128), (2048, 512), (8192, 2048)][i]
                backend = os.environ.get("HELION_BACKEND", "triton")
                m.check_bf16in(rows, kbytes, backend)
                m.check_fp4in(rows, kbytes, backend)
            elif module == "aot_compile_example":
                rows, cols = [(256, 256), (1024, 1024), (4096, 4096)][i]
                record(
                    m.add_2d,
                    torch.add,
                    (
                        rand(rows, cols, dtype=torch.bfloat16),
                        rand(rows, cols, dtype=torch.bfloat16),
                    ),
                )
            elif module == "aot_example":
                rows, cols = [(256, 256), (1024, 1024), (4096, 4096)][i]
                x = rand(rows, cols, dtype=torch.bfloat16)
                record(m.vector_scale, operator.mul, (x.flatten(), 2.5))
                record(m.row_softmax, lambda x: x.softmax(-1), (x,))
                record(m.col_reduce_sum, lambda x: x.sum(0), (x,))
                record(
                    m.rms_norm_batched,
                    lambda x: F.rms_norm(x, [x.shape[-1]], eps=1e-5),
                    (x,),
                )
                record(
                    m.rms_norm_oneshot,
                    lambda x: F.rms_norm(x, [x.shape[-1]], eps=1e-5),
                    (x,),
                )
                record(
                    m.matmul_custom_key,
                    torch.matmul,
                    (x, rand(cols, rows, dtype=x.dtype)),
                )
                record(m.sum_aot, lambda x: x.sum(-1), (x,))
            else:
                raise ValueError(module)
        for case in cases:
            case.description = description
            if module in LINEAR:
                case.check = "relative_l2"
                case.rtol = ACC_BWD_TOL if case.mode == "backward" else ACC_FWD_TOL
                case.atol = 0
        return cases
