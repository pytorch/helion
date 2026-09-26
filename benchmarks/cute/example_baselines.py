# ruff: noqa: ANN401
"""Whole-call baseline adapters and shared correctness/timing for examples."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import statistics
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    import torch


def _event_ms(fn: Callable) -> float:
    from triton.testing import do_bench

    elapsed = do_bench(fn, warmup=10, rep=50, return_mode="median")
    assert isinstance(elapsed, float), "median event timing must be a scalar"
    return elapsed


def metadata(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, (tuple, list)):
        return [metadata(v) for v in value]
    if isinstance(value, dict):
        return {k: metadata(v) for k, v in value.items()}
    return value


def captured_metadata(value: Any, depth: int = 5) -> Any:
    """Describe inputs closed over by the example's benchmark wrappers."""
    import torch

    if isinstance(value, torch.Tensor):
        return metadata(value)
    if depth <= 0:
        return None
    if isinstance(value, functools.partial):
        value = {"args": value.args, "keywords": value.keywords, "function": value.func}
    elif inspect.isfunction(value):
        value = {
            **inspect.getclosurevars(value).nonlocals,
            "defaults": value.__defaults__,
            "keyword_defaults": value.__kwdefaults__,
        }
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): item
            for key, val in value.items()
            if (item := captured_metadata(val, depth - 1)) is not None
        }
    if isinstance(value, (tuple, list)):
        return [captured_metadata(v, depth - 1) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


@dataclasses.dataclass
class PreparedCandidate:
    fn: Callable
    details: dict


def active_kernel_configs() -> list[dict]:
    import helion

    found = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("examples.") or module is None:
            continue
        for obj in list(vars(module).values()):
            if isinstance(obj, helion.Kernel) and obj._bound_kernels:
                identity = f"{obj.fn.__module__}.{obj.fn.__name__}"
                found[identity] = {
                    "kernel": identity,
                    "backend": obj.settings.backend,
                    "configs": [
                        bound._config.to_json() if bound._config is not None else None
                        for bound in obj._bound_kernels.values()
                    ],
                }
    return list(found.values())


def baseline_factories(case: Any) -> dict[str, Callable]:
    import torch
    import torch.nn.functional as F

    baseline = next(iter(case.baselines.values()))
    result: dict[str, Callable] = {
        name: (lambda fn=fn: fn) for name, fn in case.baselines.items()
    }
    if not any("compile" in n for n in result):
        result["torch.compile"] = lambda: torch.compile(baseline, fullgraph=True)
    module = case.module
    forward = case.mode == "forward"
    if forward and module in {"add", "exp", "aot_compile_example"}:

        def manual_cute() -> PreparedCandidate:
            from benchmarks.cute.cute_pointwise_manual import make_manual_fn

            operation = "exp" if module == "exp" else "add"
            call, out, config, seconds = make_manual_fn(
                operation, case.args, sweep=True
            )

            def invoke(*inputs: Any) -> torch.Tensor:
                call()
                return out

            return PreparedCandidate(
                invoke, {"config": config, "tune_seconds": seconds}
            )

        result["handwritten-cute"] = manual_cute
    if module in {"softmax", "acfs.softmax_acf", "batch_softmax"} or (
        module == "aot_example" and case.name.startswith("01_")
    ):

        def quack_softmax() -> Callable:
            softmax = importlib.import_module("quack.softmax").softmax

            return lambda x: softmax(x.reshape(-1, x.shape[-1])).reshape_as(x)

        result["quack"] = quack_softmax
        if forward:

            def quack_softmax_direct() -> Callable:
                torch2cute_dtype_map = importlib.import_module(
                    "quack.cute_dsl_utils"
                ).torch2cute_dtype_map
                Softmax = importlib.import_module("quack.softmax").Softmax

                x = case.args[0]
                out = torch.empty_like(x)
                dtype = torch2cute_dtype_map[x.dtype]
                kernel = Softmax.compile(dtype, dtype, x.shape[-1])

                def call(x: torch.Tensor) -> torch.Tensor:
                    kernel(x.reshape(-1, x.shape[-1]), out.reshape(-1, out.shape[-1]))
                    return out

                return call

            result["quack-direct"] = quack_softmax_direct
    if module == "rms_norm":

        def quack_rmsnorm() -> Callable:
            rmsnorm = importlib.import_module("quack.rmsnorm").rmsnorm

            return lambda x, w, eps: rmsnorm(x, w, eps=eps)

        result["quack"] = quack_rmsnorm
        result["aten.native_rms_norm"] = lambda: (
            lambda x, w, eps: F.rms_norm(x, [x.shape[-1]], w, eps)
        )
    if module == "aot_example" and case.name[:2] in {"03", "04"}:

        def quack_rms_no_weight() -> Callable:
            rmsnorm = importlib.import_module("quack.rmsnorm").rmsnorm

            return lambda x: rmsnorm(x, eps=1e-5)

        result["quack"] = quack_rms_no_weight
    if module == "aot_example" and case.name.startswith("05_"):

        def quack_aot_gemm() -> Callable:
            return importlib.import_module("quack.gemm_interface").gemm

        result["quack"] = quack_aot_gemm
    if module in {"rms_norm", "layer_norm"} and forward:

        def quack_norm_direct() -> Callable:
            torch2cute_dtype_map = importlib.import_module(
                "quack.cute_dsl_utils"
            ).torch2cute_dtype_map
            _compile_rmsnorm_fwd = importlib.import_module(
                "quack.rmsnorm"
            )._compile_rmsnorm_fwd

            layer = module == "layer_norm"
            x = case.args[0]
            w = case.args[2] if layer else case.args[1]
            b = case.args[3] if layer else None
            dt, wdt = torch2cute_dtype_map[x.dtype], torch2cute_dtype_map[w.dtype]
            bdt = torch2cute_dtype_map[b.dtype] if b is not None else None
            kernel = _compile_rmsnorm_fwd(
                dt, dt, None, wdt, bdt, None, x.shape[-1], False, False, layer, False
            )
            out = torch.empty_like(x)

            def call(*a: Any) -> torch.Tensor:
                kernel(
                    a[0],
                    a[2] if layer else a[1],
                    a[3] if layer else None,
                    None,
                    out,
                    None,
                    None,
                    None,
                    a[-1],
                )
                return out

            return call

        result["quack-direct"] = quack_norm_direct

        def quack_norm_tuned() -> Callable:
            rmsnorm_fwd_tuned = importlib.import_module(
                "quack.rmsnorm"
            ).rmsnorm_fwd_tuned

            layer = module == "layer_norm"
            out = torch.empty_like(case.args[0])

            def call(*a: Any) -> torch.Tensor:
                rmsnorm_fwd_tuned(
                    a[0],
                    a[2] if layer else a[1],
                    out,
                    bias=a[3] if layer else None,
                    eps=a[-1],
                    is_layernorm=layer,
                )
                return out

            return call

        result["quack-tuned"] = quack_norm_tuned
    if module == "layer_norm" and not forward:

        def quack_layernorm_autograd() -> Callable:
            layernorm_bwd = importlib.import_module("quack.rmsnorm").layernorm_bwd
            layernorm_fwd = importlib.import_module("quack.rmsnorm").layernorm_fwd

            class QuackLayerNorm(torch.autograd.Function):
                @staticmethod
                def forward(
                    ctx: Any,
                    x: torch.Tensor,
                    w: torch.Tensor,
                    b: torch.Tensor,
                    eps: float,
                ) -> torch.Tensor:
                    out, rstd, mean = layernorm_fwd(
                        x, w, b, eps, return_rstd=True, return_mean=True
                    )
                    ctx.save_for_backward(x, w, rstd, mean)
                    return out

                @staticmethod
                def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
                    (grad_output,) = grad_outputs
                    x, w, rstd, mean = ctx.saved_tensors
                    dx, dw, db = layernorm_bwd(x, w, grad_output, rstd, mean)
                    return dx, dw, db, None

            return lambda x, normalized_shape, w, b, eps: QuackLayerNorm.apply(
                x, w.float(), b.float(), eps
            )

        result["quack"] = quack_layernorm_autograd
    if module == "cross_entropy":

        def quack_ce() -> Callable:
            return importlib.import_module("quack.cross_entropy").cross_entropy

        result["quack"] = quack_ce

        def quack_ce_direct(tuned: bool) -> PreparedCandidate:
            from benchmarks.cute.compare_cross_entropy_backends import (
                _quack_compiled_kernel,
            )
            from benchmarks.cute.compare_cross_entropy_backends import (
                _quack_config_space,
            )

            x, target = case.args
            expected = F.cross_entropy(x, target)
            start = time.monotonic()
            best_ms = float("inf")
            best_fn = None
            best_config = None
            tested, failed = 0, 0
            configs: list[dict[str, Any] | None] = [None]
            if tuned:
                configs.extend(_quack_config_space(x.shape[-1], x.element_size() * 8))
            for config in configs:
                try:
                    call, losses = _quack_compiled_kernel(x, target, config)

                    def invoke(
                        *inputs: Any,
                        call: Callable = call,
                        losses: torch.Tensor = losses,
                    ) -> torch.Tensor:
                        call()
                        return losses.mean()

                    torch.testing.assert_close(
                        invoke(), expected, rtol=case.rtol, atol=case.atol
                    )
                    ms = _event_ms(invoke)
                    tested += 1
                    if ms < best_ms:
                        best_ms, best_fn, best_config = ms, invoke, config
                except Exception:
                    failed += 1
                    if not tuned:
                        raise
            assert best_fn is not None, "No correct Quack cross-entropy configuration"
            return PreparedCandidate(
                best_fn,
                {
                    "config": best_config,
                    "tested": tested,
                    "failed": failed,
                    "tune_seconds": time.monotonic() - start,
                },
            )

        result["quack-direct"] = lambda: quack_ce_direct(False)
        result["quack-tuned"] = lambda: quack_ce_direct(True)
    if module == "welford":

        def quack_welford() -> Callable:
            layernorm_fwd = importlib.import_module("quack.rmsnorm").layernorm_fwd

            return lambda weight, bias, x: layernorm_fwd(x, weight, bias, eps=1e-5)

        result["quack"] = quack_welford
    if (
        module
        in {"matmul", "bmm", "broadcast_matmul", "matmul_split_k", "split_k_barrier"}
        and forward
    ):
        # The first fixture in these examples is the unadorned product.
        if case.name.startswith("00_"):

            def quack_gemm() -> Callable:
                gemm = importlib.import_module("quack.gemm_interface").gemm

                if module == "broadcast_matmul":
                    return lambda a, b: gemm(a.reshape(-1, a.shape[-1]), b).reshape(
                        a.shape[0], a.shape[1], b.shape[-1]
                    )
                return lambda a, b: gemm(a, b)

            result["quack"] = quack_gemm
    if module == "matmul" and forward and case.name[:2] in {"01", "02"}:

        def quack_bias_gemm() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm

            if case.name.startswith("01_"):
                return lambda bias, a, b: gemm(
                    a, b, bias=bias.expand(b.shape[-1]).contiguous()
                )
            return lambda a, b, bias: gemm(a, b, bias=bias)

        result["quack"] = quack_bias_gemm
    if module == "matmul" and forward and case.name.startswith("03_"):

        def quack_relu_gemm() -> Callable:
            gemm_act = importlib.import_module("quack.gemm_interface").gemm_act

            bias = inspect.getclosurevars(baseline).nonlocals["bias"]
            return lambda a, b: gemm_act(
                a, b, bias=bias, activation="relu", store_preact=False
            )[1]

        result["quack"] = quack_relu_gemm
    if module == "epilogue_subtiling":

        def quack_gelu_epilogue() -> Callable:
            gemm_act = importlib.import_module("quack.gemm_interface").gemm_act

            if case.name.startswith("00_"):
                return lambda x, w, bias, residual: gemm_act(
                    x,
                    w,
                    C=residual.float() * 0.5,
                    bias=bias,
                    activation="gelu_erf",
                    alpha=1.25,
                    store_preact=False,
                )[1]

            def auxiliary(
                x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor
            ) -> tuple:
                pre, out = gemm_act(
                    x,
                    w,
                    bias=bias,
                    activation="gelu_erf",
                    alpha=1.25,
                    store_preact=True,
                )
                return out, pre

            return auxiliary

        result["quack"] = quack_gelu_epilogue
        result["quack.compile"] = lambda: torch.compile(
            quack_gelu_epilogue(), fullgraph=True
        )
    if module == "matmul_layernorm":

        def quack_matmul_layernorm() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm
            layernorm_fwd = importlib.import_module("quack.rmsnorm").layernorm_fwd

            return lambda x, y, w, b: layernorm_fwd(
                gemm(x, y), w.float(), b.float(), eps=1e-5
            )

        result["quack"] = quack_matmul_layernorm
        result["quack.compile"] = lambda: torch.compile(
            quack_matmul_layernorm(), fullgraph=True
        )
    if module in {"se_block", "squeeze_and_excitation_net"} and forward:

        def quack_se() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm
            gemm_act = importlib.import_module("quack.gemm_interface").gemm_act

            if module == "se_block":
                return lambda x, w: 2 * x * gemm(x, w).sigmoid()

            def network(
                x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
            ) -> torch.Tensor:
                hidden = gemm_act(x, a, activation="relu", store_preact=False)[1]
                gate = gemm(hidden, b).sigmoid()
                return x * gate

            return network

        result["quack"] = quack_se
        result["quack.compile"] = lambda: torch.compile(quack_se(), fullgraph=True)
    if module == "fp8_gemm":

        def quack_fp8_gemm() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm

            from helion._testing import HALF_DTYPE

            return lambda a, b: gemm(a, b, out_dtype=HALF_DTYPE)

        result["quack"] = quack_fp8_gemm
    if module == "gather_gemv":

        def vectorized_gather(
            w: torch.Tensor, index: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            return torch.matmul(w[index].to(x.dtype), x)

        def gemv_then_gather(
            w: torch.Tensor, index: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            return torch.matmul(w.to(x.dtype), x)[index]

        def quack_gemv() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm

            return lambda w, index, x: gemm(
                x.reshape(1, -1), w.reshape(-1, x.numel()).T
            ).reshape(w.shape[:2])[index]

        result["aten.vectorized"] = lambda: vectorized_gather
        result["compile.vectorized"] = lambda: torch.compile(
            vectorized_gather, fullgraph=True
        )
        result["aten.gemv_then_gather"] = lambda: gemv_then_gather
        result["compile.gemv_then_gather"] = lambda: torch.compile(
            gemv_then_gather, fullgraph=True
        )
        result["quack.gemv_then_gather"] = quack_gemv
    if module == "fp8_matmul":

        def quack_rowwise_fp8_gemm() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm

            scales = inspect.getclosurevars(baseline).nonlocals
            return lambda a, b: (
                gemm(a, b, out_dtype=torch.float32)
                * scales["scale_a"]
                * scales["scale_b"]
            ).to(torch.bfloat16)

        result["quack+scales"] = quack_rowwise_fp8_gemm
    if module in {"nvfp4_gemm", "nvfp4_gemv"} and case.name.startswith("01_"):

        def quack_nvfp4() -> Callable:
            BlockScaledOperand = importlib.import_module(
                "quack.blockscaled.operand"
            ).BlockScaledOperand
            gemm = importlib.import_module("quack.gemm_interface").gemm

            def operand(data: torch.Tensor, scale: torch.Tensor) -> Any:
                rows, kbytes = data.shape
                blocked = scale.reshape(
                    (rows + 127) // 128, (kbytes // 8 + 3) // 4, 32, 4, 4
                )
                return BlockScaledOperand.from_parts(data, blocked, "nvfp4")

            if module == "nvfp4_gemm":

                def gemm_call(
                    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
                ) -> torch.Tensor:
                    # The example supplies a row-major packed RHS. Its required
                    # conversion to K-contiguous storage is inside the timer.
                    return gemm(
                        operand(a, sa),
                        operand(b.T.contiguous(), sb).mT,
                        out_dtype=torch.bfloat16,
                    )

                return gemm_call

            def gemv_call(
                w: torch.Tensor, x: torch.Tensor, sw: torch.Tensor, sx: torch.Tensor
            ) -> torch.Tensor:
                return gemm(
                    operand(x.reshape(1, -1), sx),
                    operand(w, sw).mT,
                    out_dtype=torch.bfloat16,
                ).reshape(-1)

            return gemv_call

        result["quack.nvfp4"] = quack_nvfp4
        if module == "nvfp4_gemv":
            result["aten.scaled_mm"] = lambda: (
                lambda w, x, sw, sx: torch._scaled_mm(
                    x.reshape(1, -1),
                    w.T,
                    sx.reshape(-1),
                    sw.reshape(-1),
                    out_dtype=torch.bfloat16,
                ).reshape(-1)
            )
    if module == "grouped_gemm":

        def quack_grouped() -> Callable:
            gemm = importlib.import_module("quack.gemm_interface").gemm

            return lambda a, b, offsets: gemm(a, b)

        result["quack"] = quack_grouped
        result["aten.grouped_mm"] = lambda: (
            lambda a, b, o: torch._grouped_mm(
                a, b.unsqueeze(0).expand(o.numel() - 1, -1, -1), offs=o[1:].int()
            )
        )
    if module == "moe_matmul_ogs":

        def moe_provider(use_quack: bool) -> Callable:
            kernel_args = inspect.getclosurevars(
                next(iter(case.kernels.values()))
            ).nonlocals["helion_kernel_args"]
            a, weights, counts, offsets, index, maximum = kernel_args
            if use_quack:
                gemm = importlib.import_module("quack.gemm_interface").gemm

                def product() -> torch.Tensor:
                    return gemm(a, weights, cu_seqlens_m=offsets, A_idx=index)
            else:

                def product() -> torch.Tensor:
                    return torch._grouped_mm(a[index], weights, offs=offsets[1:])

            def call() -> torch.Tensor:
                packed = product()
                out = torch.empty_like(packed)
                out.index_copy_(0, index.long(), packed)
                return out

            return call

        result["quack.gather_gemm"] = lambda: moe_provider(True)
        result["aten.grouped_mm"] = lambda: moe_provider(False)
    if module == "jagged_dense_bmm":

        def grouped_jagged(use_quack: bool) -> Callable:
            if use_quack:
                gemm = importlib.import_module("quack.gemm_interface").gemm

                return lambda offsets, x, weights, bias: gemm(
                    x, weights, bias=bias, cu_seqlens_m=offsets.int()
                )

            def call(
                offsets: torch.Tensor,
                x: torch.Tensor,
                weights: torch.Tensor,
                bias: torch.Tensor,
            ) -> torch.Tensor:
                rows = torch.repeat_interleave(
                    torch.arange(weights.shape[0], device=x.device),
                    offsets[1:] - offsets[:-1],
                    output_size=x.shape[0],
                )
                return (
                    torch._grouped_mm(x, weights, offs=offsets[1:].int()) + bias[rows]
                )

            return call

        result["quack.grouped_mm"] = lambda: grouped_jagged(True)
        result["aten.grouped_mm"] = lambda: grouped_jagged(False)
    if module in {
        "jagged_sum",
        "jagged_mean",
        "jagged_softmax",
        "jagged_layer_norm",
        "jagged_dense_add",
    }:

        def segment_reference(*a: Any) -> torch.Tensor:
            x, offsets = a[:2]
            lengths = offsets[1:] - offsets[:-1]
            if module == "jagged_sum":
                return torch.segment_reduce(x, "sum", offsets=offsets, unsafe=True)
            if module == "jagged_mean":
                mean = torch.segment_reduce(x, "mean", offsets=offsets, unsafe=True)
                return torch.where(
                    torch.arange(x.shape[1], device=x.device)[None, :] < a[2][:, None],
                    mean,
                    0,
                )
            rows = torch.repeat_interleave(
                torch.arange(lengths.numel(), device=x.device),
                lengths,
                output_size=x.shape[0],
            )
            if module == "jagged_softmax":
                maxima = torch.segment_reduce(x, "max", offsets=offsets, unsafe=True)
                values = (x - maxima[rows]).exp()
                sums = torch.segment_reduce(values, "sum", offsets=offsets, unsafe=True)
                return values / sums[rows]
            if module == "jagged_layer_norm":
                means = (
                    torch.segment_reduce(
                        x.sum(-1), "mean", offsets=offsets, unsafe=True
                    )
                    / x.shape[1]
                )
                moments = (
                    torch.segment_reduce(
                        (x * x).sum(-1), "mean", offsets=offsets, unsafe=True
                    )
                    / x.shape[1]
                )
                return (x - means[rows, None]) * torch.rsqrt(
                    moments - means.square() + a[2]
                )[rows, None]
            cols = torch.arange(x.shape[0], device=x.device) - offsets[rows]
            out = a[2].clone()
            out[rows, cols] += x
            return out

        result["aten.vectorized"] = lambda: segment_reference
        result["compile.vectorized"] = lambda: torch.compile(
            segment_reference, fullgraph=True
        )
    if module in {"jagged_hstu_attn", "jagged_hstu_attn_2"}:

        def vectorized_hstu(*args: Any) -> torch.Tensor:
            if module == "jagged_hstu_attn":
                q, k, v, offsets, targets, length = args
                alpha, scale = 1 / v.shape[-1] ** 2, 1 / length
            else:
                length, alpha, scale, q, k, v, offsets = args
            lengths = offsets[1:] - offsets[:-1]
            rows = torch.repeat_interleave(
                torch.arange(lengths.numel(), device=q.device),
                lengths,
                output_size=q.shape[0],
            )
            cols = torch.arange(q.shape[0], device=q.device) - offsets[rows]

            def padded(x: torch.Tensor) -> torch.Tensor:
                out = x.new_zeros(lengths.numel(), length, *x.shape[1:])
                out[rows, cols] = x
                return out.transpose(1, 2)

            pq, pk, pv = padded(q), padded(k), padded(v)
            scores = pq @ pk.transpose(-1, -2) * alpha
            scores = (
                scores / (1 + torch.exp(-scores))
                if module == "jagged_hstu_attn"
                else F.silu(scores)
            ) * scale
            indices = torch.arange(length, device=q.device)
            causal = (
                indices[:, None] > indices[None, :]
                if module == "jagged_hstu_attn"
                else indices[:, None] >= indices[None, :]
            )
            scores = torch.where(causal, scores, 0)
            out = (scores.to(v.dtype) @ pv).transpose(1, 2)
            return out[rows, cols]

        result["aten.vectorized"] = lambda: vectorized_hstu
        result["compile.vectorized"] = lambda: torch.compile(
            vectorized_hstu, fullgraph=True
        )
    if (
        module in {"attention", "blackwell_attention", "flex_attention"}
        and forward
        and case.name.startswith("00_")
    ):
        result["aten.sdpa"] = lambda: F.scaled_dot_product_attention

        def tilegym_attention() -> Callable:
            from benchmarks.cute.tilegym_attention import fmha_variant_triton

            return lambda q, k, v: fmha_variant_triton(q, k, v, is_causal=False)

        result["tilegym-triton"] = tilegym_attention
    if module in {"attention", "blackwell_attention", "flex_attention"}:

        def selected_sdpa(kind: str) -> Callable:
            from torch.nn.attention import SDPBackend
            from torch.nn.attention import sdpa_kernel

            selected = {
                "flash": SDPBackend.FLASH_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
            }[kind]
            causal = module == "attention" and case.name[:2] in {"02", "04"}
            relu = module == "attention" and case.name[:2] in {"03", "04"}

            def call(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                bias: torch.Tensor | None = None,
            ) -> torch.Tensor:
                with sdpa_kernel(selected):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=causal
                    )
                return out.relu() if relu else out

            return call

        for kind in ["flash", "cudnn", "efficient"]:
            result[f"aten.sdpa.{kind}"] = lambda kind=kind: selected_sdpa(kind)
    if module == "fp8_attention":

        def fp8_input_sdpa() -> Callable:
            from helion._testing import HALF_DTYPE

            captured = inspect.getclosurevars(baseline).nonlocals
            q, k, v = [captured[n] for n in ["q_fp8", "k_fp8", "v_fp8"]]
            batch, heads = captured["batch"], captured["heads"]

            def call() -> torch.Tensor:
                out = F.scaled_dot_product_attention(
                    q.to(HALF_DTYPE).reshape(batch, heads, *q.shape[-2:]),
                    k.to(HALF_DTYPE).reshape(batch, heads, *k.shape[-2:]),
                    v.transpose(-1, -2)
                    .to(HALF_DTYPE)
                    .reshape(batch, heads, q.shape[-2], v.shape[-2]),
                )
                return out.to(torch.float8_e4m3fn)

            return call

        result["aten.sdpa.fp16"] = fp8_input_sdpa
    if module == "int4_gemm" and forward:

        def quack_int4_gemm() -> Callable:
            from examples.int4_gemm import _unpack_int4_matrix

            gemm = importlib.import_module("quack.gemm_interface").gemm

            def unpack_then_gemm(a: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
                decoded = _unpack_int4_matrix(packed).to(a.dtype)
                return gemm(a, decoded, out_dtype=a.dtype, split_k=None)

            return unpack_then_gemm

        result["quack"] = quack_int4_gemm
        result["quack.compile"] = lambda: torch.compile(
            quack_int4_gemm(), fullgraph=True
        )
    if module == "matmul_split_k" and forward:
        if case.name.startswith("01_"):

            def quack_split_k_bias() -> Callable:
                gemm = importlib.import_module("quack.gemm_interface").gemm

                bias = inspect.getclosurevars(baseline).nonlocals["bias"]
                return lambda a, b: gemm(a, b, bias=bias)

            result["quack"] = quack_split_k_bias
            result["quack.compile"] = lambda: torch.compile(
                quack_split_k_bias(), fullgraph=True
            )
        if case.name[:2] in {"00", "01"}:

            def quack_auto_split_k() -> Callable:
                gemm = importlib.import_module("quack.gemm_interface").gemm

                bias = (
                    inspect.getclosurevars(baseline).nonlocals["bias"]
                    if case.name.startswith("01_")
                    else None
                )
                return lambda a, b: gemm(a, b, bias=bias, split_k=None)

            result["quack.split_k"] = quack_auto_split_k
            result["quack.split_k.compile"] = lambda: torch.compile(
                quack_auto_split_k(), fullgraph=True
            )
    return result


def make_callable(fn: Callable, case: Any) -> Callable:
    import torch
    from torch.utils._pytree import tree_leaves

    if case.mode == "forward":
        return functools.partial(fn, *case.args)
    tensors = [
        v
        for v in tree_leaves(case.args)
        if isinstance(v, torch.Tensor) and v.requires_grad
    ]
    if case.gradient_inputs is not None:
        tensors = list(case.gradient_inputs)
    outputs = fn(*case.args)
    outputs = [
        v
        for v in tree_leaves(outputs)
        if isinstance(v, torch.Tensor) and v.requires_grad
    ]
    # Same logical upstream gradients, independent of output strides/dtype.
    # randn_like preserves layout, so equal seeds alone do not give equal
    # gradients when attention providers return different physical layouts.
    torch.manual_seed(73453)
    gradients = [
        torch.randn(v.shape, device=v.device, dtype=torch.float32).to(v.dtype)
        for v in outputs
    ]
    return lambda: torch.autograd.grad(
        outputs, tensors, gradients, retain_graph=True, allow_unused=True
    )


def output_contract(expected: Any, *, output_dtype: torch.dtype | None = None) -> Any:
    import torch
    from torch.utils._pytree import tree_map_only

    if output_dtype is not None:
        assert isinstance(expected, torch.Tensor), "Dtype override needs one tensor"
    return tree_map_only(
        torch.Tensor,
        lambda value: {
            "shape": list(value.shape),
            "dtype": str(value.dtype if output_dtype is None else output_dtype),
        },
        expected,
    )


def assert_correct(
    actual: Any,
    expected: Any,
    case: Any,
    *,
    output_dtype: torch.dtype | None = None,
) -> None:
    import torch
    from torch.utils._pytree import tree_flatten

    if output_dtype is not None:
        assert isinstance(expected, torch.Tensor), "Dtype override needs one tensor"
    aa, sa = tree_flatten(actual)
    bb, sb = tree_flatten(expected)
    assert sa == sb, (sa, sb)
    for a, b in zip(aa, bb, strict=True):
        if a is None or b is None:
            assert a is b
        elif isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), (
                "Output tensor structure changed"
            )
            assert a.shape == b.shape, (a.shape, b.shape)
            dtype = b.dtype if output_dtype is None else output_dtype
            assert a.dtype == dtype, (a.dtype, dtype)
            if case.check == "dropout":
                continue
            if case.check == "relative_l2":
                relative_error = (
                    a.float() - b.float()
                ).norm().item() / b.float().norm().clamp(min=1e-8).item()
                assert relative_error < case.rtol, (
                    f"Relative L2 error {relative_error} exceeds {case.rtol}"
                )
            else:
                torch.testing.assert_close(
                    a.float(), b.float(), rtol=case.rtol, atol=case.atol
                )
        else:
            assert a == b
    if case.check == "dropout":
        p, x, seed = case.args
        keep = actual != 0
        fraction = float(keep.float().mean())
        assert abs(fraction - (1 - p)) < 0.02, fraction
        torch.testing.assert_close(actual[keep], (x / (1 - p))[keep])


def measure(
    fn: Callable,
    repetitions: int,
    expected: Any,
    case: Any,
    *,
    output_dtype: torch.dtype | None = None,
) -> dict:
    import torch

    from helion.autotuner.benchmarking import _make_cudagraph_replay

    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    runs = [_event_ms(fn) for _ in range(repetitions)]
    result: dict[str, Any] = {
        "event_ms": statistics.median(runs),
        "event_runs_ms": runs,
    }
    try:
        # The forward graph of backward-only workloads is intentionally built
        # before capture. PyTorch can redirect its saved streams into capture.
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            replay = _make_cudagraph_replay(fn)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        actual = replay()
        result["graph_output"] = metadata(actual)
        assert_correct(actual, expected, case, output_dtype=output_dtype)
        graph_runs = [_event_ms(replay) for _ in range(repetitions)]
        actual = replay()
        result["graph_post_measurement_output"] = metadata(actual)
        assert_correct(actual, expected, case, output_dtype=output_dtype)
        result.update(
            graph_ms=statistics.median(graph_runs),
            graph_runs_ms=graph_runs,
            graph_correctness="pass",
        )
    except Exception as error:
        result["graph_error"] = f"{type(error).__name__}: {error}"
    return result
