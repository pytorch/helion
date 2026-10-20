# ruff: noqa: ANN401
"""Whole-call baseline adapters and shared correctness/timing for examples."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from pathlib import Path
import statistics
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    import torch

# The repository root's quack/ checkout directory has no __init__.py. With the
# root on sys.path it shadows the installed package as a namespace package, and
# quack's JIT cache then fails on quack.__file__. Like the dedicated
# comparisons, import quack from the checkout itself when it exists.
QUACK_PATH = Path(__file__).resolve().parents[2] / "quack"


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


def live_example_kernels() -> dict[str, Any]:
    """Every live Helion kernel whose function is defined in an example module.

    Kernels are found through the garbage collector, not module globals: some
    examples build theirs inside cached factory functions (nvfp4_gemv) and
    those never appear in ``vars(module)``.
    """
    import gc

    import helion

    # Some examples build their kernels in zero-argument cached factories
    # (``@functools.cache def _foo_kernel() -> helion.Kernel``); construct
    # those first so they can be pinned before the first call.
    for name, module in list(sys.modules.items()):
        if not name.startswith("examples.") or module is None:
            continue
        for obj in list(vars(module).values()):
            wrapped = getattr(obj, "__wrapped__", None)
            if (
                isinstance(obj, functools._lru_cache_wrapper)
                and callable(wrapped)
                and "Kernel" in str(inspect.signature(wrapped).return_annotation)
                and not inspect.signature(wrapped).parameters
            ):
                obj()
    # Keyed by the function's qualified name; a second kernel built from the
    # same function (examples/attention.py builds attention_dynamic from
    # attention.fn) is kept under ``name@2`` instead of silently replacing
    # the first in whatever order the collector lists them.
    found: dict[str, Any] = {}
    for obj in gc.get_objects():
        if isinstance(obj, helion.Kernel) and obj.fn.__module__.startswith("examples."):
            base = f"{obj.fn.__module__}.{obj.fn.__name__}"
            key = base
            index = 1
            while key in found and found[key] is not obj:
                index += 1
                key = f"{base}@{index}"
            found[key] = obj
    return found


def active_kernel_configs() -> list[dict]:
    found = {}
    for identity, obj in live_example_kernels().items():
        if obj._bound_kernels:
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

    if QUACK_PATH.is_dir() and str(QUACK_PATH) not in sys.path:
        sys.path.insert(0, str(QUACK_PATH))
    baseline = next(iter(case.baselines.values()))
    result: dict[str, Callable] = {
        name: (lambda fn=fn: fn) for name, fn in case.baselines.items()
    }
    if case.auto_compile and not any("compile" in n for n in result):
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

            def call(
                a: torch.Tensor,
                b: torch.Tensor,
                scale_a: torch.Tensor | None = None,
                scale_b: torch.Tensor | None = None,
            ) -> torch.Tensor:
                # The second case passes tensor-wise scales: alpha folds them
                # into quack's epilogue.
                alpha = 1.0 if scale_a is None or scale_b is None else scale_a * scale_b
                return gemm(a, b, alpha=alpha, out_dtype=HALF_DTYPE)

            return call

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
        # FP32 _grouped_mm copies offsets to the host, which CUDA graphs reject.
        # A padded batched GEMM is a capture-safe ATen formulation.
        offsets = case.args[0]
        longest = int((offsets[1:] - offsets[:-1]).max())

        def padded_bmm(
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
            cols = torch.arange(x.shape[0], device=x.device) - offsets[rows]
            padded = x.new_zeros(weights.shape[0], longest, x.shape[1])
            padded[rows, cols] = x
            return torch.baddbmm(bias[:, None, :], padded, weights)[rows, cols]

        result["aten.padded_bmm"] = lambda: padded_bmm
        result["compile.padded_bmm"] = lambda: torch.compile(padded_bmm, fullgraph=True)
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
                from helion._testing import assert_relative_l2_close

                assert_relative_l2_close(a, b, case.rtol)
            elif case.check == "mismatch_budget":
                from helion._testing import assert_close_with_mismatch_tolerance

                assert_close_with_mismatch_tolerance(
                    a.float(),
                    b.float(),
                    rtol=case.rtol,
                    atol=case.atol,
                    max_mismatch_pct=case.max_mismatch_pct,
                    max_mismatched_abs_diff=case.max_mismatched_abs_diff,
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


def graph_profile(replay: Callable, calls: int = 5) -> list[str]:
    """Names of the GPU activities (kernels, memsets, memcpys) one replay of
    a captured graph performs, in launch order; a node that a baseline has
    and we do not (or vice versa) shifts the single-replay graph time by a
    whole launch-latency quantum."""
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            replay()
        torch.cuda.synchronize()
    events = sorted(
        (
            e
            for e in prof.events() or []
            if e.device_type == torch.autograd.DeviceType.CUDA
        ),
        key=lambda e: e.time_range.start,
    )
    per_replay = len(events) // calls if calls else len(events)
    names = [
        f"{e.time_range.elapsed_us():.2f}us {e.name[:100]}" for e in events[:per_replay]
    ]
    for name in names:
        print(f"graph node {name}", file=sys.stderr)
    return names


def kernel_profile(fn: Callable, calls: int) -> dict:
    """Median per-call sum of GPU kernel durations (us) over ``calls`` eager
    calls under torch.profiler, plus a per-kernel breakdown. Measures device
    time only, so launch latency and CPU overhead drop out."""
    import torch
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    kernels = [
        e
        for e in prof.events() or []
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith("Memcpy")
        and not e.name.startswith("Memset")
    ]
    kernels.sort(key=lambda e: e.time_range.start)
    per_name: dict[str, list[float]] = {}
    for e in kernels:
        per_name.setdefault(e.name, []).append(e.time_range.elapsed_us())
    # Every kernel name should appear once per call; the per-call total is
    # the sum of per-kernel medians, which is robust to profiler stragglers.
    total = 0.0
    for name, durations in per_name.items():
        per_call = len(durations) / calls
        median = statistics.median(durations)
        total += median * per_call
        print(
            f"kernel {median:9.2f} us x{per_call:g}/call  {name[:120]}",
            file=sys.stderr,
        )
    return {"kernel_us": total, "kernel_names": list(per_name)}


def measure_batched(
    fn: Callable,
    repetitions: int,
    calls: int,
    expected: Any,
    case: Any,
    *,
    output_dtype: torch.dtype | None = None,
    sustain_ms: float = 500.0,
    min_replay_us: float = 400.0,
    max_calls: int = 256,
) -> dict:
    """Per-call ms of one CUDA graph holding ``calls`` back-to-back calls.

    A single-call graph replay carries the graph launch latency quantized to
    ~2 us steps, so kernels under ~20 us all read alike; folding many calls
    into one graph amortizes it. Each call still pays its in-graph kernel
    launch gap, which is also what a real sequence of kernels pays.

    The graph is replayed back-to-back for ``sustain_ms`` before timing so a
    power-capped GPU settles on its sustained clock (a B200 at 750 W drops
    from boost to the cap ~100 ms into a load; medians of a 70 us GEMM moved
    from 60 to 69 us), and the median of ``max(repetitions, 7)`` do_bench
    medians is reported as ``graph_calls_ms``. Readings of one implementation
    agree to ~0.3% across processes in this form.

    CUDA event elapsed times are quantized to 1/512 ms (1.953 us) on B200, so
    one replay of a 16-call graph of a 3 us kernel (47 us) reads in 4% steps
    and the median flips between adjacent steps as the true time drifts by a
    fraction of one. When one replay of the ``calls`` graph takes less than
    ``min_replay_us``, a second graph grown by powers of two (up to
    ``max_calls``) until a replay takes at least that long is measured the
    same way and reported as ``graph_long_ms`` with ``graph_long_calls`` (a
    quantum of at most 0.5%; the per-graph launch latency is also amortized
    further, so the two readings differ by a fraction of a microsecond on
    tiny kernels); otherwise ``graph_long_ms`` repeats ``graph_calls_ms``.
    The report prefers ``graph_long_ms``.
    """
    import math

    import torch

    from helion.runtime import cute_cuda_graph

    result: dict[str, Any] = {"graph_calls": calls, "sustain_ms": sustain_ms}

    def output_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, (list, tuple)):
            return sum(output_bytes(v) for v in value)
        if isinstance(value, dict):
            return sum(output_bytes(v) for v in value.values())
        return 0

    def capture(count: int) -> tuple[Any, int]:
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            outputs: list[Any] = []
            with cute_cuda_graph() as graph:
                for _ in range(count):
                    outputs.append(fn())
            torch.cuda.synchronize()
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        graph.replay()
        torch.cuda.synchronize()
        assert_correct(outputs[-1], expected, case, output_dtype=output_dtype)
        return graph, output_bytes(outputs[-1])

    def measure(graph: Any, count: int) -> list[float]:
        started = time.perf_counter()
        while (time.perf_counter() - started) * 1000.0 < sustain_ms:
            graph.replay()
        torch.cuda.synchronize()
        return [_event_ms(graph.replay) / count for _ in range(max(repetitions, 7))]

    try:
        graph, per_call_bytes = capture(calls)
        runs = measure(graph, calls)
        replay_us = statistics.median(runs) * calls * 1000.0
        result.update(
            graph_calls_ms=statistics.median(runs),
            graph_calls_runs_ms=runs,
            graph_replay_us=replay_us,
        )
        if replay_us < min_replay_us and calls < max_calls:
            factor = 2 ** max(1, math.ceil(math.log2(min_replay_us / replay_us)))
            long_calls = min(max_calls, calls * factor)
            # Every call of the grown graph keeps its own output alive; stay
            # within a quarter of the free device memory.
            free_bytes = torch.cuda.mem_get_info()[0]
            while long_calls > calls and long_calls * per_call_bytes > free_bytes // 4:
                long_calls //= 2
            del graph
            graph, _ = capture(long_calls)
            long_runs = measure(graph, long_calls)
            result.update(
                graph_long_ms=statistics.median(long_runs),
                graph_long_calls=long_calls,
                graph_long_runs_ms=long_runs,
            )
        else:
            result.update(
                graph_long_ms=result["graph_calls_ms"], graph_long_calls=calls
            )
    except Exception as error:
        result["graph_calls_error"] = f"{type(error).__name__}: {error}"
        # A failed capture leaves the capture stream current; later
        # measurements must not run on it.
        torch.cuda.set_stream(torch.cuda.default_stream())
    return result


def measure_cold(fn: Callable, repetitions: int, repeat: int = 100) -> dict:
    """Per-call ms in the autotuner's own regime: ``interleaved_bench`` flushes
    L2 before every timed call (batched into one CUDA graph of up to 16
    [flush; call] pairs for short kernels, with the flush-only graph time
    subtracted), the regime tritonbench measures and Helion's autotuner
    optimizes for, whatever HELION_AUTOTUNE_L2_FLUSH says. Median of
    ``repetitions`` bench medians."""
    import torch

    from helion.autotuner.benchmarking import _make_cudagraph_replay
    from helion.autotuner.benchmarking import interleaved_bench

    result: dict[str, Any] = {
        "bench_regime": f"interleaved_bench(repeat={repeat}, cold L2, CUDA graph)",
    }
    # Backward callables replay autograd graphs whose saved streams PyTorch
    # must redirect into the capture; without this the bench silently falls
    # back to eager calls and times host-side autograd overhead instead.
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        torch.cuda.synchronize()
        _make_cudagraph_replay(fn)  # fail loudly if the call cannot be captured
        runs = [
            interleaved_bench(
                [fn], repeat=repeat, default_cudagraph=True, l2_flush_mode="call"
            )[0]
            for _ in range(repetitions)
        ]
        result.update(bench_ms=statistics.median(runs), bench_runs_ms=runs)
    except Exception as error:
        result["bench_error"] = f"{type(error).__name__}: {error}"
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
    return result


def measure(
    fn: Callable,
    repetitions: int,
    expected: Any,
    case: Any,
    *,
    output_dtype: torch.dtype | None = None,
    graph_nodes: bool = False,
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
        if graph_nodes:
            result["graph_nodes"] = graph_profile(replay)
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
