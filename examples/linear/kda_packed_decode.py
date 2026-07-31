"""Helion versus SGLang packed Kimi Delta Attention decode.

This module implements the exact callable contract of SGLang's default
``fused_recurrent_kda_packed_decode`` kernel and provides a correctness and
latency comparison against that kernel loaded from an SGLang checkout.

The production Kimi-Linear-48B-A3B shape is K=V=128 with 32 global heads.
Tensor parallelism shards those heads, so the default benchmark uses TP=2 and
16 local heads. Activations are bfloat16 while the recurrent state is float32.
This published-model default selects SGLang's in-tree Triton decode. Explicitly
using a bfloat16 state on SM100 makes SGLang auto-select external FlashInfer and
is a different baseline.

Run from the Helion repository root:

    python -m examples.linear.kda_packed_decode
    python -m examples.linear.kda_packed_decode --tp-sizes 1 2 4 8
    HELION_PRINT_OUTPUT_CODE=1 python -m examples.linear.kda_packed_decode
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import inspect
from pathlib import Path
import statistics
import sys
from types import ModuleType
from typing import Callable
from typing import Literal
from typing import cast

import torch

import helion
import helion.language as hl

KIMI_GLOBAL_HEADS = 32
KIMI_HEAD_K_DIM = 128
KIMI_HEAD_V_DIM = 128
KIMI_KDA_LAYERS = 20
SOFTPLUS_THRESHOLD = 20.0


# CUDA x traverses V tiles, keeping the x grid at 16 across all supported TP sizes.
_KDA_CONFIG = helion.Config(
    block_sizes=[8],
    loop_orders=[[2, 1, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
    pid_type="xyz",
)


@helion.kernel(
    static_shapes=False,
    config=_KDA_CONFIG,
)
def _helion_fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Fused packed KDA decode body; mutates ``initial_state`` and ``out``."""
    B = mixed_qkv.size(0)
    HV = hl.specialize(initial_state.size(-3))
    V = hl.specialize(initial_state.size(-2))
    K = hl.specialize(initial_state.size(-1))
    H = hl.specialize((mixed_qkv.size(1) - HV * V) // (2 * K))
    heads_per_q = HV // H

    hl.specialize(
        (
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            A_log.stride(0),
            dt_bias.stride(0),
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            ssm_state_indices.stride(0),
        )
    )

    block_v = hl.register_block_size(1, V)

    for tile_b, tile_hv, tile_v in hl.tile([B, HV, V], block_size=[1, 1, block_v]):
        k_offsets = hl.arange(K)
        i_b = tile_b.id
        i_hv = tile_hv.id
        i_h = i_hv // heads_per_q

        state_index = ssm_state_indices[i_b].long()
        if state_index < 0:
            out[i_b, 0, i_hv, tile_v] = 0.0
        else:
            q_offsets = i_h * K + k_offsets
            k_input_offsets = H * K + i_h * K + k_offsets
            v_offsets = 2 * H * K + i_hv * V + tile_v.index

            gate_input = a[i_b, i_hv * K + k_offsets].float()
            gate_input = gate_input + dt_bias[i_hv * K + k_offsets].float()
            gate_exp = torch.exp(gate_input)
            softplus = torch.where(
                gate_input <= 20.0,
                torch.log(1.0 + gate_exp),
                gate_input,
            )
            A_log_value = A_log[i_hv].float()
            A = torch.exp(A_log_value)
            beta = torch.sigmoid(b[i_b, i_hv].float())
            log_decay = -A * softplus

            state = initial_state[state_index, i_hv, tile_v.index, k_offsets].float()
            decay = torch.exp(log_decay)
            state = state * decay[None, :]

            k = mixed_qkv[i_b, k_input_offsets].float()
            if use_qk_l2norm_in_kernel:
                k = k / torch.sqrt((k * k).sum() + 1e-6)
            v = mixed_qkv[i_b, v_offsets].float()
            value_residual = v - (state * k[None, :]).sum(-1)
            value_residual = value_residual * beta
            state = state + value_residual[:, None] * k[None, :]

            q = mixed_qkv[i_b, q_offsets].float()
            if use_qk_l2norm_in_kernel:
                q = q / torch.sqrt((q * q).sum() + 1e-6)
            q = q * scale
            output = (state * q[None, :]).sum(-1)

            out[i_b, 0, i_hv, tile_v] = output.to(out.dtype)
            initial_state[state_index, i_hv, tile_v.index, k_offsets] = state

    return out


def _validate_packed_decode_inputs(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """Apply the shape and layout checks from SGLang's packed wrapper."""
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            "`ssm_state_indices` must be 1D for packed decode "
            f"(got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    device = mixed_qkv.device
    if any(
        tensor.device != device
        for tensor in (
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            out,
            ssm_state_indices,
        )
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, "
            f"b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] "
            f"(got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV * K:
        raise ValueError(
            f"`a` must have shape [B, HV*K] with HV={HV}, K={K} "
            f"(got a.shape={tuple(a.shape)})."
        )
    if b.shape[1] != HV:
        raise ValueError(
            f"`b` must have shape [B, HV] with HV={HV} (got b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV:
        raise ValueError(f"`A_log` must have {HV} elements (got {A_log.numel()}).")
    if dt_bias.numel() != HV * K:
        raise ValueError(
            f"`dt_bias` must have {HV * K} elements (got {dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(
            f"Invalid packed Q size {q_dim}: must be divisible by K={K}. "
            "KDA packed decode requires num_q_heads == num_k_heads and "
            "head_q_dim == head_k_dim."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    return B, H, HV, K, V


def helion_fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helion implementation of SGLang's packed KDA decode contract.

    Inputs, mutations, padding semantics, and outputs match
    ``fused_recurrent_kda_packed_decode``:

    * ``mixed_qkv`` is ``[B, 2*H*K + HV*V]`` after the short convolution.
    * ``a`` and ``b`` are raw forget-gate and beta logits.
    * ``initial_state`` is ``[num_slots, HV, V, K]`` and is updated in place.
    * ``ssm_state_indices == -1`` writes a zero output and leaves state untouched.
    * ``out`` is ``[B, 1, HV, V]`` and is written in place.
    * The return is the same ``(out, initial_state)`` object pair supplied by the
      caller.
    """
    _validate_packed_decode_inputs(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
    )
    result = _helion_fused_recurrent_kda_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    )
    return result, initial_state


def torch_fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch reference with the same mutating contract."""
    B, H, HV, K, V = _validate_packed_decode_inputs(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
    )

    q_end = H * K
    k_end = 2 * H * K
    q = mixed_qkv[:, :q_end].reshape(B, H, K).float()
    k = mixed_qkv[:, q_end:k_end].reshape(B, H, K).float()
    v = mixed_qkv[:, k_end:].reshape(B, HV, V).float()
    if HV != H:
        repeat = HV // H
        q = q.repeat_interleave(repeat, dim=1)
        k = k.repeat_interleave(repeat, dim=1)

    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale

    gate_input = a.reshape(B, HV, K).float() + dt_bias.reshape(1, HV, K).float()
    softplus = torch.where(
        gate_input <= SOFTPLUS_THRESHOLD,
        torch.log(1.0 + torch.exp(gate_input)),
        gate_input,
    )
    log_decay = -torch.exp(A_log.reshape(1, HV, 1).float()) * softplus
    beta = torch.sigmoid(b.float())

    valid = ssm_state_indices >= 0
    safe_indices = torch.where(valid, ssm_state_indices, 0).long()
    state = initial_state.index_select(0, safe_indices).float()
    state = state * torch.exp(log_decay)[:, :, None, :]
    value_residual = v - (state * k[:, :, None, :]).sum(-1)
    value_residual = value_residual * beta[:, :, None]
    state = state + value_residual[:, :, :, None] * k[:, :, None, :]
    output = (state * q[:, :, None, :]).sum(-1)

    out[:, 0] = torch.where(valid[:, None, None], output, 0.0).to(out.dtype)
    if valid.any():
        initial_state[safe_indices[valid]] = state[valid].to(initial_state.dtype)
    return out, initial_state


PackedDecode = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ],
    tuple[torch.Tensor, torch.Tensor],
]
PackedDecodeArgs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
]


@dataclass
class KDAInputs:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    initial_state: torch.Tensor
    out: torch.Tensor
    ssm_state_indices: torch.Tensor
    use_qk_l2norm_in_kernel: bool = True

    def args(self) -> PackedDecodeArgs:
        return (
            self.mixed_qkv,
            self.a,
            self.b,
            self.A_log,
            self.dt_bias,
            self.scale,
            self.initial_state,
            self.out,
            self.ssm_state_indices,
            self.use_qk_l2norm_in_kernel,
        )

    def clone_mutable(self) -> KDAInputs:
        return KDAInputs(
            mixed_qkv=self.mixed_qkv,
            a=self.a,
            b=self.b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.scale,
            initial_state=self.initial_state.clone(),
            out=torch.empty_like(self.out),
            ssm_state_indices=self.ssm_state_indices,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
        )


def make_kda_inputs(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    *,
    device: torch.device | str = "cuda",
    activation_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.float32,
    pool_size: int | None = None,
    seed: int = 42,
    padded: bool = False,
) -> KDAInputs:
    """Create stable, production-layout packed KDA decode inputs."""
    if HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")
    if pool_size is None:
        pool_size = B + 16
    if pool_size < B:
        raise ValueError(f"pool_size={pool_size} must be at least B={B}")

    generator = torch.Generator(device=device).manual_seed(seed)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = (
        torch.randn(
            B,
            qkv_dim,
            device=device,
            dtype=activation_dtype,
            generator=generator,
        )
        * 0.1
    )
    a = (
        torch.randn(
            B,
            HV * K,
            device=device,
            dtype=activation_dtype,
            generator=generator,
        )
        * 0.5
        - 1.0
    )
    b = (
        torch.randn(
            B,
            HV,
            device=device,
            dtype=activation_dtype,
            generator=generator,
        )
        * 0.5
    )
    A_log = (
        torch.randn(HV, device=device, dtype=torch.float32, generator=generator) * 0.2
    )
    dt_bias = (
        torch.randn(HV * K, device=device, dtype=torch.float32, generator=generator)
        * 0.1
    )
    initial_state = (
        torch.randn(
            pool_size,
            HV,
            V,
            K,
            device=device,
            dtype=state_dtype,
            generator=generator,
        )
        * 0.01
    )
    ssm_state_indices = torch.arange(B, device=device, dtype=torch.int32)
    if padded:
        ssm_state_indices[1::2] = -1
    out = torch.empty(B, 1, HV, V, device=device, dtype=activation_dtype)
    return KDAInputs(
        mixed_qkv=mixed_qkv.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=A_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        scale=K**-0.5,
        initial_state=initial_state.contiguous(),
        out=out,
        ssm_state_indices=ssm_state_indices,
    )


def _install_namespace(name: str, path: Path | None = None) -> None:
    module = ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sglang_packed_decode(sglang_root: Path) -> PackedDecode:
    """Load the exact SGLang baseline without importing the full server package."""
    fla_dir = (
        sglang_root / "python" / "sglang" / "kernels" / "ops" / "attention" / "fla"
    )
    baseline_path = fla_dir / "fused_recurrent.py"
    op_path = fla_dir / "op.py"
    if not baseline_path.is_file() or not op_path.is_file():
        raise FileNotFoundError(
            f"Expected SGLang KDA sources under {fla_dir}; "
            "pass --sglang-root explicitly."
        )

    package_paths = {
        "sglang": sglang_root / "python" / "sglang",
        "sglang.kernels": sglang_root / "python" / "sglang" / "kernels",
        "sglang.kernels.ops": sglang_root / "python" / "sglang" / "kernels" / "ops",
        "sglang.kernels.ops.attention": fla_dir.parent,
        "sglang.kernels.ops.attention.fla": fla_dir,
    }
    for name, path in package_paths.items():
        _install_namespace(name, path)

    utils_name = "sglang.kernels.ops.attention.fla.utils"
    utils = ModuleType(utils_name)
    utils.input_guard = lambda fn: fn  # type: ignore[attr-defined]
    utils.is_gather_supported = hasattr(  # type: ignore[attr-defined]
        __import__("triton.language", fromlist=["gather"]), "gather"
    )
    sys.modules[utils_name] = utils

    _load_module("sglang.kernels.ops.attention.fla.op", op_path)
    module = _load_module(
        "sglang.kernels.ops.attention.fla.fused_recurrent", baseline_path
    )
    baseline = module.fused_recurrent_kda_packed_decode
    if not callable(baseline):
        raise TypeError(f"Unexpected baseline object: {baseline!r}")
    return cast("PackedDecode", baseline)


def assert_matching_signatures(baseline: PackedDecode) -> None:
    """Check argument names, order, kinds, and defaults against SGLang."""
    expected = inspect.signature(baseline)
    actual = inspect.signature(helion_fused_recurrent_kda_packed_decode)
    expected_params = list(expected.parameters.values())
    actual_params = list(actual.parameters.values())
    if len(expected_params) != len(actual_params):
        raise AssertionError(f"Signature length mismatch: {actual} != {expected}")
    for actual_param, expected_param in zip(
        actual_params, expected_params, strict=True
    ):
        if (
            actual_param.name != expected_param.name
            or actual_param.kind != expected_param.kind
            or actual_param.default != expected_param.default
        ):
            raise AssertionError(
                f"Signature mismatch at {actual_param.name}: {actual} != {expected}"
            )


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max())


def check_correctness(
    baseline: PackedDecode,
    inputs: KDAInputs,
    *,
    atol: float = 2e-2,
    rtol: float = 1e-2,
) -> tuple[float, float]:
    """Check reference, output/state values, mutations, aliases, and padding."""
    original_state = inputs.initial_state.clone()
    original_readonly = tuple(
        tensor.clone()
        for tensor in (
            inputs.mixed_qkv,
            inputs.a,
            inputs.b,
            inputs.A_log,
            inputs.dt_bias,
            inputs.ssm_state_indices,
        )
    )

    reference_inputs = inputs.clone_mutable()
    baseline_inputs = inputs.clone_mutable()
    helion_inputs = inputs.clone_mutable()

    reference_result = torch_fused_recurrent_kda_packed_decode(*reference_inputs.args())
    baseline_result = baseline(*baseline_inputs.args())
    helion_result = helion_fused_recurrent_kda_packed_decode(*helion_inputs.args())
    torch.cuda.synchronize()

    for result, call_inputs, name in (
        (reference_result, reference_inputs, "reference"),
        (baseline_result, baseline_inputs, "SGLang"),
        (helion_result, helion_inputs, "Helion"),
    ):
        if result[0].data_ptr() != call_inputs.out.data_ptr():
            raise AssertionError(f"{name} did not return the supplied out tensor")
        if result[1].data_ptr() != call_inputs.initial_state.data_ptr():
            raise AssertionError(f"{name} did not return the supplied state tensor")

    torch.testing.assert_close(
        baseline_inputs.out, reference_inputs.out, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        baseline_inputs.initial_state,
        reference_inputs.initial_state,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        helion_inputs.out, baseline_inputs.out, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        helion_inputs.initial_state,
        baseline_inputs.initial_state,
        atol=atol,
        rtol=rtol,
    )

    valid_indices = inputs.ssm_state_indices[inputs.ssm_state_indices >= 0].long()
    touched = torch.zeros(
        original_state.shape[0], dtype=torch.bool, device=original_state.device
    )
    touched[valid_indices] = True
    if not torch.equal(helion_inputs.initial_state[~touched], original_state[~touched]):
        raise AssertionError("Helion modified an unselected state-cache row")
    invalid = inputs.ssm_state_indices < 0
    if invalid.any() and torch.count_nonzero(helion_inputs.out[invalid]) != 0:
        raise AssertionError("Helion did not zero output for a padded state index")

    for current, original, name in zip(
        (
            inputs.mixed_qkv,
            inputs.a,
            inputs.b,
            inputs.A_log,
            inputs.dt_bias,
            inputs.ssm_state_indices,
        ),
        original_readonly,
        ("mixed_qkv", "a", "b", "A_log", "dt_bias", "ssm_state_indices"),
        strict=True,
    ):
        if not torch.equal(current, original):
            raise AssertionError(f"Read-only input {name} was modified")

    return (
        _max_abs_diff(helion_inputs.out, baseline_inputs.out),
        _max_abs_diff(
            helion_inputs.initial_state[valid_indices],
            baseline_inputs.initial_state[valid_indices],
        ),
    )


def benchmark_one(
    baseline: PackedDecode,
    inputs: KDAInputs,
    *,
    warmup_ms: int,
    rep_ms: int,
    rounds: int,
) -> tuple[float, float]:
    """Return median SGLang and Helion latency in milliseconds."""
    from triton.testing import do_bench

    baseline_inputs = inputs.clone_mutable()
    helion_inputs = inputs.clone_mutable()

    def run_baseline() -> None:
        baseline(*baseline_inputs.args())

    def run_helion() -> None:
        helion_fused_recurrent_kda_packed_decode(*helion_inputs.args())

    run_baseline()
    run_helion()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"SGLang": [], "Helion": []}
    benchmarks = (("SGLang", run_baseline), ("Helion", run_helion))
    for round_index in range(rounds):
        round_benchmarks = benchmarks if round_index % 2 == 0 else benchmarks[::-1]
        for name, fn in round_benchmarks:
            samples[name].append(
                cast("float", do_bench(fn, warmup=warmup_ms, rep=rep_ms))
            )
    return statistics.median(samples["SGLang"]), statistics.median(samples["Helion"])


def autotune_decode_shapes(
    batch_sizes: list[int],
    *,
    local_heads: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
    seed: int,
    aggregation: Literal["geomean", "max"],
) -> helion.Config:
    """Select one config using equal-weight relative latency across batch sizes."""
    arg_sets = [
        make_kda_inputs(
            batch_size,
            local_heads,
            local_heads,
            KIMI_HEAD_K_DIM,
            KIMI_HEAD_V_DIM,
            activation_dtype=activation_dtype,
            state_dtype=state_dtype,
            seed=seed + index,
        ).args()
        for index, batch_size in enumerate(batch_sizes)
    ]
    cache_tag = (
        "kda-packed-decode-v1-"
        f"h{local_heads}-b{'-'.join(map(str, batch_sizes))}-"
        f"{str(activation_dtype).removeprefix('torch.')}-"
        f"{str(state_dtype).removeprefix('torch.')}"
    )
    return _helion_fused_recurrent_kda_packed_decode.autotune_multi(
        arg_sets,
        aggregation=aggregation,
        relative_to="default",
        cache_tag=cache_tag,
        force=True,
    )


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _default_sglang_root() -> Path:
    return Path(__file__).resolve().parents[3] / "sglang"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Helion with SGLang's default packed KDA decode kernel."
    )
    parser.add_argument("--sglang-root", type=Path, default=_default_sglang_root())
    parser.add_argument(
        "--mode", choices=("all", "correctness", "bench"), default="all"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256]
    )
    parser.add_argument(
        "--multi-autotune",
        action="store_true",
        help="Jointly autotune one config before running the selected mode.",
    )
    parser.add_argument("--tune-batch-sizes", type=int, nargs="+", default=[1, 256])
    parser.add_argument(
        "--tune-aggregation", choices=("geomean", "max"), default="geomean"
    )
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=[2],
        help="Tensor-parallel sizes; local heads are 32 / TP (default: 2).",
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--state-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
        help="Kimi-Linear's default recurrent-state dtype is float32.",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the SGLang and Helion kernels.")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1.")
    for tp_size in args.tp_sizes:
        if KIMI_GLOBAL_HEADS % tp_size != 0:
            raise SystemExit(
                f"TP={tp_size} does not divide {KIMI_GLOBAL_HEADS} Kimi heads."
            )

    baseline = load_sglang_packed_decode(args.sglang_root.resolve())
    assert_matching_signatures(baseline)

    activation_dtype = _dtype(args.activation_dtype)
    state_dtype = _dtype(args.state_dtype)
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    print(
        f"Device: {torch.cuda.get_device_name(device)} "
        f"(SM{capability[0]}{capability[1]})"
    )
    print(f"SGLang source: {args.sglang_root.resolve()}")
    print(f"Helion config: {_KDA_CONFIG}")
    print(
        f"Contract: activation={activation_dtype}, state={state_dtype}, "
        f"K={KIMI_HEAD_K_DIM}, V={KIMI_HEAD_V_DIM}, "
        "raw gate/beta logits, q/k L2 normalization"
    )
    if capability[0] >= 10 and state_dtype is torch.bfloat16:
        print(
            "Note: this still compares the explicit Triton source; SGLang's "
            "SM100 server default for a bfloat16 state is external FlashInfer."
        )

    if args.multi_autotune:
        if len(args.tp_sizes) != 1:
            raise SystemExit("--multi-autotune requires exactly one --tp-sizes value.")
        local_heads = KIMI_GLOBAL_HEADS // args.tp_sizes[0]
        print(
            "Joint autotune: "
            f"B={args.tune_batch_sizes}, H={local_heads}, "
            f"aggregation={args.tune_aggregation}, relative_to=default"
        )
        winner = autotune_decode_shapes(
            args.tune_batch_sizes,
            local_heads=local_heads,
            activation_dtype=activation_dtype,
            state_dtype=state_dtype,
            seed=args.seed,
            aggregation=args.tune_aggregation,
        )
        print(f"Joint autotune winner: {winner}")

    if args.mode in ("all", "correctness"):
        print("\nCorrectness and mutation contract")
        for tp_size in args.tp_sizes:
            local_heads = KIMI_GLOBAL_HEADS // tp_size
            for batch_size in args.batch_sizes:
                inputs = make_kda_inputs(
                    batch_size,
                    local_heads,
                    local_heads,
                    KIMI_HEAD_K_DIM,
                    KIMI_HEAD_V_DIM,
                    device=device,
                    activation_dtype=activation_dtype,
                    state_dtype=state_dtype,
                    seed=args.seed,
                )
                output_diff, state_diff = check_correctness(
                    baseline, inputs, atol=args.atol, rtol=args.rtol
                )
                print(
                    f"  TP={tp_size} B={batch_size:>3} H={local_heads:>2}: "
                    f"PASS output_max={output_diff:.3e} "
                    f"state_max={state_diff:.3e}"
                )

            padded_inputs = make_kda_inputs(
                4,
                local_heads,
                local_heads,
                KIMI_HEAD_K_DIM,
                KIMI_HEAD_V_DIM,
                device=device,
                activation_dtype=activation_dtype,
                state_dtype=state_dtype,
                seed=args.seed + 1,
                padded=True,
            )
            check_correctness(baseline, padded_inputs, atol=args.atol, rtol=args.rtol)
            print(f"  TP={tp_size} padded cache indices: PASS")

        grouped_inputs = make_kda_inputs(
            4,
            8,
            16,
            KIMI_HEAD_K_DIM,
            KIMI_HEAD_V_DIM,
            device=device,
            activation_dtype=activation_dtype,
            state_dtype=state_dtype,
            seed=args.seed + 2,
        )
        check_correctness(baseline, grouped_inputs, atol=args.atol, rtol=args.rtol)
        print("  grouped heads H=8, HV=16: PASS")

    if args.mode in ("all", "bench"):
        print("\nLatency (microseconds, lower is better)")
        print("  TP    B   H |  SGLang   Helion  speedup  saved/layer  saved/20 layers")
        print("  " + "-" * 72)
        for tp_size in args.tp_sizes:
            local_heads = KIMI_GLOBAL_HEADS // tp_size
            for batch_size in args.batch_sizes:
                inputs = make_kda_inputs(
                    batch_size,
                    local_heads,
                    local_heads,
                    KIMI_HEAD_K_DIM,
                    KIMI_HEAD_V_DIM,
                    device=device,
                    activation_dtype=activation_dtype,
                    state_dtype=state_dtype,
                    seed=args.seed,
                )
                baseline_ms, helion_ms = benchmark_one(
                    baseline,
                    inputs,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                    rounds=args.rounds,
                )
                baseline_us = baseline_ms * 1000
                helion_us = helion_ms * 1000
                saved_us = baseline_us - helion_us
                speedup = baseline_us / helion_us
                print(
                    f"  {tp_size:>2} {batch_size:>4} {local_heads:>3} | "
                    f"{baseline_us:>7.1f} {helion_us:>8.1f} "
                    f"{speedup:>7.2f}x {saved_us:>11.1f} "
                    f"{saved_us * KIMI_KDA_LAYERS:>15.1f}"
                )


if __name__ == "__main__":
    main()
