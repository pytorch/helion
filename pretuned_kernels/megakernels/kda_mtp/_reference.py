"""Reference fixtures and SGLang verification for the Kimi-K3 KDA verify megakernel
(the torch oracle of helion-kda-decode-mtp, ReplaySSM cache ring)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

EPS = 1e-05
HEAD_DIM = 128
HIDDEN = 7168
LOWER_BOUND = -5.0
SCALE = HEAD_DIM ** (-0.5)

if TYPE_CHECKING:
    from collections.abc import Callable


def make_inputs(
    batch: int,
    heads: int,
    tokens: int,
    *,
    cache_ring: bool,
    seed: int = 42,
    padded: bool = False,
    strided: bool = False,
    device: torch.device | str = "cuda",
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    segment = heads * HEAD_DIM
    slots = batch + 5
    scratch_rows = batch + 2
    scratch_steps = tokens + 2
    ring_length = 16
    projection_width = (4 * segment + HEAD_DIM + heads + 15) // 16 * 16

    def randn(
        *shape: int, dtype: torch.dtype = torch.bfloat16, scale: float = 0.1
    ) -> torch.Tensor:
        return (
            torch.randn(shape, device=device, dtype=dtype, generator=generator) * scale
        )

    def pool(
        shape: tuple[int, ...], dtype: torch.dtype, scale: float, *, gap: int = 0
    ) -> torch.Tensor:
        value = randn(*shape, dtype=dtype, scale=scale)
        if not gap:
            return value
        strides = list(value.stride())
        strides[0] += gap
        replica = torch.empty_strided(shape, strides, device=device, dtype=dtype)
        replica.copy_(value)
        return replica

    conv = randn(3 * segment, 4, dtype=torch.float32, scale=0.15)
    indices = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
    if padded:
        indices[-1] = -1
    return {
        "hidden_states": randn(batch * tokens, HIDDEN),
        "input_weight": randn(projection_width, HIDDEN, scale=0.02),
        "f_b_weight": randn(segment, HEAD_DIM, scale=0.05),
        # This view keeps the channel-major production weights without copying.
        "conv_weight": conv.view(3, segment, 4).transpose(1, 2),
        "a_log": randn(heads, dtype=torch.float32, scale=0.2),
        "dt_bias": randn(segment, dtype=torch.float32, scale=0.1),
        "conv_state": pool(
            (slots, 3, 3 * segment),
            torch.bfloat16,
            0.1,
            gap=64 if strided else 0,
        ),
        "recurrent_state": pool(
            (slots, heads, HEAD_DIM, HEAD_DIM),
            torch.float32,
            0.02,
            gap=128 if strided else 0,
        ),
        "state_indices": indices,
        "intermediate_states": (
            None
            if cache_ring
            else torch.full(
                (scratch_rows, scratch_steps, heads, HEAD_DIM, HEAD_DIM),
                42.0,
                device=device,
                dtype=torch.float32,
            )
        ),
        "intermediate_conv": torch.full(
            (scratch_rows, scratch_steps, 3, 3 * segment),
            42.0,
            device=device,
            dtype=torch.bfloat16,
        ),
        "intermediate_indices": torch.arange(
            batch, 0, -1, device=device, dtype=torch.int32
        ),
        "norm_weight": 1 + randn(HEAD_DIM, dtype=torch.float32, scale=0.02),
        "output_weight": randn(HIDDEN, segment, scale=0.02),
        "replayssm_rawv": (
            torch.full(
                (slots, heads, ring_length, HEAD_DIM),
                42.0,
                device=device,
                dtype=torch.bfloat16,
            )
            if cache_ring
            else None
        ),
        "replayssm_rawk": (
            torch.full(
                (slots, heads, ring_length, HEAD_DIM),
                42.0,
                device=device,
                dtype=torch.bfloat16,
            )
            if cache_ring
            else None
        ),
        "replayssm_g": (
            torch.full(
                (slots, heads, ring_length, HEAD_DIM),
                42.0,
                device=device,
                dtype=torch.float32,
            )
            if cache_ring
            else None
        ),
        "replayssm_beta": (
            torch.full(
                (slots, heads, ring_length), 42.0, device=device, dtype=torch.float32
            )
            if cache_ring
            else None
        ),
    }


_WRITES = (
    "intermediate_states",
    "intermediate_conv",
    "replayssm_rawv",
    "replayssm_rawk",
    "replayssm_g",
    "replayssm_beta",
)


def clone_outputs(tensors: dict) -> dict:
    result = dict(tensors)
    for name in _WRITES:
        if tensors[name] is not None:
            result[name] = tensors[name].clone()
    return result


def project(tensors: dict) -> tuple[torch.Tensor, ...]:
    heads = tensors["recurrent_state"].shape[1]
    segment = heads * HEAD_DIM
    wide = F.linear(tensors["hidden_states"], tensors["input_weight"][: 4 * segment])
    bfa = F.linear(
        tensors["hidden_states"],
        tensors["input_weight"][4 * segment : 4 * segment + HEAD_DIM + heads],
    )
    gate = F.linear(bfa[:, :HEAD_DIM], tensors["f_b_weight"])
    return wide, bfa[:, HEAD_DIM:], gate


def torch_reference(tensors: dict, tokens: int, *, cache_ring: bool) -> torch.Tensor:
    """Independent sequential oracle for FP32 conv/recurrence and rollback."""
    batch = tensors["state_indices"].numel()
    heads = tensors["recurrent_state"].shape[1]
    segment = heads * HEAD_DIM
    wide, beta, raw_gate = project(tensors)
    wide = wide.view(batch, tokens, 4, heads, HEAD_DIM)
    beta = torch.sigmoid(beta.float()).view(batch, tokens, heads)
    raw_gate = raw_gate.float().view(batch, tokens, heads, HEAD_DIM)
    raw_gate = raw_gate + tensors["dt_bias"].view(heads, HEAD_DIM)
    log_decay = LOWER_BOUND * torch.sigmoid(
        torch.exp(tensors["a_log"])[None, None, :, None] * raw_gate
    )
    outputs = torch.zeros(
        batch,
        tokens,
        heads,
        HEAD_DIM,
        device=tensors["hidden_states"].device,
        dtype=torch.bfloat16,
    )
    for request, slot in enumerate(tensors["state_indices"].tolist()):
        if slot < 0:
            continue
        scratch = int(tensors["intermediate_indices"][request])
        window = tensors["conv_state"][slot].float().clone()
        state = tensors["recurrent_state"][slot].clone()
        for step in range(tokens):
            x = wide[request, step, :3].reshape(3 * segment).float()
            four = torch.cat((window, x[None]), dim=0)
            convolved = F.silu(
                (
                    four.view(4, 3, segment).permute(1, 0, 2) * tensors["conv_weight"]
                ).sum(dim=1)
            ).view(3, heads, HEAD_DIM)
            window = four[1:].clone()
            tensors["intermediate_conv"][scratch, step] = window
            q, k, v = convolved.unbind(0)
            if cache_ring:
                tensors["replayssm_rawk"][slot, :, step] = k
                tensors["replayssm_rawv"][slot, :, step] = v
                tensors["replayssm_g"][slot, :, step] = log_decay[request, step]
                tensors["replayssm_beta"][slot, :, step] = beta[request, step]
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            state *= torch.exp(log_decay[request, step])[:, None, :]
            residual = v - (state * k[:, None, :]).sum(-1)
            state += (residual * beta[request, step, :, None])[:, :, None] * k[
                :, None, :
            ]
            core = (state * q[:, None, :]).sum(-1)
            if not cache_ring:
                tensors["intermediate_states"][scratch, step] = state
            outputs[request, step] = (
                core
                * torch.rsqrt((core * core).mean(-1, keepdim=True) + EPS)
                * tensors["norm_weight"]
                * torch.sigmoid(wide[request, step, 3].float())
            )
    return F.linear(outputs.view(batch * tokens, segment), tensors["output_weight"])


def make_sglang_call(
    tensors: dict,
    tokens: int,
    *,
    side_stream: torch.cuda.Stream | None = None,
    gemm_backend: str = "auto",
) -> Callable[[], torch.Tensor]:
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import use_cutedsl_bf16_gemm
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    batch = tensors["state_indices"].numel()
    heads = tensors["recurrent_state"].shape[1]
    segment = heads * HEAD_DIM
    pool_size = tensors["recurrent_state"].shape[0]
    wide_weight = tensors["input_weight"][: 4 * segment]
    bfa_weight = tensors["input_weight"][4 * segment :]
    cache = SimpleNamespace(
        conv=[tensors["conv_state"]],
        temporal=tensors["recurrent_state"],
        intermediate_ssm=tensors["intermediate_states"],
        intermediate_conv_window=[tensors["intermediate_conv"]],
        replayssm_rawv=tensors["replayssm_rawv"],
        replayssm_rawk=tensors["replayssm_rawk"],
        replayssm_g=tensors["replayssm_g"],
        replayssm_beta=tensors["replayssm_beta"],
    )
    layer = SimpleNamespace(
        layer_id=0,
        bias=None,
        lower_bound=LOWER_BOUND,
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=heads,
        head_q_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        q_dim=segment,
        k_dim=segment,
        v_dim=segment,
        conv_weights=tensors["conv_weight"].transpose(1, 2).reshape(3 * segment, 4),
        A_log=tensors["a_log"],
        dt_bias=tensors["dt_bias"],
        _k3_fused_decode_args=(
            None,
            None,
            None,
            None,
            None,
            tensors["norm_weight"],
            EPS,
        ),
    )
    backend = KDAAttnBackend.__new__(KDAAttnBackend)
    backend.kernel_dispatcher = KDAKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.NV_CUTEDSL,
    )
    backend.req_to_token_pool = SimpleNamespace(size=pool_size)
    backend.state_pool = SimpleNamespace(replayssm_request_ring=False)
    backend.state_layer_cache = lambda layer_id: cache
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        * tokens,
        mamba_cache_indices=tensors["state_indices"],
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backend.verify_intermediate_state_indices = tensors["intermediate_indices"]
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=tokens),
        ragged_verify_layout=None,
        req_pool_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
    )

    # SGLang AUTO defaults to CuTe/TGV on Blackwell, then applies this exact
    # coverage heuristic. Its GEMMs and the MTP kernel launch with PDL enabled.
    def dense(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if gemm_backend == "auto" and use_cutedsl_bf16_gemm(
            x.shape[0], weight.shape[0], weight.shape[1]
        ):
            return cutedsl_bf16_gemm(x, weight)
        return F.linear(x, weight)

    def call() -> torch.Tensor:
        if side_stream is None:
            wide = dense(tensors["hidden_states"], wide_weight)
            bfa = kimi_k3_tiny_gemm(tensors["hidden_states"], bfa_weight)
            gate = kimi_k3_tiny_gemm(bfa[:, :HEAD_DIM], tensors["f_b_weight"])
        else:
            current = torch.cuda.current_stream()
            side_stream.wait_stream(current)
            wide = dense(tensors["hidden_states"], wide_weight)
            with torch.cuda.stream(side_stream):
                bfa = kimi_k3_tiny_gemm(tensors["hidden_states"], bfa_weight)
                gate = kimi_k3_tiny_gemm(bfa[:, :HEAD_DIM], tensors["f_b_weight"])
            current.wait_stream(side_stream)
        mixed_qkv = wide[:, : 3 * segment]
        beta = bfa[:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0)
        gate = gate.view(1, batch * tokens, heads, HEAD_DIM)
        layer._k3_onorm_gate = wide[:, 3 * segment :]
        layer._k3_onorm_consumed = False
        assert backend._can_run_dspark_cutedsl_mtp(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=gate,
            b=beta,
            draft_token_num=tokens,
            ragged_layout=None,
            conv_states=cache.conv[0],
            ssm_states=cache.temporal,
            intermediate_state_cache=cache.intermediate_ssm,
            intermediate_conv_window_cache=cache.intermediate_conv_window[0],
            retrieve_parent_token=None,
            replayssm_rawv=cache.replayssm_rawv,
        ), "fixture fell outside the production CuTe verification contract"
        core = backend.forward_extend(layer, forward_batch, mixed_qkv, gate, beta)
        assert layer._k3_onorm_consumed, "serving kernel did not fuse the output norm"
        return dense(core.reshape(batch * tokens, segment), tensors["output_weight"])

    return call


def assert_correct(
    actual: torch.Tensor, candidate: dict, expected: torch.Tensor, reference: dict
) -> dict:
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=2e-2)
    batch = candidate["state_indices"].numel()
    verify_tokens = candidate["hidden_states"].shape[0] // batch
    valid_requests = candidate["state_indices"] >= 0
    padded_output = actual.reshape(batch, verify_tokens, *actual.shape[1:])[
        ~valid_requests
    ]
    torch.testing.assert_close(
        padded_output,
        torch.zeros_like(padded_output),
        atol=0,
        rtol=0,
        msg="Padded verification outputs must be exactly zero",
    )
    errors = {"output_max_abs": (actual.float() - expected.float()).abs().max().item()}
    for name in _WRITES:
        if candidate[name] is None:
            continue
        value = candidate[name]
        if name in ("intermediate_states", "intermediate_conv"):
            untouched = torch.ones(
                value.shape[:2], dtype=torch.bool, device=value.device
            )
            scratch = candidate["intermediate_indices"][valid_requests].long()
            untouched[scratch, :verify_tokens] = False
            candidate_untouched = value[untouched]
            reference_untouched = reference[name][untouched]
        else:
            untouched = torch.ones(
                (value.shape[0], value.shape[2]),
                dtype=torch.bool,
                device=value.device,
            )
            slots = candidate["state_indices"][valid_requests].long()
            untouched[slots, :verify_tokens] = False
            # Replay buffers put heads before the token-ring dimension.
            candidate_untouched = value.transpose(1, 2)[untouched]
            reference_untouched = reference[name].transpose(1, 2)[untouched]
        torch.testing.assert_close(
            candidate_untouched,
            reference_untouched,
            atol=0,
            rtol=0,
            msg=f"Unexpected write in untouched {name} buffer",
        )
        if name == "intermediate_conv":
            # Windows contain projected BF16 values, so compare at their ulp.
            torch.testing.assert_close(
                candidate[name], reference[name], atol=2e-3, rtol=8e-3
            )
        elif name in ("replayssm_g", "replayssm_beta"):
            torch.testing.assert_close(
                candidate[name], reference[name], atol=2e-4, rtol=2e-3
            )
        else:
            torch.testing.assert_close(
                candidate[name].float(), reference[name].float(), atol=3e-4, rtol=2e-2
            )
        errors[name + "_max_abs"] = (
            (candidate[name].float() - reference[name].float()).abs().max().item()
        )
    return errors


def commit(tensors: dict, accepted: torch.Tensor, *, cache_ring: bool) -> None:
    if cache_ring:
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_spec,
        )

        commit_kda_replayssm_spec(
            tensors["recurrent_state"],
            tensors["replayssm_rawv"],
            tensors["replayssm_rawk"],
            tensors["replayssm_g"],
            tensors["replayssm_beta"],
            tensors["state_indices"],
            accepted,
            max_cache_len=tensors["replayssm_rawv"].shape[2],
            num_k_heads=tensors["recurrent_state"].shape[1],
            use_qk_l2norm_in_kernel=True,
            null_block_id=-1,
        )
    for request, length in enumerate(accepted.tolist()):
        slot = int(tensors["state_indices"][request])
        row = int(tensors["intermediate_indices"][request])
        if slot < 0 or length == 0:
            continue
        tensors["conv_state"][slot].copy_(tensors["intermediate_conv"][row, length - 1])
        if not cache_ring:
            tensors["recurrent_state"][slot].copy_(
                tensors["intermediate_states"][row, length - 1]
            )
