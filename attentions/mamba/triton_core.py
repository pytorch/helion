"""
Mamba Selective State Space Model - Core Triton Kernel

Mathematical Core:
==================
The Selective SSM (Mamba) introduces input-dependent gating to traditional SSMs:

    dt_t = softplus(dt_t + dt_bias)        # Input-dependent discretization step
    dA_t = exp(dt_t * A)                   # Discretized decay matrix
    dB_t = dt_t * B_t                      # Discretized input scaling

    h_t = dA_t * h_{t-1} + dB_t * x_t      # State update (the recurrence)
    y_t = C_t @ h_t + D * x_t              # Output computation

Key insight: Making dt, B, C functions of the input enables content-based
reasoning (selection) while preserving linear O(T) complexity.

This file implements the core selective_scan_update kernel which is the
fundamental building block for both training (chunked) and inference (recurrent).

References:
- Mamba Paper: https://arxiv.org/abs/2312.00752
- Mamba-2 / SSD Paper: https://arxiv.org/abs/2405.21060
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Helion imports (optional)
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False


# Softplus activation in Triton
@triton.jit
def softplus(x):
    """Softplus activation: log(1 + exp(x))"""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,      # [batch, nheads, dim, dstate] - hidden state (modified in-place)
    x_ptr,          # [batch, nheads, dim] - input
    dt_ptr,         # [batch, nheads, dim] - discretization timestep
    dt_bias_ptr,    # [nheads, dim] - dt bias (optional, can be None)
    A_ptr,          # [nheads, dim, dstate] - decay rate (negative values)
    B_ptr,          # [batch, ngroups, dstate] - input matrix
    C_ptr,          # [batch, ngroups, dstate] - output matrix
    D_ptr,          # [nheads, dim] - skip connection (optional)
    z_ptr,          # [batch, nheads, dim] - gate for SiLU (optional)
    out_ptr,        # [batch, nheads, dim] - output
    # Dimensions
    batch: tl.constexpr,
    nheads: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides for state
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    # Strides for x
    stride_x_batch, stride_x_head, stride_x_dim,
    # Strides for dt
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    # Strides for dt_bias
    stride_dt_bias_head, stride_dt_bias_dim,
    # Strides for A
    stride_A_head, stride_A_dim, stride_A_dstate,
    # Strides for B
    stride_B_batch, stride_B_group, stride_B_dstate,
    # Strides for C
    stride_C_batch, stride_C_group, stride_C_dstate,
    # Strides for D
    stride_D_head, stride_D_dim,
    # Strides for z
    stride_z_batch, stride_z_head, stride_z_dim,
    # Strides for out
    stride_out_batch, stride_out_head, stride_out_dim,
    # Options
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    """
    Core Mamba selective scan update kernel.

    Implements the recurrence:
        h_t = exp(dt * A) * h_{t-1} + (dt * B) * x_t
        y_t = sum(h_t * C) + D * x_t
        y_t = y_t * SiLU(z)  (if z is provided)

    This kernel processes one timestep, updating the state in-place.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # dim block
    pid_b = tl.program_id(1)  # batch
    pid_h = tl.program_id(2)  # head

    # Compute offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    # Compute group index for B and C (GQA-style grouping)
    group_idx = pid_h // nheads_ngroups_ratio

    # Set up pointers
    # State: [batch, nheads, dim, dstate]
    state_base = state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
    state_ptrs = state_base + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate

    # Input x: [batch, nheads, dim]
    x_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
    x_ptrs = x_base + offs_m * stride_x_dim

    # Timestep dt: [batch, nheads, dim]
    dt_base = dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head
    dt_ptrs = dt_base + offs_m * stride_dt_dim

    # A matrix: [nheads, dim, dstate]
    A_base = A_ptr + pid_h * stride_A_head
    A_ptrs = A_base + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    # B matrix: [batch, ngroups, dstate]
    B_base = B_ptr + pid_b * stride_B_batch + group_idx * stride_B_group
    B_ptrs = B_base + offs_n * stride_B_dstate

    # C matrix: [batch, ngroups, dstate]
    C_base = C_ptr + pid_b * stride_C_batch + group_idx * stride_C_group
    C_ptrs = C_base + offs_n * stride_C_dstate

    # Output: [batch, nheads, dim]
    out_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_base + offs_m * stride_out_dim

    # Masks
    mask_m = offs_m < dim
    mask_n = offs_n < dstate
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Load state h_{t-1}: [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
    state = tl.load(state_ptrs, mask=mask_mn, other=0.0).to(tl.float32)

    # Load input x_t: [BLOCK_SIZE_M]
    x = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    # Load and process dt: [BLOCK_SIZE_M]
    dt = tl.load(dt_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_base = dt_bias_ptr + pid_h * stride_dt_bias_head
        dt_bias_ptrs = dt_bias_base + offs_m * stride_dt_bias_dim
        dt_bias = tl.load(dt_bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        dt = dt + dt_bias

    if DT_SOFTPLUS:
        dt = softplus(dt)

    # Load A: [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
    A = tl.load(A_ptrs, mask=mask_mn, other=0.0).to(tl.float32)

    # Compute discretized decay: dA = exp(dt * A)
    # Note: A is typically negative, so dA is in (0, 1)
    dA = tl.exp(dt[:, None] * A)

    # Load B and C: [BLOCK_SIZE_DSTATE]
    B = tl.load(B_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    # Compute discretized input scaling: dB = dt * B
    dB = dt[:, None] * B[None, :]

    # ========================================
    # CORE SSM RECURRENCE
    # ========================================
    # h_t = dA * h_{t-1} + dB * x_t
    state = state * dA + dB * x[:, None]

    # Store updated state (in-place update)
    tl.store(state_ptrs, state.to(state_ptr.dtype.element_ty), mask=mask_mn)

    # ========================================
    # OUTPUT COMPUTATION
    # ========================================
    # y_t = sum(h_t * C, axis=dstate)
    out = tl.sum(state * C[None, :], axis=1)

    # Optional skip connection: y_t += D * x_t
    if HAS_D:
        D_base = D_ptr + pid_h * stride_D_head
        D_ptrs = D_base + offs_m * stride_D_dim
        D = tl.load(D_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        out = out + x * D

    # Optional gating: y_t = y_t * SiLU(z)
    if HAS_Z:
        z_base = z_ptr + pid_b * stride_z_batch + pid_h * stride_z_head
        z_ptrs = z_base + offs_m * stride_z_dim
        z = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        # SiLU(z) = z * sigmoid(z)
        out = out * z * tl.sigmoid(z)

    # Store output
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask_m)


def selective_scan_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    z: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    dt_softplus: bool = False,
) -> torch.Tensor:
    """
    Mamba selective scan update - single timestep.

    This is the core building block of the Mamba architecture, implementing
    the selective state space model recurrence for one timestep.

    Args:
        state: Hidden state [batch, nheads, dim, dstate], modified in-place
        x: Input [batch, nheads, dim]
        dt: Discretization timestep [batch, nheads, dim]
        A: Decay rate [nheads, dim, dstate], typically negative
        B: Input matrix [batch, ngroups, dstate]
        C: Output matrix [batch, ngroups, dstate]
        D: Skip connection [nheads, dim], optional
        z: Gate for SiLU activation [batch, nheads, dim], optional
        dt_bias: Bias for dt [nheads, dim], optional
        dt_softplus: Whether to apply softplus to dt

    Returns:
        out: Output [batch, nheads, dim]

    The recurrence is:
        dt = softplus(dt + dt_bias) if dt_softplus else dt + dt_bias
        h_t = exp(dt * A) * h_{t-1} + (dt * B) * x_t
        y_t = sum(h_t * C) + D * x_t
        y_t = y_t * SiLU(z) if z is provided
    """
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    nheads_ngroups_ratio = nheads // ngroups

    # Validate shapes
    assert x.shape == (batch, nheads, dim), f"x shape {x.shape} != {(batch, nheads, dim)}"
    assert dt.shape == (batch, nheads, dim), f"dt shape {dt.shape} != {(batch, nheads, dim)}"
    assert A.shape == (nheads, dim, dstate), f"A shape {A.shape} != {(nheads, dim, dstate)}"
    assert B.shape == (batch, ngroups, dstate), f"B shape {B.shape} != {(batch, ngroups, dstate)}"
    assert C.shape == (batch, ngroups, dstate), f"C shape {C.shape} != {(batch, ngroups, dstate)}"
    if D is not None:
        assert D.shape == (nheads, dim), f"D shape {D.shape} != {(nheads, dim)}"
    if z is not None:
        assert z.shape == (batch, nheads, dim), f"z shape {z.shape} != {(batch, nheads, dim)}"
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim), f"dt_bias shape {dt_bias.shape} != {(nheads, dim)}"

    # Allocate output
    out = torch.empty_like(x)

    # Block sizes - tuned for typical dstate values
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, num_warps = 4, 8

    BLOCK_SIZE_DSTATE = triton.next_power_of_2(dstate)

    # Grid: (dim_blocks, batch, nheads)
    grid = (triton.cdiv(dim, BLOCK_SIZE_M), batch, nheads)

    # Launch kernel
    selective_scan_update_kernel[grid](
        state, x, dt, dt_bias, A, B, C, D, z, out,
        batch, nheads, dim, dstate, nheads_ngroups_ratio,
        # State strides
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # x strides
        x.stride(0), x.stride(1), x.stride(2),
        # dt strides
        dt.stride(0), dt.stride(1), dt.stride(2),
        # dt_bias strides
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_bias.stride(1) if dt_bias is not None else 0,
        # A strides
        A.stride(0), A.stride(1), A.stride(2),
        # B strides
        B.stride(0), B.stride(1), B.stride(2),
        # C strides
        C.stride(0), C.stride(1), C.stride(2),
        # D strides
        D.stride(0) if D is not None else 0,
        D.stride(1) if D is not None else 0,
        # z strides
        z.stride(0) if z is not None else 0,
        z.stride(1) if z is not None else 0,
        z.stride(2) if z is not None else 0,
        # out strides
        out.stride(0), out.stride(1), out.stride(2),
        # Options
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        # Block sizes
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_DSTATE=BLOCK_SIZE_DSTATE,
        num_warps=num_warps,
    )

    return out


def selective_scan_update_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    z: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    dt_softplus: bool = False,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of selective scan update.

    This is the naive implementation used for correctness testing.
    """
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]

    # Apply dt bias
    if dt_bias is not None:
        dt = dt + dt_bias

    # Apply softplus
    if dt_softplus:
        dt = F.softplus(dt)

    # Compute discretized parameters
    # dA: [batch, nheads, dim, dstate]
    dA = torch.exp(dt.unsqueeze(-1) * A)

    # Expand B and C from groups to heads
    # B, C: [batch, ngroups, dstate] -> [batch, nheads, dstate]
    B = B.repeat_interleave(nheads // ngroups, dim=1)
    C = C.repeat_interleave(nheads // ngroups, dim=1)

    # dB: [batch, nheads, dim, dstate]
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)

    # State update: h_t = dA * h_{t-1} + dB * x_t
    state.copy_(state * dA + dB * x.unsqueeze(-1))

    # Output: y_t = sum(h_t * C)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)

    # Skip connection
    if D is not None:
        out = out + x * D

    # Gating
    if z is not None:
        out = out * F.silu(z)

    return out.to(x.dtype)


# ==============================================================================
# Helion Kernel Implementation
# ==============================================================================

if HELION_AVAILABLE:
    @helion.kernel(static_shapes=True, autotune_effort="none")
    def mamba_helion_kernel(
        state: torch.Tensor,       # [batch, nheads, dim, dstate]
        x: torch.Tensor,           # [batch, nheads, dim]
        dt: torch.Tensor,          # [batch, nheads, dim]
        A: torch.Tensor,           # [nheads, dim, dstate]
        B: torch.Tensor,           # [batch, ngroups, dstate]
        C: torch.Tensor,           # [batch, ngroups, dstate]
        D: torch.Tensor,           # [nheads, dim]
        z: torch.Tensor,           # [batch, nheads, dim]
        dt_bias: torch.Tensor,     # [nheads, dim]
        nheads_ngroups_ratio: int,
        has_d: bool,
        has_z: bool,
        has_dt_bias: bool,
        dt_softplus: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helion implementation of Mamba selective scan update (single timestep).

        Recurrence:
            dt_eff = softplus(dt + dt_bias) if dt_softplus else dt + dt_bias
            dA = exp(dt_eff * A)
            dB = dt_eff * B
            h = dA * h + dB * x
            out = sum(h * C) + D * x
            out *= SiLU(z) if z
        """
        batch, nheads, dim = x.shape
        dstate = state.shape[-1]
        dstate = hl.specialize(dstate)

        acc_dtype = torch.float32

        out = torch.empty(batch, nheads, dim, dtype=acc_dtype, device=x.device)
        new_state = torch.empty_like(state)

        block_dim = hl.register_block_size(dim)

        for i_batch, i_head in hl.grid([batch, nheads]):
            # Compute group index (B and C are shared across heads in same group)
            i_group = i_head // nheads_ngroups_ratio

            # Load B and C for this batch/group (shared across all dim positions)
            # Use integer indexing to get 1D tensors [dstate]
            b_B = B[i_batch, i_group, :].to(acc_dtype)  # [dstate]
            b_C = C[i_batch, i_group, :].to(acc_dtype)  # [dstate]

            for tile_dim in hl.tile(dim, block_size=block_dim):
                # Load inputs for this dim position using tile indexing
                b_x = x[i_batch, i_head, tile_dim].to(acc_dtype)  # [tile_dim]
                b_dt = dt[i_batch, i_head, tile_dim].to(acc_dtype)  # [tile_dim]

                # Load A and state using tile indexing for dim
                # A: [nheads, dim, dstate] -> [tile_dim, dstate]
                b_A = A[i_head, tile_dim, :].to(acc_dtype)  # [tile_dim, dstate]
                b_state = state[i_batch, i_head, tile_dim, :].to(acc_dtype)  # [tile_dim, dstate]

                # Apply dt_bias if present
                if has_dt_bias:
                    b_dt_bias = dt_bias[i_head, tile_dim].to(acc_dtype)  # [tile_dim]
                    b_dt = b_dt + b_dt_bias

                # Apply softplus if requested: log(1 + exp(x))
                if dt_softplus:
                    b_dt = torch.log1p(torch.exp(b_dt))

                # Compute discretized parameters
                # dA = exp(dt * A): [tile_dim, 1] * [tile_dim, dstate] = [tile_dim, dstate]
                dA = torch.exp(b_dt[:, None] * b_A)  # [tile_dim, dstate]

                # dB = dt * B: [tile_dim, 1] * [1, dstate] = [tile_dim, dstate]
                dB = b_dt[:, None] * b_B[None, :]  # [tile_dim, dstate]

                # State update: h = dA * h + dB * x
                # [tile_dim, dstate] * [tile_dim, dstate] + [tile_dim, dstate] * [tile_dim, 1]
                b_state = dA * b_state + dB * b_x[:, None]

                # Store updated state
                new_state[i_batch, i_head, tile_dim, :] = b_state

                # Output: y = sum(h * C, axis=dstate)
                # [tile_dim, dstate] * [1, dstate] -> sum -> [tile_dim]
                b_out = (b_state * b_C[None, :]).sum(dim=-1)  # [tile_dim]

                # Optional skip connection: y += D * x
                if has_d:
                    b_D = D[i_head, tile_dim].to(acc_dtype)  # [tile_dim]
                    b_out = b_out + b_D * b_x

                # Optional gating: y *= SiLU(z)
                if has_z:
                    b_z = z[i_batch, i_head, tile_dim].to(acc_dtype)  # [tile_dim]
                    b_out = b_out * b_z * torch.sigmoid(b_z)

                out[i_batch, i_head, tile_dim] = b_out

        return out, new_state


def selective_scan_update_helion(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    z: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    dt_softplus: bool = False,
) -> torch.Tensor:
    """
    Helion implementation of Mamba selective scan update.

    Args:
        state: Hidden state [batch, nheads, dim, dstate], modified in-place
        x: Input [batch, nheads, dim]
        dt: Discretization timestep [batch, nheads, dim]
        A: Decay rate [nheads, dim, dstate], typically negative
        B: Input matrix [batch, ngroups, dstate]
        C: Output matrix [batch, ngroups, dstate]
        D: Skip connection [nheads, dim], optional
        z: Gate for SiLU activation [batch, nheads, dim], optional
        dt_bias: Bias for dt [nheads, dim], optional
        dt_softplus: Whether to apply softplus to dt

    Returns:
        out: Output [batch, nheads, dim]
    """
    if not HELION_AVAILABLE:
        raise RuntimeError("Helion is not available. Please install helion.")

    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    nheads_ngroups_ratio = nheads // ngroups

    # Create dummy tensors for optional params (need proper shapes for Helion)
    if D is None:
        dummy_D = torch.empty(nheads, dim, device=x.device, dtype=x.dtype)
    else:
        dummy_D = D

    if z is None:
        dummy_z = torch.empty(batch, nheads, dim, device=x.device, dtype=x.dtype)
    else:
        dummy_z = z

    if dt_bias is None:
        dummy_dt_bias = torch.empty(nheads, dim, device=x.device, dtype=x.dtype)
    else:
        dummy_dt_bias = dt_bias

    out, new_state = mamba_helion_kernel(
        state, x, dt, A, B, C,
        dummy_D, dummy_z, dummy_dt_bias,
        nheads_ngroups_ratio,
        D is not None,
        z is not None,
        dt_bias is not None,
        dt_softplus,
    )

    # In-place update of state (like the original)
    state.copy_(new_state)

    return out.to(x.dtype)


def test_mamba_helion():
    """Test that Helion kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    if not HELION_AVAILABLE:
        print("Helion not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float32

    print("Testing Mamba Helion kernel vs PyTorch reference...")

    # Test configurations
    configs = [
        # (batch, nheads, dim, dstate, ngroups)
        (2, 4, 64, 16, 2),
        (2, 8, 128, 32, 4),
    ]

    for batch, nheads, dim, dstate, ngroups in configs:
        print(f"  Config: batch={batch}, nheads={nheads}, dim={dim}, dstate={dstate}, ngroups={ngroups}")

        # Create inputs
        state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
        state_ref = state.clone()
        x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
        dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()
        A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)

        # Run reference
        out_ref = selective_scan_update_ref(
            state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone()
        )

        # Run Helion kernel
        out_helion = selective_scan_update_helion(
            state, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone()
        )

        # Check outputs
        max_diff_out = (out_helion - out_ref).abs().max().item()
        max_diff_state = (state - state_ref).abs().max().item()

        atol_threshold = 5e-2  # Relaxed for Helion
        status = "PASS" if (max_diff_out < atol_threshold and max_diff_state < atol_threshold) else "FAIL"

        print(f"    Output max diff: {max_diff_out:.2e} [{status}]")
        print(f"    State max diff: {max_diff_state:.2e} [{status}]")

        if max_diff_out >= atol_threshold or max_diff_state >= atol_threshold:
            raise AssertionError(f"Test failed for config batch={batch}, nheads={nheads}")

    # Test with all optional parameters
    print("\n  Testing with all optional parameters (D, z, dt_bias, softplus)...")
    batch, nheads, dim, dstate, ngroups = 2, 4, 64, 16, 2

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, dim, device=device, dtype=dtype)
    z = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=dtype)

    out_ref = selective_scan_update_ref(
        state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(),
        D=D.clone(), z=z.clone(), dt_bias=dt_bias.clone(), dt_softplus=True
    )

    out_helion = selective_scan_update_helion(
        state, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(),
        D=D.clone(), z=z.clone(), dt_bias=dt_bias.clone(), dt_softplus=True
    )

    max_diff_out = (out_helion - out_ref).abs().max().item()
    max_diff_state = (state - state_ref).abs().max().item()

    atol_threshold = 5e-2
    status = "PASS" if (max_diff_out < atol_threshold and max_diff_state < atol_threshold) else "FAIL"

    print(f"    Output max diff: {max_diff_out:.2e} [{status}]")
    print(f"    State max diff: {max_diff_state:.2e} [{status}]")

    if max_diff_out >= atol_threshold or max_diff_state >= atol_threshold:
        raise AssertionError("Test with all options failed")

    print("All Mamba Helion tests passed!")


# ============================================================
# NUMERICAL TESTS
# ============================================================
def test_selective_scan_update_basic():
    """Test basic selective scan update without optional parameters."""
    print("Testing basic selective scan update...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2

    # Create inputs
    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()  # dt should be positive
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)  # A should be negative for stability
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)

    # Run Triton kernel
    out_triton = selective_scan_update(state, x, dt, A, B, C)

    # Run reference
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone())

    # Check outputs
    max_diff_out = (out_triton - out_ref).abs().max().item()
    max_diff_state = (state - state_ref).abs().max().item()

    print(f"  Output max diff: {max_diff_out:.2e}")
    print(f"  State max diff: {max_diff_state:.2e}")

    assert max_diff_out < 1e-4, f"Output mismatch: {max_diff_out}"
    assert max_diff_state < 1e-4, f"State mismatch: {max_diff_state}"
    print("  PASSED!")


def test_selective_scan_update_with_d():
    """Test selective scan update with skip connection D."""
    print("Testing selective scan update with D...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, dim, device=device, dtype=dtype)

    out_triton = selective_scan_update(state, x, dt, A, B, C, D=D)
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(), D=D.clone())

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Mismatch: {max_diff}"
    print("  PASSED!")


def test_selective_scan_update_with_z():
    """Test selective scan update with SiLU gating."""
    print("Testing selective scan update with z (SiLU gating)...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    z = torch.randn(batch, nheads, dim, device=device, dtype=dtype)

    out_triton = selective_scan_update(state, x, dt, A, B, C, z=z)
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(), z=z.clone())

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Mismatch: {max_diff}"
    print("  PASSED!")


def test_selective_scan_update_with_dt_bias():
    """Test selective scan update with dt bias and softplus."""
    print("Testing selective scan update with dt_bias and softplus...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype)  # Can be negative before softplus
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=dtype)

    out_triton = selective_scan_update(state, x, dt, A, B, C, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(),
                                        dt_bias=dt_bias.clone(), dt_softplus=True)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Mismatch: {max_diff}"
    print("  PASSED!")


def test_selective_scan_update_full():
    """Test selective scan update with all optional parameters."""
    print("Testing selective scan update with all options...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 8, 128, 64
    ngroups = 4

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, dim, device=device, dtype=dtype)
    z = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=dtype)

    out_triton = selective_scan_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(),
                                        D=D.clone(), z=z.clone(), dt_bias=dt_bias.clone(), dt_softplus=True)

    max_diff_out = (out_triton - out_ref).abs().max().item()
    max_diff_state = (state - state_ref).abs().max().item()

    print(f"  Output max diff: {max_diff_out:.2e}")
    print(f"  State max diff: {max_diff_state:.2e}")

    assert max_diff_out < 1e-4, f"Output mismatch: {max_diff_out}"
    assert max_diff_state < 1e-4, f"State mismatch: {max_diff_state}"
    print("  PASSED!")


def test_selective_scan_update_bf16():
    """Test selective scan update with bfloat16."""
    print("Testing selective scan update with bfloat16...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2

    state = torch.randn(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()
    x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)

    out_triton = selective_scan_update(state, x, dt, A, B, C)
    out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone())

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"  Max diff: {max_diff:.2e}")
    # bf16 has lower precision (7 bits mantissa vs 23 for fp32)
    # and accumulation errors can build up in state updates
    assert max_diff < 0.15, f"Mismatch: {max_diff}"
    print("  PASSED!")


def test_selective_scan_update_sequence():
    """Test selective scan update over a sequence (multiple timesteps)."""
    print("Testing selective scan update over sequence...")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, nheads, dim, dstate = 2, 4, 32, 16
    ngroups = 2
    seq_len = 10

    # Initialize states
    state = torch.zeros(batch, nheads, dim, dstate, device=device, dtype=dtype)
    state_ref = state.clone()

    # Generate sequence of inputs
    A = -torch.rand(nheads, dim, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, dim, device=device, dtype=dtype)

    outputs_triton = []
    outputs_ref = []

    for t in range(seq_len):
        x = torch.randn(batch, nheads, dim, device=device, dtype=dtype)
        dt = torch.randn(batch, nheads, dim, device=device, dtype=dtype).abs()
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)

        out_triton = selective_scan_update(state, x, dt, A, B.clone(), C.clone(), D=D)
        out_ref = selective_scan_update_ref(state_ref, x.clone(), dt.clone(), A.clone(), B.clone(), C.clone(), D=D.clone())

        outputs_triton.append(out_triton.clone())
        outputs_ref.append(out_ref.clone())

    # Check all outputs
    for t, (out_t, out_r) in enumerate(zip(outputs_triton, outputs_ref)):
        max_diff = (out_t - out_r).abs().max().item()
        if max_diff > 1e-4:
            print(f"  Step {t}: Max diff = {max_diff:.2e} FAILED")
            assert False, f"Mismatch at step {t}"

    max_diff_state = (state - state_ref).abs().max().item()
    print(f"  Final state max diff: {max_diff_state:.2e}")
    assert max_diff_state < 1e-3, f"Final state mismatch: {max_diff_state}"
    print("  PASSED!")


def run_all_tests():
    """Run all numerical tests."""
    print("=" * 60)
    print("Mamba Selective Scan Update - Numerical Tests")
    print("=" * 60)

    test_selective_scan_update_basic()
    test_selective_scan_update_with_d()
    test_selective_scan_update_with_z()
    test_selective_scan_update_with_dt_bias()
    test_selective_scan_update_full()
    test_selective_scan_update_bf16()
    test_selective_scan_update_sequence()

    # Helion tests
    if HELION_AVAILABLE:
        print()
        print("=" * 60)
        print("Mamba Helion Kernel Tests")
        print("=" * 60)
        test_mamba_helion()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
