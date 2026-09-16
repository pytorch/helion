from __future__ import annotations

import pytest
import torch

import helion
from helion._compat import get_triton_version
from helion._testing import DEVICE
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires an AMD GPU",
)


@pytest.mark.parametrize("n_tile,matrix_instr", [(8, 32), (16, 32), (8, 16)])
def test_small_n_mfma_config(n_tile: int, matrix_instr: int) -> None:
    properties = torch.cuda.get_device_properties(DEVICE)
    arch = properties.gcnArchName  # pyrefly: ignore [missing-attribute]
    if arch.split(":")[0] != "gfx950":
        pytest.skip("MFMA wide-store optimization requires gfx950")

    @helion.kernel(static_shapes=True)
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
        for m, n in hl.tile((a.size(0), b.size(1))):
            out[m, n] = a[m, :] @ b[:, n]
        return out

    a = torch.randn((128, 16), device=DEVICE, dtype=torch.float16)
    b = torch.randn((32, 16), device=DEVICE, dtype=torch.float16).T
    bound = matmul.bind((a, b))
    config = helion.Config(block_sizes=[128, n_tile], matrix_instr_nonkdim=matrix_instr)
    needs_workaround = (3, 4) <= get_triton_version().release < (3, 8)
    if needs_workaround and n_tile < 16 and matrix_instr == 32:
        with pytest.raises(InvalidConfig, match="matmul N tile smaller than 16"):
            bound.config_spec.normalize(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert config.config["matrix_instr_nonkdim"] == 0
    else:
        # Unaffected versions must compile and run the original configuration.
        bound.config_spec.normalize(config)
        assert config.config["matrix_instr_nonkdim"] == matrix_instr
    result = bound.compile_config(config)(a, b)
    torch.testing.assert_close(result, a @ b, atol=1e-2, rtol=1e-2)
