from __future__ import annotations

import ast
import textwrap

import pytest

from test.test_cute_split_k_cluster import _bind
from test.test_cute_split_k_cluster import _carrier
from test.test_cute_split_k_cluster import cpu_only as cpu_only
from test.test_cute_split_k_workspace import _embedded_sources

from helion._compiler.cute.split_k_cluster_codegen import _a_copy_coordinates


def test_a_packet_coordinates_preserve_flat_index_for_every_bk_residue():
    # Include the complete range around the 256-element packet stride, not
    # only its divisors. Nondivisors must retain the original decomposition.
    for bk in range(1, 514):
        source = _a_copy_coordinates(bk)
        code = compile(textwrap.dedent(source), "<A packet coordinates>", "exec")
        for lane in range(32):
            for ai in range(16):
                values = {"lane": lane, "ai": ai}
                exec(code, {}, values)
                flat = (lane + ai * 32) * 8
                assert (values["m"], values["k"]) == divmod(flat, bk), (
                    bk,
                    lane,
                    ai,
                )
        simplified = bk % 8 == 0 and 256 % bk == 0
        assert ("flat =" not in source) is simplified


@pytest.mark.parametrize("shape", ((32, 4096, 64), (48, 8192, 80)))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("finalizer_warps", (1, 4))
def test_emitted_a_packets_cover_each_private_slab_once(shape, bias, finalizer_warps):
    bound, args = _bind(shape, bias=bias)
    config = _carrier(bound)
    assert config is not None
    config.config["cute_split_k_finalizer_warps"] = finalizer_warps
    (source,) = _embedded_sources(bound.to_code(config))
    tree = ast.parse(source)
    (loop,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ai"
    ]
    assignments = ast.Module(body=loop.body[:-1], type_ignores=[])
    assert all(isinstance(node, ast.Assign) for node in assignments.body)
    code = compile(assignments, "<emitted A coordinates>", "exec")
    m_size, k_size, _ = shape
    # Dense and padded strides use the same emitted coordinates. The pointer
    # expression itself remains in the ordinary compiler/native proof.
    for row_stride in (k_size, k_size + 8):
        for rank in range(8):
            for wave in range(k_size // 4096):
                visited = set()
                for warp in range(4):
                    start_k = rank * (k_size // 8) + warp * (k_size // 32) + wave * 128
                    for lane in range(32):
                        for ai in range(8):
                            values = {"lane": lane, "ai": ai}
                            exec(code, {}, values)
                            m, k = values["m"], values["k"]
                            assert 0 <= m < 16 and 0 <= k <= 120
                            flat = (lane + ai * 32) * 8
                            old_m, old_k = divmod(flat, 128)
                            for tile_m in range(0, m_size, 16):
                                address = (tile_m + m) * row_stride + start_k + k
                                assert (
                                    address
                                    == (tile_m + old_m) * row_stride + start_k + old_k
                                )
                                assert address % 8 == 0
                                assert start_k + k + 7 < k_size
                            for element in range(8):
                                coord = m, k + element, warp
                                assert coord not in visited
                                visited.add(coord)
                assert len(visited) == 16 * 128 * 4
