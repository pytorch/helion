from __future__ import annotations


def select_aux_copy_layout(
    epi_m: int, epi_n: int, dtype_bits: int
) -> tuple[int, int, int]:
    """Select an exact, all-lane SIMT copy tile for one epilogue subtile.

    Return (M threads, N threads, contiguous values per thread). Prefer the
    existing 128-bit vector and eight N threads whenever they fit. Smaller
    subtiles may require fewer N threads or a narrower vector; never let CuTe
    partition a copy tile larger than its source/destination subtile.

    This pure helper runs at CuTe compile time on the actual epilogue extents,
    not the enclosing MMA tile. The 32 participating lanes must each own an
    equal, nonempty rectangular partition because all arrive on the pipeline.
    """
    if epi_m <= 0 or epi_n <= 0:
        raise ValueError("SIMT AUX copy requires positive epilogue extents")
    if dtype_bits <= 0 or dtype_bits > 128 or dtype_bits & (dtype_bits - 1):
        raise ValueError("SIMT AUX copy requires a power-of-two element bit width")
    values = 128 // dtype_bits
    while values:
        for n_threads in (8, 4, 2, 1, 16, 32):
            m_threads = 32 // n_threads
            if epi_m % m_threads == 0 and epi_n % (n_threads * values) == 0:
                return m_threads, n_threads, values
        values //= 2
    raise ValueError("Epilogue subtile has no exact all-32-lane SIMT AUX copy layout")
