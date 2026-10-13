"""Independent integer Philox stream for untimed Gumbel validation."""

from __future__ import annotations

import math

import numpy as np


def philox_words(seed: int, subsequences: np.ndarray) -> np.ndarray:
    sub = np.asarray(subsequences, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    c = [np.zeros_like(sub), np.zeros_like(sub), sub & mask, sub >> np.uint64(32)]
    k = [np.uint64(seed & 0xFFFFFFFF), np.uint64((seed >> 32) & 0xFFFFFFFF)]
    for round_index in range(10):
        p0 = c[0] * np.uint64(0xD2511F53)
        p1 = c[2] * np.uint64(0xCD9E8D57)
        c = [
            (p1 >> np.uint64(32)) ^ c[1] ^ k[0],
            p1 & mask,
            (p0 >> np.uint64(32)) ^ c[3] ^ k[1],
            p0 & mask,
        ]
        if round_index < 9:
            k = [
                (k[0] + np.uint64(0x9E3779B9)) & mask,
                (k[1] + np.uint64(0xBB67AE85)) & mask,
            ]
    return np.stack(c, axis=-1).astype(np.uint32)


def token_words(seed: int, batch: int, vocab: int) -> np.ndarray:
    vector = math.gcd(4, vocab)
    subsequences = np.arange(batch, dtype=np.uint64)[:, None] * np.uint64(
        vocab
    ) + np.arange(vocab // vector, dtype=np.uint64)[None, :] * np.uint64(vector)
    words = philox_words(seed, subsequences)
    return words[..., :vector].reshape(batch, vocab)
