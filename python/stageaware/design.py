"""Conference-matrix definitive screening designs (Xiao, Lin and Bai, 2012).

The Paley construction is used for prime-power orders q = 5, 7, 9, 11, 13,
giving conference matrices of order 6, 8, 10, 12, 14 and hence definitive
screening designs for m = 5, ..., 14 factors (one column is deleted for odd m).
GF(9) is represented as a + b t with t^2 = 2 over GF(3).
"""
from __future__ import annotations

import numpy as np

_SUPPORTED_Q = (5, 7, 9, 11, 13)


def _fsub(a: int, b: int, q: int) -> int:
    if q != 9:
        return (a - b) % q
    return ((a % 3 - b % 3) % 3) + 3 * (((a // 3) - (b // 3)) % 3)


def _fmul(a: int, b: int, q: int) -> int:
    if q != 9:
        return (a * b) % q
    aa, ab, ba, bb = a % 3, a // 3, b % 3, b // 3
    return ((aa * ba + 2 * ab * bb) % 3) + 3 * ((aa * bb + ab * ba) % 3)


def _fpow(a: int, e: int, q: int) -> int:
    v = 1
    for _ in range(e):
        v = _fmul(v, a, q)
    return v


def paley_conference(q: int) -> np.ndarray:
    """Paley conference matrix of order q + 1.

    Symmetric for q = 1 (mod 4) and skew-symmetric for q = 3 (mod 4).
    """
    if q not in _SUPPORTED_Q:
        raise ValueError(f"q must be one of {_SUPPORTED_Q}")

    def chi(v: int) -> int:
        if v == 0:
            return 0
        return 1 if _fpow(v, (q - 1) // 2, q) == 1 else -1

    Q = np.array([[chi(_fsub(a, b, q)) for b in range(q)] for a in range(q)], float)
    C = np.ones((q + 1, q + 1))
    C[0, 0] = 0.0
    C[1:, 1:] = Q
    if q % 4 == 3:
        C[1:, 0] = -1.0
    if not np.allclose(C.T @ C, q * np.eye(q + 1)):
        raise RuntimeError("conference identity C'C = qI failed")
    return C


def dsd(m: int) -> np.ndarray:
    """Definitive screening design [C; -C; 0] with 2*mm + 1 runs, mm = m rounded up to even."""
    mm = m if m % 2 == 0 else m + 1
    if mm not in (6, 8, 10, 12, 14):
        raise ValueError("supported m = 5, ..., 14")
    C = paley_conference(mm - 1)
    return np.vstack([C, -C, np.zeros(mm)])[:, :m]


def projection(m: int, active) -> np.ndarray:
    """Projection of the m-factor DSD onto the active factors (1-based indices)."""
    idx = [a - 1 for a in active]
    return dsd(m)[:, idx]
