from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def block_offsets(shape: Sequence[int]) -> NDArray[np.int64]:
    r"""Offsets of the mode blocks in X = A_1 \oplus ... \oplus A_m."""
    return np.concatenate(([0], np.cumsum(np.asarray(shape, dtype=int))))


def complementary_minor_matrix(M: ArrayLike) -> NDArray[np.float64]:
    """Return the skew matrix of complementary maximal minors.

    Parameters
    ----------
    M:
        An (m-2)-by-m matrix.

    Returns
    -------
    K:
        The m-by-m skew-symmetric matrix with, for j < k (0-based),

            K[j,k] = (-1)^(j+k+1) det(M with columns j,k deleted).

        This is the skew-matrix representation of the decomposable bivector
        determined by the complementary maximal minors of M.
    """
    M = np.asarray(M, dtype=float)
    d, m = M.shape
    if d != m - 2:
        raise ValueError(f"expected an (m-2)-by-m matrix, got {M.shape}")

    K = np.zeros((m, m), dtype=float)
    for j, k in combinations(range(m), 2):
        keep = [s for s in range(m) if s != j and s != k]
        # det of a 0-by-0 matrix is 1; this only arises for m=2, which is
        # outside the intended m>=3 setting but is mathematically consistent.
        minor = M[:, keep]
        det = 1.0 if d == 0 else float(np.linalg.det(minor))
        sign = -1.0 if (j + k + 1) % 2 else 1.0
        K[j, k] = sign * det
        K[k, j] = -K[j, k]
    return K


def alternating_contraction(
    tensor: ArrayLike,
    P: ArrayLike,
) -> NDArray[np.float64]:
    """Construct Omega_P from the tensor using the determinantal formula.

    This implements the computational form of the alternating contraction:
    for each tensor entry indexed by alpha, form the (m-2)-by-m matrix G_alpha
    by selecting from P one coordinate column in each mode block.  The
    complementary maximal minors of G_alpha are accumulated into the
    corresponding blocks of Omega_P.

    The implementation is polynomial in the dense input size N = prod r_j and
    avoids the (m-2)! permutation sum in the direct antisymmetrization formula.
    """
    T = np.asarray(tensor, dtype=float)
    if T.ndim < 3:
        raise ValueError("the LP algorithm is intended for tensors of order m >= 3")

    shape = T.shape
    m = T.ndim
    d = m - 2
    D = int(sum(shape))

    P = np.asarray(P, dtype=float)
    if P.shape != (d, D):
        raise ValueError(f"P must have shape {(d, D)}, got {P.shape}")

    offsets = block_offsets(shape)
    Omega = np.zeros((D, D), dtype=float)

    for alpha in np.ndindex(shape):
        coeff = float(T[alpha])
        if coeff == 0.0:
            continue

        # Column s of G_alpha is the coordinate of the s-th basis vector
        # chosen by alpha inside the s-th mode block.
        cols = [offsets[s] + alpha[s] for s in range(m)]
        G = P[:, cols]
        K = complementary_minor_matrix(G)

        for j, k in combinations(range(m), 2):
            value = coeff * K[j, k]
            row = offsets[j] + alpha[j]
            col = offsets[k] + alpha[k]
            Omega[row, col] += value
            Omega[col, row] -= value

    # Remove tiny loss of skew-symmetry due to floating-point arithmetic.
    return 0.5 * (Omega - Omega.T)
