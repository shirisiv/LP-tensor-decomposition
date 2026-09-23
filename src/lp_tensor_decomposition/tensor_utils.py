from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cp_tensor(factors: Sequence[ArrayLike]) -> NDArray[np.float64]:
    """Form a dense CP tensor from factor matrices A^(j) of shape r_j-by-r."""
    mats = [np.asarray(A, dtype=float) for A in factors]
    if not mats:
        raise ValueError("at least one factor matrix is required")
    ranks = {A.shape[1] for A in mats}
    if len(ranks) != 1:
        raise ValueError("all factor matrices must have the same number of columns")

    r = mats[0].shape[1]
    shape = tuple(A.shape[0] for A in mats)
    T = np.zeros(shape, dtype=float)
    for i in range(r):
        term = mats[0][:, i]
        for A in mats[1:]:
            term = np.multiply.outer(term, A[:, i])
        T += term
    return T


def rank_one_tensor(vectors: Sequence[ArrayLike]) -> NDArray[np.float64]:
    vecs = [np.asarray(v, dtype=float) for v in vectors]
    out = vecs[0]
    for v in vecs[1:]:
        out = np.multiply.outer(out, v)
    return out


def relative_residual(tensor: ArrayLike, factors: Sequence[ArrayLike]) -> float:
    T = np.asarray(tensor, dtype=float)
    R = T - cp_tensor(factors)
    denom = np.linalg.norm(T.ravel())
    return float(np.linalg.norm(R.ravel()) / (denom if denom else 1.0))


def mode_unfold(tensor: ArrayLike, mode: int) -> NDArray[np.float64]:
    T = np.asarray(tensor, dtype=float)
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def mode_product(
    tensor: ArrayLike,
    matrix: ArrayLike,
    mode: int,
) -> NDArray[np.float64]:
    """Multiply a tensor in one mode by a matrix B (new_dim-by-old_dim)."""
    T = np.asarray(tensor, dtype=float)
    B = np.asarray(matrix, dtype=float)
    if B.shape[1] != T.shape[mode]:
        raise ValueError("matrix dimension does not match tensor mode")
    out = np.tensordot(B, T, axes=(1, mode))
    return np.moveaxis(out, 0, mode)


def numerical_rank(
    A: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> int:
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.count_nonzero(s > atol + rtol * s[0]))


def compress_tensor(
    tensor: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Orthogonally compress each mode to the span of its mode unfolding.

    Returns the compressed core and orthonormal lifting matrices U_j such that
    tensor = core x_1 U_1 x_2 ... x_m U_m up to numerical precision.
    """
    T = np.asarray(tensor, dtype=float)
    bases: list[NDArray[np.float64]] = []
    for mode in range(T.ndim):
        unfold = mode_unfold(T, mode)
        U, s, _ = np.linalg.svd(unfold, full_matrices=False)
        if s.size == 0:
            rank = 0
        else:
            rank = int(np.count_nonzero(s > atol + rtol * s[0]))
        if rank == 0:
            raise ValueError("zero tensor / zero mode rank is not supported")
        bases.append(U[:, :rank])

    core = T.copy()
    for mode, U in enumerate(bases):
        core = mode_product(core, U.T, mode)
    return core, bases


def input_size(shape: Sequence[int]) -> int:
    return int(reduce(mul, shape, 1))
